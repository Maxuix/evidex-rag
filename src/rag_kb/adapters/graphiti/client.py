"""Privacy-shielded, build-scoped Graphiti client."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import copy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hashlib
import logging
import os
import re
import unicodedata
from types import SimpleNamespace
from typing import Any
from uuid import uuid5

from pydantic import ValidationError

os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

_GRAPHITI: SimpleNamespace | None = None
# OpenCode Go / mimo-v2.5 has produced four consecutive schema-shaped
# attribute payloads in a real v3 build. Keep the repair loop bounded while
# allowing one more clean response before failing the entire episode closed.
SCHEMA_ECHO_MAX_ATTEMPTS = 5

from rag_kb.domain import (
    GraphChunkSource,
    GraphitiBuildSnapshot,
    GraphitiEdgeResult,
    GraphitiPathResult,
    GraphitiSearchQuery,
    SOFTWARE_GRAPH_SCHEMA_PROFILE_DIGEST,
    SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY,
)
from rag_kb.graph.schema_profiles import (
    CompiledGraphSchema,
    GraphSchemaProfileError,
    get_graph_schema_registry,
)


_GRAPH_SCHEMA_REGISTRY = get_graph_schema_registry()
_SOFTWARE_SCHEMA = _GRAPH_SCHEMA_REGISTRY.compile(
    SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY,
    digest=SOFTWARE_GRAPH_SCHEMA_PROFILE_DIGEST,
    extractor_version="graphiti_v4",
)

# Transitional import compatibility for locked software evaluation tooling.
# Runtime calls resolve the profile from each immutable build instead of using
# these aliases as a global extraction choice.
GRAPHITI_V3_EXTRACTION_INSTRUCTIONS = _SOFTWARE_SCHEMA.extraction_instructions
GRAPHITI_V2_EXTRACTION_INSTRUCTIONS = GRAPHITI_V3_EXTRACTION_INSTRUCTIONS
GRAPHITI_ENTITY_TYPES = _SOFTWARE_SCHEMA.entity_types
GRAPHITI_EDGE_TYPES = _SOFTWARE_SCHEMA.edge_types
GRAPHITI_EDGE_TYPE_MAP = _SOFTWARE_SCHEMA.edge_type_map

_GRAPHITI_ADJACENCY_MULTIPLIER = 8
_EPISODE_COMPLETION_STATE = "rag-kb-complete-v1"


@dataclass(frozen=True, slots=True)
class _GraphitiEntityResult:
    entity_uuid: str
    name: str
    rank: int


class GraphitiSchemaEchoError(RuntimeError):
    """Provider returned JSON schema instead of the required Graphiti fields."""


def required_model_field_names(response_model: Any) -> tuple[str, ...]:
    fields = getattr(response_model, "model_fields", None)
    if not isinstance(fields, dict):
        return ()
    names: list[str] = []
    for name, field in fields.items():
        checker = getattr(field, "is_required", None)
        if callable(checker):
            if checker():
                names.append(str(name))
            continue
        if getattr(field, "is_required", False):
            names.append(str(name))
    return tuple(names)


def model_field_names(response_model: Any) -> tuple[str, ...]:
    fields = getattr(response_model, "model_fields", None)
    if not isinstance(fields, dict):
        return ()
    return tuple(str(name) for name in fields)


def is_schema_echo_payload(result: Any, response_model: Any) -> bool:
    if response_model is None or not isinstance(result, dict):
        return False
    fields = model_field_names(response_model)
    # Attribute models intentionally use optional/defaulted fields, so checking
    # only required keys misses the most common OpenAI-compatible fallback
    # failure: the model returns model_json_schema() itself. Graphiti otherwise
    # accepts these unknown schema keys because Pydantic ignores extras, then
    # attempts to persist the nested ``properties`` map into FalkorDB.
    looks_like_schema = (
        result.get("type") == "object"
        and isinstance(result.get("properties"), dict)
        and any(
            key in result
            for key in ("title", "description", "required", "$defs", "additionalProperties")
        )
    )
    if looks_like_schema and not any(field in result for field in fields):
        return True
    required = required_model_field_names(response_model)
    return bool(required) and any(name not in result for name in required)


def is_invalid_model_payload(
    result: Any,
    response_model: Any,
    *,
    strict_fields: bool = False,
) -> bool:
    if is_schema_echo_payload(result, response_model):
        return True
    validator = getattr(response_model, "model_validate", None)
    if not callable(validator) or not isinstance(result, dict):
        return False
    if strict_fields:
        allowed = set(model_field_names(response_model))
        if any(str(key) not in allowed for key in result):
            # Graphiti persists the raw attribute response after a capped merge.
            # Pydantic's default extra="ignore" would otherwise let schema
            # metadata such as type/title/properties leak into graph properties.
            return True
    try:
        validator(result)
    except ValidationError:
        # OpenAI-compatible json_object fallbacks also return field-level schema
        # fragments (for example {"short_names": {"items": ...}}). They are
        # not recognizable as a complete JSON Schema document, but validating
        # before Graphiti consumes the payload catches them at the retry boundary.
        return True
    return False


def append_schema_echo_repair_note(messages: Any, response_model: Any) -> None:
    if not messages:
        return
    last = messages[-1]
    content = getattr(last, "content", None)
    if not isinstance(content, str):
        return
    fields = required_model_field_names(response_model) or model_field_names(response_model)
    field_names = ", ".join(fields) if fields else "the required fields"
    last.content = (
        content
        + "\n\nYour previous response was invalid: you returned the JSON schema "
        "definition itself instead of the actual result. Return ONLY a JSON object "
        f"with the keys: {field_names}. Do not include the schema definition."
    )


class SchemaEchoRepairingLLMClient:
    """Retry Graphiti structured calls when the provider echoes the schema.

    The first call uses Graphiti's public generate_response path, which injects
    the schema once in json_object mode. Repairs call _generate_response so the
    schema is not appended again.
    """

    def __init__(
        self,
        inner: Any,
        *,
        max_attempts: int = SCHEMA_ECHO_MAX_ATTEMPTS,
        provider_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._inner = inner
        self._max_attempts = max_attempts
        self._provider_semaphore = provider_semaphore

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def generate_response(
        self,
        messages: Any,
        response_model: Any = None,
        max_tokens: int | None = None,
        model_size: Any = None,
        group_id: str | None = None,
        prompt_name: str | None = None,
        *,
        attribute_extraction: bool = False,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        first_kwargs: dict[str, Any] = {}
        if response_model is not None:
            first_kwargs["response_model"] = response_model
        if max_tokens is not None:
            first_kwargs["max_tokens"] = max_tokens
        if model_size is not None:
            first_kwargs["model_size"] = model_size
        if group_id is not None:
            first_kwargs["group_id"] = group_id
        if prompt_name is not None:
            first_kwargs["prompt_name"] = prompt_name
        if attribute_extraction:
            first_kwargs["attribute_extraction"] = True
        for attempt in range(self._max_attempts):
            if attempt == 0:
                result = await self._call_provider(
                    lambda: self._inner.generate_response(messages, **first_kwargs)
                )
            else:
                raw = getattr(self._inner, "_generate_response", None)
                if raw is None:
                    raise GraphitiSchemaEchoError(
                        "graphiti structured output missing required fields"
                    )
                repaired = copy.deepcopy(messages)
                append_schema_echo_repair_note(repaired, response_model)
                raw_kwargs: dict[str, Any] = {}
                if max_tokens is not None:
                    raw_kwargs["max_tokens"] = max_tokens
                if model_size is not None:
                    raw_kwargs["model_size"] = model_size
                result = await self._call_provider(
                    lambda: raw(repaired, response_model, **raw_kwargs)
                )
            if is_invalid_model_payload(
                result,
                response_model,
                strict_fields=attribute_extraction,
            ):
                last_error = GraphitiSchemaEchoError(
                    "graphiti structured output missing required fields"
                )
                continue
            if not isinstance(result, dict):
                raise TypeError("graphiti llm client must return a JSON object")
            return result
        raise last_error or GraphitiSchemaEchoError(
            "graphiti structured output missing required fields"
        )

    async def _call_provider(
        self,
        operation: Callable[[], Awaitable[Any]],
    ) -> Any:
        if self._provider_semaphore is None:
            return await operation()
        async with self._provider_semaphore:
            return await operation()


def as_graphiti_llm_client(
    inner: Any,
    *,
    max_attempts: int = SCHEMA_ECHO_MAX_ATTEMPTS,
    provider_semaphore: asyncio.Semaphore | None = None,
) -> Any:
    """Preserve the inner Graphiti LLMClient type for pydantic isinstance checks."""

    repair = SchemaEchoRepairingLLMClient(
        inner,
        max_attempts=max_attempts,
        provider_semaphore=provider_semaphore,
    )

    class _SchemaEchoRepairingLLMClient(type(inner)):
        def __init__(self) -> None:
            self._inner = inner
            self._schema_echo_repair = repair

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        def set_tracer(self, tracer: Any) -> None:
            setter = getattr(self._inner, "set_tracer", None)
            if setter is not None:
                setter(tracer)

        @property
        def token_tracker(self) -> Any:
            return getattr(self._inner, "token_tracker", None)

        async def generate_response(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return await self._schema_echo_repair.generate_response(*args, **kwargs)

    _SchemaEchoRepairingLLMClient.__name__ = "SchemaEchoRepairingLLMClient"
    _SchemaEchoRepairingLLMClient.__qualname__ = "SchemaEchoRepairingLLMClient"
    return _SchemaEchoRepairingLLMClient()


@dataclass(frozen=True, slots=True)
class GraphitiModelCredentials:
    chat_base_url: str
    chat_api_key: str
    chat_model: str
    chat_timeout_seconds: float
    chat_temperature: float
    chat_max_tokens: int
    structured_output_mode: str
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    embedding_timeout_seconds: float
    embedding_batch_size: int
    max_concurrency: int


class GraphitiRuntime:
    """Own one initialized Graphiti instance per immutable build."""

    def __init__(
        self,
        credentials: Callable[[GraphitiBuildSnapshot], Awaitable[GraphitiModelCredentials]],
        *,
        host: str = "127.0.0.1",
        port: int = 6379,
        probe_ttl_seconds: float = 15.0,
        schema_registry=None,
    ) -> None:
        self._credentials = credentials
        self._host = host
        self._port = port
        self._probe_ttl_seconds = probe_ttl_seconds
        self._clients: dict[str, tuple[Any, Any]] = {}
        self._client_lock = asyncio.Lock()
        self._probe_until: dict[str, float] = {}
        self._complete_probe_until: dict[str, float] = {}
        self._schema_registry = schema_registry or _GRAPH_SCHEMA_REGISTRY
        _shield_upstream_logs()

    async def probe(
        self,
        build: GraphitiBuildSnapshot,
        *,
        episode_uuid: str | None = None,
        require_complete: bool = False,
    ) -> bool:
        compiled = self._compiled_schema(build)
        _, driver = await self._client(build)
        loop = asyncio.get_running_loop()
        cache = self._complete_probe_until if require_complete else self._probe_until
        if cache.get(build.group_id, 0.0) > loop.time():
            return True
        try:
            result = await driver.execute_query(
                "MATCH (n:Episodic) RETURN count(n) AS count"
            )
            records = result[0] if result else []
            episode_count = int(records[0]["count"]) if records else 0
            if require_complete and episode_count != build.expected_episode_count:
                return False
            if require_complete:
                result = await driver.execute_query(
                    """
                    MATCH (source:Entity)-[edge:RELATES_TO]->(target:Entity)
                    WHERE source.uuid = target.uuid
                    RETURN count(edge) AS count
                    """
                )
                records = result[0] if result else []
                self_loop_count = int(records[0]["count"]) if records else 0
                if self_loop_count:
                    return False
                if compiled.validation_policy.standalone_alias_orphan_check:
                    result = await driver.execute_query(
                        """
                        MATCH (alias:Entity:AliasSurface)
                        OPTIONAL MATCH (alias)-[edge:RELATES_TO]-(:Entity)
                        WITH alias, count(edge) AS degree
                        WHERE degree = 0
                        RETURN count(alias) AS count
                        """
                    )
                    records = result[0] if result else []
                    orphan_alias_count = int(records[0]["count"]) if records else 0
                    if orphan_alias_count:
                        return False
                result = await driver.execute_query(
                    """
                    MATCH (:Entity)-[edge:RELATES_TO]->(:Entity)
                    WHERE edge.name IS NULL OR trim(edge.name) = ''
                    RETURN count(edge) AS count
                    """
                )
                records = result[0] if result else []
                untyped_edge_count = int(records[0]["count"]) if records else 0
                if untyped_edge_count:
                    return False
            if episode_uuid is not None:
                modules = _graphiti_modules()
                await modules.EpisodicNode.get_by_uuid(driver, episode_uuid)
        except Exception as error:
            if not isinstance(error, _graphiti_modules().NodeNotFoundError):
                raise
            return False
        self._probe_until[build.group_id] = loop.time() + self._probe_ttl_seconds
        if require_complete:
            self._complete_probe_until[build.group_id] = (
                loop.time() + self._probe_ttl_seconds
            )
        return True

    async def add_episode(
        self, build: GraphitiBuildSnapshot, chunk: GraphChunkSource
    ) -> str:
        compiled = self._compiled_schema(build)
        graphiti, driver = await self._client(build)
        modules = _graphiti_modules()
        reference_time = chunk.reference_time or datetime.now(UTC)
        episode_uuid = graphiti_episode_uuid(build, chunk)
        completed = await self._completed_episode_uuids(driver, (episode_uuid,))
        if episode_uuid in completed:
            return episode_uuid
        try:
            await modules.EpisodicNode.get_by_uuid(driver, episode_uuid)
        except modules.NodeNotFoundError:
            pass
        else:
            # The external write may have completed before its PostgreSQL mapping.
            # Replaying the same immutable chunk first removes that uncommitted
            # Episode, then reuses the deterministic identity.
            await graphiti.remove_episode(episode_uuid)
        previous = await graphiti.retrieve_episodes(
            reference_time,
            last_n=modules.RELEVANT_SCHEMA_LIMIT,
            group_ids=[build.group_id],
            source=modules.EpisodeType.text,
        )
        name = f"chunk-{chunk.index_chunk_id}"
        description = f"chunk {chunk.index_chunk_id}"
        episode = modules.EpisodicNode(
            uuid=episode_uuid,
            name=name,
            group_id=build.group_id,
            labels=[],
            source=modules.EpisodeType.text,
            source_description=description,
            content=chunk.content,
            created_at=datetime.now(UTC),
            valid_at=reference_time,
        )
        await episode.save(driver)
        result = await graphiti.add_episode(
            name=name,
            episode_body=chunk.content,
            source_description=description,
            reference_time=reference_time,
            source=modules.EpisodeType.text,
            group_id=build.group_id,
            uuid=episode_uuid,
            previous_episode_uuids=[str(item.uuid) for item in previous],
            entity_types=compiled.entity_types,
            edge_types=compiled.edge_types,
            edge_type_map=compiled.edge_type_map,
            custom_extraction_instructions=compiled.extraction_instructions,
        )
        # The extraction contract rejects self-relations, but a provider can
        # still emit one despite the prompt.  Remove that invalid topology at
        # the build boundary before the PostgreSQL mapping is committed.  The
        # query is scoped to this build's Falkor database, so it cannot touch
        # another workspace or graph generation.  Test doubles that do not
        # implement the driver query surface intentionally skip this I/O.
        if compiled.validation_policy.reject_entity_self_loops:
            await self._remove_self_loop_edges(driver, build)
        result_uuid = str(result.episode.uuid)
        await self._mark_episodes_complete(driver, (result_uuid,))
        return result_uuid

    async def add_episodes_bulk(
        self,
        build: GraphitiBuildSnapshot,
        chunks: tuple[GraphChunkSource, ...],
    ) -> tuple[str, ...]:
        """Add a bounded batch while retaining the immutable build contract.

        The database work lease remains per build, so callers can safely use
        this only while they own that lease.  Graphiti's bulk path performs
        bounded concurrent provider calls internally and returns the same
        deterministic episode identities used by ``add_episode``.
        """

        if not chunks:
            return ()
        compiled = self._compiled_schema(build)
        graphiti, driver = await self._client(build)
        modules = _graphiti_modules()
        from graphiti_core.utils.bulk_utils import RawEpisode

        episode_uuids = tuple(graphiti_episode_uuid(build, chunk) for chunk in chunks)
        completed = await self._completed_episode_uuids(driver, episode_uuids)
        pending = tuple(
            (chunk, episode_uuid)
            for chunk, episode_uuid in zip(chunks, episode_uuids, strict=True)
            if episode_uuid not in completed
        )
        if not pending:
            return episode_uuids

        raw_episodes: list[RawEpisode] = []
        pending_uuids: list[str] = []
        for chunk, episode_uuid in pending:
            reference_time = chunk.reference_time or datetime.now(UTC)
            try:
                await modules.EpisodicNode.get_by_uuid(driver, episode_uuid)
            except modules.NodeNotFoundError:
                pass
            else:
                # Preserve the single-episode recovery contract: an external
                # write without its relational mapping is replayed cleanly.
                await graphiti.remove_episode(episode_uuid)
            episode = modules.EpisodicNode(
                uuid=episode_uuid,
                name=f"chunk-{chunk.index_chunk_id}",
                group_id=build.group_id,
                labels=[],
                source=modules.EpisodeType.text,
                source_description=f"chunk {chunk.index_chunk_id}",
                content=chunk.content,
                created_at=datetime.now(UTC),
                valid_at=reference_time,
            )
            await episode.save(driver)
            pending_uuids.append(episode_uuid)
            raw_episodes.append(
                RawEpisode(
                    name=episode.name,
                    uuid=episode_uuid,
                    content=chunk.content,
                    source_description=episode.source_description,
                    source=modules.EpisodeType.text,
                    reference_time=reference_time,
                )
            )
        result = await graphiti.add_episode_bulk(
            raw_episodes,
            group_id=build.group_id,
            entity_types=compiled.entity_types,
            edge_types=compiled.edge_types,
            edge_type_map=compiled.edge_type_map,
            custom_extraction_instructions=compiled.extraction_instructions,
        )
        result_episodes = tuple(getattr(result, "episodes", ()))
        if len(result_episodes) != len(pending):
            raise RuntimeError("Graphiti bulk episode count mismatch")
        if compiled.validation_policy.reject_entity_self_loops:
            await self._remove_self_loop_edges(driver, build)
        result_uuids = tuple(str(episode.uuid) for episode in result_episodes)
        if result_uuids != tuple(pending_uuids):
            raise RuntimeError("Graphiti bulk episode identity mismatch")
        await self._mark_episodes_complete(driver, result_uuids)
        return episode_uuids

    @staticmethod
    async def _completed_episode_uuids(
        driver: Any,
        episode_uuids: tuple[str, ...],
    ) -> frozenset[str]:
        execute = getattr(driver, "execute_query", None)
        if not callable(execute) or not episode_uuids:
            return frozenset()
        result = await execute(
            """
            MATCH (episode:Episodic)
            WHERE episode.uuid IN $episode_uuids
              AND episode.rag_kb_ingestion_state = $completion_state
            RETURN episode.uuid AS uuid
            """,
            episode_uuids=list(episode_uuids),
            completion_state=_EPISODE_COMPLETION_STATE,
            routing_="r",
        )
        records = result[0] if result else []
        return frozenset(str(record["uuid"]) for record in records)

    @staticmethod
    async def _mark_episodes_complete(
        driver: Any,
        episode_uuids: tuple[str, ...],
    ) -> None:
        execute = getattr(driver, "execute_query", None)
        if not callable(execute) or not episode_uuids:
            return
        result = await execute(
            """
            MATCH (episode:Episodic)
            WHERE episode.uuid IN $episode_uuids
            SET episode.rag_kb_ingestion_state = $completion_state
            RETURN count(episode) AS count
            """,
            episode_uuids=list(episode_uuids),
            completion_state=_EPISODE_COMPLETION_STATE,
            routing_="w",
        )
        records = result[0] if result else []
        if records and int(records[0]["count"]) != len(episode_uuids):
            raise RuntimeError("Graphiti episode completion marker count mismatch")

    def _compiled_schema(self, build: GraphitiBuildSnapshot) -> CompiledGraphSchema:
        """Resolve the exact build identity before touching Graphiti."""

        key = getattr(build, "schema_profile_key", SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY)
        digest = getattr(
            build,
            "schema_profile_digest",
            SOFTWARE_GRAPH_SCHEMA_PROFILE_DIGEST,
        )
        extractor_version = getattr(
            build,
            "extractor_version",
            "graphiti_v4",
        )
        try:
            registry = getattr(self, "_schema_registry", _GRAPH_SCHEMA_REGISTRY)
            return registry.compile(
                key,
                digest=digest,
                extractor_version=extractor_version,
            )
        except GraphSchemaProfileError as error:
            raise RuntimeError("graph_schema_profile_mismatch") from error

    @staticmethod
    async def _remove_self_loop_edges(driver: Any, build: GraphitiBuildSnapshot) -> None:
        clone = getattr(driver, "clone", None)
        if not callable(clone):
            return
        scoped_driver = clone(database=build.group_id)
        execute_query = getattr(scoped_driver, "execute_query", None)
        if not callable(execute_query):
            return
        await execute_query(
            """
            MATCH (source:Entity)-[edge:RELATES_TO]->(target:Entity)
            WHERE source.uuid = target.uuid
            DELETE edge
            """,
            routing_="w",
        )

    async def search(
        self, build: GraphitiBuildSnapshot, query: GraphitiSearchQuery
    ) -> tuple[GraphitiEdgeResult, ...]:
        self._compiled_schema(build)
        graphiti, driver = await self._client(build)
        return await self._search_edges(graphiti, driver, build, query)

    async def search_paths(
        self, build: GraphitiBuildSnapshot, query: GraphitiSearchQuery
    ) -> tuple[GraphitiPathResult, ...]:
        """Resolve named query entities into bounded, connected graph paths."""

        self._compiled_schema(build)
        graphiti, driver = await self._client(build)
        ranked_edges, ranked_entities = await asyncio.gather(
            self._search_edges(graphiti, driver, build, query),
            self._search_entities(graphiti, driver, build, query),
        )
        endpoint_edges = await self._edges_by_uuid(
            driver,
            build,
            tuple(edge.edge_uuid for edge in ranked_edges),
        )
        ranked_edges = _with_endpoints(ranked_edges, endpoint_edges)
        seed_entity_uuids = tuple(
            dict.fromkeys(
                (
                    *_grounded_entity_ids(query.query, ranked_entities),
                    *_grounded_edge_entity_ids(query.query, ranked_edges),
                )
            )
        )
        native_edges = await self._search_centered_edges(
            graphiti, driver, build, query, seed_entity_uuids
        )
        native_edges = _with_endpoints(
            native_edges,
            await self._edges_by_uuid(
                driver, build, tuple(edge.edge_uuid for edge in native_edges)
            ),
        )
        ranked_first_hops = _merge_ranked_edges(native_edges, ranked_edges)
        return _rank_graphiti_paths(
            query.query,
            ranked_first_hops,
            ranked_first_hops,
            limit=min(query.limit, 20),
            seed_entity_uuids=seed_entity_uuids,
        )

    async def _search_centered_edges(
        self,
        graphiti: Any,
        driver: Any,
        build: GraphitiBuildSnapshot,
        query: GraphitiSearchQuery,
        seed_entity_uuids: tuple[str, ...],
    ) -> tuple[GraphitiEdgeResult, ...]:
        """Use Graphiti's node-distance reranker and native bounded BFS."""

        if not seed_entity_uuids:
            return ()
        modules = _graphiti_modules()
        limit = min(64, max(query.limit * _GRAPHITI_ADJACENCY_MULTIPLIER, 16))
        config = modules.EDGE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(deep=True)
        config.limit = limit
        config.edge_config.search_methods = [
            modules.EdgeSearchMethod.bm25,
            modules.EdgeSearchMethod.cosine_similarity,
            modules.EdgeSearchMethod.bfs,
        ]
        config.edge_config.bfs_max_depth = 3
        merged: list[GraphitiEdgeResult] = []
        seen: set[str] = set()
        for seed_uuid in seed_entity_uuids[:3]:
            result = await graphiti.search_(
                query.query,
                config=config,
                group_ids=[build.group_id],
                center_node_uuid=seed_uuid,
                bfs_origin_node_uuids=[seed_uuid],
                driver=driver.clone(database=build.group_id),
            )
            for edge in result.edges:
                edge_uuid = str(edge.uuid)
                if edge_uuid in seen:
                    continue
                seen.add(edge_uuid)
                merged.append(_edge_result_from_graphiti(edge, len(merged) + 1))
                if len(merged) >= limit:
                    return tuple(merged)
        return tuple(merged)

    async def _search_edges(
        self,
        graphiti: Any,
        driver: Any,
        build: GraphitiBuildSnapshot,
        query: GraphitiSearchQuery,
    ) -> tuple[GraphitiEdgeResult, ...]:
        modules = _graphiti_modules()
        config = modules.EDGE_HYBRID_SEARCH_RRF.model_copy(update={"limit": query.limit})
        result = await graphiti.search_(
            query.query,
            config=config,
            group_ids=[build.group_id],
            driver=driver.clone(database=build.group_id),
        )
        return tuple(
            _edge_result_from_graphiti(edge, rank)
            for rank, edge in enumerate(result.edges[: query.limit], start=1)
        )

    async def _search_entities(
        self,
        graphiti: Any,
        driver: Any,
        build: GraphitiBuildSnapshot,
        query: GraphitiSearchQuery,
    ) -> tuple[_GraphitiEntityResult, ...]:
        modules = _graphiti_modules()
        entity_limit = min(32, max(query.limit * 4, 16))
        config = modules.NODE_HYBRID_SEARCH_RRF.model_copy(
            update={"limit": entity_limit}
        )
        result = await graphiti.search_(
            query.query,
            config=config,
            group_ids=[build.group_id],
            driver=driver.clone(database=build.group_id),
        )
        return tuple(
            _GraphitiEntityResult(
                entity_uuid=str(node.uuid),
                name=str(node.name or ""),
                rank=rank,
            )
            for rank, node in enumerate(result.nodes[:entity_limit], start=1)
            if node.uuid is not None and str(node.name or "").strip()
        )

    async def diagnostic_edges(
        self,
        build: GraphitiBuildSnapshot,
    ) -> tuple[GraphitiEdgeResult, ...]:
        """Read every build-scoped relation for content-safe offline scoring.

        Callers must reduce the returned source content to aggregate counters or
        synthetic relation IDs before persisting diagnostics.
        """

        self._compiled_schema(build)
        _, driver = await self._client(build)
        records, _, _ = await driver.clone(database=build.group_id).execute_query(
            """
            MATCH (source:Entity)-[edge:RELATES_TO]->(target:Entity)
            RETURN source.uuid AS source_uuid,
                   source.name AS source_name,
                   edge.uuid AS edge_uuid,
                   edge.name AS relation_type,
                   edge.fact AS fact,
                   edge.episodes AS episodes,
                   target.uuid AS target_uuid,
                   target.name AS target_name
            ORDER BY edge.uuid
            """,
            routing_="r",
        )
        return tuple(
            _edge_result_from_record(row, rank)
            for rank, row in enumerate(records, start=1)
        )

    @staticmethod
    async def _edges_by_uuid(
        driver: Any,
        build: GraphitiBuildSnapshot,
        edge_uuids: tuple[str, ...],
    ) -> tuple[GraphitiEdgeResult, ...]:
        if not edge_uuids:
            return ()
        records, _, _ = await driver.clone(database=build.group_id).execute_query(
            """
            MATCH (source:Entity)-[edge:RELATES_TO]->(target:Entity)
            WHERE edge.uuid IN $edge_uuids
            RETURN source.uuid AS source_uuid,
                   source.name AS source_name,
                   edge.uuid AS edge_uuid,
                   edge.name AS relation_type,
                   edge.fact AS fact,
                   edge.episodes AS episodes,
                   target.uuid AS target_uuid,
                   target.name AS target_name
            """,
            edge_uuids=list(edge_uuids),
            routing_="r",
        )
        rank_by_id = {
            edge_uuid: rank for rank, edge_uuid in enumerate(edge_uuids, start=1)
        }
        return tuple(
            _edge_result_from_record(row, rank_by_id[str(row["edge_uuid"])])
            for row in records
            if str(row["edge_uuid"]) in rank_by_id
        )

    @staticmethod
    async def _adjacent_edges(
        driver: Any,
        build: GraphitiBuildSnapshot,
        entity_uuids: tuple[str, ...],
        *,
        limit: int,
    ) -> tuple[GraphitiEdgeResult, ...]:
        records, _, _ = await driver.clone(database=build.group_id).execute_query(
            """
            MATCH (source:Entity)-[edge:RELATES_TO]->(target:Entity)
            WHERE source.uuid IN $entity_uuids OR target.uuid IN $entity_uuids
            RETURN source.uuid AS source_uuid,
                   source.name AS source_name,
                   edge.uuid AS edge_uuid,
                   edge.name AS relation_type,
                   edge.fact AS fact,
                   edge.episodes AS episodes,
                   target.uuid AS target_uuid,
                   target.name AS target_name
            ORDER BY edge.created_at, edge.uuid
            LIMIT $limit
            """,
            entity_uuids=list(entity_uuids),
            limit=limit,
            routing_="r",
        )
        return tuple(
            _edge_result_from_record(row, rank)
            for rank, row in enumerate(records, start=1)
        )

    async def delete_graph(self, build: GraphitiBuildSnapshot) -> None:
        cached = self._clients.pop(build.group_id, None)
        self._probe_until.pop(build.group_id, None)
        self._complete_probe_until.pop(build.group_id, None)
        if cached is not None:
            _, driver = cached
            try:
                await driver.client.select_graph(build.group_id).delete()
            finally:
                await driver.close()
            return
        modules = _graphiti_modules()
        driver = modules.FalkorDriver(
            host=self._host,
            port=self._port,
            database=build.group_id,
        )
        try:
            task = getattr(driver, "_init_task", None)
            if task is not None:
                await task
            await driver.client.select_graph(build.group_id).delete()
        finally:
            await driver.close()

    async def close(self) -> None:
        clients = tuple(self._clients.values())
        self._clients.clear()
        self._probe_until.clear()
        self._complete_probe_until.clear()
        await asyncio.gather(
            *(driver.close() for _, driver in clients),
            return_exceptions=True,
        )

    async def _client(self, build: GraphitiBuildSnapshot):
        cached = self._clients.get(build.group_id)
        if cached is not None:
            return cached
        async with self._client_lock:
            cached = self._clients.get(build.group_id)
            if cached is not None:
                return cached
            modules = _graphiti_modules()
            values = await self._credentials(build)
            provider_semaphore = asyncio.Semaphore(values.max_concurrency)
            llm = as_graphiti_llm_client(
                modules.OpenAIGenericClient(
                    config=modules.LLMConfig(
                        api_key=values.chat_api_key,
                        model=values.chat_model,
                        base_url=values.chat_base_url,
                        temperature=values.chat_temperature,
                        max_tokens=values.chat_max_tokens,
                    ),
                    client=modules.AsyncOpenAI(
                        api_key=values.chat_api_key,
                        base_url=values.chat_base_url,
                        timeout=values.chat_timeout_seconds,
                    ),
                    structured_output_mode=values.structured_output_mode,
                ),
                provider_semaphore=provider_semaphore,
            )
            embedder = modules.BoundedEmbedder(
                modules.OpenAIEmbedder(
                    config=modules.OpenAIEmbedderConfig(
                        api_key=values.embedding_api_key,
                        base_url=values.embedding_base_url,
                        embedding_model=values.embedding_model,
                        embedding_dim=build.embedding_dimension,
                    ),
                    client=modules.AsyncOpenAI(
                        api_key=values.embedding_api_key,
                        base_url=values.embedding_base_url,
                        timeout=values.embedding_timeout_seconds,
                    ),
                ),
                values.embedding_batch_size,
                provider_semaphore,
            )
            driver = modules.FalkorDriver(
                host=self._host,
                port=self._port,
                database=build.group_id,
            )
            task = getattr(driver, "_init_task", None)
            if task is not None:
                await task
            graphiti = modules.Graphiti(
                graph_driver=driver,
                llm_client=llm,
                embedder=embedder,
                cross_encoder=modules.NeverRerank(),
                store_raw_episode_content=True,
                # Keep Graphiti's own scheduler aligned with the provider
                # budget.  The shared request semaphore above remains the
                # effective boundary for helpers that ignore this setting.
                max_coroutines=values.max_concurrency,
            )
            cached = (graphiti, driver)
            self._clients[build.group_id] = cached
            return cached


def _normalized_surface(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _entity_surface_variants(value: str) -> tuple[str, ...]:
    variants = [_normalized_surface(value)]
    words = re.findall(r"[A-Za-z0-9]+", value)
    if len(words) >= 2:
        variants.append("".join(word[0] for word in words).casefold())
    variants.extend(
        _normalized_surface(item)
        for item in re.findall(r"[（(]([^）)]+)[）)]", value)
    )
    for suffix in ("基金会", "有限公司", "公司", "组织", "项目", "平台", "服务"):
        if value.endswith(suffix):
            variants.append(_normalized_surface(value[: -len(suffix)]))
    return tuple(dict.fromkeys(item for item in variants if len(item) >= 2))


def _edge_result_from_record(row: Any, rank: int) -> GraphitiEdgeResult:
    return GraphitiEdgeResult(
        edge_uuid=str(row["edge_uuid"]),
        relation_type=str(row.get("relation_type") or ""),
        fact=str(row.get("fact") or ""),
        episode_uuids=tuple(str(value) for value in (row.get("episodes") or ())),
        rank=rank,
        source_entity_uuid=str(row["source_uuid"]),
        source_entity_name=str(row.get("source_name") or ""),
        target_entity_uuid=str(row["target_uuid"]),
        target_entity_name=str(row.get("target_name") or ""),
    )


def _edge_result_from_graphiti(edge: Any, rank: int) -> GraphitiEdgeResult:
    return GraphitiEdgeResult(
        edge_uuid=str(edge.uuid),
        relation_type=str(getattr(edge, "name", "") or ""),
        fact=str(edge.fact or ""),
        episode_uuids=tuple(str(value) for value in (edge.episodes or ())),
        rank=rank,
        source_entity_uuid=(
            str(edge.source_node_uuid) if edge.source_node_uuid is not None else ""
        ),
        source_entity_name="",
        target_entity_uuid=(
            str(edge.target_node_uuid) if edge.target_node_uuid is not None else ""
        ),
        target_entity_name="",
    )


def _entry_endpoint(
    query: str,
    edge: GraphitiEdgeResult,
    *,
    seed_entity_uuids: tuple[str, ...] = (),
) -> tuple[str, bool]:
    for entity_uuid in seed_entity_uuids:
        if entity_uuid in edge.endpoint_uuids:
            return entity_uuid, True
    normalized_query = _normalized_surface(query)
    for entity_uuid, entity_name in (
        (edge.source_entity_uuid, edge.source_entity_name),
        (edge.target_entity_uuid, edge.target_entity_name),
    ):
        if any(
            _query_mentions_entity(normalized_query, surface)
            for surface in _entity_surface_variants(entity_name)
        ):
            return entity_uuid, True
    return edge.source_entity_uuid, False


def _query_mentions_entity(normalized_query: str, normalized_name: str) -> bool:
    if len(normalized_name) < 2:
        return False
    # Legal/corporate suffix omission is handled explicitly by
    # _entity_surface_variants. A generic leading-prefix match incorrectly
    # grounds repositories such as scikit-learn/scikit-learn when the question
    # names the project scikit-learn, duplicating seeds and path budgets.
    return normalized_name in normalized_query


def _grounded_entity_ids(
    query: str,
    entities: tuple[_GraphitiEntityResult, ...],
) -> tuple[str, ...]:
    normalized_query = _normalized_surface(query)
    return tuple(
        entity.entity_uuid
        for entity in entities
        if any(
            _query_mentions_entity(normalized_query, surface)
            for surface in _entity_surface_variants(entity.name)
        )
    )


def _grounded_edge_entity_ids(
    query: str,
    edges: tuple[GraphitiEdgeResult, ...],
) -> tuple[str, ...]:
    entity_ids: list[str] = []
    for edge in edges:
        entry_uuid, grounded = _entry_endpoint(query, edge)
        if grounded:
            entity_ids.append(entry_uuid)
    return tuple(dict.fromkeys(entity_ids))


def _merge_ranked_edges(
    ranked_edges: tuple[GraphitiEdgeResult, ...],
    adjacent_edges: tuple[GraphitiEdgeResult, ...],
) -> tuple[GraphitiEdgeResult, ...]:
    merged: list[GraphitiEdgeResult] = []
    seen: set[str] = set()
    for edge in (*ranked_edges, *adjacent_edges):
        if edge.edge_uuid in seen:
            continue
        seen.add(edge.edge_uuid)
        merged.append(replace(edge, rank=len(merged) + 1))
    return tuple(merged)


def _with_endpoints(
    ranked_edges: tuple[GraphitiEdgeResult, ...],
    hydrated_edges: tuple[GraphitiEdgeResult, ...],
) -> tuple[GraphitiEdgeResult, ...]:
    hydrated_by_id = {edge.edge_uuid: edge for edge in hydrated_edges}
    return tuple(
        replace(
            edge,
            relation_type=hydrated.relation_type or edge.relation_type,
            source_entity_uuid=hydrated.source_entity_uuid,
            source_entity_name=hydrated.source_entity_name,
            target_entity_uuid=hydrated.target_entity_uuid,
            target_entity_name=hydrated.target_entity_name,
        )
        if (hydrated := hydrated_by_id.get(edge.edge_uuid)) is not None
        else edge
        for edge in ranked_edges
    )


def _rank_graphiti_paths(
    query: str,
    ranked_edges: tuple[GraphitiEdgeResult, ...],
    adjacent_edges: tuple[GraphitiEdgeResult, ...],
    *,
    limit: int,
    seed_entity_uuids: tuple[str, ...] = (),
) -> tuple[GraphitiPathResult, ...]:
    """Enumerate simple one-to-three-hop paths from query-grounded nodes."""

    all_edges = _merge_ranked_edges(ranked_edges, adjacent_edges)
    usable = tuple(
        edge
        for edge in all_edges
        if edge.source_entity_uuid
        and edge.target_entity_uuid
        and edge.source_entity_uuid != edge.target_entity_uuid
    )
    adjacency: dict[str, list[GraphitiEdgeResult]] = {}
    for edge in usable:
        for endpoint in edge.endpoint_uuids:
            adjacency.setdefault(endpoint, []).append(edge)
    folded_query = query.casefold()
    explicit_chain_markers = (
        "最终", "通过", "经由", "间接", "依赖链", "关系链", "起源于",
        "through", "ultimately", "originated",
    )
    relation_markers = (
        "托管", "支持", "负责", "依赖", "建立在", "采用", "许可证", "仓库",
        "文档", "属于", "host", "support", "depend", "built on", "license",
        "repository", "documentation", "belong",
    )
    explicit_chain_query = any(
        marker in folded_query for marker in explicit_chain_markers
    )
    chain_query = explicit_chain_query or sum(
        marker in folded_query for marker in relation_markers
    ) >= 2
    candidates: list[
        tuple[tuple[int, int, int, int, int, str], str, tuple[GraphitiEdgeResult, ...], bool]
    ] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    def add_path(
        entry_uuid: str,
        hops: tuple[GraphitiEdgeResult, ...],
        *,
        seed_entry: bool,
        reverse_count: int,
    ) -> None:
        edge_ids = tuple(edge.edge_uuid for edge in hops)
        key = (entry_uuid, edge_ids)
        if key in seen:
            return
        seen.add(key)
        path_id = _graphiti_path_id(entry_uuid, edge_ids)
        depth_order = -len(hops) if chain_query else len(hops)
        candidates.append(
            (
                (
                    0 if seed_entry else 1,
                    depth_order,
                    max(edge.rank for edge in hops),
                    sum(edge.rank for edge in hops),
                    reverse_count,
                    path_id,
                ),
                entry_uuid,
                hops,
                seed_entry,
            )
        )

    def walk(
        entry_uuid: str,
        current_uuid: str,
        hops: tuple[GraphitiEdgeResult, ...],
        visited_nodes: frozenset[str],
        reverse_count: int,
        candidate_start: int,
    ) -> None:
        # A dense early seed must not consume the entire enumeration budget and
        # suppress later query-grounded entities. Keep each seed independently
        # bounded; the seed list itself is bounded below.
        if len(candidates) - candidate_start >= max(limit * 16, 64):
            return
        if hops:
            add_path(
                entry_uuid,
                hops,
                seed_entry=True,
                reverse_count=reverse_count,
            )
        if len(hops) == 3:
            return
        for edge in adjacency.get(current_uuid, ()):
            if any(edge.edge_uuid == hop.edge_uuid for hop in hops):
                continue
            source_uuid, target_uuid = edge.endpoint_uuids
            next_uuid = target_uuid if current_uuid == source_uuid else source_uuid
            if next_uuid in visited_nodes:
                continue
            walk(
                entry_uuid,
                next_uuid,
                (*hops, edge),
                visited_nodes | {next_uuid},
                reverse_count + int(current_uuid != source_uuid),
                candidate_start,
            )

    resolved_seed_uuids = seed_entity_uuids or tuple(
        dict.fromkeys(
            entry_uuid
            for edge in usable
            for entry_uuid, grounded in (_entry_endpoint(query, edge),)
            if grounded
        )
    )
    if resolved_seed_uuids:
        for entry_uuid in resolved_seed_uuids[:8]:
            candidate_start = len(candidates)
            walk(
                entry_uuid,
                entry_uuid,
                (),
                frozenset({entry_uuid}),
                0,
                candidate_start,
            )
    else:
        usable_by_id = {edge.edge_uuid: edge for edge in usable}
        for ranked_edge in ranked_edges:
            edge = usable_by_id.get(ranked_edge.edge_uuid)
            if edge is None:
                continue
            entry_uuid, grounded = _entry_endpoint(query, edge)
            add_path(entry_uuid, (edge,), seed_entry=grounded, reverse_count=0)

    candidates.sort(key=lambda item: item[0])
    # Preserve query-grounded seed diversity first, then alternative first-hop
    # hypotheses. This prevents one dense neighborhood from monopolizing K.
    selected = []
    selected_ids: set[str] = set()
    selected_entries: set[str] = set()
    reservation_order = (
        resolved_seed_uuids[:8]
        if resolved_seed_uuids
        else tuple(dict.fromkeys(candidate[1] for candidate in candidates))
    )
    for entry_uuid in reservation_order:
        entry_candidates = tuple(
            item for item in candidates if item[1] == entry_uuid
        )
        if explicit_chain_query:
            candidate = entry_candidates[0] if entry_candidates else None
        else:
            # An inferred compound query can mention several independent
            # relations. Reserve the cheapest direct fact for every grounded
            # entity before spending the evidence budget on longer paths.
            candidate = min(
                entry_candidates,
                key=lambda item: (
                    len(item[2]),
                    max(edge.rank for edge in item[2]),
                    sum(edge.rank for edge in item[2]),
                    item[0][-1],
                ),
                default=None,
            )
        if candidate is None:
            continue
        path_id = candidate[0][-1]
        selected_entries.add(entry_uuid)
        selected_ids.add(path_id)
        selected.append(candidate)
        if len(selected) == limit:
            break
    first_edges: set[str] = set()
    for candidate in candidates:
        path_id = candidate[0][-1]
        first_edge = candidate[2][0].edge_uuid
        if path_id in selected_ids or first_edge in first_edges:
            continue
        first_edges.add(first_edge)
        selected_ids.add(path_id)
        selected.append(candidate)
        if len(selected) == limit:
            break
    if len(selected) < limit:
        selected.extend(
            item
            for item in candidates
            if item[0][-1] not in selected_ids
        )
    selected = selected[:limit]
    return tuple(
        GraphitiPathResult(
            path_id=path_id,
            entry_entity_uuid=entry_uuid,
            hops=hops,
            rank=rank,
            seed_entry=seed_entry,
        )
        for rank, (_sort_key, entry_uuid, hops, seed_entry) in enumerate(
            selected,
            start=1,
        )
        for path_id in (_graphiti_path_id(entry_uuid, tuple(hop.edge_uuid for hop in hops)),)
    )


def _graphiti_path_id(entry_uuid: str, edge_uuids: tuple[str, ...]) -> str:
    return hashlib.sha256(
        f"{entry_uuid}:{':'.join(edge_uuids)}".encode("utf-8")
    ).hexdigest()


def graphiti_episode_uuid(
    build: GraphitiBuildSnapshot,
    chunk: GraphChunkSource,
) -> str:
    """Return the stable Graphiti identity for one immutable build chunk."""

    identity = f"{chunk.index_chunk_id}:{chunk.content_hash}"
    return str(uuid5(build.build_id, identity))


def _graphiti_modules() -> SimpleNamespace:
    global _GRAPHITI
    if _GRAPHITI is not None:
        return _GRAPHITI
    from graphiti_core.cross_encoder.client import CrossEncoderClient
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.embedder.client import EmbedderClient
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.errors import NodeNotFoundError
    from graphiti_core.graphiti import Graphiti, RELEVANT_SCHEMA_LIMIT
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.nodes import EpisodeType, EpisodicNode
    from graphiti_core.search.search_config_recipes import (
        EDGE_HYBRID_SEARCH_NODE_DISTANCE,
        EDGE_HYBRID_SEARCH_RRF,
        NODE_HYBRID_SEARCH_RRF,
    )
    from graphiti_core.search.search_config import EdgeSearchMethod
    from openai import AsyncOpenAI

    class NeverRerank(CrossEncoderClient):
        async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
            raise RuntimeError("Graphiti cross-encoder is disabled for RRF search")

    class BoundedEmbedder(EmbedderClient):
        def __init__(
            self,
            inner: EmbedderClient,
            limit: int,
            provider_semaphore: asyncio.Semaphore,
        ) -> None:
            self._inner = inner
            self._limit = min(16, max(1, limit))
            self._provider_semaphore = provider_semaphore

        async def create(self, input_data: Any) -> list[float]:
            async with self._provider_semaphore:
                return await self._inner.create(input_data)

        async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
            vectors: list[list[float]] = []
            for start in range(0, len(input_data_list), self._limit):
                async with self._provider_semaphore:
                    vectors.extend(
                        await self._inner.create_batch(
                            input_data_list[start : start + self._limit]
                        )
                    )
            return vectors

    _GRAPHITI = SimpleNamespace(
        AsyncOpenAI=AsyncOpenAI,
        BoundedEmbedder=BoundedEmbedder,
        EDGE_HYBRID_SEARCH_NODE_DISTANCE=EDGE_HYBRID_SEARCH_NODE_DISTANCE,
        EDGE_HYBRID_SEARCH_RRF=EDGE_HYBRID_SEARCH_RRF,
        EdgeSearchMethod=EdgeSearchMethod,
        NODE_HYBRID_SEARCH_RRF=NODE_HYBRID_SEARCH_RRF,
        EpisodeType=EpisodeType,
        EpisodicNode=EpisodicNode,
        FalkorDriver=FalkorDriver,
        Graphiti=Graphiti,
        RELEVANT_SCHEMA_LIMIT=RELEVANT_SCHEMA_LIMIT,
        LLMConfig=LLMConfig,
        NeverRerank=NeverRerank,
        NodeNotFoundError=NodeNotFoundError,
        OpenAIEmbedder=OpenAIEmbedder,
        OpenAIEmbedderConfig=OpenAIEmbedderConfig,
        OpenAIGenericClient=OpenAIGenericClient,
    )
    return _GRAPHITI


def _shield_upstream_logs() -> None:
    for name in ("graphiti_core", "graphiti_core.driver", "graphiti_core.llm_client"):
        logger = logging.getLogger(name)
        logger.propagate = False
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
