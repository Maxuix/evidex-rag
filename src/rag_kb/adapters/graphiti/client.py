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
import unicodedata
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

_GRAPHITI: SimpleNamespace | None = None
SCHEMA_ECHO_MAX_ATTEMPTS = 3

from rag_kb.domain import (
    GraphChunkSource,
    GraphitiBuildSnapshot,
    GraphitiEdgeResult,
    GraphitiPathResult,
    GraphitiSearchQuery,
)


GRAPHITI_V2_EXTRACTION_INSTRUCTIONS = """
Extract only factual relations explicitly stated in the episode. Treat names,
aliases, organization names, people, places, products, projects, and stable
identifiers as possible relation participants. Preserve codes and short product
identifiers exactly. When the text explicitly states an alias, former name, or
renaming, classify the alias/former/pre-rename surface as AliasSurface and keep
both surface names as distinct nodes connected by that stated relation; do not
collapse the relation into a self-loop.

Do not extract document structure or authoring instructions as knowledge-graph
facts. In particular, ignore section labels, relation IDs, evidence-unit IDs,
filenames, headings that only organize the document, and generic descriptions
of how evidence or graph extraction should work. Do not infer missing entities,
relations, directions, dates, or world knowledge. Every extracted edge must be
supported by a specific sentence in this episode and must connect two distinct
participants named or unambiguously referenced in that sentence.
""".strip()

_GRAPHITI_ADJACENCY_MULTIPLIER = 8


class AliasSurfaceEntity(BaseModel):
    """An explicit alias surface that remains distinct from its canonical node."""


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


def is_schema_echo_payload(result: Any, response_model: Any) -> bool:
    if response_model is None or not isinstance(result, dict):
        return False
    required = required_model_field_names(response_model)
    return bool(required) and any(name not in result for name in required)


def append_schema_echo_repair_note(messages: Any, response_model: Any) -> None:
    if not messages:
        return
    last = messages[-1]
    content = getattr(last, "content", None)
    if not isinstance(content, str):
        return
    fields = required_model_field_names(response_model)
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
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._inner = inner
        self._max_attempts = max_attempts

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
                result = await self._inner.generate_response(messages, **first_kwargs)
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
                result = await raw(repaired, response_model, **raw_kwargs)
            if is_schema_echo_payload(result, response_model):
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


def as_graphiti_llm_client(
    inner: Any,
    *,
    max_attempts: int = SCHEMA_ECHO_MAX_ATTEMPTS,
) -> Any:
    """Preserve the inner Graphiti LLMClient type for pydantic isinstance checks."""

    repair = SchemaEchoRepairingLLMClient(inner, max_attempts=max_attempts)

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


class GraphitiRuntime:
    """Own one initialized Graphiti instance per immutable build."""

    def __init__(
        self,
        credentials: Callable[[GraphitiBuildSnapshot], Awaitable[GraphitiModelCredentials]],
        *,
        host: str = "127.0.0.1",
        port: int = 6379,
        probe_ttl_seconds: float = 15.0,
    ) -> None:
        self._credentials = credentials
        self._host = host
        self._port = port
        self._probe_ttl_seconds = probe_ttl_seconds
        self._clients: dict[str, tuple[Any, Any]] = {}
        self._client_lock = asyncio.Lock()
        self._probe_until: dict[str, float] = {}
        self._complete_probe_until: dict[str, float] = {}
        _shield_upstream_logs()

    async def probe(
        self,
        build: GraphitiBuildSnapshot,
        *,
        episode_uuid: str | None = None,
        require_complete: bool = False,
    ) -> bool:
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
        graphiti, _ = await self._client(build)
        modules = _graphiti_modules()
        result = await graphiti.add_episode(
            name=f"chunk-{chunk.ordinal}",
            episode_body=chunk.content,
            source_description=f"chunk {chunk.index_chunk_id}",
            reference_time=chunk.reference_time or datetime.now(UTC),
            source=modules.EpisodeType.text,
            group_id=build.group_id,
            entity_types={"AliasSurface": AliasSurfaceEntity},
            custom_extraction_instructions=GRAPHITI_V2_EXTRACTION_INSTRUCTIONS,
        )
        return str(result.episode.uuid)

    async def search(
        self, build: GraphitiBuildSnapshot, query: GraphitiSearchQuery
    ) -> tuple[GraphitiEdgeResult, ...]:
        graphiti, driver = await self._client(build)
        return await self._search_edges(graphiti, driver, build, query)

    async def search_paths(
        self, build: GraphitiBuildSnapshot, query: GraphitiSearchQuery
    ) -> tuple[GraphitiPathResult, ...]:
        """Resolve named query entities into bounded, connected graph paths."""

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
        first_hop_edges = (
            await self._adjacent_edges(
                driver,
                build,
                seed_entity_uuids,
                limit=min(
                    64,
                    max(
                        query.limit,
                        query.limit * _GRAPHITI_ADJACENCY_MULTIPLIER,
                    ),
                ),
            )
            if seed_entity_uuids
            else ()
        )
        ranked_first_hops = _merge_ranked_edges(ranked_edges, first_hop_edges)
        bridge_ids = _outward_bridge_entity_ids(
            query.query,
            ranked_first_hops,
            seed_entity_uuids=seed_entity_uuids,
        )
        adjacent_edges = (
            await self._adjacent_edges(
                driver,
                build,
                bridge_ids,
                limit=min(
                    64,
                    max(
                        query.limit,
                        query.limit * _GRAPHITI_ADJACENCY_MULTIPLIER,
                    ),
                ),
            )
            if bridge_ids
            else ()
        )
        return _rank_graphiti_paths(
            query.query,
            ranked_first_hops,
            adjacent_edges,
            limit=min(query.limit, 20),
            seed_entity_uuids=seed_entity_uuids,
        )

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
            GraphitiEdgeResult(
                edge_uuid=str(edge.uuid),
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
        config = modules.NODE_HYBRID_SEARCH_RRF.model_copy(
            update={"limit": query.limit}
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
            for rank, node in enumerate(result.nodes[: query.limit], start=1)
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

        _, driver = await self._client(build)
        records, _, _ = await driver.clone(database=build.group_id).execute_query(
            """
            MATCH (source:Entity)-[edge:RELATES_TO]->(target:Entity)
            RETURN source.uuid AS source_uuid,
                   source.name AS source_name,
                   edge.uuid AS edge_uuid,
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
                )
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
            )
            cached = (graphiti, driver)
            self._clients[build.group_id] = cached
            return cached


def _normalized_surface(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _edge_result_from_record(row: Any, rank: int) -> GraphitiEdgeResult:
    return GraphitiEdgeResult(
        edge_uuid=str(row["edge_uuid"]),
        fact=str(row.get("fact") or ""),
        episode_uuids=tuple(str(value) for value in (row.get("episodes") or ())),
        rank=rank,
        source_entity_uuid=str(row["source_uuid"]),
        source_entity_name=str(row.get("source_name") or ""),
        target_entity_uuid=str(row["target_uuid"]),
        target_entity_name=str(row.get("target_name") or ""),
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
        normalized_name = _normalized_surface(entity_name)
        if _query_mentions_entity(normalized_query, normalized_name):
            return entity_uuid, True
    return edge.source_entity_uuid, False


def _query_mentions_entity(normalized_query: str, normalized_name: str) -> bool:
    if len(normalized_name) < 2:
        return False
    if normalized_name in normalized_query:
        return True
    # Corporate/legal suffixes are frequently omitted in questions. Match the
    # longest available leading surface instead of a fixed short prefix: four
    # Han characters or eight other alphanumerics are the minimum. This keeps
    # the rule language-shape based and avoids a benchmark/legal-suffix list.
    has_han = any(
        "\u3400" <= character <= "\u9fff" for character in normalized_name
    )
    minimum = 4 if has_han else 8
    return any(
        normalized_name[:length] in normalized_query
        for length in range(len(normalized_name) - 1, minimum - 1, -1)
    )


def _outward_bridge_entity_ids(
    query: str,
    edges: tuple[GraphitiEdgeResult, ...],
    *,
    seed_entity_uuids: tuple[str, ...] = (),
) -> tuple[str, ...]:
    bridge_ids: list[str] = []
    for edge in edges:
        entry_uuid, grounded = _entry_endpoint(
            query,
            edge,
            seed_entity_uuids=seed_entity_uuids,
        )
        if not grounded or entry_uuid not in edge.endpoint_uuids:
            continue
        source_uuid, target_uuid = edge.endpoint_uuids
        bridge_ids.append(target_uuid if entry_uuid == source_uuid else source_uuid)
    return tuple(dict.fromkeys(bridge_ids))


def _grounded_entity_ids(
    query: str,
    entities: tuple[_GraphitiEntityResult, ...],
) -> tuple[str, ...]:
    normalized_query = _normalized_surface(query)
    return tuple(
        entity.entity_uuid
        for entity in entities
        if _query_mentions_entity(
            normalized_query,
            _normalized_surface(entity.name),
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
    """Create only connected paths that expand away from a query-grounded node."""

    candidates: list[
        tuple[tuple[int, int, int, int, str], str, tuple[GraphitiEdgeResult, ...], bool]
    ] = []
    seen: set[tuple[str, ...]] = set()
    ranked_edge_rank = {edge.edge_uuid: edge.rank for edge in ranked_edges}
    adjacent_rank = {edge.edge_uuid: edge.rank for edge in adjacent_edges}
    for edge in ranked_edges:
        if (
            not edge.source_entity_uuid
            or not edge.target_entity_uuid
            or edge.source_entity_uuid == edge.target_entity_uuid
        ):
            continue
        entry_uuid, seed_entry = _entry_endpoint(
            query,
            edge,
            seed_entity_uuids=seed_entity_uuids,
        )
        one_hop_key = (edge.edge_uuid,)
        if one_hop_key not in seen:
            seen.add(one_hop_key)
            path_id = _graphiti_path_id(entry_uuid, one_hop_key)
            candidates.append(
                (
                    (
                        0 if seed_entry else 1,
                        1,
                        edge.rank,
                        edge.rank,
                        path_id,
                    ),
                    entry_uuid,
                    (edge,),
                    seed_entry,
                )
            )
        if not seed_entry:
            continue
        bridge_uuid = (
            edge.target_entity_uuid
            if entry_uuid == edge.source_entity_uuid
            else edge.source_entity_uuid
        )
        for adjacent in adjacent_edges:
            if adjacent.edge_uuid == edge.edge_uuid:
                continue
            endpoints = set(adjacent.endpoint_uuids)
            if (
                bridge_uuid not in endpoints
                or entry_uuid in endpoints
                or len(endpoints) != 2
            ):
                continue
            path_key = tuple(sorted((edge.edge_uuid, adjacent.edge_uuid)))
            if path_key in seen:
                continue
            seen.add(path_key)
            path_id = _graphiti_path_id(entry_uuid, (edge.edge_uuid, adjacent.edge_uuid))
            candidates.append(
                (
                    (
                        0,
                        0,
                        edge.rank,
                        ranked_edge_rank.get(
                            adjacent.edge_uuid,
                            len(ranked_edges)
                            + adjacent_rank.get(adjacent.edge_uuid, limit + 1),
                        ),
                        path_id,
                    ),
                    entry_uuid,
                    (edge, adjacent),
                    True,
                )
            )
    candidates.sort(key=lambda item: item[0])
    return tuple(
        GraphitiPathResult(
            path_id=path_id,
            entry_entity_uuid=entry_uuid,
            hops=hops,
            rank=rank,
            seed_entry=seed_entry,
        )
        for rank, (_sort_key, entry_uuid, hops, seed_entry) in enumerate(
            candidates[:limit],
            start=1,
        )
        for path_id in (_graphiti_path_id(entry_uuid, tuple(hop.edge_uuid for hop in hops)),)
    )


def _graphiti_path_id(entry_uuid: str, edge_uuids: tuple[str, ...]) -> str:
    return hashlib.sha256(
        f"{entry_uuid}:{':'.join(edge_uuids)}".encode("utf-8")
    ).hexdigest()


def _graphiti_modules() -> SimpleNamespace:
    global _GRAPHITI
    if _GRAPHITI is not None:
        return _GRAPHITI
    from graphiti_core.cross_encoder.client import CrossEncoderClient
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.embedder.client import EmbedderClient
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.errors import NodeNotFoundError
    from graphiti_core.graphiti import Graphiti
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.nodes import EpisodeType, EpisodicNode
    from graphiti_core.search.search_config_recipes import (
        EDGE_HYBRID_SEARCH_RRF,
        NODE_HYBRID_SEARCH_RRF,
    )
    from openai import AsyncOpenAI

    class NeverRerank(CrossEncoderClient):
        async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
            raise RuntimeError("Graphiti cross-encoder is disabled for RRF search")

    class BoundedEmbedder(EmbedderClient):
        def __init__(self, inner: EmbedderClient, limit: int) -> None:
            self._inner = inner
            self._limit = min(16, max(1, limit))

        async def create(self, input_data: Any) -> list[float]:
            return await self._inner.create(input_data)

        async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
            vectors: list[list[float]] = []
            for start in range(0, len(input_data_list), self._limit):
                vectors.extend(
                    await self._inner.create_batch(
                        input_data_list[start : start + self._limit]
                    )
                )
            return vectors

    _GRAPHITI = SimpleNamespace(
        AsyncOpenAI=AsyncOpenAI,
        BoundedEmbedder=BoundedEmbedder,
        EDGE_HYBRID_SEARCH_RRF=EDGE_HYBRID_SEARCH_RRF,
        NODE_HYBRID_SEARCH_RRF=NODE_HYBRID_SEARCH_RRF,
        EpisodeType=EpisodeType,
        EpisodicNode=EpisodicNode,
        FalkorDriver=FalkorDriver,
        Graphiti=Graphiti,
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
