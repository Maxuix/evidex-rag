"""Privacy-shielded, build-scoped Graphiti client."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import copy
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
import os
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

_GRAPHITI: SimpleNamespace | None = None
SCHEMA_ECHO_MAX_ATTEMPTS = 3

from rag_kb.domain import (
    GraphChunkSource,
    GraphitiBuildSnapshot,
    GraphitiEdgeResult,
    GraphitiSearchQuery,
)


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
        )
        return str(result.episode.uuid)

    async def search(
        self, build: GraphitiBuildSnapshot, query: GraphitiSearchQuery
    ) -> tuple[GraphitiEdgeResult, ...]:
        graphiti, driver = await self._client(build)
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
            )
            for rank, edge in enumerate(result.edges[: query.limit], start=1)
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
    from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
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
