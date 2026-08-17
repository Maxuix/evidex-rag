"""Privacy-shielded, build-scoped Graphiti client."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
import os
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

_GRAPHITI: SimpleNamespace | None = None

from rag_kb.domain import (
    GraphChunkSource,
    GraphitiBuildSnapshot,
    GraphitiEdgeResult,
    GraphitiSearchQuery,
)


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
            llm = modules.OpenAIGenericClient(
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
