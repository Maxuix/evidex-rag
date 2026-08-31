"""External Graphiti/FalkorDB contracts."""

from __future__ import annotations

from typing import Protocol

from rag_kb.domain import (
    GraphChunkSource,
    GraphitiBuildSnapshot,
    GraphitiEdgeResult,
    GraphitiPathResult,
    GraphitiSearchQuery,
)


class GraphitiGraph(Protocol):
    async def probe(
        self,
        build: GraphitiBuildSnapshot,
        *,
        episode_uuid: str | None = None,
        require_complete: bool = False,
    ) -> bool: ...

    async def add_episode(
        self, build: GraphitiBuildSnapshot, chunk: GraphChunkSource
    ) -> str: ...

    async def add_episodes_bulk(
        self,
        build: GraphitiBuildSnapshot,
        chunks: tuple[GraphChunkSource, ...],
    ) -> tuple[str, ...]: ...

    async def search(
        self, build: GraphitiBuildSnapshot, query: GraphitiSearchQuery
    ) -> tuple[GraphitiEdgeResult, ...]: ...

    async def search_paths(
        self, build: GraphitiBuildSnapshot, query: GraphitiSearchQuery
    ) -> tuple[GraphitiPathResult, ...]: ...

    async def delete_graph(self, build: GraphitiBuildSnapshot) -> None: ...
