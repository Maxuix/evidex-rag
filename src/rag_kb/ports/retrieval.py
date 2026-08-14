"""Application-facing lexical and vector retrieval contracts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from rag_kb.domain import (
    AdjacentChunkQuery,
    AdjacentChunkResult,
    EmbeddingSpaceDefinition,
    GraphEntityCandidate,
    GraphEntityLookupQuery,
    GraphConfigSnapshot,
    GraphTraversalQuery,
    GraphTraversalResult,
    LexicalSearchResult,
    RetrievalQueryPlan,
    VectorSearchResult,
)


@runtime_checkable
class VectorStore(Protocol):
    async def adjacent_chunks(
        self,
        query: AdjacentChunkQuery,
    ) -> AdjacentChunkResult | None: ...

    async def resolve_spaces(
        self,
        plan: RetrievalQueryPlan,
    ) -> dict[str, EmbeddingSpaceDefinition]: ...

    async def resolve_space(
        self,
        plan: RetrievalQueryPlan,
        space_role: str,
    ) -> EmbeddingSpaceDefinition | None: ...

    async def has_space_role(
        self,
        plan: RetrievalQueryPlan,
        space_role: str,
    ) -> bool: ...

    async def search(
        self,
        plan: RetrievalQueryPlan,
        query_embedding: tuple[float, ...],
    ) -> VectorSearchResult | None: ...

    async def search_space(
        self,
        plan: RetrievalQueryPlan,
        query_embedding: tuple[float, ...],
        *,
        space_role: str,
        representation_kinds: tuple[str, ...],
        expected_space: EmbeddingSpaceDefinition,
    ) -> VectorSearchResult | None: ...


@runtime_checkable
class LexicalStore(Protocol):
    async def search(
        self,
        plan: RetrievalQueryPlan,
        query: str,
        query_embedding: tuple[float, ...],
        *,
        analyzer_version: str,
        query_version: str,
        candidate_count: int,
    ) -> LexicalSearchResult | None: ...


@runtime_checkable
class GraphStore(Protocol):
    """Fixed-scope graph lookup; callers cannot submit arbitrary SQL or depth."""

    async def get_config(
        self, workspace_id: UUID, knowledge_base_id: UUID
    ) -> GraphConfigSnapshot | None: ...

    async def find_entity_candidates(
        self, query: GraphEntityLookupQuery
    ) -> tuple[GraphEntityCandidate, ...]: ...

    async def traverse(
        self, query: GraphTraversalQuery
    ) -> GraphTraversalResult | None: ...
