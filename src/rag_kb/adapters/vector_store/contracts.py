"""Application-facing vector retrieval contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_kb.domain import EmbeddingSpaceDefinition, RetrievalQueryPlan, VectorSearchResult


@runtime_checkable
class VectorStore(Protocol):
    async def has_space_role(
        self, plan: RetrievalQueryPlan, space_role: str
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
