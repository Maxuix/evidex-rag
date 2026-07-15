"""Application-facing vector retrieval contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_kb.domain import RetrievalQueryPlan, VectorSearchResult


@runtime_checkable
class VectorStore(Protocol):
    async def search(
        self,
        plan: RetrievalQueryPlan,
        query_embedding: tuple[float, ...],
    ) -> VectorSearchResult | None: ...
