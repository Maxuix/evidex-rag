"""Application-facing lexical and vector retrieval contracts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_kb.domain import (
    EmbeddingSpaceDefinition,
    LexicalSearchResult,
    RetrievalQueryPlan,
    VectorSearchResult,
)


@runtime_checkable
class VectorStore(Protocol):
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
