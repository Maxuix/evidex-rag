"""Application-facing lexical and vector retrieval contracts."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from rag_kb.domain import (
    AdjacentChunkQuery,
    AdjacentChunkResult,
    EmbeddingSpaceDefinition,
    GraphConfigSnapshot,
    GraphitiBuildSnapshot,
    GraphitiEdgeResult,
    GraphitiPathResult,
    GraphTraversalResult,
    LexicalManifestStatus,
    LexicalSearchResult,
    RetrievalQueryPlan,
    ServingDocumentList,
    ServingScopeQuery,
    VectorSearchResult,
)


class VectorStore(Protocol):
    async def rerank_document_contexts(
        self,
        *,
        workspace_id: UUID,
        knowledge_base_id: UUID,
        index_revision_id: UUID,
        indexed_document_version_ids: tuple[UUID, ...],
    ) -> dict[UUID, str]: ...

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

    async def list_serving_documents(
        self, query: ServingScopeQuery
    ) -> ServingDocumentList | None: ...


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

    async def manifest_status(
        self, query: ServingScopeQuery
    ) -> LexicalManifestStatus | None: ...


class GraphStore(Protocol):
    """Fixed-scope graph lookup; callers cannot submit arbitrary SQL or depth."""

    async def get_config(
        self, workspace_id: UUID, knowledge_base_id: UUID
    ) -> GraphConfigSnapshot | None: ...

    async def get_active_graphiti_build(
        self, workspace_id: UUID, knowledge_base_id: UUID
    ) -> GraphitiBuildSnapshot | None: ...

    async def first_graphiti_episode_uuid(
        self, workspace_id: UUID, knowledge_base_id: UUID, build_id: UUID
    ) -> str | None: ...

    async def hydrate_graphiti_edges(
        self,
        *,
        workspace_id: UUID,
        knowledge_base_id: UUID,
        build_id: UUID,
        index_revision_id: UUID,
        edges: tuple[GraphitiEdgeResult, ...],
    ) -> GraphTraversalResult | None: ...

    async def hydrate_graphiti_paths(
        self,
        *,
        workspace_id: UUID,
        knowledge_base_id: UUID,
        build_id: UUID,
        index_revision_id: UUID,
        paths: tuple[GraphitiPathResult, ...],
    ) -> GraphTraversalResult | None: ...

    async def schedule_graphiti_rebuild(
        self,
        workspace_id: UUID,
        knowledge_base_id: UUID,
        failed_build_id: UUID,
    ) -> bool: ...
