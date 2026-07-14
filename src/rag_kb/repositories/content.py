"""Async persistence contracts for content lifecycle aggregates."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from rag_kb.domain import (
    ContentMutation,
    Document,
    DocumentMutationResult,
    DocumentSource,
    EmbeddingSpaceDefinition,
    IdempotencyScope,
    IndexProfileDefinition,
    KnowledgeBase,
    Page,
)


@runtime_checkable
class KnowledgeBaseRepository(Protocol):
    async def create(
        self,
        *,
        name: str,
        retrieval_defaults: dict[str, Any],
        embedding_space: EmbeddingSpaceDefinition,
        index_profile: IndexProfileDefinition,
    ) -> KnowledgeBase: ...

    async def get(self, kb_id: UUID) -> KnowledgeBase | None: ...

    async def list(
        self,
        *,
        limit: int,
        sort: str,
        after: tuple[str, ...] | None,
    ) -> Page[KnowledgeBase]: ...

    async def update(
        self,
        kb_id: UUID,
        *,
        name: str | None,
        retrieval_defaults: dict[str, Any] | None,
    ) -> KnowledgeBase | None: ...


@runtime_checkable
class DocumentRepository(Protocol):
    async def get(self, document_id: UUID) -> Document | None: ...

    async def list(
        self,
        *,
        kb_id: UUID,
        limit: int,
        sort: str,
        after: tuple[str, ...] | None,
    ) -> Page[Document]: ...

    async def reserve_version(
        self,
        *,
        kb_id: UUID,
        document_id: UUID | None,
        display_name: str,
        source: DocumentSource,
    ) -> DocumentMutationResult: ...

    async def activate_version(
        self,
        *,
        document_id: UUID,
        document_version_id: UUID,
    ) -> DocumentMutationResult: ...

    async def soft_delete(self, document_id: UUID) -> DocumentMutationResult | None: ...


@runtime_checkable
class ContentMutationRepository(Protocol):
    async def lock(self, scope: IdempotencyScope) -> None: ...

    async def get(self, scope: IdempotencyScope) -> ContentMutation | None: ...

    async def add(
        self,
        *,
        scope: IdempotencyScope,
        request_hash: str,
        operation: str,
        status: str,
        result: DocumentMutationResult | KnowledgeBase,
    ) -> ContentMutation: ...

    async def complete(
        self,
        scope: IdempotencyScope,
        result: DocumentMutationResult,
    ) -> ContentMutation: ...
