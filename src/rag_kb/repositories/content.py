"""Async persistence contracts for content lifecycle aggregates."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from rag_kb.domain import (
    ContentMutation,
    Document,
    DocumentChunkInspection,
    DocumentDetail,
    DocumentMutationResult,
    DocumentSource,
    EmbeddingSpaceDefinition,
    IdempotencyScope,
    IndexProfileDefinition,
    KnowledgeBase,
    Page,
    PendingFileMutation,
    SourceFileCleanup,
    SourceFileReference,
)


class KnowledgeBaseRepository(Protocol):
    async def create(
        self,
        *,
        name: str,
        retrieval_defaults: dict[str, Any],
        answer_policy_defaults: dict[str, Any],
        embedding_space: EmbeddingSpaceDefinition,
        cross_modal_embedding_space: EmbeddingSpaceDefinition | None,
        index_profile: IndexProfileDefinition,
    ) -> KnowledgeBase: ...

    async def get(
        self, kb_id: UUID, *, include_deleted: bool = False
    ) -> KnowledgeBase | None: ...

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
        answer_policy_defaults: dict[str, Any] | None,
    ) -> KnowledgeBase | None: ...

    async def soft_delete(self, kb_id: UUID) -> KnowledgeBase | None: ...


class DocumentRepository(Protocol):
    async def get(self, document_id: UUID) -> Document | None: ...

    async def get_detail(self, document_id: UUID) -> DocumentDetail | None: ...

    async def inspect_chunks(
        self,
        document_id: UUID,
        *,
        limit: int,
        after: tuple[str, ...] | None,
    ) -> DocumentChunkInspection | None: ...

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

    async def exclude_chunk(
        self, *, document_id: UUID, chunk_id: UUID
    ) -> datetime | None: ...


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

    async def add_indexing_retry(
        self,
        *,
        scope: IdempotencyScope,
        request_hash: str,
        kb_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
        indexed_document_version_id: UUID,
        index_revision_id: UUID,
        job_id: UUID,
    ) -> ContentMutation: ...


class FileConsistencyRepository(Protocol):
    async def delete_expired_cleanup_records(
        self, *, before: datetime, limit: int
    ) -> int: ...

    async def list_references(self) -> tuple[SourceFileReference, ...]: ...

    async def list_pending_mutations(
        self, *, limit: int
    ) -> tuple[PendingFileMutation, ...]: ...

    async def list_due_cleanup(
        self, *, now: datetime, limit: int
    ) -> tuple[SourceFileCleanup, ...]: ...

    async def schedule_cleanup(
        self,
        *,
        document_version_id: UUID,
        storage_uri: str,
        reason: str,
    ) -> None: ...

    async def complete_cleanup(self, cleanup_id: UUID, *, now: datetime) -> bool: ...

    async def fail_cleanup(
        self,
        cleanup_id: UUID,
        *,
        expected_attempt_count: int,
        error_code: str,
        next_attempt_at: datetime,
        terminal: bool,
    ) -> bool: ...

    async def fail_pending_file_mutation(
        self,
        *,
        scope: IdempotencyScope,
        document_version_id: UUID,
        failure_code: str,
        failed_at: datetime,
    ) -> bool: ...

    async def compensate_missing_file(self, document_version_id: UUID) -> bool: ...
