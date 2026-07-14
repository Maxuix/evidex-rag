"""Framework-independent knowledge-base and document lifecycle facts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generic, TypeVar
from uuid import UUID

from rag_kb.domain.idempotency import IdempotencyScope


ItemT = TypeVar("ItemT")


@dataclass(frozen=True, slots=True)
class Page(Generic[ItemT]):
    items: tuple[ItemT, ...]
    next_values: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingSpaceDefinition:
    provider_identity: str
    endpoint_identity: str
    requested_model: str
    resolved_model: str
    model_version: str
    deployment_revision: str | None
    dimension: int
    distance_metric: str
    vector_data_type: str
    normalization: str
    configuration_fingerprint: str
    tokenizer_fingerprint: str | None
    compatibility_fingerprint: str


@dataclass(frozen=True, slots=True)
class IndexProfileDefinition:
    parser_config: dict[str, Any]
    chunking_config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class KnowledgeBase:
    id: UUID
    workspace_id: UUID
    name: str
    source_change_seq: int
    active_index_revision_id: UUID
    embedding_space_id: UUID
    retrieval_defaults: dict[str, Any]
    provisioned_at: datetime
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class DocumentVersion:
    id: UUID
    document_id: UUID
    version_number: int
    source_status: str
    checksum_sha256: str
    storage_uri: str
    original_filename: str
    media_type: str
    size_bytes: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Document:
    id: UUID
    workspace_id: UUID
    kb_id: UUID
    display_name: str
    current_version: DocumentVersion | None
    deleted_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class DocumentSource:
    checksum_sha256: str
    storage_uri: str
    original_filename: str
    media_type: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class DocumentMutationResult:
    document: Document
    document_version_id: UUID | None = None
    source_change_id: UUID | None = None
    source_change_seq: int | None = None
    indexed_document_version_id: UUID | None = None
    index_revision_id: UUID | None = None
    job_id: UUID | None = None
    job_status: str | None = None


@dataclass(frozen=True, slots=True)
class ContentMutation:
    scope: IdempotencyScope
    request_hash: str
    operation: str
    status: str
    kb_id: UUID | None
    document_id: UUID | None
    document_version_id: UUID | None
    source_change_id: UUID | None
    indexed_document_version_id: UUID | None
    index_revision_id: UUID | None
    job_id: UUID | None


class ContentLifecycleError(RuntimeError):
    """Base class for safe lifecycle failures mapped at the API boundary."""


class ResourceNotFoundError(ContentLifecycleError):
    pass


class ResourceNameConflictError(ContentLifecycleError):
    pass


class ResourceStateConflictError(ContentLifecycleError):
    pass


class IdempotencyKeyReusedError(ContentLifecycleError):
    pass
