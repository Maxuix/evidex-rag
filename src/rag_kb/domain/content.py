"""Framework-independent knowledge-base and document lifecycle facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
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
    model_profile_revision_id: UUID | None = None
    dimension_request_mode: str = "explicit"


class EmbeddingExecutionMode(StrEnum):
    TEXT_ONLY = "text_only"
    DUAL_SPACE_MULTIMODAL = "dual_space_multimodal"
    UNIFIED_MULTIMODAL = "unified_multimodal"


def derive_embedding_execution_mode(
    text_space_id: UUID,
    cross_modal_space_id: UUID | None,
) -> EmbeddingExecutionMode:
    if cross_modal_space_id is None:
        return EmbeddingExecutionMode.TEXT_ONLY
    if cross_modal_space_id == text_space_id:
        return EmbeddingExecutionMode.UNIFIED_MULTIMODAL
    return EmbeddingExecutionMode.DUAL_SPACE_MULTIMODAL


@dataclass(frozen=True, slots=True)
class IndexProfileDefinition:
    parser_config: dict[str, Any]
    chunking_config: dict[str, Any]
    enrichment_config: dict[str, Any] = field(default_factory=dict)
    representation_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class KnowledgeBaseAutoQASummary:
    enabled: bool = False
    questions_per_chunk: int = 5
    model_profile_revision_id: UUID | None = None
    model_name: str | None = None
    model_revision: int | None = None


@dataclass(frozen=True, slots=True)
class KnowledgeBase:
    id: UUID
    workspace_id: UUID
    name: str
    source_change_seq: int
    active_index_revision_id: UUID
    embedding_space_id: UUID
    chunking_config: dict[str, Any]
    retrieval_defaults: dict[str, Any]
    answer_policy_defaults: dict[str, Any]
    provisioned_at: datetime
    created_at: datetime
    updated_at: datetime
    parser_config: dict[str, Any] = field(default_factory=dict)
    embedding: KnowledgeBaseEmbeddingSummary | None = None
    deleted_at: datetime | None = None
    description: str = ""
    auto_qa: KnowledgeBaseAutoQASummary = field(default_factory=KnowledgeBaseAutoQASummary)


@dataclass(frozen=True, slots=True)
class EmbeddingRoleSummary:
    embedding_space_id: UUID
    profile_revision_id: UUID | None
    dimension: int


@dataclass(frozen=True, slots=True)
class KnowledgeBaseEmbeddingSummary:
    strategy: str
    text: EmbeddingRoleSummary
    cross_modal: EmbeddingRoleSummary | None = None


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
class DocumentIndexSummary:
    indexed_document_version_id: UUID
    index_revision_id: UUID
    build_status: str
    serving_status: str
    unit_count: int | None
    asset_count: int | None
    representation_count: int | None
    composite_chunk_count: int | None = None
    visual_unit_count: int | None = None
    relation_count: int | None = None
    text_representation_count: int | None = None
    native_image_representation_count: int | None = None
    table_representation_count: int | None = None


@dataclass(frozen=True, slots=True)
class DocumentDetail:
    document: Document
    index: DocumentIndexSummary | None


@dataclass(frozen=True, slots=True)
class DocumentChunkAsset:
    id: UUID
    media_type: str
    checksum_sha256: str
    width: int | None
    height: int | None


@dataclass(frozen=True, slots=True)
class DocumentChunkRelation:
    visual_unit_id: UUID
    asset: DocumentChunkAsset
    relation_type: str
    confidence_micros: int
    provenance: str
    figure_label: str | None


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    id: UUID
    ordinal: int
    modality: str
    content: str
    token_count: int
    source_location: dict[str, Any]
    hierarchy: dict[str, Any]
    source_metadata: dict[str, Any]
    evidence_group_key: str | None
    representations: tuple[str, ...]
    asset: DocumentChunkAsset | None = None
    related_visuals: tuple[DocumentChunkRelation, ...] = ()
    excluded_at: datetime | None = None
    generated_questions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DocumentChunkInspection:
    document_id: UUID
    document_version_id: UUID
    indexed_document_version_id: UUID
    index_revision_id: UUID
    total_chunks: int
    items: tuple[DocumentChunk, ...]
    next_values: tuple[str, ...] | None = None


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
    failure_code: str | None
    failed_at: datetime | None
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


class DuplicateDocumentError(ResourceNameConflictError):
    """A new document has the same current content as an existing document."""

    def __init__(self, existing_document_id: UUID) -> None:
        self.existing_document_id = existing_document_id
        super().__init__("document content already exists")


class ResourceStateConflictError(ContentLifecycleError):
    pass


class IdempotencyKeyReusedError(ContentLifecycleError):
    pass
