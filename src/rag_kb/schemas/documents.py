"""Public document lifecycle API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from rag_kb.schemas.common import OpaqueCursor, PublicSchema


class DocumentVersionResponse(PublicSchema):
    id: UUID
    version_number: int
    source_status: Literal["available", "unavailable", "deleted"]
    checksum_sha256: str
    original_filename: str
    media_type: str
    size_bytes: int
    created_at: datetime


class DocumentResponse(PublicSchema):
    id: UUID
    kb_id: UUID
    display_name: str
    current_version: DocumentVersionResponse | None
    deleted_at: datetime | None
    created_at: datetime
    updated_at: datetime


class DocumentIndexSummaryResponse(PublicSchema):
    indexed_document_version_id: UUID
    index_revision_id: UUID
    build_status: Literal["queued", "processing", "ready", "failed"]
    serving_status: Literal["candidate", "serving", "retired"]
    unit_count: int | None
    asset_count: int | None
    representation_count: int | None
    composite_chunk_count: int | None
    visual_unit_count: int | None
    relation_count: int | None
    text_representation_count: int | None
    native_image_representation_count: int | None
    table_representation_count: int | None


class DocumentDetailResponse(DocumentResponse):
    index: DocumentIndexSummaryResponse | None


class DocumentChunkAssetResponse(PublicSchema):
    id: UUID
    media_type: str
    checksum_sha256: str
    content_url: str
    width: int | None = None
    height: int | None = None


class DocumentChunkRelationResponse(PublicSchema):
    visual_unit_id: UUID
    asset: DocumentChunkAssetResponse
    relation_type: str
    confidence_micros: int
    provenance: str
    figure_label: str | None = None


class DocumentChunkResponse(PublicSchema):
    id: UUID
    ordinal: int
    modality: Literal["text", "image", "table"]
    content: str
    token_count: int
    source_location: dict[str, Any]
    hierarchy: dict[str, Any]
    source_metadata: dict[str, Any]
    evidence_group_key: str | None = None
    representations: tuple[str, ...]
    asset: DocumentChunkAssetResponse | None = None
    related_visuals: tuple[DocumentChunkRelationResponse, ...] = ()
    excluded_at: datetime | None = None
    generated_questions: tuple[str, ...] = ()


class DocumentChunkDeleteResponse(PublicSchema):
    document_id: UUID
    chunk_id: UUID
    excluded_at: datetime


class DocumentChunkInspectionResponse(PublicSchema):
    document_id: UUID
    document_version_id: UUID
    indexed_document_version_id: UUID
    index_revision_id: UUID
    total_chunks: int
    items: tuple[DocumentChunkResponse, ...]
    next_cursor: OpaqueCursor | None = None


class DocumentPage(PublicSchema):
    items: tuple[DocumentResponse, ...]
    next_cursor: OpaqueCursor | None = None


class DocumentDeleteResponse(PublicSchema):
    document: DocumentResponse
    source_change_id: UUID | None
    source_change_seq: int | None
    index_revision_id: UUID | None


class DocumentUploadResponse(PublicSchema):
    document: DocumentResponse
    document_version_id: UUID
    source_change_id: UUID
    source_change_seq: int
    indexed_document_version_id: UUID
    index_revision_id: UUID
    job_id: UUID
    job_status: Literal["queued"]
