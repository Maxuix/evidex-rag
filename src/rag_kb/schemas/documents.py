"""Public document lifecycle API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
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
