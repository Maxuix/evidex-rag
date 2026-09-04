"""Public indexing status and explicit retry schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Union
from uuid import UUID

from pydantic import Field

from rag_kb.domain import ErrorCode
from rag_kb.schemas.common import PublicSchema
from rag_kb.schemas.common import OpaqueCursor


class IndexingErrorResponse(PublicSchema):
    code: ErrorCode
    detail: dict[str, Any]


class PdfParsingProgressResponse(PublicSchema):
    schema_version: Literal["pdf_parsing_progress_v1"]
    stage: str
    total_pages: int
    completed_pages: int
    segment_number: int
    segment_count: int
    page_from: int
    page_to: int
    stage_pages: dict[str, int]
    ocr_pages: int
    ocr_regions: int
    table_candidates: int
    elapsed_ms: int
    child_peak_rss_bytes: int | None = None


class AutoQAGenerationProgressResponse(PublicSchema):
    schema_version: Literal["auto_qa_generation_v1"]
    eligible_chunks: int
    processed_chunks: int
    question_count: int
    model_calls: int
    prompt_tokens: int = 0
    completion_tokens: int = 0


IndexingProgressResponse = Annotated[
    Union[PdfParsingProgressResponse, AutoQAGenerationProgressResponse],
    Field(discriminator="schema_version"),
]


class IndexingJobResponse(PublicSchema):
    job_id: UUID
    kb_id: UUID
    document_id: UUID
    document_version_id: UUID
    indexed_document_version_id: UUID
    index_revision_id: UUID
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    phase: str
    progress: IndexingProgressResponse | None
    attempt: int
    build_status: Literal["queued", "processing", "ready", "failed"]
    serving_status: Literal["candidate", "serving", "retired"]
    claimed_at: datetime | None
    heartbeat_at: datetime | None
    next_attempt_at: datetime | None
    error: IndexingErrorResponse | None
    can_retry: bool
    created_at: datetime
    updated_at: datetime


class IndexingJobPage(PublicSchema):
    items: tuple[IndexingJobResponse, ...]
    next_cursor: OpaqueCursor | None = None
