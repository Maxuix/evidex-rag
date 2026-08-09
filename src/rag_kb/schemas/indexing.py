"""Public indexing status and explicit retry schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from rag_kb.domain import ErrorCode
from rag_kb.schemas.common import PublicSchema
from rag_kb.schemas.common import OpaqueCursor


class IndexingErrorResponse(PublicSchema):
    code: ErrorCode
    detail: dict[str, Any]


class IndexingJobResponse(PublicSchema):
    job_id: UUID
    kb_id: UUID
    document_id: UUID
    document_version_id: UUID
    indexed_document_version_id: UUID
    index_revision_id: UUID
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    phase: str
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
