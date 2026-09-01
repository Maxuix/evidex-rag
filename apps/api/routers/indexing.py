"""Queryable indexing state and explicit bounded retry."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Query, Request, Response

from apps.api.errors import ApiProblem
from apps.api.idempotency import RequiredIdempotencyKey
from apps.api.pagination import (
    API_CURSOR_MAX_LENGTH,
    API_PAGINATION_MAX_LIMIT,
    decode_cursor,
    encode_cursor,
)
from rag_kb.schemas import (
    CursorPayload,
    ErrorCode,
    IndexingErrorResponse,
    IndexingJobPage,
    IndexingJobResponse,
)


router = APIRouter(tags=["indexing"])


@router.get(
    "/knowledge-bases/{kb_id}/indexing-jobs",
    response_model=IndexingJobPage,
)
async def list_indexing_jobs(
    request: Request,
    kb_id: UUID,
    limit: Annotated[int, Query(ge=1, le=API_PAGINATION_MAX_LIMIT)] = API_PAGINATION_MAX_LIMIT,
    cursor: Annotated[str | None, Query(min_length=1, max_length=API_CURSOR_MAX_LENGTH)] = None,
) -> IndexingJobPage:
    page = await request.app.state.dependencies.indexing_job_service.list(
        kb_id=kb_id,
        limit=limit,
        after=_after(cursor),
    )
    return IndexingJobPage(
        items=tuple(_response(item) for item in page.items),
        next_cursor=(
            encode_cursor(
                CursorPayload(sort="-created_at", values=page.next_values)
            )
            if page.next_values is not None
            else None
        ),
    )


@router.get(
    "/indexing-jobs/{job_id}",
    response_model=IndexingJobResponse,
)
async def get_indexing_job(
    request: Request,
    job_id: UUID,
) -> IndexingJobResponse:
    value = await request.app.state.dependencies.indexing_job_service.get(
        job_id
    )
    return _response(value)


@router.post(
    "/indexing-jobs/{job_id}/retry",
    response_model=IndexingJobResponse,
    status_code=202,
)
async def retry_indexing_job(
    request: Request,
    response: Response,
    job_id: UUID,
    idempotency_key: RequiredIdempotencyKey,
) -> IndexingJobResponse:
    value = await request.app.state.dependencies.indexing_job_service.retry(
        idempotency_key,
        job_id,
    )
    response.headers["Location"] = f"/api/v1/indexing-jobs/{job_id}"
    return _response(value)


def _response(value: Any) -> IndexingJobResponse:
    error = None
    if value.error_code is not None:
        error = IndexingErrorResponse(
            code=value.error_code,
            detail=dict(value.error_detail or {}),
        )
    return IndexingJobResponse(
        job_id=value.job_id,
        kb_id=value.kb_id,
        document_id=value.document_id,
        document_version_id=value.document_version_id,
        indexed_document_version_id=value.indexed_document_version_id,
        index_revision_id=value.index_revision_id,
        status=value.job_status,
        phase=value.phase,
        progress=value.progress or None,
        attempt=value.attempt,
        build_status=value.build_status,
        serving_status=value.serving_status,
        claimed_at=value.claimed_at,
        heartbeat_at=value.heartbeat_at,
        next_attempt_at=value.next_attempt_at,
        error=error,
        can_retry=value.can_retry,
        created_at=value.created_at,
        updated_at=value.updated_at,
    )


def _after(cursor: str | None) -> tuple[str, ...] | None:
    if cursor is None:
        return None
    decoded = decode_cursor(cursor)
    if decoded.sort != "-created_at" or len(decoded.values) != 2:
        _invalid_cursor("The pagination cursor does not match indexing-job ordering.")
    try:
        datetime.fromisoformat(decoded.values[0])
        UUID(decoded.values[1])
    except ValueError:
        _invalid_cursor("The indexing-job cursor has an invalid position.")
    return decoded.values


def _invalid_cursor(detail: str) -> None:
    raise ApiProblem(
        code=ErrorCode.INVALID_CURSOR,
        status=400,
        title="Invalid cursor",
        detail=detail,
    )
