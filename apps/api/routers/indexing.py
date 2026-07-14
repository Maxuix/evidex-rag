"""Queryable indexing state and explicit bounded retry."""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response

from apps.api.idempotency import RequiredIdempotencyKey
from apps.api.openapi import problem_responses
from apps.api.security import get_auth_context
from rag_kb.auth import AuthContext
from rag_kb.schemas import IndexingErrorResponse, IndexingJobResponse


router = APIRouter(prefix="/indexing-jobs", tags=["indexing"])


@router.get(
    "/{job_id}",
    response_model=IndexingJobResponse,
    responses=problem_responses(404, 422),
)
async def get_indexing_job(
    request: Request,
    job_id: UUID,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> IndexingJobResponse:
    value = await request.app.state.dependencies.indexing_job_service.get(
        context, job_id
    )
    return _response(value)


@router.post(
    "/{job_id}/retry",
    response_model=IndexingJobResponse,
    status_code=202,
    responses=problem_responses(404, 409, 422),
)
async def retry_indexing_job(
    request: Request,
    response: Response,
    job_id: UUID,
    idempotency_key: RequiredIdempotencyKey,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> IndexingJobResponse:
    value = await request.app.state.dependencies.indexing_job_service.retry(
        context,
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
