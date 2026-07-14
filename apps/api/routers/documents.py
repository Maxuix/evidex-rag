"""Read and soft-delete document HTTP transport."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request

from apps.api.errors import ApiProblem
from apps.api.idempotency import RequiredIdempotencyKey
from apps.api.openapi import problem_responses
from apps.api.pagination import decode_cursor, encode_cursor
from apps.api.security import get_auth_context
from rag_kb.auth import AuthContext
from rag_kb.services import Document, DocumentMutationResult
from rag_kb.schemas import (
    CursorPayload,
    DocumentDeleteResponse,
    DocumentPage,
    DocumentResponse,
    DocumentVersionResponse,
    ErrorCode,
)


router = APIRouter(tags=["documents"])
DocumentSort = Literal["created_at", "-created_at", "display_name", "-display_name"]


@router.get(
    "/knowledge-bases/{kb_id}/documents",
    response_model=DocumentPage,
    responses=problem_responses(400, 404, 422),
)
async def list_documents(
    request: Request,
    kb_id: UUID,
    context: Annotated[AuthContext, Depends(get_auth_context)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[str | None, Query(min_length=1, max_length=2048)] = None,
    sort: DocumentSort = "created_at",
) -> DocumentPage:
    after = _after(cursor, sort)
    page = await request.app.state.dependencies.document_service.list(
        context, kb_id=kb_id, limit=limit, sort=sort, after=after
    )
    next_cursor = (
        encode_cursor(CursorPayload(sort=sort, values=page.next_values))
        if page.next_values is not None
        else None
    )
    return DocumentPage(
        items=tuple(_response(item) for item in page.items),
        next_cursor=next_cursor,
    )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    responses=problem_responses(404, 422),
)
async def get_document(
    request: Request,
    document_id: UUID,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> DocumentResponse:
    return _response(
        await request.app.state.dependencies.document_service.get(context, document_id)
    )


@router.delete(
    "/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    responses=problem_responses(404, 409, 422),
)
async def delete_document(
    request: Request,
    document_id: UUID,
    idempotency_key: RequiredIdempotencyKey,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> DocumentDeleteResponse:
    result = await request.app.state.dependencies.document_service.delete(
        context, idempotency_key, document_id
    )
    return _delete_response(result)


def _after(cursor: str | None, sort: str) -> tuple[str, ...] | None:
    if cursor is None:
        return None
    decoded = decode_cursor(cursor)
    if decoded.sort != sort:
        _invalid_cursor("The pagination cursor does not match the requested sort.")
    try:
        if len(decoded.values) != 2:
            raise ValueError
        if sort.removeprefix("-") == "created_at":
            datetime.fromisoformat(decoded.values[0])
        UUID(decoded.values[1])
    except ValueError:
        _invalid_cursor("The pagination cursor has an invalid position.")
    return decoded.values


def _invalid_cursor(detail: str) -> None:
    raise ApiProblem(
        code=ErrorCode.INVALID_CURSOR,
        status=400,
        title="Invalid cursor",
        detail=detail,
    )


def _response(value: Document) -> DocumentResponse:
    version = value.current_version
    return DocumentResponse(
        id=value.id,
        kb_id=value.kb_id,
        display_name=value.display_name,
        current_version=(
            DocumentVersionResponse(
                id=version.id,
                version_number=version.version_number,
                source_status=version.source_status,
                checksum_sha256=version.checksum_sha256,
                original_filename=version.original_filename,
                media_type=version.media_type,
                size_bytes=version.size_bytes,
                created_at=version.created_at,
            )
            if version is not None
            else None
        ),
        deleted_at=value.deleted_at,
        created_at=value.created_at,
        updated_at=value.updated_at,
    )


def _delete_response(value: DocumentMutationResult) -> DocumentDeleteResponse:
    return DocumentDeleteResponse(
        document=_response(value.document),
        source_change_id=value.source_change_id,
        source_change_seq=value.source_change_seq,
        index_revision_id=value.index_revision_id,
    )
