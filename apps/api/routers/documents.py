"""Bounded upload, read, version, and soft-delete document transport."""

from __future__ import annotations

from datetime import datetime
import tempfile
from typing import Annotated, Literal
import unicodedata
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, Request

from apps.api.errors import ApiProblem
from apps.api.idempotency import RequiredIdempotencyKey
from apps.api.openapi import problem_responses
from apps.api.pagination import decode_cursor, encode_cursor
from apps.api.security import get_auth_context
from apps.api.upload_metadata import resolve_upload_metadata
from rag_kb.auth import AuthContext
from rag_kb.services import Document, DocumentMutationResult, FileAdmissionError
from rag_kb.schemas import (
    CursorPayload,
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentIndexSummaryResponse,
    DocumentPage,
    DocumentResponse,
    DocumentUploadResponse,
    DocumentVersionResponse,
    ErrorCode,
)


router = APIRouter(tags=["documents"])
DocumentSort = Literal["created_at", "-created_at", "display_name", "-display_name"]
_BINARY_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "text/plain": {"schema": {"type": "string", "format": "binary"}},
            "text/markdown": {"schema": {"type": "string", "format": "binary"}},
            "application/pdf": {"schema": {"type": "string", "format": "binary"}},
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": {
                "schema": {"type": "string", "format": "binary"}
            },
        },
    }
}


@router.post(
    "/knowledge-bases/{kb_id}/documents",
    response_model=DocumentUploadResponse,
    status_code=202,
    responses=problem_responses(404, 409, 413, 415, 422),
    openapi_extra=_BINARY_BODY,
)
async def upload_document(
    request: Request,
    kb_id: UUID,
    idempotency_key: RequiredIdempotencyKey,
    context: Annotated[AuthContext, Depends(get_auth_context)],
    encoded_metadata: Annotated[
        str | None,
        Header(alias="X-Document-Metadata", min_length=1, max_length=4096),
    ] = None,
    original_filename: Annotated[
        str | None,
        Header(alias="X-Document-Filename", min_length=1, max_length=255),
    ] = None,
    display_name: Annotated[
        str | None,
        Header(alias="X-Document-Display-Name", min_length=1, max_length=255),
    ] = None,
) -> DocumentUploadResponse:
    metadata = resolve_upload_metadata(
        encoded_metadata=encoded_metadata,
        legacy_filename=original_filename,
        legacy_display_name=display_name,
    )
    return await _accept_upload(
        request,
        context=context,
        idempotency_key=idempotency_key,
        kb_id=kb_id,
        document_id=None,
        original_filename=metadata.original_filename,
        display_name=_clean_display_name(
            metadata.display_name or metadata.original_filename
        ),
    )


@router.post(
    "/documents/{document_id}/versions",
    response_model=DocumentUploadResponse,
    status_code=202,
    responses=problem_responses(404, 409, 413, 415, 422),
    openapi_extra=_BINARY_BODY,
)
async def upload_document_version(
    request: Request,
    document_id: UUID,
    idempotency_key: RequiredIdempotencyKey,
    context: Annotated[AuthContext, Depends(get_auth_context)],
    encoded_metadata: Annotated[
        str | None,
        Header(alias="X-Document-Metadata", min_length=1, max_length=4096),
    ] = None,
    original_filename: Annotated[
        str | None,
        Header(alias="X-Document-Filename", min_length=1, max_length=255),
    ] = None,
    display_name: Annotated[
        str | None,
        Header(alias="X-Document-Display-Name", min_length=1, max_length=255),
    ] = None,
) -> DocumentUploadResponse:
    metadata = resolve_upload_metadata(
        encoded_metadata=encoded_metadata,
        legacy_filename=original_filename,
        legacy_display_name=display_name,
    )
    document = await request.app.state.dependencies.document_service.get(
        context, document_id
    )
    return await _accept_upload(
        request,
        context=context,
        idempotency_key=idempotency_key,
        kb_id=document.kb_id,
        document_id=document_id,
        original_filename=metadata.original_filename,
        display_name=_clean_display_name(
            metadata.display_name or document.display_name
        ),
    )


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
    response_model=DocumentDetailResponse,
    responses=problem_responses(404, 422),
)
async def get_document(
    request: Request,
    document_id: UUID,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> DocumentDetailResponse:
    detail = await request.app.state.dependencies.document_service.get_detail(
        context, document_id
    )
    document = _response(detail.document)
    summary = detail.index
    return DocumentDetailResponse(
        **document.model_dump(),
        index=(
            DocumentIndexSummaryResponse(
                indexed_document_version_id=summary.indexed_document_version_id,
                index_revision_id=summary.index_revision_id,
                build_status=summary.build_status,
                serving_status=summary.serving_status,
                unit_count=summary.unit_count,
                asset_count=summary.asset_count,
                representation_count=summary.representation_count,
                composite_chunk_count=summary.composite_chunk_count,
                visual_unit_count=summary.visual_unit_count,
                relation_count=summary.relation_count,
                text_representation_count=summary.text_representation_count,
                native_image_representation_count=(
                    summary.native_image_representation_count
                ),
                table_representation_count=summary.table_representation_count,
            )
            if summary is not None
            else None
        ),
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


async def _accept_upload(
    request: Request,
    *,
    context: AuthContext,
    idempotency_key: UUID,
    kb_id: UUID,
    document_id: UUID | None,
    original_filename: str,
    display_name: str,
) -> DocumentUploadResponse:
    dependencies = request.app.state.dependencies
    maximum = dependencies.file_admission_service.limits.max_bytes
    source = tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b")
    size = 0
    try:
        async for block in request.stream():
            size += len(block)
            if size > maximum:
                raise FileAdmissionError(
                    ErrorCode.FILE_TOO_LARGE, limit=maximum, observed=size
                )
            source.write(block)
        source.seek(0)
        admitted = dependencies.file_admission_service.validate(
            source,
            original_filename=original_filename,
            media_type=request.headers.get("content-type", ""),
        )
        result = await dependencies.source_file_service.store_and_activate(
            context,
            idempotency_key,
            kb_id=kb_id,
            document_id=document_id,
            display_name=display_name,
            original_filename=admitted.original_filename,
            media_type=admitted.media_type,
            source=source,
        )
        return _upload_response(result)
    finally:
        source.close()


def _clean_display_name(value: str) -> str:
    cleaned = unicodedata.normalize("NFC", value.strip())
    if (
        not cleaned
        or len(cleaned) > 255
        or any(
            unicodedata.category(character) in {"Cc", "Cs"}
            for character in cleaned
        )
    ):
        raise ApiProblem(
            code=ErrorCode.REQUEST_VALIDATION_FAILED,
            status=422,
            title="Request validation failed",
            detail="The document display name is invalid.",
        )
    return cleaned


def _upload_response(value: DocumentMutationResult) -> DocumentUploadResponse:
    if None in {
        value.document_version_id,
        value.source_change_id,
        value.source_change_seq,
        value.indexed_document_version_id,
        value.index_revision_id,
        value.job_id,
    } or value.job_status != "queued":
        raise RuntimeError("activated upload result is incomplete")
    return DocumentUploadResponse(
        document=_response(value.document),
        document_version_id=value.document_version_id,
        source_change_id=value.source_change_id,
        source_change_seq=value.source_change_seq,
        indexed_document_version_id=value.indexed_document_version_id,
        index_revision_id=value.index_revision_id,
        job_id=value.job_id,
        job_status="queued",
    )
