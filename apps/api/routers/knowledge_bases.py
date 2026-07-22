"""Knowledge-base HTTP transport."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, status

from apps.api.errors import ApiProblem
from apps.api.idempotency import RequiredIdempotencyKey
from apps.api.openapi import problem_responses
from apps.api.pagination import decode_cursor, encode_cursor
from apps.api.security import get_auth_context
from rag_kb.auth import AuthContext
from rag_kb.services import KnowledgeBase
from rag_kb.schemas import (
    CursorPayload,
    ErrorCode,
    KnowledgeBaseCreate,
    KnowledgeBaseChunkingResponse,
    KnowledgeBaseAnswerPolicyDefaults,
    KnowledgeBasePage,
    KnowledgeBaseParsingResponse,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
    RetrievalDefaults,
)


router = APIRouter(tags=["knowledge-bases"])
KnowledgeBaseSort = Literal["created_at", "-created_at", "name", "-name"]


@router.post(
    "/knowledge-bases",
    response_model=KnowledgeBaseResponse,
    status_code=status.HTTP_201_CREATED,
    responses=problem_responses(409, 422),
)
async def create_knowledge_base(
    request: Request,
    payload: KnowledgeBaseCreate,
    idempotency_key: RequiredIdempotencyKey,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> KnowledgeBaseResponse:
    created = await request.app.state.dependencies.knowledge_base_service.create(
        context,
        idempotency_key,
        name=payload.name,
        parsing_preset=payload.parsing.preset,
        chunking_preset=payload.chunking.preset,
        retrieval_defaults=payload.retrieval_defaults.model_dump(mode="json"),
        answer_policy_defaults=payload.answer_policy_defaults.model_dump(mode="json"),
    )
    return _response(created)


@router.get(
    "/knowledge-bases",
    response_model=KnowledgeBasePage,
    responses=problem_responses(400, 422),
)
async def list_knowledge_bases(
    request: Request,
    context: Annotated[AuthContext, Depends(get_auth_context)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[str | None, Query(min_length=1, max_length=2048)] = None,
    sort: KnowledgeBaseSort = "created_at",
) -> KnowledgeBasePage:
    after = _after(cursor, sort)
    page = await request.app.state.dependencies.knowledge_base_service.list(
        context, limit=limit, sort=sort, after=after
    )
    next_cursor = (
        encode_cursor(CursorPayload(sort=sort, values=page.next_values))
        if page.next_values is not None
        else None
    )
    return KnowledgeBasePage(
        items=tuple(_response(item) for item in page.items),
        next_cursor=next_cursor,
    )


@router.get(
    "/knowledge-bases/{kb_id}",
    response_model=KnowledgeBaseResponse,
    responses=problem_responses(404, 422),
)
async def get_knowledge_base(
    request: Request,
    kb_id: UUID,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> KnowledgeBaseResponse:
    return _response(
        await request.app.state.dependencies.knowledge_base_service.get(context, kb_id)
    )


@router.patch(
    "/knowledge-bases/{kb_id}",
    response_model=KnowledgeBaseResponse,
    responses=problem_responses(404, 409, 422),
)
async def update_knowledge_base(
    request: Request,
    kb_id: UUID,
    payload: KnowledgeBaseUpdate,
    idempotency_key: RequiredIdempotencyKey,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> KnowledgeBaseResponse:
    updated = await request.app.state.dependencies.knowledge_base_service.update(
        context,
        idempotency_key,
        kb_id,
        name=payload.name,
        retrieval_defaults=(
            payload.retrieval_defaults.model_dump(mode="json")
            if payload.retrieval_defaults is not None
            else None
        ),
        answer_policy_defaults=(
            payload.answer_policy_defaults.model_dump(mode="json")
            if payload.answer_policy_defaults is not None
            else None
        ),
    )
    return _response(updated)


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


def _response(value: KnowledgeBase) -> KnowledgeBaseResponse:
    from rag_kb.document_processing import (
        UNSTRUCTURED_PARSER_CONFIG,
        public_descriptor,
        public_parsing_descriptor,
    )

    parser_config = value.parser_config or UNSTRUCTURED_PARSER_CONFIG

    return KnowledgeBaseResponse(
        id=value.id,
        name=value.name,
        source_change_seq=value.source_change_seq,
        active_index_revision_id=value.active_index_revision_id,
        embedding_space_id=value.embedding_space_id,
        parsing=KnowledgeBaseParsingResponse.model_validate(
            public_parsing_descriptor(parser_config)
        ),
        chunking=KnowledgeBaseChunkingResponse.model_validate(
            public_descriptor(value.chunking_config)
        ),
        retrieval_defaults=RetrievalDefaults.model_validate(value.retrieval_defaults),
        answer_policy_defaults=KnowledgeBaseAnswerPolicyDefaults.model_validate(
            value.answer_policy_defaults
        ),
        provisioned_at=value.provisioned_at,
        created_at=value.created_at,
        updated_at=value.updated_at,
    )
