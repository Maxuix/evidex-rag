"""Knowledge-base HTTP transport."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Query, Request, status

from apps.api.errors import ApiProblem
from apps.api.idempotency import RequiredIdempotencyKey
from apps.api.pagination import (
    API_CURSOR_MAX_LENGTH,
    API_PAGINATION_DEFAULT_LIMIT,
    API_PAGINATION_MAX_LIMIT,
    decode_cursor,
    encode_cursor,
)
from rag_kb.domain import KnowledgeBase
from rag_kb.schemas import (
    CursorPayload,
    ErrorCode,
    KnowledgeBaseAutoQAResponse,
    KnowledgeBaseCreate,
    KnowledgeBaseDeleteResponse,
    KnowledgeBaseChunkingResponse,
    KnowledgeBasePage,
    KnowledgeBaseParsingResponse,
    KnowledgeBaseResponse,
    KnowledgeBaseEmbeddingResponse,
    KnowledgeBaseUpdate,
    RetrievalDefaults,
)


router = APIRouter(tags=["knowledge-bases"])
KnowledgeBaseSort = Literal["created_at", "-created_at", "name", "-name"]


@router.post(
    "/knowledge-bases",
    response_model=KnowledgeBaseResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_knowledge_base(
    request: Request,
    payload: KnowledgeBaseCreate,
    idempotency_key: RequiredIdempotencyKey,
) -> KnowledgeBaseResponse:
    created = await request.app.state.dependencies.knowledge_base_service.create(
        idempotency_key,
        name=payload.name,
        parsing_preset=payload.parsing.preset,
        chunking_preset=payload.chunking.preset,
        retrieval_defaults=payload.retrieval_defaults.model_dump(mode="json"),
        embedding_selection=(
            payload.embedding.model_dump(mode="json")
            if payload.embedding is not None
            else None
        ),
        auto_qa=payload.auto_qa.model_dump(mode="json"),
    )
    return _response(created)


@router.get(
    "/knowledge-bases",
    response_model=KnowledgeBasePage,
)
async def list_knowledge_bases(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=API_PAGINATION_MAX_LIMIT)] = API_PAGINATION_DEFAULT_LIMIT,
    cursor: Annotated[str | None, Query(min_length=1, max_length=API_CURSOR_MAX_LENGTH)] = None,
    sort: KnowledgeBaseSort = "created_at",
) -> KnowledgeBasePage:
    after = _after(cursor, sort)
    page = await request.app.state.dependencies.knowledge_base_service.list(
        limit=limit, sort=sort, after=after
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
)
async def get_knowledge_base(
    request: Request,
    kb_id: UUID,
) -> KnowledgeBaseResponse:
    return _response(
        await request.app.state.dependencies.knowledge_base_service.get(kb_id)
    )


@router.patch(
    "/knowledge-bases/{kb_id}",
    response_model=KnowledgeBaseResponse,
)
async def update_knowledge_base(
    request: Request,
    kb_id: UUID,
    payload: KnowledgeBaseUpdate,
    idempotency_key: RequiredIdempotencyKey,
) -> KnowledgeBaseResponse:
    updated = await request.app.state.dependencies.knowledge_base_service.update(
        idempotency_key,
        kb_id,
        name=payload.name,
        retrieval_defaults=(
            payload.retrieval_defaults.model_dump(mode="json")
            if payload.retrieval_defaults is not None
            else None
        ),
    )
    return _response(updated)


@router.delete(
    "/knowledge-bases/{kb_id}",
    response_model=KnowledgeBaseDeleteResponse,
)
async def delete_knowledge_base(
    request: Request,
    kb_id: UUID,
    idempotency_key: RequiredIdempotencyKey,
) -> KnowledgeBaseDeleteResponse:
    deleted = await request.app.state.dependencies.knowledge_base_service.delete(
        idempotency_key,
        kb_id,
    )
    assert deleted.deleted_at is not None
    return KnowledgeBaseDeleteResponse(
        id=deleted.id,
        name=deleted.name,
        deleted_at=deleted.deleted_at,
    )


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
    from rag_kb.document_processing.profiles import (
        public_descriptor,
        public_parsing_descriptor,
    )

    assert value.embedding is not None
    return KnowledgeBaseResponse(
        id=value.id,
        name=value.name,
        source_change_seq=value.source_change_seq,
        active_index_revision_id=value.active_index_revision_id,
        embedding_space_id=value.embedding_space_id,
        embedding=KnowledgeBaseEmbeddingResponse.model_validate(
            {
                "strategy": value.embedding.strategy,
                "text": {
                    "embedding_space_id": value.embedding.text.embedding_space_id,
                    "profile_revision_id": value.embedding.text.profile_revision_id,
                    "dimension": value.embedding.text.dimension,
                },
                "cross_modal": (
                    {
                        "embedding_space_id": (
                            value.embedding.cross_modal.embedding_space_id
                        ),
                        "profile_revision_id": (
                            value.embedding.cross_modal.profile_revision_id
                        ),
                        "dimension": value.embedding.cross_modal.dimension,
                    }
                    if value.embedding.cross_modal is not None
                    else None
                ),
            }
        ),
        parsing=KnowledgeBaseParsingResponse.model_validate(
            public_parsing_descriptor(value.parser_config)
        ),
        chunking=KnowledgeBaseChunkingResponse.model_validate(
            public_descriptor(value.chunking_config)
        ),
        retrieval_defaults=RetrievalDefaults.model_validate(value.retrieval_defaults),
        answer_policy_defaults=dict(value.answer_policy_defaults),
        auto_qa=KnowledgeBaseAutoQAResponse(
            enabled=value.auto_qa.enabled,
            questions_per_chunk=value.auto_qa.questions_per_chunk,
            model_profile_revision_id=value.auto_qa.model_profile_revision_id,
            model_name=value.auto_qa.model_name,
            model_revision=value.auto_qa.model_revision,
        ),
        provisioned_at=value.provisioned_at,
        created_at=value.created_at,
        updated_at=value.updated_at,
    )
