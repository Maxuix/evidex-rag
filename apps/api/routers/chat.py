"""Durable chat session and ChatRun HTTP transport."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, Request, Response, status
from fastapi.sse import EventSourceResponse, ServerSentEvent

from apps.api.errors import ApiProblem
from apps.api.idempotency import RequiredIdempotencyKey
from apps.api.openapi import problem_responses
from apps.api.pagination import decode_cursor, encode_cursor
from apps.api.security import get_auth_context
from rag_kb.auth import AuthContext
from rag_kb.services import (
    ChatMessage,
    ChatRun,
    ChatSession,
    ChatSseSubscription,
)
from rag_kb.schemas import (
    ChatAnswerCompletedEvent,
    ChatCitationResponse,
    ChatMessagePage,
    ChatMessageResponse,
    ChatRunCreate,
    ChatRunErrorResponse,
    ChatRunFailedEvent,
    ChatRunResponse,
    ChatRunQueryContextResponse,
    ChatSessionCreate,
    ChatSessionPage,
    ChatSessionResponse,
    CursorPayload,
    EffectiveAnswerPolicyResponse,
    ErrorCode,
)
from rag_kb.memory import (
    empty_conversation_context,
    hydrate_contextualized_query,
    hydrate_conversation_context,
)


router = APIRouter(prefix="/chat", tags=["chat"])
SessionSort = Literal["created_at", "-created_at", "updated_at", "-updated_at"]
MessageSort = Literal["created_at", "-created_at"]


@router.post(
    "/sessions",
    response_model=ChatSessionResponse,
    status_code=status.HTTP_201_CREATED,
    responses=problem_responses(404, 422),
)
async def create_chat_session(
    request: Request,
    response: Response,
    payload: ChatSessionCreate,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> ChatSessionResponse:
    value = await request.app.state.dependencies.chat_service.create_session(
        context,
        kb_id=payload.knowledge_base_id,
        title=payload.title,
    )
    response.headers["Location"] = f"/api/v1/chat/sessions/{value.id}/messages"
    return _session_response(value)


@router.get(
    "/sessions",
    response_model=ChatSessionPage,
    responses=problem_responses(400, 422),
)
async def list_chat_sessions(
    request: Request,
    context: Annotated[AuthContext, Depends(get_auth_context)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[str | None, Query(min_length=1, max_length=2048)] = None,
    sort: SessionSort = "-updated_at",
) -> ChatSessionPage:
    after = _after(cursor, sort)
    page = await request.app.state.dependencies.chat_service.list_sessions(
        context, limit=limit, sort=sort, after=after
    )
    return ChatSessionPage(
        items=tuple(_session_response(item) for item in page.items),
        next_cursor=_next_cursor(sort, page.next_values),
    )


@router.get(
    "/sessions/{session_id}/messages",
    response_model=ChatMessagePage,
    responses=problem_responses(400, 404, 422),
)
async def list_chat_messages(
    request: Request,
    session_id: UUID,
    context: Annotated[AuthContext, Depends(get_auth_context)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[str | None, Query(min_length=1, max_length=2048)] = None,
    sort: MessageSort = "created_at",
) -> ChatMessagePage:
    after = _after(cursor, sort)
    page = await request.app.state.dependencies.chat_service.list_messages(
        context,
        session_id,
        limit=limit,
        sort=sort,
        after=after,
    )
    return ChatMessagePage(
        items=tuple(_message_response(item) for item in page.items),
        next_cursor=_next_cursor(sort, page.next_values),
    )


@router.post(
    "/runs",
    response_model=ChatRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=problem_responses(404, 409, 422),
)
async def create_chat_run(
    request: Request,
    response: Response,
    payload: ChatRunCreate,
    idempotency_key: RequiredIdempotencyKey,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> ChatRunResponse:
    value = await request.app.state.dependencies.chat_service.create_run(
        context,
        idempotency_key,
        session_id=payload.session_id,
        kb_id=payload.knowledge_base_id,
        message=payload.message,
        answer_style=payload.answer_policy.answer_style,
        insufficiency_policy=payload.answer_policy.insufficiency_policy,
        retrieval_mode=payload.retrieval.mode,
        top_k=payload.retrieval.top_k,
        rerank=payload.retrieval.rerank,
    )
    response.headers["Location"] = _status_url(value.id)
    return _run_response(value)


@router.get(
    "/runs/{run_id}",
    response_model=ChatRunResponse,
    responses=problem_responses(404, 422),
)
async def get_chat_run(
    request: Request,
    run_id: UUID,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> ChatRunResponse:
    return _run_response(
        await request.app.state.dependencies.chat_service.get_run(context, run_id)
    )


async def _prepare_chat_sse_subscription(
    request: Request,
    run_id: UUID,
    context: Annotated[AuthContext, Depends(get_auth_context)],
    last_event_id: Annotated[
        str | None, Header(alias="Last-Event-ID", max_length=2048)
    ] = None,
) -> AsyncIterator[ChatSseSubscription]:
    if last_event_id is not None:
        raise ApiProblem(
            code=ErrorCode.REQUEST_VALIDATION_FAILED,
            status=400,
            title="Event replay is not supported",
            detail="Last-Event-ID is not accepted by the P1A terminal stream.",
        )
    chat = request.app.state.dependencies.chat_service
    value = await chat.get_run(context, run_id)
    limiter = request.app.state.dependencies.chat_sse_connection_limiter
    acquired = await limiter.acquire(context.principal_id, run_id)
    if not acquired:
        raise ApiProblem(
            code=ErrorCode.CHAT_SSE_CONNECTION_LIMIT_EXCEEDED,
            status=429,
            title="Chat event connection limit exceeded",
            detail="Use the authoritative ChatRun status URL and retry later.",
            retryable=True,
        )
    try:
        yield ChatSseSubscription(context=context, run=value)
    finally:
        await limiter.release(context.principal_id, run_id)


@router.get(
    "/runs/{run_id}/events",
    response_class=EventSourceResponse,
    responses=problem_responses(400, 404, 429, 422),
)
async def stream_chat_run_events(
    request: Request,
    subscription: Annotated[
        ChatSseSubscription, Depends(_prepare_chat_sse_subscription)
    ],
) -> AsyncIterator[ServerSentEvent]:
    watcher = request.app.state.dependencies.chat_terminal_watcher
    async for value in watcher.watch(
        subscription.context,
        subscription.run.id,
        initial=subscription.run,
        disconnected=request.is_disconnected,
    ):
        if value.status == "completed":
            if not value.assistant_content:
                raise RuntimeError("completed ChatRun is missing its answer")
            yield ServerSentEvent(
                event="answer.completed",
                data=ChatAnswerCompletedEvent(
                    run_id=value.id,
                    message_id=value.assistant_message_id,
                    answer=value.assistant_content,
                    citations=_citation_responses(value),
                    effective_answer_policy=_policy_response(value),
                    status_url=_status_url(value.id),
                ),
            )
        else:
            yield ServerSentEvent(
                event="run.failed",
                data=ChatRunFailedEvent(
                    run_id=value.id,
                    status=value.status,
                    error=_terminal_error(value),
                    effective_answer_policy=_policy_response(value),
                    status_url=_status_url(value.id),
                ),
            )


def _session_response(value: ChatSession) -> ChatSessionResponse:
    return ChatSessionResponse(
        id=value.id,
        knowledge_base_id=value.kb_id,
        title=value.title,
        created_at=value.created_at,
        updated_at=value.updated_at,
    )


def _message_response(value: ChatMessage) -> ChatMessageResponse:
    return ChatMessageResponse(
        id=value.id,
        session_id=value.session_id,
        run_id=value.chat_run_id,
        role=value.role,
        assistant_status=value.assistant_status,
        content=value.content,
        created_at=value.created_at,
    )


def _run_response(value: ChatRun) -> ChatRunResponse:
    return ChatRunResponse(
        run_id=value.id,
        knowledge_base_id=value.kb_id,
        session_id=value.session_id,
        user_message_id=value.user_message_id,
        assistant_message_id=value.assistant_message_id,
        index_revision_id=value.index_revision_id,
        status=value.status,
        assistant_status=value.assistant_status,
        answer=value.assistant_content or None,
        citations=_citation_responses(value),
        status_url=_status_url(value.id),
        events_url=f"{_status_url(value.id)}/events",
        effective_answer_policy=_policy_response(value),
        retrieval=dict(value.retrieval_strategy),
        query_context=_query_context_response(value),
        attempt=value.attempt,
        error=_run_error(value),
        usage=dict(value.usage) if value.usage is not None else None,
        timing=dict(value.timing) if value.timing is not None else None,
        created_at=value.created_at,
        updated_at=value.updated_at,
        completed_at=value.completed_at,
    )


def _query_context_response(value: ChatRun) -> ChatRunQueryContextResponse:
    try:
        snapshot = (
            hydrate_conversation_context(value.conversation_context)
            if value.conversation_context is not None
            else empty_conversation_context()
        )
        artifact = (
            hydrate_contextualized_query(value.contextualized_query)
            if value.contextualized_query is not None
            else None
        )
    except (TypeError, ValueError) as error:
        raise ApiProblem(
            code=ErrorCode.CHAT_CONTEXT_INVALID,
            status=500,
            title="Chat context invalid",
            detail="The persisted ChatRun context is invalid.",
        ) from error
    return ChatRunQueryContextResponse(
        strategy=snapshot.strategy,
        status=(
            artifact.status.value
            if artifact is not None
            else ("original" if not snapshot.turns else "pending")
        ),
        history_turn_count=len(snapshot.turns),
        history_token_count=snapshot.token_count,
        history_truncated=snapshot.truncated,
        standalone_query=(
            artifact.standalone_query if artifact is not None else None
        ),
    )


def _citation_responses(value: ChatRun) -> tuple[ChatCitationResponse, ...]:
    return tuple(
        ChatCitationResponse(
            ordinal=item.ordinal,
            index_chunk_id=item.index_chunk_id,
            document_id=item.document_id,
            document_version_id=item.document_version_id,
            quoted_text=item.quoted_text,
            source_location=dict(item.source_location),
            score=item.score,
        )
        for item in value.citations
    )


def _policy_response(value: ChatRun) -> EffectiveAnswerPolicyResponse:
    return EffectiveAnswerPolicyResponse.model_validate(value.effective_policy)


def _run_error(value: ChatRun) -> ChatRunErrorResponse | None:
    if value.error_code is None:
        return None
    return ChatRunErrorResponse(
        code=value.error_code,
        detail=dict(value.error_detail or {}),
        retryable=bool(value.error_retryable),
    )


def _cancelled_error() -> ChatRunErrorResponse:
    return ChatRunErrorResponse(
        code=ErrorCode.CHAT_RUN_CANCELLED.value,
        detail={},
        retryable=False,
    )


def _terminal_error(value: ChatRun) -> ChatRunErrorResponse:
    committed = _run_error(value)
    if committed is not None:
        return committed
    if value.status == "cancelled":
        return _cancelled_error()
    return ChatRunErrorResponse(
        code=ErrorCode.INTERNAL_SERVER_ERROR.value,
        detail={},
        retryable=False,
    )


def _status_url(run_id: UUID) -> str:
    return f"/api/v1/chat/runs/{run_id}"


def _next_cursor(sort: str, values: tuple[str, ...] | None) -> str | None:
    if values is None:
        return None
    return encode_cursor(CursorPayload(sort=sort, values=values))


def _after(cursor: str | None, sort: str) -> tuple[str, ...] | None:
    if cursor is None:
        return None
    decoded = decode_cursor(cursor)
    if decoded.sort != sort:
        _invalid_cursor("The pagination cursor does not match the requested sort.")
    try:
        if len(decoded.values) != 2:
            raise ValueError
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
