"""SQLAlchemy persistence for sessions, messages, and ChatRuns."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Select, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_kb.db.models import (
    AssistantMessageStatus,
    ChatMessage as ChatMessageRow,
    ChatMessageRole,
    ChatRun as ChatRunRow,
    ChatRunStatus,
    ChatSession as ChatSessionRow,
)
from rag_kb.domain import (
    ChatMessage,
    ChatRun,
    ChatSession,
    IdempotencyScope,
    Page,
)


class SqlAlchemyChatRepository:
    def __init__(
        self,
        session: AsyncSession,
        workspace_id: UUID,
        ensure_active: Callable[[], None],
    ) -> None:
        self._session = session
        self._workspace_id = workspace_id
        self._ensure_active = ensure_active

    async def create_session(
        self, *, kb_id: UUID, principal_id: str, title: str | None
    ) -> ChatSession:
        self._ensure_active()
        row = ChatSessionRow(
            workspace_id=self._workspace_id,
            kb_id=kb_id,
            principal_id=principal_id,
            title=title,
        )
        self._session.add(row)
        await self._session.flush()
        return _session(row)

    async def get_session(
        self, session_id: UUID, *, principal_id: str
    ) -> ChatSession | None:
        self._ensure_active()
        row = await self._session.scalar(
            select(ChatSessionRow).where(
                ChatSessionRow.workspace_id == self._workspace_id,
                ChatSessionRow.id == session_id,
                ChatSessionRow.principal_id == principal_id,
            )
        )
        return _session(row) if row is not None else None

    async def list_sessions(
        self,
        *,
        principal_id: str,
        limit: int,
        sort: str,
        after: tuple[str, ...] | None,
    ) -> Page[ChatSession]:
        self._ensure_active()
        descending = sort.startswith("-")
        field = sort.removeprefix("-")
        column = {
            "created_at": ChatSessionRow.created_at,
            "updated_at": ChatSessionRow.updated_at,
        }[field]
        statement = select(ChatSessionRow).where(
            ChatSessionRow.workspace_id == self._workspace_id,
            ChatSessionRow.principal_id == principal_id,
        )
        statement = _with_after(
            statement, column, ChatSessionRow.id, after, descending
        )
        ordering = column.desc() if descending else column.asc()
        id_ordering = (
            ChatSessionRow.id.desc() if descending else ChatSessionRow.id.asc()
        )
        rows = (
            await self._session.scalars(
                statement.order_by(ordering, id_ordering).limit(limit + 1)
            )
        ).all()
        has_more = len(rows) > limit
        rows = rows[:limit]
        items = tuple(_session(row) for row in rows)
        next_values = None
        if has_more and items:
            value = getattr(items[-1], field)
            next_values = (value.isoformat(), str(items[-1].id))
        return Page(items=items, next_values=next_values)

    async def list_messages(
        self,
        *,
        session_id: UUID,
        principal_id: str,
        limit: int,
        sort: str,
        after: tuple[str, ...] | None,
    ) -> Page[ChatMessage] | None:
        self._ensure_active()
        authorized_session = await self._session.scalar(
            select(ChatSessionRow.id).where(
                ChatSessionRow.workspace_id == self._workspace_id,
                ChatSessionRow.id == session_id,
                ChatSessionRow.principal_id == principal_id,
            )
        )
        if authorized_session is None:
            return None
        descending = sort.startswith("-")
        statement = select(ChatMessageRow).where(
            ChatMessageRow.workspace_id == self._workspace_id,
            ChatMessageRow.session_id == session_id,
        )
        statement = _with_after(
            statement,
            ChatMessageRow.created_at,
            ChatMessageRow.id,
            after,
            descending,
        )
        ordering = (
            ChatMessageRow.created_at.desc()
            if descending
            else ChatMessageRow.created_at.asc()
        )
        id_ordering = (
            ChatMessageRow.id.desc() if descending else ChatMessageRow.id.asc()
        )
        rows = (
            await self._session.scalars(
                statement.order_by(ordering, id_ordering).limit(limit + 1)
            )
        ).all()
        has_more = len(rows) > limit
        rows = rows[:limit]
        items = tuple(_message(row) for row in rows)
        next_values = None
        if has_more and items:
            next_values = (items[-1].created_at.isoformat(), str(items[-1].id))
        return Page(items=items, next_values=next_values)

    async def lock_idempotency(self, scope: IdempotencyScope) -> None:
        self._ensure_active()
        await self._session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
            {
                "key": (
                    f"chat-run:{scope.principal_id}:{scope.client_id}:"
                    f"{scope.endpoint}:{scope.idempotency_key}"
                )
            },
        )

    async def get_run_by_scope(self, scope: IdempotencyScope) -> ChatRun | None:
        self._ensure_active()
        statement = _run_statement().where(
            ChatRunRow.workspace_id == self._workspace_id,
            ChatRunRow.principal_id == scope.principal_id,
            ChatRunRow.client_id == scope.client_id,
            ChatRunRow.endpoint == scope.endpoint,
            ChatRunRow.idempotency_key == scope.idempotency_key,
        )
        row = (await self._session.execute(statement)).one_or_none()
        return _run(row[0], row[1]) if row is not None else None

    async def get_run(
        self, run_id: UUID, *, principal_id: str, client_id: str
    ) -> ChatRun | None:
        self._ensure_active()
        statement = _run_statement().where(
            ChatRunRow.workspace_id == self._workspace_id,
            ChatRunRow.id == run_id,
            ChatRunRow.principal_id == principal_id,
            ChatRunRow.client_id == client_id,
        )
        row = (await self._session.execute(statement)).one_or_none()
        return _run(row[0], row[1]) if row is not None else None

    async def create_run(
        self,
        *,
        scope: IdempotencyScope,
        request_hash: str,
        kb_id: UUID,
        session_id: UUID,
        index_revision_id: UUID,
        message: str,
        requested_policy: dict[str, Any],
        effective_policy: dict[str, Any],
        retrieval_strategy: dict[str, Any],
        model_configuration: dict[str, Any],
    ) -> ChatRun:
        self._ensure_active()
        user_message = ChatMessageRow(
            workspace_id=self._workspace_id,
            session_id=session_id,
            role=ChatMessageRole.USER,
            assistant_status=None,
            client_request_id=scope.idempotency_key,
            content=message,
        )
        self._session.add(user_message)
        await self._session.flush()

        run = ChatRunRow(
            workspace_id=self._workspace_id,
            kb_id=kb_id,
            session_id=session_id,
            user_message_id=user_message.id,
            index_revision_id=index_revision_id,
            status=ChatRunStatus.QUEUED,
            principal_id=scope.principal_id,
            client_id=scope.client_id,
            endpoint=scope.endpoint,
            idempotency_key=scope.idempotency_key,
            request_hash=request_hash,
            requested_policy=dict(requested_policy),
            effective_policy=dict(effective_policy),
            retrieval_strategy=dict(retrieval_strategy),
            model_configuration=dict(model_configuration),
        )
        self._session.add(run)
        await self._session.flush()

        assistant = ChatMessageRow(
            workspace_id=self._workspace_id,
            session_id=session_id,
            chat_run_id=run.id,
            role=ChatMessageRole.ASSISTANT,
            assistant_status=AssistantMessageStatus.GENERATING,
            client_request_id=None,
            content="",
        )
        self._session.add(assistant)
        await self._session.flush()
        return _run(run, assistant)


def _run_statement():
    return (
        select(ChatRunRow, ChatMessageRow)
        .join(
            ChatMessageRow,
            (ChatMessageRow.chat_run_id == ChatRunRow.id)
            & (ChatMessageRow.role == ChatMessageRole.ASSISTANT),
        )
    )


def _with_after(
    statement: Select,
    column,
    id_column,
    after: tuple[str, ...] | None,
    descending: bool,
) -> Select:
    if after is None:
        return statement
    position = datetime.fromisoformat(after[0])
    resource_id = UUID(after[1])
    if descending:
        return statement.where(
            (column < position) | ((column == position) & (id_column < resource_id))
        )
    return statement.where(
        (column > position) | ((column == position) & (id_column > resource_id))
    )


def _session(row: ChatSessionRow) -> ChatSession:
    return ChatSession(
        id=row.id,
        workspace_id=row.workspace_id,
        kb_id=row.kb_id,
        principal_id=row.principal_id,
        title=row.title,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _message(row: ChatMessageRow) -> ChatMessage:
    return ChatMessage(
        id=row.id,
        session_id=row.session_id,
        chat_run_id=row.chat_run_id,
        role=row.role.value,
        assistant_status=(
            row.assistant_status.value if row.assistant_status is not None else None
        ),
        client_request_id=row.client_request_id,
        content=row.content,
        created_at=row.created_at,
    )


def _run(run: ChatRunRow, assistant: ChatMessageRow) -> ChatRun:
    if run.effective_policy is None:
        raise RuntimeError("persisted ChatRun is missing its effective policy")
    if assistant.assistant_status is None:
        raise RuntimeError("persisted assistant message is missing its status")
    return ChatRun(
        id=run.id,
        workspace_id=run.workspace_id,
        kb_id=run.kb_id,
        session_id=run.session_id,
        user_message_id=run.user_message_id,
        assistant_message_id=assistant.id,
        index_revision_id=run.index_revision_id,
        status=run.status.value,
        principal_id=run.principal_id,
        client_id=run.client_id,
        endpoint=run.endpoint,
        idempotency_key=run.idempotency_key,
        request_hash=run.request_hash,
        requested_policy=dict(run.requested_policy),
        effective_policy=dict(run.effective_policy),
        retrieval_strategy=dict(run.retrieval_strategy),
        model_configuration=dict(run.model_configuration),
        assistant_status=assistant.assistant_status.value,
        assistant_content=assistant.content,
        attempt=run.attempt,
        error_code=run.error_code,
        error_detail=dict(run.error_detail) if run.error_detail is not None else None,
        error_retryable=run.error_retryable,
        usage=dict(run.usage) if run.usage is not None else None,
        timing=dict(run.timing) if run.timing is not None else None,
        created_at=run.created_at,
        updated_at=run.updated_at,
        completed_at=run.completed_at,
    )
