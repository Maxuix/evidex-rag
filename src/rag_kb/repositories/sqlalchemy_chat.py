"""SQLAlchemy persistence for sessions, messages, and ChatRuns."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Select, delete, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from rag_kb.db.models import (
    AssistantMessageStatus,
    ChatMessage as ChatMessageRow,
    ChatMessageRole,
    ChatRun as ChatRunRow,
    ChatRunStatus,
    ChatSession as ChatSessionRow,
    Citation as CitationRow,
)
from rag_kb.domain import (
    ChatCitation,
    ChatExecutionContext,
    ChatFailureSettlementCommand,
    ChatMessage,
    ChatRun,
    ChatRunLease,
    ChatTerminalSuccessCommand,
    ChatTerminalWriteStatus,
    ChatSession,
    ContextualizedQuery,
    ConversationTurn,
    ErrorCode,
    IdempotencyScope,
    Page,
    ReconciliationResult,
)
from rag_kb.memory import (
    hydrate_contextualized_query,
    hydrate_conversation_context,
    serialize_contextualized_query,
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

    async def oldest_claimable_at(
        self, *, observed_at: datetime, max_attempts: int
    ) -> datetime | None:
        self._ensure_active()
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        return await self._session.scalar(
            select(ChatRunRow.created_at)
            .where(
                *_claimable_run(
                    observed_at,
                    max_attempts,
                    self._workspace_id,
                )
            )
            .order_by(ChatRunRow.created_at, ChatRunRow.id)
            .limit(1)
        )

    async def claim_run(
        self, *, worker_id: str, observed_at: datetime, max_attempts: int
    ) -> ChatRunLease | None:
        self._ensure_active()
        if not worker_id.strip() or max_attempts < 1:
            raise ValueError("worker_id and max_attempts must be valid")
        row = await self._session.scalar(
            select(ChatRunRow)
            .where(*_claimable_run(observed_at, max_attempts, self._workspace_id))
            .order_by(
                ChatRunRow.next_attempt_at.asc().nullsfirst(),
                ChatRunRow.created_at.asc(),
                ChatRunRow.id.asc(),
            )
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if row is None:
            return None
        row.status = ChatRunStatus.RUNNING
        row.attempt += 1
        row.claimed_by = worker_id
        row.claimed_at = observed_at
        row.heartbeat_at = observed_at
        row.next_attempt_at = None
        row.error_code = None
        row.error_detail = None
        row.error_retryable = None
        await self._session.flush()
        return ChatRunLease(
            run_id=row.id,
            workspace_id=row.workspace_id,
            claimed_by=worker_id,
            attempt=row.attempt,
            claimed_at=observed_at,
        )

    async def reconcile_stale_runs(
        self,
        *,
        stale_before: datetime,
        observed_at: datetime,
        max_attempts: int,
        retry_at_by_attempt: tuple[datetime, ...],
        limit: int,
    ) -> ReconciliationResult:
        self._ensure_active()
        if max_attempts < 1 or limit < 1:
            raise ValueError("reconciliation limits must be positive")
        if len(retry_at_by_attempt) < max_attempts:
            raise ValueError("retry schedule must cover every configured attempt")
        rows = (
            await self._session.execute(
                select(ChatRunRow, ChatMessageRow)
                .join(
                    ChatMessageRow,
                    (ChatMessageRow.chat_run_id == ChatRunRow.id)
                    & (ChatMessageRow.role == ChatMessageRole.ASSISTANT),
                )
                .where(
                    ChatRunRow.workspace_id == self._workspace_id,
                    ChatMessageRow.workspace_id == self._workspace_id,
                    ChatRunRow.status == ChatRunStatus.RUNNING,
                    ChatRunRow.heartbeat_at <= stale_before,
                )
                .order_by(ChatRunRow.heartbeat_at, ChatRunRow.id)
                .limit(limit)
                .with_for_update(
                    of=(ChatRunRow, ChatMessageRow),
                    skip_locked=True,
                )
            )
        ).all()
        requeued = 0
        failed = 0
        for run, assistant in rows:
            claimed_at = run.claimed_at or run.created_at
            exhausted = run.attempt >= max_attempts
            next_attempt_at = (
                None
                if exhausted
                else retry_at_by_attempt[run.attempt - 1]
            )
            result = "failed" if exhausted else "requeued"
            record = {
                "result": result,
                "phase": "load_context",
                "error_code": ErrorCode.CHAT_STALE_WORKER.value,
                "retryable": True,
                "exhausted": exhausted,
                "diagnostic": {"check": "stale_heartbeat"},
                "claimed_by": run.claimed_by or "unknown",
                "claimed_at": claimed_at.isoformat(),
                "finished_at": observed_at.isoformat(),
                "duration_ms": _duration_ms(claimed_at, observed_at),
            }
            context_calls = _contextualization_calls(run)
            if context_calls:
                run.usage = _merge_usage(run.usage, context_calls)
            run.timing = _merge_timing(run.timing, run.attempt, record)
            run.status = (
                ChatRunStatus.FAILED if exhausted else ChatRunStatus.QUEUED
            )
            run.error_code = ErrorCode.CHAT_STALE_WORKER.value
            run.error_detail = {"check": "stale_heartbeat"}
            run.error_retryable = True
            run.next_attempt_at = next_attempt_at
            run.claimed_by = None
            run.claimed_at = None
            run.heartbeat_at = None
            run.completed_at = observed_at if exhausted else None
            assistant.content = ""
            assistant.assistant_status = (
                AssistantMessageStatus.FAILED
                if exhausted
                else AssistantMessageStatus.GENERATING
            )
            if exhausted:
                failed += 1
            else:
                requeued += 1
        await self._session.flush()
        return ReconciliationResult(requeued=requeued, failed=failed)

    async def complete_owned_run(
        self, command: ChatTerminalSuccessCommand
    ) -> ChatTerminalWriteStatus:
        self._ensure_active()
        locked = await self._lock_terminal_rows(command.lease)
        if locked is None:
            return ChatTerminalWriteStatus.STALE
        run, assistant = locked
        calls = _combine_calls(
            _contextualization_calls(run),
            _serialized_calls(command.lease.attempt, command.model_calls),
        )
        validation = _serialized_validation(command)
        stable_success = {
            **validation,
            "claimed_by": command.lease.claimed_by,
            "claimed_at": command.lease.claimed_at.isoformat(),
        }
        citations = await self._citation_rows(assistant.id)

        if run.status == ChatRunStatus.COMPLETED:
            if (
                assistant.id == command.assistant_message_id
                and assistant.assistant_status == AssistantMessageStatus.COMPLETED
                and assistant.content == command.rendered.content
                and _citations_equal(citations, command)
                and _stored_attempt_calls(run.usage, command.lease.attempt) == calls
                and _stored_success_facts(run.timing, command.lease.attempt)
                == stable_success
            ):
                return ChatTerminalWriteStatus.IDEMPOTENT
            return ChatTerminalWriteStatus.STALE

        if not _owns_running_lease(run, command.lease) or (
            assistant.id != command.assistant_message_id
            or assistant.assistant_status != AssistantMessageStatus.GENERATING
        ):
            return ChatTerminalWriteStatus.STALE
        if citations:
            return ChatTerminalWriteStatus.STALE

        usage = _merge_usage(run.usage, calls)
        timing = _merge_timing(
            run.timing,
            command.lease.attempt,
            {
                **validation,
                "claimed_by": command.lease.claimed_by,
                "claimed_at": command.lease.claimed_at.isoformat(),
                "finished_at": command.finished_at.isoformat(),
                "duration_ms": _duration_ms(
                    command.lease.claimed_at, command.finished_at
                ),
            },
        )
        assistant.assistant_status = AssistantMessageStatus.COMPLETED
        assistant.content = command.rendered.content
        self._session.add_all(
            [
                CitationRow(
                    workspace_id=self._workspace_id,
                    assistant_message_id=assistant.id,
                    ordinal=item.ordinal,
                    index_chunk_id=item.index_chunk_id,
                    document_id_snapshot=item.document_id,
                    document_version_id_snapshot=item.document_version_id,
                    quoted_text=item.quoted_text,
                    source_location=dict(item.source_location),
                    score=item.score,
                )
                for item in command.rendered.citations
            ]
        )
        run.status = ChatRunStatus.COMPLETED
        run.usage = usage
        run.timing = timing
        run.error_code = None
        run.error_detail = None
        run.error_retryable = None
        run.next_attempt_at = None
        run.claimed_by = None
        run.claimed_at = None
        run.heartbeat_at = None
        run.completed_at = command.finished_at
        await self._session.flush()
        return ChatTerminalWriteStatus.APPLIED

    async def settle_owned_failure(
        self, command: ChatFailureSettlementCommand
    ) -> ChatTerminalWriteStatus:
        self._ensure_active()
        locked = await self._lock_terminal_rows(command.lease)
        if locked is None:
            return ChatTerminalWriteStatus.STALE
        run, assistant = locked
        calls = _combine_calls(
            _contextualization_calls(run),
            _serialized_calls(command.lease.attempt, command.model_calls),
        )
        facts = _serialized_failure(command)
        stable_failure = {
            **facts,
            "claimed_by": command.lease.claimed_by,
            "claimed_at": command.lease.claimed_at.isoformat(),
        }
        expected_status = (
            ChatRunStatus.QUEUED
            if command.next_attempt_at is not None
            else ChatRunStatus.FAILED
        )
        expected_assistant = (
            AssistantMessageStatus.GENERATING
            if expected_status == ChatRunStatus.QUEUED
            else AssistantMessageStatus.FAILED
        )
        if run.status == expected_status:
            if (
                assistant.assistant_status == expected_assistant
                and assistant.content == ""
                and _stored_attempt_calls(run.usage, command.lease.attempt) == calls
                and _stored_failure_facts(run.timing, command.lease.attempt)
                == stable_failure
            ):
                return ChatTerminalWriteStatus.IDEMPOTENT
            return ChatTerminalWriteStatus.STALE
        if not _owns_running_lease(run, command.lease):
            return ChatTerminalWriteStatus.STALE

        usage = _merge_usage(run.usage, calls)
        timing = _merge_timing(
            run.timing,
            command.lease.attempt,
            {
                **facts,
                "claimed_by": command.lease.claimed_by,
                "claimed_at": command.lease.claimed_at.isoformat(),
                "finished_at": command.finished_at.isoformat(),
                "duration_ms": _duration_ms(
                    command.lease.claimed_at, command.finished_at
                ),
            },
        )
        await self._session.execute(
            delete(CitationRow).where(
                CitationRow.workspace_id == self._workspace_id,
                CitationRow.assistant_message_id == assistant.id,
            )
        )
        assistant.content = ""
        assistant.assistant_status = expected_assistant
        run.status = expected_status
        run.usage = usage
        run.timing = timing
        run.error_code = command.code.value
        run.error_detail = dict(command.diagnostic)
        run.error_retryable = command.retryable
        run.next_attempt_at = command.next_attempt_at
        run.claimed_by = None
        run.claimed_at = None
        run.heartbeat_at = None
        run.completed_at = (
            command.finished_at if expected_status == ChatRunStatus.FAILED else None
        )
        await self._session.flush()
        return ChatTerminalWriteStatus.APPLIED

    async def _lock_terminal_rows(
        self, lease: ChatRunLease
    ) -> tuple[ChatRunRow, ChatMessageRow] | None:
        if lease.workspace_id != self._workspace_id:
            return None
        run = await self._session.scalar(
            select(ChatRunRow)
            .where(
                ChatRunRow.workspace_id == self._workspace_id,
                ChatRunRow.id == lease.run_id,
            )
            .with_for_update()
        )
        if run is None:
            return None
        assistant = await self._session.scalar(
            select(ChatMessageRow)
            .where(
                ChatMessageRow.workspace_id == self._workspace_id,
                ChatMessageRow.chat_run_id == run.id,
                ChatMessageRow.role == ChatMessageRole.ASSISTANT,
            )
            .with_for_update()
        )
        if assistant is None:
            return None
        return run, assistant

    async def _citation_rows(self, assistant_message_id: UUID):
        return tuple(
            (
                await self._session.scalars(
                    select(CitationRow)
                    .where(
                        CitationRow.workspace_id == self._workspace_id,
                        CitationRow.assistant_message_id == assistant_message_id,
                    )
                    .order_by(CitationRow.ordinal)
                    .with_for_update()
                )
            ).all()
        )

    async def heartbeat_run(
        self, lease: ChatRunLease, *, observed_at: datetime
    ) -> bool:
        self._ensure_active()
        if lease.workspace_id != self._workspace_id:
            return False
        result = await self._session.execute(
            update(ChatRunRow)
            .where(
                ChatRunRow.workspace_id == self._workspace_id,
                ChatRunRow.id == lease.run_id,
                ChatRunRow.status == ChatRunStatus.RUNNING,
                ChatRunRow.claimed_by == lease.claimed_by,
                ChatRunRow.attempt == lease.attempt,
            )
            .values(heartbeat_at=observed_at, updated_at=observed_at)
        )
        return bool(result.rowcount == 1)

    async def load_execution_context(
        self, lease: ChatRunLease
    ) -> ChatExecutionContext | None:
        self._ensure_active()
        if lease.workspace_id != self._workspace_id:
            return None
        user = aliased(ChatMessageRow)
        assistant = aliased(ChatMessageRow)
        statement = (
            select(ChatRunRow, user, assistant)
            .join(user, user.id == ChatRunRow.user_message_id)
            .join(
                assistant,
                (assistant.chat_run_id == ChatRunRow.id)
                & (assistant.role == ChatMessageRole.ASSISTANT),
            )
            .where(
                ChatRunRow.workspace_id == self._workspace_id,
                ChatRunRow.id == lease.run_id,
                ChatRunRow.status == ChatRunStatus.RUNNING,
                ChatRunRow.claimed_by == lease.claimed_by,
                ChatRunRow.attempt == lease.attempt,
                user.role == ChatMessageRole.USER,
            )
        )
        result = (await self._session.execute(statement)).one_or_none()
        if result is None:
            return None
        run, user_message, assistant_message = result
        if run.effective_policy is None:
            return None
        try:
            conversation_context = hydrate_conversation_context(
                run.conversation_context
            )
            contextualized_query = (
                hydrate_contextualized_query(run.contextualized_query)
                if run.contextualized_query is not None
                else None
            )
        except (TypeError, ValueError):
            return None
        return ChatExecutionContext(
            lease=lease,
            run_id=run.id,
            workspace_id=run.workspace_id,
            knowledge_base_id=run.kb_id,
            session_id=run.session_id,
            user_message_id=run.user_message_id,
            assistant_message_id=assistant_message.id,
            index_revision_id=run.index_revision_id,
            principal_id=run.principal_id,
            client_id=run.client_id,
            query=user_message.content,
            effective_policy=run.effective_policy,
            retrieval_strategy=run.retrieval_strategy,
            model_configuration=run.model_configuration,
            attempt=run.attempt,
            conversation_context=conversation_context,
            contextualized_query=contextualized_query,
        )

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

    async def lock_session(
        self, session_id: UUID, *, principal_id: str
    ) -> ChatSession | None:
        self._ensure_active()
        row = await self._session.scalar(
            select(ChatSessionRow)
            .where(
                ChatSessionRow.workspace_id == self._workspace_id,
                ChatSessionRow.id == session_id,
                ChatSessionRow.principal_id == principal_id,
            )
            .with_for_update()
        )
        return _session(row) if row is not None else None

    async def has_nonterminal_run(self, session_id: UUID) -> bool:
        self._ensure_active()
        return bool(
            await self._session.scalar(
                select(ChatRunRow.id)
                .where(
                    ChatRunRow.workspace_id == self._workspace_id,
                    ChatRunRow.session_id == session_id,
                    ChatRunRow.status.in_(
                        (ChatRunStatus.QUEUED, ChatRunStatus.RUNNING)
                    ),
                )
                .limit(1)
            )
        )

    async def list_completed_turns(
        self,
        *,
        session_id: UUID,
        principal_id: str,
        kb_id: UUID,
        limit: int,
    ) -> tuple[ConversationTurn, ...]:
        self._ensure_active()
        if limit < 1:
            raise ValueError("completed turn limit must be positive")
        user = aliased(ChatMessageRow)
        assistant = aliased(ChatMessageRow)
        rows = (
            await self._session.execute(
                select(user, assistant)
                .join(ChatRunRow, ChatRunRow.user_message_id == user.id)
                .join(
                    assistant,
                    (assistant.id == ChatRunRow.assistant_message_id)
                    & (assistant.chat_run_id == ChatRunRow.id)
                    & (assistant.role == ChatMessageRole.ASSISTANT),
                )
                .where(
                    ChatRunRow.workspace_id == self._workspace_id,
                    ChatRunRow.session_id == session_id,
                    ChatRunRow.kb_id == kb_id,
                    ChatRunRow.principal_id == principal_id,
                    ChatRunRow.status == ChatRunStatus.COMPLETED,
                    user.workspace_id == self._workspace_id,
                    user.session_id == session_id,
                    user.role == ChatMessageRole.USER,
                    assistant.workspace_id == self._workspace_id,
                    assistant.session_id == session_id,
                    assistant.assistant_status == AssistantMessageStatus.COMPLETED,
                )
                .order_by(ChatRunRow.created_at.desc(), ChatRunRow.id.desc())
                .limit(limit)
            )
        ).all()
        return tuple(
            ConversationTurn(
                user_message_id=user_row.id,
                user_content=user_row.content,
                assistant_message_id=assistant_row.id,
                assistant_content=assistant_row.content,
            )
            for user_row, assistant_row in rows
        )

    async def save_contextualized_query(
        self,
        lease: ChatRunLease,
        value: ContextualizedQuery,
    ) -> ContextualizedQuery | None:
        self._ensure_active()
        if lease.workspace_id != self._workspace_id:
            return None
        row = await self._session.scalar(
            select(ChatRunRow)
            .where(
                ChatRunRow.workspace_id == self._workspace_id,
                ChatRunRow.id == lease.run_id,
                ChatRunRow.status == ChatRunStatus.RUNNING,
                ChatRunRow.claimed_by == lease.claimed_by,
                ChatRunRow.attempt == lease.attempt,
            )
            .with_for_update()
        )
        if row is None:
            return None
        try:
            snapshot = hydrate_conversation_context(row.conversation_context)
        except (TypeError, ValueError):
            return None
        if snapshot.content_hash != value.context_hash:
            return None
        if row.contextualized_query is None:
            row.contextualized_query = serialize_contextualized_query(value)
            await self._session.flush()
            return value
        try:
            return hydrate_contextualized_query(row.contextualized_query)
        except (TypeError, ValueError):
            return None

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
        rows = (await self._session.execute(statement)).all()
        if not rows:
            return None
        return _run_rows(rows)

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
        rows = (await self._session.execute(statement)).all()
        if not rows:
            return None
        return _run_rows(rows)

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
        conversation_context: dict[str, Any],
        contextualized_query: dict[str, Any] | None,
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
            conversation_context=dict(conversation_context),
            contextualized_query=(
                dict(contextualized_query)
                if contextualized_query is not None
                else None
            ),
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
        select(ChatRunRow, ChatMessageRow, CitationRow)
        .join(
            ChatMessageRow,
            (ChatMessageRow.chat_run_id == ChatRunRow.id)
            & (ChatMessageRow.role == ChatMessageRole.ASSISTANT),
        )
        .outerjoin(
            CitationRow,
            (CitationRow.assistant_message_id == ChatMessageRow.id)
            & (CitationRow.workspace_id == ChatRunRow.workspace_id),
        )
        .order_by(CitationRow.ordinal)
    )


def _claimable_run(
    observed_at: datetime,
    max_attempts: int,
    workspace_id: UUID,
):
    return (
        ChatRunRow.workspace_id == workspace_id,
        ChatRunRow.status == ChatRunStatus.QUEUED,
        ChatRunRow.attempt < max_attempts,
        or_(
            ChatRunRow.next_attempt_at.is_(None),
            ChatRunRow.next_attempt_at <= observed_at,
        ),
    )


def _run_rows(rows) -> ChatRun:
    run, assistant, _ = rows[0]
    citations = tuple(
        ChatCitation(
            ordinal=citation.ordinal,
            index_chunk_id=citation.index_chunk_id,
            document_id=citation.document_id_snapshot,
            document_version_id=citation.document_version_id_snapshot,
            quoted_text=citation.quoted_text,
            source_location=dict(citation.source_location),
            score=citation.score,
        )
        for _, _, citation in rows
        if citation is not None
    )
    return _run(run, assistant, citations)


def _owns_running_lease(run: ChatRunRow, lease: ChatRunLease) -> bool:
    return bool(
        run.status == ChatRunStatus.RUNNING
        and run.workspace_id == lease.workspace_id
        and run.id == lease.run_id
        and run.claimed_by == lease.claimed_by
        and run.attempt == lease.attempt
    )


def _serialized_calls(attempt: int, calls) -> dict[str, dict[str, Any]]:
    return {
        f"{attempt}:{sequence}:{call.operation.value}": {
            "attempt": attempt,
            "sequence": sequence,
            "operation": call.operation.value,
            "model": call.model,
            "provider_request_id": call.provider_request_id,
            "usage": dict(call.usage),
        }
        for sequence, call in enumerate(calls, start=1)
    }


def _contextualization_calls(run: ChatRunRow) -> dict[str, dict[str, Any]]:
    if run.contextualized_query is None:
        return {}
    value = hydrate_contextualized_query(run.contextualized_query)
    if value.origin_attempt is None:
        return {}
    return _serialized_calls(value.origin_attempt, value.model_calls)


def _combine_calls(
    *ledgers: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}
    for ledger in ledgers:
        for key, value in ledger.items():
            if key in combined and combined[key] != value:
                raise RuntimeError("chat model call ledger conflict")
            combined[key] = value
    return combined


def _merge_usage(
    stored: dict[str, Any] | None, calls: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    existing = dict((stored or {}).get("calls", {}))
    for key, value in calls.items():
        if key in existing and existing[key] != value:
            raise RuntimeError("chat model call ledger conflict")
        existing[key] = value
    totals: dict[str, int] = {}
    for item in existing.values():
        for name, value in item["usage"].items():
            totals[name] = totals.get(name, 0) + value
    return {"calls": existing, "totals": totals}


def _stored_attempt_calls(
    stored: dict[str, Any] | None, attempt: int
) -> dict[str, dict[str, Any]]:
    prefix = f"{attempt}:"
    return {
        key: value
        for key, value in (stored or {}).get("calls", {}).items()
        if key.startswith(prefix)
    }


def _merge_timing(
    stored: dict[str, Any] | None, attempt: int, record: dict[str, Any]
) -> dict[str, Any]:
    attempts = dict((stored or {}).get("attempts", {}))
    key = str(attempt)
    if key in attempts and attempts[key] != record:
        raise RuntimeError("chat attempt timing ledger conflict")
    attempts[key] = record
    return {"attempts": attempts}


def _serialized_validation(command: ChatTerminalSuccessCommand) -> dict[str, Any]:
    validation = command.validation
    return {
        "result": "completed",
        "phase": "persist_result",
        "outcome": command.rendered.outcome.value,
        "citation_ids": [item.citation_id for item in command.rendered.citations],
        "validation": {
            "initial_issues": [item.value for item in validation.initial_issues],
            "repair_issues": [item.value for item in validation.repair_issues],
            "repair_attempted": validation.repair_attempted,
            "repair_succeeded": validation.repair_succeeded,
            "safe_fallback": validation.safe_fallback,
        },
    }


def _serialized_failure(command: ChatFailureSettlementCommand) -> dict[str, Any]:
    return {
        "result": "requeued" if command.next_attempt_at is not None else "failed",
        "phase": command.phase.value,
        "error_code": command.code.value,
        "retryable": command.retryable,
        "exhausted": command.exhausted,
        "diagnostic": dict(command.diagnostic),
    }


def _stable_attempt_facts(
    stored: dict[str, Any] | None, attempt: int
) -> dict[str, Any] | None:
    record = (stored or {}).get("attempts", {}).get(str(attempt))
    if record is None:
        return None
    return {
        key: value
        for key, value in record.items()
        if key not in {"finished_at", "duration_ms"}
    }


def _stored_success_facts(
    stored: dict[str, Any] | None, attempt: int
) -> dict[str, Any] | None:
    return _stable_attempt_facts(stored, attempt)


def _stored_failure_facts(
    stored: dict[str, Any] | None, attempt: int
) -> dict[str, Any] | None:
    return _stable_attempt_facts(stored, attempt)


def _citations_equal(rows, command: ChatTerminalSuccessCommand) -> bool:
    expected = command.rendered.citations
    return len(rows) == len(expected) and all(
        row.ordinal == item.ordinal
        and row.index_chunk_id == item.index_chunk_id
        and row.document_id_snapshot == item.document_id
        and row.document_version_id_snapshot == item.document_version_id
        and row.quoted_text == item.quoted_text
        and row.source_location == dict(item.source_location)
        and row.score == item.score
        for row, item in zip(rows, expected, strict=True)
    )


def _duration_ms(started_at: datetime, finished_at: datetime) -> int:
    return max(0, round((finished_at - started_at).total_seconds() * 1000))


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


def _run(
    run: ChatRunRow,
    assistant: ChatMessageRow,
    citations: tuple[ChatCitation, ...] = (),
) -> ChatRun:
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
        citations=citations,
        attempt=run.attempt,
        error_code=run.error_code,
        error_detail=dict(run.error_detail) if run.error_detail is not None else None,
        error_retryable=run.error_retryable,
        usage=dict(run.usage) if run.usage is not None else None,
        timing=dict(run.timing) if run.timing is not None else None,
        created_at=run.created_at,
        updated_at=run.updated_at,
        completed_at=run.completed_at,
        conversation_context=dict(run.conversation_context),
        contextualized_query=(
            dict(run.contextualized_query)
            if run.contextualized_query is not None
            else None
        ),
    )
