"""Async persistence contract for durable chat state."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from rag_kb.domain import (
    ChatExecutionContext,
    ChatFailureSettlementCommand,
    ChatMessage,
    ContextualizedQuery,
    ConversationTurn,
    ChatRun,
    ChatRunLease,
    ChatSession,
    ChatTerminalSuccessCommand,
    ChatTerminalWriteStatus,
    IdempotencyScope,
    Page,
    ReconciliationResult,
)


class ChatRepository(Protocol):
    async def reconcile_stale_runs(
        self,
        *,
        stale_before: datetime,
        observed_at: datetime,
        max_attempts: int,
        retry_at_by_attempt: tuple[datetime, ...],
        limit: int,
    ) -> ReconciliationResult: ...

    async def complete_owned_run(
        self, command: ChatTerminalSuccessCommand
    ) -> ChatTerminalWriteStatus: ...

    async def settle_owned_failure(
        self, command: ChatFailureSettlementCommand
    ) -> ChatTerminalWriteStatus: ...

    async def claim_run(
        self, *, worker_id: str, observed_at: datetime, max_attempts: int
    ) -> ChatRunLease | None: ...

    async def heartbeat_run(
        self, lease: ChatRunLease, *, observed_at: datetime
    ) -> bool: ...

    async def load_execution_context(
        self, lease: ChatRunLease
    ) -> ChatExecutionContext | None: ...

    async def create_session(
        self, *, kb_id: UUID, principal_id: str, title: str | None
    ) -> ChatSession: ...

    async def get_session(
        self, session_id: UUID, *, principal_id: str
    ) -> ChatSession | None: ...

    async def lock_session(
        self, session_id: UUID, *, principal_id: str
    ) -> ChatSession | None: ...

    async def has_nonterminal_run(self, session_id: UUID) -> bool: ...

    async def list_completed_turns(
        self,
        *,
        session_id: UUID,
        principal_id: str,
        kb_id: UUID,
        limit: int,
    ) -> tuple[ConversationTurn, ...]: ...

    async def save_contextualized_query(
        self,
        lease: ChatRunLease,
        value: ContextualizedQuery,
    ) -> ContextualizedQuery | None: ...

    async def list_sessions(
        self,
        *,
        principal_id: str,
        limit: int,
        sort: str,
        after: tuple[str, ...] | None,
        kb_id: UUID | None = None,
    ) -> Page[ChatSession]: ...

    async def list_messages(
        self,
        *,
        session_id: UUID,
        principal_id: str,
        limit: int,
        sort: str,
        after: tuple[str, ...] | None,
    ) -> Page[ChatMessage] | None: ...

    async def lock_idempotency(self, scope: IdempotencyScope) -> None: ...

    async def get_run_by_scope(self, scope: IdempotencyScope) -> ChatRun | None: ...

    async def get_run(
        self, run_id: UUID, *, principal_id: str, client_id: str
    ) -> ChatRun | None: ...

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
    ) -> ChatRun: ...
