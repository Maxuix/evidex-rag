"""Async persistence contract for durable chat state."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from rag_kb.domain import (
    ChatExecutionContext,
    ChatFailureSettlementCommand,
    ChatMessage,
    ChatRun,
    ChatRunLease,
    ChatTerminalSuccessCommand,
    ChatTerminalWriteStatus,
    ChatSession,
    IdempotencyScope,
    Page,
)


@runtime_checkable
class ChatRepository(Protocol):
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

    async def list_sessions(
        self,
        *,
        principal_id: str,
        limit: int,
        sort: str,
        after: tuple[str, ...] | None,
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
    ) -> ChatRun: ...
