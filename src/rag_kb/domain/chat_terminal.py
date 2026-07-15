"""Immutable commands for lease-owned chat terminal writes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any
from uuid import UUID

from rag_kb.domain.answering import (
    AnswerValidationRecord,
    ChatModelCallRecord,
    RenderedAnswer,
)
from rag_kb.domain.chat_pipeline import ChatPipelinePhase, ChatRunLease
from rag_kb.domain.errors import ErrorCode


class ChatTerminalWriteStatus(StrEnum):
    APPLIED = "applied"
    IDEMPOTENT = "idempotent"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class ChatTerminalSuccessCommand:
    lease: ChatRunLease
    assistant_message_id: UUID
    rendered: RenderedAnswer
    validation: AnswerValidationRecord
    model_calls: tuple[ChatModelCallRecord, ...]
    finished_at: datetime

    def __post_init__(self) -> None:
        _require_finish_time(self.lease, self.finished_at)


@dataclass(frozen=True, slots=True)
class ChatFailureSettlementCommand:
    lease: ChatRunLease
    phase: ChatPipelinePhase
    code: ErrorCode
    diagnostic: Mapping[str, Any]
    model_calls: tuple[ChatModelCallRecord, ...]
    retryable: bool
    exhausted: bool
    finished_at: datetime
    next_attempt_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_finish_time(self.lease, self.finished_at)
        should_requeue = self.retryable and not self.exhausted
        if should_requeue != (self.next_attempt_at is not None):
            raise ValueError("next attempt time must match retry disposition")
        if self.next_attempt_at is not None:
            if (
                self.next_attempt_at.tzinfo is None
                or self.next_attempt_at.utcoffset() is None
                or self.next_attempt_at <= self.finished_at
            ):
                raise ValueError("next attempt time must be timezone-aware and future")
        object.__setattr__(self, "diagnostic", MappingProxyType(dict(self.diagnostic)))


def _require_finish_time(lease: ChatRunLease, finished_at: datetime) -> None:
    if finished_at.tzinfo is None or finished_at.utcoffset() is None:
        raise ValueError("finished_at must be timezone-aware")
    if finished_at < lease.claimed_at:
        raise ValueError("finished_at cannot precede claim")
