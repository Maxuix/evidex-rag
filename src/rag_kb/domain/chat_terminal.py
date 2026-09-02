"""Immutable commands for lease-owned chat terminal writes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any
from uuid import UUID

from rag_kb.domain.answering import ChatModelCallRecord, RenderedAnswer
from rag_kb.domain.chat_agent import CHAT_AGENT_ACCEPTED_VERSIONS
from rag_kb.domain.chat_pipeline import ChatPipelinePhase, ChatRunLease
from rag_kb.domain.errors import ErrorCode
from rag_kb.domain.composite import VisualEvidenceDecision


class ChatTerminalWriteStatus(StrEnum):
    APPLIED = "applied"
    IDEMPOTENT = "idempotent"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class ChatTerminalSuccessCommand:
    lease: ChatRunLease
    assistant_message_id: UUID
    rendered: RenderedAnswer
    model_calls: tuple[ChatModelCallRecord, ...]
    finished_at: datetime
    retrieval_diagnostics: Mapping[str, int] = field(default_factory=dict)
    visual_decisions: tuple[VisualEvidenceDecision, ...] = ()
    visual_image_count: int = 0
    visual_total_bytes: int = 0
    agent_trace: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_finish_time(self.lease, self.finished_at)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.retrieval_diagnostics.values()
        ):
            raise ValueError("retrieval diagnostic counts must be non-negative")
        if not 0 <= self.visual_image_count <= 4 or self.visual_total_bytes < 0:
            raise ValueError("visual terminal counters are invalid")
        if sum(item.selected for item in self.visual_decisions) < self.visual_image_count:
            raise ValueError("attached images require selected visual decisions")
        object.__setattr__(
            self,
            "retrieval_diagnostics",
            MappingProxyType(dict(self.retrieval_diagnostics)),
        )
        if self.agent_trace is not None:
            if (
                self.agent_trace.get("version") not in CHAT_AGENT_ACCEPTED_VERSIONS
                or not isinstance(self.agent_trace.get("events"), (list, tuple))
                or len(self.agent_trace["events"]) > 32
            ):
                raise ValueError("terminal agent trace is invalid")
            object.__setattr__(
                self, "agent_trace", MappingProxyType(dict(self.agent_trace))
            )


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
    agent_trace: Mapping[str, Any] | None = None

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
        if self.agent_trace is not None:
            if (
                self.agent_trace.get("version") not in CHAT_AGENT_ACCEPTED_VERSIONS
                or self.agent_trace.get("outcome") is not None
                or not isinstance(self.agent_trace.get("events"), (list, tuple))
                or len(self.agent_trace["events"]) > 32
                or not isinstance(self.agent_trace.get("usage"), Mapping)
                or not isinstance(self.agent_trace.get("diagnostics"), Mapping)
                or self.agent_trace["diagnostics"].get("partial") is not True
            ):
                raise ValueError("partial terminal agent trace is invalid")
            object.__setattr__(
                self, "agent_trace", MappingProxyType(dict(self.agent_trace))
            )


def _require_finish_time(lease: ChatRunLease, finished_at: datetime) -> None:
    if finished_at.tzinfo is None or finished_at.utcoffset() is None:
        raise ValueError("finished_at must be timezone-aware")
    if finished_at < lease.claimed_at:
        raise ValueError("finished_at cannot precede claim")
