"""Immutable contracts for one claimed direct chat execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any
from uuid import UUID

from rag_kb.domain.answering import ChatAnsweringState, ChatModelCallRecord
from rag_kb.domain.errors import ErrorCode
from rag_kb.domain.retrieval import EvidencePack


class ChatPipelinePhase(StrEnum):
    LOAD_CONTEXT = "load_context"
    RETRIEVE_EVIDENCE = "retrieve_evidence"
    ASSESS_EVIDENCE = "assess_evidence"
    GENERATE_OR_REFUSE = "generate_or_refuse"
    VALIDATE_STRUCTURE = "validate_structure"
    PERSIST_RESULT = "persist_result"


@dataclass(frozen=True, slots=True)
class ChatRunLease:
    run_id: UUID
    workspace_id: UUID
    claimed_by: str
    attempt: int
    claimed_at: datetime

    def __post_init__(self) -> None:
        if not self.claimed_by.strip():
            raise ValueError("claimed_by must not be empty")
        if self.attempt < 1:
            raise ValueError("attempt must be positive")
        if self.claimed_at.tzinfo is None or self.claimed_at.utcoffset() is None:
            raise ValueError("claimed_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class ChatExecutionCommand:
    lease: ChatRunLease


@dataclass(frozen=True, slots=True)
class ChatExecutionContext:
    lease: ChatRunLease
    run_id: UUID
    workspace_id: UUID
    knowledge_base_id: UUID
    session_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID
    index_revision_id: UUID
    principal_id: str
    client_id: str
    query: str
    effective_policy: Mapping[str, Any]
    retrieval_strategy: Mapping[str, Any]
    model_configuration: Mapping[str, Any]
    attempt: int

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("chat query must not be empty")
        if self.attempt < 1:
            raise ValueError("attempt must be positive")
        if (
            self.lease.run_id != self.run_id
            or self.lease.workspace_id != self.workspace_id
            or self.lease.attempt != self.attempt
        ):
            raise ValueError("execution context must preserve its claimed lease")
        object.__setattr__(
            self, "effective_policy", _frozen_mapping(self.effective_policy)
        )
        object.__setattr__(
            self, "retrieval_strategy", _frozen_mapping(self.retrieval_strategy)
        )
        object.__setattr__(
            self, "model_configuration", _frozen_mapping(self.model_configuration)
        )


@dataclass(frozen=True, slots=True)
class ChatPipelineState:
    context: ChatExecutionContext | None = None
    evidence_pack: EvidencePack | None = None
    answering: ChatAnsweringState | None = None
    artifacts: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", _frozen_mapping(self.artifacts))


@dataclass(frozen=True, slots=True)
class ChatModelMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError("unsupported chat model message role")
        if not self.content:
            raise ValueError("chat model message content must not be empty")


@dataclass(frozen=True, slots=True)
class ChatModelRequest:
    messages: tuple[ChatModelMessage, ...]

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("chat model request must contain messages")


@dataclass(frozen=True, slots=True)
class ChatModelResponse:
    content: str
    model: str
    finish_reason: str | None
    provider_request_id: str | None
    usage: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.content or not self.model:
            raise ValueError("chat model response content and model are required")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.usage.values()
        ):
            raise ValueError("chat model usage values must be non-negative integers")
        object.__setattr__(self, "usage", MappingProxyType(dict(self.usage)))


class ChatPipelineExecutionError(RuntimeError):
    """Stable, content-safe failure at a direct pipeline boundary."""

    def __init__(
        self,
        code: ErrorCode,
        *,
        phase: ChatPipelinePhase,
        diagnostic: Mapping[str, Any] | None = None,
        model_calls: tuple[ChatModelCallRecord, ...] = (),
    ) -> None:
        super().__init__(code.value)
        self.code = code
        self.phase = phase
        self.diagnostic = dict(diagnostic or {})
        self.model_calls = model_calls

    def retain_model_calls(
        self, calls: tuple[ChatModelCallRecord, ...]
    ) -> ChatPipelineExecutionError:
        """Retain completed provider calls without duplicating an existing prefix."""

        if len(calls) > len(self.model_calls):
            self.model_calls = calls
        return self


class ChatModelExecutionError(RuntimeError):
    """Stable, content-safe chat provider failure."""

    def __init__(
        self, code: ErrorCode, *, diagnostic: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(code.value)
        self.code = code
        self.diagnostic = dict(diagnostic or {})


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {key: _frozen_value(item) for key, item in value.items()}
    )


def _frozen_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _frozen_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_frozen_value(item) for item in value)
    return value
