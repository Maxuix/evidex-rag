"""Immutable contracts for one claimed native tool-calling execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any
from uuid import UUID

from rag_kb.domain.answering import (
    ChatAnsweringState,
    ChatModelCallRecord,
    ChatModelVisualContent,
)
from rag_kb.domain.errors import ErrorCode
from rag_kb.domain.retrieval import EvidencePack
from rag_kb.domain.memory import (
    ConversationContextSnapshot,
    empty_context_snapshot,
)


class ChatPipelinePhase(StrEnum):
    LOAD_CONTEXT = "load_context"
    RETRIEVE_EVIDENCE = "retrieve_evidence"
    PREPARE_VISUAL_EVIDENCE = "prepare_visual_evidence"
    GENERATE_OR_REFUSE = "generate_or_refuse"
    PERSIST_RESULT = "persist_result"


class ChatToolChoice(StrEnum):
    AUTO = "auto"
    REQUIRED = "required"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class ChatToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not self.name.strip()
            or len(self.name) > 64
            or not self.description.strip()
            or len(self.description) > 1024
            or self.input_schema.get("type") != "object"
        ):
            raise ValueError("chat tool definition is invalid")
        object.__setattr__(self, "input_schema", _frozen_mapping(self.input_schema))


@dataclass(frozen=True, slots=True)
class ChatToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.id.strip() or len(self.id) > 128 or not self.name.strip():
            raise ValueError("chat tool call is invalid")
        object.__setattr__(self, "arguments", _frozen_mapping(self.arguments))


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
    retrieval_strategy: Mapping[str, Any]
    model_configuration: Mapping[str, Any]
    attempt: int
    conversation_context: ConversationContextSnapshot = field(
        default_factory=empty_context_snapshot
    )
    agent_configuration: Mapping[str, Any] = field(
        default_factory=lambda: {
            "version": "native_tool_calling_agent_v3",
            "budget": {"max_model_rounds": 8, "max_graph_calls": 2},
        }
    )

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
            self, "retrieval_strategy", _frozen_mapping(self.retrieval_strategy)
        )
        object.__setattr__(
            self, "model_configuration", _frozen_mapping(self.model_configuration)
        )
        object.__setattr__(
            self, "agent_configuration", _frozen_mapping(self.agent_configuration)
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
    visual_content: tuple[ChatModelVisualContent, ...] = ()
    tool_calls: tuple[ChatToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool", "evidence"}:
            raise ValueError("unsupported chat model message role")
        if not self.content and not (self.role == "assistant" and self.tool_calls):
            raise ValueError("chat model message content must not be empty")
        if self.visual_content and self.role not in {"user", "evidence"}:
            raise ValueError("visual content is allowed only on evidence messages")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("tool calls are allowed only on assistant messages")
        if self.tool_call_id is not None and (
            self.role != "tool" or not self.tool_call_id.strip()
        ):
            raise ValueError("tool call identifier is allowed only on tool results")
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError("tool result requires a tool call identifier")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("non-tool message cannot carry a tool call identifier")
        asset_ids = [item.asset_id for item in self.visual_content]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("chat model message visual assets must be unique")
        call_ids = [item.id for item in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("chat model tool call identifiers must be unique")


@dataclass(frozen=True, slots=True)
class ChatModelRequest:
    messages: tuple[ChatModelMessage, ...]
    max_output_tokens: int | None = None
    model_profile_revision_id: UUID | None = None
    thinking_enabled: bool | None = None
    tools: tuple[ChatToolDefinition, ...] = ()
    tool_choice: ChatToolChoice | str | None = None
    parallel_tool_calls: bool = False
    response_format: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("chat model request must contain messages")
        if (
            self.max_output_tokens is not None
            and (
                isinstance(self.max_output_tokens, bool)
                or not 1 <= self.max_output_tokens <= 8192
            )
        ):
            raise ValueError("chat model output token limit is invalid")
        if self.thinking_enabled is not None and not isinstance(
            self.thinking_enabled, bool
        ):
            raise ValueError("chat model thinking override is invalid")
        tool_names = [item.name for item in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("chat model tools must be unique")
        if self.tool_choice is not None:
            choice = str(self.tool_choice)
            if choice not in {item.value for item in ChatToolChoice} and choice not in tool_names:
                raise ValueError("chat model tool choice is invalid")
            if not self.tools and choice != ChatToolChoice.NONE.value:
                raise ValueError("chat model tool choice requires tools")
        if self.parallel_tool_calls:
            raise ValueError("parallel chat tool calls are not supported")
        if self.response_format is not None:
            if not isinstance(self.response_format, Mapping):
                raise ValueError("chat response format must be an object")
            object.__setattr__(self, "response_format", dict(self.response_format))


@dataclass(frozen=True, slots=True)
class ChatModelResponse:
    content: str
    model: str
    finish_reason: str | None
    provider_request_id: str | None
    usage: Mapping[str, int]
    tool_calls: tuple[ChatToolCall, ...] = ()

    def __post_init__(self) -> None:
        if (not self.content and not self.tool_calls) or not self.model:
            raise ValueError("chat model response content or tool calls and model are required")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.usage.values()
        ):
            raise ValueError("chat model usage values must be non-negative integers")
        object.__setattr__(self, "usage", MappingProxyType(dict(self.usage)))


class ChatPipelineExecutionError(RuntimeError):
    """Stable, content-safe failure at the Chat pipeline boundary."""

    def __init__(
        self,
        code: ErrorCode,
        *,
        phase: ChatPipelinePhase,
        diagnostic: Mapping[str, Any] | None = None,
        model_calls: tuple[ChatModelCallRecord, ...] = (),
        agent_trace: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(code.value)
        self.code = code
        self.phase = phase
        self.diagnostic = dict(diagnostic or {})
        self.model_calls = model_calls
        self.agent_trace = dict(agent_trace) if agent_trace is not None else None

    def retain_model_calls(
        self, calls: tuple[ChatModelCallRecord, ...]
    ) -> ChatPipelineExecutionError:
        """Retain completed provider calls without duplicating an existing prefix."""

        if len(calls) > len(self.model_calls):
            self.model_calls = calls
        return self

    def retain_agent_trace(
        self, trace: Mapping[str, Any]
    ) -> ChatPipelineExecutionError:
        if self.agent_trace is None:
            self.agent_trace = dict(trace)
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
