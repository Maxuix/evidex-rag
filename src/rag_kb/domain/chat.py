"""Framework-independent durable chat facts and P1A policy values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from rag_kb.domain.chat_workflow import ChatWorkflowMode, initial_chat_workflow


class AnswerStyle(StrEnum):
    CONCISE = "concise"
    SUMMARY = "summary"


class InsufficiencyPolicy(StrEnum):
    REFUSE = "refuse"
    PARTIAL_ANSWER = "partial_answer"


class GroundingPolicy(StrEnum):
    EVIDENCE_ONLY = "evidence_only"


class CitationGranularity(StrEnum):
    CLAIM_LEVEL = "claim_level"


class AnswerTask(StrEnum):
    ANSWER = "answer"


class AnswerPolicyVersion(StrEnum):
    P1 = "p1"


_SUPPORTED_P1_COMBINATIONS = frozenset(
    {
        (AnswerStyle.CONCISE, InsufficiencyPolicy.REFUSE),
        (AnswerStyle.CONCISE, InsufficiencyPolicy.PARTIAL_ANSWER),
        (AnswerStyle.SUMMARY, InsufficiencyPolicy.REFUSE),
        (AnswerStyle.SUMMARY, InsufficiencyPolicy.PARTIAL_ANSWER),
    }
)


class AnswerPolicyNotSupportedError(ValueError):
    """A requested or persisted answer policy is outside the frozen P1 surface."""


class ChatSessionBusyError(RuntimeError):
    """A Session already owns a queued or running ChatRun."""


@dataclass(frozen=True, slots=True)
class AnswerPolicyDefaults:
    answer_style: AnswerStyle = AnswerStyle.CONCISE
    insufficiency_policy: InsufficiencyPolicy = InsufficiencyPolicy.REFUSE

    def as_dict(self) -> dict[str, str]:
        return {
            "answer_style": self.answer_style.value,
            "insufficiency_policy": self.insufficiency_policy.value,
        }


def validate_p1_answer_policy_defaults(
    values: Mapping[str, object],
) -> AnswerPolicyDefaults:
    permitted_dimensions = {"answer_style", "insufficiency_policy"}
    if set(values) != permitted_dimensions:
        raise AnswerPolicyNotSupportedError(
            "answer policy defaults contain unsupported dimensions"
        )
    try:
        return AnswerPolicyDefaults(
            answer_style=AnswerStyle(values["answer_style"]),
            insufficiency_policy=InsufficiencyPolicy(
                values["insufficiency_policy"]
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AnswerPolicyNotSupportedError(
            "answer policy default is not supported"
        ) from error


@dataclass(frozen=True, slots=True)
class EffectiveAnswerPolicy:
    grounding_policy: GroundingPolicy
    answer_style: AnswerStyle
    insufficiency_policy: InsufficiencyPolicy
    citation_required: bool
    citation_granularity: CitationGranularity
    answer_task: AnswerTask
    policy_version: AnswerPolicyVersion

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "grounding_policy": self.grounding_policy.value,
            "answer_style": self.answer_style.value,
            "insufficiency_policy": self.insufficiency_policy.value,
            "citation_required": self.citation_required,
            "citation_granularity": self.citation_granularity.value,
            "answer_task": self.answer_task.value,
            "policy_version": self.policy_version.value,
        }


def resolve_p1_policy(
    *,
    requested_policy: Mapping[str, object] | None = None,
    knowledge_base_defaults: Mapping[str, object] | None = None,
) -> EffectiveAnswerPolicy:
    """Resolve the immutable P1 policy using server > request > KB precedence."""

    requested = dict(requested_policy or {})
    defaults = dict(
        AnswerPolicyDefaults().as_dict()
        if knowledge_base_defaults is None
        else knowledge_base_defaults
    )
    permitted_dimensions = {"answer_style", "insufficiency_policy"}
    if set(requested) - permitted_dimensions:
        raise AnswerPolicyNotSupportedError(
            "answer policy contains unsupported dimensions"
        )
    resolved_defaults = validate_p1_answer_policy_defaults(defaults)
    try:
        answer_style = AnswerStyle(
            requested.get("answer_style", resolved_defaults.answer_style)
        )
        insufficiency_policy = InsufficiencyPolicy(
            requested.get(
                "insufficiency_policy", resolved_defaults.insufficiency_policy
            )
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AnswerPolicyNotSupportedError(
            "answer policy value is not supported"
        ) from error

    if (answer_style, insufficiency_policy) not in _SUPPORTED_P1_COMBINATIONS:
        raise AnswerPolicyNotSupportedError(
            "answer policy combination is not supported"
        )

    return EffectiveAnswerPolicy(
        grounding_policy=GroundingPolicy.EVIDENCE_ONLY,
        answer_style=answer_style,
        insufficiency_policy=insufficiency_policy,
        citation_required=True,
        citation_granularity=CitationGranularity.CLAIM_LEVEL,
        answer_task=AnswerTask.ANSWER,
        policy_version=AnswerPolicyVersion.P1,
    )


@dataclass(frozen=True, slots=True)
class ChatSession:
    id: UUID
    workspace_id: UUID
    kb_id: UUID
    principal_id: str
    title: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ChatMessage:
    id: UUID
    session_id: UUID
    chat_run_id: UUID | None
    role: str
    assistant_status: str | None
    client_request_id: UUID | None
    content: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ChatCitation:
    ordinal: int
    index_chunk_id: UUID | None
    document_id: UUID
    document_version_id: UUID
    document_display_name: str
    document_original_filename: str
    quoted_text: str
    source_location: dict[str, Any]
    score: float | None
    modality: str = "text"
    asset_snapshot: dict[str, Any] | None = None
    matched_representations: tuple[str, ...] = ("text",)

    def __post_init__(self) -> None:
        if (
            self.ordinal < 0
            or not self.quoted_text
            or not self.document_display_name.strip()
            or not self.document_original_filename.strip()
        ):
            raise ValueError("chat citation snapshot is invalid")

    @property
    def asset(self) -> dict[str, Any] | None:
        return self.asset_snapshot


@dataclass(frozen=True, slots=True)
class ChatRun:
    id: UUID
    workspace_id: UUID
    kb_id: UUID
    session_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID
    index_revision_id: UUID
    status: str
    principal_id: str
    client_id: str
    endpoint: str
    idempotency_key: UUID
    request_hash: str
    requested_policy: dict[str, Any]
    effective_policy: dict[str, Any]
    retrieval_strategy: dict[str, Any]
    model_configuration: dict[str, Any]
    assistant_status: str
    assistant_content: str
    citations: tuple[ChatCitation, ...]
    attempt: int
    error_code: str | None
    error_detail: dict[str, Any] | None
    error_retryable: bool | None
    usage: dict[str, Any] | None
    timing: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    conversation_context: dict[str, Any]
    contextualized_query: dict[str, Any] | None = None
    final_llm_context: dict[str, Any] | None = None
    workflow_configuration: dict[str, Any] = field(
        default_factory=lambda: initial_chat_workflow(ChatWorkflowMode.SIMPLE)[
            0
        ].as_dict()
    )
    workflow_state: dict[str, Any] = field(
        default_factory=lambda: initial_chat_workflow(ChatWorkflowMode.SIMPLE)[
            1
        ].as_dict()
    )
