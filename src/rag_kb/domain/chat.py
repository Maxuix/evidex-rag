"""Framework-independent durable chat facts and P1A policy values."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID


class AnswerStyle(StrEnum):
    CONCISE = "concise"
    SUMMARY = "summary"


class InsufficiencyPolicy(StrEnum):
    REFUSE = "refuse"
    PARTIAL_ANSWER = "partial_answer"


@dataclass(frozen=True, slots=True)
class EffectiveAnswerPolicy:
    grounding_policy: str
    answer_style: AnswerStyle
    insufficiency_policy: InsufficiencyPolicy
    citation_required: bool
    citation_granularity: str
    answer_task: str
    policy_version: str

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "grounding_policy": self.grounding_policy,
            "answer_style": self.answer_style.value,
            "insufficiency_policy": self.insufficiency_policy.value,
            "citation_required": self.citation_required,
            "citation_granularity": self.citation_granularity,
            "answer_task": self.answer_task,
            "policy_version": self.policy_version,
        }


def resolve_p1_policy(
    *,
    answer_style: AnswerStyle | None,
    insufficiency_policy: InsufficiencyPolicy | None,
) -> EffectiveAnswerPolicy:
    """Resolve the narrow W01 policy handoff without external work.

    Knowledge-base-specific defaults and the complete precedence matrix are
    deliberately owned by S05-W02. W01 freezes the safe P1 defaults so every
    created run already carries a complete, non-weakenable effective policy.
    """

    return EffectiveAnswerPolicy(
        grounding_policy="evidence_only",
        answer_style=answer_style or AnswerStyle.CONCISE,
        insufficiency_policy=insufficiency_policy or InsufficiencyPolicy.REFUSE,
        citation_required=True,
        citation_granularity="claim_level",
        answer_task="answer",
        policy_version="p1",
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
    attempt: int
    error_code: str | None
    error_detail: dict[str, Any] | None
    error_retryable: bool | None
    usage: dict[str, Any] | None
    timing: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
