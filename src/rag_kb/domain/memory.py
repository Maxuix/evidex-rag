"""Immutable Session-scoped conversational context contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from rag_kb.domain.answering import ChatModelCallRecord


SESSION_CONTEXT_VERSION = "session_context_v1"
SESSION_CONTEXT_STRATEGY = "recent_completed_turns_v1"
CONTEXTUAL_QUERY_VERSION = "contextual_query_v1"


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    user_message_id: UUID
    user_content: str
    assistant_message_id: UUID
    assistant_content: str

    def __post_init__(self) -> None:
        if not self.user_content.strip() or not self.assistant_content.strip():
            raise ValueError("conversation turn messages must not be empty")


@dataclass(frozen=True, slots=True)
class ConversationContextSnapshot:
    version: str
    strategy: str
    turns: tuple[ConversationTurn, ...]
    token_budget: int
    token_count: int
    candidate_turn_count: int
    truncated: bool
    content_hash: str

    def __post_init__(self) -> None:
        if self.version != SESSION_CONTEXT_VERSION:
            raise ValueError("unsupported conversation context version")
        if self.strategy != SESSION_CONTEXT_STRATEGY:
            raise ValueError("unsupported conversation context strategy")
        if self.token_budget != 4000 or not 0 <= self.token_count <= self.token_budget:
            raise ValueError("conversation context token bounds are invalid")
        if self.candidate_turn_count < len(self.turns):
            raise ValueError("conversation context candidate count is invalid")
        if len(self.turns) > 6:
            raise ValueError("conversation context contains too many turns")
        if self.truncated != (self.candidate_turn_count > len(self.turns)):
            raise ValueError("conversation context truncation fact is invalid")
        if not self.content_hash.startswith("sha256:") or len(self.content_hash) != 71:
            raise ValueError("conversation context hash is invalid")


def empty_context_snapshot() -> ConversationContextSnapshot:
    """Return the canonical migration-compatible empty Session snapshot."""

    return ConversationContextSnapshot(
        version=SESSION_CONTEXT_VERSION,
        strategy=SESSION_CONTEXT_STRATEGY,
        turns=(),
        token_budget=4000,
        token_count=0,
        candidate_turn_count=0,
        truncated=False,
        content_hash=(
            "sha256:45f76aa530878a50f94da86cc77ac1584f58c20c82988739d888b7bbb637652c"
        ),
    )


class QueryContextStatus(StrEnum):
    ORIGINAL = "original"
    CONTEXTUALIZED = "contextualized"
    NEEDS_CLARIFICATION = "needs_clarification"


@dataclass(frozen=True, slots=True)
class ContextualizedQuery:
    version: str
    status: QueryContextStatus
    original_query: str
    standalone_query: str | None
    context_hash: str
    model_calls: tuple[ChatModelCallRecord, ...] = ()
    created_at: datetime | None = None
    origin_attempt: int | None = None

    def __post_init__(self) -> None:
        if self.version != CONTEXTUAL_QUERY_VERSION:
            raise ValueError("unsupported contextual query version")
        if not self.original_query.strip() or len(self.original_query) > 32768:
            raise ValueError("original query is invalid")
        if not self.context_hash.startswith("sha256:") or len(self.context_hash) != 71:
            raise ValueError("contextual query hash is invalid")
        if self.status is QueryContextStatus.NEEDS_CLARIFICATION:
            if self.standalone_query is not None:
                raise ValueError("clarification query must not contain a standalone query")
        elif (
            self.standalone_query is None
            or not self.standalone_query.strip()
            or len(self.standalone_query) > 32768
        ):
            raise ValueError("standalone query is invalid")
        if self.status is QueryContextStatus.ORIGINAL:
            if self.standalone_query != self.original_query or self.model_calls:
                raise ValueError("original query artifact is inconsistent")
            if self.created_at is not None:
                raise ValueError("original query artifact must not have a creation time")
            if self.origin_attempt is not None:
                raise ValueError("original query artifact must not have an attempt")
        else:
            if (
                not self.model_calls
                or self.created_at is None
                or self.origin_attempt is None
                or isinstance(self.origin_attempt, bool)
                or self.origin_attempt < 1
            ):
                raise ValueError("model-produced query artifact is incomplete")
        if self.created_at is not None and (
            self.created_at.tzinfo is None or self.created_at.utcoffset() is None
        ):
            raise ValueError("contextual query creation time must be timezone-aware")

    def model_calls_for_attempt(
        self, attempt: int
    ) -> tuple[ChatModelCallRecord, ...]:
        return self.model_calls if self.origin_attempt == attempt else ()
