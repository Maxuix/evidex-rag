"""Immutable Session-scoped conversational context contracts."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


SESSION_CONTEXT_VERSION = "session_context_v1"
SESSION_CONTEXT_STRATEGY = "recent_completed_turns_v1"


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
