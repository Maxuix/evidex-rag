"""Framework-independent durable chat facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from rag_kb.domain.chat_agent import CHAT_AGENT_VERSION, ChatAgentBudget


class ChatSessionBusyError(RuntimeError):
    """A Session already owns a queued or running ChatRun."""


@dataclass(frozen=True, slots=True)
class ChatSession:
    id: UUID
    workspace_id: UUID
    kb_id: UUID
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
    agent_configuration: dict[str, Any] = field(
        default_factory=lambda: {
            "version": CHAT_AGENT_VERSION,
            "budget": ChatAgentBudget().as_dict(),
        }
    )
    agent_trace: dict[str, Any] | None = None
