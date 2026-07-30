"""Framework-independent values for best-effort Chat answer previews."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID


CHAT_PREVIEW_VERSION = "chat_preview_v1"


class ChatPreviewResetReason(StrEnum):
    GENERATION_FAILED = "generation_failed"
    VALIDATION_REPAIR = "validation_repair"
    PREVIEW_INVALID = "preview_invalid"


@dataclass(frozen=True, slots=True)
class ChatPreviewDelta:
    run_id: UUID
    attempt: int
    seq: int
    delta: str

    def __post_init__(self) -> None:
        if self.attempt < 1 or self.seq < 1:
            raise ValueError("chat preview attempt and sequence must be positive")
        if not self.delta:
            raise ValueError("chat preview delta must not be empty")


@dataclass(frozen=True, slots=True)
class ChatPreviewReset:
    run_id: UUID
    attempt: int
    seq: int
    reason: ChatPreviewResetReason

    def __post_init__(self) -> None:
        if self.attempt < 1 or self.seq < 1:
            raise ValueError("chat preview attempt and sequence must be positive")


ChatPreviewEvent = ChatPreviewDelta | ChatPreviewReset
