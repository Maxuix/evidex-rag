"""Application-owned delivery contract for non-authoritative Chat previews."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from rag_kb.domain.chat_preview import ChatPreviewResetReason


@runtime_checkable
class ChatPreviewSink(Protocol):
    @property
    def enabled(self) -> bool: ...

    async def emit_delta(
        self,
        *,
        run_id: UUID,
        attempt: int,
        delta: str,
    ) -> None: ...

    async def emit_reset(
        self,
        *,
        run_id: UUID,
        attempt: int,
        reason: ChatPreviewResetReason,
    ) -> None: ...
