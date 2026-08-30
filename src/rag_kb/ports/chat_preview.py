"""Application-owned delivery contract for non-authoritative Chat previews."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from rag_kb.domain.chat_preview import (
    ChatProgressUpdate,
    ChatPreviewEvent,
)


@runtime_checkable
class ChatPreviewSink(Protocol):
    @property
    def enabled(self) -> bool: ...

    async def emit_progress(
        self,
        *,
        run_id: UUID,
        attempt: int,
        update: ChatProgressUpdate,
    ) -> None: ...


@runtime_checkable
class ChatPreviewSubscription(Protocol):
    async def next_event(self) -> ChatPreviewEvent: ...

    def discard_pending(self) -> None: ...

    async def close(self) -> None: ...
