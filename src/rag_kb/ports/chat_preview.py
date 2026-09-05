"""Application-owned delivery contract for non-authoritative Chat previews."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from rag_kb.domain.chat_activity import ChatActivityEvent
from rag_kb.domain.chat_preview import (
    ChatProgressUpdate,
    ChatPreviewEvent,
)


class ChatPreviewSink(Protocol):
    @property
    def enabled(self) -> bool: ...

    def emit_activity(self, event: ChatActivityEvent) -> None: ...

    async def emit_progress(
        self,
        *,
        run_id: UUID,
        attempt: int,
        update: ChatProgressUpdate,
    ) -> None: ...


class ChatPreviewSubscription(Protocol):
    async def next_event(self) -> ChatPreviewEvent: ...

    def discard_pending(self) -> None: ...

    async def close(self) -> None: ...
