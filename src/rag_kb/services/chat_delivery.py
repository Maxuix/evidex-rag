"""Bounded live delivery of already committed chat terminal facts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
import random
import time
from uuid import UUID

from rag_kb.domain import ChatPreviewEvent, ChatRun
from rag_kb.ports.chat_preview import ChatPreviewSubscription
from rag_kb.services.chat import ChatService


DisconnectCheck = Callable[[], Awaitable[bool]]
Sleep = Callable[[float], Awaitable[None]]


class ChatTerminalWatcher:
    """Poll through short ChatService reads and yield at most one terminal run."""

    def __init__(
        self,
        chat: ChatService,
        *,
        poll_interval_seconds: float,
        jitter_ratio: float,
        max_duration_seconds: float,
        sleep: Sleep = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        uniform: Callable[[float, float], float] = random.uniform,
    ) -> None:
        if poll_interval_seconds <= 0 or max_duration_seconds <= 0:
            raise ValueError("chat delivery timing must be positive")
        if not 0 <= jitter_ratio <= 0.5:
            raise ValueError("chat delivery jitter ratio is invalid")
        self._chat = chat
        self._poll_interval_seconds = poll_interval_seconds
        self._jitter_ratio = jitter_ratio
        self._max_duration_seconds = max_duration_seconds
        self._sleep = sleep
        self._monotonic = monotonic
        self._uniform = uniform

    async def watch(
        self,
        run_id: UUID,
        *,
        initial: ChatRun,
        disconnected: DisconnectCheck,
    ) -> AsyncIterator[ChatRun]:
        if initial.id != run_id:
            raise ValueError("initial ChatRun does not match requested run")
        started_at = self._monotonic()
        current = initial
        while True:
            if current.status in {"completed", "failed", "cancelled"}:
                yield current
                return
            if await disconnected():
                return
            remaining = self._max_duration_seconds - (
                self._monotonic() - started_at
            )
            if remaining <= 0:
                return
            jitter = self._uniform(-self._jitter_ratio, self._jitter_ratio)
            delay = min(remaining, self._poll_interval_seconds * (1 + jitter))
            await self._sleep(delay)
            if await disconnected():
                return
            if self._monotonic() - started_at >= self._max_duration_seconds:
                return
            current = await self._chat.get_run(run_id)


class ChatEventWatcher:
    """Merge ephemeral preview events with one authoritative terminal fact."""

    def __init__(self, terminal: ChatTerminalWatcher) -> None:
        self._terminal = terminal

    async def watch(
        self,
        run_id: UUID,
        *,
        initial: ChatRun,
        disconnected: DisconnectCheck,
        preview: ChatPreviewSubscription | None,
    ) -> AsyncIterator[ChatPreviewEvent | ChatRun]:
        terminal_iterator = self._terminal.watch(
            run_id,
            initial=initial,
            disconnected=disconnected,
        )
        terminal_task = asyncio.create_task(
            anext(terminal_iterator, None),
            name="chat-terminal-watch",
        )
        preview_task = (
            asyncio.create_task(
                preview.next_event(),
                name="chat-preview-watch",
            )
            if preview is not None
            else None
        )
        try:
            while True:
                pending = (
                    (terminal_task, preview_task)
                    if preview_task is not None
                    else (terminal_task,)
                )
                done, _ = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if terminal_task in done:
                    terminal = terminal_task.result()
                    if terminal is not None:
                        if preview is not None:
                            preview.discard_pending()
                        yield terminal
                    return
                assert preview_task is not None
                try:
                    event = preview_task.result()
                except Exception:
                    preview_task = None
                    continue
                yield event
                preview_task = asyncio.create_task(
                    preview.next_event(),  # type: ignore[union-attr]
                    name="chat-preview-watch",
                )
        finally:
            terminal_task.cancel()
            tasks = [terminal_task]
            if preview_task is not None:
                preview_task.cancel()
                tasks.append(preview_task)
            await asyncio.gather(*tasks, return_exceptions=True)
            await terminal_iterator.aclose()


class ChatSseConnectionLimiter:
    """Process-local connection limit for one ChatRun."""

    def __init__(self, max_connections_per_run: int) -> None:
        if max_connections_per_run < 1:
            raise ValueError("chat SSE connection limit must be positive")
        self._maximum = max_connections_per_run
        self._counts: dict[UUID, int] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, run_id: UUID) -> bool:
        async with self._lock:
            current = self._counts.get(run_id, 0)
            if current >= self._maximum:
                return False
            self._counts[run_id] = current + 1
            return True

    async def release(self, run_id: UUID) -> None:
        async with self._lock:
            current = self._counts.get(run_id)
            if current is None:
                raise RuntimeError("chat SSE connection was not acquired")
            if current == 1:
                del self._counts[run_id]
            else:
                self._counts[run_id] = current - 1

    async def active(self, run_id: UUID) -> int:
        async with self._lock:
            return self._counts.get(run_id, 0)


@dataclass(frozen=True, slots=True)
class ChatSseSubscription:
    run: ChatRun
    preview: ChatPreviewSubscription | None = None
