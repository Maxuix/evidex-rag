"""Bounded live delivery of already committed chat terminal facts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
import random
import time
from uuid import UUID

from rag_kb.auth import AuthContext
from rag_kb.domain import ChatRun
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
        context: AuthContext,
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
            current = await self._chat.get_run(context, run_id)


class ChatSseConnectionLimiter:
    """Process-local P1A limit for one principal and ChatRun pair."""

    def __init__(self, max_connections_per_principal_run: int) -> None:
        if max_connections_per_principal_run < 1:
            raise ValueError("chat SSE connection limit must be positive")
        self._maximum = max_connections_per_principal_run
        self._counts: dict[tuple[str, UUID], int] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, principal_id: str, run_id: UUID) -> bool:
        key = (principal_id, run_id)
        async with self._lock:
            current = self._counts.get(key, 0)
            if current >= self._maximum:
                return False
            self._counts[key] = current + 1
            return True

    async def release(self, principal_id: str, run_id: UUID) -> None:
        key = (principal_id, run_id)
        async with self._lock:
            current = self._counts.get(key)
            if current is None:
                raise RuntimeError("chat SSE connection was not acquired")
            if current == 1:
                del self._counts[key]
            else:
                self._counts[key] = current - 1

    async def active(self, principal_id: str, run_id: UUID) -> int:
        async with self._lock:
            return self._counts.get((principal_id, run_id), 0)


@dataclass(frozen=True, slots=True)
class ChatSseSubscription:
    context: AuthContext
    run: ChatRun
