"""Non-blocking activity recorder shared by live delivery and terminal history."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import replace
import time
from typing import Any
from uuid import UUID

from rag_kb.domain.chat_activity import (
    ACTIVE_STATUSES, MAX_ACTIVITY_BYTES, MAX_ACTIVITY_STEPS,
    ActivityStep, ChatActivityEvent, ChatActivitySnapshot, activity_json,
)


class ChatActivityRecorder:
    def __init__(self, run_id: UUID, attempt: int, *, emit: Callable[[ChatActivityEvent], None] | None = None, started_at: float | None = None) -> None:
        self.run_id = run_id
        self.attempt = attempt
        self._emit = emit
        self._started_at = time.monotonic() if started_at is None else started_at
        self._steps: OrderedDict[str, ActivityStep] = OrderedDict()
        self._total = 0
        self._seq = 0
        self._bytes = 0

    def elapsed(self) -> int:
        return max(0, round((time.monotonic() - self._started_at) * 1000))

    def begin(self, kind: str, name: str, *, round: int | None = None, pending: bool = False, **facts: Any) -> str:
        self._total += 1
        self._seq += 1
        step = ActivityStep(
            step_id=f"step_{self._total}", ordinal=self._total, seq=self._seq,
            kind=kind, name=name, status="pending" if pending else "running", round=round,
            started_offset_ms=None if pending else self.elapsed(), **facts,
        )
        self._save(step)
        return step.step_id

    def update(self, step_id: str, status: str, **facts: Any) -> None:
        old = self._steps.get(step_id)
        if old is None:
            return
        self._seq += 1
        started = old.started_offset_ms
        if started is None and status in {"running", "processing", "succeeded", "failed"}:
            started = self.elapsed()
        ended = self.elapsed() if status not in ACTIVE_STATUSES and started is not None else None
        self._save(replace(old, seq=self._seq, status=status, started_offset_ms=started, ended_offset_ms=ended, **facts))

    def terminate(self, status: str, *, code: str | None = None) -> None:
        for step in tuple(self._steps.values()):
            if step.status in ACTIVE_STATUSES:
                self.update(step.step_id, status, result_code=code)

    def snapshot(self, status: str = "running") -> ChatActivitySnapshot:
        return ChatActivitySnapshot(
            attempt=self.attempt, steps=tuple(self._steps.values()), total_steps=self._total,
            omitted_step_count=self._total - len(self._steps), status=status, elapsed_ms=self.elapsed(),
        )

    def _save(self, step: ActivityStep) -> None:
        previous = self._steps.pop(step.step_id, None)
        if previous is not None:
            self._bytes -= _size(previous)
        self._steps[step.step_id] = step
        # Preserve dispatch order even when parallel tools finish out of order.
        self._steps = OrderedDict(sorted(self._steps.items(), key=lambda item: item[1].ordinal))
        self._bytes += _size(step)
        if self._bytes > MAX_ACTIVITY_BYTES - 4096:
            for key, value in tuple(self._steps.items()):
                compact = replace(value, queries=(), refs=(), expression=None, sources=(), result_value=None, details_truncated=True)
                self._bytes += _size(compact) - _size(value)
                self._steps[key] = compact
                if self._bytes <= MAX_ACTIVITY_BYTES - 4096:
                    break
        while len(self._steps) > MAX_ACTIVITY_STEPS or self._bytes > MAX_ACTIVITY_BYTES - 4096:
            _, discarded = self._steps.popitem(last=False)
            self._bytes -= _size(discarded)
        if self._emit is not None:
            try:
                self._emit(ChatActivityEvent(self.run_id, self.attempt, step.seq, step, self.elapsed()))
            except Exception:
                # Delivery never changes the answer or the retained observation.
                pass


def _size(step: ActivityStep) -> int:
    return len(activity_json(step.as_dict()).encode("utf-8")) + 1
