"""Deterministic weighted lane choice with starvation-preventing aging."""

from __future__ import annotations

from datetime import datetime

from rag_kb.domain import WorkLane


class WeightedLaneSelector:
    def __init__(
        self,
        *,
        chat_weight: int,
        indexing_weight: int,
        aging_seconds: float,
    ) -> None:
        if chat_weight <= 0 or indexing_weight <= 0 or aging_seconds <= 0:
            raise ValueError("lane weights and aging must be positive")
        self._cycle = (
            (WorkLane.CHAT,) * chat_weight
            + (WorkLane.INDEXING,) * indexing_weight
        )
        self._aging_seconds = aging_seconds
        self._offset = 0

    def choose(
        self,
        available: set[WorkLane],
        *,
        oldest_queued_at: dict[WorkLane, datetime | None],
        observed_at: datetime,
    ) -> WorkLane | None:
        if not available:
            return None
        aged = [
            lane
            for lane in available
            if oldest_queued_at.get(lane) is not None
            and (observed_at - oldest_queued_at[lane]).total_seconds()
            >= self._aging_seconds
        ]
        if aged:
            return min(aged, key=lambda lane: (oldest_queued_at[lane], lane.value))
        for _ in self._cycle:
            lane = self._cycle[self._offset]
            self._offset = (self._offset + 1) % len(self._cycle)
            if lane in available:
                return lane
        return min(available, key=lambda lane: lane.value)
