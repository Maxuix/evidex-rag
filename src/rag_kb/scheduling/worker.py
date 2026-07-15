"""Fair single-process dispatch across reserved chat and indexing lanes."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol

from rag_kb.domain import WorkLane
from rag_kb.scheduling.fairness import WeightedLaneSelector


Clock = Callable[[], datetime]


class LaneScheduler(Protocol):
    async def oldest_claimable_at(self) -> datetime | None: ...

    async def claim_once(self) -> Any | None: ...

    async def reconcile_once(self) -> Any: ...

    async def execute(self, lease: Any, stopped: asyncio.Event) -> None: ...


class FairWorkerScheduler:
    """Own lane capacity and dispatch durable work through weighted aging."""

    def __init__(
        self,
        chat: LaneScheduler,
        indexing: LaneScheduler,
        selector: WeightedLaneSelector,
        *,
        chat_concurrency: int,
        indexing_concurrency: int,
        poll_interval_seconds: float,
        clock: Clock | None = None,
    ) -> None:
        if (
            chat_concurrency <= 0
            or indexing_concurrency <= 0
            or poll_interval_seconds <= 0
        ):
            raise ValueError("worker lane limits are invalid")
        self._schedulers = {
            WorkLane.CHAT: chat,
            WorkLane.INDEXING: indexing,
        }
        self._selector = selector
        self._capacity = {
            WorkLane.CHAT: chat_concurrency,
            WorkLane.INDEXING: indexing_concurrency,
        }
        self._semaphores = {
            lane: asyncio.BoundedSemaphore(capacity)
            for lane, capacity in self._capacity.items()
        }
        self._poll_interval_seconds = poll_interval_seconds
        self._clock = clock or (lambda: datetime.now(UTC))

    async def run(self, stopped: asyncio.Event) -> None:
        active: dict[asyncio.Task[None], WorkLane] = {}
        try:
            while not stopped.is_set():
                await _reap(active)
                await asyncio.gather(
                    *(scheduler.reconcile_once() for scheduler in self._schedulers.values()),
                    return_exceptions=True,
                )
                await self._dispatch(active, stopped)
                await _wait_for_activity(
                    stopped,
                    set(active),
                    timeout=self._poll_interval_seconds,
                )
        finally:
            if active:
                _done, pending = await asyncio.wait(set(active), timeout=5)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*active, return_exceptions=True)

    async def _dispatch(
        self,
        active: dict[asyncio.Task[None], WorkLane],
        stopped: asyncio.Event,
    ) -> None:
        unavailable: set[WorkLane] = set()
        while not stopped.is_set():
            candidates = {
                lane
                for lane, capacity in self._capacity.items()
                if lane not in unavailable
                and sum(item == lane for item in active.values()) < capacity
            }
            if not candidates:
                return
            oldest_values = await asyncio.gather(
                *(self._schedulers[lane].oldest_claimable_at() for lane in candidates),
                return_exceptions=True,
            )
            oldest = dict(zip(candidates, oldest_values, strict=True))
            available = {
                lane
                for lane, value in oldest.items()
                if isinstance(value, datetime)
            }
            if not available:
                return
            observed_at = self._clock()
            lane = self._selector.choose(
                available,
                oldest_queued_at={
                    item: (
                        oldest[item]
                        if isinstance(oldest.get(item), datetime)
                        else None
                    )
                    for item in WorkLane
                },
                observed_at=observed_at,
            )
            if lane is None:
                return
            semaphore = self._semaphores[lane]
            await semaphore.acquire()
            try:
                lease = await self._schedulers[lane].claim_once()
            except Exception:
                semaphore.release()
                unavailable.add(lane)
                continue
            if lease is None:
                semaphore.release()
                unavailable.add(lane)
                continue
            task = asyncio.create_task(
                self._run_one(lane, lease, stopped, semaphore)
            )
            active[task] = lane

    async def _run_one(
        self,
        lane: WorkLane,
        lease: Any,
        stopped: asyncio.Event,
        semaphore: asyncio.BoundedSemaphore,
    ) -> None:
        try:
            await self._schedulers[lane].execute(lease, stopped)
        finally:
            semaphore.release()


async def _reap(active: dict[asyncio.Task[None], WorkLane]) -> None:
    finished = {task for task in active if task.done()}
    if not finished:
        return
    await asyncio.gather(*finished, return_exceptions=True)
    for task in finished:
        del active[task]


async def _wait_for_activity(
    stopped: asyncio.Event,
    active: set[asyncio.Task[None]],
    *,
    timeout: float,
) -> None:
    stop_task = asyncio.create_task(stopped.wait())
    try:
        await asyncio.wait(
            (*active, stop_task),
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        stop_task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)
