from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from rag_kb.domain import IndexingLease, IndexingResult, WorkLane
from rag_kb.scheduling import IndexingJobScheduler, RetryPolicy, WeightedLaneSelector


NOW = datetime(2026, 7, 14, 9, 0, tzinfo=UTC)


class SchedulingPolicyTests(unittest.TestCase):
    def test_weighted_selection_and_aging_preserve_both_lanes(self) -> None:
        selector = WeightedLaneSelector(
            chat_weight=3,
            indexing_weight=1,
            aging_seconds=30,
        )
        available = {WorkLane.CHAT, WorkLane.INDEXING}
        recent = {
            WorkLane.CHAT: NOW,
            WorkLane.INDEXING: NOW,
        }

        selected = tuple(
            selector.choose(
                available,
                oldest_queued_at=recent,
                observed_at=NOW,
            )
            for _ in range(8)
        )

        self.assertEqual(selected.count(WorkLane.CHAT), 6)
        self.assertEqual(selected.count(WorkLane.INDEXING), 2)
        aged = selector.choose(
            available,
            oldest_queued_at={
                WorkLane.CHAT: NOW,
                WorkLane.INDEXING: NOW - timedelta(seconds=31),
            },
            observed_at=NOW,
        )
        self.assertEqual(aged, WorkLane.INDEXING)

    def test_retry_policy_is_exponential_and_capped(self) -> None:
        policy = RetryPolicy(5, 2, 5)
        self.assertEqual(policy.retry_at(1, NOW), NOW + timedelta(seconds=2))
        self.assertEqual(policy.retry_at(2, NOW), NOW + timedelta(seconds=4))
        self.assertEqual(policy.retry_at(3, NOW), NOW + timedelta(seconds=5))


class IndexingSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_heartbeat_uses_independent_transactions(self) -> None:
        repository = _Repository()
        factory = _Factory(repository)
        pipeline = _Pipeline(factory, delay=0.04)
        scheduler = _scheduler(factory, pipeline, deadline=1)

        await scheduler._execute(_lease(attempt=1), asyncio.Event())  # noqa: SLF001

        self.assertGreaterEqual(repository.heartbeats, 1)
        self.assertEqual(repository.released, 1)
        self.assertFalse(factory.active)

    async def test_deadline_requeues_before_attempt_exhaustion(self) -> None:
        repository = _Repository()
        factory = _Factory(repository)
        scheduler = _scheduler(factory, _Pipeline(factory, never=True), deadline=0.01)

        await scheduler._execute(_lease(attempt=1), asyncio.Event())  # noqa: SLF001

        self.assertEqual(repository.rescheduled["error_code"], "INDEXING_DEADLINE_EXCEEDED")
        self.assertEqual(repository.failed, 0)

    async def test_attempt_exhaustion_is_terminal(self) -> None:
        repository = _Repository()
        factory = _Factory(repository)
        scheduler = _scheduler(factory, _Pipeline(factory, never=True), deadline=0.01)

        await scheduler._execute(_lease(attempt=2), asyncio.Event())  # noqa: SLF001

        self.assertEqual(repository.failed, 1)
        self.assertIsNone(repository.rescheduled)

    async def test_stop_requeues_owned_execution(self) -> None:
        repository = _Repository()
        factory = _Factory(repository)
        scheduler = _scheduler(factory, _Pipeline(factory, never=True), deadline=1)
        stopped = asyncio.Event()
        execution = asyncio.create_task(
            scheduler._execute(_lease(attempt=1), stopped)  # noqa: SLF001
        )
        await asyncio.sleep(0.01)

        stopped.set()
        await execution

        self.assertEqual(repository.rescheduled["error_code"], "INDEXING_WORKER_STOPPED")


class _Factory:
    def __init__(self, repository) -> None:
        self.repository = repository
        self.active = False

    def __call__(self, *, purpose, mode):
        del purpose, mode
        return _UnitOfWork(self)


class _UnitOfWork:
    def __init__(self, factory) -> None:
        self.factory = factory
        self.indexing = factory.repository

    async def __aenter__(self):
        if self.factory.active:
            raise AssertionError("transactions must not overlap")
        self.factory.active = True
        return self

    async def commit(self):
        return None

    async def __aexit__(self, *args):
        self.factory.active = False


class _Repository:
    def __init__(self) -> None:
        self.heartbeats = 0
        self.released = 0
        self.rescheduled = None
        self.failed = 0

    async def heartbeat(self, lease, *, observed_at):
        del lease, observed_at
        self.heartbeats += 1
        return True

    async def reschedule(self, lease, **values):
        del lease
        self.rescheduled = values
        return True

    async def fail_owned(self, lease, **values):
        del lease, values
        self.failed += 1
        return True

    async def release_terminal(self, lease):
        del lease
        self.released += 1
        return True


class _Pipeline:
    def __init__(self, factory, *, delay=0, never=False) -> None:
        self.factory = factory
        self.delay = delay
        self.never = never

    async def execute(self, command):
        del command
        if self.factory.active:
            raise AssertionError("pipeline ran inside a transaction")
        if self.never:
            await asyncio.Event().wait()
        await asyncio.sleep(self.delay)
        return IndexingResult(uuid4(), uuid4(), "ready", 1, serving_status="serving")


def _lease(*, attempt):
    return IndexingLease(uuid4(), uuid4(), "worker-a", attempt, NOW)


def _scheduler(factory, pipeline, *, deadline):
    return IndexingJobScheduler(
        factory,
        pipeline,
        worker_id="worker-a",
        concurrency=1,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.005,
        stale_after_seconds=1,
        deadline_seconds=deadline,
        retry_policy=RetryPolicy(2, 0.01, 0.02),
        reconciliation_batch_size=10,
        clock=lambda: NOW,
    )


if __name__ == "__main__":
    unittest.main()
