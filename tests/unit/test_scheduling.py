from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from rag_kb.domain import (
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatRunLease,
    ErrorCode,
    IndexingExecutionError,
    IndexingLease,
    IndexingPhase,
    IndexingResult,
    ReconciliationResult,
    WorkLane,
)
from rag_kb.scheduling import (
    ChatRunScheduler,
    FairWorkerScheduler,
    IndexingJobScheduler,
    RetryPolicy,
    WeightedLaneSelector,
)


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

    async def test_parser_wall_timeout_is_terminal_without_retry_cascade(self) -> None:
        repository = _Repository()
        factory = _Factory(repository)
        pipeline = _Pipeline(
            factory,
            error=IndexingExecutionError(
                ErrorCode.PARSER_RESOURCE_LIMIT,
                phase=IndexingPhase.PARSING,
                diagnostic={
                    "limit_name": "document_timeout",
                    "limit": 600,
                },
            ),
        )
        scheduler = _scheduler(factory, pipeline, deadline=900)

        await scheduler._execute(_lease(attempt=1), asyncio.Event())  # noqa: SLF001

        self.assertEqual(repository.failed, 1)
        self.assertIsNone(repository.rescheduled)


class ChatSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_heartbeat_and_pipeline_failure_are_settled(self) -> None:
        coordinator = _ChatCoordinator()
        settler = _ChatSettler()
        scheduler = _chat_scheduler(
            coordinator,
            _ChatPipeline(
                delay=0.03,
                error=ChatPipelineExecutionError(
                    ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
                    phase=ChatPipelinePhase.GENERATE_OR_REFUSE,
                ),
            ),
            settler,
        )

        await scheduler.execute(_chat_lease(), asyncio.Event())

        self.assertGreaterEqual(coordinator.heartbeats, 1)
        self.assertEqual(settler.errors[0].code, ErrorCode.CHAT_PROVIDER_UNAVAILABLE)

    async def test_stop_requeues_through_stable_failure(self) -> None:
        coordinator = _ChatCoordinator()
        settler = _ChatSettler()
        scheduler = _chat_scheduler(
            coordinator,
            _ChatPipeline(never=True),
            settler,
        )
        stopped = asyncio.Event()
        execution = asyncio.create_task(
            scheduler.execute(_chat_lease(), stopped)
        )
        await asyncio.sleep(0.01)

        stopped.set()
        await execution

        self.assertEqual(settler.errors[0].code, ErrorCode.CHAT_WORKER_STOPPED)

    async def test_reconciliation_uses_bounded_retry_schedule(self) -> None:
        coordinator = _ChatCoordinator()
        scheduler = _chat_scheduler(
            coordinator,
            _ChatPipeline(),
            _ChatSettler(),
        )

        result = await scheduler.reconcile_once()

        self.assertEqual(result, ReconciliationResult(1, 0))
        self.assertEqual(
            coordinator.reconciliation["retry_at_by_attempt"],
            (
                NOW + timedelta(seconds=0.01),
                NOW + timedelta(seconds=0.02),
            ),
        )


class FairWorkerSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_reserved_lane_capacity_starts_chat_under_indexing_load(self) -> None:
        release = asyncio.Event()
        chat = _Lane(WorkLane.CHAT, ["chat-1", "chat-2"], release)
        indexing = _Lane(
            WorkLane.INDEXING,
            ["index-1", "index-2", "index-3"],
            release,
        )
        scheduler = FairWorkerScheduler(
            chat,
            indexing,
            WeightedLaneSelector(
                chat_weight=3,
                indexing_weight=1,
                aging_seconds=30,
            ),
            chat_concurrency=1,
            indexing_concurrency=1,
            poll_interval_seconds=0.01,
            clock=lambda: NOW,
        )
        active = {}

        await scheduler._dispatch(active, asyncio.Event())  # noqa: SLF001
        await asyncio.sleep(0)

        self.assertEqual(set(active.values()), {WorkLane.CHAT, WorkLane.INDEXING})
        self.assertEqual(chat.claimed, ["chat-1"])
        self.assertEqual(indexing.claimed, ["index-1"])
        release.set()
        await asyncio.gather(*active)


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
    def __init__(self, factory, *, delay=0, never=False, error=None) -> None:
        self.factory = factory
        self.delay = delay
        self.never = never
        self.error = error

    async def execute(self, command):
        del command
        if self.factory.active:
            raise AssertionError("pipeline ran inside a transaction")
        if self.never:
            await asyncio.Event().wait()
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return IndexingResult(uuid4(), uuid4(), "ready", 1, serving_status="serving")


class _ChatCoordinator:
    def __init__(self) -> None:
        self.heartbeats = 0
        self.reconciliation = None

    async def heartbeat(self, lease, *, observed_at):
        del lease, observed_at
        self.heartbeats += 1
        return True

    async def reconcile_stale(self, **values):
        self.reconciliation = values
        return ReconciliationResult(1, 0)

    async def oldest_claimable_at(self, **values):
        del values
        return NOW

    async def claim(self, **values):
        del values
        return _chat_lease()


class _ChatPipeline:
    def __init__(self, *, delay=0, never=False, error=None) -> None:
        self.delay = delay
        self.never = never
        self.error = error

    async def execute(self, command):
        del command
        if self.never:
            await asyncio.Event().wait()
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error


class _ChatSettler:
    def __init__(self) -> None:
        self.errors = []

    async def settle(self, lease, error):
        del lease
        self.errors.append(error)


class _Lane:
    def __init__(self, lane, leases, release) -> None:
        self.lane = lane
        self.leases = list(leases)
        self.release = release
        self.claimed = []

    async def oldest_claimable_at(self):
        return NOW if self.leases else None

    async def claim_once(self):
        if not self.leases:
            return None
        lease = self.leases.pop(0)
        self.claimed.append(lease)
        return lease

    async def reconcile_once(self):
        return ReconciliationResult(0, 0)

    async def execute(self, lease, stopped):
        del lease, stopped
        await self.release.wait()


def _lease(*, attempt):
    return IndexingLease(uuid4(), uuid4(), "worker-a", attempt, NOW)


def _chat_lease():
    return ChatRunLease(uuid4(), uuid4(), "worker-a", 1, NOW)


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


def _chat_scheduler(coordinator, pipeline, settler):
    return ChatRunScheduler(
        coordinator,
        pipeline,
        settler,
        worker_id="worker-a",
        heartbeat_interval_seconds=0.005,
        stale_after_seconds=1,
        retry_policy=RetryPolicy(2, 0.01, 0.02),
        reconciliation_batch_size=10,
        clock=lambda: NOW,
    )


if __name__ == "__main__":
    unittest.main()
