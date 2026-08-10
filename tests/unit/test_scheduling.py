from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import patch
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
)
from rag_kb.scheduling.chat import ChatRunScheduler
from rag_kb.scheduling.indexing import IndexingJobScheduler, RetryPolicy
from rag_kb.scheduling.worker import consume_lane, reconcile_lanes


NOW = datetime(2026, 7, 14, 9, 0, tzinfo=UTC)


class SchedulingPolicyTests(unittest.TestCase):
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

    async def test_heartbeat_exception_emits_content_safe_event(self) -> None:
        class FailingHeartbeatRepository(_Repository):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            async def heartbeat(self, lease, *, observed_at):
                del lease, observed_at
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("document-body-must-not-leak")
                return False

        repository = FailingHeartbeatRepository()
        factory = _Factory(repository)
        scheduler = _scheduler(factory, _Pipeline(factory), deadline=1)
        ownership_lost = asyncio.Event()

        with patch("rag_kb.scheduling.indexing.log_exception") as logged:
            await scheduler._heartbeat(  # noqa: SLF001
                _lease(attempt=1),
                ownership_lost,
            )

        self.assertTrue(ownership_lost.is_set())
        self.assertEqual(logged.call_args.args[1], "indexing_heartbeat_failed")
        self.assertEqual(logged.call_args.kwargs["lane"], "indexing")
        self.assertIsInstance(logged.call_args.args[2], RuntimeError)


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

    async def test_heartbeat_exception_emits_content_safe_event(self) -> None:
        class FailingHeartbeatCoordinator(_ChatCoordinator):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            async def heartbeat(self, lease, *, observed_at):
                del lease, observed_at
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("chat-content-must-not-leak")
                return False

        coordinator = FailingHeartbeatCoordinator()
        scheduler = _chat_scheduler(
            coordinator,
            _ChatPipeline(),
            _ChatSettler(),
        )
        ownership_lost = asyncio.Event()

        with patch("rag_kb.scheduling.chat.log_exception") as logged:
            await scheduler._heartbeat(  # noqa: SLF001
                _chat_lease(),
                ownership_lost,
            )

        self.assertTrue(ownership_lost.is_set())
        self.assertEqual(logged.call_args.args[1], "chat_heartbeat_failed")
        self.assertEqual(logged.call_args.kwargs["lane"], "chat")
        self.assertIsInstance(logged.call_args.args[2], RuntimeError)


class WorkerConsumerTests(unittest.IsolatedAsyncioTestCase):
    async def test_independent_consumers_start_both_lanes(self) -> None:
        stopped = asyncio.Event()
        release = asyncio.Event()
        chat = _Lane(["chat-1"], release)
        indexing = _Lane(["index-1"], release)
        tasks = (
            asyncio.create_task(
                consume_lane(
                    "chat",
                    chat,
                    stopped,
                    poll_interval_seconds=0.01,
                )
            ),
            asyncio.create_task(
                consume_lane(
                    "indexing",
                    indexing,
                    stopped,
                    poll_interval_seconds=0.01,
                )
            ),
        )
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    chat.claimed_event.wait(),
                    indexing.claimed_event.wait(),
                ),
                timeout=1,
            )
            self.assertEqual(chat.claimed, ["chat-1"])
            self.assertEqual(indexing.claimed, ["index-1"])
            self.assertEqual(chat.reconciliations, 0)
            self.assertEqual(indexing.reconciliations, 0)
        finally:
            stopped.set()
            release.set()
            await asyncio.gather(*tasks)

    async def test_reconciliation_exception_emits_content_safe_event(self) -> None:
        stopped = asyncio.Event()
        release = asyncio.Event()
        chat = _Lane([], release)
        chat.reconciliation_error = RuntimeError("query-must-not-leak")
        chat.stop_after_reconciliation = stopped

        with patch("rag_kb.scheduling.worker.log_exception") as logged:
            await reconcile_lanes(
                {"chat": chat, "indexing": _Lane([], release)},
                stopped,
                interval_seconds=0.01,
            )

        self.assertEqual(
            logged.call_args.args[1],
            "worker_reconciliation_failed",
        )
        self.assertEqual(logged.call_args.kwargs["lane"], "chat")
        self.assertIsInstance(logged.call_args.args[2], RuntimeError)

    async def test_claim_exception_emits_stable_event(self) -> None:
        stopped = asyncio.Event()
        release = asyncio.Event()
        claim_lane = _Lane(["chat-1"], release)
        claim_lane.claim_error = RuntimeError("claim-content-must-not-leak")
        claim_lane.stop_after_claim = stopped

        with patch("rag_kb.scheduling.worker.log_exception") as logged:
            await consume_lane(
                "chat",
                claim_lane,
                stopped,
                poll_interval_seconds=0.01,
            )

        events = [call.args[1] for call in logged.call_args_list]
        self.assertEqual(events, ["worker_claim_failed"])
        self.assertEqual(logged.call_args.kwargs["lane"], "chat")
        self.assertIsInstance(logged.call_args.args[2], RuntimeError)

    async def test_execution_exception_emits_stable_event(self) -> None:
        stopped = asyncio.Event()
        lane = _Lane(["index-1"], asyncio.Event())
        lane.execution_error = RuntimeError("pipeline-content-must-not-leak")
        lane.stop_after_execution = stopped

        with (
            patch("rag_kb.scheduling.worker.log_event") as events,
            patch("rag_kb.scheduling.worker.log_exception") as logged,
        ):
            await consume_lane(
                "indexing",
                lane,
                stopped,
                poll_interval_seconds=0.01,
            )

        self.assertEqual(
            logged.call_args.args[1],
            "worker_execution_failed",
        )
        self.assertEqual(logged.call_args.kwargs["lane"], "indexing")
        self.assertIsInstance(logged.call_args.args[2], RuntimeError)
        self.assertEqual(events.call_args_list[0].args[1], "worker_job_claimed")


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
    def __init__(self, leases, release) -> None:
        self.leases = list(leases)
        self.release = release
        self.claimed = []
        self.claimed_event = asyncio.Event()
        self.claim_error = None
        self.reconciliation_error = None
        self.stop_after_reconciliation = None
        self.stop_after_claim = None
        self.execution_error = None
        self.stop_after_execution = None
        self.reconciliations = 0

    async def claim_once(self):
        if self.stop_after_claim is not None:
            self.stop_after_claim.set()
        if self.claim_error is not None:
            raise self.claim_error
        if not self.leases:
            return None
        lease = self.leases.pop(0)
        self.claimed.append(lease)
        self.claimed_event.set()
        return lease

    async def reconcile_once(self):
        self.reconciliations += 1
        if self.stop_after_reconciliation is not None:
            self.stop_after_reconciliation.set()
        if self.reconciliation_error is not None:
            raise self.reconciliation_error
        return ReconciliationResult(0, 0)

    async def execute(self, lease, stopped):
        del lease, stopped
        if self.stop_after_execution is not None:
            self.stop_after_execution.set()
        if self.execution_error is not None:
            raise self.execution_error
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
