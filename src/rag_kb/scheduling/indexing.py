"""PostgreSQL-backed indexing scheduling with bounded recovery."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from rag_kb.domain import (
    ErrorCode,
    IndexingCommand,
    IndexingExecutionError,
    IndexingLease,
    ReconciliationResult,
)
from rag_kb.indexing import IndexingPipeline
from rag_kb.uow import UnitOfWork, UnitOfWorkFactory, UnitOfWorkPurpose, execute_in_transaction


Clock = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int
    base_delay_seconds: float
    max_delay_seconds: float

    def __post_init__(self) -> None:
        if (
            self.max_attempts <= 0
            or self.base_delay_seconds <= 0
            or self.max_delay_seconds < self.base_delay_seconds
        ):
            raise ValueError("retry limits must be positive and ordered")

    def retry_at(self, attempt: int, observed_at: datetime) -> datetime:
        if attempt <= 0:
            raise ValueError("attempt must be positive")
        delay = min(
            self.base_delay_seconds * (2 ** (attempt - 1)),
            self.max_delay_seconds,
        )
        return observed_at + timedelta(seconds=delay)


class IndexingJobScheduler:
    def __init__(
        self,
        unit_of_work: UnitOfWorkFactory,
        pipeline: IndexingPipeline,
        *,
        worker_id: str,
        concurrency: int,
        poll_interval_seconds: float,
        heartbeat_interval_seconds: float,
        stale_after_seconds: float,
        deadline_seconds: float,
        retry_policy: RetryPolicy,
        reconciliation_batch_size: int,
        clock: Clock | None = None,
    ) -> None:
        if (
            not worker_id
            or concurrency <= 0
            or poll_interval_seconds <= 0
            or heartbeat_interval_seconds <= 0
            or stale_after_seconds <= heartbeat_interval_seconds
            or deadline_seconds <= 0
            or reconciliation_batch_size <= 0
        ):
            raise ValueError("scheduler limits are invalid")
        self._unit_of_work = unit_of_work
        self._pipeline = pipeline
        self._worker_id = worker_id
        self._concurrency = concurrency
        self._poll_interval_seconds = poll_interval_seconds
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._stale_after_seconds = stale_after_seconds
        self._deadline_seconds = deadline_seconds
        self._retry = retry_policy
        self._reconciliation_batch_size = reconciliation_batch_size
        self._clock = clock or (lambda: datetime.now(UTC))

    async def oldest_claimable_at(self) -> datetime | None:
        observed_at = self._clock()
        return await execute_in_transaction(
            self._unit_of_work,
            lambda uow: uow.indexing.oldest_claimable_at(
                observed_at=observed_at,
                max_attempts=self._retry.max_attempts,
            ),
            purpose=UnitOfWorkPurpose.POLL,
        )

    async def execute(
        self,
        lease: IndexingLease,
        stopped: asyncio.Event,
    ) -> None:
        await self._execute(lease, stopped)

    async def run(self, stopped: asyncio.Event) -> None:
        active: set[asyncio.Task[None]] = set()
        try:
            while not stopped.is_set():
                finished = {task for task in active if task.done()}
                if finished:
                    await asyncio.gather(*finished, return_exceptions=True)
                    active -= finished
                try:
                    await self.reconcile_once()
                except Exception:
                    await _wait_for_activity(
                        stopped,
                        active,
                        timeout=self._poll_interval_seconds,
                    )
                    continue
                while len(active) < self._concurrency and not stopped.is_set():
                    try:
                        lease = await self.claim_once()
                    except Exception:
                        break
                    if lease is None:
                        break
                    active.add(asyncio.create_task(self._execute(lease, stopped)))
                await _wait_for_activity(
                    stopped,
                    active,
                    timeout=self._poll_interval_seconds,
                )
        finally:
            if active:
                _done, pending = await asyncio.wait(active, timeout=5)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*active, return_exceptions=True)

    async def claim_once(self) -> IndexingLease | None:
        observed_at = self._clock()
        return await execute_in_transaction(
            self._unit_of_work,
            lambda uow: uow.indexing.claim(
                worker_id=self._worker_id,
                observed_at=observed_at,
                max_attempts=self._retry.max_attempts,
            ),
            purpose=UnitOfWorkPurpose.CLAIM,
        )

    async def reconcile_once(self) -> ReconciliationResult:
        observed_at = self._clock()
        retry_schedule = tuple(
            self._retry.retry_at(attempt, observed_at)
            for attempt in range(1, self._retry.max_attempts + 1)
        )
        return await execute_in_transaction(
            self._unit_of_work,
            lambda uow: uow.indexing.reconcile_stale(
                stale_before=observed_at
                - timedelta(seconds=self._stale_after_seconds),
                observed_at=observed_at,
                max_attempts=self._retry.max_attempts,
                retry_at_by_attempt=retry_schedule,
                limit=self._reconciliation_batch_size,
            ),
            purpose=UnitOfWorkPurpose.RECONCILIATION,
        )

    async def _execute(
        self,
        lease: IndexingLease,
        stopped: asyncio.Event,
    ) -> None:
        pipeline_task = asyncio.create_task(
            self._pipeline.execute(
                IndexingCommand(
                    lease.job_id,
                    lease.indexed_document_version_id,
                )
            )
        )
        ownership_lost = asyncio.Event()
        heartbeat_task = asyncio.create_task(
            self._heartbeat(lease, ownership_lost)
        )
        stop_task = asyncio.create_task(stopped.wait())
        deadline_task = asyncio.create_task(asyncio.sleep(self._deadline_seconds))
        ownership_task = asyncio.create_task(ownership_lost.wait())
        try:
            done, _ = await asyncio.wait(
                (pipeline_task, stop_task, deadline_task, ownership_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if pipeline_task in done:
                try:
                    pipeline_task.result()
                except IndexingExecutionError as error:
                    await self._settle_error(lease, error)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    await self._settle(
                        lease,
                        code=ErrorCode.INDEX_PERSISTENCE_FAILED,
                        detail={"operation": "worker_execution"},
                        retryable=True,
                    )
                else:
                    await self._release_terminal(lease)
            elif stop_task in done:
                await _cancel(pipeline_task)
                await self._settle(
                    lease,
                    code=ErrorCode.INDEXING_WORKER_STOPPED,
                    detail={"operation": "worker_shutdown"},
                    retryable=True,
                )
            elif deadline_task in done:
                await _cancel(pipeline_task)
                await self._settle(
                    lease,
                    code=ErrorCode.INDEXING_DEADLINE_EXCEEDED,
                    detail={"limit": self._deadline_seconds},
                    retryable=True,
                )
            else:
                await _cancel(pipeline_task)
        except asyncio.CancelledError:
            await _cancel(pipeline_task)
            await self._settle(
                lease,
                code=ErrorCode.INDEXING_WORKER_STOPPED,
                detail={"operation": "worker_shutdown"},
                retryable=True,
            )
            raise
        finally:
            heartbeat_task.cancel()
            stop_task.cancel()
            deadline_task.cancel()
            ownership_task.cancel()
            await asyncio.gather(
                heartbeat_task,
                stop_task,
                deadline_task,
                ownership_task,
                return_exceptions=True,
            )

    async def _heartbeat(
        self,
        lease: IndexingLease,
        ownership_lost: asyncio.Event,
    ) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_interval_seconds)
            try:
                owned = await execute_in_transaction(
                    self._unit_of_work,
                    lambda uow: uow.indexing.heartbeat(
                        lease,
                        observed_at=self._clock(),
                    ),
                    purpose=UnitOfWorkPurpose.HEARTBEAT,
                )
            except Exception:
                continue
            if not owned:
                ownership_lost.set()
                return

    async def _settle_error(
        self,
        lease: IndexingLease,
        error: IndexingExecutionError,
    ) -> None:
        await self._settle(
            lease,
            code=error.code,
            detail=error.diagnostic,
            retryable=_is_retryable(error),
        )

    async def _settle(
        self,
        lease: IndexingLease,
        *,
        code: ErrorCode,
        detail: dict,
        retryable: bool,
    ) -> None:
        observed_at = self._clock()
        safe_detail = _safe_detail(detail, attempt=lease.attempt)

        async def persist(uow: UnitOfWork) -> bool:
            if retryable and lease.attempt < self._retry.max_attempts:
                return await uow.indexing.reschedule(
                    lease,
                    observed_at=observed_at,
                    next_attempt_at=self._retry.retry_at(
                        lease.attempt, observed_at
                    ),
                    error_code=code.value,
                    error_detail=safe_detail,
                )
            return await uow.indexing.fail_owned(
                lease,
                observed_at=observed_at,
                error_code=code.value,
                error_detail=safe_detail,
            )

        changed = await execute_in_transaction(
            self._unit_of_work,
            persist,
            purpose=UnitOfWorkPurpose.RECONCILIATION,
        )
        if not changed:
            await self._release_terminal(lease)

    async def _release_terminal(self, lease: IndexingLease) -> None:
        await execute_in_transaction(
            self._unit_of_work,
            lambda uow: uow.indexing.release_terminal(lease),
            purpose=UnitOfWorkPurpose.RECONCILIATION,
        )


def _is_retryable(error: IndexingExecutionError) -> bool:
    if error.code is ErrorCode.INDEX_PERSISTENCE_FAILED:
        return "operation" in error.diagnostic
    return error.code in {
        ErrorCode.PARSER_TIMEOUT,
        ErrorCode.PARSER_CRASHED,
        ErrorCode.PARSER_ISOLATION_FAILED,
        ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE,
        ErrorCode.INDEX_INCOMPLETE,
    }


def _safe_detail(detail: dict, *, attempt: int) -> dict:
    allowed = {"check", "operation", "http_status", "retryable", "limit"}
    return {
        **{key: value for key, value in detail.items() if key in allowed},
        "attempt": attempt,
    }


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


async def _cancel(task: asyncio.Task) -> None:
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)
