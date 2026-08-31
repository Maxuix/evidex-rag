"""PostgreSQL-backed indexing scheduling with bounded recovery."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import inspect
import logging
from typing import TYPE_CHECKING

from rag_kb.domain import (
    ErrorCode,
    IndexingCommand,
    IndexingExecutionError,
    IndexingLease,
    ReconciliationResult,
    GraphWorkItem,
)
from rag_kb.observability import get_logger, log_event, log_exception
from rag_kb.uow import UnitOfWork, UnitOfWorkFactory, execute_in_transaction

if TYPE_CHECKING:
    from rag_kb.indexing.pipeline import IndexingPipeline
    from rag_kb.graph.service import GraphExtractionWorker


Clock = Callable[[], datetime]
LOGGER = get_logger("rag_kb.scheduling.indexing")


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
        heartbeat_interval_seconds: float,
        stale_after_seconds: float,
        retry_policy: RetryPolicy,
        reconciliation_batch_size: int,
        graph_worker: GraphExtractionWorker | None = None,
        clock: Clock | None = None,
    ) -> None:
        if (
            not worker_id
            or heartbeat_interval_seconds <= 0
            or stale_after_seconds <= heartbeat_interval_seconds
            or reconciliation_batch_size <= 0
        ):
            raise ValueError("scheduler limits are invalid")
        self._unit_of_work = unit_of_work
        self._pipeline = pipeline
        self._worker_id = worker_id
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._stale_after_seconds = stale_after_seconds
        self._retry = retry_policy
        self._reconciliation_batch_size = reconciliation_batch_size
        self._graph_worker = graph_worker
        self._clock = clock or (lambda: datetime.now(UTC))

    async def execute(
        self,
        lease: IndexingLease | GraphWorkItem,
        stopped: asyncio.Event,
    ) -> None:
        if isinstance(lease, GraphWorkItem):
            if self._graph_worker is None:
                return
            await self._graph_worker.process_work_item(lease)
            return
        await self._execute(lease, stopped)

    async def claim_once(self) -> IndexingLease | GraphWorkItem | None:
        observed_at = self._clock()

        async def claim_or_graph(uow: UnitOfWork):
            lease = await uow.indexing.claim(
                worker_id=self._worker_id,
                observed_at=observed_at,
                max_attempts=self._retry.max_attempts,
            )
            if lease is not None or self._graph_worker is None:
                return lease, ()
            work = await _claim_graph_work_item(
                uow.graph,
                worker_id=self._worker_id,
                observed_at=observed_at,
            )
            return work, uow.graph.take_retired_graphiti_builds()

        claimed, retired = await execute_in_transaction(
            self._unit_of_work,
            claim_or_graph,
        )
        if retired and self._graph_worker is not None:
            await self._graph_worker.recycle_retired(retired)
        return claimed

    async def reconcile_once(self) -> ReconciliationResult:
        observed_at = self._clock()
        retry_schedule = tuple(
            self._retry.retry_at(attempt, observed_at)
            for attempt in range(1, self._retry.max_attempts + 1)
        )
        async def reconcile(uow: UnitOfWork) -> ReconciliationResult:
            result = await uow.indexing.reconcile_stale(
                stale_before=observed_at
                - timedelta(seconds=self._stale_after_seconds),
                observed_at=observed_at,
                max_attempts=self._retry.max_attempts,
                retry_at_by_attempt=retry_schedule,
                limit=self._reconciliation_batch_size,
            )
            reconcile_graph = getattr(uow.graph, "reconcile_graph_work_leases", None)
            if callable(reconcile_graph):
                recovered = await reconcile_graph(
                    observed_at=observed_at,
                    limit=self._reconciliation_batch_size,
                )
                if recovered:
                    log_event(
                        LOGGER,
                        "graphiti_work_lease_reconciled",
                        recovered_count=recovered,
                    )
            return result

        return await execute_in_transaction(
            self._unit_of_work,
            reconcile,
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
        ownership_task = asyncio.create_task(ownership_lost.wait())
        try:
            done, _ = await asyncio.wait(
                (pipeline_task, stop_task, ownership_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if pipeline_task in done:
                try:
                    pipeline_task.result()
                except IndexingExecutionError as error:
                    await self._settle_error(lease, error)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    log_exception(
                        LOGGER,
                        "indexing_attempt_crashed",
                        error,
                        lane="indexing",
                        job_id=lease.job_id,
                        indexed_document_version_id=(
                            lease.indexed_document_version_id
                        ),
                        attempt=lease.attempt,
                    )
                    await self._settle(
                        lease,
                        code=ErrorCode.INDEX_PERSISTENCE_FAILED,
                        detail={"operation": "worker_execution"},
                        retryable=True,
                        phase="worker_execution",
                    )
                else:
                    await self._release_terminal(lease)
                    log_event(
                        LOGGER,
                        "indexing_attempt_completed",
                        lane="indexing",
                        job_id=lease.job_id,
                        indexed_document_version_id=(
                            lease.indexed_document_version_id
                        ),
                        attempt=lease.attempt,
                        outcome="released",
                    )
            elif stop_task in done:
                await _cancel(pipeline_task)
                await self._settle(
                    lease,
                    code=ErrorCode.INDEXING_WORKER_STOPPED,
                    detail={"operation": "worker_shutdown"},
                    retryable=True,
                    phase="worker_shutdown",
                )
            else:
                await _cancel(pipeline_task)
                log_event(
                    LOGGER,
                    "indexing_attempt_ownership_lost",
                    level=logging.WARNING,
                    lane="indexing",
                    job_id=lease.job_id,
                    indexed_document_version_id=(
                        lease.indexed_document_version_id
                    ),
                    attempt=lease.attempt,
                )
        except asyncio.CancelledError:
            await _cancel(pipeline_task)
            await self._settle(
                lease,
                code=ErrorCode.INDEXING_WORKER_STOPPED,
                detail={"operation": "worker_shutdown"},
                retryable=True,
                phase="worker_shutdown",
            )
            raise
        finally:
            heartbeat_task.cancel()
            stop_task.cancel()
            ownership_task.cancel()
            await asyncio.gather(
                heartbeat_task,
                stop_task,
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
                )
            except Exception as error:
                log_exception(
                    LOGGER,
                    "indexing_heartbeat_failed",
                    error,
                    lane="indexing",
                    job_id=lease.job_id,
                    indexed_document_version_id=(
                        lease.indexed_document_version_id
                    ),
                    attempt=lease.attempt,
                )
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
            phase=error.phase.value,
        )

    async def _settle(
        self,
        lease: IndexingLease,
        *,
        code: ErrorCode,
        detail: dict,
        retryable: bool,
        phase: str,
    ) -> None:
        observed_at = self._clock()
        safe_detail = _safe_detail(detail, attempt=lease.attempt)
        will_retry = retryable and lease.attempt < self._retry.max_attempts

        async def persist(uow: UnitOfWork) -> bool:
            if will_retry:
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
        )
        if not changed:
            log_event(
                LOGGER,
                "indexing_attempt_settlement_skipped",
                level=logging.WARNING,
                lane="indexing",
                job_id=lease.job_id,
                indexed_document_version_id=(
                    lease.indexed_document_version_id
                ),
                attempt=lease.attempt,
                error_code=code.value,
                phase=phase,
                retryable=retryable,
                outcome="ownership_lost",
            )
            await self._release_terminal(lease)
            return
        log_event(
            LOGGER,
            (
                "indexing_attempt_rescheduled"
                if will_retry
                else "indexing_attempt_failed"
            ),
            level=logging.WARNING if will_retry else logging.ERROR,
            lane="indexing",
            job_id=lease.job_id,
            indexed_document_version_id=lease.indexed_document_version_id,
            attempt=lease.attempt,
            error_code=code.value,
            phase=phase,
            retryable=retryable,
            outcome="requeued" if will_retry else "terminal",
        )

    async def _release_terminal(self, lease: IndexingLease) -> None:
        await execute_in_transaction(
            self._unit_of_work,
            lambda uow: uow.indexing.release_terminal(lease),
        )


def _is_retryable(error: IndexingExecutionError) -> bool:
    if error.code is ErrorCode.INDEX_PERSISTENCE_FAILED:
        return "operation" in error.diagnostic
    return error.code in {
        ErrorCode.PARSER_CRASHED,
        ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE,
        ErrorCode.INDEX_INCOMPLETE,
    }


def _safe_detail(detail: dict, *, attempt: int) -> dict:
    allowed = {
        "check",
        "operation",
        "http_status",
        "retryable",
        "limit_name",
        "limit",
        "unit_count",
        "chunk_count",
        "analysis_batch_count",
        "page_from",
        "page_to",
    }
    return {
        **{key: value for key, value in detail.items() if key in allowed},
        "attempt": attempt,
    }


async def _cancel(task: asyncio.Task) -> None:
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _claim_graph_work_item(graph, *, worker_id: str, observed_at: datetime):
    """Pass lease identity while keeping lightweight scheduler fakes compatible."""

    method = graph.next_work_item
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "worker_id" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return await method(worker_id=worker_id, observed_at=observed_at)
    return await method()
