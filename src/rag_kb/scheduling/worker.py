"""Simple single-process consumers for chat and indexing work."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from time import perf_counter
from typing import TYPE_CHECKING, TypeAlias

from rag_kb.domain import ChatRunLease, GraphWorkItem, IndexingLease

from rag_kb.observability import (
    bind_log_context,
    get_logger,
    log_event,
    log_exception,
)

if TYPE_CHECKING:
    from rag_kb.scheduling.chat import ChatRunScheduler
    from rag_kb.scheduling.indexing import IndexingJobScheduler


DEFAULT_RECONCILIATION_INTERVAL_SECONDS = 30.0
LOGGER = get_logger("rag_kb.scheduling.worker")


LaneLease: TypeAlias = ChatRunLease | IndexingLease | GraphWorkItem


async def consume_lane(
    lane: str,
    scheduler: ChatRunScheduler | IndexingJobScheduler,
    stopped: asyncio.Event,
    *,
    poll_interval_seconds: float,
) -> None:
    """Claim and execute one job at a time for an independently reserved lane."""

    if not lane.strip() or poll_interval_seconds <= 0:
        raise ValueError("lane and poll interval must be valid")
    while not stopped.is_set():
        try:
            lease = await scheduler.claim_once()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            _log_failure("worker_claim_failed", lane, error)
            await _wait_or_stop(stopped, poll_interval_seconds)
            continue
        if lease is None:
            await _wait_or_stop(stopped, poll_interval_seconds)
            continue
        with bind_log_context(**_lease_context(lane, lease)):
            started = perf_counter()
            log_event(LOGGER, "worker_job_claimed")
            try:
                await scheduler.execute(lease, stopped)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                _log_failure("worker_execution_failed", lane, error)
            else:
                log_event(
                    LOGGER,
                    "worker_attempt_finished",
                    duration_ms=round((perf_counter() - started) * 1000, 3),
                    outcome="returned",
                )


async def reconcile_lanes(
    schedulers: Mapping[str, ChatRunScheduler | IndexingJobScheduler],
    stopped: asyncio.Event,
    *,
    interval_seconds: float = DEFAULT_RECONCILIATION_INTERVAL_SECONDS,
) -> None:
    """Recover stale work at a cadence independent from ordinary polling."""

    if not schedulers or any(not lane.strip() for lane in schedulers):
        raise ValueError("at least one named lane is required")
    if interval_seconds <= 0:
        raise ValueError("reconciliation interval must be positive")
    lanes = tuple(schedulers)
    while not stopped.is_set():
        results = await asyncio.gather(
            *(schedulers[lane].reconcile_once() for lane in lanes),
            return_exceptions=True,
        )
        for lane, result in zip(lanes, results, strict=True):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, Exception):
                _log_failure("worker_reconciliation_failed", lane, result)
                continue
            requeued = getattr(result, "requeued", 0)
            failed = getattr(result, "failed", 0)
            if requeued or failed:
                log_event(
                    LOGGER,
                    "worker_reconciliation_completed",
                    lane=lane,
                    requeued=requeued,
                    failed=failed,
                )
        await _wait_or_stop(stopped, interval_seconds)


async def _wait_or_stop(stopped: asyncio.Event, timeout: float) -> None:
    try:
        await asyncio.wait_for(stopped.wait(), timeout=timeout)
    except TimeoutError:
        pass


def _log_failure(event: str, lane: str, error: Exception) -> None:
    log_exception(
        LOGGER,
        event,
        error,
        lane=lane,
    )


def _lease_context(lane: str, lease: LaneLease) -> dict[str, object]:
    context: dict[str, object] = {"lane": lane}
    for field in (
        "attempt",
        "indexed_document_version_id",
        "job_id",
        "run_id",
        "workspace_id",
    ):
        value = getattr(lease, field, None)
        if value is not None:
            context[field] = value
    return context
