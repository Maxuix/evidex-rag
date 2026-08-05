"""Simple single-process consumers for chat and indexing work."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import logging
from typing import Any, Protocol

from rag_kb.observability import get_logger, log_event


DEFAULT_RECONCILIATION_INTERVAL_SECONDS = 30.0
LOGGER = get_logger("rag_kb.scheduling.worker")


class LaneScheduler(Protocol):
    async def claim_once(self) -> Any | None: ...

    async def reconcile_once(self) -> Any: ...

    async def execute(self, lease: Any, stopped: asyncio.Event) -> None: ...


async def consume_lane(
    lane: str,
    scheduler: LaneScheduler,
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
        try:
            await scheduler.execute(lease, stopped)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            _log_failure("worker_execution_failed", lane, error)


async def reconcile_lanes(
    schedulers: Mapping[str, LaneScheduler],
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
        await _wait_or_stop(stopped, interval_seconds)


async def _wait_or_stop(stopped: asyncio.Event, timeout: float) -> None:
    try:
        await asyncio.wait_for(stopped.wait(), timeout=timeout)
    except TimeoutError:
        pass


def _log_failure(event: str, lane: str, error: Exception) -> None:
    log_event(
        LOGGER,
        event,
        level=logging.ERROR,
        lane=lane,
        error_type=type(error).__name__,
    )
