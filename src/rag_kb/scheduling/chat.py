"""PostgreSQL-backed ChatRun execution, heartbeat, and stale recovery."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
import logging
from typing import Any, Protocol

from rag_kb.domain import (
    ChatExecutionCommand,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatRunLease,
    ErrorCode,
    ReconciliationResult,
)
from rag_kb.observability import get_logger, log_event
from rag_kb.scheduling.indexing import RetryPolicy
from rag_kb.workflows.contracts import GraphRunner


Clock = Callable[[], datetime]
LOGGER = get_logger("rag_kb.scheduling.chat")


class ChatCoordinator(Protocol):
    async def oldest_claimable_at(self, **values: Any) -> datetime | None: ...

    async def claim(self, **values: Any) -> ChatRunLease | None: ...

    async def heartbeat(self, lease: ChatRunLease, **values: Any) -> bool: ...

    async def reconcile_stale(self, **values: Any) -> ReconciliationResult: ...


class FailureSettler(Protocol):
    async def settle(
        self, lease: ChatRunLease, error: ChatPipelineExecutionError
    ) -> Any: ...


class ChatRunScheduler:
    """Execute one claimed ChatRun at a time through lease-owned services."""

    def __init__(
        self,
        coordinator: ChatCoordinator,
        runner: GraphRunner,
        failure_settler: FailureSettler,
        *,
        worker_id: str,
        heartbeat_interval_seconds: float,
        stale_after_seconds: float,
        retry_policy: RetryPolicy,
        reconciliation_batch_size: int,
        clock: Clock | None = None,
    ) -> None:
        if (
            not worker_id.strip()
            or heartbeat_interval_seconds <= 0
            or stale_after_seconds <= heartbeat_interval_seconds
            or reconciliation_batch_size <= 0
        ):
            raise ValueError("chat scheduler limits are invalid")
        self._coordinator = coordinator
        self._runner = runner
        self._failure_settler = failure_settler
        self._worker_id = worker_id
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._stale_after_seconds = stale_after_seconds
        self._retry = retry_policy
        self._reconciliation_batch_size = reconciliation_batch_size
        self._clock = clock or (lambda: datetime.now(UTC))

    async def oldest_claimable_at(self) -> datetime | None:
        return await self._coordinator.oldest_claimable_at(
            observed_at=self._clock(),
            max_attempts=self._retry.max_attempts,
        )

    async def claim_once(self) -> ChatRunLease | None:
        return await self._coordinator.claim(
            worker_id=self._worker_id,
            observed_at=self._clock(),
            max_attempts=self._retry.max_attempts,
        )

    async def reconcile_once(self) -> ReconciliationResult:
        observed_at = self._clock()
        retry_schedule = tuple(
            self._retry.retry_at(attempt, observed_at)
            for attempt in range(1, self._retry.max_attempts + 1)
        )
        return await self._coordinator.reconcile_stale(
            stale_before=observed_at
            - timedelta(seconds=self._stale_after_seconds),
            observed_at=observed_at,
            max_attempts=self._retry.max_attempts,
            retry_at_by_attempt=retry_schedule,
            limit=self._reconciliation_batch_size,
        )

    async def execute(
        self,
        lease: ChatRunLease,
        stopped: asyncio.Event,
    ) -> None:
        pipeline_task = asyncio.create_task(
            self._runner.execute(ChatExecutionCommand(lease))
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
                except ChatPipelineExecutionError as error:
                    await self._failure_settler.settle(lease, error)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    await self._failure_settler.settle(
                        lease,
                        ChatPipelineExecutionError(
                            ErrorCode.CHAT_PIPELINE_STEP_FAILED,
                            phase=ChatPipelinePhase.LOAD_CONTEXT,
                            diagnostic={"check": "scheduler_execution"},
                        ),
                    )
            elif stop_task in done:
                await _cancel(pipeline_task)
                await self._settle_stopped(lease)
            else:
                await _cancel(pipeline_task)
        except asyncio.CancelledError:
            await _cancel(pipeline_task)
            await self._settle_stopped(lease)
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
        lease: ChatRunLease,
        ownership_lost: asyncio.Event,
    ) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_interval_seconds)
            try:
                owned = await self._coordinator.heartbeat(
                    lease,
                    observed_at=self._clock(),
                )
            except Exception as error:
                log_event(
                    LOGGER,
                    "chat_heartbeat_failed",
                    level=logging.ERROR,
                    lane="chat",
                    attempt=lease.attempt,
                    error_type=type(error).__name__,
                )
                continue
            if not owned:
                ownership_lost.set()
                return

    async def _settle_stopped(self, lease: ChatRunLease) -> None:
        await self._failure_settler.settle(
            lease,
            ChatPipelineExecutionError(
                ErrorCode.CHAT_WORKER_STOPPED,
                phase=ChatPipelinePhase.LOAD_CONTEXT,
                diagnostic={"operation": "worker_shutdown"},
            ),
        )


async def _cancel(task: asyncio.Task) -> None:
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)
