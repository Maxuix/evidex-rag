"""PostgreSQL-backed ChatRun execution, heartbeat, and stale recovery."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
import logging
from typing import TYPE_CHECKING

from rag_kb.domain import (
    ChatExecutionCommand,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatRunLease,
    ErrorCode,
    ReconciliationResult,
)
from rag_kb.observability import get_logger, log_event, log_exception
from rag_kb.scheduling.indexing import RetryPolicy

if TYPE_CHECKING:
    from rag_kb.answering.runner import NativeAgentRunner
    from rag_kb.services.chat_execution import ChatRunCoordinator
    from rag_kb.services.chat_terminal import ChatFailureSettlementService


Clock = Callable[[], datetime]
LOGGER = get_logger("rag_kb.scheduling.chat")


class ChatRunScheduler:
    """Execute one claimed ChatRun at a time through lease-owned services."""

    def __init__(
        self,
        coordinator: ChatRunCoordinator,
        runner: NativeAgentRunner,
        failure_settler: ChatFailureSettlementService,
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
                    log_event(
                        LOGGER,
                        "chat_attempt_failed",
                        level=logging.ERROR,
                        lane="chat",
                        run_id=lease.run_id,
                        attempt=lease.attempt,
                        error_code=error.code.value,
                        phase=error.phase.value,
                    )
                    await self._failure_settler.settle(lease, error)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    log_exception(
                        LOGGER,
                        "chat_attempt_crashed",
                        error,
                        lane="chat",
                        run_id=lease.run_id,
                        attempt=lease.attempt,
                        phase=ChatPipelinePhase.LOAD_CONTEXT.value,
                    )
                    await self._failure_settler.settle(
                        lease,
                        ChatPipelineExecutionError(
                            ErrorCode.CHAT_PIPELINE_STEP_FAILED,
                            phase=ChatPipelinePhase.LOAD_CONTEXT,
                            diagnostic={"check": "scheduler_execution"},
                        ),
                    )
                else:
                    log_event(
                        LOGGER,
                        "chat_attempt_completed",
                        lane="chat",
                        run_id=lease.run_id,
                        attempt=lease.attempt,
                        outcome="terminal",
                    )
            elif stop_task in done:
                await _cancel(pipeline_task)
                log_event(
                    LOGGER,
                    "chat_attempt_stopped",
                    lane="chat",
                    run_id=lease.run_id,
                    attempt=lease.attempt,
                )
                await self._settle_stopped(lease)
            else:
                await _cancel(pipeline_task)
                log_event(
                    LOGGER,
                    "chat_attempt_ownership_lost",
                    level=logging.WARNING,
                    lane="chat",
                    run_id=lease.run_id,
                    attempt=lease.attempt,
                )
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
                log_exception(
                    LOGGER,
                    "chat_heartbeat_failed",
                    error,
                    lane="chat",
                    run_id=lease.run_id,
                    attempt=lease.attempt,
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
