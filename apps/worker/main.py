"""Foundation Worker process with startup validation and graceful shutdown."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable
from datetime import UTC, datetime
import logging
from pathlib import Path
import signal
import stat
from typing import Any

from rag_kb.config import load_settings
from rag_kb.observability import configure_logging, get_logger, log_event
from rag_kb.scheduling.worker import consume_lane, reconcile_lanes


LOGGER = get_logger("rag_kb.worker.runtime")
WORKER_RUNTIME_DIRECTORY = ".worker-runtime"
WORKER_HEARTBEAT_FILENAME = ".worker-heartbeat"
WORKER_HEARTBEAT_INTERVAL_SECONDS = 5.0
WORKER_HEARTBEAT_MAX_AGE_SECONDS = 30.0


class WorkerHeartbeatError(RuntimeError):
    """The serving Worker heartbeat is absent or outside its freshness bound."""


class WorkerBackgroundTaskError(RuntimeError):
    """A supervised Worker background task stopped before process shutdown."""


async def check_runtime() -> None:
    settings = load_settings()
    configure_logging(level=settings.observability.log_level)
    _require_fresh_heartbeat(_heartbeat_path(settings.file_store.root_path))
    from apps.worker.dependencies import build_worker_dependencies

    dependencies = build_worker_dependencies(settings)
    try:
        await dependencies.start()
        log_event(
            LOGGER,
            "runtime_check_passed",
            process="worker",
            database="ready",
            queue="ready",
        )
    finally:
        await dependencies.close()


async def serve() -> None:
    settings = load_settings()
    configure_logging(level=settings.observability.log_level)
    heartbeat_path = _heartbeat_path(settings.file_store.root_path)
    _remove_heartbeat(heartbeat_path)
    from apps.worker.dependencies import build_worker_dependencies

    dependencies = build_worker_dependencies(settings)
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for shutdown_signal in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(shutdown_signal, stopped.set)
    try:
        await dependencies.start()
        _publish_heartbeat(heartbeat_path)
        log_event(
            LOGGER,
            "process_ready",
            process="worker",
            component="foundation_runtime",
            database="ready",
            queue="ready",
        )
        background = {
            "file_reconciliation_janitor": _run_janitor(
                dependencies,
                stopped,
                settings.file_store.reconciliation_interval_seconds,
            ),
            "chat_consumer": consume_lane(
                "chat",
                dependencies.chat_scheduler,
                stopped,
                poll_interval_seconds=settings.job_poller.poll_interval_seconds,
            ),
            "indexing_consumer": consume_lane(
                "indexing",
                dependencies.indexing_scheduler,
                stopped,
                poll_interval_seconds=settings.job_poller.poll_interval_seconds,
            ),
            "stale_work_reconciler": reconcile_lanes(
                {
                    "chat": dependencies.chat_scheduler,
                    "indexing": dependencies.indexing_scheduler,
                },
                stopped,
            ),
            "worker_liveness_heartbeat": _run_liveness_heartbeat(
                heartbeat_path,
                stopped,
                interval=WORKER_HEARTBEAT_INTERVAL_SECONDS,
            ),
        }
        log_event(
            LOGGER,
            "worker_consumers_started",
            process="worker",
            queue_backend="postgresql",
            lane="chat,indexing",
        )
        await _supervise_background_tasks(background, stopped)
        log_event(
            LOGGER,
            "worker_consumers_stopped",
            process="worker",
            lane="chat,indexing",
        )
    finally:
        stopped.set()
        _remove_heartbeat(heartbeat_path, best_effort=True)
        await dependencies.close()
        log_event(LOGGER, "process_stopped", process="worker")
        for shutdown_signal in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(shutdown_signal)


async def _run_janitor(dependencies, stopped: asyncio.Event, interval: float) -> None:
    context = dependencies.auth_provider.get_context()
    while not stopped.is_set():
        try:
            result = await dependencies.reconciliation_service.run_once(context)
            log_event(
                LOGGER,
                "source_file_reconciliation_completed",
                process="worker",
                pending_activated=result.pending_activated,
                missing_compensated=result.missing_compensated,
                cleanup_completed=result.cleanup_completed,
                cleanup_failed=result.cleanup_failed,
                orphans_removed=result.orphans_removed,
            )
        except Exception as error:
            log_event(
                LOGGER,
                "source_file_reconciliation_failed",
                process="worker",
                error_type=type(error).__name__,
            )
        try:
            await asyncio.wait_for(stopped.wait(), timeout=interval)
        except TimeoutError:
            pass


async def _run_liveness_heartbeat(
    path: Path,
    stopped: asyncio.Event,
    *,
    interval: float,
) -> None:
    while not stopped.is_set():
        _publish_heartbeat(path)
        try:
            await asyncio.wait_for(stopped.wait(), timeout=interval)
        except TimeoutError:
            pass


async def _supervise_background_tasks(
    background: dict[str, Awaitable[None]],
    stopped: asyncio.Event,
) -> None:
    tasks = {
        component: asyncio.create_task(awaitable, name=component)
        for component, awaitable in background.items()
    }
    stop_task = asyncio.create_task(stopped.wait(), name="worker_stop_signal")
    try:
        done, _pending = await asyncio.wait(
            (*tasks.values(), stop_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if stop_task in done or stopped.is_set():
            results = await asyncio.gather(
                *tasks.values(),
                return_exceptions=True,
            )
            failure = _log_background_failures(tasks, results)
            if failure:
                raise WorkerBackgroundTaskError(
                    "worker background task failed during shutdown"
                )
            return

        finished = {
            component: task
            for component, task in tasks.items()
            if task in done
        }
        for component, task in finished.items():
            error_type = _task_error_type(task)
            log_event(
                LOGGER,
                (
                    "worker_background_task_failed"
                    if error_type is not None
                    else "worker_background_task_stopped"
                ),
                level=logging.ERROR,
                process="worker",
                component=component,
                error_type=error_type,
            )
        stopped.set()
        for task in tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        raise WorkerBackgroundTaskError(
            "worker background task stopped unexpectedly"
        )
    finally:
        stopped.set()
        for task in tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        stop_task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)


def _log_background_failures(
    tasks: dict[str, asyncio.Task[None]],
    results: list[Any],
) -> bool:
    failed = False
    for (component, _task), result in zip(
        tasks.items(),
        results,
        strict=True,
    ):
        if not isinstance(result, Exception):
            continue
        failed = True
        log_event(
            LOGGER,
            "worker_background_task_failed",
            level=logging.ERROR,
            process="worker",
            component=component,
            error_type=type(result).__name__,
        )
    return failed


def _task_error_type(task: asyncio.Task[None]) -> str | None:
    if task.cancelled():
        return "CancelledError"
    error = task.exception()
    return type(error).__name__ if error is not None else None


def _heartbeat_path(root: Path) -> Path:
    return root / WORKER_RUNTIME_DIRECTORY / WORKER_HEARTBEAT_FILENAME


def _publish_heartbeat(path: Path) -> None:
    path.touch(exist_ok=True)


def _require_fresh_heartbeat(
    path: Path,
    *,
    observed_at: datetime | None = None,
) -> None:
    try:
        status = path.lstat()
    except FileNotFoundError as error:
        raise WorkerHeartbeatError("worker heartbeat is missing") from error
    if not stat.S_ISREG(status.st_mode):
        raise WorkerHeartbeatError("worker heartbeat is not a regular file")
    now = observed_at or datetime.now(UTC)
    age_seconds = now.timestamp() - status.st_mtime
    if age_seconds > WORKER_HEARTBEAT_MAX_AGE_SECONDS:
        raise WorkerHeartbeatError("worker heartbeat is stale")


def _remove_heartbeat(path: Path, *, best_effort: bool = False) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        if not best_effort:
            raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "validate Worker heartbeat, storage, database, migration, "
            "pgvector, and queue access"
        ),
    )
    arguments = parser.parse_args()
    try:
        asyncio.run(check_runtime() if arguments.check else serve())
    except Exception as error:
        log_event(
            LOGGER,
            "process_failed",
            level=logging.ERROR,
            process="worker",
            error_type=type(error).__name__,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
