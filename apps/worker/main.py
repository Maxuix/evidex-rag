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

from rag_kb.config import Settings, load_settings
from rag_kb.observability import (
    bind_log_context,
    configure_logging,
    get_logger,
    log_event,
    log_exception,
)
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


def _load_runtime_settings(env_file: Path | None) -> Settings:
    """Preserve the normal default while allowing an isolated env file."""

    if env_file is None:
        return load_settings()
    return load_settings(env_file=env_file)


async def check_runtime(*, env_file: Path | None = None) -> None:
    settings = _load_runtime_settings(env_file)
    configure_logging(
        level=settings.observability.log_level,
        process="worker-check",
        log_directory=settings.observability.log_directory,
    )
    _require_fresh_heartbeat(_heartbeat_path(settings.file_store.root_path))
    from apps.worker.dependencies import build_worker_dependencies

    dependencies = build_worker_dependencies(settings)
    try:
        await dependencies.start()
        log_event(
            LOGGER,
            "runtime_check_passed",
            level=logging.DEBUG,
            database="ready",
        )
    finally:
        await dependencies.close()


async def serve(*, env_file: Path | None = None) -> None:
    settings = _load_runtime_settings(env_file)
    configure_logging(
        level=settings.observability.log_level,
        process="worker",
        log_directory=settings.observability.log_directory,
    )
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
            component="foundation_runtime",
            database="ready",
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
            queue_backend="postgresql",
            lane="chat,indexing",
        )
        await _supervise_background_tasks(background, stopped)
        log_event(
            LOGGER,
            "worker_consumers_stopped",
            lane="chat,indexing",
        )
    finally:
        stopped.set()
        _remove_heartbeat(heartbeat_path, best_effort=True)
        await dependencies.close()
        log_event(LOGGER, "process_stopped")
        for shutdown_signal in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(shutdown_signal)


async def _run_janitor(dependencies, stopped: asyncio.Event, interval: float) -> None:
    context = dependencies.auth_provider.get_context()
    with bind_log_context(
        principal_id=context.principal_id,
        client_id=context.client_id,
        workspace_id=context.workspace_id,
    ):
        while not stopped.is_set():
            try:
                result = await dependencies.reconciliation_service.run_once(context)
                secret_result = await dependencies.model_secret_reconciliation_service.run_once(context)
                changed = any(
                    (
                        result.pending_activated,
                        result.pending_waiting,
                        result.pending_failed,
                        result.pending_conflicted,
                        secret_result.removed,
                        secret_result.failed,
                        result.missing_compensated,
                        result.cleanup_completed,
                        result.cleanup_failed,
                        result.orphans_removed,
                    )
                )
                log_event(
                    LOGGER,
                    "source_file_reconciliation_completed",
                    level=logging.INFO if changed else logging.DEBUG,
                    pending_activated=result.pending_activated,
                    pending_waiting=result.pending_waiting,
                    pending_failed=result.pending_failed,
                    pending_conflicted=result.pending_conflicted,
                    orphan_secrets_removed=secret_result.removed,
                    orphan_secrets_retained=secret_result.retained,
                    orphan_secrets_failed=secret_result.failed,
                    orphan_secrets_invalid=secret_result.invalid,
                    missing_compensated=result.missing_compensated,
                    cleanup_completed=result.cleanup_completed,
                    cleanup_failed=result.cleanup_failed,
                    orphans_removed=result.orphans_removed,
                )
            except Exception as error:
                log_exception(
                    LOGGER,
                    "source_file_reconciliation_failed",
                    error,
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
            error = _task_error(task)
            if error is not None:
                log_exception(
                    LOGGER,
                    "worker_background_task_failed",
                    error,
                    component=component,
                )
            else:
                log_event(
                    LOGGER,
                    "worker_background_task_stopped",
                    level=logging.ERROR,
                    component=component,
                    error_type=("CancelledError" if task.cancelled() else None),
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
        log_exception(
            LOGGER,
            "worker_background_task_failed",
            result,
            component=component,
        )
    return failed


def _task_error(task: asyncio.Task[None]) -> BaseException | None:
    if task.cancelled():
        return None
    return task.exception()


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
            "validate Worker heartbeat, storage, database, and migration"
        ),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help=(
            "load settings from this explicit environment file; useful for an "
            "isolated host runtime"
        ),
    )
    arguments = parser.parse_args()
    configure_logging(
        level="INFO",
        process="worker-check" if arguments.check else "worker",
    )
    try:
        asyncio.run(
            check_runtime(env_file=arguments.env_file)
            if arguments.check
            else serve(env_file=arguments.env_file)
        )
    except Exception as error:
        log_exception(
            LOGGER,
            "process_failed",
            error,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
