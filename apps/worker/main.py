"""Foundation Worker process with startup validation and graceful shutdown."""

from __future__ import annotations

import argparse
import asyncio
import signal

from apps.worker.dependencies import build_worker_dependencies
from rag_kb.config import load_settings
from rag_kb.observability import configure_logging, get_logger, log_event


LOGGER = get_logger("rag_kb.worker.runtime")


async def check_runtime() -> None:
    settings = load_settings()
    configure_logging(level=settings.observability.log_level)
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
    dependencies = build_worker_dependencies(settings)
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for shutdown_signal in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(shutdown_signal, stopped.set)
    try:
        await dependencies.start()
        log_event(
            LOGGER,
            "process_ready",
            process="worker",
            component="foundation_runtime",
            database="ready",
            queue="ready",
        )
        janitor = asyncio.create_task(
            _run_janitor(
                dependencies,
                stopped,
                settings.file_store.reconciliation_interval_seconds,
            )
        )
        scheduler = asyncio.create_task(dependencies.worker_scheduler.run(stopped))
        log_event(
            LOGGER,
            "worker_scheduler_started",
            process="worker",
            queue_backend="postgresql",
            lane="chat,indexing",
        )
        await stopped.wait()
        await asyncio.gather(janitor, scheduler)
        log_event(
            LOGGER,
            "worker_scheduler_stopped",
            process="worker",
            lane="chat,indexing",
        )
    finally:
        await dependencies.close()
        log_event(LOGGER, "process_stopped", process="worker")


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate storage, database, migration, pgvector, and queue access",
    )
    arguments = parser.parse_args()
    asyncio.run(check_runtime() if arguments.check else serve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
