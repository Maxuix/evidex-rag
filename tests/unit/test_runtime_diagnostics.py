from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import io
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from apps.api.main import server_address
from apps.worker.main import (
    WORKER_HEARTBEAT_MAX_AGE_SECONDS,
    WorkerBackgroundTaskError,
    WorkerHeartbeatError,
    _heartbeat_path,
    _publish_heartbeat,
    _remove_heartbeat,
    _require_fresh_heartbeat,
    _run_liveness_heartbeat,
    _supervise_background_tasks,
    check_runtime,
    main as worker_main,
)
from rag_kb.observability import ContentSafeJsonFormatter, log_event

from tests.unit.test_settings import build_settings


class RuntimeDiagnosticsTests(unittest.TestCase):
    def test_direct_api_binding_remains_loopback_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = build_settings(Path(directory))

        self.assertEqual(
            server_address(settings, container_listen=False),
            ("127.0.0.1", 8000),
        )
        self.assertEqual(
            server_address(settings, container_listen=True),
            ("0.0.0.0", 8000),
        )

    def test_content_safe_formatter_drops_message_and_exception_text(self) -> None:
        formatter = ContentSafeJsonFormatter()
        record = logging.LogRecord(
            "third.party",
            logging.ERROR,
            __file__,
            1,
            "password=must-not-leak",
            (),
            RuntimeError("body-must-not-leak"),
        )

        rendered = formatter.format(record)
        payload = json.loads(rendered)
        self.assertEqual(payload["event"], "external_log")
        self.assertNotIn("must-not-leak", rendered)

    def test_structured_events_accept_only_allowlisted_metadata(self) -> None:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(ContentSafeJsonFormatter())
        logger = logging.getLogger("tests.runtime.safe")
        logger.handlers[:] = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)

        log_event(
            logger,
            "request_completed",
            trace_id="trace-1",
            method="GET",
            path="/health/live",
            status_code=200,
            cleanup_completed=1,
            error_type="OSError",
        )
        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["trace_id"], "trace-1")
        self.assertEqual(payload["cleanup_completed"], 1)
        self.assertNotIn("message", payload)

        with self.assertRaises(ValueError):
            log_event(logger, "unsafe", request_body="must-not-leak")

    def test_worker_background_failure_returns_nonzero_process_status(self) -> None:
        with (
            patch("sys.argv", ["worker"]),
            patch(
                "apps.worker.main.serve",
                AsyncMock(
                    side_effect=WorkerBackgroundTaskError(
                        "scheduler stopped unexpectedly"
                    )
                ),
            ),
            patch("apps.worker.main.log_event") as logged,
        ):
            self.assertEqual(worker_main(), 1)

        self.assertEqual(logged.call_args.args[1], "process_failed")
        self.assertEqual(
            logged.call_args.kwargs["error_type"],
            "WorkerBackgroundTaskError",
        )


class WorkerRuntimeDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    async def test_heartbeat_gate_rejects_missing_and_stale_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _heartbeat_path(Path(directory))

            with self.assertRaises(WorkerHeartbeatError):
                _require_fresh_heartbeat(path)

            _publish_heartbeat(path)
            modified_at = datetime.fromtimestamp(path.stat().st_mtime, UTC)
            _require_fresh_heartbeat(path, observed_at=modified_at)
            with self.assertRaises(WorkerHeartbeatError):
                _require_fresh_heartbeat(
                    path,
                    observed_at=modified_at
                    + timedelta(
                        seconds=WORKER_HEARTBEAT_MAX_AGE_SECONDS + 1
                    ),
                )
            _remove_heartbeat(path)
            self.assertFalse(path.exists())

    async def test_liveness_loop_publishes_heartbeat_until_stopped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _heartbeat_path(Path(directory))
            stopped = asyncio.Event()
            task = asyncio.create_task(
                _run_liveness_heartbeat(path, stopped, interval=0.01)
            )
            for _ in range(10):
                if path.exists():
                    break
                await asyncio.sleep(0.01)
            else:
                self.fail("Worker liveness heartbeat was not published")

            stopped.set()
            await task

        self.assertTrue(task.done())

    async def test_fresh_heartbeat_preserves_dependency_readiness_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = build_settings(Path(directory))
            _publish_heartbeat(_heartbeat_path(settings.file_store.root_path))
            dependencies = Mock()
            dependencies.start = AsyncMock()
            dependencies.close = AsyncMock()

            with (
                patch("apps.worker.main.load_settings", return_value=settings),
                patch("apps.worker.main.configure_logging"),
                patch(
                    "apps.worker.dependencies.build_worker_dependencies",
                    return_value=dependencies,
                ) as build,
                patch("apps.worker.main.log_event"),
            ):
                await check_runtime()

        build.assert_called_once_with(settings)
        dependencies.start.assert_awaited_once()
        dependencies.close.assert_awaited_once()

    async def test_missing_heartbeat_skips_dependency_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = build_settings(Path(directory))
            with (
                patch("apps.worker.main.load_settings", return_value=settings),
                patch("apps.worker.main.configure_logging"),
                patch(
                    "apps.worker.dependencies.build_worker_dependencies"
                ) as build,
            ):
                with self.assertRaises(WorkerHeartbeatError):
                    await check_runtime()

        build.assert_not_called()

    async def test_scheduler_or_janitor_termination_fails_supervision(self) -> None:
        async def stop_early() -> None:
            return

        async def wait_until_cancelled(cancelled: asyncio.Event) -> None:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        for early_component in (
            "worker_scheduler",
            "file_reconciliation_janitor",
        ):
            with self.subTest(component=early_component):
                stopped = asyncio.Event()
                cancelled = {
                    component: asyncio.Event()
                    for component in (
                        "worker_scheduler",
                        "file_reconciliation_janitor",
                        "worker_liveness_heartbeat",
                    )
                    if component != early_component
                }
                background = {
                    component: (
                        stop_early()
                        if component == early_component
                        else wait_until_cancelled(cancelled[component])
                    )
                    for component in (
                        "worker_scheduler",
                        "file_reconciliation_janitor",
                        "worker_liveness_heartbeat",
                    )
                }

                with patch("apps.worker.main.log_event") as logged:
                    with self.assertRaises(WorkerBackgroundTaskError):
                        await _supervise_background_tasks(background, stopped)

                self.assertTrue(stopped.is_set())
                self.assertTrue(
                    all(event.is_set() for event in cancelled.values())
                )
                self.assertEqual(
                    logged.call_args.args[1],
                    "worker_background_task_stopped",
                )
                self.assertEqual(
                    logged.call_args.kwargs["component"],
                    early_component,
                )

    async def test_background_exception_log_omits_exception_message(self) -> None:
        async def fail() -> None:
            raise RuntimeError("worker-content-must-not-leak")

        async def wait_forever() -> None:
            await asyncio.Event().wait()

        with patch("apps.worker.main.log_event") as logged:
            with self.assertRaises(WorkerBackgroundTaskError):
                await _supervise_background_tasks(
                    {
                        "worker_scheduler": fail(),
                        "file_reconciliation_janitor": wait_forever(),
                        "worker_liveness_heartbeat": wait_forever(),
                    },
                    asyncio.Event(),
                )

        self.assertEqual(
            logged.call_args.args[1],
            "worker_background_task_failed",
        )
        self.assertEqual(logged.call_args.kwargs["error_type"], "RuntimeError")
        self.assertNotIn("worker-content-must-not-leak", repr(logged.mock_calls))

    async def test_stop_signal_allows_graceful_background_shutdown(self) -> None:
        stopped = asyncio.Event()

        async def wait_for_stop() -> None:
            await stopped.wait()

        async def request_stop() -> None:
            await asyncio.sleep(0)
            stopped.set()

        stop_request = asyncio.create_task(request_stop())
        with patch("apps.worker.main.log_event") as logged:
            await _supervise_background_tasks(
                {
                    "worker_scheduler": wait_for_stop(),
                    "file_reconciliation_janitor": wait_for_stop(),
                    "worker_liveness_heartbeat": wait_for_stop(),
                },
                stopped,
            )
        await stop_request

        logged.assert_not_called()


if __name__ == "__main__":
    unittest.main()
