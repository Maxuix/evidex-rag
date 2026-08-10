from __future__ import annotations

from datetime import UTC, datetime, timedelta
import io
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from rag_kb.observability import ContentSafeJsonFormatter, log_event, log_exception
from tools.collect_diagnostics import (
    _parse_compose_json,
    collect_safe_events,
    main,
    sanitize_event,
)


class DiagnosticsBundleTests(unittest.TestCase):
    def test_compose_json_lines_are_parsed_without_labels_or_environment(self) -> None:
        rows = _parse_compose_json(
            '{"Name":"rag-api-1","Service":"api"}\n'
            '{"Name":"rag-worker-1","Service":"worker"}\n'
        )

        self.assertEqual([row["Service"] for row in rows], ["api", "worker"])

    def test_collection_keeps_safe_events_and_rejects_unknown_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "api.jsonl"
            stream = io.StringIO()
            handler = logging.StreamHandler(stream)
            handler.setFormatter(ContentSafeJsonFormatter(process="api"))
            logger = logging.getLogger("tests.diagnostics")
            logger.handlers[:] = [handler]
            logger.propagate = False
            logger.setLevel(logging.INFO)
            log_event(
                logger,
                "request_completed",
                trace_id="trace-1",
                method="GET",
                path="/api/v1/test",
                status_code=200,
            )
            safe_line = stream.getvalue()
            unsafe = {**json.loads(safe_line), "message": "secret-must-not-leak"}
            log_path.write_text(
                safe_line + json.dumps(unsafe) + "\nnot-json\n",
                encoding="utf-8",
            )

            events, summary = collect_safe_events(
                root,
                since=datetime.now(UTC) - timedelta(minutes=1),
                max_events=10,
            )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["trace_id"], "trace-1")
        self.assertEqual(summary["invalid_or_unsafe"], 2)
        self.assertNotIn("secret-must-not-leak", json.dumps(events))

    def test_sanitizer_accepts_safe_exception_shape_without_message(self) -> None:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(ContentSafeJsonFormatter(process="worker"))
        logger = logging.getLogger("tests.diagnostics.exception")
        logger.handlers[:] = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)
        try:
            raise RuntimeError("model-output-must-not-leak")
        except RuntimeError as error:
            log_exception(logger, "worker_failed", error, lane="chat")

        sanitized = sanitize_event(json.loads(stream.getvalue()))

        self.assertEqual(sanitized["exception"]["type"], "RuntimeError")
        self.assertNotIn("model-output-must-not-leak", json.dumps(sanitized))

    def test_cli_creates_private_bounded_archive_without_raw_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = root / "logs"
            output = root / "output"
            logs.mkdir()
            stream = io.StringIO()
            handler = logging.StreamHandler(stream)
            handler.setFormatter(ContentSafeJsonFormatter(process="api"))
            logger = logging.getLogger("tests.diagnostics.bundle")
            logger.handlers[:] = [handler]
            logger.propagate = False
            logger.setLevel(logging.INFO)
            log_event(logger, "process_ready")
            (logs / "api.jsonl").write_text(stream.getvalue(), encoding="utf-8")

            stdout = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "collect-diagnostics",
                        "--log-directory",
                        str(logs),
                        "--output-directory",
                        str(output),
                    ],
                ),
                patch(
                    "tools.collect_diagnostics.collect_docker_state",
                    return_value={"available": False},
                ),
                patch(
                    "tools.collect_diagnostics.collect_health",
                    return_value=[],
                ),
                patch(
                    "tools.collect_diagnostics.collect_git_state",
                    return_value={"available": True, "dirty": False},
                ),
                patch("sys.stdout", stdout),
            ):
                self.assertEqual(main(), 0)

            result = json.loads(stdout.getvalue())
            archive_path = Path(result["bundle"])
            self.assertEqual(archive_path.stat().st_mode & 0o777, 0o600)
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {
                        "README.txt",
                        "docker.json",
                        "events.jsonl",
                        "git.json",
                        "health.json",
                        "manifest.json",
                    },
                )
                combined = b"".join(archive.read(name) for name in archive.namelist())
            self.assertNotIn(b"API_KEY=", combined)
            self.assertNotIn(b"must-not-leak", combined)


if __name__ == "__main__":
    unittest.main()
