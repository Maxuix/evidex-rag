from __future__ import annotations

import io
import json
import logging
import tempfile
import unittest
from pathlib import Path

from apps.api.main import server_address
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
        )
        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["trace_id"], "trace-1")
        self.assertNotIn("message", payload)

        with self.assertRaises(ValueError):
            log_event(logger, "unsafe", request_body="must-not-leak")


if __name__ == "__main__":
    unittest.main()
