from __future__ import annotations

import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.run_e2e_integration import (
    PublicE2E,
    inherited_e2e_environment,
    parse_sse,
    report,
)
from datetime import UTC, datetime


class EndToEndRunnerTests(unittest.TestCase):
    def test_environment_does_not_inherit_runtime_inputs_or_passwords(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PATH": "/usr/bin",
                "RAG_KB_ENV_FILE": "private.env",
                "RAG_KB__MODEL_PROVIDER__CHAT__API_KEY": "secret",
                "POSTGRES_ADMIN_PASSWORD": "secret",
                "RAG_KB_MIGRATION_PASSWORD": "secret",
                "RAG_KB_RUNTIME_PASSWORD": "secret",
            },
            clear=True,
        ):
            self.assertEqual(inherited_e2e_environment(), {"PATH": "/usr/bin"})

    def test_sse_parser_accepts_named_json_events_only(self) -> None:
        body = (
            b": keepalive\r\n\r\n"
            b"event: answer.completed\r\n"
            b'data: {"run_id":"run-1","answer":"safe"}\r\n\r\n'
        )
        self.assertEqual(
            parse_sse(body),
            [("answer.completed", {"run_id": "run-1", "answer": "safe"})],
        )

    def test_report_contains_no_ephemeral_secret_or_identifier(self) -> None:
        runner = PublicE2E(Path("."))
        runner.mark("closed_loop")
        value = report(
            runner,
            started_at=datetime(2026, 7, 16, tzinfo=UTC),
            duration_seconds=12.5,
        )
        encoded = json.dumps(value)
        self.assertEqual(value["status"], "passed")
        self.assertEqual(value["scenarios"], {"closed_loop": "passed"})
        self.assertNotIn(runner.project, encoded)
        self.assertNotIn("password", encoded.lower())


if __name__ == "__main__":
    unittest.main()
