from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.local_runtime import LocalRuntimeError
from tools.smoke_local import main


class SmokeLocalTests(unittest.TestCase):
    def test_defaults_come_from_local_runtime_identity(self) -> None:
        runtime = SimpleNamespace(
            api_origin="http://127.0.0.1:18000",
            frontend_origin="http://127.0.0.1:13000",
        )
        responses = (
            {"status": "alive"},
            {"status": "ready"},
            {"status": "ok"},
            {
                "paths": {
                    "/api/v1/knowledge-bases": {},
                    "/api/v1/retrieval/query": {},
                    "/api/v1/chat/sessions": {},
                }
            },
        )
        with (
            patch("tools.smoke_local.resolve_local_runtime", return_value=runtime),
            patch("tools.smoke_local.read_json", side_effect=responses) as read,
            patch("tools.smoke_local.check_frontend") as frontend,
            patch("sys.argv", ["smoke-local"]),
        ):
            self.assertEqual(main(), 0)

        frontend.assert_called_once_with("http://127.0.0.1:13000")

        self.assertEqual(
            [call.args[0] for call in read.call_args_list],
            [
                "http://127.0.0.1:18000/health/live",
                "http://127.0.0.1:18000/health/ready",
                "http://127.0.0.1:13000/health",
                "http://127.0.0.1:18000/api/v1/openapi.json",
            ],
        )

    def test_invalid_manifest_fails_without_exception_detail(self) -> None:
        with patch(
            "tools.smoke_local.resolve_local_runtime",
            side_effect=LocalRuntimeError("contains-sensitive-context"),
        ):
            self.assertEqual(main(), 1)


if __name__ == "__main__":
    unittest.main()
