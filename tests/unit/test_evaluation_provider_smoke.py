from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_kb.domain import ChatModelExecutionError, ErrorCode
from rag_kb.services.model_settings import ModelProfileValidationError
from tools.run_evaluation_provider_smoke import (
    _load_checkpoint,
    _safe_failure_summary,
)


class EvaluationProviderSmokeTests(unittest.TestCase):
    def test_new_checkpoint_has_no_provider_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = _load_checkpoint(Path(directory) / "checkpoint.json")
        self.assertEqual(checkpoint["status"], "started")
        self.assertEqual(checkpoint["providers"], {})

    def test_chat_failure_summary_excludes_unknown_diagnostics(self) -> None:
        self.assertEqual(
            _safe_failure_summary(
                ChatModelExecutionError(
                    ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
                    diagnostic={"check": "transport", "untrusted": "ignored"},
                )
            ),
            {
                "type": "ChatModelExecutionError",
                "code": "CHAT_PROVIDER_UNAVAILABLE",
                "diagnostic": {"check": "transport"},
            },
        )

    def test_profile_validation_failure_summary_retains_only_code(self) -> None:
        self.assertEqual(
            _safe_failure_summary(
                ModelProfileValidationError("provider_validation_failed")
            ),
            {
                "type": "ModelProfileValidationError",
                "code": "provider_validation_failed",
            },
        )

    def test_smoke_failure_summary_retains_stable_code(self) -> None:
        from tools.run_evaluation_provider_smoke import ProviderSmokeError

        self.assertEqual(
            _safe_failure_summary(ProviderSmokeError("chat_no_tool_call")),
            {"type": "ProviderSmokeError", "code": "chat_no_tool_call"},
        )


if __name__ == "__main__":
    unittest.main()
