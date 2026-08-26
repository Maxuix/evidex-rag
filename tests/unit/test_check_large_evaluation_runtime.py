from __future__ import annotations

import json
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools.check_large_evaluation_runtime import (
    LargeEvaluationRuntimeError,
    _assert_loopback_port,
    _assert_resolved_chat_model,
    _load_private_plan,
)
from tools.prepare_large_evaluation import build_plan
from tools.run_adaptive_graph_r4 import R4RunnerError


class CheckLargeEvaluationRuntimeTests(unittest.TestCase):
    def test_accepts_owner_only_plan_with_the_required_provider_contract(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            path.write_text(json.dumps(build_plan()), encoding="utf-8")
            path.chmod(0o600)
            plan = _load_private_plan(path)
            self.assertEqual(plan["plan_binding"]["provider_contract"]["chat_model"], "mimo-v2.5")

    def test_rejects_non_private_plan_and_provider_substitution(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            plan = build_plan()
            plan["plan_binding"]["provider_contract"]["chat_model"] = "fallback"
            path.write_text(json.dumps(plan), encoding="utf-8")
            path.chmod(0o644)
            with self.assertRaisesRegex(LargeEvaluationRuntimeError, "permissions"):
                _load_private_plan(path)
            path.chmod(0o600)
            with self.assertRaisesRegex(LargeEvaluationRuntimeError, "provider_contract"):
                _load_private_plan(path)
            plan["plan_binding"]["provider_contract"]["chat_model"] = "mimo-v2.5"
            plan["plan_binding_sha256"] = "0" * 64
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(LargeEvaluationRuntimeError, "binding_digest"):
                _load_private_plan(path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_names_an_unreachable_host_dependency(self) -> None:
        with patch("tools.check_large_evaluation_runtime.socket.create_connection", side_effect=OSError):
            with self.assertRaisesRegex(LargeEvaluationRuntimeError, "host_postgres_unreachable"):
                _assert_loopback_port(name="postgres", port=25432)

    def test_requires_the_bound_mimo_chat_model(self) -> None:
        _assert_resolved_chat_model({"resolved_model": "mimo-v2.5"})
        with self.assertRaisesRegex(LargeEvaluationRuntimeError, "chat_model_mismatch"):
            _assert_resolved_chat_model({"resolved_model": "fallback"})

    def test_r4_identity_error_has_stable_failure_code(self) -> None:
        self.assertEqual(
            str(R4RunnerError("r4_knowledge_base_not_found")),
            "r4_knowledge_base_not_found",
        )


if __name__ == "__main__":
    unittest.main()
