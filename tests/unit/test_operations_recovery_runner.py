from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import unittest

from tools.run_operations_recovery import (
    CONFIRMATION,
    EVIDENCE_INPUTS,
    OperationsRuntime,
    build_report,
    inherited_operations_environment,
    reset_invocation,
)


class OperationsRecoveryRunnerTests(unittest.TestCase):
    def test_environment_scrubs_runtime_inputs_and_credentials(self) -> None:
        inherited = inherited_operations_environment(
            {
                "PATH": "/usr/bin",
                "DOCKER_HOST": "unix:///test.sock",
                "RAG_KB_ENV_FILE": "secret.env",
                "RAG_KB__MODEL_PROVIDER__CHAT__API_KEY": "secret",
                "POSTGRES_ADMIN_PASSWORD": "admin-secret",
                "RAG_KB_MIGRATION_PASSWORD": "migration-secret",
                "RAG_KB_RUNTIME_PASSWORD": "runtime-secret",
            }
        )
        self.assertEqual(
            inherited,
            {"PATH": "/usr/bin", "DOCKER_HOST": "unix:///test.sock"},
        )

    def test_runtime_uses_unique_project_ports_and_random_role_passwords(self) -> None:
        root = Path(__file__).resolve().parents[2]
        first = OperationsRuntime(root)
        second = OperationsRuntime(root)
        self.assertNotEqual(first.project, second.project)
        self.assertEqual(len(set(first.password_values)), 3)
        self.assertTrue(set(first.password_values).isdisjoint(second.password_values))
        self.assertEqual(
            len(
                {
                    first.environment["RAG_KB_POSTGRES_PORT"],
                    first.environment["RAG_KB_API_PORT"],
                    first.environment["RAG_KB_FRONTEND_PORT"],
                }
            ),
            3,
        )
        self.assertIn("deploy/compose-operations.override.yaml", first.base)

    def test_reset_invocation_requires_the_explicit_project_and_token(self) -> None:
        command = reset_invocation(Path("/repo"), "disposable-project", CONFIRMATION)
        self.assertEqual(command[0], "/repo/.venv/bin/python")
        self.assertEqual(
            command[-4:],
            ["--project-name", "disposable-project", "--confirm", CONFIRMATION],
        )

    def test_report_contains_hashes_and_never_contains_password_values(self) -> None:
        root = Path(__file__).resolve().parents[2]
        runtime = OperationsRuntime(root)
        runtime.scenarios["example"] = "passed"
        runtime.metrics["count"] = 1
        report = build_report(
            runtime,
            started_at=datetime.now(UTC),
            duration_seconds=1.25,
        )
        rendered = json.dumps(report)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(set(report["inputs"]["files"]), set(EVIDENCE_INPUTS))
        for value in runtime.password_values:
            self.assertNotIn(value, rendered)
        self.assertIn("no host-failure recovery", rendered)

if __name__ == "__main__":
    unittest.main()
