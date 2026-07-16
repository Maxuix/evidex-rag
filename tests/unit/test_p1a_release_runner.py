from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import unittest

from tools.run_p1a_release_validation import (
    EVIDENCE_INPUTS,
    build_report,
    command_matrix,
    safe_command_text,
    scrubbed_environment,
)


def _quality_report() -> dict:
    checks = [
        {"check_id": f"check_{index}", "status": "passed", "observations": {}}
        for index in range(18)
    ]
    checks[0] = {
        "check_id": "python_unit",
        "status": "passed",
        "observations": {"test_count": 176},
    }
    checks[1] = {
        "check_id": "asgi_contract",
        "status": "passed",
        "observations": {"test_count": 34},
    }
    checks[2] = {
        "check_id": "postgresql_pgvector_integration",
        "status": "passed",
        "observations": {"test_count": 61},
    }
    checks[3] = {
        "check_id": "frontend_tests",
        "status": "passed",
        "observations": {"test_file_count": 6, "test_count": 27},
    }
    return {
        "status": "passed",
        "matrix": checks,
        "golden_metrics": {"citation_identifier_validity": 1.0},
        "security_assertions": {"prompt_authority": "passed"},
        "tested_platform": "darwin/arm64",
    }


class P1AReleaseRunnerTests(unittest.TestCase):
    def test_environment_scrubs_project_inputs_and_credentials(self) -> None:
        value = scrubbed_environment(
            {
                "PATH": "/usr/bin",
                "RAG_KB_ENV_FILE": "secret.env",
                "RAG_KB__MODEL_PROVIDER__CHAT__API_KEY": "secret",
                "POSTGRES_ADMIN_PASSWORD": "secret",
                "RAG_KB_MIGRATION_PASSWORD": "secret",
                "RAG_KB_RUNTIME_PASSWORD": "secret",
            }
        )
        self.assertEqual(value, {"PATH": "/usr/bin", "PYTHONPATH": "src:."})

    def test_matrix_contains_only_the_three_release_boundaries(self) -> None:
        values = command_matrix(Path("/tmp/release"))
        self.assertEqual(
            tuple(item[0] for item in values),
            (
                "release_package",
                "quality_security_regression",
                "operations_recovery",
            ),
        )
        self.assertIn("quality-security.json", " ".join(values[1][1]))
        self.assertIn("operations-recovery.json", " ".join(values[2][1]))

    def test_command_text_removes_host_temporary_directory(self) -> None:
        value = safe_command_text(
            (
                ".venv/bin/python",
                "runner.py",
                "/private/tmp/rag-kb-p1a-release-random/quality-security.json",
            )
        )
        self.assertEqual(
            value,
            ".venv/bin/python runner.py <temporary>/quality-security.json",
        )

    def test_report_binds_inputs_counts_and_limits_without_secrets(self) -> None:
        root = Path(__file__).resolve().parents[2]
        operations = {
            "status": "passed",
            "scenarios": {f"scenario_{index}": "passed" for index in range(9)},
            "metrics": {"post_reset_knowledge_bases": 0},
            "tested_platform": "linux/arm64",
        }
        results = [
            {"check_id": name, "status": "passed"}
            for name in (
                "release_package",
                "quality_security_regression",
                "operations_recovery",
            )
        ]
        report = build_report(
            root,
            results,
            _quality_report(),
            operations,
            started_at=datetime.now(UTC),
            duration_seconds=1.0,
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["fresh_quality_security"]["checks"], 18)
        self.assertEqual(report["fresh_operations_recovery"]["scenarios"], 9)
        self.assertEqual(set(report["inputs"]["files"]), set(EVIDENCE_INPUTS))
        rendered = json.dumps(report)
        self.assertNotIn("api_key", rendered.lower())
        self.assertNotIn("password", rendered.lower())


if __name__ == "__main__":
    unittest.main()
