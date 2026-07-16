from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from tools.run_quality_security_regression import (
    EVIDENCE_INPUTS,
    MATRIX,
    safe_observations,
    scrubbed_environment,
)


class QualitySecurityRunnerTests(unittest.TestCase):
    def test_matrix_covers_every_frozen_regression_boundary(self) -> None:
        check_ids = {check_id for check_id, _ in MATRIX}

        self.assertEqual(len(check_ids), len(MATRIX))
        self.assertTrue(
            {
                "architecture_boundaries",
                "application_lock",
                "frontend_lock",
                "python_unit",
                "asgi_contract",
                "openapi_compatibility",
                "frontend_api_contract",
                "compose_contract",
                "golden_dataset",
                "lexical_evaluation",
                "retrieval_evaluation",
                "answer_security_evaluation",
                "frontend_tests",
                "frontend_typecheck",
                "frontend_build",
                "postgresql_pgvector_integration",
                "public_e2e",
                "compose_smoke",
            }
            <= check_ids
        )

    def test_environment_does_not_inherit_project_runtime_credentials(self) -> None:
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
            self.assertEqual(
                scrubbed_environment(),
                {"PATH": "/usr/bin", "PYTHONPATH": "src:."},
            )

    def test_evidence_binds_golden_security_and_integration_reports(self) -> None:
        self.assertIn(
            "evaluation/reports/quality-security-regression-synthetic-v1-v1.0.json",
            EVIDENCE_INPUTS,
        )
        self.assertIn("verification/e2e/s06-w02-report-v1.0.json", EVIDENCE_INPUTS)
        self.assertIn(
            "evaluation/golden/p1a-security-probes-v1.0.jsonl", EVIDENCE_INPUTS
        )

    def test_only_numeric_test_observations_are_retained(self) -> None:
        output = "secret body\nRan 162 tests in 2.5s\n"
        self.assertEqual(
            safe_observations("python_unit", output), {"test_count": 162}
        )
        frontend = "Test Files  6 passed\nTests  27 passed\nsecret answer"
        self.assertEqual(
            safe_observations("frontend_tests", frontend),
            {"test_file_count": 6, "test_count": 27},
        )


if __name__ == "__main__":
    unittest.main()
