#!/usr/bin/env python3
"""Run and record the complete Stage 06 quality/security regression matrix."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any


MATRIX: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "architecture_boundaries",
        (".venv/bin/python", "tools/check_architecture.py"),
    ),
    ("application_lock", (".venv/bin/python", "tools/check_application_lock.py")),
    ("frontend_lock", (".venv/bin/python", "tools/check_frontend_lock.py")),
    (
        "python_unit",
        (
            ".venv/bin/python",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests/unit",
            "-q",
        ),
    ),
    (
        "asgi_contract",
        (
            ".venv/bin/python",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests/contract",
            "-q",
        ),
    ),
    (
        "openapi_compatibility",
        (".venv/bin/python", "tools/check_openapi_compatibility.py"),
    ),
    (
        "frontend_api_contract",
        (".venv/bin/python", "tools/check_frontend_api_contract.py"),
    ),
    (
        "compose_contract",
        (".venv/bin/python", "tools/check_compose_contract.py"),
    ),
    (
        "golden_dataset",
        (
            ".venv/bin/python",
            "tools/validate_golden_dataset.py",
            "evaluation/golden/synthetic-v1-golden-v1.0.jsonl",
            "--corpus-manifest",
            "evaluation/corpus/synthetic-v1/manifest.json",
            "--dataset-manifest",
            "evaluation/golden/synthetic-v1-golden-manifest-v1.0.json",
            "--provider-declaration",
            "verification/providers/provider-declarations-deepseek-qwen-v1.0.json",
            "--evaluation-config",
            "evaluation/configs/p1a-quality-baseline-v1.0.json",
            "--report-schema",
            "evaluation/schemas/evaluation-report-v1.0.schema.json",
        ),
    ),
    (
        "lexical_evaluation",
        (".venv/bin/python", "tools/lexical_comparison.py", "--check"),
    ),
    (
        "retrieval_evaluation",
        (".venv/bin/python", "tools/retrieval_evaluation.py", "--check"),
    ),
    (
        "answer_security_evaluation",
        (".venv/bin/python", "tools/quality_security_evaluation.py", "--check"),
    ),
    ("frontend_tests", ("npm", "--prefix", "apps/web-test", "test")),
    (
        "frontend_typecheck",
        ("npm", "--prefix", "apps/web-test", "run", "typecheck"),
    ),
    ("frontend_build", ("npm", "--prefix", "apps/web-test", "run", "build")),
    (
        "postgresql_pgvector_integration",
        (".venv/bin/python", "tools/run_db_integration.py"),
    ),
    (
        "public_e2e",
        (".venv/bin/python", "tools/run_e2e_integration.py"),
    ),
    (
        "compose_smoke",
        (".venv/bin/python", "tools/run_compose_smoke.py"),
    ),
)

EVIDENCE_INPUTS = (
    "tools/run_quality_security_regression.py",
    "tools/quality_security_evaluation.py",
    "evaluation/configs/quality-security-regression-v1.0.json",
    "evaluation/configs/p1a-quality-baseline-v1.0.json",
    "evaluation/golden/synthetic-v1-golden-v1.0.jsonl",
    "evaluation/golden/synthetic-v1-golden-manifest-v1.0.json",
    "evaluation/golden/synthetic-v1-answer-responses-v1.0.jsonl",
    "evaluation/golden/p1a-security-probes-v1.0.jsonl",
    "evaluation/corpus/synthetic-v1/manifest.json",
    "src/rag_kb/adapters/parser/plain_text.py",
    "evaluation/reports/retrieval-evaluation-synthetic-v1-v1.0.json",
    "evaluation/reports/quality-security-regression-synthetic-v1-v1.0.json",
    "verification/e2e/s06-w02-report-v1.0.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the matrix without executing it",
    )
    return parser.parse_args()


def scrubbed_environment() -> dict[str, str]:
    exact = {
        "POSTGRES_ADMIN_PASSWORD",
        "RAG_KB_MIGRATION_PASSWORD",
        "RAG_KB_RUNTIME_PASSWORD",
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("RAG_KB") and key not in exact
    }
    environment["PYTHONPATH"] = "src:."
    return environment


def command_text(command: tuple[str, ...]) -> str:
    return " ".join(command)


def safe_observations(check_id: str, output: str) -> dict[str, int]:
    """Extract numeric test counts without retaining command output content."""

    observations: dict[str, int] = {}
    unittest_match = re.search(r"Ran (\d+) tests?", output)
    if unittest_match:
        observations["test_count"] = int(unittest_match.group(1))
    if check_id == "frontend_tests":
        file_match = re.search(r"Test Files\s+(\d+) passed", output)
        test_match = re.search(r"Tests\s+(\d+) passed", output)
        if file_match:
            observations["test_file_count"] = int(file_match.group(1))
        if test_match:
            observations["test_count"] = int(test_match.group(1))
    return observations


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_matrix(root: Path) -> list[dict[str, Any]]:
    environment = scrubbed_environment()
    results: list[dict[str, Any]] = []
    for index, (check_id, command) in enumerate(MATRIX, start=1):
        print(f"[{index}/{len(MATRIX)}] {check_id}", flush=True)
        started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
        )
        duration = round(time.monotonic() - started, 3)
        if completed.returncode != 0:
            raise RuntimeError(
                f"quality/security check failed: {check_id} (exit {completed.returncode})"
            )
        results.append(
            {
                "check_id": check_id,
                "command": command_text(command),
                "status": "passed",
                "duration_seconds": duration,
                "observations": safe_observations(
                    check_id, completed.stdout + completed.stderr
                ),
            }
        )
    return results


def report(
    root: Path,
    results: list[dict[str, Any]],
    *,
    started_at: datetime,
    duration_seconds: float,
) -> dict[str, Any]:
    files = {
        path: sha256(root / path)
        for path in EVIDENCE_INPUTS
    }
    quality = json.loads(
        (root / "evaluation/reports/quality-security-regression-synthetic-v1-v1.0.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "schema_version": "1.0",
        "artifact": "stage06-quality-security-regression",
        "status": "passed",
        "verified_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "duration_seconds": round(duration_seconds, 3),
        "tested_platform": f"{platform.system().lower()}/{platform.machine()}",
        "matrix": results,
        "coverage": {
            "unit_contract_and_static": True,
            "real_postgresql_pgvector_and_migrations": True,
            "real_database_scheduling_load": True,
            "frontend_test_typecheck_build": True,
            "golden_retrieval_answer_and_security": True,
            "public_http_sse_e2e": True,
            "clean_compose_smoke": True,
        },
        "security_assertions": {
            "prompt_authority": "passed",
            "mandatory_access_filters": "passed",
            "citation_authority": "passed",
            "credential_and_tool_authority": "passed",
            "safe_refusal_and_output": "passed",
            "content_safe_logs_errors_api_frontend": "passed",
            "runtime_role_no_ddl": "passed",
            "non_development_profiles_fail_closed": "passed",
        },
        "golden_metrics": quality["answer_metrics"],
        "golden_answer_pipeline_latency_ms": quality["latency"][
            "answer_pipeline_ms"
        ],
        "inputs": {"algorithm": "sha256", "files": files},
        "recorded_limits": [
            "the deterministic chat adapter proves application and reviewed-label regression, not real-provider robustness or answer quality",
            "the prior real embedding/retrieval report is reused by checksum; this matrix performs no external model call",
            "the synthetic single-workspace fixture is not a hostile multi-tenant, production-capacity, or compliance claim",
            "linux/amd64 remains unexecuted",
        ],
    }


def main() -> int:
    arguments = parse_args()
    root = Path(__file__).resolve().parents[1]
    if arguments.list:
        for check_id, command in MATRIX:
            print(f"{check_id}: {command_text(command)}")
        return 0
    started_at = datetime.now(UTC)
    started = time.monotonic()
    try:
        results = run_matrix(root)
        value = report(
            root,
            results,
            started_at=started_at,
            duration_seconds=time.monotonic() - started,
        )
        if arguments.report is not None:
            arguments.report.parent.mkdir(parents=True, exist_ok=True)
            arguments.report.write_text(
                json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(
            "Quality/security regression passed: full matrix, golden metrics, "
            "malicious-document boundaries, and clean integration checks",
            flush=True,
        )
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"Quality/security regression failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
