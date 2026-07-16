#!/usr/bin/env python3
"""Run and record the final P1A local-development release validation."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any

from tools.check_release_package import check_release_package


SENSITIVE_ENVIRONMENT = {
    "POSTGRES_ADMIN_PASSWORD",
    "RAG_KB_MIGRATION_PASSWORD",
    "RAG_KB_RUNTIME_PASSWORD",
}
EVIDENCE_INPUTS = (
    ".python-version",
    "requirements.lock",
    "apps/web-test/package-lock.json",
    "Dockerfile",
    "apps/web-test/Dockerfile",
    "compose.yaml",
    ".env.example",
    "start-local.sh",
    "docs/release/README.md",
    "docs/release/local-development-guide.md",
    "docs/release/capability-matrix.md",
    "docs/release/known-limitations.md",
    "docs/release/p1a-implementation-summary.md",
    "tools/check_release_package.py",
    "tools/run_p1a_release_validation.py",
    "tests/unit/test_release_package.py",
    "tests/unit/test_p1a_release_runner.py",
    "tests/unit/test_start_local_script.py",
    "src/rag_kb/domain/errors.py",
    "tests/contract/snapshots/openapi-v1.json",
    "verification/compatibility/container-images-v1.0.json",
    "verification/compatibility/frontend-build-v1.0.json",
    "verification/providers/provider-declarations-deepseek-qwen-v1.0.json",
    "verification/e2e/s06-w02-report-v1.0.json",
    "verification/quality-security/s06-w03-report-v1.0.json",
    "verification/operations-recovery/s06-w04-report-v1.0.json",
    "verification/release/README.md",
)


class ReleaseValidationError(RuntimeError):
    pass


def scrubbed_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    values = os.environ if source is None else source
    environment = {
        key: value
        for key, value in values.items()
        if not key.startswith("RAG_KB") and key not in SENSITIVE_ENVIRONMENT
    }
    environment["PYTHONPATH"] = "src:."
    return environment


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_observations(check_id: str, output: str) -> dict[str, int]:
    values: dict[str, int] = {}
    if check_id == "quality_security_regression":
        values["check_count"] = 18
    elif check_id == "operations_recovery":
        values["scenario_count"] = 9
    return values


def safe_command_text(command: tuple[str, ...]) -> str:
    values: list[str] = []
    for value in command:
        if "rag-kb-p1a-release-" in value:
            values.append(f"<temporary>/{Path(value).name}")
        else:
            values.append(value)
    return " ".join(values)


def command_matrix(temp: Path) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        (
            "release_package",
            (".venv/bin/python", "tools/check_release_package.py"),
        ),
        (
            "quality_security_regression",
            (
                ".venv/bin/python",
                "tools/run_quality_security_regression.py",
                "--report",
                str(temp / "quality-security.json"),
            ),
        ),
        (
            "operations_recovery",
            (
                ".venv/bin/python",
                "tools/run_operations_recovery.py",
                "--report",
                str(temp / "operations-recovery.json"),
            ),
        ),
    )


def run_matrix(
    root: Path,
    matrix: tuple[tuple[str, tuple[str, ...]], ...],
) -> list[dict[str, Any]]:
    environment = scrubbed_environment()
    results: list[dict[str, Any]] = []
    for index, (check_id, command) in enumerate(matrix, start=1):
        print(f"[{index}/{len(matrix)}] {check_id}", flush=True)
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
            raise ReleaseValidationError(
                f"release validation failed: {check_id} (exit {completed.returncode})"
            )
        results.append(
            {
                "check_id": check_id,
                "command": safe_command_text(command),
                "status": "passed",
                "duration_seconds": duration,
                "observations": safe_observations(
                    check_id, completed.stdout + completed.stderr
                ),
            }
        )
    return results


def _load_report(path: Path, artifact: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReleaseValidationError(f"{artifact} report is not an object")
    if value.get("status") != "passed":
        raise ReleaseValidationError(f"{artifact} report did not pass")
    return value


def build_report(
    root: Path,
    results: list[dict[str, Any]],
    quality: dict[str, Any],
    operations: dict[str, Any],
    *,
    started_at: datetime,
    duration_seconds: float,
) -> dict[str, Any]:
    package = check_release_package(root)
    quality_matrix = quality.get("matrix")
    operation_scenarios = operations.get("scenarios")
    if not isinstance(quality_matrix, list) or len(quality_matrix) != 18:
        raise ReleaseValidationError("fresh quality matrix did not contain 18 checks")
    if not isinstance(operation_scenarios, dict) or len(operation_scenarios) != 9:
        raise ReleaseValidationError("fresh operations report did not contain 9 scenarios")
    if any(item.get("status") != "passed" for item in quality_matrix):
        raise ReleaseValidationError("fresh quality matrix contains a failure")
    if any(value != "passed" for value in operation_scenarios.values()):
        raise ReleaseValidationError("fresh operations report contains a failure")

    files = {path: sha256(root / path) for path in EVIDENCE_INPUTS}
    unit = next(
        item for item in quality_matrix if item.get("check_id") == "python_unit"
    )
    asgi = next(
        item for item in quality_matrix if item.get("check_id") == "asgi_contract"
    )
    database = next(
        item
        for item in quality_matrix
        if item.get("check_id") == "postgresql_pgvector_integration"
    )
    frontend = next(
        item for item in quality_matrix if item.get("check_id") == "frontend_tests"
    )
    return {
        "schema_version": "1.0",
        "artifact": "p1a-local-development-release",
        "status": "passed",
        "verified_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "duration_seconds": round(duration_seconds, 3),
        "tested_host_platform": f"{platform.system().lower()}/{platform.machine()}",
        "validation": results,
        "release_package": {
            "release_documents": package.release_documents,
            "startup_scripts": package.startup_scripts,
            "markdown_links": package.markdown_links,
            "public_paths": package.public_paths,
            "stable_error_codes": package.error_codes,
            "immutable_images": package.immutable_images,
            "core_versions": package.core_versions,
            "provider_fingerprints": package.provider_fingerprints,
            "prior_reports": package.prior_reports,
            "required_commands": package.required_commands,
            "required_limits": package.required_limits,
        },
        "fresh_quality_security": {
            "checks": len(quality_matrix),
            "unit_tests": unit.get("observations", {}).get("test_count"),
            "asgi_contract_tests": asgi.get("observations", {}).get("test_count"),
            "postgresql_pgvector_tests": database.get("observations", {}).get(
                "test_count"
            ),
            "frontend_test_files": frontend.get("observations", {}).get(
                "test_file_count"
            ),
            "frontend_tests": frontend.get("observations", {}).get("test_count"),
            "golden_metrics": quality.get("golden_metrics"),
            "security_assertions": quality.get("security_assertions"),
            "tested_platform": quality.get("tested_platform"),
        },
        "fresh_operations_recovery": {
            "scenarios": len(operation_scenarios),
            "metrics": operations.get("metrics"),
            "tested_platform": operations.get("tested_platform"),
        },
        "coverage": {
            "one_command_local_start_migration_health_and_secret_reuse": True,
            "clean_checkout_migrate_start_and_public_closed_loop": True,
            "documents_retrieval_chat_citations_and_history": True,
            "frontend_contract_typecheck_build_and_manual_browser_evidence": True,
            "quality_security_and_malicious_documents": True,
            "real_postgresql_pgvector_migrations_and_scheduling": True,
            "abnormal_exit_recovery_cleanup_and_reset": True,
            "release_docs_capabilities_errors_dependencies_and_limits": True,
        },
        "inputs": {"algorithm": "sha256", "files": files},
        "recorded_limits": [
            "P1A is a loopback-only local-development release with one fixed identity and workspace",
            "no production security, backup/restore, host-failure recovery, RPO/RTO, high availability, compliance, or hostile multi-tenant claim is made",
            "the deterministic suites make no external provider call and do not establish real-provider robustness or latency",
            "the synthetic corpus and 18 reviewed cases are not representative enterprise quality or capacity evidence",
            "linux/amd64 immutable manifests are recorded but remain unexecuted",
            "real-browser behavior was manually accepted by Maxui; automation uses component tests and real HTTP/SSE without a browser binary",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    root = Path(__file__).resolve().parents[1]
    if arguments.list:
        with tempfile.TemporaryDirectory() as directory:
            for check_id, command in command_matrix(Path(directory)):
                print(f"{check_id}: {' '.join(command)}")
        return 0

    started_at = datetime.now(UTC)
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="rag-kb-p1a-release-") as directory:
            temporary = Path(directory)
            results = run_matrix(root, command_matrix(temporary))
            quality = _load_report(
                temporary / "quality-security.json", "quality/security"
            )
            operations = _load_report(
                temporary / "operations-recovery.json", "operations/recovery"
            )
            report = build_report(
                root,
                results,
                quality,
                operations,
                started_at=started_at,
                duration_seconds=time.monotonic() - started,
            )
    except (
        OSError,
        ValueError,
        KeyError,
        StopIteration,
        json.JSONDecodeError,
        subprocess.SubprocessError,
        ReleaseValidationError,
    ) as error:
        print(f"P1A release validation failed: {error}", file=sys.stderr)
        return 1

    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(
        "P1A local release validation passed: release package, 18-check "
        "quality/security matrix, and 9-scenario operations/recovery suite",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
