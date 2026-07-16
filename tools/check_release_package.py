#!/usr/bin/env python3
"""Validate the checked P1A local release documentation and evidence links."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Iterable

from rag_kb.domain.errors import ErrorCode


RELEASE_DOCUMENTS = (
    "docs/release/README.md",
    "docs/release/local-development-guide.md",
    "docs/release/capability-matrix.md",
    "docs/release/known-limitations.md",
    "docs/release/p1a-implementation-summary.md",
)
PRIOR_REPORTS = {
    "frontend": "verification/compatibility/frontend-build-v1.0.json",
    "e2e": "verification/e2e/s06-w02-report-v1.0.json",
    "quality_security": "verification/quality-security/s06-w03-report-v1.0.json",
    "operations_recovery": "verification/operations-recovery/s06-w04-report-v1.0.json",
}
REQUIRED_GUIDE_HEADINGS = (
    "Boundary and Prerequisites",
    "Create Local Configuration",
    "Configure Model Providers",
    "Migrate and Start",
    "Verify Health and Logs",
    "Import the Checked Synthetic Sample",
    "Demonstrate the Closed Loop",
    "Evaluation and Release Checks",
    "Stop, Maintain, and Reset",
    "Troubleshooting",
)
REQUIRED_COMMANDS = (
    "docker compose up -d --wait postgres storage-init",
    "docker compose --profile tools run --rm migrate",
    "docker compose up -d --wait api worker frontend",
    "curl --fail http://127.0.0.1:8000/health/live",
    "curl --fail http://127.0.0.1:8000/health/ready",
    "docker compose --profile tools run --rm maintenance",
    "DESTROY_RAG_KB_LOCAL_DATA",
    "tools/run_e2e_integration.py",
    "tools/run_p1a_release_validation.py",
)
REQUIRED_LIMITS = (
    "no host-failure recovery",
    "RPO",
    "RTO",
    "high availability",
    "production",
    "linux/amd64",
    "fixed development",
    "single Worker",
    "exact vector",
)
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"Authorization:\s*Bearer\s+\S+", re.IGNORECASE),
    re.compile(r"postgresql(?:\+asyncpg)?://[^:\s]+:(?!replace|choose)[^@\s]+@"),
)
CORE_VERSIONS = (
    "CPython `3.12.13`",
    "FastAPI `0.135.4`",
    "Uvicorn `0.51.0`",
    "Pydantic `2.13.4`",
    "SQLAlchemy `2.0.51`",
    "Alembic `1.18.5`",
    "asyncpg `0.31.0`",
    "pgvector Python `0.4.2`",
)


class ReleasePackageError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ReleaseCheckResult:
    release_documents: int
    markdown_links: int
    public_paths: int
    error_codes: int
    immutable_images: int
    core_versions: int
    provider_fingerprints: int
    prior_reports: int
    required_commands: int
    required_limits: int


def markdown_headings(text: str) -> set[str]:
    return {
        re.sub(r"^\d+\.\s+", "", match.group(1).strip())
        for line in text.splitlines()
        if (match := re.match(r"^#{1,6}\s+(.+?)\s*$", line))
    }


def markdown_links(path: Path, text: str) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for raw in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        target = raw.strip().strip("<>").split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved.append((path.parent / target).resolve())
    return tuple(resolved)


def validate_links(root: Path, relative_paths: Iterable[str]) -> int:
    count = 0
    for relative in relative_paths:
        path = root / relative
        for target in markdown_links(path, path.read_text(encoding="utf-8")):
            count += 1
            if not target.exists():
                raise ReleasePackageError(
                    f"broken release link in {relative}: {target}"
                )
    return count


def validate_no_secrets(text: str) -> None:
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise ReleasePackageError(
                f"release documentation matched secret pattern {pattern.pattern!r}"
            )


def _require_all(text: str, values: Iterable[str], label: str) -> int:
    missing = tuple(value for value in values if value not in text)
    if missing:
        raise ReleasePackageError(f"{label} missing: {', '.join(missing)}")
    return len(tuple(values))


def _load_json(root: Path, relative: str) -> dict:
    value = json.loads((root / relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReleasePackageError(f"{relative} is not a JSON object")
    return value


def check_release_package(root: Path) -> ReleaseCheckResult:
    for relative in RELEASE_DOCUMENTS:
        if not (root / relative).is_file():
            raise ReleasePackageError(f"missing release document: {relative}")
    texts = {
        relative: (root / relative).read_text(encoding="utf-8")
        for relative in RELEASE_DOCUMENTS
    }
    combined = "\n".join(texts.values())
    validate_no_secrets(combined)

    guide = texts["docs/release/local-development-guide.md"]
    headings = markdown_headings(guide)
    missing_headings = tuple(
        heading for heading in REQUIRED_GUIDE_HEADINGS if heading not in headings
    )
    if missing_headings:
        raise ReleasePackageError(
            f"local guide headings missing: {', '.join(missing_headings)}"
        )
    command_count = _require_all(guide, REQUIRED_COMMANDS, "release commands")

    limitations = texts["docs/release/known-limitations.md"]
    limit_count = _require_all(limitations, REQUIRED_LIMITS, "release limits")

    capability = texts["docs/release/capability-matrix.md"]
    openapi = _load_json(root, "tests/contract/snapshots/openapi-v1.json")
    paths = tuple(sorted(openapi.get("paths", {})))
    _require_all(capability, paths, "public API paths")

    codes = tuple(item.value for item in ErrorCode)
    _require_all(capability, codes, "stable error codes")
    documented_count = re.search(
        r"following\s+(\d+)\s+stable identifiers", capability
    )
    if documented_count is None or int(documented_count.group(1)) != len(codes):
        raise ReleasePackageError("documented stable error-code count is stale")

    summary = texts["docs/release/p1a-implementation-summary.md"]
    _require_all(summary, CORE_VERSIONS, "core dependency versions")
    if (root / ".python-version").read_text(encoding="utf-8").strip() != "3.12.13":
        raise ReleasePackageError(".python-version is not the release version")

    images = _load_json(
        root, "verification/compatibility/container-images-v1.0.json"
    ).get("images")
    if not isinstance(images, list) or len(images) != 3:
        raise ReleasePackageError("container image manifest is incomplete")
    image_sources = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in ("Dockerfile", "compose.yaml", "apps/web-test/Dockerfile", summary_path())
    )
    for image in images:
        reference = image.get("pinned_reference")
        if not isinstance(reference, str) or reference not in image_sources:
            raise ReleasePackageError(f"pinned image is not used/referenced: {reference}")

    provider = _load_json(
        root,
        "verification/providers/provider-declarations-deepseek-qwen-v1.0.json",
    )
    env = (root / ".env.example").read_text(encoding="utf-8")
    provider_values = (
        provider["chat"]["capability_fingerprint"],
        provider["embedding"]["embedding_space"]["compatibility_fingerprint"],
    )
    for value in provider_values:
        if value not in env or value not in summary:
            raise ReleasePackageError(f"provider fingerprint is not release-bound: {value}")

    for label, relative in PRIOR_REPORTS.items():
        report = _load_json(root, relative)
        if report.get("status") != "passed":
            raise ReleasePackageError(f"prior report did not pass: {label}")

    return ReleaseCheckResult(
        release_documents=len(RELEASE_DOCUMENTS),
        markdown_links=validate_links(root, RELEASE_DOCUMENTS),
        public_paths=len(paths),
        error_codes=len(codes),
        immutable_images=len(images),
        core_versions=len(CORE_VERSIONS),
        provider_fingerprints=len(provider_values),
        prior_reports=len(PRIOR_REPORTS),
        required_commands=command_count,
        required_limits=limit_count,
    )


def summary_path() -> str:
    return "docs/release/p1a-implementation-summary.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        result = check_release_package(root)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, ReleasePackageError) as error:
        print(f"P1A release package check failed: {error}")
        return 1
    print(
        "P1A release package check passed: "
        + json.dumps(asdict(result), separators=(",", ":"), sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
