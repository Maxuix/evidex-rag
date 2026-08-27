#!/usr/bin/env python3
"""Validate and atomically publish the report produced by quality evaluation."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import uuid
from typing import Any

from tools.evaluation_campaign_state import canonical_bytes, write_private_json


REPORT_SCHEMA = "large_evaluation_final_report_v1"
CHECKPOINT_SCHEMA = "large_evaluation_report_publish_v1"


class ReportPublishError(RuntimeError):
    """A content-safe report publication failure."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--source-markdown", type=Path, required=True)
    parser.add_argument("--locked-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser


def _private_file(path: Path, *, reason: str) -> Path:
    try:
        status = path.lstat()
    except FileNotFoundError as error:
        raise ReportPublishError(f"report_{reason}_missing") from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise ReportPublishError(f"report_{reason}_invalid")
    if stat.S_IMODE(status.st_mode) & 0o077:
        raise ReportPublishError(f"report_{reason}_permissions_invalid")
    return path.resolve()


def _atomic_bytes(destination: Path, payload: bytes, mode: int) -> str:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination.parent.chmod(0o700)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        os.chmod(destination, mode)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def _validate_report(path: Path) -> tuple[dict[str, Any], bytes]:
    source = _private_file(path, reason="source_report")
    try:
        report = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ReportPublishError("report_source_json_invalid") from error
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != REPORT_SCHEMA
        or report.get("status") != "completed"
        or not isinstance(report.get("artifact_sha256"), str)
    ):
        raise ReportPublishError("report_source_schema_invalid")
    unsigned = dict(report)
    artifact_sha256 = unsigned.pop("artifact_sha256")
    expected = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    if artifact_sha256 != expected:
        raise ReportPublishError("report_source_digest_invalid")
    return report, source.read_bytes()


def publish(arguments: argparse.Namespace) -> dict[str, Any]:
    report, report_bytes = _validate_report(arguments.source_report)
    source_markdown = _private_file(arguments.source_markdown, reason="source_markdown")
    markdown_bytes = source_markdown.read_bytes()
    if not markdown_bytes.strip():
        raise ReportPublishError("report_source_markdown_empty")

    locked_sha256 = _atomic_bytes(
        arguments.locked_output,
        report_bytes,
        0o600,
    )
    markdown_sha256 = _atomic_bytes(
        arguments.markdown_output,
        markdown_bytes,
        0o600,
    )
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "report_schema_version": report["schema_version"],
        "report_artifact_sha256": report["artifact_sha256"],
        "locked_report_sha256": locked_sha256,
        "markdown_report_sha256": markdown_sha256,
        "locked_report_path": str(arguments.locked_output),
        "markdown_report_path": str(arguments.markdown_output),
    }
    checkpoint_sha256 = write_private_json(arguments.checkpoint, checkpoint)
    return {
        "status": "completed",
        "locked_report_sha256": locked_sha256,
        "markdown_report_sha256": markdown_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }


def main() -> int:
    arguments = _parser().parse_args()
    try:
        result = publish(arguments)
    except (OSError, ReportPublishError, TypeError, ValueError) as error:
        print(json.dumps({"status": "failed", "failure_code": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
