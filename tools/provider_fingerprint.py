#!/usr/bin/env python3
"""Validate provider declarations and derive immutable configuration fingerprints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROHIBITED_SECRET_KEYS = {
    "api_key",
    "authorization",
    "password",
    "secret",
    "token",
}
REQUIRED_COMMON = (
    "logical_endpoint_id",
    "provider_id",
    "requested_model",
    "resolved_model",
    "model_version",
    "configuration_fingerprint",
    "timeout_seconds",
    "max_retries",
    "max_concurrency",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("declaration", type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Validate a template and report missing values without failing",
    )
    return parser.parse_args()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def find_secret_fields(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.lower()
            if normalized in PROHIBITED_SECRET_KEYS:
                findings.append(f"{path}.{key}")
            findings.extend(find_secret_fields(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(find_secret_fields(child, f"{path}[{index}]"))
    return findings


def missing_required(declaration: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for provider_name in ("chat", "embedding"):
        provider = declaration.get(provider_name, {})
        for field in REQUIRED_COMMON:
            if provider.get(field) is None:
                missing.append(f"{provider_name}.{field}")
        capabilities = provider.get("capabilities", {})
        for field, value in capabilities.items():
            if value is None:
                missing.append(f"{provider_name}.capabilities.{field}")
    space = declaration.get("embedding", {}).get("embedding_space", {})
    for field in ("metric", "vector_data_type", "normalization"):
        if space.get(field) is None:
            missing.append(f"embedding.embedding_space.{field}")
    validation = declaration.get("validation", {})
    if validation.get("real_endpoint_verified") is not True:
        missing.append("validation.real_endpoint_verified")
    if validation.get("verified_at") is None:
        missing.append("validation.verified_at")
    if validation.get("report_path") is None:
        missing.append("validation.report_path")
    return sorted(missing)


def fingerprint_inputs(provider: dict[str, Any], *, embedding: bool) -> dict[str, Any]:
    values = {
        "logical_endpoint_id": provider["logical_endpoint_id"],
        "provider_id": provider["provider_id"],
        "protocol": provider["protocol"],
        "requested_model": provider["requested_model"],
        "resolved_model": provider["resolved_model"],
        "model_version": provider["model_version"],
        "deployment_revision": provider.get("deployment_revision"),
        "configuration_fingerprint": provider["configuration_fingerprint"],
        "capabilities": provider["capabilities"],
    }
    if embedding:
        values["embedding_space"] = {
            key: value
            for key, value in provider["embedding_space"].items()
            if key != "compatibility_fingerprint"
        }
    return values


def main() -> int:
    args = parse_args()
    try:
        declaration = json.loads(args.declaration.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"provider declaration failed: {exc}", file=sys.stderr)
        return 1

    secret_fields = find_secret_fields(declaration)
    missing = missing_required(declaration)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "declaration": str(args.declaration),
        "secret_fields_found": secret_fields,
        "missing_required_fields": missing,
        "complete": not missing and not secret_fields,
    }
    if not missing and not secret_fields:
        report["chat_fingerprint"] = canonical_hash(
            fingerprint_inputs(declaration["chat"], embedding=False)
        )
        report["embedding_space_fingerprint"] = canonical_hash(
            fingerprint_inputs(declaration["embedding"], embedding=True)
        )

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if secret_fields:
        return 1
    if missing and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
