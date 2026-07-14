#!/usr/bin/env python3
"""Check the current OpenAPI document against its reviewed v1 snapshot."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from apps.api.app import application


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = PROJECT_ROOT / "tests/contract/snapshots/openapi-v1.json"
HTTP_METHODS = frozenset(
    {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
)


def find_breaking_changes(
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> list[str]:
    """Report removals and narrowing changes covered by the v1 policy."""

    changes: list[str] = []
    baseline_paths = baseline.get("paths", {})
    current_paths = current.get("paths", {})
    for path, baseline_path in baseline_paths.items():
        current_path = current_paths.get(path)
        if current_path is None:
            changes.append(f"removed path: {path}")
            continue
        for method, baseline_operation in baseline_path.items():
            if method not in HTTP_METHODS:
                continue
            current_operation = current_path.get(method)
            if current_operation is None:
                changes.append(f"removed operation: {method.upper()} {path}")
                continue
            baseline_responses = baseline_operation.get("responses", {})
            current_responses = current_operation.get("responses", {})
            for status in baseline_responses:
                if status not in current_responses:
                    changes.append(
                        f"removed response: {method.upper()} {path} {status}"
                    )

    baseline_schemas = baseline.get("components", {}).get("schemas", {})
    current_schemas = current.get("components", {}).get("schemas", {})
    for name, baseline_schema in baseline_schemas.items():
        current_schema = current_schemas.get(name)
        if current_schema is None:
            changes.append(f"removed schema: {name}")
            continue
        _compare_schema(name, baseline_schema, current_schema, changes)
    return changes


def _compare_schema(
    name: str,
    baseline: dict[str, Any],
    current: dict[str, Any],
    changes: list[str],
) -> None:
    if baseline.get("type") != current.get("type"):
        changes.append(f"changed schema type: {name}")

    baseline_properties = baseline.get("properties", {})
    current_properties = current.get("properties", {})
    for property_name, baseline_property in baseline_properties.items():
        current_property = current_properties.get(property_name)
        if current_property is None:
            changes.append(f"removed schema property: {name}.{property_name}")
            continue
        baseline_enum = set(baseline_property.get("enum", ()))
        current_enum = set(current_property.get("enum", ()))
        if not baseline_enum.issubset(current_enum):
            changes.append(f"removed enum value: {name}.{property_name}")

    added_required = set(current.get("required", ())) - set(
        baseline.get("required", ())
    )
    for property_name in sorted(added_required):
        changes.append(f"added required property: {name}.{property_name}")


def main(snapshot_path: Path = DEFAULT_SNAPSHOT) -> int:
    baseline = json.loads(snapshot_path.read_text(encoding="utf-8"))
    current = application.openapi()
    breaking = find_breaking_changes(baseline, current)
    if breaking:
        for change in breaking:
            print(f"breaking OpenAPI change: {change}", file=sys.stderr)
        return 1
    if current != baseline:
        print(
            "OpenAPI snapshot drifted; review compatibility and update the snapshot",
            file=sys.stderr,
        )
        return 1
    print("OpenAPI compatibility check passed: snapshot is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
