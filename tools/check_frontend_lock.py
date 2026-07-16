#!/usr/bin/env python3
"""Verify the Stage 06 frontend declarations and npm lock boundary."""

from __future__ import annotations

import base64
import binascii
import json
import re
import sys
from pathlib import Path
from typing import Any


EXACT_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?$")
APPROVED_LICENSES = {
    "0BSD",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "BlueOak-1.0.0",
    "CC0-1.0",
    "ISC",
    "MIT",
    "MIT-0",
    "MPL-2.0",
}
REGISTRY_PREFIX = "https://registry.npmjs.org/"


def validate_frontend_lock(
    package: dict[str, Any],
    lock: dict[str, Any],
) -> tuple[list[str], int, int]:
    errors: list[str] = []
    declared = {
        **_dependency_map(package, "dependencies", errors),
        **_dependency_map(package, "devDependencies", errors),
    }
    engines = package.get("engines")
    if not isinstance(engines, dict):
        errors.append("package.json engines must declare exact Node and npm versions")
    else:
        for tool in ("node", "npm"):
            value = engines.get(tool)
            if not isinstance(value, str) or not EXACT_VERSION.fullmatch(value):
                errors.append(f"package.json engine {tool} is not exactly pinned")
    expected_package_manager = (
        f"npm@{engines.get('npm')}" if isinstance(engines, dict) else None
    )
    if package.get("packageManager") != expected_package_manager:
        errors.append("packageManager does not match the exact npm engine")

    if lock.get("lockfileVersion") != 3:
        errors.append("package-lock.json is not lockfileVersion 3")
    packages = lock.get("packages")
    if not isinstance(packages, dict):
        return [*errors, "package-lock.json packages must be an object"], len(declared), 0
    root = packages.get("")
    if not isinstance(root, dict):
        errors.append("package-lock.json is missing its root package entry")
    else:
        locked_declared = {
            **_dependency_map(root, "dependencies", errors, source="lock root"),
            **_dependency_map(root, "devDependencies", errors, source="lock root"),
        }
        if locked_declared != declared:
            errors.append("package-lock root dependencies differ from package.json")
        if root.get("engines") != engines:
            errors.append("package-lock root engines differ from package.json")

    resolved_count = 0
    for path, metadata in packages.items():
        if path == "":
            continue
        if not isinstance(metadata, dict):
            errors.append(f"lock entry {path} is not an object")
            continue
        resolved_count += 1
        resolved = metadata.get("resolved")
        integrity = metadata.get("integrity")
        license_name = metadata.get("license")
        if not isinstance(resolved, str) or not resolved.startswith(REGISTRY_PREFIX):
            errors.append(f"lock entry {path} is outside the approved npm registry")
        if not isinstance(integrity, str) or not _valid_sha512_integrity(integrity):
            errors.append(f"lock entry {path} has no sha512 integrity value")
        if license_name not in APPROVED_LICENSES:
            errors.append(f"lock entry {path} has an unreviewed license: {license_name}")

    return errors, len(declared), resolved_count


def _valid_sha512_integrity(value: str) -> bool:
    tokens = value.split()
    if not tokens:
        return False
    for token in tokens:
        if not token.startswith("sha512-"):
            return False
        try:
            digest = base64.b64decode(token.removeprefix("sha512-"), validate=True)
        except (binascii.Error, ValueError):
            return False
        if len(digest) != 64:
            return False
    return True


def _dependency_map(
    value: dict[str, Any],
    field: str,
    errors: list[str],
    *,
    source: str = "package.json",
) -> dict[str, str]:
    dependencies = value.get(field, {})
    if not isinstance(dependencies, dict):
        errors.append(f"{source} {field} must be an object")
        return {}
    result: dict[str, str] = {}
    for name, version in dependencies.items():
        if not isinstance(name, str) or not isinstance(version, str):
            errors.append(f"{source} {field} contains an invalid dependency")
            continue
        if not EXACT_VERSION.fullmatch(version):
            errors.append(f"{source} dependency {name} is not exactly pinned")
        result[name] = version
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    package_path = root / "apps" / "web-test" / "package.json"
    lock_path = root / "apps" / "web-test" / "package-lock.json"
    try:
        package = json.loads(package_path.read_text(encoding="utf-8"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if not isinstance(package, dict) or not isinstance(lock, dict):
            raise ValueError("frontend dependency files must contain JSON objects")
        errors, direct_count, resolved_count = validate_frontend_lock(package, lock)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"frontend lock check error: {error}", file=sys.stderr)
        return 2

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(
        "frontend lock check passed: "
        f"{direct_count} direct and {resolved_count} resolved packages; "
        "exact versions, npm registry, sha512 integrity, and reviewed licenses"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
