#!/usr/bin/env python3
"""Verify the application dependency set against the Stage 01 frozen baseline."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path


EXACT_DEPENDENCY = re.compile(r"^([A-Za-z0-9_.-]+)==([^;\s]+)$")
LOCK_ENTRY = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)")


def normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_exact_dependencies(values: list[str], source: Path) -> dict[str, str]:
    dependencies: dict[str, str] = {}
    for value in values:
        match = EXACT_DEPENDENCY.fullmatch(value)
        if match is None:
            raise ValueError(f"{source}: dependency is not exactly pinned: {value}")
        dependencies[normalized_name(match.group(1))] = match.group(2)
    return dependencies


def parse_requirements_in(path: Path) -> dict[str, str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return parse_exact_dependencies(values, path)


def parse_lock(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LOCK_ENTRY.match(line)
        if match:
            packages[normalized_name(match.group(1))] = match.group(2)
    if not packages:
        raise ValueError(f"{path}: no locked packages found")
    return packages


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pyproject_path = root / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    try:
        project_dependencies = parse_exact_dependencies(
            pyproject["project"]["dependencies"], pyproject_path
        )
        baseline_dependencies = parse_requirements_in(
            root / "verification/compatibility/requirements.in"
        )
        application_lock = parse_lock(root / "requirements.lock")
        baseline_lock = parse_lock(
            root / "verification/compatibility/requirements.lock"
        )
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        tomllib.TOMLDecodeError,
    ) as error:
        print(f"application lock check error: {error}", file=sys.stderr)
        return 2

    errors: list[str] = []
    missing_verified_direct = baseline_dependencies.keys() - project_dependencies.keys()
    if missing_verified_direct:
        errors.append(
            "pyproject is missing verified direct dependencies: "
            f"{sorted(missing_verified_direct)}"
        )
    for name, version in project_dependencies.items():
        locked_version = baseline_lock.get(name)
        if locked_version != version:
            errors.append(
                f"pyproject dependency {name}=={version} is not in the verified lock"
            )
    if application_lock != baseline_lock:
        errors.append("application lock package versions differ from the verified lock")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(
        "application lock check passed: "
        f"{len(project_dependencies)} direct and {len(application_lock)} total packages"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
