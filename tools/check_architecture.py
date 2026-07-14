#!/usr/bin/env python3
"""Check local Python imports against the declared module boundaries."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    message: str

    def render(self, project_root: Path) -> str:
        try:
            display_path = self.path.relative_to(project_root)
        except ValueError:
            display_path = self.path
        return f"{display_path}:{self.line}: {self.message}"


@dataclass(frozen=True)
class BoundaryConfig:
    scan_roots: tuple[str, ...]
    allowed_dependencies: dict[str, frozenset[str]]

    @classmethod
    def load(cls, path: Path) -> BoundaryConfig:
        with path.open("rb") as config_file:
            raw = tomllib.load(config_file)

        scan_roots = raw.get("scan_roots")
        boundaries = raw.get("boundaries")
        if not isinstance(scan_roots, list) or not all(
            isinstance(item, str) for item in scan_roots
        ):
            raise ValueError("scan_roots must be an array of paths")
        if not isinstance(boundaries, dict):
            raise ValueError("boundaries must be a table")

        known = set(boundaries)
        allowed: dict[str, frozenset[str]] = {}
        for namespace, dependencies in boundaries.items():
            if not isinstance(namespace, str) or not isinstance(dependencies, list):
                raise ValueError("each boundary must map a namespace to an array")
            if not all(isinstance(item, str) for item in dependencies):
                raise ValueError(f"{namespace} dependencies must be namespace strings")
            unknown = set(dependencies) - known
            if unknown:
                raise ValueError(
                    f"{namespace} contains unknown dependencies: {sorted(unknown)}"
                )
            allowed[namespace] = frozenset(dependencies)

        return cls(tuple(scan_roots), allowed)

    @property
    def namespaces(self) -> tuple[str, ...]:
        return tuple(sorted(self.allowed_dependencies, key=len, reverse=True))

    @property
    def local_roots(self) -> frozenset[str]:
        return frozenset(
            namespace.split(".", maxsplit=1)[0] for namespace in self.namespaces
        )

    def owner_of(self, module: str) -> str | None:
        for namespace in self.namespaces:
            if module == namespace or module.startswith(f"{namespace}."):
                return namespace
        return None


def module_name(path: Path, scan_root: Path) -> str:
    relative = path.relative_to(scan_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    if scan_root.name != "src":
        parts.insert(0, scan_root.name)
    return ".".join(parts)


def imported_modules(
    tree: ast.AST, current_module: str, is_package: bool
) -> list[tuple[str, int]]:
    imports: list[tuple[str, int]] = []
    package = current_module if is_package else current_module.rpartition(".")[0]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            resolved = ""
            if node.level:
                relative_name = f"{'.' * node.level}{node.module or ''}"
                try:
                    resolved = importlib.util.resolve_name(relative_name, package)
                except (ImportError, ValueError):
                    resolved = ""
            elif node.module:
                resolved = node.module
            if resolved:
                imports.append((resolved, node.lineno))
                imports.extend(
                    (f"{resolved}.{alias.name}", node.lineno)
                    for alias in node.names
                    if alias.name != "*"
                )
    return imports


def check_project(project_root: Path, config: BoundaryConfig) -> list[Violation]:
    violations: list[Violation] = []

    for configured_root in config.scan_roots:
        scan_root = project_root / configured_root
        if not scan_root.is_dir():
            violations.append(
                Violation(
                    scan_root,
                    1,
                    f"configured scan root does not exist: {configured_root}",
                )
            )
            continue

        for path in sorted(scan_root.rglob("*.py")):
            module = module_name(path, scan_root)
            owner = config.owner_of(module)
            if owner is None:
                if module not in config.local_roots:
                    violations.append(
                        Violation(
                            path,
                            1,
                            f"module is outside every declared boundary: {module}",
                        )
                    )
                continue

            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as error:
                violations.append(
                    Violation(path, error.lineno or 1, f"cannot parse module: {error.msg}")
                )
                continue

            allowed = config.allowed_dependencies[owner]
            for imported, line in imported_modules(
                tree, module, path.name == "__init__.py"
            ):
                imported_root = imported.split(".", maxsplit=1)[0]
                if imported_root not in config.local_roots:
                    continue
                imported_owner = config.owner_of(imported)
                if imported_owner is None:
                    if imported not in config.local_roots:
                        violations.append(
                            Violation(
                                path,
                                line,
                                f"{owner} imports unowned local module {imported}",
                            )
                        )
                    continue
                if imported_owner != owner and imported_owner not in allowed:
                    violations.append(
                        Violation(
                            path,
                            line,
                            f"{owner} must not depend on {imported_owner} (import {imported})",
                        )
                    )

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--config", type=Path, default=Path("architecture.toml"))
    arguments = parser.parse_args(argv)

    project_root = arguments.project_root.resolve()
    config_path = arguments.config
    if not config_path.is_absolute():
        config_path = project_root / config_path

    try:
        config = BoundaryConfig.load(config_path)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"architecture configuration error: {error}", file=sys.stderr)
        return 2

    violations = check_project(project_root, config)
    if violations:
        for violation in violations:
            print(violation.render(project_root), file=sys.stderr)
        print(
            f"architecture boundary check failed: {len(violations)} violation(s)",
            file=sys.stderr,
        )
        return 1

    print(
        "architecture boundary check passed: "
        f"{len(config.allowed_dependencies)} boundaries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
