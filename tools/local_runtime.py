#!/usr/bin/env python3
"""Resolve and diagnose the one supported personal local runtime identity."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import stat
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_COMPOSE_PROJECT = "rag"
LOCAL_MANIFEST_NAME = ".env.local"
LEGACY_APP_ENV_NAME = ".env"
LEGACY_SOURCE_OVERRIDE = Path(
    ".runtime/routing-rag-current-source.override.yaml"
)
DEFAULT_PORTS = {
    "RAG_KB_API_PORT": 8000,
    "RAG_KB_FRONTEND_PORT": 3000,
    "RAG_KB_POSTGRES_PORT": 5432,
    "RAG_KB_FALKORDB_PORT": 6379,
}
REQUIRED_MANIFEST_KEYS = frozenset(
    {
        "COMPOSE_PROJECT_NAME",
        *DEFAULT_PORTS,
        "POSTGRES_ADMIN_PASSWORD",
        "RAG_KB_MIGRATION_PASSWORD",
        "RAG_KB_RUNTIME_PASSWORD",
        "RAG_KB__APP__BIND_HOST",
        "RAG_KB__IDENTITY__PRINCIPAL_ID",
        "RAG_KB__IDENTITY__CLIENT_ID",
        "RAG_KB__IDENTITY__WORKSPACE_ID",
        "RAG_KB__DATABASE__RUNTIME_DSN",
        "RAG_KB__DATABASE__MIGRATION_DSN",
    }
)


class LocalRuntimeError(ValueError):
    """The local runtime manifest cannot produce a safe canonical identity."""


@dataclass(frozen=True, slots=True)
class LocalRuntime:
    checkout: Path
    canonical_checkout: Path
    manifest: Path
    manifest_present: bool
    manifest_mode: int | None
    present_keys: frozenset[str]
    compose_project: str
    api_port: int
    frontend_port: int
    postgres_port: int
    falkordb_port: int

    @property
    def api_origin(self) -> str:
        return f"http://127.0.0.1:{self.api_port}"

    @property
    def frontend_origin(self) -> str:
        return f"http://127.0.0.1:{self.frontend_port}"

    @property
    def is_primary_checkout(self) -> bool:
        return self.checkout == self.canonical_checkout

    @property
    def is_canonical_manifest(self) -> bool:
        return self.manifest == self.canonical_checkout / LOCAL_MANIFEST_NAME


def resolve_local_runtime(
    *,
    checkout: Path = PROJECT_ROOT,
    canonical_checkout: Path | None = None,
    env_file: Path | None = None,
    require_manifest: bool = False,
) -> LocalRuntime:
    checkout = checkout.resolve()
    canonical = (
        canonical_checkout.resolve()
        if canonical_checkout is not None
        else discover_canonical_checkout(checkout)
    )
    manifest = (env_file or canonical / LOCAL_MANIFEST_NAME).absolute()
    manifest_present = manifest.exists()
    if require_manifest and not manifest_present:
        raise LocalRuntimeError("local manifest is missing")

    values: dict[str, str] = {}
    manifest_mode: int | None = None
    if manifest_present:
        file_status = manifest.lstat()
        if stat.S_ISLNK(file_status.st_mode) or not stat.S_ISREG(file_status.st_mode):
            raise LocalRuntimeError("local manifest is not a regular file")
        manifest_mode = stat.S_IMODE(file_status.st_mode)
        values = _parse_manifest(manifest)

    compose_project = values.get(
        "COMPOSE_PROJECT_NAME",
        CANONICAL_COMPOSE_PROJECT,
    )
    if compose_project != CANONICAL_COMPOSE_PROJECT:
        raise LocalRuntimeError("noncanonical Compose project")

    ports = {
        name: _parse_port(name, values.get(name, str(default)))
        for name, default in DEFAULT_PORTS.items()
    }
    if len(set(ports.values())) != len(ports):
        raise LocalRuntimeError("local service ports must be distinct")

    return LocalRuntime(
        checkout=checkout,
        canonical_checkout=canonical,
        manifest=manifest,
        manifest_present=manifest_present,
        manifest_mode=manifest_mode,
        present_keys=frozenset(values),
        compose_project=compose_project,
        api_port=ports["RAG_KB_API_PORT"],
        frontend_port=ports["RAG_KB_FRONTEND_PORT"],
        postgres_port=ports["RAG_KB_POSTGRES_PORT"],
        falkordb_port=ports["RAG_KB_FALKORDB_PORT"],
    )


def discover_canonical_checkout(checkout: Path = PROJECT_ROOT) -> Path:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    common_directory = Path(result.stdout.strip()).resolve()
    if common_directory.name != ".git":
        raise LocalRuntimeError("Git common directory is not a primary checkout")
    return common_directory.parent


def collect_active_compose_projects() -> tuple[str, ...] | None:
    try:
        result = subprocess.run(
            ["docker", "compose", "ls", "--format", "json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        payload = json.loads(result.stdout or "[]")
    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list):
        return None
    names = {
        str(item["Name"])
        for item in payload
        if isinstance(item, dict) and isinstance(item.get("Name"), str)
    }
    return tuple(sorted(names))


def build_doctor_report(
    runtime: LocalRuntime,
    *,
    active_compose_projects: tuple[str, ...] | None,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    _add_check(
        checks,
        "canonical_checkout",
        runtime.is_primary_checkout,
        "linked_worktree",
    )
    _add_check(
        checks,
        "manifest_present",
        runtime.manifest_present,
        "manifest_missing",
    )
    _add_check(
        checks,
        "manifest_permissions",
        runtime.manifest_mode == 0o600,
        "manifest_must_be_owner_only",
    )

    missing_keys = REQUIRED_MANIFEST_KEYS - runtime.present_keys
    _add_check(
        checks,
        "manifest_identity",
        not missing_keys,
        "missing_identity_keys",
        count=len(missing_keys),
    )
    _add_check(
        checks,
        "single_manifest",
        "RAG_KB_ENV_FILE" not in runtime.present_keys,
        "legacy_state_indirection",
    )
    _add_check(
        checks,
        "legacy_app_env",
        not (runtime.canonical_checkout / LEGACY_APP_ENV_NAME).exists(),
        "legacy_app_env_present",
    )
    _add_check(
        checks,
        "legacy_source_override",
        not (runtime.canonical_checkout / LEGACY_SOURCE_OVERRIDE).exists(),
        "legacy_source_override_present",
    )

    if active_compose_projects is None:
        checks.append(
            {
                "name": "compose_projects",
                "status": "warn",
                "reason": "docker_unavailable",
            }
        )
    else:
        noncanonical = tuple(
            name
            for name in active_compose_projects
            if name != CANONICAL_COMPOSE_PROJECT and name.startswith("rag")
        )
        _add_check(
            checks,
            "compose_projects",
            not noncanonical,
            "noncanonical_compose_project_active",
            count=len(noncanonical),
        )

    status = "fail" if any(item["status"] == "fail" for item in checks) else (
        "warn" if any(item["status"] == "warn" for item in checks) else "pass"
    )
    return {
        "schema_version": 1,
        "status": status,
        "identity": {
            "compose_project": runtime.compose_project,
            "checkout": (
                "primary"
                if runtime.is_primary_checkout
                else "linked"
            ),
            "ports": {
                "api": runtime.api_port,
                "frontend": runtime.frontend_port,
                "postgres": runtime.postgres_port,
                "falkordb": runtime.falkordb_port,
            },
        },
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="report content-safe local drift")
    doctor.add_argument("--env-file", type=Path)
    doctor.add_argument(
        "--skip-docker",
        action="store_true",
        help="skip the read-only active Compose project check",
    )
    arguments = parser.parse_args()

    try:
        runtime = resolve_local_runtime(
            checkout=PROJECT_ROOT,
            env_file=arguments.env_file,
        )
    except (LocalRuntimeError, OSError, subprocess.SubprocessError):
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "fail",
                    "checks": [
                        {
                            "name": "manifest",
                            "status": "fail",
                            "reason": "manifest_invalid",
                        }
                    ],
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 1

    active_projects = (
        None if arguments.skip_docker else collect_active_compose_projects()
    )
    report = build_doctor_report(
        runtime,
        active_compose_projects=active_projects,
    )
    print(json.dumps(report, separators=(",", ":"), sort_keys=True))
    return 1 if report["status"] == "fail" else 0


def _parse_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        if not separator or not name or not _is_env_name(name):
            raise LocalRuntimeError("local manifest contains an invalid assignment")
        if name in values:
            raise LocalRuntimeError("local manifest contains a duplicate key")
        values[name] = _unquote(value.strip())
    return values


def _is_env_name(value: str) -> bool:
    return value.replace("_", "A").isalnum() and not value[0].isdigit()


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_port(name: str, value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise LocalRuntimeError(f"{name} is not an integer") from error
    if not 1024 <= port <= 65535:
        raise LocalRuntimeError(f"{name} is outside the local port range")
    return port


def _add_check(
    checks: list[dict[str, object]],
    name: str,
    passed: bool,
    reason: str,
    *,
    count: int | None = None,
) -> None:
    check: dict[str, object] = {
        "name": name,
        "status": "pass" if passed else "fail",
    }
    if not passed:
        check["reason"] = reason
        if count is not None:
            check["count"] = count
    checks.append(check)


if __name__ == "__main__":
    raise SystemExit(main())
