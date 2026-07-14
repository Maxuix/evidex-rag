#!/usr/bin/env python3
"""Render and verify the frozen Stage 02 local Compose boundary."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


POSTGRES_IMAGE = (
    "docker.io/pgvector/pgvector:0.8.2-pg18-bookworm@"
    "sha256:42e7f6b4e1eceb02ff14e3e6bc6108bbe259abbe83879dc1845d0da1ddeb555d"
)
PYTHON_IMAGE = (
    "docker.io/library/python:3.12.13-slim-bookworm@"
    "sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b"
)
SOURCE_TARGET = "/var/lib/rag-kb/sources"


def render_compose(project_root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            ".env.example",
            "--profile",
            "tools",
            "config",
            "--format",
            "json",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def named_mount(service: dict[str, Any], target: str) -> dict[str, Any] | None:
    return next(
        (
            mount
            for mount in service.get("volumes", [])
            if mount.get("type") == "volume" and mount.get("target") == target
        ),
        None,
    )


def validate_compose(config: dict[str, Any], dockerfile: str) -> list[str]:
    errors: list[str] = []
    services = config.get("services", {})
    required = {"api", "worker", "postgres", "storage-init", "migrate", "frontend"}
    if set(services) != required:
        errors.append(f"service set differs: {sorted(set(services) ^ required)}")
        return errors

    if services["postgres"].get("image") != POSTGRES_IMAGE:
        errors.append("PostgreSQL/pgvector image is not the frozen digest")
    if dockerfile.splitlines()[0] != f"FROM {PYTHON_IMAGE}":
        errors.append("application Python base image is not the frozen digest")

    for name in ("postgres", "api", "frontend"):
        ports = services[name].get("ports", [])
        if not ports or any(port.get("host_ip") != "127.0.0.1" for port in ports):
            errors.append(f"{name} has a non-loopback or missing published port")

    api_mount = named_mount(services["api"], SOURCE_TARGET)
    worker_mount = named_mount(services["worker"], SOURCE_TARGET)
    storage_mount = named_mount(services["storage-init"], SOURCE_TARGET)
    if not api_mount or api_mount != worker_mount or api_mount != storage_mount:
        errors.append("API, Worker, and storage initializer do not share one source volume")
    if named_mount(services["frontend"], SOURCE_TARGET):
        errors.append("frontend must not mount private source storage")

    for name in ("api", "worker"):
        environment = services[name].get("environment", {})
        migration_dsn = environment.get("RAG_KB__DATABASE__MIGRATION_DSN", "")
        runtime_dsn = environment.get("RAG_KB__DATABASE__RUNTIME_DSN", "")
        if "migration-disabled.invalid" not in migration_dsn:
            errors.append(f"{name} receives usable migration credentials")
        if "rag_kb_runtime" not in runtime_dsn or "@postgres:5432/" not in runtime_dsn:
            errors.append(f"{name} does not receive the runtime role DSN")

    migrate = services["migrate"]
    if migrate.get("profiles") != ["tools"]:
        errors.append("migration service is not isolated behind the tools profile")
    migration_dsn = migrate.get("environment", {}).get(
        "RAG_KB__DATABASE__MIGRATION_DSN", ""
    )
    if set(migrate.get("environment", {})) != {
        "RAG_KB__DATABASE__MIGRATION_DSN"
    }:
        errors.append("migration service receives credentials outside its role")
    if "rag_kb_migration" not in migration_dsn or "@postgres:5432/" not in migration_dsn:
        errors.append("migration service does not receive the migration role DSN")
    if migrate.get("command") != ["python", "-m", "alembic", "upgrade", "head"]:
        errors.append("migration command is not the explicit Alembic upgrade")
    for name in ("api", "worker"):
        if "alembic" in " ".join(services[name].get("command", [])):
            errors.append(f"{name} performs migrations at startup")

    for name in ("postgres", "api", "worker", "frontend"):
        if "healthcheck" not in services[name]:
            errors.append(f"{name} has no health check")
    storage_condition = services["api"].get("depends_on", {}).get("storage-init", {})
    if storage_condition.get("condition") != "service_completed_successfully":
        errors.append("API does not wait for successful storage initialization")
    if "environment" in services["frontend"]:
        errors.append("frontend receives private application environment")

    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        config = render_compose(root)
        errors = validate_compose(
            config,
            (root / "Dockerfile").read_text(encoding="utf-8"),
        )
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"Compose contract check error: {error}", file=sys.stderr)
        return 2

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(
        "Compose contract check passed: 6 services, loopback ports, "
        "shared storage, isolated migration, and pinned images"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
