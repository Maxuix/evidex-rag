#!/usr/bin/env python3
"""Render and verify the frozen local Compose boundary."""

from __future__ import annotations

import json
import os
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
NODE_IMAGE = (
    "docker.io/library/node:24.18.0-bookworm-slim@"
    "sha256:6f7b03f7c2c8e2e784dcf9295400527b9b1270fd37b7e9a7285cf83b6951452d"
)
FRONTEND_IMAGE = "rag-kb-frontend:s06-w01"
SOURCE_TARGET = "/var/lib/rag-kb/sources"
SENSITIVE_COMPOSE_ENVIRONMENT = {
    "POSTGRES_ADMIN_PASSWORD",
    "RAG_KB_MIGRATION_PASSWORD",
    "RAG_KB_RUNTIME_PASSWORD",
}


def compose_environment() -> dict[str, str]:
    """Return deterministic non-production inputs for Compose interpolation."""

    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("RAG_KB") and key not in SENSITIVE_COMPOSE_ENVIRONMENT
    }
    environment.update(
        {
            "POSTGRES_ADMIN_PASSWORD": "contract-admin-password",
            "RAG_KB_MIGRATION_PASSWORD": "contract-migration-password",
            "RAG_KB_RUNTIME_PASSWORD": "contract-runtime-password",
            "RAG_KB_POSTGRES_PORT": "5432",
            "RAG_KB_API_PORT": "8000",
            "RAG_KB_FRONTEND_PORT": "3000",
            "RAG_KB_ENV_FILE": ".env.example",
        }
    )
    return environment


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
        env=compose_environment(),
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


def dockerfile_base_images(dockerfile: str) -> tuple[str, ...]:
    """Return image references from Dockerfile ``FROM`` instructions."""

    images: list[str] = []
    for raw_line in dockerfile.splitlines():
        parts = raw_line.strip().split()
        if not parts or parts[0].upper() != "FROM":
            continue
        index = 1
        while index < len(parts) and parts[index].startswith("--"):
            index += 1
        if index < len(parts):
            images.append(parts[index])
    return tuple(images)


def published_port(service: dict[str, Any], target: int) -> str | None:
    matches = [
        port
        for port in service.get("ports", [])
        if port.get("target") == target and port.get("published") is not None
    ]
    if len(matches) != 1:
        return None
    return str(matches[0]["published"])


def validate_compose(
    config: dict[str, Any],
    dockerfile: str,
    frontend_dockerfile: str,
    project_root: Path,
) -> list[str]:
    errors: list[str] = []
    services = config.get("services", {})
    required = {
        "api",
        "worker",
        "postgres",
        "storage-init",
        "migrate",
        "maintenance",
        "frontend",
    }
    if set(services) != required:
        errors.append(f"service set differs: {sorted(set(services) ^ required)}")
        return errors

    if services["postgres"].get("image") != POSTGRES_IMAGE:
        errors.append("PostgreSQL/pgvector image is not the frozen digest")
    if dockerfile_base_images(dockerfile) != (PYTHON_IMAGE,):
        errors.append("application Python base image is not the frozen digest")
    if dockerfile_base_images(frontend_dockerfile) != (NODE_IMAGE, PYTHON_IMAGE):
        errors.append(
            "frontend base images are not the frozen Node build and Python "
            "runtime digests"
        )

    frontend = services["frontend"]
    if frontend.get("image") != FRONTEND_IMAGE:
        errors.append("frontend image is not the independent frozen Stage 06 image")
    frontend_build = frontend.get("build", {})
    expected_context = (project_root / "apps" / "web-test").resolve()
    context = frontend_build.get("context")
    if not isinstance(context, str) or Path(context).resolve() != expected_context:
        errors.append(
            "frontend build context is not the narrow apps/web-test directory"
        )
    if frontend_build.get("dockerfile") != "Dockerfile":
        errors.append("frontend Dockerfile is not scoped to its narrow build context")

    for name in ("postgres", "api", "frontend"):
        ports = services[name].get("ports", [])
        if not ports or any(port.get("host_ip") != "127.0.0.1" for port in ports):
            errors.append(f"{name} has a non-loopback or missing published port")

    api_mount = named_mount(services["api"], SOURCE_TARGET)
    worker_mount = named_mount(services["worker"], SOURCE_TARGET)
    storage_mount = named_mount(services["storage-init"], SOURCE_TARGET)
    maintenance_mount = named_mount(services["maintenance"], SOURCE_TARGET)
    if (
        not api_mount
        or api_mount != worker_mount
        or api_mount != storage_mount
        or api_mount != maintenance_mount
    ):
        errors.append(
            "API, Worker, maintenance, and storage initializer do not share one source volume"
        )
    if named_mount(frontend, SOURCE_TARGET):
        errors.append("frontend must not mount private source storage")
    for field in ("environment", "env_file", "volumes", "secrets", "configs"):
        if field in frontend:
            errors.append(f"frontend must not declare {field}")

    for name in ("api", "worker", "maintenance"):
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
    maintenance = services["maintenance"]
    if maintenance.get("profiles") != ["tools"]:
        errors.append("maintenance service is not isolated behind the tools profile")
    if maintenance.get("command") != [
        "python",
        "-m",
        "apps.maintenance.main",
        "cleanup",
    ]:
        errors.append("maintenance command is not the bounded cleanup entrypoint")
    for name in ("api", "worker"):
        if "alembic" in " ".join(services[name].get("command", [])):
            errors.append(f"{name} performs migrations at startup")

    for name in ("postgres", "api", "worker", "frontend"):
        if "healthcheck" not in services[name]:
            errors.append(f"{name} has no health check")
    storage_condition = services["api"].get("depends_on", {}).get("storage-init", {})
    if storage_condition.get("condition") != "service_completed_successfully":
        errors.append("API does not wait for successful storage initialization")
    frontend_condition = frontend.get("depends_on", {}).get("api", {})
    if frontend_condition.get("condition") != "service_healthy":
        errors.append("frontend does not wait for API health")

    api_port = published_port(services["api"], 8000)
    if api_port is None:
        errors.append("API does not publish exactly one container port 8000")
    else:
        expected_command = [
            "python",
            "/app/server.py",
            "--api-base-url",
            f"http://127.0.0.1:{api_port}/api/v1",
        ]
        if frontend.get("command") != expected_command:
            errors.append(
                "frontend runtime API URL command does not match the API port"
            )

    frontend_port = published_port(frontend, 3000)
    if frontend_port is None or len(frontend.get("ports", [])) != 1:
        errors.append("frontend does not publish exactly one container port 3000")

    frontend_health = " ".join(frontend.get("healthcheck", {}).get("test", []))
    if "http://127.0.0.1:3000/health" not in frontend_health:
        errors.append(
            "frontend health check does not use the dedicated /health endpoint"
        )

    api_environment = services["api"].get("environment", {})
    raw_origins = api_environment.get("RAG_KB__SECURITY__ALLOWED_CORS_ORIGINS")
    try:
        origins = json.loads(raw_origins) if isinstance(raw_origins, str) else None
    except json.JSONDecodeError:
        origins = None
    if frontend_port is not None:
        expected_origins = [f"http://127.0.0.1:{frontend_port}"]
        if origins != expected_origins:
            errors.append(
                "API CORS origin does not match the published frontend loopback port"
            )
    if api_environment.get("RAG_KB__SECURITY__CORS_ALLOW_CREDENTIALS") != "false":
        errors.append("API CORS credentials are not disabled")

    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        config = render_compose(root)
        errors = validate_compose(
            config,
            (root / "Dockerfile").read_text(encoding="utf-8"),
            (root / "apps" / "web-test" / "Dockerfile").read_text(
                encoding="utf-8"
            ),
            root,
        )
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"Compose contract check error: {error}", file=sys.stderr)
        return 2

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(
        "Compose contract check passed: 7 services, loopback ports, shared "
        "storage, isolated migration/maintenance tools, pinned images, and "
        "an independent public-API frontend"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
