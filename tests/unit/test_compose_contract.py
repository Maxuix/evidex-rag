from __future__ import annotations

import copy
import json
import os
import unittest
from unittest import mock
from pathlib import Path
from typing import Any

from tools.check_compose_contract import (
    FRONTEND_IMAGE,
    NODE_IMAGE,
    POSTGRES_IMAGE,
    PYTHON_IMAGE,
    SOURCE_TARGET,
    compose_environment,
    validate_compose,
)


PROJECT_ROOT = Path("/project")
APPLICATION_DOCKERFILE = f"""\
# application runtime
FROM {PYTHON_IMAGE}
"""
FRONTEND_DOCKERFILE = f"""\
FROM {NODE_IMAGE} AS build
RUN npm run build
FROM {PYTHON_IMAGE}
"""


def _published_port(target: int, published: str) -> list[dict[str, object]]:
    return [
        {
            "host_ip": "127.0.0.1",
            "target": target,
            "published": published,
            "protocol": "tcp",
        }
    ]


def _runtime_environment(frontend_port: str) -> dict[str, str]:
    return {
        "RAG_KB__DATABASE__MIGRATION_DSN": (
            "postgresql+asyncpg://rag_kb_migration:not-available@"
            "migration-disabled.invalid/rag_kb"
        ),
        "RAG_KB__DATABASE__RUNTIME_DSN": (
            "postgresql+asyncpg://rag_kb_runtime:runtime-password@"
            "postgres:5432/rag_kb"
        ),
        "RAG_KB__SECURITY__ALLOWED_CORS_ORIGINS": json.dumps(
            [f"http://127.0.0.1:{frontend_port}"]
        ),
        "RAG_KB__SECURITY__CORS_ALLOW_CREDENTIALS": "false",
    }


def valid_config(
    *, api_port: str = "8000", frontend_port: str = "3000"
) -> dict[str, Any]:
    source_mount = {
        "type": "volume",
        "source": "source-data",
        "target": SOURCE_TARGET,
        "volume": {},
    }
    runtime_environment = _runtime_environment(frontend_port)
    return {
        "services": {
            "postgres": {
                "image": POSTGRES_IMAGE,
                "ports": _published_port(5432, "5432"),
                "healthcheck": {"test": ["CMD", "true"]},
            },
            "storage-init": {"volumes": [copy.deepcopy(source_mount)]},
            "migrate": {
                "profiles": ["tools"],
                "environment": {
                    "RAG_KB__DATABASE__MIGRATION_DSN": (
                        "postgresql+asyncpg://rag_kb_migration:"
                        "migration-password@postgres:5432/rag_kb"
                    )
                },
                "command": ["python", "-m", "alembic", "upgrade", "head"],
            },
            "maintenance": {
                "profiles": ["tools"],
                "environment": copy.deepcopy(runtime_environment),
                "command": ["python", "-m", "apps.maintenance.main", "cleanup"],
                "volumes": [copy.deepcopy(source_mount)],
            },
            "api": {
                "environment": copy.deepcopy(runtime_environment),
                "command": ["python", "-m", "apps.api.main", "--container-listen"],
                "ports": _published_port(8000, api_port),
                "volumes": [copy.deepcopy(source_mount)],
                "depends_on": {
                    "storage-init": {"condition": "service_completed_successfully"}
                },
                "healthcheck": {"test": ["CMD", "true"]},
            },
            "worker": {
                "environment": copy.deepcopy(runtime_environment),
                "command": ["python", "-m", "apps.worker.main"],
                "volumes": [copy.deepcopy(source_mount)],
                "healthcheck": {"test": ["CMD", "true"]},
            },
            "frontend": {
                "image": FRONTEND_IMAGE,
                "build": {
                    "context": str(PROJECT_ROOT / "apps" / "web-test"),
                    "dockerfile": "Dockerfile",
                },
                "command": [
                    "python",
                    "/app/server.py",
                    "--api-base-url",
                    f"http://127.0.0.1:{api_port}/api/v1",
                ],
                "ports": _published_port(3000, frontend_port),
                "depends_on": {"api": {"condition": "service_healthy"}},
                "healthcheck": {
                    "test": [
                        "CMD",
                        "python",
                        "-c",
                        (
                            "import urllib.request; urllib.request.urlopen("
                            "'http://127.0.0.1:3000/health', timeout=2).read()"
                        ),
                    ]
                },
            },
        }
    }


def validate(config: dict[str, Any]) -> list[str]:
    return validate_compose(
        config,
        APPLICATION_DOCKERFILE,
        FRONTEND_DOCKERFILE,
        PROJECT_ROOT,
    )


class ComposeContractTests(unittest.TestCase):
    def test_compose_render_environment_replaces_caller_runtime_inputs(self) -> None:
        contaminated = {
            "RAG_KB_ENV_FILE": ".env.production",
            "RAG_KB__PROVIDER__API_KEY": "must-not-survive",
            "POSTGRES_ADMIN_PASSWORD": "real-admin-password",
            "RAG_KB_MIGRATION_PASSWORD": "real-migration-password",
            "RAG_KB_RUNTIME_PASSWORD": "real-runtime-password",
        }
        with mock.patch.dict(os.environ, contaminated, clear=False):
            environment = compose_environment()

        self.assertEqual(environment["RAG_KB_ENV_FILE"], ".env.example")
        self.assertNotIn("RAG_KB__PROVIDER__API_KEY", environment)
        self.assertEqual(
            environment["POSTGRES_ADMIN_PASSWORD"],
            "contract-admin-password",
        )

    def test_valid_frontend_contract_passes_with_dynamic_ports(self) -> None:
        self.assertEqual(
            validate(valid_config(api_port="48123", frontend_port="43123")),
            [],
        )

    def test_backend_and_frontend_base_images_are_frozen(self) -> None:
        application_errors = validate_compose(
            valid_config(),
            "FROM docker.io/library/python:latest",
            FRONTEND_DOCKERFILE,
            PROJECT_ROOT,
        )
        self.assertIn(
            "application Python base image is not the frozen digest",
            application_errors,
        )

        for changed_dockerfile in (
            FRONTEND_DOCKERFILE.replace(NODE_IMAGE, "docker.io/library/node:latest"),
            FRONTEND_DOCKERFILE.replace(
                PYTHON_IMAGE, "docker.io/library/python:latest"
            ),
        ):
            with self.subTest(changed_dockerfile=changed_dockerfile):
                errors = validate_compose(
                    valid_config(),
                    APPLICATION_DOCKERFILE,
                    changed_dockerfile,
                    PROJECT_ROOT,
                )
                self.assertIn(
                    "frontend base images are not the frozen Node build and "
                    "Python runtime digests",
                    errors,
                )

    def test_frontend_requires_independent_image_and_narrow_build_context(self) -> None:
        config = valid_config()
        frontend = config["services"]["frontend"]
        frontend["image"] = "rag-kb-app:s03-w07"
        frontend["build"] = {
            "context": str(PROJECT_ROOT),
            "dockerfile": "Frontend.Dockerfile",
        }

        errors = validate(config)

        self.assertIn(
            "frontend image is not the independent frozen Stage 06 image", errors
        )
        self.assertIn(
            "frontend build context is not the narrow apps/web-test directory", errors
        )
        self.assertIn(
            "frontend Dockerfile is not scoped to its narrow build context", errors
        )

    def test_frontend_rejects_every_private_runtime_attachment(self) -> None:
        forbidden_values: dict[str, object] = {
            "environment": {},
            "env_file": [],
            "volumes": [],
            "secrets": [],
            "configs": [],
        }
        for field, value in forbidden_values.items():
            with self.subTest(field=field):
                config = valid_config()
                frontend = config["services"]["frontend"]
                frontend[field] = value

                self.assertIn(f"frontend must not declare {field}", validate(config))

    def test_frontend_port_must_be_single_loopback_publication(self) -> None:
        config = valid_config()
        frontend = config["services"]["frontend"]
        frontend["ports"][0]["host_ip"] = "0.0.0.0"
        frontend["ports"].append(
            {
                "host_ip": "127.0.0.1",
                "target": 3001,
                "published": "3001",
            }
        )

        errors = validate(config)

        self.assertIn("frontend has a non-loopback or missing published port", errors)
        self.assertIn(
            "frontend does not publish exactly one container port 3000", errors
        )

    def test_frontend_waits_for_api_and_uses_dedicated_health_endpoint(self) -> None:
        config = valid_config()
        frontend = config["services"]["frontend"]
        frontend["depends_on"] = {"api": {"condition": "service_started"}}
        frontend["healthcheck"] = {
            "test": ["CMD", "python", "-c", "open('/')"]
        }

        errors = validate(config)

        self.assertIn("frontend does not wait for API health", errors)
        self.assertIn(
            "frontend health check does not use the dedicated /health endpoint",
            errors,
        )

    def test_runtime_api_url_tracks_published_api_port(self) -> None:
        config = valid_config(api_port="48123")
        frontend = config["services"]["frontend"]
        frontend["command"][-1] = "http://127.0.0.1:8000/api/v1"

        self.assertIn(
            "frontend runtime API URL command does not match the API port",
            validate(config),
        )

    def test_api_cors_origin_tracks_frontend_port_without_credentials(self) -> None:
        config = valid_config(frontend_port="43123")
        api_environment = config["services"]["api"]["environment"]
        api_environment["RAG_KB__SECURITY__ALLOWED_CORS_ORIGINS"] = (
            '["http://127.0.0.1:3000"]'
        )
        api_environment["RAG_KB__SECURITY__CORS_ALLOW_CREDENTIALS"] = "true"

        errors = validate(config)

        self.assertIn(
            "API CORS origin does not match the published frontend loopback port",
            errors,
        )
        self.assertIn("API CORS credentials are not disabled", errors)

    def test_invalid_cors_json_fails_closed(self) -> None:
        config = valid_config()
        api_environment = config["services"]["api"]["environment"]
        api_environment["RAG_KB__SECURITY__ALLOWED_CORS_ORIGINS"] = "not-json"

        self.assertIn(
            "API CORS origin does not match the published frontend loopback port",
            validate(config),
        )


if __name__ == "__main__":
    unittest.main()
