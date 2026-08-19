#!/usr/bin/env python3
"""Run database integration tests against an isolated temporary PostgreSQL."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSTGRES_INIT = PROJECT_ROOT / "deploy/postgres/init-runtime.sh"
POSTGRES_BOOTSTRAP = PROJECT_ROOT / "deploy/postgres/bootstrap-roles.sql"
TEST_DATABASE = "rag_kb"
TEST_PASSWORD = "isolated-test-only"
CONTAINER_LABEL = "rag-kb.database-test-owner"
PORT_PATTERN = re.compile(r"^127\.0\.0\.1:(?P<port>[1-9][0-9]*)$")


@dataclass(frozen=True, slots=True)
class ContainerIdentity:
    name: str
    owner: str


def create_container_identity(
    project_root: Path = PROJECT_ROOT,
) -> ContainerIdentity:
    root_hash = hashlib.sha256(str(project_root).encode("utf-8")).hexdigest()[:8]
    owner = uuid.uuid4().hex
    return ContainerIdentity(
        name=f"rag-kb-db-test-{root_hash}-{owner[:8]}",
        owner=owner,
    )


def parse_published_port(output: str) -> int:
    endpoints = [line.strip() for line in output.splitlines() if line.strip()]
    if len(endpoints) != 1:
        raise RuntimeError("temporary PostgreSQL must expose exactly one host port")
    match = PORT_PATTERN.fullmatch(endpoints[0])
    if match is None:
        raise RuntimeError("temporary PostgreSQL must bind only to 127.0.0.1")
    port = int(match.group("port"))
    if port > 65_535:
        raise RuntimeError("Docker returned an invalid PostgreSQL port")
    return port


def database_environment(port: int) -> dict[str, str]:
    if not 1 <= port <= 65_535:
        raise ValueError("PostgreSQL port must be between 1 and 65535")
    host = f"127.0.0.1:{port}/{TEST_DATABASE}"
    migration = f"postgresql://rag_kb_migration@{host}"
    runtime = f"postgresql://rag_kb_runtime@{host}"
    return {
        "RAG_KB__DATABASE__MIGRATION_DSN": migration.replace(
            "postgresql://", "postgresql+asyncpg://", 1
        ),
        "RAG_KB_TEST_MIGRATION_DSN": migration,
        "RAG_KB_TEST_RUNTIME_DSN": runtime,
        "RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN": runtime.replace(
            "postgresql://", "postgresql+asyncpg://", 1
        ),
    }


def docker_run_command(
    *,
    image: str,
    identity: ContainerIdentity,
) -> list[str]:
    if not image.strip():
        raise ValueError("PostgreSQL image must not be empty")
    return [
        "docker",
        "run",
        "--pull",
        "never",
        "--detach",
        "--rm",
        "--name",
        identity.name,
        "--label",
        f"{CONTAINER_LABEL}={identity.owner}",
        "--env",
        f"POSTGRES_DB={TEST_DATABASE}",
        "--env",
        "POSTGRES_USER=postgres",
        "--env",
        f"POSTGRES_PASSWORD={TEST_PASSWORD}",
        "--env",
        f"RAG_KB_MIGRATION_PASSWORD={TEST_PASSWORD}",
        "--env",
        f"RAG_KB_RUNTIME_PASSWORD={TEST_PASSWORD}",
        "--env",
        "POSTGRES_HOST_AUTH_METHOD=trust",
        "--publish",
        "127.0.0.1::5432",
        "--tmpfs",
        "/var/lib/postgresql:rw,nosuid,size=1024m",
        "--mount",
        (
            f"type=bind,src={POSTGRES_INIT},"
            "dst=/docker-entrypoint-initdb.d/10-init-runtime.sh,readonly"
        ),
        "--mount",
        (
            f"type=bind,src={POSTGRES_BOOTSTRAP},"
            "dst=/opt/rag-kb/bootstrap-roles.sql,readonly"
        ),
        image,
    ]


def _run_capture(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _postgres_image() -> str:
    raw_config = _run_capture(
        [
            "docker",
            "compose",
            "--env-file",
            str(PROJECT_ROOT / ".env.example"),
            "--file",
            str(PROJECT_ROOT / "compose.yaml"),
            "config",
            "--format",
            "json",
        ]
    )
    configuration = json.loads(raw_config)
    image = configuration.get("services", {}).get("postgres", {}).get("image")
    if not isinstance(image, str) or not image.strip():
        raise RuntimeError("compose.yaml does not define the PostgreSQL image")
    return image


def _primary_worktree() -> Path | None:
    try:
        common = _run_capture(
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ]
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    common_path = Path(common)
    if common_path.name != ".git":
        return None
    return common_path.parent


def _test_python() -> Path:
    candidates = [PROJECT_ROOT / ".venv/bin/python"]
    primary = _primary_worktree()
    if primary is not None:
        candidates.append(primary / ".venv/bin/python")
    candidates.append(Path(sys.executable))

    observed: set[Path] = set()
    for candidate in candidates:
        candidate_key = candidate.resolve()
        if candidate_key in observed or not candidate.is_file():
            continue
        observed.add(candidate_key)
        version = subprocess.run(
            [
                str(candidate),
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if version.returncode == 0 and version.stdout.strip() == "3.12":
            return candidate
    raise RuntimeError(
        "a repository Python 3.12 virtual environment is required; "
        "dependency installation was not attempted"
    )


def _wait_until_ready(identity: ContainerIdentity) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        ready = subprocess.run(
            [
                "docker",
                "exec",
                identity.name,
                "pg_isready",
                "--username",
                "postgres",
                "--dbname",
                TEST_DATABASE,
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if ready.returncode == 0:
            return
        status = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", identity.name],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if status.returncode != 0 or status.stdout.strip() != "true":
            raise RuntimeError("temporary PostgreSQL exited before becoming ready")
        time.sleep(0.5)
    raise RuntimeError("temporary PostgreSQL did not become ready within 60 seconds")


def _cleanup(identity: ContainerIdentity) -> None:
    inspected_owner = subprocess.run(
        [
            "docker",
            "inspect",
            "--format",
            f'{{{{index .Config.Labels "{CONTAINER_LABEL}"}}}}',
            identity.name,
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if inspected_owner.returncode != 0:
        return
    if inspected_owner.stdout.strip() != identity.owner:
        raise RuntimeError("refusing to stop a temporary container with a wrong owner")
    stopped = subprocess.run(
        ["docker", "stop", "--time", "5", identity.name],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if stopped.returncode != 0:
        raise RuntimeError("failed to stop the owned temporary PostgreSQL container")


def _test_arguments(raw_arguments: list[str]) -> list[str]:
    arguments = list(raw_arguments)
    if arguments[:1] == ["--"]:
        arguments.pop(0)
    if arguments:
        return arguments
    return ["discover", "-s", "tests/integration/db", "-v"]


def run_database_tests(unittest_arguments: list[str]) -> int:
    identity = create_container_identity()
    python = _test_python()
    created = False
    try:
        image = _postgres_image()
        print(
            "Starting an isolated passwordless PostgreSQL test container...",
            flush=True,
        )
        _run_capture(docker_run_command(image=image, identity=identity))
        created = True
        _wait_until_ready(identity)
        _run_capture(
            [
                "docker",
                "exec",
                identity.name,
                "/docker-entrypoint-initdb.d/10-init-runtime.sh",
            ]
        )
        port = parse_published_port(
            _run_capture(["docker", "port", identity.name, "5432/tcp"])
        )
        environment = {
            **os.environ,
            **database_environment(port),
            "PYTHONPATH": os.pathsep.join(
                (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT))
            ),
        }
        print(
            f"Applying migrations to the temporary database on port {port}...",
            flush=True,
        )
        migration = subprocess.run(
            [str(python), "-m", "alembic", "upgrade", "head"],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
        )
        if migration.returncode != 0:
            return migration.returncode
        print(
            "Running database integration tests against the temporary database...",
            flush=True,
        )
        completed = subprocess.run(
            [str(python), "-m", "unittest", *unittest_arguments],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
        )
        return completed.returncode
    finally:
        if created:
            _cleanup(identity)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run database integration tests in a unique, temporary PostgreSQL "
            "container without reading local application credentials."
        )
    )
    parser.add_argument(
        "unittest_arguments",
        nargs=argparse.REMAINDER,
        help=(
            "arguments passed to python -m unittest; defaults to discovery in "
            "tests/integration/db"
        ),
    )
    arguments = parser.parse_args()
    try:
        return run_database_tests(_test_arguments(arguments.unittest_arguments))
    except KeyboardInterrupt:
        return 130
    except (
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as error:
        print(f"Database test setup failed: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
