#!/usr/bin/env python3
"""Run S02-W03 against the pinned PostgreSQL/pgvector container."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import asyncpg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE = (
    "docker.io/pgvector/pgvector:0.8.2-pg18-bookworm@"
    "sha256:42e7f6b4e1eceb02ff14e3e6bc6108bbe259abbe83879dc1845d0da1ddeb555d"
)
MIGRATION_PASSWORD = "migration-integration-secret"
RUNTIME_PASSWORD = "runtime-integration-secret"
POSTGRES_PASSWORD = "postgres-integration-secret"


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def container_port(container_name: str) -> int:
    result = subprocess.run(
        ["docker", "port", container_name, "5432/tcp"],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip().rsplit(":", maxsplit=1)[1])


def wait_until_ready(container_name: str) -> None:
    deadline = time.monotonic() + 30
    consecutive_successes = 0
    while time.monotonic() < deadline:
        result = subprocess.run(
            [
                "docker",
                "exec",
                container_name,
                "pg_isready",
                "-U",
                "postgres",
                "-d",
                "rag_kb",
            ],
            capture_output=True,
        )
        if result.returncode == 0:
            consecutive_successes += 1
            if consecutive_successes >= 5:
                return
        else:
            consecutive_successes = 0
        time.sleep(0.25)
    raise RuntimeError("PostgreSQL container did not become ready within 30 seconds")


def bootstrap(container_name: str) -> None:
    sql = (PROJECT_ROOT / "deploy/postgres/bootstrap-roles.sql").read_text(
        encoding="utf-8"
    )
    subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            container_name,
            "psql",
            "-U",
            "postgres",
            "-d",
            "rag_kb",
            "-v",
            "database_name=rag_kb",
            "-v",
            f"migration_password={MIGRATION_PASSWORD}",
            "-v",
            f"runtime_password={RUNTIME_PASSWORD}",
        ],
        input=sql,
        text=True,
        check=True,
    )


async def verify_clean_downgrade(port: int) -> None:
    connection = await asyncpg.connect(
        host="127.0.0.1",
        port=port,
        user="rag_kb_migration",
        password=MIGRATION_PASSWORD,
        database="rag_kb",
    )
    try:
        application_tables = await connection.fetchval(
            """
            SELECT count(*) FROM pg_tables
            WHERE schemaname = 'public' AND tablename <> 'alembic_version'
            """
        )
        application_enums = await connection.fetchval(
            """
            SELECT count(*) FROM pg_type type
            JOIN pg_namespace namespace ON namespace.oid = type.typnamespace
            WHERE namespace.nspname = 'public' AND type.typtype = 'e'
            """
        )
        vector_version = await connection.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
    finally:
        await connection.close()

    if application_tables != 0 or application_enums != 0:
        raise RuntimeError(
            "downgrade left application schema objects: "
            f"tables={application_tables}, enums={application_enums}"
        )
    if vector_version != "0.8.2":
        raise RuntimeError("cluster-owned pgvector extension was removed or changed")


def main() -> int:
    container_name = f"rag-kb-s02-w03-{os.getpid()}"
    started = False
    try:
        run(
            [
                "docker",
                "run",
                "--detach",
                "--rm",
                "--name",
                container_name,
                "-e",
                f"POSTGRES_PASSWORD={POSTGRES_PASSWORD}",
                "-e",
                "POSTGRES_DB=rag_kb",
                "-p",
                "127.0.0.1::5432",
                IMAGE,
            ]
        )
        started = True
        wait_until_ready(container_name)
        port = container_port(container_name)
        bootstrap(container_name)

        migration_sqlalchemy_dsn = (
            "postgresql+asyncpg://rag_kb_migration:"
            f"{MIGRATION_PASSWORD}@127.0.0.1:{port}/rag_kb"
        )
        runtime_sqlalchemy_dsn = (
            "postgresql+asyncpg://rag_kb_runtime:"
            f"{RUNTIME_PASSWORD}@127.0.0.1:{port}/rag_kb"
        )
        test_environment = os.environ.copy()
        test_environment.update(
            {
                "PYTHONPATH": "src:.",
                "RAG_KB__DATABASE__MIGRATION_DSN": migration_sqlalchemy_dsn,
                "RAG_KB_TEST_MIGRATION_DSN": migration_sqlalchemy_dsn.replace(
                    "+asyncpg", ""
                ),
                "RAG_KB_TEST_RUNTIME_DSN": runtime_sqlalchemy_dsn.replace(
                    "+asyncpg", ""
                ),
                "RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN": runtime_sqlalchemy_dsn,
            }
        )

        run([sys.executable, "-m", "alembic", "upgrade", "head"], env=test_environment)
        run([sys.executable, "-m", "alembic", "check"], env=test_environment)
        run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests/integration/db",
                "-v",
            ],
            env=test_environment,
        )
        run([sys.executable, "-m", "alembic", "downgrade", "base"], env=test_environment)
        asyncio.run(verify_clean_downgrade(port))
        run([sys.executable, "-m", "alembic", "upgrade", "head"], env=test_environment)
        run([sys.executable, "-m", "alembic", "check"], env=test_environment)
        print("database integration suite passed")
        return 0
    finally:
        if started:
            subprocess.run(
                ["docker", "stop", "--time", "5", container_name],
                cwd=PROJECT_ROOT,
                check=False,
                stdout=subprocess.DEVNULL,
            )


if __name__ == "__main__":
    raise SystemExit(main())
