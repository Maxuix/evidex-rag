#!/usr/bin/env python3
"""Run the real W05 evaluation against a disposable pinned pgvector database."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from run_db_integration import (
    IMAGE,
    MIGRATION_PASSWORD,
    POSTGRES_PASSWORD,
    PROJECT_ROOT,
    RUNTIME_PASSWORD,
    bootstrap,
    container_port,
    run,
    wait_until_ready,
)


BASE_URL_ENV = "RAG_KB__MODEL_PROVIDER__EMBEDDING__BASE_URL"
API_KEY_ENV = "RAG_KB__MODEL_PROVIDER__EMBEDDING__API_KEY"
PROVIDER_KEYS = frozenset({BASE_URL_ENV, API_KEY_ENV})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Local ignored file from which only the two embedding credentials are read",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("evaluation/configs/retrieval-evaluation-v1.0.json"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def provider_environment(env_file: Path) -> dict[str, str]:
    environment = os.environ.copy()
    path = env_file if env_file.is_absolute() else PROJECT_ROOT / env_file
    if path.is_file():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line.removeprefix("export ").lstrip()
            key, separator, value = line.partition("=")
            key = key.strip()
            if not separator or key not in PROVIDER_KEYS or environment.get(key):
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            environment[key] = value

    missing = [key for key in sorted(PROVIDER_KEYS) if not environment.get(key)]
    if missing:
        raise RuntimeError(
            "real retrieval evaluation requires local embedding base URL and API key; "
            f"missing: {', '.join(missing)}"
        )
    return environment


def main() -> int:
    args = parse_args()
    environment = provider_environment(args.env_file)
    container_name = f"rag-kb-retrieval-evaluation-{os.getpid()}"
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
        environment.update(
            {
                "PYTHONPATH": "src:.",
                "RAG_KB__DATABASE__MIGRATION_DSN": migration_sqlalchemy_dsn,
                "RAG_KB_TEST_RUNTIME_DSN": runtime_sqlalchemy_dsn.replace(
                    "+asyncpg", ""
                ),
                "RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN": runtime_sqlalchemy_dsn,
            }
        )
        run([sys.executable, "-m", "alembic", "upgrade", "head"], env=environment)
        run([sys.executable, "-m", "alembic", "check"], env=environment)

        evaluation_command = [
            sys.executable,
            "tools/retrieval_evaluation.py",
            "--config",
            str(args.config),
        ]
        if args.output is not None:
            evaluation_command.extend(("--output", str(args.output)))
        run(evaluation_command, env=environment)
        run([*evaluation_command, "--check"], env=environment)
        print("real retrieval evaluation passed and its report was reproduced")
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
