#!/usr/bin/env python3
"""Run the Stage 01 component compatibility probe against real PostgreSQL/pgvector."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any

import asyncpg
from pydantic import BaseModel, ConfigDict
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


DIRECT_EXPECTED = {
    "alembic": "1.18.5",
    "asyncpg": "0.31.0",
    "fastapi": "0.135.4",
    "pgvector": "0.4.2",
    "pydantic-settings": "2.14.2",
    "SQLAlchemy": "2.0.51",
}


class ProbeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    dimension: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path")
    return parser.parse_args()


def package_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in sorted(DIRECT_EXPECTED, key=str.lower)
    }


async def probe() -> dict[str, Any]:
    host = os.getenv("PGHOST", "host.docker.internal")
    port = int(os.getenv("PGPORT", "55432"))
    user = os.getenv("PGUSER", "compat_user")
    password = os.environ["PGPASSWORD"]
    database = os.getenv("PGDATABASE", "compat_db")
    expected_postgres = os.getenv("EXPECTED_POSTGRESQL_VERSION", "18.4")
    expected_pgvector = os.getenv("EXPECTED_PGVECTOR_VERSION", "0.8.2")
    failures: list[str] = []

    python_version = platform.python_version()
    if python_version != "3.12.13":
        failures.append(f"Python version {python_version} != 3.12.13")

    installed = package_versions()
    for name, expected in DIRECT_EXPECTED.items():
        actual = installed[name]
        if actual != expected:
            failures.append(f"{name} version {actual} != {expected}")

    from fastapi.sse import EventSourceResponse, ServerSentEvent

    sse_event = ServerSentEvent(data={"status": "ok"}, event="compatibility")
    sse_response = EventSourceResponse(iter([sse_event]))
    fastapi_sse = {
        "module_import": True,
        "event_type": type(sse_event).__name__,
        "response_type": type(sse_response).__name__,
        "media_type": sse_response.media_type,
    }
    if sse_response.media_type != "text/event-stream":
        failures.append("FastAPI native SSE media type is not text/event-stream")

    model = ProbeModel(name="fixed-p1a-space", dimension=3)
    pydantic_v2 = {
        "model_validate": ProbeModel.model_validate(model.model_dump()).model_dump(),
        "extra_forbid": model.model_config.get("extra") == "forbid",
    }

    connection = await asyncpg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        timeout=10,
    )
    try:
        await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        server_version_raw = await connection.fetchval("SHOW server_version")
        server_match = re.match(r"(\d+\.\d+)", server_version_raw)
        server_version = server_match.group(1) if server_match else server_version_raw
        extension_version = await connection.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
        if server_version != expected_postgres:
            failures.append(
                f"PostgreSQL version {server_version} != {expected_postgres}"
            )
        if extension_version != expected_pgvector:
            failures.append(
                f"pgvector version {extension_version} != {expected_pgvector}"
            )

        await connection.execute("DROP TABLE IF EXISTS compatibility_vector_probe")
        await connection.execute(
            """
            CREATE TABLE compatibility_vector_probe (
                id integer PRIMARY KEY,
                category text NOT NULL,
                embedding vector(3) NOT NULL
            )
            """
        )
        await connection.execute(
            """
            INSERT INTO compatibility_vector_probe (id, category, embedding)
            VALUES
                (1, 'keep', '[1,0,0]'),
                (2, 'drop', '[0,1,0]'),
                (3, 'keep', '[0,0,1]')
            """
        )
        distances = await connection.fetchrow(
            """
            SELECT
                '[1,0,0]'::vector <=> '[1,0,0]'::vector AS cosine,
                '[1,0,0]'::vector <-> '[0,1,0]'::vector AS l2,
                '[1,0,0]'::vector <#> '[1,0,0]'::vector AS negative_inner_product
            """
        )
        exact_ids = await connection.fetch(
            """
            SELECT id
            FROM compatibility_vector_probe
            WHERE category = 'keep'
            ORDER BY embedding <=> '[1,0,0]'::vector
            LIMIT 2
            """
        )
        await connection.execute(
            """
            CREATE INDEX compatibility_vector_probe_hnsw
            ON compatibility_vector_probe
            USING hnsw (embedding vector_cosine_ops)
            """
        )
        await connection.execute("SET hnsw.iterative_scan = strict_order")
        iterative_scan = await connection.fetchval("SHOW hnsw.iterative_scan")
        operator_names = await connection.fetch(
            """
            SELECT DISTINCT oprname
            FROM pg_operator
            WHERE oprname IN ('<=>', '<->', '<#>')
            ORDER BY oprname
            """
        )
        operator_classes = await connection.fetch(
            """
            SELECT DISTINCT opcname
            FROM pg_opclass
            WHERE opcname IN ('vector_cosine_ops', 'vector_ip_ops', 'vector_l2_ops')
            ORDER BY opcname
            """
        )
        vector_type = await connection.fetchrow(
            "SELECT typname, typlen FROM pg_type WHERE typname = 'vector'"
        )
    finally:
        await connection.close()

    sqlalchemy_url = (
        f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
    )
    engine = create_async_engine(sqlalchemy_url, pool_pre_ping=True)
    try:
        async with engine.connect() as sql_connection:
            sqlalchemy_probe = (
                await sql_connection.execute(text("SELECT 1 AS compatibility_ok"))
            ).scalar_one()
    finally:
        await engine.dispose()

    if [record["id"] for record in exact_ids] != [1, 3]:
        failures.append("filtered exact cosine ordering returned unexpected ids")
    if iterative_scan != "strict_order":
        failures.append("hnsw.iterative_scan strict_order capability unavailable")
    if sqlalchemy_probe != 1:
        failures.append("SQLAlchemy async probe did not return 1")

    return {
        "schema_version": "1.0",
        "status": "passed" if not failures else "failed",
        "runtime": {
            "python": python_version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "direct_package_versions": installed,
        "fastapi_native_sse": fastapi_sse,
        "pydantic_v2": pydantic_v2,
        "postgresql": {
            "server_version": server_version,
            "server_version_raw": server_version_raw,
            "asyncpg_connection": True,
            "sqlalchemy_async_connection": sqlalchemy_probe == 1,
        },
        "pgvector": {
            "extension_version": extension_version,
            "type": dict(vector_type),
            "dimension_probe": 3,
            "distance_operators": [record["oprname"] for record in operator_names],
            "operator_classes": [record["opcname"] for record in operator_classes],
            "distance_results": dict(distances),
            "exact_filtered_result_ids": [record["id"] for record in exact_ids],
            "hnsw_index_create": True,
            "iterative_scan": iterative_scan,
            "fixed_p1a_type": "vector (single-precision float32)",
            "fixed_p1a_metric": "cosine (<=> / vector_cosine_ops)",
        },
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    try:
        report = asyncio.run(probe())
    except Exception as exc:  # compatibility failures must become visible evidence
        report = {
            "schema_version": "1.0",
            "status": "failed",
            "fatal_error": f"{type(exc).__name__}: {exc}",
        }

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
