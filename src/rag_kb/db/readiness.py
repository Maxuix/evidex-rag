"""Minimal database readiness check for the local runtime."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

EXPECTED_REVISION = "0026_simplify_attempt_ownership"


class DatabaseReadinessError(RuntimeError):
    """The local database is not at the schema expected by this checkout."""


async def check_database_ready(engine: AsyncEngine) -> None:
    """Confirm connectivity and the current-only Alembic head."""

    async with engine.connect() as connection:
        revision = await connection.scalar(
            text("SELECT version_num FROM alembic_version")
        )
    if revision != EXPECTED_REVISION:
        raise DatabaseReadinessError(
            f"expected migration {EXPECTED_REVISION}, found {revision}"
        )


async def ensure_local_workspace(engine: AsyncEngine, workspace_id: UUID) -> None:
    """Create the configured single-user namespace when a database is empty."""

    async with engine.begin() as connection:
        await connection.execute(
            text(
                """
                INSERT INTO workspace (id, name)
                VALUES (:workspace_id, :name)
                ON CONFLICT (id) DO NOTHING
                """
            ),
            {
                "workspace_id": workspace_id,
                "name": f"local-{workspace_id}",
            },
        )
