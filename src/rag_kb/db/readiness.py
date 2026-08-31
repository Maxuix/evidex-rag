"""Minimal database readiness check for the local runtime."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

EXPECTED_REVISION = "0024_remove_agent_deadline_reserve"


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
