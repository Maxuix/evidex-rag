"""Read-only runtime readiness checks for PostgreSQL-backed P1A services."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from rag_kb.db.compatibility import (
    DatabaseCompatibility,
    validate_database_compatibility,
)


QUEUE_PROBES = (
    text("SELECT 1 FROM chat_run LIMIT 0"),
    text("SELECT 1 FROM indexing_job LIMIT 0"),
)


@dataclass(frozen=True, slots=True)
class RuntimeReadiness:
    database: str
    queue: str
    queue_backend: str
    compatibility: DatabaseCompatibility


async def validate_runtime_readiness(engine: AsyncEngine) -> RuntimeReadiness:
    """Validate schema and PostgreSQL queue access without writes or repair."""

    async with engine.connect() as connection:
        compatibility = await validate_database_compatibility(connection)
        for probe in QUEUE_PROBES:
            await connection.execute(probe)
    return RuntimeReadiness(
        database="ready",
        queue="ready",
        queue_backend="postgresql",
        compatibility=compatibility,
    )
