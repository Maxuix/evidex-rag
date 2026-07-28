"""Process-owned asynchronous SQLAlchemy engine and session resources."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


class DatabaseProcess(StrEnum):
    API = "api"
    WORKER = "worker"
    MAINTENANCE = "maintenance"


_DEFAULT_STATEMENT_TIMEOUT_MS = {
    DatabaseProcess.API: 30_000,
    DatabaseProcess.WORKER: 60_000,
    DatabaseProcess.MAINTENANCE: 300_000,
}
_MAX_SESSION_TIMEOUT_MS = 86_400_000


@dataclass(slots=True)
class DatabaseResources:
    """One process's engine and session factory; no global engine is created."""

    engine: AsyncEngine
    sessions: async_sessionmaker[AsyncSession]
    process: DatabaseProcess
    _closed: bool = False

    async def close(self) -> None:
        """Release the process pool exactly once during application shutdown."""

        if not self._closed:
            await self.engine.dispose()
            self._closed = True

    async def __aenter__(self) -> DatabaseResources:
        if self._closed:
            raise RuntimeError("database resources are already closed")
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


def create_database_resources(
    runtime_dsn: str,
    *,
    pool_size: int,
    max_overflow: int,
    process: DatabaseProcess,
    statement_timeout_ms: int | None = None,
    lock_timeout_ms: int = 5_000,
    idle_in_transaction_session_timeout_ms: int = 30_000,
) -> DatabaseResources:
    """Build a lazy runtime-role pool sized for the selected process."""

    resolved_statement_timeout_ms = (
        _DEFAULT_STATEMENT_TIMEOUT_MS[process]
        if statement_timeout_ms is None
        else statement_timeout_ms
    )
    timeout_settings = {
        "statement_timeout": resolved_statement_timeout_ms,
        "lock_timeout": lock_timeout_ms,
        "idle_in_transaction_session_timeout": (
            idle_in_transaction_session_timeout_ms
        ),
    }
    if any(
        value <= 0 or value > _MAX_SESSION_TIMEOUT_MS
        for value in timeout_settings.values()
    ):
        raise ValueError(
            "database session timeouts must be between 1 and 86400000 milliseconds"
        )
    server_settings = {
        "application_name": f"rag-kb-{process.value}",
        **{name: str(value) for name, value in timeout_settings.items()},
    }
    engine = create_async_engine(
        runtime_dsn,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        connect_args={"server_settings": server_settings},
    )
    sessions = async_sessionmaker(
        engine,
        class_=AsyncSession,
        autoflush=False,
        expire_on_commit=False,
    )
    return DatabaseResources(engine=engine, sessions=sessions, process=process)
