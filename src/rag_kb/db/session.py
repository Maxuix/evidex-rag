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
) -> DatabaseResources:
    """Build a lazy runtime-role pool sized for the selected process."""

    engine = create_async_engine(
        runtime_dsn,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        connect_args={
            "server_settings": {"application_name": f"rag-kb-{process.value}"}
        },
    )
    sessions = async_sessionmaker(
        engine,
        class_=AsyncSession,
        autoflush=False,
        expire_on_commit=False,
    )
    return DatabaseResources(engine=engine, sessions=sessions, process=process)
