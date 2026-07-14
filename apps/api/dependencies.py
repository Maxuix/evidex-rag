"""API composition root for process-wide foundation dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_kb.auth import DevelopmentAuthProvider, SingleWorkspaceAccessPolicy
from rag_kb.config import (
    Settings,
    StartupValidation,
    load_settings,
    validate_startup_environment,
)
from rag_kb.db import (
    DatabaseProcess,
    DatabaseResources,
    RuntimeReadiness,
    create_database_resources,
    validate_runtime_readiness,
)
from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWorkFactory


@dataclass(frozen=True)
class ApiDependencies:
    """Dependencies currently safe to construct before API contracts exist."""

    settings: Settings
    startup: StartupValidation
    database: DatabaseResources
    unit_of_work: SqlAlchemyUnitOfWorkFactory
    auth_provider: DevelopmentAuthProvider
    access_policy: SingleWorkspaceAccessPolicy

    async def close(self) -> None:
        """Release process-owned database resources during API shutdown."""

        await self.database.close()

    async def start(self) -> RuntimeReadiness:
        """Fail startup when the migration-created runtime is incompatible."""

        return await self.check_readiness()

    async def check_readiness(self) -> RuntimeReadiness:
        return await validate_runtime_readiness(self.database.engine)


def build_api_dependencies(
    settings: Settings | None = None,
    *,
    env_file: str | Path | None = ".env",
) -> ApiDependencies:
    """Load configuration explicitly and fail before constructing an API app."""

    resolved_settings = settings or load_settings(env_file=env_file)
    startup = validate_startup_environment(resolved_settings)
    database_settings = resolved_settings.database
    identity = resolved_settings.identity
    database = create_database_resources(
        database_settings.runtime_dsn.get_secret_value(),
        pool_size=database_settings.api_pool_size,
        max_overflow=database_settings.api_max_overflow,
        process=DatabaseProcess.API,
    )
    return ApiDependencies(
        settings=resolved_settings,
        startup=startup,
        database=database,
        unit_of_work=SqlAlchemyUnitOfWorkFactory(
            database.sessions,
            identity.workspace_id,
        ),
        auth_provider=DevelopmentAuthProvider(
            deployment_profile=resolved_settings.app.deployment_profile.value,
            principal_id=identity.principal_id,
            client_id=identity.client_id,
            workspace_id=identity.workspace_id,
        ),
        access_policy=SingleWorkspaceAccessPolicy(identity.workspace_id),
    )
