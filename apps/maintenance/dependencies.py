"""Composition root for bounded one-shot maintenance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_kb.adapters import LocalFileStore, LocalIndexAssetStore
from rag_kb.auth import DevelopmentAuthProvider, SingleWorkspaceAccessPolicy
from rag_kb.config import Settings, load_settings, validate_startup_environment
from rag_kb.db import DatabaseProcess, DatabaseResources, create_database_resources
from rag_kb.services import (
    FileReconciliationService,
    LexicalBackfillService,
    MaintenanceCleanupService,
    build_content_services,
)
from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWorkFactory


@dataclass(frozen=True)
class MaintenanceDependencies:
    settings: Settings
    database: DatabaseResources
    auth_provider: DevelopmentAuthProvider
    cleanup: MaintenanceCleanupService
    lexical_backfill: LexicalBackfillService

    async def close(self) -> None:
        await self.database.close()


def build_maintenance_dependencies(
    settings: Settings | None = None,
    *,
    env_file: str | Path | None = ".env",
) -> MaintenanceDependencies:
    resolved = settings or load_settings(env_file=env_file)
    validate_startup_environment(resolved)
    identity = resolved.identity
    access_policy = SingleWorkspaceAccessPolicy(identity.workspace_id)
    database = create_database_resources(
        resolved.database.runtime_dsn.get_secret_value(),
        pool_size=resolved.database.worker_pool_size,
        max_overflow=resolved.database.worker_max_overflow,
        process=DatabaseProcess.MAINTENANCE,
        statement_timeout_ms=(
            resolved.database.maintenance_statement_timeout_ms
        ),
        lock_timeout_ms=resolved.database.lock_timeout_ms,
        idle_in_transaction_session_timeout_ms=(
            resolved.database.idle_in_transaction_session_timeout_ms
        ),
    )
    unit_of_work = SqlAlchemyUnitOfWorkFactory(
        database.sessions,
        identity.workspace_id,
    )
    content = build_content_services(
        unit_of_work,
        access_policy,
        resolved.model_provider.embedding,
        resolved.model_provider.multimodal_embedding,
    )
    file_store = LocalFileStore(
        resolved.file_store.staging_path,
        resolved.file_store.final_path,
    )
    files = FileReconciliationService(
        unit_of_work,
        content.documents,
        file_store,
        batch_size=resolved.file_store.reconciliation_batch_size,
        orphan_grace_seconds=resolved.file_store.orphan_grace_seconds,
        cleanup_max_attempts=resolved.file_store.cleanup_max_attempts,
        cleanup_base_delay_seconds=resolved.file_store.cleanup_base_delay_seconds,
    )
    maintenance = resolved.maintenance
    asset_store = None
    if resolved.model_provider.multimodal_embedding is not None:
        assert resolved.file_store.asset_staging_path is not None
        assert resolved.file_store.asset_final_path is not None
        asset_store = LocalIndexAssetStore(
            resolved.file_store.asset_staging_path,
            resolved.file_store.asset_final_path,
        )
    return MaintenanceDependencies(
        settings=resolved,
        database=database,
        auth_provider=DevelopmentAuthProvider(
            deployment_profile=resolved.app.deployment_profile.value,
            principal_id=identity.principal_id,
            client_id=identity.client_id,
            workspace_id=identity.workspace_id,
        ),
        cleanup=MaintenanceCleanupService(
            unit_of_work,
            files,
            batch_size=maintenance.batch_size,
            retired_data_grace_seconds=maintenance.retired_data_grace_seconds,
            task_retention_seconds=maintenance.task_retention_seconds,
            asset_store=asset_store,
        ),
        lexical_backfill=LexicalBackfillService(
            database.sessions,
            identity.workspace_id,
        ),
    )
