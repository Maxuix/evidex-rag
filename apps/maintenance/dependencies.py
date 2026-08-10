"""Composition root for bounded one-shot maintenance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from apps.model_asset_runtime import assemble_model_asset_runtime
from rag_kb.adapters.file_store.local import LocalFileStore
from rag_kb.auth import DevelopmentAuthProvider, SingleWorkspaceAccessPolicy
from rag_kb.config import (
    Settings,
    StartupValidation,
    load_settings,
    validate_startup_environment,
)
from rag_kb.db import DatabaseProcess, DatabaseResources, create_database_resources
from rag_kb.ports.files import IndexAssetStore
from rag_kb.services.content import build_content_services
from rag_kb.services.files import FileReconciliationService
from rag_kb.services.maintenance import MaintenanceCleanupService
from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWorkFactory


@dataclass(frozen=True)
class MaintenanceDependencies:
    settings: Settings
    startup: StartupValidation
    database: DatabaseResources
    auth_provider: DevelopmentAuthProvider
    asset_store: IndexAssetStore
    cleanup: MaintenanceCleanupService

    async def close(self) -> None:
        await self.database.close()


def build_maintenance_dependencies(
    settings: Settings | None = None,
    *,
    env_file: str | Path | None = ".env",
) -> MaintenanceDependencies:
    resolved = settings or load_settings(env_file=env_file)
    startup = validate_startup_environment(resolved)
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
    model_assets = assemble_model_asset_runtime(resolved)
    content = build_content_services(
        unit_of_work,
        access_policy,
        model_assets.embedding_settings,
        model_assets.multimodal_settings,
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
    return MaintenanceDependencies(
        settings=resolved,
        startup=startup,
        database=database,
        auth_provider=DevelopmentAuthProvider(
            deployment_profile=resolved.app.deployment_profile.value,
            principal_id=identity.principal_id,
            client_id=identity.client_id,
            workspace_id=identity.workspace_id,
        ),
        asset_store=model_assets.asset_store,
        cleanup=MaintenanceCleanupService(
            unit_of_work,
            files,
            batch_size=maintenance.batch_size,
            retired_data_grace_seconds=maintenance.retired_data_grace_seconds,
            task_retention_seconds=maintenance.task_retention_seconds,
            asset_store=model_assets.asset_store,
        ),
    )
