"""Bounded, repeatable local-development cleanup orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from rag_kb.domain import (
    FileReconciliationResult,
    IndexCleanupResult,
    ModelSecretReconciliationResult,
)
from rag_kb.ports.files import IndexAssetStore
from rag_kb.services.files import FileReconciliationService
from rag_kb.services.secrets import ModelSecretReconciliationService
from rag_kb.uow import execute_in_transaction

if TYPE_CHECKING:
    from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWork, SqlAlchemyUnitOfWorkFactory


@dataclass(frozen=True, slots=True)
class MaintenanceCleanupResult:
    files: FileReconciliationResult
    index: IndexCleanupResult
    secrets: ModelSecretReconciliationResult = field(
        default_factory=ModelSecretReconciliationResult
    )


class MaintenanceCleanupService:
    def __init__(
        self,
        unit_of_work: SqlAlchemyUnitOfWorkFactory,
        file_reconciliation: FileReconciliationService,
        workspace_id: UUID,
        *,
        batch_size: int,
        retired_data_grace_seconds: float,
        task_retention_seconds: float,
        asset_store: IndexAssetStore | None = None,
        model_secret_reconciliation: ModelSecretReconciliationService | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._file_reconciliation = file_reconciliation
        self._workspace_id = workspace_id
        self._batch_size = batch_size
        self._retired_data_grace = timedelta(seconds=retired_data_grace_seconds)
        self._task_retention = timedelta(seconds=task_retention_seconds)
        self._asset_store = asset_store
        self._model_secret_reconciliation = model_secret_reconciliation

    async def run_once(
        self,
        *,
        now: datetime | None = None,
    ) -> MaintenanceCleanupResult:
        observed_at = now or datetime.now(UTC)
        files = await self._file_reconciliation.run_once(now=observed_at)
        secrets = (
            await self._model_secret_reconciliation.run_once(now=observed_at)
            if self._model_secret_reconciliation is not None
            else ModelSecretReconciliationResult()
        )
        data_before = observed_at - self._retired_data_grace

        async def list_targets(uow: SqlAlchemyUnitOfWork):
            return await uow.indexing.list_retired_target_assets(
                data_before=data_before,
                limit=self._batch_size,
            )

        targets = await execute_in_transaction(
            self._unit_of_work,
            list_targets,
        )
        approved_target_ids: list[UUID] = []
        for target in targets:
            if not target.assets:
                approved_target_ids.append(target.indexed_document_version_id)
                continue
            if self._asset_store is None:
                continue
            all_deleted = True
            for asset in target.assets:
                try:
                    identity = self._asset_store.parse_uri(asset.storage_uri)
                    if (
                        asset.workspace_id != self._workspace_id
                        or asset.indexed_document_version_id
                        != target.indexed_document_version_id
                        or identity.workspace_id != self._workspace_id
                        or identity.indexed_document_version_id
                        != target.indexed_document_version_id
                    ):
                        raise ValueError("index asset identity differs from target")
                    await self._asset_store.delete(identity)
                except Exception:
                    all_deleted = False
            if all_deleted:
                approved_target_ids.append(target.indexed_document_version_id)

        async def clean(uow: SqlAlchemyUnitOfWork) -> IndexCleanupResult:
            index = await uow.indexing.cleanup_retired(
                target_ids=tuple(approved_target_ids),
                data_before=data_before,
                tasks_before=observed_at - self._task_retention,
                limit=self._batch_size,
            )
            records = await uow.file_consistency.delete_expired_cleanup_records(
                before=observed_at - self._task_retention,
                limit=self._batch_size,
            )
            return IndexCleanupResult(
                retired_targets_cleaned=index.retired_targets_cleaned,
                vectors_deleted=index.vectors_deleted,
                chunks_deleted=index.chunks_deleted,
                plans_deleted=index.plans_deleted,
                manifests_deleted=index.manifests_deleted,
                assets_deleted=index.assets_deleted,
                relations_deleted=index.relations_deleted,
                jobs_deleted=index.jobs_deleted,
                file_cleanup_records_deleted=records,
            )

        index = await execute_in_transaction(
            self._unit_of_work,
            clean,
        )
        return MaintenanceCleanupResult(files=files, index=index, secrets=secrets)
