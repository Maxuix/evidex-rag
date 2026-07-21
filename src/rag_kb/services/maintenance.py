"""Bounded, repeatable local-development cleanup orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from rag_kb.auth import AuthContext
from rag_kb.domain import FileReconciliationResult, IndexCleanupResult
from rag_kb.services.files import FileReconciliationService
from rag_kb.uow import UnitOfWork, UnitOfWorkFactory, UnitOfWorkPurpose, execute_in_transaction


@dataclass(frozen=True, slots=True)
class MaintenanceCleanupResult:
    files: FileReconciliationResult
    index: IndexCleanupResult


class MaintenanceCleanupService:
    def __init__(
        self,
        unit_of_work: UnitOfWorkFactory,
        file_reconciliation: FileReconciliationService,
        *,
        batch_size: int,
        retired_data_grace_seconds: float,
        task_retention_seconds: float,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._file_reconciliation = file_reconciliation
        self._batch_size = batch_size
        self._retired_data_grace = timedelta(seconds=retired_data_grace_seconds)
        self._task_retention = timedelta(seconds=task_retention_seconds)

    async def run_once(
        self,
        context: AuthContext,
        *,
        now: datetime | None = None,
    ) -> MaintenanceCleanupResult:
        observed_at = now or datetime.now(UTC)
        files = await self._file_reconciliation.run_once(context, now=observed_at)

        async def clean(uow: UnitOfWork) -> IndexCleanupResult:
            if uow.workspace_id != context.workspace_id:
                raise RuntimeError("maintenance workspace does not match identity")
            index = await uow.indexing.cleanup_retired(
                data_before=observed_at - self._retired_data_grace,
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
                jobs_deleted=index.jobs_deleted,
                file_cleanup_records_deleted=records,
            )

        index = await execute_in_transaction(
            self._unit_of_work,
            clean,
            purpose=UnitOfWorkPurpose.RECONCILIATION,
        )
        return MaintenanceCleanupResult(files=files, index=index)
