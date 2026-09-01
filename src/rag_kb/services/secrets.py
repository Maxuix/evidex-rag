"""Restart-safe reconciliation for local model-provider secret files."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import logging
from typing import TYPE_CHECKING

from rag_kb.domain import ModelSecretReconciliationResult
from rag_kb.ports.model_secrets import ModelSecretStore
from rag_kb.uow import (
    TransactionMode,
    execute_in_transaction,
)

if TYPE_CHECKING:
    from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWork, SqlAlchemyUnitOfWorkFactory


LOGGER = logging.getLogger("rag_kb.model_secrets.reconciliation")


class ModelSecretReconciliationService:
    def __init__(
        self,
        unit_of_work: SqlAlchemyUnitOfWorkFactory,
        secret_store: ModelSecretStore,
        *,
        batch_size: int,
        orphan_grace_seconds: float,
    ) -> None:
        if batch_size <= 0 or orphan_grace_seconds <= 0:
            raise ValueError("secret reconciliation bounds must be positive")
        self._unit_of_work = unit_of_work
        self._secret_store = secret_store
        self._batch_size = batch_size
        self._orphan_grace = timedelta(seconds=orphan_grace_seconds)

    async def run_once(
        self,
        *,
        now: datetime | None = None,
    ) -> ModelSecretReconciliationResult:
        observed_at = now or datetime.now(UTC)

        async def load(uow: SqlAlchemyUnitOfWork) -> tuple[str, ...]:
            return await uow.model_settings.list_secret_references()

        # A failed read-only transaction must happen before any filesystem
        # mutation.  This is the fail-closed boundary for deletion.
        references = await execute_in_transaction(
            self._unit_of_work,
            load,
            mode=TransactionMode.REPEATABLE_READ_ONLY,
        )
        entries = await asyncio.to_thread(self._secret_store.list_entries)
        referenced = set(references)
        cutoff = observed_at - self._orphan_grace
        removed = retained = failed = invalid = 0

        for entry in entries[: self._batch_size]:
            if not entry.is_regular_file:
                invalid += 1
                continue
            if entry.canonical_reference is None and not entry.is_temporary:
                invalid += 1
                continue
            if entry.modified_at > cutoff:
                retained += 1
                continue
            if (
                not entry.is_temporary
                and entry.canonical_reference in referenced
            ):
                retained += 1
                continue
            try:
                await asyncio.to_thread(self._secret_store.delete_entry, entry)
            except FileNotFoundError:
                removed += 1
            except (OSError, ValueError):
                failed += 1
            else:
                removed += 1

        result = ModelSecretReconciliationResult(
            removed=removed,
            retained=retained,
            failed=failed,
            invalid=invalid,
        )
        LOGGER.log(
            logging.INFO if removed or failed else logging.DEBUG,
            "model_secret_reconciliation_completed removed=%d retained=%d "
            "failed=%d invalid=%d",
            result.removed,
            result.retained,
            result.failed,
            result.invalid,
        )
        return result
