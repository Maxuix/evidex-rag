"""Short-transaction persistence service for offline evaluation runs."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from rag_kb.domain import (
    EvaluationCaseResult,
    EvaluationRunDefinition,
    EvaluationRunSnapshot,
)
from rag_kb.uow import UnitOfWork, UnitOfWorkFactory, UnitOfWorkPurpose, execute_in_transaction


class EvaluationPersistenceService:
    def __init__(self, unit_of_work: UnitOfWorkFactory) -> None:
        self._unit_of_work = unit_of_work

    async def start(self, definition: EvaluationRunDefinition) -> EvaluationRunSnapshot:
        return await execute_in_transaction(
            self._unit_of_work,
            lambda uow: self._start(uow, definition),
            purpose=UnitOfWorkPurpose.COMMAND,
        )

    async def complete(
        self,
        definition: EvaluationRunDefinition,
        results: tuple[EvaluationCaseResult, ...],
        *,
        completed_at: datetime,
    ) -> EvaluationRunSnapshot:
        return await execute_in_transaction(
            self._unit_of_work,
            lambda uow: self._complete(uow, definition, results, completed_at),
            purpose=UnitOfWorkPurpose.COMMAND,
        )

    async def fail(
        self,
        definition: EvaluationRunDefinition,
        *,
        completed_at: datetime,
        error_code: str,
        error_detail: dict[str, Any],
    ) -> EvaluationRunSnapshot:
        return await execute_in_transaction(
            self._unit_of_work,
            lambda uow: self._fail(
                uow,
                definition,
                completed_at,
                error_code,
                error_detail,
            ),
            purpose=UnitOfWorkPurpose.COMMAND,
        )

    async def get(self, run_id: UUID) -> EvaluationRunSnapshot | None:
        async def load(uow: UnitOfWork) -> EvaluationRunSnapshot | None:
            return await uow.evaluations.get(run_id)

        return await execute_in_transaction(
            self._unit_of_work,
            load,
            purpose=UnitOfWorkPurpose.REQUEST,
        )

    @staticmethod
    async def _start(
        uow: UnitOfWork,
        definition: EvaluationRunDefinition,
    ) -> EvaluationRunSnapshot:
        _require_workspace(uow, definition)
        return await uow.evaluations.start(definition)

    @staticmethod
    async def _complete(
        uow: UnitOfWork,
        definition: EvaluationRunDefinition,
        results: tuple[EvaluationCaseResult, ...],
        completed_at: datetime,
    ) -> EvaluationRunSnapshot:
        _require_workspace(uow, definition)
        return await uow.evaluations.complete(
            definition,
            results,
            completed_at=completed_at,
        )

    @staticmethod
    async def _fail(
        uow: UnitOfWork,
        definition: EvaluationRunDefinition,
        completed_at: datetime,
        error_code: str,
        error_detail: dict[str, Any],
    ) -> EvaluationRunSnapshot:
        _require_workspace(uow, definition)
        return await uow.evaluations.fail(
            definition,
            completed_at=completed_at,
            error_code=error_code,
            error_detail=error_detail,
        )


def _require_workspace(uow: UnitOfWork, definition: EvaluationRunDefinition) -> None:
    metadata_workspace = definition.run_config.get("workspace_id")
    if metadata_workspace != str(uow.workspace_id):
        raise RuntimeError("evaluation run config does not match Unit of Work scope")
