"""SQLAlchemy persistence for versioned evaluation runs."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_kb.db.models import (
    EvalCase as EvalCaseRow,
    EvalDataset as EvalDatasetRow,
    EvalResult as EvalResultRow,
    EvalRun as EvalRunRow,
    EvalRunStatus,
)
from rag_kb.domain import (
    EvaluationCaseResult,
    EvaluationConflictError,
    EvaluationDatasetDefinition,
    EvaluationRunDefinition,
    EvaluationRunSnapshot,
    EvaluationRunState,
)


class SqlAlchemyEvaluationRepository:
    def __init__(
        self,
        session: AsyncSession,
        workspace_id: UUID,
        ensure_active: Callable[[], None],
    ) -> None:
        self._session = session
        self._workspace_id = workspace_id
        self._ensure_active = ensure_active

    async def ensure_dataset(self, definition: EvaluationDatasetDefinition) -> UUID:
        self._ensure_active()
        row = await self._session.scalar(
            select(EvalDatasetRow)
            .where(
                EvalDatasetRow.workspace_id == self._workspace_id,
                EvalDatasetRow.name == definition.name,
                EvalDatasetRow.version == definition.version,
            )
            .with_for_update()
        )
        if row is None:
            row = EvalDatasetRow(
                workspace_id=self._workspace_id,
                name=definition.name,
                version=definition.version,
                manifest_hash=definition.manifest_hash,
                dataset_metadata=definition.metadata,
            )
            self._session.add(row)
            await self._session.flush()
        elif (
            row.manifest_hash != definition.manifest_hash
            or row.dataset_metadata != definition.metadata
        ):
            raise EvaluationConflictError(
                "evaluation dataset version was reused with different metadata"
            )

        existing = {
            case.case_key: case
            for case in (
                await self._session.scalars(
                    select(EvalCaseRow)
                    .where(EvalCaseRow.dataset_id == row.id)
                    .order_by(EvalCaseRow.case_key)
                )
            ).all()
        }
        expected_keys = {case.case_key for case in definition.cases}
        if set(existing) - expected_keys:
            raise EvaluationConflictError(
                "evaluation dataset contains cases absent from its manifest"
            )
        for case in definition.cases:
            prior = existing.get(case.case_key)
            if prior is None:
                self._session.add(
                    EvalCaseRow(
                        dataset_id=row.id,
                        case_key=case.case_key,
                        question=case.question,
                        expected=case.expected,
                        tags=list(case.tags),
                    )
                )
            elif (
                prior.question != case.question
                or prior.expected != case.expected
                or prior.tags != list(case.tags)
            ):
                raise EvaluationConflictError(
                    f"evaluation case changed within dataset version: {case.case_key}"
                )
        await self._session.flush()
        return row.id

    async def start(self, definition: EvaluationRunDefinition) -> EvaluationRunSnapshot:
        self._ensure_active()
        dataset_id = await self.ensure_dataset(definition.dataset)
        row = await self._session.get(EvalRunRow, definition.run_id, with_for_update=True)
        if row is None:
            row = EvalRunRow(
                id=definition.run_id,
                workspace_id=self._workspace_id,
                kb_id=definition.knowledge_base_id,
                dataset_id=dataset_id,
                index_revision_id=definition.index_revision_id,
                status=EvalRunStatus.RUNNING,
                run_config=definition.run_config,
                started_at=definition.started_at,
            )
            self._session.add(row)
            await self._session.flush()
        else:
            self._require_same_run(row, definition, dataset_id)
        return await self._snapshot(row)

    async def complete(
        self,
        definition: EvaluationRunDefinition,
        results: tuple[EvaluationCaseResult, ...],
        *,
        completed_at: datetime,
    ) -> EvaluationRunSnapshot:
        self._ensure_active()
        if completed_at.tzinfo is None:
            raise ValueError("evaluation completion time must be timezone-aware")
        started = await self.start(definition)
        row = await self._session.get(EvalRunRow, definition.run_id, with_for_update=True)
        assert row is not None
        case_rows = {
            case.case_key: case
            for case in (
                await self._session.scalars(
                    select(EvalCaseRow).where(EvalCaseRow.dataset_id == started.dataset_id)
                )
            ).all()
        }
        result_by_key = {result.case_key: result for result in results}
        if len(result_by_key) != len(results) or set(result_by_key) != set(case_rows):
            raise EvaluationConflictError(
                "completed evaluation must contain exactly one result per dataset case"
            )
        existing = {
            result.eval_case_id: result
            for result in (
                await self._session.scalars(
                    select(EvalResultRow).where(
                        EvalResultRow.eval_run_id == definition.run_id
                    )
                )
            ).all()
        }
        for case_key, result in result_by_key.items():
            case_id = case_rows[case_key].id
            prior = existing.get(case_id)
            values = _result_values(result)
            if prior is None:
                self._session.add(
                    EvalResultRow(
                        eval_run_id=definition.run_id,
                        eval_case_id=case_id,
                        **values,
                    )
                )
            elif any(getattr(prior, key) != value for key, value in values.items()):
                raise EvaluationConflictError(
                    f"evaluation result replay changed case: {case_key}"
                )
        if row.status is EvalRunStatus.FAILED:
            raise EvaluationConflictError("failed evaluation run cannot become completed")
        if (
            row.status is EvalRunStatus.COMPLETED
            and row.completed_at != completed_at
        ):
            raise EvaluationConflictError(
                "evaluation completion replay changed completion time"
            )
        row.status = EvalRunStatus.COMPLETED
        row.completed_at = completed_at
        row.error_code = None
        row.error_detail = None
        await self._session.flush()
        return await self._snapshot(row)

    async def fail(
        self,
        definition: EvaluationRunDefinition,
        *,
        completed_at: datetime,
        error_code: str,
        error_detail: dict[str, Any],
    ) -> EvaluationRunSnapshot:
        self._ensure_active()
        if not error_code or not error_detail or completed_at.tzinfo is None:
            raise ValueError("failed evaluation requires bounded error metadata")
        await self.start(definition)
        row = await self._session.get(EvalRunRow, definition.run_id, with_for_update=True)
        assert row is not None
        if row.status is EvalRunStatus.COMPLETED:
            raise EvaluationConflictError("completed evaluation run cannot become failed")
        if row.status is EvalRunStatus.FAILED and (
            row.error_code != error_code
            or row.error_detail != error_detail
            or row.completed_at != completed_at
        ):
            raise EvaluationConflictError("evaluation failure replay changed diagnostics")
        row.status = EvalRunStatus.FAILED
        row.completed_at = completed_at
        row.error_code = error_code
        row.error_detail = dict(error_detail)
        await self._session.flush()
        return await self._snapshot(row)

    async def get(self, run_id: UUID) -> EvaluationRunSnapshot | None:
        self._ensure_active()
        row = await self._session.scalar(
            select(EvalRunRow).where(
                EvalRunRow.workspace_id == self._workspace_id,
                EvalRunRow.id == run_id,
            )
        )
        return await self._snapshot(row) if row is not None else None

    def _require_same_run(
        self,
        row: EvalRunRow,
        definition: EvaluationRunDefinition,
        dataset_id: UUID,
    ) -> None:
        if (
            row.workspace_id != self._workspace_id
            or row.kb_id != definition.knowledge_base_id
            or row.dataset_id != dataset_id
            or row.index_revision_id != definition.index_revision_id
            or row.run_config != definition.run_config
            or row.started_at != definition.started_at
        ):
            raise EvaluationConflictError(
                "evaluation run ID was reused with different immutable input"
            )

    async def _snapshot(self, row: EvalRunRow) -> EvaluationRunSnapshot:
        count = await self._session.scalar(
            select(func.count())
            .select_from(EvalResultRow)
            .where(EvalResultRow.eval_run_id == row.id)
        )
        state = {
            EvalRunStatus.RUNNING: EvaluationRunState.RUNNING,
            EvalRunStatus.COMPLETED: EvaluationRunState.COMPLETED,
            EvalRunStatus.FAILED: EvaluationRunState.FAILED,
        }.get(row.status)
        if state is None:
            raise EvaluationConflictError("queued evaluation rows are not owned by W05")
        return EvaluationRunSnapshot(
            run_id=row.id,
            dataset_id=row.dataset_id,
            state=state,
            result_count=int(count or 0),
            error_code=row.error_code,
        )


def _result_values(result: EvaluationCaseResult) -> dict[str, Any]:
    return {
        "generated_answer": None,
        "evidence": result.evidence,
        "metrics": result.metrics,
        "error_code": result.error_code,
        "error_detail": result.error_detail,
    }
