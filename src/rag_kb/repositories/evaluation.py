"""Evaluation persistence contract."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from rag_kb.domain.evaluation import (
    EvaluationCaseResult,
    EvaluationDatasetDefinition,
    EvaluationRunDefinition,
    EvaluationRunSnapshot,
)


@runtime_checkable
class EvaluationRepository(Protocol):
    async def start(self, definition: EvaluationRunDefinition) -> EvaluationRunSnapshot: ...

    async def complete(
        self,
        definition: EvaluationRunDefinition,
        results: tuple[EvaluationCaseResult, ...],
        *,
        completed_at: datetime,
    ) -> EvaluationRunSnapshot: ...

    async def fail(
        self,
        definition: EvaluationRunDefinition,
        *,
        completed_at: datetime,
        error_code: str,
        error_detail: dict[str, Any],
    ) -> EvaluationRunSnapshot: ...

    async def get(self, run_id: UUID) -> EvaluationRunSnapshot | None: ...

    async def ensure_dataset(
        self, definition: EvaluationDatasetDefinition
    ) -> UUID: ...
