"""Transactional candidate-to-serving promotion boundary."""

from __future__ import annotations

from rag_kb.domain import (
    ErrorCode,
    IndexingExecutionError,
    IndexingPhase,
    PromotionCommand,
    PromotionResult,
)
from rag_kb.uow import UnitOfWorkFactory, UnitOfWorkPurpose, execute_in_transaction


class CandidatePromotionService:
    """Promote a completed target without coupling promotion to index building."""

    def __init__(self, unit_of_work: UnitOfWorkFactory) -> None:
        self._unit_of_work = unit_of_work

    async def promote(self, command: PromotionCommand) -> PromotionResult:
        result = await execute_in_transaction(
            self._unit_of_work,
            lambda uow: uow.indexing.promote(command),
            purpose=UnitOfWorkPurpose.INDEXING,
        )
        if result is None:
            raise IndexingExecutionError(
                ErrorCode.INDEX_TARGET_INVALID,
                phase=IndexingPhase.COMPLETED,
                diagnostic={"check": "promotion_target"},
            )
        return result
