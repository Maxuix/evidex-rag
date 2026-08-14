"""Transactional candidate-to-serving promotion boundary."""

from __future__ import annotations

from rag_kb.domain import (
    ErrorCode,
    IndexingExecutionError,
    IndexingPhase,
    PromotionCommand,
    PromotionResult,
    PromotionReason,
)
from rag_kb.uow import UnitOfWorkFactory, UnitOfWorkPurpose, execute_in_transaction


class CandidatePromotionService:
    """Promote a completed target without coupling promotion to index building."""

    def __init__(self, unit_of_work: UnitOfWorkFactory) -> None:
        self._unit_of_work = unit_of_work

    async def promote(self, command: PromotionCommand) -> PromotionResult:
        async def persist(uow):
            result = await uow.indexing.promote(command)
            graph = getattr(uow, "graph", None)
            if (
                result is not None
                and result.reason is PromotionReason.PROMOTED
                and graph is not None
            ):
                await graph.invalidate_for_indexed_target(
                    command.indexed_document_version_id
                )
            return result

        result = await execute_in_transaction(
            self._unit_of_work,
            persist,
            purpose=UnitOfWorkPurpose.INDEXING,
        )
        if result is None:
            raise IndexingExecutionError(
                ErrorCode.INDEX_TARGET_INVALID,
                phase=IndexingPhase.COMPLETED,
                diagnostic={"check": "promotion_target"},
            )
        return result
