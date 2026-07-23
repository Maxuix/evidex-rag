"""Bounded active-revision hydration for composite evidence relations."""

from __future__ import annotations

from uuid import UUID

from rag_kb.auth import AuthContext
from rag_kb.domain import IndexChunkAssetRelationSnapshot
from rag_kb.uow import (
    TransactionMode,
    UnitOfWork,
    UnitOfWorkFactory,
    UnitOfWorkPurpose,
    execute_in_transaction,
)


class CompositeEvidenceHydrationService:
    def __init__(self, unit_of_work: UnitOfWorkFactory, *, limit: int = 500) -> None:
        if not 1 <= limit <= 2_000:
            raise ValueError("relation hydration limit must be between 1 and 2000")
        self._unit_of_work = unit_of_work
        self._limit = limit

    async def hydrate(
        self,
        context: AuthContext,
        *,
        kb_id: UUID,
        index_revision_id: UUID,
        chunk_ids: tuple[UUID, ...],
        asset_ids: tuple[UUID, ...],
    ) -> tuple[IndexChunkAssetRelationSnapshot, ...]:
        async def load(
            unit_of_work: UnitOfWork,
        ) -> tuple[IndexChunkAssetRelationSnapshot, ...]:
            if unit_of_work.workspace_id != context.workspace_id:
                raise RuntimeError("unit of work workspace scope mismatch")
            return await unit_of_work.indexing.list_relations(
                kb_id=kb_id,
                index_revision_id=index_revision_id,
                chunk_ids=chunk_ids,
                asset_ids=asset_ids,
                limit=self._limit,
            )

        return await execute_in_transaction(
            self._unit_of_work,
            load,
            purpose=UnitOfWorkPurpose.READ_SNAPSHOT,
            mode=TransactionMode.REPEATABLE_READ_ONLY,
        )
