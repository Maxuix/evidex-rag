"""Workspace-scoped indexing status and explicit retry operations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from rag_kb.domain import (
    IdempotencyKeyReusedError,
    IdempotencyScope,
    IndexingJobSnapshot,
    Page,
    ResourceNotFoundError,
    canonical_request_hash,
)
from rag_kb.uow import execute_in_transaction

if TYPE_CHECKING:
    from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWork, SqlAlchemyUnitOfWorkFactory


RETRY_INDEXING_JOB_ENDPOINT = "POST /api/v1/indexing-jobs/{job_id}/retry"


class IndexingJobService:
    def __init__(self, unit_of_work: SqlAlchemyUnitOfWorkFactory) -> None:
        self._unit_of_work = unit_of_work

    async def get(
        self,
        job_id: UUID,
    ) -> IndexingJobSnapshot:
        async def load(uow: SqlAlchemyUnitOfWork) -> IndexingJobSnapshot:
            result = await uow.indexing.get_job(job_id)
            if result is None:
                raise ResourceNotFoundError("indexing job was not found")
            return result

        return await execute_in_transaction(
            self._unit_of_work,
            load,
        )

    async def list(
        self,
        *,
        kb_id: UUID,
        limit: int,
        after: tuple[str, ...] | None,
    ) -> Page[IndexingJobSnapshot]:
        async def load(uow: SqlAlchemyUnitOfWork) -> Page[IndexingJobSnapshot]:
            if await uow.knowledge_bases.get(kb_id) is None:
                raise ResourceNotFoundError("knowledge base was not found")
            return await uow.indexing.list_jobs(
                kb_id=kb_id,
                limit=limit,
                after=after,
            )

        return await execute_in_transaction(
            self._unit_of_work,
            load,
        )

    async def retry(
        self,
        idempotency_key: UUID,
        job_id: UUID,
    ) -> IndexingJobSnapshot:
        scope = IdempotencyScope(
            RETRY_INDEXING_JOB_ENDPOINT,
            idempotency_key,
        )
        request_hash = canonical_request_hash({"job_id": str(job_id)})

        async def persist(uow: SqlAlchemyUnitOfWork) -> IndexingJobSnapshot:
            await uow.content_mutations.lock(scope)
            prior = await uow.content_mutations.get(scope)
            if prior is not None:
                if prior.request_hash != request_hash or prior.job_id != job_id:
                    raise IdempotencyKeyReusedError(
                        "idempotency key targets another indexing retry"
                    )
                replay = await uow.indexing.get_job(job_id)
                if replay is None:
                    raise ResourceNotFoundError("indexing job was not found")
                return replay
            retried = await uow.indexing.retry_failed(
                job_id,
                observed_at=datetime.now(UTC),
            )
            if retried is None:
                raise ResourceNotFoundError("indexing job was not found")
            await uow.content_mutations.add_indexing_retry(
                scope=scope,
                request_hash=request_hash,
                kb_id=retried.kb_id,
                document_id=retried.document_id,
                document_version_id=retried.document_version_id,
                indexed_document_version_id=retried.indexed_document_version_id,
                index_revision_id=retried.index_revision_id,
                job_id=retried.job_id,
            )
            return retried

        return await execute_in_transaction(self._unit_of_work, persist)
