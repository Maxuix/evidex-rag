"""Workspace-scoped indexing status and explicit retry operations."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from rag_kb.auth import AccessPolicy, AuthContext
from rag_kb.domain import (
    IdempotencyKeyReusedError,
    IdempotencyScope,
    IndexingJobSnapshot,
    Page,
    ResourceNotFoundError,
    canonical_request_hash,
)
from rag_kb.uow import UnitOfWork, UnitOfWorkFactory, UnitOfWorkPurpose, execute_in_transaction


RETRY_INDEXING_JOB_ENDPOINT = "POST /api/v1/indexing-jobs/{job_id}/retry"


class IndexingJobService:
    def __init__(
        self,
        unit_of_work: UnitOfWorkFactory,
        access_policy: AccessPolicy,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._access_policy = access_policy

    async def get(
        self,
        context: AuthContext,
        job_id: UUID,
    ) -> IndexingJobSnapshot:
        self._authorize(context)

        async def load(uow: UnitOfWork) -> IndexingJobSnapshot:
            _require_scope(uow, context)
            result = await uow.indexing.get_job(job_id)
            if result is None:
                raise ResourceNotFoundError("indexing job was not found")
            return result

        return await execute_in_transaction(
            self._unit_of_work,
            load,
            purpose=UnitOfWorkPurpose.REQUEST,
        )

    async def list(
        self,
        context: AuthContext,
        *,
        kb_id: UUID,
        limit: int,
        after: tuple[str, ...] | None,
    ) -> Page[IndexingJobSnapshot]:
        self._authorize(context)

        async def load(uow: UnitOfWork) -> Page[IndexingJobSnapshot]:
            _require_scope(uow, context)
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
            purpose=UnitOfWorkPurpose.REQUEST,
        )

    async def retry(
        self,
        context: AuthContext,
        idempotency_key: UUID,
        job_id: UUID,
    ) -> IndexingJobSnapshot:
        self._authorize(context)
        scope = IdempotencyScope(
            context.principal_id,
            context.client_id,
            RETRY_INDEXING_JOB_ENDPOINT,
            idempotency_key,
        )
        request_hash = canonical_request_hash({"job_id": str(job_id)})

        async def persist(uow: UnitOfWork) -> IndexingJobSnapshot:
            _require_scope(uow, context)
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

    def _authorize(self, context: AuthContext) -> None:
        self._access_policy.metadata_filter(context)


def _require_scope(uow: UnitOfWork, context: AuthContext) -> None:
    if uow.workspace_id != context.workspace_id:
        raise RuntimeError("Unit of Work scope does not match authenticated workspace")
