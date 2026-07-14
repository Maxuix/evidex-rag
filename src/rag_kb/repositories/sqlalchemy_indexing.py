"""SQLAlchemy persistence for idempotent chunk and pgvector writes."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import and_, exists, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from rag_kb.db.models import (
    Document as DocumentRow,
    DocumentSourceStatus,
    DocumentVersion as DocumentVersionRow,
    EmbeddingSpace as EmbeddingSpaceRow,
    IndexBuildStatus,
    IndexChunk as IndexChunkRow,
    IndexedDocumentVersion as IndexedDocumentVersionRow,
    IndexingJob as IndexingJobRow,
    IndexRevision as IndexRevisionRow,
    IndexRevisionStatus,
    IndexServingStatus,
    JobStatus,
    KnowledgeBase as KnowledgeBaseRow,
    SourceChange as SourceChangeRow,
    VectorRecord as VectorRecordRow,
)
from rag_kb.domain import (
    EmbeddingSpaceDefinition,
    ErrorCode,
    IndexChunkWrite,
    IndexingCancelled,
    IndexingCommand,
    IndexingExecutionError,
    IndexingLease,
    IndexingPhase,
    IndexingTarget,
    PromotionCommand,
    PromotionReason,
    PromotionResult,
    PromotionStatus,
    ReconciliationResult,
    VectorRecordWrite,
    stable_chunk_id,
)


class SqlAlchemyIndexingRepository:
    def __init__(
        self,
        session: AsyncSession,
        workspace_id: UUID,
        ensure_active: Callable[[], None],
    ) -> None:
        self._session = session
        self._workspace_id = workspace_id
        self._ensure_active = ensure_active

    async def claim(
        self,
        *,
        worker_id: str,
        observed_at: datetime,
        max_attempts: int,
    ) -> IndexingLease | None:
        self._ensure_active()
        row = (
            await self._session.execute(
                select(IndexingJobRow, IndexedDocumentVersionRow)
                .join(
                    IndexedDocumentVersionRow,
                    IndexedDocumentVersionRow.id
                    == IndexingJobRow.indexed_document_version_id,
                )
                .where(
                    IndexingJobRow.workspace_id == self._workspace_id,
                    IndexedDocumentVersionRow.workspace_id == self._workspace_id,
                    or_(
                        and_(
                            IndexingJobRow.status == JobStatus.QUEUED,
                            IndexingJobRow.attempt < max_attempts,
                            or_(
                                IndexingJobRow.next_attempt_at.is_(None),
                                IndexingJobRow.next_attempt_at <= observed_at,
                            ),
                            IndexedDocumentVersionRow.build_status.in_(
                                (IndexBuildStatus.QUEUED, IndexBuildStatus.FAILED)
                            ),
                        ),
                        and_(
                            IndexingJobRow.status == JobStatus.COMPLETED,
                            IndexingJobRow.claimed_by.is_(None),
                            IndexedDocumentVersionRow.build_status
                            == IndexBuildStatus.READY,
                        ),
                    ),
                    IndexedDocumentVersionRow.serving_status
                    == IndexServingStatus.CANDIDATE,
                )
                .order_by(
                    IndexingJobRow.next_attempt_at.asc().nullsfirst(),
                    IndexingJobRow.created_at,
                    IndexingJobRow.id,
                )
                .limit(1)
                .with_for_update(of=IndexingJobRow, skip_locked=True)
            )
        ).one_or_none()
        if row is None:
            return None
        job, target = row
        if job.status is JobStatus.QUEUED:
            job.status = JobStatus.RUNNING
            job.phase = "claimed"
            job.attempt += 1
        elif job.attempt == 0:
            job.attempt = 1
        job.claimed_by = worker_id
        job.claimed_at = observed_at
        job.heartbeat_at = observed_at
        job.next_attempt_at = None
        job.error_code = None
        job.error_detail = None
        job.updated_at = observed_at
        await self._session.flush()
        return IndexingLease(
            job_id=job.id,
            indexed_document_version_id=target.id,
            claimed_by=worker_id,
            attempt=job.attempt,
            claimed_at=observed_at,
        )

    async def heartbeat(
        self,
        lease: IndexingLease,
        *,
        observed_at: datetime,
    ) -> bool:
        self._ensure_active()
        updated = await self._session.scalar(
            update(IndexingJobRow)
            .where(*_owned_execution(lease, self._workspace_id))
            .values(heartbeat_at=observed_at, updated_at=observed_at)
            .returning(IndexingJobRow.id)
        )
        return updated is not None

    async def reschedule(
        self,
        lease: IndexingLease,
        *,
        observed_at: datetime,
        next_attempt_at: datetime,
        error_code: str,
        error_detail: dict[str, Any],
    ) -> bool:
        self._ensure_active()
        target_id = await self._session.scalar(
            update(IndexingJobRow)
            .where(
                *_owned(lease, self._workspace_id),
                IndexingJobRow.status.in_((JobStatus.RUNNING, JobStatus.FAILED)),
            )
            .values(
                status=JobStatus.QUEUED,
                phase="queued",
                claimed_by=None,
                claimed_at=None,
                heartbeat_at=None,
                next_attempt_at=next_attempt_at,
                error_code=error_code,
                error_detail=dict(error_detail),
                updated_at=observed_at,
            )
            .returning(IndexingJobRow.indexed_document_version_id)
        )
        if target_id is None:
            return False
        await self._mark_target_failed(
            target_id,
            error_code=error_code,
            error_detail=error_detail,
            observed_at=observed_at,
        )
        return True

    async def fail_owned(
        self,
        lease: IndexingLease,
        *,
        observed_at: datetime,
        error_code: str,
        error_detail: dict[str, Any],
    ) -> bool:
        self._ensure_active()
        target_id = await self._session.scalar(
            update(IndexingJobRow)
            .where(
                *_owned(lease, self._workspace_id),
                IndexingJobRow.status.in_((JobStatus.RUNNING, JobStatus.FAILED)),
            )
            .values(
                status=JobStatus.FAILED,
                phase="failed",
                claimed_by=None,
                claimed_at=None,
                heartbeat_at=None,
                next_attempt_at=None,
                error_code=error_code,
                error_detail=dict(error_detail),
                updated_at=observed_at,
            )
            .returning(IndexingJobRow.indexed_document_version_id)
        )
        if target_id is None:
            return False
        await self._mark_target_failed(
            target_id,
            error_code=error_code,
            error_detail=error_detail,
            observed_at=observed_at,
        )
        return True

    async def release_terminal(self, lease: IndexingLease) -> bool:
        self._ensure_active()
        updated = await self._session.scalar(
            update(IndexingJobRow)
            .where(
                *_owned(lease, self._workspace_id),
                IndexingJobRow.status.in_(
                    (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
                ),
            )
            .values(
                claimed_by=None,
                claimed_at=None,
                heartbeat_at=None,
                next_attempt_at=None,
                updated_at=func.now(),
            )
            .returning(IndexingJobRow.id)
        )
        return updated is not None

    async def reconcile_stale(
        self,
        *,
        stale_before: datetime,
        observed_at: datetime,
        max_attempts: int,
        retry_at_by_attempt: tuple[datetime, ...],
        limit: int,
    ) -> ReconciliationResult:
        self._ensure_active()
        if len(retry_at_by_attempt) < max_attempts:
            raise ValueError("retry schedule must cover every configured attempt")
        rows = (
            await self._session.execute(
                select(IndexingJobRow, IndexedDocumentVersionRow)
                .join(
                    IndexedDocumentVersionRow,
                    IndexedDocumentVersionRow.id
                    == IndexingJobRow.indexed_document_version_id,
                )
                .where(
                    IndexingJobRow.workspace_id == self._workspace_id,
                    IndexedDocumentVersionRow.workspace_id == self._workspace_id,
                    IndexingJobRow.status.in_(
                        (JobStatus.RUNNING, JobStatus.COMPLETED)
                    ),
                    IndexingJobRow.heartbeat_at <= stale_before,
                )
                .order_by(IndexingJobRow.heartbeat_at, IndexingJobRow.id)
                .limit(limit)
                .with_for_update(
                    of=(IndexingJobRow, IndexedDocumentVersionRow),
                    skip_locked=True,
                )
            )
        ).all()
        requeued = 0
        failed = 0
        for job, target in rows:
            if job.status is JobStatus.COMPLETED:
                job.claimed_by = None
                job.claimed_at = None
                job.heartbeat_at = None
                job.updated_at = observed_at
                if (
                    target.build_status is IndexBuildStatus.READY
                    and target.serving_status is IndexServingStatus.CANDIDATE
                ):
                    requeued += 1
                continue
            detail = {"attempt": job.attempt, "stale": True}
            target.build_status = IndexBuildStatus.FAILED
            target.error_code = ErrorCode.INDEXING_STALE_WORKER.value
            target.error_detail = detail
            target.updated_at = observed_at
            job.claimed_by = None
            job.claimed_at = None
            job.heartbeat_at = None
            job.error_code = ErrorCode.INDEXING_STALE_WORKER.value
            job.error_detail = detail
            job.updated_at = observed_at
            if job.attempt < max_attempts:
                job.status = JobStatus.QUEUED
                job.phase = "queued"
                job.next_attempt_at = retry_at_by_attempt[job.attempt - 1]
                requeued += 1
            else:
                job.status = JobStatus.FAILED
                job.phase = "failed"
                job.next_attempt_at = None
                failed += 1
        await self._session.flush()
        return ReconciliationResult(requeued=requeued, failed=failed)

    async def _mark_target_failed(
        self,
        target_id: UUID,
        *,
        error_code: str,
        error_detail: dict[str, Any],
        observed_at: datetime,
    ) -> None:
        await self._session.execute(
            update(IndexedDocumentVersionRow)
            .where(
                IndexedDocumentVersionRow.workspace_id == self._workspace_id,
                IndexedDocumentVersionRow.id == target_id,
                IndexedDocumentVersionRow.serving_status
                == IndexServingStatus.CANDIDATE,
                IndexedDocumentVersionRow.build_status != IndexBuildStatus.READY,
            )
            .values(
                build_status=IndexBuildStatus.FAILED,
                error_code=error_code,
                error_detail=dict(error_detail),
                updated_at=observed_at,
            )
        )

    async def promote(self, command: PromotionCommand) -> PromotionResult | None:
        """Conditionally switch one complete candidate using lifecycle lock order."""

        self._ensure_active()
        document_id = await self._session.scalar(
            select(IndexedDocumentVersionRow.document_id)
            .join(
                IndexingJobRow,
                IndexingJobRow.indexed_document_version_id
                == IndexedDocumentVersionRow.id,
            )
            .where(
                IndexedDocumentVersionRow.workspace_id == self._workspace_id,
                IndexingJobRow.workspace_id == self._workspace_id,
                IndexedDocumentVersionRow.id
                == command.indexed_document_version_id,
                IndexingJobRow.id == command.job_id,
            )
        )
        if document_id is None:
            return None

        document = await self._session.scalar(
            select(DocumentRow)
            .where(
                DocumentRow.workspace_id == self._workspace_id,
                DocumentRow.id == document_id,
            )
            .with_for_update(of=DocumentRow)
        )
        if document is None:
            return None
        knowledge_base = await self._session.scalar(
            select(KnowledgeBaseRow)
            .where(
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.id == document.kb_id,
            )
            .with_for_update(of=KnowledgeBaseRow)
        )
        if knowledge_base is None:
            return None

        locked = (
            await self._session.execute(
                select(IndexingJobRow, IndexedDocumentVersionRow)
                .join(
                    IndexedDocumentVersionRow,
                    IndexedDocumentVersionRow.id
                    == IndexingJobRow.indexed_document_version_id,
                )
                .where(
                    IndexingJobRow.workspace_id == self._workspace_id,
                    IndexedDocumentVersionRow.workspace_id == self._workspace_id,
                    IndexingJobRow.id == command.job_id,
                    IndexedDocumentVersionRow.id
                    == command.indexed_document_version_id,
                    IndexedDocumentVersionRow.document_id == document.id,
                )
                .with_for_update(
                    of=(IndexingJobRow, IndexedDocumentVersionRow)
                )
            )
        ).one_or_none()
        if locked is None:
            return None
        job, target = locked

        associations = tuple(
            (
                await self._session.execute(
                    select(IndexedDocumentVersionRow)
                    .where(
                        IndexedDocumentVersionRow.workspace_id
                        == self._workspace_id,
                        IndexedDocumentVersionRow.document_id == document.id,
                        IndexedDocumentVersionRow.index_revision_id
                        == target.index_revision_id,
                    )
                    .order_by(IndexedDocumentVersionRow.id)
                    .with_for_update(of=IndexedDocumentVersionRow)
                )
            ).scalars()
        )

        if target.serving_status is IndexServingStatus.SERVING:
            return _promotion_result(
                command,
                PromotionStatus.SERVING,
                PromotionReason.ALREADY_SERVING,
            )
        if target.serving_status is IndexServingStatus.RETIRED:
            return _promotion_result(
                command,
                PromotionStatus.RETIRED,
                PromotionReason.ALREADY_RETIRED,
            )
        if target.build_status is not IndexBuildStatus.READY:
            return _promotion_result(
                command,
                PromotionStatus.NOT_READY,
                PromotionReason.NOT_READY,
            )
        if job.status is not JobStatus.COMPLETED:
            return _promotion_result(
                command,
                PromotionStatus.NOT_READY,
                PromotionReason.JOB_INCOMPLETE,
            )

        revision_status = await self._session.scalar(
            select(IndexRevisionRow.status).where(
                IndexRevisionRow.workspace_id == self._workspace_id,
                IndexRevisionRow.kb_id == document.kb_id,
                IndexRevisionRow.id == target.index_revision_id,
            )
        )
        later_change = bool(
            await self._session.scalar(
                select(
                    exists().where(
                        SourceChangeRow.workspace_id == self._workspace_id,
                        SourceChangeRow.kb_id == document.kb_id,
                        SourceChangeRow.document_id == document.id,
                        SourceChangeRow.source_change_seq
                        > target.source_change_seq,
                    )
                )
            )
        )
        retirement_reason: PromotionReason | None = None
        if document.deleted_at is not None:
            retirement_reason = PromotionReason.DOCUMENT_DELETED
        elif (
            knowledge_base.active_index_revision_id != target.index_revision_id
            or revision_status is not IndexRevisionStatus.ACTIVE
        ):
            retirement_reason = PromotionReason.REVISION_INACTIVE
        elif document.current_version_id != target.document_version_id:
            retirement_reason = PromotionReason.SUPERSEDED
        elif later_change:
            retirement_reason = PromotionReason.LATER_SOURCE_CHANGE

        now = datetime.now(UTC)
        if retirement_reason is not None:
            target.serving_status = IndexServingStatus.RETIRED
            target.updated_at = now
            await self._session.flush()
            return _promotion_result(
                command,
                PromotionStatus.RETIRED,
                retirement_reason,
            )

        previous_serving: UUID | None = None
        for association in associations:
            if (
                association.id != target.id
                and association.serving_status is IndexServingStatus.SERVING
            ):
                association.serving_status = IndexServingStatus.RETIRED
                association.updated_at = now
                previous_serving = association.id
        await self._session.flush()
        target.serving_status = IndexServingStatus.SERVING
        target.updated_at = now
        await self._session.flush()
        return _promotion_result(
            command,
            PromotionStatus.SERVING,
            PromotionReason.PROMOTED,
            previous_serving_target_id=previous_serving,
        )

    async def prepare(self, command: IndexingCommand) -> IndexingTarget | None:
        self._ensure_active()
        row = await self._load(command, lock=True)
        if row is None:
            return None
        job, target, version, revision, embedding, knowledge_base = row
        if (
            job.status is JobStatus.COMPLETED
            and target.build_status is IndexBuildStatus.READY
        ):
            return _target(row, already_complete=True)
        if job.status is JobStatus.CANCELLED or target.serving_status is IndexServingStatus.RETIRED:
            raise IndexingCancelled
        if target.build_status is IndexBuildStatus.READY or job.status is JobStatus.COMPLETED:
            raise _execution_error(
                ErrorCode.INDEX_TARGET_INVALID,
                IndexingPhase.SOURCE_READ,
                "terminal_state",
            )
        if version.source_status is not DocumentSourceStatus.AVAILABLE:
            raise _execution_error(
                ErrorCode.SOURCE_NOT_AVAILABLE,
                IndexingPhase.SOURCE_READ,
                "source_status",
            )
        if (
            revision.status is not IndexRevisionStatus.ACTIVE
            or knowledge_base.active_index_revision_id != revision.id
        ):
            raise _execution_error(
                ErrorCode.INDEX_REVISION_INCOMPATIBLE,
                IndexingPhase.SOURCE_READ,
                "active_revision",
            )
        target.build_status = IndexBuildStatus.PROCESSING
        target.error_code = None
        target.error_detail = None
        job.status = JobStatus.RUNNING
        job.phase = IndexingPhase.SOURCE_READ.value
        job.error_code = None
        job.error_detail = None
        await self._session.flush()
        return _target(row)

    async def set_phase(self, command: IndexingCommand, phase: IndexingPhase) -> bool:
        self._ensure_active()
        row = await self._load(command, lock=True)
        if row is None:
            return False
        job, target, *_ = row
        if not _is_writable(job, target):
            return False
        job.phase = phase.value
        await self._session.flush()
        return True

    async def upsert_batch(
        self,
        command: IndexingCommand,
        chunks: tuple[IndexChunkWrite, ...],
        vectors: tuple[VectorRecordWrite, ...],
    ) -> bool:
        self._ensure_active()
        if len(chunks) != len(vectors):
            raise ValueError("every persisted chunk requires one vector")
        row = await self._load(command, lock=True)
        if row is None:
            return False
        job, target, *_ = row
        if not _is_writable(job, target):
            return False
        if not chunks:
            return True

        chunk_values = [
            {
                "id": chunk.id,
                "workspace_id": self._workspace_id,
                "kb_id": target.kb_id,
                "indexed_document_version_id": target.id,
                "ordinal": chunk.ordinal,
                "content": chunk.content,
                "content_hash": chunk.content_hash,
                "token_count": chunk.token_count,
                "source_location": chunk.source_location,
                "hierarchy": chunk.hierarchy,
                "source_metadata": chunk.source_metadata,
            }
            for chunk in chunks
        ]
        chunk_insert = pg_insert(IndexChunkRow).values(chunk_values)
        stored_chunks = (
            await self._session.execute(
                chunk_insert.on_conflict_do_update(
                    index_elements=["indexed_document_version_id", "ordinal"],
                    set_={
                        "content": chunk_insert.excluded.content,
                        "content_hash": chunk_insert.excluded.content_hash,
                        "token_count": chunk_insert.excluded.token_count,
                        "source_location": chunk_insert.excluded.source_location,
                        "hierarchy": chunk_insert.excluded.hierarchy,
                        "source_metadata": chunk_insert.excluded.source_metadata,
                    },
                    where=and_(
                        IndexChunkRow.id == chunk_insert.excluded.id,
                        IndexChunkRow.content_hash
                        == chunk_insert.excluded.content_hash,
                    ),
                ).returning(IndexChunkRow.id, IndexChunkRow.ordinal)
            )
        ).all()
        expected_chunk_ids = {chunk.ordinal: chunk.id for chunk in chunks}
        if {item.ordinal: item.id for item in stored_chunks} != expected_chunk_ids:
            raise _execution_error(
                ErrorCode.INDEX_PERSISTENCE_FAILED,
                IndexingPhase.PERSISTING,
                "stable_chunk_key",
            )

        vector_values = [
            {
                "id": vector.id,
                "workspace_id": self._workspace_id,
                "kb_id": target.kb_id,
                "index_chunk_id": vector.index_chunk_id,
                "embedding_space_id": vector.embedding_space_id,
                "embedding": list(vector.embedding),
            }
            for vector in vectors
        ]
        vector_insert = pg_insert(VectorRecordRow).values(vector_values)
        stored_vectors = (
            await self._session.execute(
                vector_insert.on_conflict_do_update(
                    index_elements=["index_chunk_id"],
                    set_={"embedding": vector_insert.excluded.embedding},
                    where=(
                        VectorRecordRow.embedding_space_id
                        == vector_insert.excluded.embedding_space_id
                    ),
                ).returning(VectorRecordRow.id, VectorRecordRow.index_chunk_id)
            )
        ).all()
        expected_vector_ids = {
            vector.index_chunk_id: vector.id for vector in vectors
        }
        if {item.index_chunk_id: item.id for item in stored_vectors} != expected_vector_ids:
            raise _execution_error(
                ErrorCode.INDEX_PERSISTENCE_FAILED,
                IndexingPhase.PERSISTING,
                "stable_vector_key",
            )
        job.phase = IndexingPhase.PERSISTING.value
        await self._session.flush()
        return True

    async def complete(self, command: IndexingCommand, *, expected_chunks: int) -> bool:
        self._ensure_active()
        row = await self._load(command, lock=True)
        if row is None:
            return False
        job, target, version, revision, _embedding_row, knowledge_base = row
        embedding = row[4]
        if job.status is JobStatus.COMPLETED and target.build_status is IndexBuildStatus.READY:
            return True
        if not _is_writable(job, target):
            return False
        if version.source_status is not DocumentSourceStatus.AVAILABLE:
            raise _execution_error(
                ErrorCode.SOURCE_NOT_AVAILABLE,
                IndexingPhase.VALIDATING,
                "source_status",
            )
        if (
            revision.status is not IndexRevisionStatus.ACTIVE
            or knowledge_base.active_index_revision_id != revision.id
        ):
            raise _execution_error(
                ErrorCode.INDEX_REVISION_INCOMPATIBLE,
                IndexingPhase.VALIDATING,
                "active_revision",
            )
        records = (
            await self._session.execute(
                select(
                    IndexChunkRow.ordinal,
                    IndexChunkRow.id.label("chunk_id"),
                    VectorRecordRow.id.label("vector_id"),
                    VectorRecordRow.embedding_space_id,
                )
                .outerjoin(
                    VectorRecordRow,
                    VectorRecordRow.index_chunk_id == IndexChunkRow.id,
                )
                .where(IndexChunkRow.indexed_document_version_id == target.id)
                .order_by(IndexChunkRow.ordinal)
            )
        ).all()
        valid = (
            len(records) == expected_chunks
            and tuple(record.ordinal for record in records) == tuple(range(expected_chunks))
            and all(
                record.chunk_id == stable_chunk_id(target.id, record.ordinal)
                and record.vector_id is not None
                and record.embedding_space_id == embedding.id
                for record in records
            )
        )
        if not valid:
            raise IndexingExecutionError(
                ErrorCode.INDEX_INCOMPLETE,
                phase=IndexingPhase.VALIDATING,
                diagnostic={
                    "expected_chunks": expected_chunks,
                    "observed_chunks": len(records),
                    "observed_vectors": sum(
                        record.vector_id is not None for record in records
                    ),
                },
            )
        target.build_status = IndexBuildStatus.READY
        target.error_code = None
        target.error_detail = None
        job.status = JobStatus.COMPLETED
        job.phase = IndexingPhase.COMPLETED.value
        job.error_code = None
        job.error_detail = None
        await self._session.flush()
        return True

    async def count_chunks(self, command: IndexingCommand) -> int:
        self._ensure_active()
        count = await self._session.scalar(
            select(func.count(IndexChunkRow.id))
            .join(
                IndexedDocumentVersionRow,
                IndexedDocumentVersionRow.id
                == IndexChunkRow.indexed_document_version_id,
            )
            .join(
                IndexingJobRow,
                IndexingJobRow.indexed_document_version_id
                == IndexedDocumentVersionRow.id,
            )
            .where(
                IndexingJobRow.workspace_id == self._workspace_id,
                IndexingJobRow.id == command.job_id,
                IndexedDocumentVersionRow.id
                == command.indexed_document_version_id,
            )
        )
        return int(count or 0)

    async def fail(
        self,
        command: IndexingCommand,
        *,
        phase: IndexingPhase,
        error_code: str,
        error_detail: dict[str, Any],
    ) -> bool:
        self._ensure_active()
        row = await self._load(command, lock=True)
        if row is None:
            return False
        job, target, *_ = row
        if (
            job.status in (JobStatus.COMPLETED, JobStatus.CANCELLED)
            or target.build_status is IndexBuildStatus.READY
            or target.serving_status is IndexServingStatus.RETIRED
        ):
            return False
        target.build_status = IndexBuildStatus.FAILED
        target.error_code = error_code
        target.error_detail = dict(error_detail)
        job.status = JobStatus.FAILED
        job.phase = phase.value
        job.error_code = error_code
        job.error_detail = dict(error_detail)
        await self._session.flush()
        return True

    async def _load(self, command: IndexingCommand, *, lock: bool):
        statement = (
            select(
                IndexingJobRow,
                IndexedDocumentVersionRow,
                DocumentVersionRow,
                IndexRevisionRow,
                EmbeddingSpaceRow,
                KnowledgeBaseRow,
            )
            .join(
                IndexedDocumentVersionRow,
                IndexedDocumentVersionRow.id
                == IndexingJobRow.indexed_document_version_id,
            )
            .join(
                DocumentVersionRow,
                DocumentVersionRow.id
                == IndexedDocumentVersionRow.document_version_id,
            )
            .join(
                IndexRevisionRow,
                IndexRevisionRow.id == IndexedDocumentVersionRow.index_revision_id,
            )
            .join(
                EmbeddingSpaceRow,
                EmbeddingSpaceRow.id == IndexRevisionRow.embedding_space_id,
            )
            .join(
                KnowledgeBaseRow,
                KnowledgeBaseRow.id == IndexedDocumentVersionRow.kb_id,
            )
            .where(
                IndexingJobRow.workspace_id == self._workspace_id,
                IndexedDocumentVersionRow.workspace_id == self._workspace_id,
                IndexingJobRow.id == command.job_id,
                IndexedDocumentVersionRow.id
                == command.indexed_document_version_id,
            )
        )
        if lock:
            statement = statement.with_for_update(
                of=(IndexingJobRow, IndexedDocumentVersionRow)
            )
        return (await self._session.execute(statement)).one_or_none()


def _is_writable(job: IndexingJobRow, target: IndexedDocumentVersionRow) -> bool:
    return (
        job.status is JobStatus.RUNNING
        and target.build_status is IndexBuildStatus.PROCESSING
        and target.serving_status is IndexServingStatus.CANDIDATE
    )


def _target(row, *, already_complete: bool = False) -> IndexingTarget:
    job, target, version, revision, embedding, _knowledge_base = row
    return IndexingTarget(
        job_id=job.id,
        indexed_document_version_id=target.id,
        workspace_id=target.workspace_id,
        kb_id=target.kb_id,
        document_id=target.document_id,
        document_version_id=target.document_version_id,
        index_revision_id=target.index_revision_id,
        embedding_space_id=embedding.id,
        source_change_seq=target.source_change_seq,
        storage_uri=version.storage_uri,
        checksum_sha256=version.checksum_sha256,
        size_bytes=version.size_bytes,
        original_filename=version.original_filename,
        media_type=version.media_type,
        parser_config=dict(revision.parser_config),
        chunking_config=dict(revision.chunking_config),
        embedding_space=_embedding(embedding),
        already_complete=already_complete,
    )


def _embedding(row: EmbeddingSpaceRow) -> EmbeddingSpaceDefinition:
    return EmbeddingSpaceDefinition(
        provider_identity=row.provider_identity,
        endpoint_identity=row.endpoint_identity,
        requested_model=row.requested_model,
        resolved_model=row.resolved_model,
        model_version=row.model_version,
        deployment_revision=row.deployment_revision,
        dimension=row.dimension,
        distance_metric=row.distance_metric,
        vector_data_type=row.vector_data_type,
        normalization=row.normalization,
        configuration_fingerprint=row.configuration_fingerprint,
        tokenizer_fingerprint=row.tokenizer_fingerprint,
        compatibility_fingerprint=row.compatibility_fingerprint,
    )


def _execution_error(
    code: ErrorCode,
    phase: IndexingPhase,
    check: str,
) -> IndexingExecutionError:
    return IndexingExecutionError(code, phase=phase, diagnostic={"check": check})


def _promotion_result(
    command: PromotionCommand,
    status: PromotionStatus,
    reason: PromotionReason,
    *,
    previous_serving_target_id: UUID | None = None,
) -> PromotionResult:
    return PromotionResult(
        job_id=command.job_id,
        indexed_document_version_id=command.indexed_document_version_id,
        status=status,
        reason=reason,
        previous_serving_target_id=previous_serving_target_id,
    )


def _owned(lease: IndexingLease, workspace_id: UUID) -> tuple[Any, ...]:
    return (
        IndexingJobRow.workspace_id == workspace_id,
        IndexingJobRow.id == lease.job_id,
        IndexingJobRow.indexed_document_version_id
        == lease.indexed_document_version_id,
        IndexingJobRow.claimed_by == lease.claimed_by,
        IndexingJobRow.attempt == lease.attempt,
    )


def _owned_execution(lease: IndexingLease, workspace_id: UUID) -> tuple[Any, ...]:
    return (
        *_owned(lease, workspace_id),
        IndexingJobRow.status.in_(
            (JobStatus.RUNNING, JobStatus.FAILED, JobStatus.COMPLETED)
        ),
    )
