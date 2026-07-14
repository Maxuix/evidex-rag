"""SQLAlchemy persistence for idempotent chunk and pgvector writes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from rag_kb.db.models import (
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
    VectorRecord as VectorRecordRow,
)
from rag_kb.domain import (
    EmbeddingSpaceDefinition,
    ErrorCode,
    IndexChunkWrite,
    IndexingCancelled,
    IndexingCommand,
    IndexingExecutionError,
    IndexingPhase,
    IndexingTarget,
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
