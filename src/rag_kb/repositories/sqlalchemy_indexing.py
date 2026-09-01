"""SQLAlchemy persistence for idempotent chunk and pgvector writes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import and_, delete, exists, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from rag_kb.db.models import (
    Document as DocumentRow,
    DocumentSourceStatus,
    DocumentVersion as DocumentVersionRow,
    EmbeddingSpace as EmbeddingSpaceRow,
    IndexArtifactManifest as IndexArtifactManifestRow,
    IndexAsset as IndexAssetRow,
    IndexBuildStatus,
    IndexChunk as IndexChunkRow,
    IndexChunkLexical as IndexChunkLexicalRow,
    IndexLexicalManifest as IndexLexicalManifestRow,
    IndexChunkAssetRelation as IndexChunkAssetRelationRow,
    IndexChunkPlan as IndexChunkPlanRow,
    IndexedDocumentVersion as IndexedDocumentVersionRow,
    IndexingJob as IndexingJobRow,
    IndexRevision as IndexRevisionRow,
    IndexRevisionEmbeddingSpace as IndexRevisionEmbeddingSpaceRow,
    IndexRevisionStatus,
    IndexServingStatus,
    JobStatus,
    KnowledgeBase as KnowledgeBaseRow,
    ModelProfileRevision as ModelProfileRevisionRow,
    SourceChange as SourceChangeRow,
    VectorRecord as VectorRecordRow,
)
from rag_kb.domain import (
    EmbeddingSpaceDefinition,
    ErrorCode,
    IndexChunkWrite,
    IndexChunkLexicalWrite,
    IndexLexicalManifest,
    IndexChunkAssetRelationSnapshot,
    IndexChunkAssetRelationWrite,
    IndexChunkPlan,
    IndexArtifactManifest,
    IndexAssetWrite,
    IndexAssetSnapshot,
    IndexCleanupResult,
    IndexingCancelled,
    IndexingCommand,
    IndexingExecutionError,
    IndexingLease,
    IndexingJobSnapshot,
    IndexingPhase,
    IndexingTarget,
    Page,
    PromotionCommand,
    PromotionReason,
    PromotionResult,
    PromotionStatus,
    RetiredIndexTargetAssets,
    ReconciliationResult,
    ResourceStateConflictError,
    VectorRecordWrite,
)
from rag_kb.document_processing.profiles import profile_fingerprint
from rag_kb.document_processing.lexical import (
    LEXICAL_ANALYZER_VERSION,
    lexical_manifest_hash,
)


class SqlAlchemyIndexingRepository:
    def __init__(
        self,
        session: AsyncSession,
        workspace_id: UUID,
    ) -> None:
        self._session = session
        self._workspace_id = workspace_id

    async def get_asset(self, asset_id: UUID) -> IndexAssetSnapshot | None:
        row = await self._session.scalar(
            select(IndexAssetRow)
            .join(
                KnowledgeBaseRow,
                KnowledgeBaseRow.id == IndexAssetRow.kb_id,
            )
            .where(
                IndexAssetRow.workspace_id == self._workspace_id,
                IndexAssetRow.id == asset_id,
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.deleted_at.is_(None),
            )
        )
        if row is None:
            return None
        return IndexAssetSnapshot(
            id=row.id,
            workspace_id=row.workspace_id,
            kb_id=row.kb_id,
            document_id=row.document_id,
            document_version_id=row.document_version_id,
            indexed_document_version_id=row.indexed_document_version_id,
            storage_uri=row.storage_uri,
            media_type=row.media_type,
            checksum_sha256=row.checksum_sha256,
        )

    async def list_retired_target_assets(
        self, *, data_before: datetime, limit: int
    ) -> tuple[RetiredIndexTargetAssets, ...]:
        if limit < 1:
            raise ValueError("retired target limit must be positive")
        target_ids = tuple(
            (
                await self._session.execute(
                    select(IndexedDocumentVersionRow.id)
                    .where(
                        IndexedDocumentVersionRow.workspace_id
                        == self._workspace_id,
                        IndexedDocumentVersionRow.serving_status
                        == IndexServingStatus.RETIRED,
                        IndexedDocumentVersionRow.updated_at <= data_before,
                        or_(
                            exists(
                                select(IndexChunkRow.id).where(
                                    IndexChunkRow.workspace_id
                                    == self._workspace_id,
                                    IndexChunkRow.indexed_document_version_id
                                    == IndexedDocumentVersionRow.id,
                                )
                            ),
                            exists(
                                select(IndexAssetRow.id).where(
                                    IndexAssetRow.workspace_id
                                    == self._workspace_id,
                                    IndexAssetRow.indexed_document_version_id
                                    == IndexedDocumentVersionRow.id,
                                )
                            ),
                            exists(
                                select(IndexChunkAssetRelationRow.id).where(
                                    IndexChunkAssetRelationRow.workspace_id
                                    == self._workspace_id,
                                    IndexChunkAssetRelationRow.indexed_document_version_id
                                    == IndexedDocumentVersionRow.id,
                                )
                            ),
                            exists(
                                select(
                                    IndexArtifactManifestRow.indexed_document_version_id
                                ).where(
                                    IndexArtifactManifestRow.indexed_document_version_id
                                    == IndexedDocumentVersionRow.id
                                )
                            ),
                            exists(
                                select(IndexChunkPlanRow.indexed_document_version_id)
                                .where(
                                    IndexChunkPlanRow.indexed_document_version_id
                                    == IndexedDocumentVersionRow.id
                                )
                            ),
                        ),
                    )
                    .order_by(
                        IndexedDocumentVersionRow.updated_at,
                        IndexedDocumentVersionRow.id,
                    )
                    .limit(limit)
                )
            ).scalars()
        )
        if not target_ids:
            return ()
        rows = (
            await self._session.scalars(
                select(IndexAssetRow)
                .where(
                    IndexAssetRow.workspace_id == self._workspace_id,
                    IndexAssetRow.indexed_document_version_id.in_(target_ids),
                )
                .order_by(
                    IndexAssetRow.indexed_document_version_id,
                    IndexAssetRow.created_at,
                    IndexAssetRow.id,
                )
            )
        ).all()
        assets_by_target: dict[UUID, list[IndexAssetSnapshot]] = {
            target_id: [] for target_id in target_ids
        }
        for row in rows:
            assets_by_target[row.indexed_document_version_id].append(
                IndexAssetSnapshot(
                    id=row.id,
                    workspace_id=row.workspace_id,
                    kb_id=row.kb_id,
                    document_id=row.document_id,
                    document_version_id=row.document_version_id,
                    indexed_document_version_id=row.indexed_document_version_id,
                    storage_uri=row.storage_uri,
                    media_type=row.media_type,
                    checksum_sha256=row.checksum_sha256,
                )
            )
        return tuple(
            RetiredIndexTargetAssets(
                indexed_document_version_id=target_id,
                assets=tuple(assets_by_target[target_id]),
            )
            for target_id in target_ids
        )

    async def list_relations(
        self,
        *,
        kb_id: UUID,
        index_revision_id: UUID,
        chunk_ids: tuple[UUID, ...] = (),
        asset_ids: tuple[UUID, ...] = (),
        limit: int = 500,
    ) -> tuple[IndexChunkAssetRelationSnapshot, ...]:
        if not chunk_ids and not asset_ids:
            return ()
        if not 1 <= limit <= 2_000:
            raise ValueError("relation hydration limit must be between 1 and 2000")
        selectors = []
        if chunk_ids:
            selectors.extend(
                (
                    IndexChunkAssetRelationRow.chunk_id.in_(chunk_ids),
                    IndexChunkAssetRelationRow.visual_unit_id.in_(chunk_ids),
                )
            )
        if asset_ids:
            selectors.append(IndexChunkAssetRelationRow.asset_id.in_(asset_ids))
        parent_chunk = aliased(IndexChunkRow)
        visual_chunk = aliased(IndexChunkRow)
        rows = (
            await self._session.execute(
                select(
                    IndexChunkAssetRelationRow,
                    IndexedDocumentVersionRow.index_revision_id,
                    IndexedDocumentVersionRow.document_id,
                    IndexedDocumentVersionRow.document_version_id,
                    parent_chunk,
                    visual_chunk,
                    IndexAssetRow,
                )
                .join(
                    IndexedDocumentVersionRow,
                    IndexedDocumentVersionRow.id
                    == IndexChunkAssetRelationRow.indexed_document_version_id,
                )
                .join(
                    DocumentRow,
                    DocumentRow.id == IndexedDocumentVersionRow.document_id,
                )
                .join(
                    DocumentVersionRow,
                    and_(
                        DocumentVersionRow.id
                        == IndexedDocumentVersionRow.document_version_id,
                        DocumentVersionRow.document_id
                        == IndexedDocumentVersionRow.document_id,
                        DocumentVersionRow.kb_id == IndexedDocumentVersionRow.kb_id,
                        DocumentVersionRow.workspace_id
                        == IndexedDocumentVersionRow.workspace_id,
                    ),
                )
                .join(
                    KnowledgeBaseRow,
                    KnowledgeBaseRow.id == IndexedDocumentVersionRow.kb_id,
                )
                .join(
                    parent_chunk,
                    and_(
                        parent_chunk.indexed_document_version_id
                        == IndexChunkAssetRelationRow.indexed_document_version_id,
                        parent_chunk.id == IndexChunkAssetRelationRow.chunk_id,
                    ),
                )
                .join(
                    visual_chunk,
                    and_(
                        visual_chunk.indexed_document_version_id
                        == IndexChunkAssetRelationRow.indexed_document_version_id,
                        visual_chunk.id
                        == IndexChunkAssetRelationRow.visual_unit_id,
                    ),
                )
                .join(
                    IndexAssetRow,
                    and_(
                        IndexAssetRow.indexed_document_version_id
                        == IndexChunkAssetRelationRow.indexed_document_version_id,
                        IndexAssetRow.id == IndexChunkAssetRelationRow.asset_id,
                    ),
                )
                .where(
                    IndexChunkAssetRelationRow.workspace_id == self._workspace_id,
                    IndexChunkAssetRelationRow.kb_id == kb_id,
                    IndexedDocumentVersionRow.workspace_id == self._workspace_id,
                    IndexedDocumentVersionRow.kb_id == kb_id,
                    IndexedDocumentVersionRow.index_revision_id == index_revision_id,
                    KnowledgeBaseRow.workspace_id == self._workspace_id,
                    KnowledgeBaseRow.id == kb_id,
                    KnowledgeBaseRow.deleted_at.is_(None),
                    KnowledgeBaseRow.active_index_revision_id == index_revision_id,
                    IndexedDocumentVersionRow.build_status == IndexBuildStatus.READY,
                    IndexedDocumentVersionRow.serving_status == IndexServingStatus.SERVING,
                    DocumentRow.workspace_id == self._workspace_id,
                    DocumentRow.kb_id == kb_id,
                    DocumentRow.deleted_at.is_(None),
                    parent_chunk.excluded_at.is_(None),
                    visual_chunk.excluded_at.is_(None),
                    DocumentVersionRow.source_status
                    == DocumentSourceStatus.AVAILABLE,
                    or_(*selectors),
                )
                .order_by(
                    IndexChunkAssetRelationRow.ordinal,
                    IndexChunkAssetRelationRow.id,
                )
                .limit(limit)
            )
        ).all()
        return tuple(
            IndexChunkAssetRelationSnapshot(
                id=row.id,
                workspace_id=row.workspace_id,
                kb_id=row.kb_id,
                indexed_document_version_id=row.indexed_document_version_id,
                index_revision_id=revision_id,
                chunk_id=row.chunk_id,
                visual_unit_id=row.visual_unit_id,
                asset_id=row.asset_id,
                relation_type=row.relation_type,
                confidence_micros=row.confidence_micros,
                figure_label=row.figure_label,
                ordinal=row.ordinal,
                provenance=row.provenance,
                evidence_group_key=row.evidence_group_key,
                document_id=document_id,
                document_version_id=document_version_id,
                chunk_ordinal=parent.ordinal,
                chunk_content=parent.content,
                chunk_modality=parent.modality,
                chunk_source_location=parent.source_location,
                chunk_hierarchy=parent.hierarchy,
                chunk_source_metadata=parent.source_metadata,
                visual_ordinal=visual.ordinal,
                visual_content=visual.content,
                visual_modality=visual.modality,
                visual_source_location=visual.source_location,
                visual_hierarchy=visual.hierarchy,
                visual_source_metadata=visual.source_metadata,
                asset_media_type=asset.media_type,
                asset_checksum_sha256=asset.checksum_sha256,
                asset_width=asset.width,
                asset_height=asset.height,
            )
            for (
                row,
                revision_id,
                document_id,
                document_version_id,
                parent,
                visual,
                asset,
            ) in rows
        )

    async def get_job(self, job_id: UUID) -> IndexingJobSnapshot | None:
        row = await self._job_row(job_id)
        return _job_snapshot(row) if row is not None else None

    async def list_jobs(
        self,
        *,
        kb_id: UUID,
        limit: int,
        after: tuple[str, ...] | None,
    ) -> Page[IndexingJobSnapshot]:
        statement = (
            select(
                IndexingJobRow,
                IndexedDocumentVersionRow,
                DocumentRow,
                DocumentVersionRow,
                IndexRevisionRow,
                KnowledgeBaseRow,
            )
            .join(
                IndexedDocumentVersionRow,
                IndexedDocumentVersionRow.id
                == IndexingJobRow.indexed_document_version_id,
            )
            .join(DocumentRow, DocumentRow.id == IndexedDocumentVersionRow.document_id)
            .join(
                DocumentVersionRow,
                DocumentVersionRow.id
                == IndexedDocumentVersionRow.document_version_id,
            )
            .join(
                IndexRevisionRow,
                IndexRevisionRow.id == IndexedDocumentVersionRow.index_revision_id,
            )
            .join(KnowledgeBaseRow, KnowledgeBaseRow.id == IndexingJobRow.kb_id)
            .where(
                IndexingJobRow.workspace_id == self._workspace_id,
                IndexedDocumentVersionRow.workspace_id == self._workspace_id,
                IndexingJobRow.kb_id == kb_id,
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.deleted_at.is_(None),
            )
        )
        if after is not None:
            if len(after) != 2:
                raise ValueError("job cursor must contain created_at and id")
            created_at, job_id = datetime.fromisoformat(after[0]), UUID(after[1])
            statement = statement.where(
                or_(
                    IndexingJobRow.created_at < created_at,
                    and_(
                        IndexingJobRow.created_at == created_at,
                        IndexingJobRow.id < job_id,
                    ),
                )
            )
        rows = (
            await self._session.execute(
                statement.order_by(
                    IndexingJobRow.created_at.desc(),
                    IndexingJobRow.id.desc(),
                ).limit(limit + 1)
            )
        ).all()
        has_more = len(rows) > limit
        rows = rows[:limit]
        items = tuple(_job_snapshot(row) for row in rows)
        next_values = (
            (items[-1].created_at.isoformat(), str(items[-1].job_id))
            if has_more and items
            else None
        )
        return Page(items=items, next_values=next_values)

    async def retry_failed(
        self,
        job_id: UUID,
        *,
        observed_at: datetime,
    ) -> IndexingJobSnapshot | None:
        identity = (
            await self._session.execute(
                select(
                    IndexingJobRow.kb_id,
                    IndexedDocumentVersionRow.document_id,
                )
                .join(
                    IndexedDocumentVersionRow,
                    IndexedDocumentVersionRow.id
                    == IndexingJobRow.indexed_document_version_id,
                )
                .where(
                    IndexingJobRow.workspace_id == self._workspace_id,
                    IndexingJobRow.id == job_id,
                )
            )
        ).one_or_none()
        if identity is None:
            return None
        kb_id, document_id = identity
        document = await self._session.scalar(
            select(DocumentRow)
            .where(
                DocumentRow.workspace_id == self._workspace_id,
                DocumentRow.id == document_id,
            )
            .with_for_update()
        )
        knowledge_base = await self._session.scalar(
            select(KnowledgeBaseRow)
            .where(
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.id == kb_id,
                KnowledgeBaseRow.deleted_at.is_(None),
            )
            .with_for_update()
        )
        locked = (
            await self._session.execute(
                select(
                    IndexingJobRow,
                    IndexedDocumentVersionRow,
                    DocumentVersionRow,
                    IndexRevisionRow,
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
                    IndexRevisionRow.id
                    == IndexedDocumentVersionRow.index_revision_id,
                )
                .where(
                    IndexingJobRow.workspace_id == self._workspace_id,
                    IndexingJobRow.id == job_id,
                )
                .with_for_update(
                    of=(IndexingJobRow, IndexedDocumentVersionRow)
                )
            )
        ).one_or_none()
        if locked is None or document is None or knowledge_base is None:
            return None
        job, target, version, revision = locked
        eligible = (
            job.status is JobStatus.FAILED
            and target.serving_status is IndexServingStatus.CANDIDATE
            and target.build_status is IndexBuildStatus.FAILED
            and document.deleted_at is None
            and document.current_version_id == target.document_version_id
            and version.source_status is DocumentSourceStatus.AVAILABLE
            and revision.status is IndexRevisionStatus.ACTIVE
            and knowledge_base.deleted_at is None
            and knowledge_base.active_index_revision_id == target.index_revision_id
        )
        if not eligible:
            raise ResourceStateConflictError("indexing job is not retryable")
        job.status = JobStatus.QUEUED
        job.phase = "queued"
        job.attempt = 0
        job.claimed_at = None
        job.heartbeat_at = None
        job.continuation_pending = False
        job.next_attempt_at = observed_at
        job.error_code = None
        job.error_detail = None
        job.updated_at = observed_at
        target.build_status = IndexBuildStatus.QUEUED
        target.error_code = None
        target.error_detail = None
        target.updated_at = observed_at
        await self._session.flush()
        row = await self._job_row(job_id)
        assert row is not None
        return _job_snapshot(row)

    async def cleanup_retired(
        self,
        *,
        target_ids: tuple[UUID, ...],
        data_before: datetime,
        tasks_before: datetime,
        limit: int,
    ) -> IndexCleanupResult:
        if limit < 1:
            raise ValueError("retired cleanup limit must be positive")
        requested_target_ids = tuple(dict.fromkeys(target_ids))
        if len(requested_target_ids) > limit:
            raise ValueError("approved retired targets exceed cleanup limit")
        targets = tuple(
            (
                await self._session.execute(
                    select(IndexedDocumentVersionRow.id)
                    .where(
                        IndexedDocumentVersionRow.workspace_id
                        == self._workspace_id,
                        IndexedDocumentVersionRow.id.in_(requested_target_ids),
                        IndexedDocumentVersionRow.serving_status
                        == IndexServingStatus.RETIRED,
                        IndexedDocumentVersionRow.updated_at <= data_before,
                    )
                    .order_by(
                        IndexedDocumentVersionRow.updated_at,
                        IndexedDocumentVersionRow.id,
                    )
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            ).scalars()
        )
        chunk_ids: tuple[UUID, ...] = ()
        if targets:
            chunk_ids = tuple(
                (
                    await self._session.execute(
                        select(IndexChunkRow.id).where(
                            IndexChunkRow.workspace_id == self._workspace_id,
                            IndexChunkRow.indexed_document_version_id.in_(targets),
                        )
                    )
                ).scalars()
            )
        vectors_deleted = 0
        chunks_deleted = 0
        plans_deleted = 0
        manifests_deleted = 0
        assets_deleted = 0
        relations_deleted = 0
        if targets:
            relations_deleted = int(
                (
                    await self._session.execute(
                        delete(IndexChunkAssetRelationRow).where(
                            IndexChunkAssetRelationRow.workspace_id
                            == self._workspace_id,
                            IndexChunkAssetRelationRow.indexed_document_version_id.in_(
                                targets
                            ),
                        )
                    )
                ).rowcount
                or 0
            )
        if chunk_ids:
            vectors_deleted = int(
                (
                    await self._session.execute(
                        delete(VectorRecordRow).where(
                            VectorRecordRow.workspace_id == self._workspace_id,
                            VectorRecordRow.index_chunk_id.in_(chunk_ids),
                        )
                    )
                ).rowcount
                or 0
            )
            chunks_deleted = int(
                (
                    await self._session.execute(
                        delete(IndexChunkRow).where(
                            IndexChunkRow.workspace_id == self._workspace_id,
                            IndexChunkRow.id.in_(chunk_ids),
                        )
                    )
                ).rowcount
                or 0
            )
        if targets:
            plans_deleted = int(
                (
                    await self._session.execute(
                        delete(IndexChunkPlanRow).where(
                            IndexChunkPlanRow.indexed_document_version_id.in_(targets)
                        )
                    )
                ).rowcount
                or 0
            )
            assets_deleted = int(
                (
                    await self._session.execute(
                        delete(IndexAssetRow).where(
                            IndexAssetRow.workspace_id == self._workspace_id,
                            IndexAssetRow.indexed_document_version_id.in_(targets)
                        )
                    )
                ).rowcount
                or 0
            )
            manifests_deleted = int(
                (
                    await self._session.execute(
                        delete(IndexArtifactManifestRow).where(
                            IndexArtifactManifestRow.indexed_document_version_id.in_(targets)
                        )
                    )
                ).rowcount
                or 0
            )
        expired_jobs = tuple(
            (
                await self._session.execute(
                    select(IndexingJobRow.id)
                    .join(
                        IndexedDocumentVersionRow,
                        IndexedDocumentVersionRow.id
                        == IndexingJobRow.indexed_document_version_id,
                    )
                    .where(
                        IndexingJobRow.workspace_id == self._workspace_id,
                        IndexingJobRow.status.in_(
                            (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
                        ),
                        IndexingJobRow.updated_at <= tasks_before,
                        IndexedDocumentVersionRow.serving_status
                        == IndexServingStatus.RETIRED,
                    )
                    .order_by(IndexingJobRow.updated_at, IndexingJobRow.id)
                    .limit(limit)
                    .with_for_update(of=IndexingJobRow, skip_locked=True)
                )
            ).scalars()
        )
        jobs_deleted = 0
        if expired_jobs:
            jobs_deleted = int(
                (
                    await self._session.execute(
                        delete(IndexingJobRow).where(
                            IndexingJobRow.workspace_id == self._workspace_id,
                            IndexingJobRow.id.in_(expired_jobs),
                        )
                    )
                ).rowcount
                or 0
            )
        return IndexCleanupResult(
            retired_targets_cleaned=len(targets),
            vectors_deleted=vectors_deleted,
            chunks_deleted=chunks_deleted,
            plans_deleted=plans_deleted,
            manifests_deleted=manifests_deleted,
            assets_deleted=assets_deleted,
            relations_deleted=relations_deleted,
            jobs_deleted=jobs_deleted,
        )

    async def _job_row(self, job_id: UUID):
        return (
            await self._session.execute(
                select(
                    IndexingJobRow,
                    IndexedDocumentVersionRow,
                    DocumentRow,
                    DocumentVersionRow,
                    IndexRevisionRow,
                    KnowledgeBaseRow,
                )
                .join(
                    IndexedDocumentVersionRow,
                    IndexedDocumentVersionRow.id
                    == IndexingJobRow.indexed_document_version_id,
                )
                .join(DocumentRow, DocumentRow.id == IndexedDocumentVersionRow.document_id)
                .join(
                    DocumentVersionRow,
                    DocumentVersionRow.id
                    == IndexedDocumentVersionRow.document_version_id,
                )
                .join(
                    IndexRevisionRow,
                    IndexRevisionRow.id
                    == IndexedDocumentVersionRow.index_revision_id,
                )
                .join(KnowledgeBaseRow, KnowledgeBaseRow.id == IndexingJobRow.kb_id)
                .where(
                    IndexingJobRow.workspace_id == self._workspace_id,
                    IndexedDocumentVersionRow.workspace_id == self._workspace_id,
                    IndexingJobRow.id == job_id,
                    KnowledgeBaseRow.deleted_at.is_(None),
                )
            )
        ).one_or_none()

    async def claim(
        self,
        *,
        observed_at: datetime,
        max_attempts: int,
    ) -> IndexingLease | None:
        row = (
            await self._session.execute(
                select(IndexingJobRow, IndexedDocumentVersionRow)
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
                    KnowledgeBaseRow,
                    KnowledgeBaseRow.id == IndexingJobRow.kb_id,
                )
                .where(
                    IndexingJobRow.workspace_id == self._workspace_id,
                    IndexedDocumentVersionRow.workspace_id == self._workspace_id,
                    KnowledgeBaseRow.workspace_id == self._workspace_id,
                    KnowledgeBaseRow.deleted_at.is_(None),
                    *_claimable_job(observed_at, max_attempts),
                )
                .order_by(
                    # A continuation receives the time at which its previous
                    # segment yielded. Fresh jobs that arrived during that
                    # segment therefore run first, while later arrivals sort
                    # after the already-due continuation and cannot starve it.
                    func.coalesce(
                        IndexingJobRow.next_attempt_at,
                        IndexingJobRow.created_at,
                    ),
                    DocumentVersionRow.size_bytes,
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
            if job.continuation_pending:
                job.continuation_pending = False
            else:
                job.attempt += 1
        elif job.attempt == 0:
            job.attempt = 1
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
            attempt=job.attempt,
            claimed_at=observed_at,
        )

    async def heartbeat(
        self,
        lease: IndexingLease,
        *,
        observed_at: datetime,
    ) -> bool:
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
        target_id = await self._session.scalar(
            update(IndexingJobRow)
            .where(
                *_owned(lease, self._workspace_id),
                IndexingJobRow.status.in_((JobStatus.RUNNING, JobStatus.FAILED)),
            )
            .values(
                status=JobStatus.QUEUED,
                phase="queued",
                continuation_pending=False,
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
        target_id = await self._session.scalar(
            update(IndexingJobRow)
            .where(
                *_owned(lease, self._workspace_id),
                IndexingJobRow.status.in_((JobStatus.RUNNING, JobStatus.FAILED)),
            )
            .values(
                status=JobStatus.FAILED,
                phase="failed",
                continuation_pending=False,
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
        updated = await self._session.scalar(
            update(IndexingJobRow)
            .where(
                *_owned(lease, self._workspace_id),
                IndexingJobRow.status.in_(
                    (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
                ),
            )
            .values(
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
            job.claimed_at = None
            job.heartbeat_at = None
            job.error_code = ErrorCode.INDEXING_STALE_WORKER.value
            job.error_detail = detail
            job.updated_at = observed_at
            if job.attempt < max_attempts:
                job.status = JobStatus.QUEUED
                job.phase = "queued"
                job.continuation_pending = False
                job.next_attempt_at = retry_at_by_attempt[job.attempt - 1]
                requeued += 1
            else:
                job.status = JobStatus.FAILED
                job.phase = "failed"
                job.continuation_pending = False
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
                IndexingJobRow.attempt == command.attempt,
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
                    IndexingJobRow.attempt == command.attempt,
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
            knowledge_base.deleted_at is not None
            or knowledge_base.active_index_revision_id != target.index_revision_id
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
        row = await self._load(command, lock=True, allow_initial_attempt=True)
        if row is None:
            return None
        job, target, version, revision, embedding, knowledge_base = row
        if (
            job.status is JobStatus.COMPLETED
            and target.build_status is IndexBuildStatus.READY
        ):
            return _target(
                row,
                already_complete=True,
                space_roles=await self._space_roles(revision.id),
            )
        if (
            job.status is JobStatus.CANCELLED
            or target.serving_status is IndexServingStatus.RETIRED
        ):
            raise IndexingCancelled
        if job.attempt == 0:
            job.attempt = command.attempt
        if knowledge_base.deleted_at is not None:
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
        await self._discard_partial_build(target.id)
        target.build_status = IndexBuildStatus.PROCESSING
        target.error_code = None
        target.error_detail = None
        job.status = JobStatus.RUNNING
        job.phase = IndexingPhase.SOURCE_READ.value
        job.error_code = None
        job.error_detail = None
        await self._session.flush()
        return _target(row, space_roles=await self._space_roles(revision.id))

    async def discard_partial_assets(self, command: IndexingCommand) -> bool:
        """Forget candidate asset rows after their local files are removed."""

        row = await self._load(command, lock=True)
        if row is None:
            return False
        job, target, _version, _revision, _embedding, _knowledge_base = row
        if not _is_writable(job, target):
            return False
        await self._session.execute(
            delete(IndexAssetRow).where(
                IndexAssetRow.indexed_document_version_id == target.id
            )
        )
        await self._session.flush()
        return True

    async def _discard_partial_build(self, target_id: UUID) -> None:
        """Discard derived rows while retaining assets until file cleanup."""

        chunk_ids = select(IndexChunkRow.id).where(
            IndexChunkRow.indexed_document_version_id == target_id
        )
        await self._session.execute(
            delete(VectorRecordRow).where(
                VectorRecordRow.index_chunk_id.in_(chunk_ids)
            )
        )
        for model in (
            IndexChunkAssetRelationRow,
            IndexChunkLexicalRow,
            IndexLexicalManifestRow,
            IndexChunkRow,
            IndexArtifactManifestRow,
            IndexChunkPlanRow,
        ):
            await self._session.execute(
                delete(model).where(
                    model.indexed_document_version_id == target_id
                )
            )

    async def _space_roles(
        self, revision_id: UUID
    ) -> tuple[dict[str, UUID], dict[str, EmbeddingSpaceDefinition]]:
        rows = (
            await self._session.execute(
                select(
                    IndexRevisionEmbeddingSpaceRow,
                    EmbeddingSpaceRow,
                    ModelProfileRevisionRow.validation_snapshot,
                )
                .join(
                    EmbeddingSpaceRow,
                    and_(
                        EmbeddingSpaceRow.id
                        == IndexRevisionEmbeddingSpaceRow.embedding_space_id,
                        EmbeddingSpaceRow.workspace_id
                        == IndexRevisionEmbeddingSpaceRow.workspace_id,
                    ),
                )
                .outerjoin(
                    ModelProfileRevisionRow,
                    ModelProfileRevisionRow.id
                    == EmbeddingSpaceRow.model_profile_revision_id,
                )
                .where(
                    IndexRevisionEmbeddingSpaceRow.workspace_id == self._workspace_id,
                    IndexRevisionEmbeddingSpaceRow.index_revision_id == revision_id,
                    IndexRevisionEmbeddingSpaceRow.required.is_(True),
                )
            )
        ).all()
        ids = {binding.role: embedding.id for binding, embedding, _ in rows}
        definitions = {
            binding.role: _embedding(embedding, snapshot)
            for binding, embedding, snapshot in rows
        }
        return ids, definitions

    async def save_chunk_plan(
        self,
        command: IndexingCommand,
        proposed: IndexChunkPlan,
    ) -> bool:
        row = await self._load(command, lock=True)
        if row is None:
            raise IndexingCancelled
        job, target, *_ = row
        if not _is_writable(job, target):
            raise IndexingCancelled
        if proposed.indexed_document_version_id != target.id:
            raise _execution_error(
                ErrorCode.INDEX_CHUNK_PLAN_MISMATCH,
                IndexingPhase.SEMANTIC_ANALYSIS,
                "chunk_plan_target",
            )
        self._session.add(
            IndexChunkPlanRow(
                indexed_document_version_id=proposed.indexed_document_version_id,
                source_checksum_sha256=proposed.source_checksum_sha256,
                profile_fingerprint=proposed.profile_fingerprint,
                unit_sequence_hash=proposed.unit_sequence_hash,
                unit_count=proposed.unit_count,
                chunk_count=proposed.chunk_count,
                boundaries=[
                    {
                        "after_unit_ordinal": item.after_unit_ordinal,
                        "reason": item.reason.value,
                        "score_micros": item.score_micros,
                    }
                    for item in proposed.boundaries
                ],
                plan_hash=proposed.plan_hash,
            )
        )
        await self._session.flush()
        return True

    async def save_artifact_manifest(
        self, command: IndexingCommand, proposed: IndexArtifactManifest
    ) -> bool:
        row = await self._load(command, lock=True)
        if row is None or not _is_writable(row[0], row[1]):
            raise IndexingCancelled
        if proposed.indexed_document_version_id != row[1].id:
            raise _execution_error(
                ErrorCode.INDEX_CHUNK_PLAN_MISMATCH,
                IndexingPhase.PERSISTING,
                "artifact_manifest_target",
            )
        self._session.add(
            IndexArtifactManifestRow(
                indexed_document_version_id=proposed.indexed_document_version_id,
                source_checksum_sha256=proposed.source_checksum_sha256,
                profile_fingerprint=proposed.profile_fingerprint,
                element_sequence_hash=proposed.element_sequence_hash,
                asset_manifest_hash=proposed.asset_manifest_hash,
                unit_plan=list(proposed.unit_plan),
                representation_matrix=list(proposed.representation_matrix),
                unit_count=proposed.unit_count,
                asset_count=proposed.asset_count,
                representation_count=proposed.representation_count,
                relation_plan=list(proposed.relation_plan),
                relation_count=proposed.relation_count,
                relation_manifest_hash=proposed.relation_manifest_hash,
                manifest_hash=proposed.manifest_hash,
            )
        )
        await self._session.flush()
        return True

    async def upsert_assets(
        self, command: IndexingCommand, assets: tuple[IndexAssetWrite, ...]
    ) -> bool:
        row = await self._load(command, lock=True)
        if row is None:
            return False
        job, target, version, *_ = row
        if not _is_writable(job, target):
            return False
        if not assets:
            return True
        values = [
            {
                "id": asset.id,
                "workspace_id": self._workspace_id,
                "kb_id": target.kb_id,
                "document_id": target.document_id,
                "document_version_id": version.id,
                "indexed_document_version_id": target.id,
                "asset_key": asset.asset_key,
                "kind": asset.kind,
                "storage_uri": asset.storage_uri,
                "media_type": asset.media_type,
                "checksum_sha256": asset.checksum_sha256,
                "width": asset.width,
                "height": asset.height,
                "source_location": asset.source_location,
                "processing_metadata": asset.processing_metadata,
            }
            for asset in assets
        ]
        inserted = pg_insert(IndexAssetRow).values(values)
        stored = (
            await self._session.execute(
                inserted.on_conflict_do_update(
                    index_elements=["indexed_document_version_id", "asset_key"],
                    set_={"processing_metadata": IndexAssetRow.processing_metadata},
                    where=and_(
                        IndexAssetRow.id == inserted.excluded.id,
                        IndexAssetRow.checksum_sha256 == inserted.excluded.checksum_sha256,
                        IndexAssetRow.storage_uri == inserted.excluded.storage_uri,
                        IndexAssetRow.media_type == inserted.excluded.media_type,
                        IndexAssetRow.kind == inserted.excluded.kind,
                        IndexAssetRow.source_location == inserted.excluded.source_location,
                        IndexAssetRow.processing_metadata
                        == inserted.excluded.processing_metadata,
                    ),
                ).returning(IndexAssetRow.id, IndexAssetRow.asset_key)
            )
        ).all()
        if {item.asset_key: item.id for item in stored} != {
            item.asset_key: item.id for item in assets
        }:
            raise _execution_error(
                ErrorCode.INDEX_PERSISTENCE_FAILED,
                IndexingPhase.PERSISTING,
                "stable_asset_key",
            )
        return True

    async def upsert_relations(
        self,
        command: IndexingCommand,
        relations: tuple[IndexChunkAssetRelationWrite, ...],
    ) -> bool:
        row = await self._load(command, lock=True)
        if row is None:
            return False
        job, target, *_ = row
        if not _is_writable(job, target):
            return False
        if not relations:
            return True
        values = [
            {
                "id": relation.id,
                "workspace_id": self._workspace_id,
                "kb_id": target.kb_id,
                "indexed_document_version_id": target.id,
                "chunk_id": relation.chunk_id,
                "visual_unit_id": relation.visual_unit_id,
                "asset_id": relation.asset_id,
                "relation_type": relation.relation_type,
                "confidence_micros": relation.confidence_micros,
                "figure_label": relation.figure_label,
                "ordinal": relation.ordinal,
                "provenance": relation.provenance,
                "evidence_group_key": relation.evidence_group_key,
            }
            for relation in relations
        ]
        await self._session.execute(
            pg_insert(IndexChunkAssetRelationRow)
            .values(values)
            .on_conflict_do_nothing(
                index_elements=[
                    "indexed_document_version_id",
                    "chunk_id",
                    "asset_id",
                    "relation_type",
                ]
            )
        )
        stored = (
            await self._session.execute(
                select(IndexChunkAssetRelationRow).where(
                    IndexChunkAssetRelationRow.indexed_document_version_id == target.id,
                    IndexChunkAssetRelationRow.id.in_(
                        tuple(relation.id for relation in relations)
                    ),
                )
            )
        ).scalars()
        observed = {
            item.id: (
                item.chunk_id,
                item.visual_unit_id,
                item.asset_id,
                item.relation_type,
                item.confidence_micros,
                item.figure_label,
                item.ordinal,
                item.provenance,
                item.evidence_group_key,
            )
            for item in stored
        }
        expected = {
            item.id: (
                item.chunk_id,
                item.visual_unit_id,
                item.asset_id,
                item.relation_type,
                item.confidence_micros,
                item.figure_label,
                item.ordinal,
                item.provenance,
                item.evidence_group_key,
            )
            for item in relations
        }
        if observed != expected:
            raise _execution_error(
                ErrorCode.INDEX_PERSISTENCE_FAILED,
                IndexingPhase.PERSISTING,
                "stable_chunk_asset_relation",
            )
        return True

    async def set_phase(self, command: IndexingCommand, phase: IndexingPhase) -> bool:
        row = await self._load(command, lock=True)
        if row is None:
            return False
        job, target, *_ = row
        if not _is_writable(job, target):
            return False
        job.phase = phase.value
        await self._session.flush()
        return True

    async def set_progress(
        self,
        command: IndexingCommand,
        progress: dict[str, Any],
    ) -> bool:
        row = await self._load(command, lock=True)
        if row is None:
            return False
        job, target, *_ = row
        if not _is_writable(job, target):
            return False
        stage = progress.get("stage")
        if not isinstance(stage, str) or not stage:
            raise ValueError("indexing progress stage is required")
        job.progress = dict(progress)
        job.phase = f"parsing_{stage}"[:64]
        await self._session.flush()
        return True

    async def yield_continuation(
        self,
        command: IndexingCommand,
        progress: dict[str, Any],
    ) -> bool:
        row = await self._load(command, lock=True)
        if row is None:
            return False
        job, target, *_ = row
        if not _is_writable(job, target):
            return False
        job.status = JobStatus.QUEUED
        job.phase = "parsing_queued"
        job.progress = dict(progress)
        job.continuation_pending = True
        job.continuation_count += 1
        job.claimed_at = None
        job.heartbeat_at = None
        job.next_attempt_at = func.now()
        await self._session.flush()
        return True

    async def upsert_batch(
        self,
        command: IndexingCommand,
        chunks: tuple[IndexChunkWrite, ...],
        vectors: tuple[VectorRecordWrite, ...],
    ) -> bool:
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
                "unit_key": chunk.unit_key,
                "modality": chunk.modality.value,
                "index_asset_id": chunk.index_asset_id,
                "evidence_group_key": chunk.evidence_group_key,
                "relations": dict(chunk.relations or {}),
                "content": chunk.content,
                "content_hash": chunk.content_hash,
                "embedding_text": chunk.embedding_text,
                "embedding_text_hash": chunk.embedding_text_hash,
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
                    index_elements=["indexed_document_version_id", "unit_key"],
                    set_={
                        "content": chunk_insert.excluded.content,
                        "ordinal": chunk_insert.excluded.ordinal,
                        "modality": chunk_insert.excluded.modality,
                        "index_asset_id": chunk_insert.excluded.index_asset_id,
                        "evidence_group_key": chunk_insert.excluded.evidence_group_key,
                        "relations": chunk_insert.excluded.relations,
                        "content_hash": chunk_insert.excluded.content_hash,
                        "embedding_text": chunk_insert.excluded.embedding_text,
                        "embedding_text_hash": chunk_insert.excluded.embedding_text_hash,
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
                "embedding_dimension": vector.embedding_dimension,
                "representation_kind": vector.representation_kind,
                "embedding": list(vector.embedding),
            }
            for vector in vectors
        ]
        if vector_values:
            if any(
                len(vector.embedding) != vector.embedding_dimension
                for vector in vectors
            ):
                raise _execution_error(
                    ErrorCode.INDEX_PERSISTENCE_FAILED,
                    IndexingPhase.PERSISTING,
                    "vector_dimension",
                )
            vector_insert = pg_insert(VectorRecordRow).values(vector_values)
            stored_vectors = (
                await self._session.execute(
                    vector_insert.on_conflict_do_update(
                        index_elements=[
                            "index_chunk_id",
                            "embedding_space_id",
                            "representation_kind",
                        ],
                        set_={"embedding": vector_insert.excluded.embedding},
                        where=VectorRecordRow.id == vector_insert.excluded.id,
                    ).returning(VectorRecordRow.id, VectorRecordRow.index_chunk_id)
                )
            ).all()
            expected_vector_ids = {vector.id for vector in vectors}
            if {item.id for item in stored_vectors} != expected_vector_ids:
                raise _execution_error(
                    ErrorCode.INDEX_PERSISTENCE_FAILED,
                    IndexingPhase.PERSISTING,
                    "stable_vector_key",
                )
        job.phase = IndexingPhase.PERSISTING.value
        await self._session.flush()
        return True

    async def upsert_lexical_rows(
        self,
        command: IndexingCommand,
        rows: tuple[IndexChunkLexicalWrite, ...],
    ) -> bool:
        loaded = await self._load(command, lock=True)
        if loaded is None:
            return False
        job, target, *_ = loaded
        if not _is_writable(job, target):
            return False
        if not rows:
            return True
        values = [
            {
                "index_chunk_id": item.index_chunk_id,
                "analyzer_version": item.analyzer_version,
                "workspace_id": self._workspace_id,
                "kb_id": target.kb_id,
                "indexed_document_version_id": target.id,
                "lexical_text": item.lexical_text,
                "lexical_text_hash": item.lexical_text_hash,
            }
            for item in rows
        ]
        inserted = pg_insert(IndexChunkLexicalRow).values(values)
        await self._session.execute(
            inserted.on_conflict_do_nothing(
                index_elements=["index_chunk_id", "analyzer_version"]
            )
        )
        stored = (
            await self._session.execute(
                select(IndexChunkLexicalRow).where(
                    IndexChunkLexicalRow.indexed_document_version_id == target.id,
                    IndexChunkLexicalRow.analyzer_version
                    == rows[0].analyzer_version,
                    IndexChunkLexicalRow.index_chunk_id.in_(
                        tuple(item.index_chunk_id for item in rows)
                    ),
                )
            )
        ).scalars().all()
        observed = {
            item.index_chunk_id: (item.lexical_text, item.lexical_text_hash)
            for item in stored
        }
        expected = {
            item.index_chunk_id: (item.lexical_text, item.lexical_text_hash)
            for item in rows
        }
        if observed != expected:
            raise _execution_error(
                ErrorCode.INDEX_PERSISTENCE_FAILED,
                IndexingPhase.PERSISTING,
                "stable_lexical_row",
            )
        return True

    async def complete_lexical_manifest(
        self,
        command: IndexingCommand,
        proposed: IndexLexicalManifest,
    ) -> bool:
        loaded = await self._load(command, lock=True)
        if loaded is None:
            return False
        job, target, *_ = loaded
        if not _is_writable(job, target) or (
            proposed.indexed_document_version_id != target.id
        ):
            return False
        stored_rows = (
            await self._session.execute(
                select(
                    IndexChunkLexicalRow.index_chunk_id,
                    IndexChunkLexicalRow.lexical_text_hash,
                )
                .where(
                    IndexChunkLexicalRow.indexed_document_version_id == target.id,
                    IndexChunkLexicalRow.analyzer_version
                    == proposed.analyzer_version,
                )
                .order_by(IndexChunkLexicalRow.index_chunk_id)
            )
        ).all()
        observed_hash = lexical_manifest_hash(
            proposed.analyzer_version,
            (
                (item.index_chunk_id, item.lexical_text_hash)
                for item in stored_rows
            ),
        )
        if (
            len(stored_rows) != proposed.lexical_chunk_count
            or observed_hash != proposed.lexical_manifest_hash
        ):
            raise _execution_error(
                ErrorCode.INDEX_INCOMPLETE,
                IndexingPhase.VALIDATING,
                "lexical_manifest_content",
            )
        inserted = pg_insert(IndexLexicalManifestRow).values(
            indexed_document_version_id=target.id,
            analyzer_version=proposed.analyzer_version,
            workspace_id=self._workspace_id,
            kb_id=target.kb_id,
            lexical_chunk_count=proposed.lexical_chunk_count,
            lexical_manifest_hash=proposed.lexical_manifest_hash,
        )
        await self._session.execute(
            inserted.on_conflict_do_nothing(
                index_elements=[
                    "indexed_document_version_id",
                    "analyzer_version",
                ]
            )
        )
        manifest = await self._session.get(
            IndexLexicalManifestRow,
            (target.id, proposed.analyzer_version),
        )
        if manifest is None or (
            manifest.lexical_chunk_count != proposed.lexical_chunk_count
            or manifest.lexical_manifest_hash != proposed.lexical_manifest_hash
        ):
            raise _execution_error(
                ErrorCode.INDEX_PERSISTENCE_FAILED,
                IndexingPhase.VALIDATING,
                "stable_lexical_manifest",
            )
        return True

    async def complete(self, command: IndexingCommand, *, expected_chunks: int) -> bool:
        row = await self._load(command, lock=True)
        if row is None:
            return False
        job, target, version, revision, _embedding_row, knowledge_base = row
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
        chunk_records = (
            await self._session.execute(
                select(
                    IndexChunkRow.ordinal,
                    IndexChunkRow.id.label("chunk_id"),
                    IndexChunkRow.unit_key,
                    IndexChunkRow.index_asset_id,
                    IndexChunkRow.embedding_text_hash,
                )
                .where(IndexChunkRow.indexed_document_version_id == target.id)
                .order_by(IndexChunkRow.ordinal)
            )
        ).all()
        vector_records = (
            await self._session.execute(
                select(
                    VectorRecordRow.index_chunk_id.label("chunk_id"),
                    VectorRecordRow.id.label("vector_id"),
                    VectorRecordRow.embedding_space_id,
                    VectorRecordRow.representation_kind,
                    VectorRecordRow.embedding_dimension.label("physical_dimension"),
                )
                .join(
                    IndexChunkRow,
                    IndexChunkRow.id == VectorRecordRow.index_chunk_id,
                )
                .where(IndexChunkRow.indexed_document_version_id == target.id)
            )
        ).all()
        vectors_by_chunk: dict[str, list[tuple[UUID, str, int]]] = {
            str(record.chunk_id): [] for record in chunk_records
        }
        for record in vector_records:
            vectors_by_chunk.setdefault(str(record.chunk_id), []).append(
                (
                    record.embedding_space_id,
                    record.representation_kind,
                    record.physical_dimension,
                )
            )
        manifest_row = await self._session.get(IndexArtifactManifestRow, target.id)
        if manifest_row is None:
            valid = False
        else:
            manifest = _artifact_manifest(manifest_row)
            role_spaces, role_definitions = await self._space_roles(revision.id)
            asset_rows = (
                await self._session.execute(
                    select(IndexAssetRow.id, IndexAssetRow.asset_key).where(
                        IndexAssetRow.indexed_document_version_id == target.id
                    )
                )
            ).all()
            asset_ids = {item.id for item in asset_rows}
            asset_ids_by_key = {item.asset_key: item.id for item in asset_rows}
            relation_rows = (
                await self._session.execute(
                    select(IndexChunkAssetRelationRow).where(
                        IndexChunkAssetRelationRow.indexed_document_version_id
                        == target.id
                    )
                )
            ).scalars().all()
            planned_units = {
                item.get("unit_id"): item for item in manifest.unit_plan
            }
            valid = (
                manifest.unit_count == expected_chunks
                and manifest.asset_count == len(asset_ids)
                and manifest.source_checksum_sha256 == version.checksum_sha256
                and manifest.profile_fingerprint
                == profile_fingerprint(
                    revision.parser_config,
                    revision.chunking_config,
                    revision.enrichment_config,
                    revision.representation_config,
                )
                and manifest.unit_count == len(manifest.unit_plan)
                and manifest.representation_count
                == len(manifest.representation_matrix)
                and len({record.chunk_id for record in chunk_records})
                == expected_chunks
                and tuple(sorted({record.ordinal for record in chunk_records}))
                == tuple(range(expected_chunks))
                and len(planned_units) == expected_chunks
            )
            planned_relations = {
                item.get("relation_id"): item
                for item in manifest.relation_plan
            }
            valid = valid and (
                manifest.relation_count == len(relation_rows)
                and manifest.relation_count == len(manifest.relation_plan)
                and len(planned_relations) == len(relation_rows)
                and all(
                    (planned := planned_relations.get(str(record.id)))
                    is not None
                    and planned.get("chunk_id") == str(record.chunk_id)
                    and planned.get("visual_unit_id")
                    == str(record.visual_unit_id)
                    and planned.get("asset_id") == str(record.asset_id)
                    and planned.get("relation_type") == record.relation_type
                    and planned.get("confidence_micros")
                    == record.confidence_micros
                    and planned.get("figure_label") == record.figure_label
                    and planned.get("ordinal") == record.ordinal
                    and planned.get("provenance") == record.provenance
                    and planned.get("evidence_group_key")
                    == record.evidence_group_key
                    for record in relation_rows
                )
            )
            dimension_by_space = {
                role_spaces[role]: definition.dimension
                for role, definition in role_definitions.items()
                if role in role_spaces
            }
            if not all(
                space_id in dimension_by_space
                and physical_dimension == dimension_by_space[space_id]
                for values in vectors_by_chunk.values()
                for space_id, _kind, physical_dimension in values
            ):
                valid = False
            for record in chunk_records:
                planned = planned_units.get(str(record.chunk_id))
                if (
                    planned is None
                    or planned.get("unit_key") != record.unit_key
                    or planned.get("ordinal") != record.ordinal
                    or planned.get("embedding_text_hash")
                    != record.embedding_text_hash
                    or (
                        planned.get("asset_key") is not None
                        and record.index_asset_id
                        != asset_ids_by_key.get(planned.get("asset_key"))
                    )
                    or (
                        planned.get("asset_key") is None
                        and record.index_asset_id is not None
                    )
                ):
                    valid = False
                    break
            for requirement in manifest.representation_matrix:
                if not requirement.get("required", True):
                    continue
                required_space = role_spaces.get(requirement.get("space_role"))
                definition = role_definitions.get(requirement.get("space_role"))
                identity = (
                    required_space,
                    requirement.get("representation_kind"),
                    definition.dimension if definition is not None else None,
                )
                if required_space is None or identity not in vectors_by_chunk.get(
                    requirement.get("unit_id", ""), []
                ):
                    valid = False
                    break
        if not valid:
            raise IndexingExecutionError(
                ErrorCode.INDEX_INCOMPLETE,
                phase=IndexingPhase.VALIDATING,
                diagnostic={
                    "expected_chunks": expected_chunks,
                    "observed_chunks": len(chunk_records),
                    "observed_vectors": len(vector_records),
                },
            )
        lexical_manifest = await self._session.get(
            IndexLexicalManifestRow,
            (target.id, LEXICAL_ANALYZER_VERSION),
        )
        if lexical_manifest is None:
            raise IndexingExecutionError(
                ErrorCode.INDEX_INCOMPLETE,
                phase=IndexingPhase.VALIDATING,
                diagnostic={"check": "lexical_manifest"},
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
                IndexingJobRow.attempt == command.attempt,
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

    async def _load(
        self,
        command: IndexingCommand,
        *,
        lock: bool,
        allow_initial_attempt: bool = False,
    ):
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
                or_(
                    IndexingJobRow.attempt == command.attempt,
                    and_(
                        allow_initial_attempt,
                        command.attempt == 1,
                        or_(
                            and_(
                                IndexingJobRow.attempt == 0,
                                IndexingJobRow.status == JobStatus.QUEUED,
                            ),
                            IndexingJobRow.status == JobStatus.CANCELLED,
                        ),
                    ),
                ),
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


def _target(
    row,
    *,
    already_complete: bool = False,
    space_roles: tuple[
        dict[str, UUID], dict[str, EmbeddingSpaceDefinition]
    ] | None = None,
) -> IndexingTarget:
    job, target, version, revision, embedding, _knowledge_base = row
    space_ids, spaces = space_roles or (
        {"text_retrieval": embedding.id},
        {"text_retrieval": _embedding(embedding)},
    )
    primary_space = spaces.get("text_retrieval", _embedding(embedding))
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
        embedding_space=primary_space,
        already_complete=already_complete,
        enrichment_config=dict(revision.enrichment_config),
        representation_config=dict(revision.representation_config),
        embedding_space_ids=space_ids,
        embedding_spaces=spaces,
    )


def _artifact_manifest(row: IndexArtifactManifestRow) -> IndexArtifactManifest:
    try:
        return IndexArtifactManifest(
            indexed_document_version_id=row.indexed_document_version_id,
            source_checksum_sha256=row.source_checksum_sha256,
            profile_fingerprint=row.profile_fingerprint,
            element_sequence_hash=row.element_sequence_hash,
            asset_manifest_hash=row.asset_manifest_hash,
            unit_plan=tuple(dict(item) for item in row.unit_plan),
            representation_matrix=tuple(
                dict(item) for item in row.representation_matrix
            ),
            unit_count=row.unit_count,
            asset_count=row.asset_count,
            representation_count=row.representation_count,
            relation_plan=tuple(dict(item) for item in row.relation_plan),
            relation_count=row.relation_count,
            relation_manifest_hash=row.relation_manifest_hash,
            manifest_hash=row.manifest_hash,
        )
    except (TypeError, ValueError) as error:
        raise _execution_error(
            ErrorCode.INDEX_CHUNK_PLAN_MISMATCH,
            IndexingPhase.PERSISTING,
            "artifact_manifest_shape",
        ) from error


def _embedding(
    row: EmbeddingSpaceRow,
    validation_snapshot: dict[str, Any] | None = None,
) -> EmbeddingSpaceDefinition:
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
        model_profile_revision_id=row.model_profile_revision_id,
        dimension_request_mode=(
            str(validation_snapshot.get("dimension_request_mode", "explicit"))
            if validation_snapshot is not None
            else "explicit"
        ),
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
        IndexingJobRow.attempt == lease.attempt,
    )


def _claimable_job(
    observed_at: datetime,
    max_attempts: int,
) -> tuple[Any, ...]:
    return (
        or_(
            and_(
                IndexingJobRow.status == JobStatus.QUEUED,
                or_(
                    IndexingJobRow.continuation_pending.is_(True),
                    IndexingJobRow.attempt < max_attempts,
                ),
                or_(
                    IndexingJobRow.next_attempt_at.is_(None),
                    IndexingJobRow.next_attempt_at <= observed_at,
                ),
                or_(
                    IndexedDocumentVersionRow.build_status.in_(
                        (IndexBuildStatus.QUEUED, IndexBuildStatus.FAILED)
                    ),
                    and_(
                        IndexingJobRow.continuation_pending.is_(True),
                        IndexedDocumentVersionRow.build_status
                        == IndexBuildStatus.PROCESSING,
                    ),
                ),
            ),
            and_(
                IndexingJobRow.status == JobStatus.COMPLETED,
                IndexingJobRow.heartbeat_at.is_(None),
                IndexedDocumentVersionRow.build_status == IndexBuildStatus.READY,
            ),
        ),
        IndexedDocumentVersionRow.serving_status
        == IndexServingStatus.CANDIDATE,
    )


def _owned_execution(lease: IndexingLease, workspace_id: UUID) -> tuple[Any, ...]:
    return (
        *_owned(lease, workspace_id),
        IndexingJobRow.status.in_(
            (JobStatus.RUNNING, JobStatus.FAILED, JobStatus.COMPLETED)
        ),
    )


def _job_snapshot(row) -> IndexingJobSnapshot:
    job, target, document, version, revision, knowledge_base = row
    can_retry = (
        job.status is JobStatus.FAILED
        and target.build_status is IndexBuildStatus.FAILED
        and target.serving_status is IndexServingStatus.CANDIDATE
        and document.deleted_at is None
        and document.current_version_id == target.document_version_id
        and version.source_status is DocumentSourceStatus.AVAILABLE
        and revision.status is IndexRevisionStatus.ACTIVE
        and knowledge_base.active_index_revision_id == target.index_revision_id
    )
    return IndexingJobSnapshot(
        job_id=job.id,
        kb_id=job.kb_id,
        document_id=target.document_id,
        document_version_id=target.document_version_id,
        indexed_document_version_id=target.id,
        index_revision_id=target.index_revision_id,
        job_status=job.status.value,
        phase=job.phase,
        progress=dict(job.progress or {}),
        attempt=job.attempt,
        build_status=target.build_status.value,
        serving_status=target.serving_status.value,
        claimed_at=job.claimed_at,
        heartbeat_at=job.heartbeat_at,
        next_attempt_at=job.next_attempt_at,
        error_code=job.error_code or target.error_code,
        error_detail=(
            dict(job.error_detail or target.error_detail or {})
            if job.error_code or target.error_code
            else None
        ),
        can_retry=can_retry,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
