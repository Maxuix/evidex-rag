"""SQLAlchemy persistence for knowledge-base and document lifecycle state."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import and_, delete, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from rag_kb.db.models import (
    ContentMutation as ContentMutationRow,
    Document as DocumentRow,
    DocumentSourceStatus,
    DocumentVersion as DocumentVersionRow,
    EmbeddingSpace as EmbeddingSpaceRow,
    IndexedDocumentVersion as IndexedDocumentVersionRow,
    IndexArtifactManifest as IndexArtifactManifestRow,
    IndexAsset as IndexAssetRow,
    IndexBuildStatus,
    IndexChunk as IndexChunkRow,
    IndexChunkAssetRelation as IndexChunkAssetRelationRow,
    IndexingJob as IndexingJobRow,
    IndexRevision as IndexRevisionRow,
    IndexRevisionEmbeddingSpace as IndexRevisionEmbeddingSpaceRow,
    IndexRevisionStatus,
    IndexServingStatus,
    JobStatus,
    KnowledgeBase as KnowledgeBaseRow,
    SourceChange as SourceChangeRow,
    SourceChangeKind,
    SourceFileCleanup as SourceFileCleanupRow,
    VectorRecord as VectorRecordRow,
    Workspace as WorkspaceRow,
)
from rag_kb.domain import (
    ContentMutation,
    Document,
    DocumentChunk,
    DocumentChunkAsset,
    DocumentChunkInspection,
    DocumentChunkRelation,
    DocumentDetail,
    DocumentIndexSummary,
    DocumentMutationResult,
    DocumentSource,
    DocumentVersion,
    DuplicateDocumentError,
    EmbeddingSpaceRole,
    EmbeddingSpaceDefinition,
    EmbeddingRoleSummary,
    IdempotencyScope,
    IndexProfileDefinition,
    KnowledgeBase,
    KnowledgeBaseEmbeddingSummary,
    Page,
    PendingFileMutation,
    ResourceNameConflictError,
    ResourceStateConflictError,
    SourceFileCleanup,
    SourceFileReference,
)


class SqlAlchemyKnowledgeBaseRepository:
    def __init__(
        self,
        session: AsyncSession,
        workspace_id: UUID,
        ensure_active: Callable[[], None],
    ) -> None:
        self._session = session
        self._workspace_id = workspace_id
        self._ensure_active = ensure_active

    async def create(
        self,
        *,
        name: str,
        retrieval_defaults: dict[str, Any],
        answer_policy_defaults: dict[str, Any],
        embedding_space: EmbeddingSpaceDefinition,
        cross_modal_embedding_space: EmbeddingSpaceDefinition | None,
        index_profile: IndexProfileDefinition,
    ) -> KnowledgeBase:
        self._ensure_active()
        await self._session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
            {"key": f"content-foundation:{self._workspace_id}"},
        )
        workspace = await self._session.get(WorkspaceRow, self._workspace_id)
        if workspace is None:
            workspace = WorkspaceRow(
                id=self._workspace_id,
                name=f"development-workspace-{self._workspace_id}",
            )
            self._session.add(workspace)
            await self._session.flush()

        embedding = await self._find_or_create_embedding(embedding_space)
        cross_modal_embedding = (
            await self._find_or_create_embedding(cross_modal_embedding_space)
            if cross_modal_embedding_space is not None
            else None
        )

        duplicate = await self._session.scalar(
            select(KnowledgeBaseRow.id).where(
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.name == name,
                KnowledgeBaseRow.deleted_at.is_(None),
            )
        )
        if duplicate is not None:
            raise ResourceNameConflictError("knowledge-base name already exists")

        kb = KnowledgeBaseRow(
            workspace_id=self._workspace_id,
            name=name,
            retrieval_defaults=dict(retrieval_defaults),
            answer_policy_defaults=dict(answer_policy_defaults),
        )
        self._session.add(kb)
        await self._session.flush()
        revision = IndexRevisionRow(
            workspace_id=self._workspace_id,
            kb_id=kb.id,
            embedding_space_id=embedding.id,
            status=IndexRevisionStatus.ACTIVE,
            source_snapshot_seq=0,
            parser_config=dict(index_profile.parser_config),
            chunking_config=dict(index_profile.chunking_config),
            enrichment_config=dict(index_profile.enrichment_config),
            representation_config=dict(index_profile.representation_config),
        )
        self._session.add(revision)
        await self._session.flush()
        self._session.add(
            IndexRevisionEmbeddingSpaceRow(
                workspace_id=self._workspace_id,
                index_revision_id=revision.id,
                role=EmbeddingSpaceRole.TEXT_RETRIEVAL.value,
                embedding_space_id=embedding.id,
                required=True,
                retrieval_weight_micros=1_000_000,
            )
        )
        required_roles = index_profile.chunking_config.get(
            "required_embedding_roles", ()
        )
        if EmbeddingSpaceRole.SEMANTIC_ANALYSIS.value in required_roles:
            self._session.add(
                IndexRevisionEmbeddingSpaceRow(
                    workspace_id=self._workspace_id,
                    index_revision_id=revision.id,
                    role=EmbeddingSpaceRole.SEMANTIC_ANALYSIS.value,
                    embedding_space_id=embedding.id,
                    required=True,
                )
            )
        if cross_modal_embedding is not None:
            self._session.add(
                IndexRevisionEmbeddingSpaceRow(
                    workspace_id=self._workspace_id,
                    index_revision_id=revision.id,
                    role=EmbeddingSpaceRole.CROSS_MODAL_RETRIEVAL.value,
                    embedding_space_id=cross_modal_embedding.id,
                    required=True,
                    retrieval_weight_micros=1_000_000,
                )
            )
        now = datetime.now(UTC)
        kb.active_index_revision_id = revision.id
        kb.provisioned_at = now
        kb.updated_at = now
        await self._session.flush()
        return _knowledge_base(
            kb,
            embedding.id,
            revision.parser_config,
            revision.chunking_config,
            _embedding_summary_from_spaces(embedding, cross_modal_embedding),
        )

    async def _find_or_create_embedding(
        self, definition: EmbeddingSpaceDefinition
    ) -> EmbeddingSpaceRow:
        embedding = await self._session.scalar(
            select(EmbeddingSpaceRow).where(
                EmbeddingSpaceRow.compatibility_fingerprint
                == definition.compatibility_fingerprint
            )
        )
        if embedding is None:
            embedding = EmbeddingSpaceRow(
                workspace_id=self._workspace_id,
                provider_identity=definition.provider_identity,
                endpoint_identity=definition.endpoint_identity,
                requested_model=definition.requested_model,
                resolved_model=definition.resolved_model,
                model_version=definition.model_version,
                deployment_revision=definition.deployment_revision,
                dimension=definition.dimension,
                distance_metric=definition.distance_metric,
                vector_data_type=definition.vector_data_type,
                normalization=definition.normalization,
                configuration_fingerprint=definition.configuration_fingerprint,
                tokenizer_fingerprint=definition.tokenizer_fingerprint,
                compatibility_fingerprint=definition.compatibility_fingerprint,
                model_profile_revision_id=definition.model_profile_revision_id,
            )
            self._session.add(embedding)
            await self._session.flush()
        elif not _embedding_matches(embedding, self._workspace_id, definition):
            raise ResourceStateConflictError(
                "the configured embedding space conflicts with persisted state"
            )
        return embedding

    async def get(
        self, kb_id: UUID, *, include_deleted: bool = False
    ) -> KnowledgeBase | None:
        self._ensure_active()
        filters = [
            KnowledgeBaseRow.workspace_id == self._workspace_id,
            KnowledgeBaseRow.id == kb_id,
        ]
        if not include_deleted:
            filters.append(KnowledgeBaseRow.deleted_at.is_(None))
        row = (
            await self._session.execute(
                select(
                    KnowledgeBaseRow,
                    IndexRevisionRow.embedding_space_id,
                    IndexRevisionRow.parser_config,
                    IndexRevisionRow.chunking_config,
                )
                .join(
                    IndexRevisionRow,
                    IndexRevisionRow.id == KnowledgeBaseRow.active_index_revision_id,
                )
                .where(*filters)
            )
        ).one_or_none()
        if row is None:
            return None
        summary = await self._embedding_summary(row[0].active_index_revision_id)
        return _knowledge_base(row[0], row[1], row[2], row[3], summary)

    async def list(
        self,
        *,
        limit: int,
        sort: str,
        after: tuple[str, ...] | None,
    ) -> Page[KnowledgeBase]:
        self._ensure_active()
        descending = sort.startswith("-")
        field = sort.removeprefix("-")
        column = {
            "created_at": KnowledgeBaseRow.created_at,
            "name": KnowledgeBaseRow.name,
        }[field]
        statement = (
            select(
                KnowledgeBaseRow,
                IndexRevisionRow.embedding_space_id,
                IndexRevisionRow.parser_config,
                IndexRevisionRow.chunking_config,
            )
            .join(
                IndexRevisionRow,
                IndexRevisionRow.id == KnowledgeBaseRow.active_index_revision_id,
            )
            .where(
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.deleted_at.is_(None),
            )
        )
        statement = _with_after(statement, column, KnowledgeBaseRow.id, after, descending)
        ordering = column.desc() if descending else column.asc()
        id_ordering = KnowledgeBaseRow.id.desc() if descending else KnowledgeBaseRow.id.asc()
        rows = (await self._session.execute(statement.order_by(ordering, id_ordering).limit(limit + 1))).all()
        has_more = len(rows) > limit
        rows = rows[:limit]
        summaries = await self._embedding_summaries(
            tuple(row[0].active_index_revision_id for row in rows)
        )
        items = tuple(
            _knowledge_base(
                row[0],
                row[1],
                row[2],
                row[3],
                summaries[row[0].active_index_revision_id],
            )
            for row in rows
        )
        next_values = _cursor_values(items[-1], field) if has_more and items else None
        return Page(items=items, next_values=next_values)

    async def update(
        self,
        kb_id: UUID,
        *,
        name: str | None,
        retrieval_defaults: dict[str, Any] | None,
        answer_policy_defaults: dict[str, Any] | None,
    ) -> KnowledgeBase | None:
        self._ensure_active()
        kb = await self._session.scalar(
            select(KnowledgeBaseRow).where(
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.id == kb_id,
                KnowledgeBaseRow.deleted_at.is_(None),
            ).with_for_update()
        )
        if kb is None:
            return None
        if name is not None and name != kb.name:
            duplicate = await self._session.scalar(
                select(KnowledgeBaseRow.id).where(
                    KnowledgeBaseRow.workspace_id == self._workspace_id,
                    KnowledgeBaseRow.name == name,
                    KnowledgeBaseRow.id != kb_id,
                    KnowledgeBaseRow.deleted_at.is_(None),
                )
            )
            if duplicate is not None:
                raise ResourceNameConflictError("knowledge-base name already exists")
            kb.name = name
        if retrieval_defaults is not None:
            kb.retrieval_defaults = dict(retrieval_defaults)
        if answer_policy_defaults is not None:
            kb.answer_policy_defaults = dict(answer_policy_defaults)
        kb.updated_at = datetime.now(UTC)
        await self._session.flush()
        assert kb.active_index_revision_id is not None
        revision_facts = (
            await self._session.execute(
                select(
                    IndexRevisionRow.embedding_space_id,
                    IndexRevisionRow.parser_config,
                    IndexRevisionRow.chunking_config,
                ).where(
                IndexRevisionRow.id == kb.active_index_revision_id,
                IndexRevisionRow.kb_id == kb.id,
            )
            )
        ).one_or_none()
        assert revision_facts is not None
        return _knowledge_base(
            kb,
            revision_facts[0],
            revision_facts[1],
            revision_facts[2],
            await self._embedding_summary(kb.active_index_revision_id),
        )

    async def soft_delete(self, kb_id: UUID) -> KnowledgeBase | None:
        self._ensure_active()
        kb = await self._session.scalar(
            select(KnowledgeBaseRow).where(
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.id == kb_id,
            ).with_for_update()
        )
        if kb is None:
            return None
        assert kb.active_index_revision_id is not None
        revision_facts = (
            await self._session.execute(
                select(
                    IndexRevisionRow.embedding_space_id,
                    IndexRevisionRow.parser_config,
                    IndexRevisionRow.chunking_config,
                ).where(
                    IndexRevisionRow.workspace_id == self._workspace_id,
                    IndexRevisionRow.kb_id == kb.id,
                    IndexRevisionRow.id == kb.active_index_revision_id,
                )
            )
        ).one_or_none()
        if revision_facts is None:
            raise ResourceStateConflictError(
                "knowledge base active revision is unavailable"
            )
        embedding = await self._embedding_summary(kb.active_index_revision_id)
        if kb.deleted_at is None:
            now = datetime.now(UTC)
            versions = (
                await self._session.execute(
                    select(
                        DocumentVersionRow.id,
                        DocumentVersionRow.storage_uri,
                    ).where(
                        DocumentVersionRow.workspace_id == self._workspace_id,
                        DocumentVersionRow.kb_id == kb.id,
                    )
                )
            ).all()
            if versions:
                await self._session.execute(
                    pg_insert(SourceFileCleanupRow)
                    .values(
                        [
                            {
                                "workspace_id": self._workspace_id,
                                "document_version_id": version.id,
                                "storage_uri": version.storage_uri,
                                "reason": "knowledge_base_deleted",
                            }
                            for version in versions
                        ]
                    )
                    .on_conflict_do_nothing(index_elements=["document_version_id"])
                )
            await self._session.execute(
                update(DocumentRow)
                .where(
                    DocumentRow.workspace_id == self._workspace_id,
                    DocumentRow.kb_id == kb.id,
                )
                .values(deleted_at=now, updated_at=now)
            )
            await self._session.execute(
                update(DocumentVersionRow)
                .where(
                    DocumentVersionRow.workspace_id == self._workspace_id,
                    DocumentVersionRow.kb_id == kb.id,
                )
                .values(source_status=DocumentSourceStatus.DELETED)
            )
            target_ids = select(IndexedDocumentVersionRow.id).where(
                IndexedDocumentVersionRow.workspace_id == self._workspace_id,
                IndexedDocumentVersionRow.kb_id == kb.id,
            )
            await self._session.execute(
                update(IndexedDocumentVersionRow)
                .where(IndexedDocumentVersionRow.id.in_(target_ids))
                .values(
                    serving_status=IndexServingStatus.RETIRED,
                    updated_at=now,
                )
            )
            await self._session.execute(
                update(IndexingJobRow)
                .where(
                    IndexingJobRow.workspace_id == self._workspace_id,
                    IndexingJobRow.kb_id == kb.id,
                    IndexingJobRow.status.in_((JobStatus.QUEUED, JobStatus.RUNNING)),
                )
                .values(
                    status=JobStatus.CANCELLED,
                    phase="knowledge_base_deleted",
                    claimed_by=None,
                    claimed_at=None,
                    heartbeat_at=None,
                    next_attempt_at=None,
                    updated_at=now,
                )
            )
            await self._session.execute(
                update(IndexRevisionRow)
                .where(
                    IndexRevisionRow.workspace_id == self._workspace_id,
                    IndexRevisionRow.kb_id == kb.id,
                    IndexRevisionRow.status.in_(
                        (IndexRevisionStatus.ACTIVE, IndexRevisionStatus.READY)
                    ),
                )
                .values(status=IndexRevisionStatus.RETIRED, updated_at=now)
            )
            kb.deleted_at = now
            kb.updated_at = now
            await self._session.flush()
        return _knowledge_base(
            kb,
            revision_facts[0],
            revision_facts[1],
            revision_facts[2],
            embedding,
        )

    async def _embedding_summary(
        self, revision_id: UUID | None
    ) -> KnowledgeBaseEmbeddingSummary:
        assert revision_id is not None
        return (await self._embedding_summaries((revision_id,)))[revision_id]

    async def _embedding_summaries(
        self, revision_ids: tuple[UUID, ...]
    ) -> dict[UUID, KnowledgeBaseEmbeddingSummary]:
        if not revision_ids:
            return {}
        rows = (
            await self._session.execute(
                select(IndexRevisionEmbeddingSpaceRow, EmbeddingSpaceRow)
                .join(
                    EmbeddingSpaceRow,
                    and_(
                        EmbeddingSpaceRow.workspace_id
                        == IndexRevisionEmbeddingSpaceRow.workspace_id,
                        EmbeddingSpaceRow.id
                        == IndexRevisionEmbeddingSpaceRow.embedding_space_id,
                    ),
                )
                .where(
                    IndexRevisionEmbeddingSpaceRow.workspace_id
                    == self._workspace_id,
                    IndexRevisionEmbeddingSpaceRow.index_revision_id.in_(revision_ids),
                    IndexRevisionEmbeddingSpaceRow.role.in_(
                        (
                            EmbeddingSpaceRole.TEXT_RETRIEVAL.value,
                            EmbeddingSpaceRole.CROSS_MODAL_RETRIEVAL.value,
                        )
                    ),
                )
            )
        ).all()
        spaces_by_revision: dict[UUID, dict[str, EmbeddingSpaceRow]] = {}
        for binding, space in rows:
            spaces_by_revision.setdefault(binding.index_revision_id, {})[
                binding.role
            ] = space
        summaries: dict[UUID, KnowledgeBaseEmbeddingSummary] = {}
        for revision_id in revision_ids:
            by_role = spaces_by_revision.get(revision_id, {})
            text_space = by_role.get(EmbeddingSpaceRole.TEXT_RETRIEVAL.value)
            assert text_space is not None
            summaries[revision_id] = _embedding_summary_from_spaces(
                text_space,
                by_role.get(EmbeddingSpaceRole.CROSS_MODAL_RETRIEVAL.value),
            )
        return summaries


class SqlAlchemyDocumentRepository:
    def __init__(self, session: AsyncSession, workspace_id: UUID, ensure_active: Callable[[], None]) -> None:
        self._session = session
        self._workspace_id = workspace_id
        self._ensure_active = ensure_active

    async def get(self, document_id: UUID) -> Document | None:
        self._ensure_active()
        row = (
            await self._session.execute(
                select(DocumentRow, DocumentVersionRow)
                .join(KnowledgeBaseRow, KnowledgeBaseRow.id == DocumentRow.kb_id)
                .outerjoin(
                    DocumentVersionRow,
                    DocumentVersionRow.id == DocumentRow.current_version_id,
                )
                .where(
                    DocumentRow.workspace_id == self._workspace_id,
                    DocumentRow.id == document_id,
                    KnowledgeBaseRow.workspace_id == self._workspace_id,
                    KnowledgeBaseRow.deleted_at.is_(None),
                )
            )
        ).one_or_none()
        return _document(row[0], row[1]) if row is not None else None

    async def get_detail(self, document_id: UUID) -> DocumentDetail | None:
        self._ensure_active()
        row = (
            await self._session.execute(
                select(
                    DocumentRow,
                    DocumentVersionRow,
                    IndexedDocumentVersionRow,
                    IndexArtifactManifestRow,
                )
                .outerjoin(
                    DocumentVersionRow,
                    DocumentVersionRow.id == DocumentRow.current_version_id,
                )
                .join(
                    KnowledgeBaseRow,
                    KnowledgeBaseRow.id == DocumentRow.kb_id,
                )
                .outerjoin(
                    IndexedDocumentVersionRow,
                    and_(
                        IndexedDocumentVersionRow.workspace_id
                        == self._workspace_id,
                        IndexedDocumentVersionRow.document_id == DocumentRow.id,
                        IndexedDocumentVersionRow.document_version_id
                        == DocumentRow.current_version_id,
                        IndexedDocumentVersionRow.index_revision_id
                        == KnowledgeBaseRow.active_index_revision_id,
                    ),
                )
                .outerjoin(
                    IndexArtifactManifestRow,
                    IndexArtifactManifestRow.indexed_document_version_id
                    == IndexedDocumentVersionRow.id,
                )
                .where(
                    DocumentRow.workspace_id == self._workspace_id,
                    KnowledgeBaseRow.workspace_id == self._workspace_id,
                    KnowledgeBaseRow.deleted_at.is_(None),
                    DocumentRow.id == document_id,
                )
            )
        ).one_or_none()
        if row is None:
            return None
        document_row, version_row, indexed_row, manifest_row = row
        v2_counts = _v2_manifest_counts(manifest_row)
        summary = (
            DocumentIndexSummary(
                indexed_document_version_id=indexed_row.id,
                index_revision_id=indexed_row.index_revision_id,
                build_status=indexed_row.build_status.value,
                serving_status=indexed_row.serving_status.value,
                unit_count=(manifest_row.unit_count if manifest_row else None),
                asset_count=(manifest_row.asset_count if manifest_row else None),
                representation_count=(
                    manifest_row.representation_count if manifest_row else None
                ),
                **v2_counts,
            )
            if indexed_row is not None
            else None
        )
        return DocumentDetail(
            document=_document(document_row, version_row),
            index=summary,
        )

    async def inspect_chunks(
        self,
        document_id: UUID,
        *,
        limit: int,
        after: tuple[str, ...] | None,
    ) -> DocumentChunkInspection | None:
        self._ensure_active()
        row = (
            await self._session.execute(
                select(DocumentRow, DocumentVersionRow, IndexedDocumentVersionRow)
                .join(KnowledgeBaseRow, KnowledgeBaseRow.id == DocumentRow.kb_id)
                .outerjoin(
                    DocumentVersionRow,
                    DocumentVersionRow.id == DocumentRow.current_version_id,
                )
                .outerjoin(
                    IndexedDocumentVersionRow,
                    and_(
                        IndexedDocumentVersionRow.workspace_id == self._workspace_id,
                        IndexedDocumentVersionRow.document_id == DocumentRow.id,
                        IndexedDocumentVersionRow.document_version_id
                        == DocumentRow.current_version_id,
                        IndexedDocumentVersionRow.index_revision_id
                        == KnowledgeBaseRow.active_index_revision_id,
                    ),
                )
                .where(
                    DocumentRow.workspace_id == self._workspace_id,
                    KnowledgeBaseRow.workspace_id == self._workspace_id,
                    KnowledgeBaseRow.deleted_at.is_(None),
                    DocumentRow.id == document_id,
                )
            )
        ).one_or_none()
        if row is None:
            return None
        document, version, target = row
        if (
            document.deleted_at is not None
            or version is None
            or target is None
            or target.build_status is not IndexBuildStatus.READY
        ):
            raise ResourceStateConflictError(
                "a complete chunk snapshot is not ready for the current document version"
            )

        statement = select(IndexChunkRow).where(
            IndexChunkRow.workspace_id == self._workspace_id,
            IndexChunkRow.indexed_document_version_id == target.id,
        )
        if after is not None:
            if len(after) != 2:
                raise ValueError("chunk cursor must contain ordinal and id")
            ordinal, chunk_id = int(after[0]), UUID(after[1])
            statement = statement.where(
                or_(
                    IndexChunkRow.ordinal > ordinal,
                    and_(IndexChunkRow.ordinal == ordinal, IndexChunkRow.id > chunk_id),
                )
            )
        rows = tuple(
            (
                await self._session.execute(
                    statement.order_by(IndexChunkRow.ordinal, IndexChunkRow.id).limit(
                        limit + 1
                    )
                )
            ).scalars()
        )
        has_more = len(rows) > limit
        rows = rows[:limit]
        total_chunks = await self._session.scalar(
            select(func.count()).select_from(IndexChunkRow).where(
                IndexChunkRow.workspace_id == self._workspace_id,
                IndexChunkRow.indexed_document_version_id == target.id,
            )
        )
        chunk_ids = tuple(item.id for item in rows)
        representations: dict[UUID, set[str]] = {item.id: set() for item in rows}
        if chunk_ids:
            vector_rows = (
                await self._session.execute(
                    select(
                        VectorRecordRow.index_chunk_id,
                        VectorRecordRow.representation_kind,
                    ).where(
                        VectorRecordRow.workspace_id == self._workspace_id,
                        VectorRecordRow.index_chunk_id.in_(chunk_ids),
                    )
                )
            ).all()
            for chunk_id, representation_kind in vector_rows:
                representations[chunk_id].add(representation_kind)

        relation_rows = ()
        if chunk_ids:
            relation_rows = (
                await self._session.execute(
                    select(IndexChunkAssetRelationRow).where(
                        IndexChunkAssetRelationRow.workspace_id == self._workspace_id,
                        IndexChunkAssetRelationRow.indexed_document_version_id == target.id,
                        IndexChunkAssetRelationRow.chunk_id.in_(chunk_ids),
                    ).order_by(
                        IndexChunkAssetRelationRow.chunk_id,
                        IndexChunkAssetRelationRow.ordinal,
                        IndexChunkAssetRelationRow.id,
                    )
                )
            ).scalars().all()
        asset_ids = {
            relation.asset_id for relation in relation_rows
        } | {item.index_asset_id for item in rows if item.index_asset_id is not None}
        assets: dict[UUID, DocumentChunkAsset] = {}
        if asset_ids:
            asset_rows = (
                await self._session.execute(
                    select(IndexAssetRow).where(
                        IndexAssetRow.workspace_id == self._workspace_id,
                        IndexAssetRow.indexed_document_version_id == target.id,
                        IndexAssetRow.id.in_(asset_ids),
                    )
                )
            ).scalars()
            assets = {
                item.id: DocumentChunkAsset(
                    id=item.id,
                    media_type=item.media_type,
                    checksum_sha256=item.checksum_sha256,
                    width=item.width,
                    height=item.height,
                )
                for item in asset_rows
            }
        related: dict[UUID, list[DocumentChunkRelation]] = {
            item.id: [] for item in rows
        }
        for relation in relation_rows:
            asset = assets.get(relation.asset_id)
            if asset is not None:
                related[relation.chunk_id].append(
                    DocumentChunkRelation(
                        visual_unit_id=relation.visual_unit_id,
                        asset=asset,
                        relation_type=relation.relation_type,
                        confidence_micros=relation.confidence_micros,
                        provenance=relation.provenance,
                        figure_label=relation.figure_label,
                    )
                )
        return DocumentChunkInspection(
            document_id=document.id,
            document_version_id=version.id,
            indexed_document_version_id=target.id,
            index_revision_id=target.index_revision_id,
            total_chunks=total_chunks or 0,
            items=tuple(
                DocumentChunk(
                    id=item.id,
                    ordinal=item.ordinal,
                    modality=item.modality,
                    content=item.content,
                    token_count=item.token_count,
                    source_location=dict(item.source_location),
                    hierarchy=dict(item.hierarchy),
                    source_metadata=dict(item.source_metadata),
                    evidence_group_key=item.evidence_group_key,
                    representations=tuple(sorted(representations[item.id])),
                    asset=assets.get(item.index_asset_id),
                    related_visuals=tuple(related[item.id]),
                    excluded_at=item.excluded_at,
                )
                for item in rows
            ),
            next_values=(str(rows[-1].ordinal), str(rows[-1].id))
            if has_more and rows
            else None,
        )

    async def list(
        self,
        *,
        kb_id: UUID,
        limit: int,
        sort: str,
        after: tuple[str, ...] | None,
    ) -> Page[Document]:
        self._ensure_active()
        kb_exists = await self._session.scalar(
            select(KnowledgeBaseRow.id).where(
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.id == kb_id,
                KnowledgeBaseRow.deleted_at.is_(None),
            )
        )
        if kb_exists is None:
            return Page(items=())
        descending = sort.startswith("-")
        field = sort.removeprefix("-")
        column = {"created_at": DocumentRow.created_at, "display_name": DocumentRow.display_name}[field]
        statement = (
            select(DocumentRow, DocumentVersionRow)
            .outerjoin(DocumentVersionRow, DocumentVersionRow.id == DocumentRow.current_version_id)
            .where(
                DocumentRow.workspace_id == self._workspace_id,
                DocumentRow.kb_id == kb_id,
                DocumentRow.deleted_at.is_(None),
            )
        )
        statement = _with_after(statement, column, DocumentRow.id, after, descending)
        ordering = column.desc() if descending else column.asc()
        id_ordering = DocumentRow.id.desc() if descending else DocumentRow.id.asc()
        rows = (await self._session.execute(statement.order_by(ordering, id_ordering).limit(limit + 1))).all()
        has_more = len(rows) > limit
        rows = rows[:limit]
        items = tuple(_document(row[0], row[1]) for row in rows)
        next_values = _cursor_values(items[-1], field) if has_more and items else None
        return Page(items=items, next_values=next_values)

    async def reserve_version(
        self,
        *,
        kb_id: UUID,
        document_id: UUID | None,
        display_name: str,
        source: DocumentSource,
    ) -> DocumentMutationResult:
        self._ensure_active()
        await self._session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
            {
                "key": (
                    f"document-content:{self._workspace_id}:{kb_id}"
                )
            },
        )
        kb_exists = await self._session.scalar(
            select(KnowledgeBaseRow.id).where(
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.id == kb_id,
                KnowledgeBaseRow.provisioned_at.is_not(None),
                KnowledgeBaseRow.deleted_at.is_(None),
            )
        )
        if kb_exists is None:
            raise ResourceStateConflictError("knowledge base is unavailable")
        if document_id is None:
            duplicate = await self._session.scalar(
                select(DocumentRow.id)
                .join(
                    DocumentVersionRow,
                    DocumentVersionRow.document_id == DocumentRow.id,
                )
                .where(
                    DocumentRow.workspace_id == self._workspace_id,
                    DocumentRow.kb_id == kb_id,
                    DocumentRow.deleted_at.is_(None),
                    or_(
                        and_(
                            DocumentRow.current_version_id.is_not(None),
                            DocumentVersionRow.source_status
                            == DocumentSourceStatus.AVAILABLE,
                            DocumentVersionRow.checksum_sha256
                            == source.checksum_sha256,
                        ),
                        and_(
                            DocumentRow.current_version_id.is_(None),
                            DocumentVersionRow.source_status
                            == DocumentSourceStatus.UNAVAILABLE,
                            DocumentVersionRow.checksum_sha256
                            == source.checksum_sha256,
                        ),
                    ),
                )
                .order_by(DocumentRow.id)
            )
            if duplicate is not None:
                raise DuplicateDocumentError(duplicate)
            document = DocumentRow(
                workspace_id=self._workspace_id,
                kb_id=kb_id,
                display_name=display_name,
            )
            self._session.add(document)
            await self._session.flush()
            version_number = 1
        else:
            document = await self._session.scalar(
                select(DocumentRow).where(
                    DocumentRow.workspace_id == self._workspace_id,
                    DocumentRow.kb_id == kb_id,
                    DocumentRow.id == document_id,
                ).with_for_update()
            )
            if document is None:
                raise ResourceStateConflictError("document is unavailable")
            maximum = await self._session.scalar(
                select(func.max(DocumentVersionRow.version_number)).where(
                    DocumentVersionRow.workspace_id == self._workspace_id,
                    DocumentVersionRow.document_id == document.id,
                )
            )
            version_number = (maximum or 0) + 1
            document.display_name = display_name
            document.updated_at = datetime.now(UTC)
        version = DocumentVersionRow(
            workspace_id=self._workspace_id,
            kb_id=kb_id,
            document_id=document.id,
            version_number=version_number,
            source_status=DocumentSourceStatus.UNAVAILABLE,
            checksum_sha256=source.checksum_sha256,
            storage_uri=source.storage_uri,
            original_filename=source.original_filename,
            media_type=source.media_type,
            size_bytes=source.size_bytes,
        )
        self._session.add(version)
        await self._session.flush()
        return DocumentMutationResult(
            document=_document(document, None),
            document_version_id=version.id,
        )

    async def activate_version(
        self,
        *,
        document_id: UUID,
        document_version_id: UUID,
    ) -> DocumentMutationResult:
        self._ensure_active()
        document = await self._session.scalar(
            select(DocumentRow).where(
                DocumentRow.workspace_id == self._workspace_id,
                DocumentRow.id == document_id,
            ).with_for_update()
        )
        version = await self._session.scalar(
            select(DocumentVersionRow).where(
                DocumentVersionRow.workspace_id == self._workspace_id,
                DocumentVersionRow.document_id == document_id,
                DocumentVersionRow.id == document_version_id,
            ).with_for_update()
        )
        if document is None or version is None:
            raise ResourceStateConflictError("reserved document version is unavailable")
        if version.source_status is not DocumentSourceStatus.UNAVAILABLE:
            raise ResourceStateConflictError("reserved document version is not provisional")

        allocation = await self._session.execute(
            update(KnowledgeBaseRow)
            .where(
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.id == document.kb_id,
                KnowledgeBaseRow.provisioned_at.is_not(None),
                KnowledgeBaseRow.deleted_at.is_(None),
            )
            .values(
                source_change_seq=KnowledgeBaseRow.source_change_seq + 1,
                updated_at=func.now(),
            )
            .returning(
                KnowledgeBaseRow.source_change_seq,
                KnowledgeBaseRow.active_index_revision_id,
            )
        )
        allocated = allocation.one_or_none()
        if allocated is None or allocated.active_index_revision_id is None:
            raise ResourceStateConflictError("knowledge base has no active revision")
        version.source_status = DocumentSourceStatus.AVAILABLE
        document.current_version_id = version.id
        document.deleted_at = None
        document.updated_at = datetime.now(UTC)
        change = SourceChangeRow(
            workspace_id=self._workspace_id,
            kb_id=document.kb_id,
            source_change_seq=allocated.source_change_seq,
            document_id=document.id,
            document_version_id=version.id,
            change_kind=SourceChangeKind.UPSERT,
        )
        self._session.add(change)
        target = IndexedDocumentVersionRow(
            workspace_id=self._workspace_id,
            kb_id=document.kb_id,
            document_id=document.id,
            document_version_id=version.id,
            index_revision_id=allocated.active_index_revision_id,
            source_change_seq=allocated.source_change_seq,
            build_status=IndexBuildStatus.QUEUED,
            serving_status=IndexServingStatus.CANDIDATE,
        )
        self._session.add(target)
        await self._session.flush()
        job = IndexingJobRow(
            workspace_id=self._workspace_id,
            kb_id=document.kb_id,
            indexed_document_version_id=target.id,
            status=JobStatus.QUEUED,
            phase="queued",
        )
        self._session.add(job)
        await self._session.flush()
        return DocumentMutationResult(
            document=_document(document, version),
            document_version_id=version.id,
            source_change_id=change.id,
            source_change_seq=allocated.source_change_seq,
            indexed_document_version_id=target.id,
            index_revision_id=allocated.active_index_revision_id,
            job_id=job.id,
            job_status=job.status.value,
        )

    async def soft_delete(self, document_id: UUID) -> DocumentMutationResult | None:
        self._ensure_active()
        document = await self._session.scalar(
            select(DocumentRow)
            .join(KnowledgeBaseRow, KnowledgeBaseRow.id == DocumentRow.kb_id)
            .where(
                DocumentRow.workspace_id == self._workspace_id,
                DocumentRow.id == document_id,
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.deleted_at.is_(None),
            ).with_for_update()
        )
        if document is None:
            return None
        current = None
        if document.current_version_id is not None:
            current = await self._session.get(DocumentVersionRow, document.current_version_id)
        await self._schedule_file_cleanup(document.id)
        if document.deleted_at is not None:
            return DocumentMutationResult(document=_document(document, current))
        allocation = await self._session.execute(
            update(KnowledgeBaseRow)
            .where(
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.id == document.kb_id,
            )
            .values(
                source_change_seq=KnowledgeBaseRow.source_change_seq + 1,
                updated_at=func.now(),
            )
            .returning(
                KnowledgeBaseRow.source_change_seq,
                KnowledgeBaseRow.active_index_revision_id,
            )
        )
        allocated = allocation.one()
        now = datetime.now(UTC)
        document.deleted_at = now
        document.updated_at = now
        await self._session.execute(
            update(DocumentVersionRow)
            .where(
                DocumentVersionRow.workspace_id == self._workspace_id,
                DocumentVersionRow.document_id == document.id,
            )
            .values(source_status=DocumentSourceStatus.DELETED)
        )
        change = SourceChangeRow(
            workspace_id=self._workspace_id,
            kb_id=document.kb_id,
            source_change_seq=allocated.source_change_seq,
            document_id=document.id,
            document_version_id=None,
            change_kind=SourceChangeKind.DELETE,
        )
        self._session.add(change)
        if allocated.active_index_revision_id is not None:
            targets = select(IndexedDocumentVersionRow.id).where(
                IndexedDocumentVersionRow.workspace_id == self._workspace_id,
                IndexedDocumentVersionRow.document_id == document.id,
                IndexedDocumentVersionRow.index_revision_id == allocated.active_index_revision_id,
            )
            await self._session.execute(
                update(IndexedDocumentVersionRow)
                .where(IndexedDocumentVersionRow.id.in_(targets))
                .values(serving_status=IndexServingStatus.RETIRED, updated_at=func.now())
            )
            await self._session.execute(
                update(IndexingJobRow)
                .where(
                    IndexingJobRow.indexed_document_version_id.in_(targets),
                    IndexingJobRow.status.in_((JobStatus.QUEUED, JobStatus.RUNNING)),
                )
                .values(
                    status=JobStatus.CANCELLED,
                    phase="cancelled",
                    claimed_by=None,
                    claimed_at=None,
                    heartbeat_at=None,
                    next_attempt_at=None,
                    updated_at=func.now(),
                )
            )
        await self._session.flush()
        if current is not None:
            current.source_status = DocumentSourceStatus.DELETED
        return DocumentMutationResult(
            document=_document(document, current),
            source_change_id=change.id,
            source_change_seq=allocated.source_change_seq,
            index_revision_id=allocated.active_index_revision_id,
        )

    async def exclude_chunk(
        self, *, document_id: UUID, chunk_id: UUID
    ) -> datetime | None:
        self._ensure_active()
        document = await self._session.scalar(
            select(DocumentRow)
            .join(KnowledgeBaseRow, KnowledgeBaseRow.id == DocumentRow.kb_id)
            .where(
                DocumentRow.workspace_id == self._workspace_id,
                DocumentRow.id == document_id,
                DocumentRow.deleted_at.is_(None),
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.deleted_at.is_(None),
            )
            .with_for_update(of=DocumentRow)
        )
        if document is None:
            return None
        target = await self._session.scalar(
            select(IndexedDocumentVersionRow).where(
                IndexedDocumentVersionRow.workspace_id == self._workspace_id,
                IndexedDocumentVersionRow.kb_id == document.kb_id,
                IndexedDocumentVersionRow.document_id == document.id,
                IndexedDocumentVersionRow.document_version_id
                == document.current_version_id,
                IndexedDocumentVersionRow.build_status == IndexBuildStatus.READY,
                IndexedDocumentVersionRow.serving_status
                == IndexServingStatus.SERVING,
            )
        )
        if target is None:
            raise ResourceStateConflictError(
                "the current document version has no serving chunk snapshot"
            )
        chunk = await self._session.scalar(
            select(IndexChunkRow).where(
                IndexChunkRow.workspace_id == self._workspace_id,
                IndexChunkRow.kb_id == document.kb_id,
                IndexChunkRow.indexed_document_version_id == target.id,
                IndexChunkRow.id == chunk_id,
            ).with_for_update()
        )
        if chunk is None:
            return None
        if chunk.excluded_at is None:
            chunk.excluded_at = datetime.now(UTC)
            await self._session.flush()
        return chunk.excluded_at

    async def _schedule_file_cleanup(self, document_id: UUID) -> None:
        versions = (
            await self._session.execute(
                select(DocumentVersionRow.id, DocumentVersionRow.storage_uri).where(
                    DocumentVersionRow.workspace_id == self._workspace_id,
                    DocumentVersionRow.document_id == document_id,
                )
            )
        ).all()
        if not versions:
            return
        await self._session.execute(
            pg_insert(SourceFileCleanupRow)
            .values(
                [
                    {
                        "workspace_id": self._workspace_id,
                        "document_version_id": version.id,
                        "storage_uri": version.storage_uri,
                        "reason": "document_deleted",
                    }
                    for version in versions
                ]
            )
            .on_conflict_do_nothing(index_elements=["document_version_id"])
        )


class SqlAlchemyContentMutationRepository:
    def __init__(self, session: AsyncSession, workspace_id: UUID, ensure_active: Callable[[], None]) -> None:
        self._session = session
        self._workspace_id = workspace_id
        self._ensure_active = ensure_active

    async def lock(self, scope: IdempotencyScope) -> None:
        self._ensure_active()
        key = f"{scope.principal_id}\x1f{scope.client_id}\x1f{scope.endpoint}\x1f{scope.idempotency_key}"
        await self._session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
            {"key": key},
        )

    async def get(self, scope: IdempotencyScope) -> ContentMutation | None:
        self._ensure_active()
        row = await self._session.scalar(
            select(ContentMutationRow).where(
                ContentMutationRow.workspace_id == self._workspace_id,
                ContentMutationRow.principal_id == scope.principal_id,
                ContentMutationRow.client_id == scope.client_id,
                ContentMutationRow.endpoint == scope.endpoint,
                ContentMutationRow.idempotency_key == scope.idempotency_key,
            )
        )
        return _mutation(row) if row is not None else None

    async def add(
        self,
        *,
        scope: IdempotencyScope,
        request_hash: str,
        operation: str,
        status: str,
        result: DocumentMutationResult | KnowledgeBase,
    ) -> ContentMutation:
        self._ensure_active()
        values = _result_ids(result)
        row = ContentMutationRow(
            workspace_id=self._workspace_id,
            principal_id=scope.principal_id,
            client_id=scope.client_id,
            endpoint=scope.endpoint,
            idempotency_key=scope.idempotency_key,
            request_hash=request_hash,
            operation=operation,
            status=status,
            **values,
        )
        self._session.add(row)
        await self._session.flush()
        return _mutation(row)

    async def complete(
        self, scope: IdempotencyScope, result: DocumentMutationResult
    ) -> ContentMutation:
        self._ensure_active()
        row = await self._session.scalar(
            select(ContentMutationRow).where(
                ContentMutationRow.workspace_id == self._workspace_id,
                ContentMutationRow.principal_id == scope.principal_id,
                ContentMutationRow.client_id == scope.client_id,
                ContentMutationRow.endpoint == scope.endpoint,
                ContentMutationRow.idempotency_key == scope.idempotency_key,
            ).with_for_update()
        )
        if row is None:
            raise ResourceStateConflictError("content mutation reservation is missing")
        for name, value in _result_ids(result).items():
            setattr(row, name, value)
        row.status = "completed"
        row.failure_code = None
        row.failed_at = None
        row.updated_at = datetime.now(UTC)
        await self._session.flush()
        return _mutation(row)

    async def add_indexing_retry(
        self,
        *,
        scope: IdempotencyScope,
        request_hash: str,
        kb_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
        indexed_document_version_id: UUID,
        index_revision_id: UUID,
        job_id: UUID,
    ) -> ContentMutation:
        self._ensure_active()
        row = ContentMutationRow(
            workspace_id=self._workspace_id,
            principal_id=scope.principal_id,
            client_id=scope.client_id,
            endpoint=scope.endpoint,
            idempotency_key=scope.idempotency_key,
            request_hash=request_hash,
            operation="indexing_job.retry",
            status="completed",
            kb_id=kb_id,
            document_id=document_id,
            document_version_id=document_version_id,
            source_change_id=None,
            indexed_document_version_id=indexed_document_version_id,
            index_revision_id=index_revision_id,
            job_id=job_id,
        )
        self._session.add(row)
        await self._session.flush()
        return _mutation(row)


class SqlAlchemyFileConsistencyRepository:
    def __init__(
        self,
        session: AsyncSession,
        workspace_id: UUID,
        ensure_active: Callable[[], None],
    ) -> None:
        self._session = session
        self._workspace_id = workspace_id
        self._ensure_active = ensure_active

    async def delete_expired_cleanup_records(
        self,
        *,
        before: datetime,
        limit: int,
    ) -> int:
        self._ensure_active()
        ids = tuple(
            (
                await self._session.execute(
                    select(SourceFileCleanupRow.id)
                    .where(
                        SourceFileCleanupRow.workspace_id == self._workspace_id,
                        SourceFileCleanupRow.status.in_(("completed", "failed")),
                        SourceFileCleanupRow.updated_at <= before,
                    )
                    .order_by(SourceFileCleanupRow.updated_at, SourceFileCleanupRow.id)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            ).scalars()
        )
        if not ids:
            return 0
        result = await self._session.execute(
            delete(SourceFileCleanupRow).where(
                SourceFileCleanupRow.workspace_id == self._workspace_id,
                SourceFileCleanupRow.id.in_(ids),
            )
        )
        return int(result.rowcount or 0)

    async def list_references(self) -> tuple[SourceFileReference, ...]:
        self._ensure_active()
        rows = (
            await self._session.execute(
                select(DocumentVersionRow).where(
                    DocumentVersionRow.workspace_id == self._workspace_id
                )
            )
        ).scalars()
        return tuple(
            SourceFileReference(
                document_id=row.document_id,
                document_version_id=row.id,
                source_status=row.source_status.value,
                storage_uri=row.storage_uri,
                checksum_sha256=row.checksum_sha256,
                size_bytes=row.size_bytes,
            )
            for row in rows
        )

    async def list_pending_mutations(
        self, *, limit: int
    ) -> tuple[PendingFileMutation, ...]:
        self._ensure_active()
        rows = (
            await self._session.execute(
                select(ContentMutationRow, DocumentVersionRow)
                .join(
                    DocumentVersionRow,
                    DocumentVersionRow.id == ContentMutationRow.document_version_id,
                )
                .where(
                    ContentMutationRow.workspace_id == self._workspace_id,
                    ContentMutationRow.status == "pending",
                    ContentMutationRow.operation == "document.version.reserve",
                    DocumentVersionRow.source_status == DocumentSourceStatus.UNAVAILABLE,
                )
                .order_by(ContentMutationRow.created_at, ContentMutationRow.id)
                .limit(limit)
            )
        ).all()
        return tuple(
            PendingFileMutation(
                scope=IdempotencyScope(
                    mutation.principal_id,
                    mutation.client_id,
                    mutation.endpoint,
                    mutation.idempotency_key,
                ),
                document_id=mutation.document_id,
                document_version_id=version.id,
                storage_uri=version.storage_uri,
                checksum_sha256=version.checksum_sha256,
                size_bytes=version.size_bytes,
                reserved_at=mutation.created_at,
            )
            for mutation, version in rows
            if mutation.document_id is not None
        )

    async def list_due_cleanup(
        self, *, now: datetime, limit: int
    ) -> tuple[SourceFileCleanup, ...]:
        self._ensure_active()
        rows = (
            await self._session.execute(
                select(SourceFileCleanupRow)
                .where(
                    SourceFileCleanupRow.workspace_id == self._workspace_id,
                    SourceFileCleanupRow.status == "pending",
                    SourceFileCleanupRow.next_attempt_at <= now,
                )
                .order_by(SourceFileCleanupRow.next_attempt_at, SourceFileCleanupRow.id)
                .limit(limit)
            )
        ).scalars()
        return tuple(_file_cleanup(row) for row in rows)

    async def schedule_cleanup(
        self,
        *,
        document_version_id: UUID,
        storage_uri: str,
        reason: str,
    ) -> None:
        self._ensure_active()
        await self._session.execute(
            pg_insert(SourceFileCleanupRow)
            .values(
                workspace_id=self._workspace_id,
                document_version_id=document_version_id,
                storage_uri=storage_uri,
                reason=reason,
            )
            .on_conflict_do_nothing(index_elements=["document_version_id"])
        )

    async def complete_cleanup(self, cleanup_id: UUID, *, now: datetime) -> bool:
        self._ensure_active()
        result = await self._session.execute(
            update(SourceFileCleanupRow)
            .where(
                SourceFileCleanupRow.workspace_id == self._workspace_id,
                SourceFileCleanupRow.id == cleanup_id,
                SourceFileCleanupRow.status == "pending",
            )
            .values(
                status="completed",
                completed_at=now,
                last_error_code=None,
                updated_at=now,
            )
        )
        return bool(result.rowcount)

    async def fail_cleanup(
        self,
        cleanup_id: UUID,
        *,
        expected_attempt_count: int,
        error_code: str,
        next_attempt_at: datetime,
        terminal: bool,
    ) -> bool:
        self._ensure_active()
        result = await self._session.execute(
            update(SourceFileCleanupRow)
            .where(
                SourceFileCleanupRow.workspace_id == self._workspace_id,
                SourceFileCleanupRow.id == cleanup_id,
                SourceFileCleanupRow.status == "pending",
                SourceFileCleanupRow.attempt_count == expected_attempt_count,
            )
            .values(
                status="failed" if terminal else "pending",
                attempt_count=SourceFileCleanupRow.attempt_count + 1,
                last_error_code=error_code,
                next_attempt_at=next_attempt_at,
                updated_at=func.now(),
            )
        )
        return bool(result.rowcount)

    async def fail_pending_file_mutation(
        self,
        *,
        scope: IdempotencyScope,
        document_version_id: UUID,
        failure_code: str,
        failed_at: datetime,
    ) -> bool:
        """Atomically make one irrecoverable reservation terminal.

        The mutation row is the serialization point shared with activation.  A
        concurrent successful activation therefore wins without being replaced
        by a janitor failure transition.
        """

        self._ensure_active()
        mutation = await self._session.scalar(
            select(ContentMutationRow)
            .where(
                ContentMutationRow.workspace_id == self._workspace_id,
                ContentMutationRow.principal_id == scope.principal_id,
                ContentMutationRow.client_id == scope.client_id,
                ContentMutationRow.endpoint == scope.endpoint,
                ContentMutationRow.idempotency_key == scope.idempotency_key,
            )
            .with_for_update()
        )
        if (
            mutation is None
            or mutation.status != "pending"
            or mutation.operation != "document.version.reserve"
            or mutation.document_version_id != document_version_id
        ):
            return False

        version = await self._session.scalar(
            select(DocumentVersionRow)
            .where(
                DocumentVersionRow.workspace_id == self._workspace_id,
                DocumentVersionRow.id == document_version_id,
            )
            .with_for_update()
        )
        if (
            version is None
            or version.source_status is not DocumentSourceStatus.UNAVAILABLE
        ):
            return False
        document = await self._session.scalar(
            select(DocumentRow)
            .where(
                DocumentRow.workspace_id == self._workspace_id,
                DocumentRow.id == version.document_id,
            )
            .with_for_update()
        )
        if document is None:
            return False

        version.source_status = DocumentSourceStatus.DELETED
        if document.current_version_id is None:
            document.deleted_at = failed_at
        document.updated_at = failed_at
        mutation.status = "failed"
        mutation.failure_code = failure_code
        mutation.failed_at = failed_at
        mutation.updated_at = failed_at
        await self._session.execute(
            pg_insert(SourceFileCleanupRow)
            .values(
                workspace_id=self._workspace_id,
                document_version_id=version.id,
                storage_uri=version.storage_uri,
                reason="pending_mutation_failed",
            )
            .on_conflict_do_nothing(index_elements=["document_version_id"])
        )
        await self._session.flush()
        return True

    async def compensate_missing_file(self, document_version_id: UUID) -> bool:
        self._ensure_active()
        version = await self._session.scalar(
            select(DocumentVersionRow)
            .where(
                DocumentVersionRow.workspace_id == self._workspace_id,
                DocumentVersionRow.id == document_version_id,
            )
            .with_for_update()
        )
        if version is None or version.source_status is not DocumentSourceStatus.AVAILABLE:
            return False
        document = await self._session.scalar(
            select(DocumentRow)
            .where(
                DocumentRow.workspace_id == self._workspace_id,
                DocumentRow.id == version.document_id,
            )
            .with_for_update()
        )
        if document is None:
            return False
        allocation = await self._session.execute(
            update(KnowledgeBaseRow)
            .where(
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.id == version.kb_id,
            )
            .values(
                source_change_seq=KnowledgeBaseRow.source_change_seq + 1,
                updated_at=func.now(),
            )
            .returning(KnowledgeBaseRow.source_change_seq)
        )
        source_change_seq = allocation.scalar_one()
        version.source_status = DocumentSourceStatus.UNAVAILABLE
        self._session.add(
            SourceChangeRow(
                workspace_id=self._workspace_id,
                kb_id=version.kb_id,
                source_change_seq=source_change_seq,
                document_id=version.document_id,
                document_version_id=None,
                change_kind=SourceChangeKind.DELETE,
            )
        )
        targets = select(IndexedDocumentVersionRow.id).where(
            IndexedDocumentVersionRow.workspace_id == self._workspace_id,
            IndexedDocumentVersionRow.document_version_id == version.id,
        )
        await self._session.execute(
            update(IndexedDocumentVersionRow)
            .where(IndexedDocumentVersionRow.id.in_(targets))
            .values(serving_status=IndexServingStatus.RETIRED, updated_at=func.now())
        )
        await self._session.execute(
            update(IndexingJobRow)
            .where(
                IndexingJobRow.indexed_document_version_id.in_(targets),
                IndexingJobRow.status.in_((JobStatus.QUEUED, JobStatus.RUNNING)),
            )
            .values(
                status=JobStatus.CANCELLED,
                phase="source_missing",
                claimed_by=None,
                claimed_at=None,
                heartbeat_at=None,
                next_attempt_at=None,
                updated_at=func.now(),
            )
        )
        await self._session.flush()
        return True


def _embedding_matches(row: EmbeddingSpaceRow, workspace_id: UUID, value: EmbeddingSpaceDefinition) -> bool:
    return (
        row.workspace_id == workspace_id
        and row.provider_identity == value.provider_identity
        and row.endpoint_identity == value.endpoint_identity
        and row.requested_model == value.requested_model
        and row.resolved_model == value.resolved_model
        and row.model_version == value.model_version
        and row.deployment_revision == value.deployment_revision
        and row.dimension == value.dimension
        and row.distance_metric == value.distance_metric
        and row.vector_data_type == value.vector_data_type
        and row.normalization == value.normalization
        and row.configuration_fingerprint == value.configuration_fingerprint
        and row.tokenizer_fingerprint == value.tokenizer_fingerprint
        and row.model_profile_revision_id == value.model_profile_revision_id
    )


def _knowledge_base(
    row: KnowledgeBaseRow,
    embedding_space_id: UUID,
    parser_config: dict[str, Any],
    chunking_config: dict[str, Any],
    embedding: KnowledgeBaseEmbeddingSummary,
) -> KnowledgeBase:
    assert row.active_index_revision_id is not None
    assert row.provisioned_at is not None
    return KnowledgeBase(
        id=row.id,
        workspace_id=row.workspace_id,
        name=row.name,
        source_change_seq=row.source_change_seq,
        active_index_revision_id=row.active_index_revision_id,
        embedding_space_id=embedding_space_id,
        parser_config=dict(parser_config),
        chunking_config=dict(chunking_config),
        retrieval_defaults=dict(row.retrieval_defaults),
        answer_policy_defaults=dict(row.answer_policy_defaults),
        provisioned_at=row.provisioned_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
        embedding=embedding,
        deleted_at=row.deleted_at,
    )


def _embedding_summary_from_spaces(
    text_space: EmbeddingSpaceRow,
    cross_modal_space: EmbeddingSpaceRow | None,
) -> KnowledgeBaseEmbeddingSummary:
    text = EmbeddingRoleSummary(
        embedding_space_id=text_space.id,
        profile_revision_id=text_space.model_profile_revision_id,
        dimension=text_space.dimension,
    )
    cross_modal = (
        EmbeddingRoleSummary(
            embedding_space_id=cross_modal_space.id,
            profile_revision_id=cross_modal_space.model_profile_revision_id,
            dimension=cross_modal_space.dimension,
        )
        if cross_modal_space is not None
        else None
    )
    if cross_modal is None:
        strategy = "text_only"
    elif cross_modal.embedding_space_id == text.embedding_space_id:
        strategy = "unified_multimodal"
    else:
        strategy = "dual_space"
    return KnowledgeBaseEmbeddingSummary(
        strategy=strategy,
        text=text,
        cross_modal=cross_modal,
    )


def _version(row: DocumentVersionRow) -> DocumentVersion:
    return DocumentVersion(
        id=row.id,
        document_id=row.document_id,
        version_number=row.version_number,
        source_status=row.source_status.value,
        checksum_sha256=row.checksum_sha256,
        storage_uri=row.storage_uri,
        original_filename=row.original_filename,
        media_type=row.media_type,
        size_bytes=row.size_bytes,
        created_at=row.created_at,
    )


def _v2_manifest_counts(manifest) -> dict[str, int | None]:
    empty = {
        "composite_chunk_count": None,
        "visual_unit_count": None,
        "relation_count": None,
        "text_representation_count": None,
        "native_image_representation_count": None,
        "table_representation_count": None,
    }
    if manifest is None:
        return empty
    units = tuple(manifest.unit_plan or ())
    representations = tuple(manifest.representation_matrix or ())
    return {
        "composite_chunk_count": sum(
            item.get("modality") == "text" for item in units
        ),
        "visual_unit_count": sum(
            item.get("modality") in {"image", "table"} for item in units
        ),
        "relation_count": manifest.relation_count,
        "text_representation_count": sum(
            item.get("representation_kind") in {"text", "ocr_text"}
            for item in representations
        ),
        "native_image_representation_count": sum(
            item.get("representation_kind") == "native_image"
            for item in representations
        ),
        "table_representation_count": sum(
            item.get("representation_kind") in {"table_text", "table_image"}
            for item in representations
        ),
    }


def _document(row: DocumentRow, version: DocumentVersionRow | None) -> Document:
    return Document(
        id=row.id,
        workspace_id=row.workspace_id,
        kb_id=row.kb_id,
        display_name=row.display_name,
        current_version=_version(version) if version is not None else None,
        deleted_at=row.deleted_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _mutation(row: ContentMutationRow) -> ContentMutation:
    return ContentMutation(
        scope=IdempotencyScope(row.principal_id, row.client_id, row.endpoint, row.idempotency_key),
        request_hash=row.request_hash,
        operation=row.operation,
        status=row.status,
        failure_code=row.failure_code,
        failed_at=row.failed_at,
        kb_id=row.kb_id,
        document_id=row.document_id,
        document_version_id=row.document_version_id,
        source_change_id=row.source_change_id,
        indexed_document_version_id=row.indexed_document_version_id,
        index_revision_id=row.index_revision_id,
        job_id=row.job_id,
    )


def _file_cleanup(row: SourceFileCleanupRow) -> SourceFileCleanup:
    return SourceFileCleanup(
        id=row.id,
        document_version_id=row.document_version_id,
        storage_uri=row.storage_uri,
        reason=row.reason,
        status=row.status,
        attempt_count=row.attempt_count,
        next_attempt_at=row.next_attempt_at,
    )


def _result_ids(result: DocumentMutationResult | KnowledgeBase) -> dict[str, UUID | None]:
    if isinstance(result, KnowledgeBase):
        return {
            "kb_id": result.id,
            "document_id": None,
            "document_version_id": None,
            "source_change_id": None,
            "indexed_document_version_id": None,
            "index_revision_id": result.active_index_revision_id,
            "job_id": None,
        }
    return {
        "kb_id": result.document.kb_id,
        "document_id": result.document.id,
        "document_version_id": result.document_version_id,
        "source_change_id": result.source_change_id,
        "indexed_document_version_id": result.indexed_document_version_id,
        "index_revision_id": result.index_revision_id,
        "job_id": result.job_id,
    }


def _with_after(statement, column, id_column, after: tuple[str, ...] | None, descending: bool):
    if after is None:
        return statement
    if len(after) != 2:
        raise ValueError("cursor must contain sort value and id")
    raw_value, raw_id = after
    value: Any = datetime.fromisoformat(raw_value) if getattr(column.type, "python_type", None) is datetime else raw_value
    item_id = UUID(raw_id)
    compare = column < value if descending else column > value
    compare_id = id_column < item_id if descending else id_column > item_id
    return statement.where(or_(compare, and_(column == value, compare_id)))


def _cursor_values(item: KnowledgeBase | Document, field: str) -> tuple[str, str]:
    value = getattr(item, field)
    rendered = value.isoformat() if isinstance(value, datetime) else str(value)
    return rendered, str(item.id)


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))
