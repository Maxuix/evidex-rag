"""SQLAlchemy persistence for the current-only entity graph projection."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_kb.db.models import (
    Document as DocumentRow,
    DocumentSourceStatus,
    DocumentVersion as DocumentVersionRow,
    GraphEntityMention as GraphEntityMentionRow,
    GraphRelationAssertion as GraphRelationAssertionRow,
    IndexBuildStatus,
    IndexChunk as IndexChunkRow,
    IndexGraphChunk as IndexGraphChunkRow,
    IndexRevision as IndexRevisionRow,
    IndexRevisionStatus,
    IndexServingStatus,
    IndexedDocumentVersion as IndexedDocumentVersionRow,
    KnowledgeBase as KnowledgeBaseRow,
    KnowledgeBaseGraphConfig as KnowledgeBaseGraphConfigRow,
    ModelProfile as ModelProfileRow,
    ModelProfileRevision as ModelProfileRevisionRow,
    ModelProvider as ModelProviderRow,
)
from rag_kb.domain import (
    GRAPH_EXTRACTOR_VERSION,
    GraphChunkExtraction,
    GraphConfigSnapshot,
    GraphConfigStatus,
    GraphEntityType,
    GraphWorkItem,
    GraphWorkKind,
    GraphChunkResultStatus,
    ResourceNotFoundError,
    ResourceStateConflictError,
    allowed_graph_skips,
)


class SqlAlchemyGraphRepository:
    def __init__(
        self,
        session: AsyncSession,
        workspace_id: UUID,
        ensure_active: Callable[[], None],
    ) -> None:
        self._session = session
        self._workspace_id = workspace_id
        self._ensure_active = ensure_active

    async def get_config(self, kb_id: UUID) -> GraphConfigSnapshot | None:
        self._ensure_active()
        row = await self._session.scalar(
            select(KnowledgeBaseGraphConfigRow).where(
                KnowledgeBaseGraphConfigRow.workspace_id == self._workspace_id,
                KnowledgeBaseGraphConfigRow.kb_id == kb_id,
            )
        )
        if row is None:
            return None
        return await self._snapshot(row)

    async def ensure_config(self, kb_id: UUID) -> GraphConfigSnapshot:
        self._ensure_active()
        kb = await self._session.scalar(
            select(KnowledgeBaseRow).where(
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.id == kb_id,
                KnowledgeBaseRow.deleted_at.is_(None),
            )
        )
        if kb is None:
            raise ResourceNotFoundError("knowledge base was not found")
        row = await self._session.scalar(
            select(KnowledgeBaseGraphConfigRow)
            .where(
                KnowledgeBaseGraphConfigRow.workspace_id == self._workspace_id,
                KnowledgeBaseGraphConfigRow.kb_id == kb_id,
            )
            .with_for_update()
        )
        if row is None:
            row = KnowledgeBaseGraphConfigRow(
                workspace_id=self._workspace_id,
                kb_id=kb_id,
                status=GraphConfigStatus.DISABLED.value,
                build_id=uuid4(),
                extractor_version=GRAPH_EXTRACTOR_VERSION,
            )
            self._session.add(row)
            await self._session.flush()
        return await self._snapshot(row)

    async def configure(
        self,
        kb_id: UUID,
        *,
        chat_profile_revision_id: UUID | None,
        enabled: bool,
        extractor_version: str,
        force_rebuild: bool = False,
    ) -> GraphConfigSnapshot:
        self._ensure_active()
        if not extractor_version.strip():
            raise ValueError("graph extractor version must not be empty")
        row = await self._locked_config(kb_id)
        if row is None:
            await self.ensure_config(kb_id)
            row = await self._locked_config(kb_id)
        assert row is not None
        if not enabled:
            row.status = GraphConfigStatus.DISABLED.value
            row.chat_profile_revision_id = None
            row.build_id = uuid4()
            row.preflight_extractor_version = None
            row.last_error_code = None
            row.extractor_version = extractor_version
            await self._session.flush()
            return await self._snapshot(row)

        if chat_profile_revision_id is None:
            raise ResourceStateConflictError(
                "an enabled Graph configuration requires a Chat Profile Revision"
            )
        await self._require_valid_chat_profile(chat_profile_revision_id)
        rotate = (
            force_rebuild
            or row.chat_profile_revision_id != chat_profile_revision_id
            or row.extractor_version != extractor_version
            or row.status == GraphConfigStatus.DISABLED.value
        )
        if rotate:
            row.build_id = uuid4()
            row.preflight_extractor_version = None
        elif row.status in {
            GraphConfigStatus.BUILDING.value,
            GraphConfigStatus.READY.value,
        }:
            # Repeating the same PUT is idempotent.  An explicit retry or
            # force-rebuild operation is required to turn an unchanged ready
            # build back into work.
            return await self._snapshot(row)
        row.chat_profile_revision_id = chat_profile_revision_id
        row.extractor_version = extractor_version
        row.status = GraphConfigStatus.BUILDING.value
        row.last_error_code = None
        await self._session.flush()
        return await self._snapshot(row)

    async def retry(
        self,
        kb_id: UUID,
        *,
        extractor_version: str,
        force_rebuild: bool = False,
    ) -> GraphConfigSnapshot:
        self._ensure_active()
        row = await self._locked_config(kb_id)
        if row is None or row.status == GraphConfigStatus.DISABLED.value:
            raise ResourceStateConflictError("Graph is not enabled for this knowledge base")
        if row.chat_profile_revision_id is None:
            raise ResourceStateConflictError("Graph Chat Profile Revision is missing")
        await self._require_valid_chat_profile(row.chat_profile_revision_id)
        if not extractor_version.strip():
            raise ValueError("graph extractor version must not be empty")
        version_changed = row.extractor_version != extractor_version
        snapshot = await self._snapshot(row)
        if (
            not force_rebuild
            and not version_changed
            and snapshot.status is GraphConfigStatus.READY
            and snapshot.protocol_skipped_count == 0
        ):
            raise ResourceStateConflictError(
                "Graph retry requires protocol-skipped chunks or force_rebuild"
            )
        if not force_rebuild and not version_changed:
            skipped_protocol_ids = select(IndexGraphChunkRow.index_chunk_id).where(
                IndexGraphChunkRow.workspace_id == self._workspace_id,
                IndexGraphChunkRow.kb_id == kb_id,
                IndexGraphChunkRow.build_id == row.build_id,
                IndexGraphChunkRow.result_status
                == GraphChunkResultStatus.SKIPPED_PROTOCOL.value,
            )
            await self._session.execute(
                delete(GraphRelationAssertionRow).where(
                    GraphRelationAssertionRow.workspace_id == self._workspace_id,
                    GraphRelationAssertionRow.kb_id == kb_id,
                    GraphRelationAssertionRow.build_id == row.build_id,
                    GraphRelationAssertionRow.index_chunk_id.in_(skipped_protocol_ids),
                )
            )
            await self._session.execute(
                delete(GraphEntityMentionRow).where(
                    GraphEntityMentionRow.workspace_id == self._workspace_id,
                    GraphEntityMentionRow.kb_id == kb_id,
                    GraphEntityMentionRow.build_id == row.build_id,
                    GraphEntityMentionRow.index_chunk_id.in_(skipped_protocol_ids),
                )
            )
            await self._session.execute(
                delete(IndexGraphChunkRow).where(
                    IndexGraphChunkRow.workspace_id == self._workspace_id,
                    IndexGraphChunkRow.kb_id == kb_id,
                    IndexGraphChunkRow.build_id == row.build_id,
                    IndexGraphChunkRow.result_status
                    == GraphChunkResultStatus.SKIPPED_PROTOCOL.value,
                )
            )
        if force_rebuild or version_changed:
            row.build_id = uuid4()
            row.preflight_extractor_version = None
        row.extractor_version = extractor_version
        row.status = GraphConfigStatus.BUILDING.value
        row.last_error_code = None
        await self._session.flush()
        return await self._snapshot(row)

    async def invalidate_for_serving_change(self, kb_id: UUID) -> bool:
        self._ensure_active()
        changed = await self._session.execute(
            update(KnowledgeBaseGraphConfigRow)
            .where(
                KnowledgeBaseGraphConfigRow.workspace_id == self._workspace_id,
                KnowledgeBaseGraphConfigRow.kb_id == kb_id,
                KnowledgeBaseGraphConfigRow.status == GraphConfigStatus.READY.value,
            )
            .values(
                status=GraphConfigStatus.BUILDING.value,
                last_error_code=None,
                updated_at=datetime.now(UTC),
            )
        )
        return changed.rowcount > 0

    async def invalidate_for_indexed_target(self, target_id: UUID) -> bool:
        self._ensure_active()
        kb_id = await self._session.scalar(
            select(IndexedDocumentVersionRow.kb_id).where(
                IndexedDocumentVersionRow.workspace_id == self._workspace_id,
                IndexedDocumentVersionRow.id == target_id,
            )
        )
        if kb_id is None:
            return False
        return await self.invalidate_for_serving_change(kb_id)

    async def next_work_item(self) -> GraphWorkItem | None:
        self._ensure_active()
        configs = (
            await self._session.scalars(
                select(KnowledgeBaseGraphConfigRow)
                .where(
                    KnowledgeBaseGraphConfigRow.workspace_id == self._workspace_id,
                    KnowledgeBaseGraphConfigRow.status == GraphConfigStatus.BUILDING.value,
                )
                .order_by(
                    KnowledgeBaseGraphConfigRow.updated_at,
                    KnowledgeBaseGraphConfigRow.kb_id,
                )
            )
        ).all()
        for row in configs:
            if row.extractor_version != GRAPH_EXTRACTOR_VERSION:
                row.status = GraphConfigStatus.FAILED.value
                row.last_error_code = "graph_extractor_version_stale"
                row.updated_at = datetime.now(UTC)
                await self._session.flush()
                continue
            config = await self._snapshot(row)
            if config.preflight_extractor_version != config.extractor_version:
                return GraphWorkItem(GraphWorkKind.PREFLIGHT, config)
            chunk = await self._next_missing_chunk(config)
            if chunk is not None:
                return GraphWorkItem(GraphWorkKind.CHUNK, config, chunk)
            return GraphWorkItem(GraphWorkKind.FINALIZE, config)
        return None

    async def save_preflight_success(
        self, kb_id: UUID, *, build_id: UUID, extractor_version: str
    ) -> bool:
        self._ensure_active()
        result = await self._session.execute(
            update(KnowledgeBaseGraphConfigRow)
            .where(
                KnowledgeBaseGraphConfigRow.workspace_id == self._workspace_id,
                KnowledgeBaseGraphConfigRow.kb_id == kb_id,
                KnowledgeBaseGraphConfigRow.build_id == build_id,
                KnowledgeBaseGraphConfigRow.status == GraphConfigStatus.BUILDING.value,
                KnowledgeBaseGraphConfigRow.extractor_version == extractor_version,
            )
            .values(
                preflight_extractor_version=extractor_version,
                last_error_code=None,
                updated_at=datetime.now(UTC),
            )
        )
        return result.rowcount > 0

    async def save_chunk_extraction(
        self,
        *,
        kb_id: UUID,
        build_id: UUID,
        index_chunk_id: UUID,
        content_hash: str,
        extractor_version: str,
        extraction: GraphChunkExtraction,
    ) -> bool:
        self._ensure_active()
        config = await self._session.scalar(
            select(KnowledgeBaseGraphConfigRow)
            .where(
                KnowledgeBaseGraphConfigRow.workspace_id == self._workspace_id,
                KnowledgeBaseGraphConfigRow.kb_id == kb_id,
                KnowledgeBaseGraphConfigRow.build_id == build_id,
                KnowledgeBaseGraphConfigRow.status == GraphConfigStatus.BUILDING.value,
                KnowledgeBaseGraphConfigRow.extractor_version == extractor_version,
            )
            .with_for_update()
        )
        if config is None:
            return False
        chunk = await self._session.scalar(
            select(IndexChunkRow).where(
                IndexChunkRow.workspace_id == self._workspace_id,
                IndexChunkRow.kb_id == kb_id,
                IndexChunkRow.id == index_chunk_id,
                IndexChunkRow.content_hash == content_hash,
                IndexChunkRow.modality.in_(("text", "table")),
            )
        )
        if chunk is None:
            return False
        await self._session.execute(
            delete(GraphRelationAssertionRow).where(
                GraphRelationAssertionRow.workspace_id == self._workspace_id,
                GraphRelationAssertionRow.kb_id == kb_id,
                GraphRelationAssertionRow.build_id == build_id,
                GraphRelationAssertionRow.index_chunk_id == index_chunk_id,
            )
        )
        await self._session.execute(
            delete(GraphEntityMentionRow).where(
                GraphEntityMentionRow.workspace_id == self._workspace_id,
                GraphEntityMentionRow.kb_id == kb_id,
                GraphEntityMentionRow.build_id == build_id,
                GraphEntityMentionRow.index_chunk_id == index_chunk_id,
            )
        )
        await self._session.execute(
            delete(IndexGraphChunkRow).where(
                IndexGraphChunkRow.workspace_id == self._workspace_id,
                IndexGraphChunkRow.kb_id == kb_id,
                IndexGraphChunkRow.build_id == build_id,
                IndexGraphChunkRow.index_chunk_id == index_chunk_id,
            )
        )
        graph_chunk = IndexGraphChunkRow(
            workspace_id=self._workspace_id,
            kb_id=kb_id,
            build_id=build_id,
            index_chunk_id=index_chunk_id,
            content_hash=content_hash,
            extractor_version=extractor_version,
            result_status=extraction.result_status.value,
            entity_count=len(extraction.mentions),
            relation_count=len(extraction.relations),
            result_hash=extraction.result_hash or None,
            error_code=extraction.error_code,
        )
        self._session.add(graph_chunk)
        await self._session.flush()
        for mention in extraction.mentions:
            self._session.add(
                GraphEntityMentionRow(
                    workspace_id=self._workspace_id,
                    kb_id=kb_id,
                    build_id=build_id,
                    index_chunk_id=index_chunk_id,
                    mention_id=mention.mention_id,
                    ordinal=mention.ordinal,
                    entity_type=mention.entity_type.value,
                    surface=mention.surface,
                    normalized_surface=mention.normalized_surface,
                    disambiguator=mention.disambiguator,
                    disambiguator_support_start=mention.disambiguator_support_start,
                    disambiguator_support_end=mention.disambiguator_support_end,
                    surface_start=mention.surface_start,
                    surface_end=mention.surface_end,
                    entity_key=mention.entity_key,
                )
            )
        await self._session.flush()
        for relation in extraction.relations:
            self._session.add(
                GraphRelationAssertionRow(
                    workspace_id=self._workspace_id,
                    kb_id=kb_id,
                    build_id=build_id,
                    index_chunk_id=index_chunk_id,
                    relation_id=relation.relation_id,
                    ordinal=relation.ordinal,
                    subject_mention_id=relation.subject_mention_id,
                    object_mention_id=relation.object_mention_id,
                    subject_entity_key=relation.subject_entity_key,
                    object_entity_key=relation.object_entity_key,
                    predicate=relation.predicate,
                    normalized_predicate=relation.normalized_predicate,
                    support_start=relation.support_start,
                    support_end=relation.support_end,
                )
            )
        await self._session.flush()
        return True

    async def mark_failed(
        self,
        kb_id: UUID,
        *,
        build_id: UUID,
        error_code: str,
    ) -> bool:
        self._ensure_active()
        result = await self._session.execute(
            update(KnowledgeBaseGraphConfigRow)
            .where(
                KnowledgeBaseGraphConfigRow.workspace_id == self._workspace_id,
                KnowledgeBaseGraphConfigRow.kb_id == kb_id,
                KnowledgeBaseGraphConfigRow.build_id == build_id,
                KnowledgeBaseGraphConfigRow.status == GraphConfigStatus.BUILDING.value,
            )
            .values(
                status=GraphConfigStatus.FAILED.value,
                last_error_code=error_code,
                updated_at=datetime.now(UTC),
            )
        )
        return result.rowcount > 0

    async def finalize_if_complete(
        self,
        kb_id: UUID,
        *,
        build_id: UUID,
        observed_at: datetime,
    ) -> GraphConfigSnapshot | None:
        del observed_at
        self._ensure_active()
        row = await self._session.scalar(
            select(KnowledgeBaseGraphConfigRow)
            .where(
                KnowledgeBaseGraphConfigRow.workspace_id == self._workspace_id,
                KnowledgeBaseGraphConfigRow.kb_id == kb_id,
                KnowledgeBaseGraphConfigRow.build_id == build_id,
                KnowledgeBaseGraphConfigRow.status == GraphConfigStatus.BUILDING.value,
            )
            .with_for_update()
        )
        if row is None:
            return None
        snapshot = await self._snapshot(row)
        if snapshot.processed_chunk_count < snapshot.eligible_chunk_count:
            return snapshot
        skipped = snapshot.protocol_skipped_count + snapshot.resource_skipped_count
        has_non_skipped = (
            snapshot.extracted_chunk_count + snapshot.empty_chunk_count > 0
        )
        if skipped > allowed_graph_skips(snapshot.eligible_chunk_count):
            row.status = GraphConfigStatus.FAILED.value
            row.last_error_code = "graph_skip_limit_exceeded"
        elif snapshot.eligible_chunk_count > 0 and not has_non_skipped:
            row.status = GraphConfigStatus.FAILED.value
            row.last_error_code = "graph_no_non_skipped_result"
        else:
            row.status = GraphConfigStatus.READY.value
            row.last_error_code = None
        row.updated_at = datetime.now(UTC)
        await self._session.flush()
        return await self._snapshot(row)

    async def _locked_config(self, kb_id: UUID) -> KnowledgeBaseGraphConfigRow | None:
        return await self._session.scalar(
            select(KnowledgeBaseGraphConfigRow)
            .where(
                KnowledgeBaseGraphConfigRow.workspace_id == self._workspace_id,
                KnowledgeBaseGraphConfigRow.kb_id == kb_id,
            )
            .with_for_update()
        )

    async def _require_valid_chat_profile(self, revision_id: UUID) -> None:
        row = (
            await self._session.execute(
                select(ModelProfileRevisionRow.id)
                .join(
                    ModelProfileRow,
                    and_(
                        ModelProfileRow.id == ModelProfileRevisionRow.profile_id,
                        ModelProfileRow.workspace_id == ModelProfileRevisionRow.workspace_id,
                    ),
                )
                .join(
                    ModelProviderRow,
                    and_(
                        ModelProviderRow.id == ModelProfileRow.provider_id,
                        ModelProviderRow.workspace_id == ModelProfileRow.workspace_id,
                    ),
                )
                .where(
                    ModelProfileRevisionRow.workspace_id == self._workspace_id,
                    ModelProfileRevisionRow.id == revision_id,
                    ModelProfileRevisionRow.validation_status == "valid",
                    ModelProfileRow.kind == "chat",
                    ModelProfileRow.enabled.is_(True),
                    ModelProviderRow.enabled.is_(True),
                )
            )
        ).scalar_one_or_none()
        if row is None:
            raise ResourceStateConflictError(
                "Graph Chat Profile Revision is not an enabled valid Chat profile"
            )

    async def _next_missing_chunk(
        self, config: GraphConfigSnapshot
    ) -> object | None:
        statement = _eligible_chunk_statement(self._workspace_id, config.knowledge_base_id)
        rows = (
            await self._session.execute(
                statement.order_by(
                    IndexChunkRow.indexed_document_version_id,
                    IndexChunkRow.ordinal,
                    IndexChunkRow.id,
                )
            )
        ).all()
        for row in rows:
            (
                chunk,
                target,
                document_version,
                document,
                _kb,
            ) = row
            existing = await self._session.execute(
                select(
                    IndexGraphChunkRow.id,
                    IndexGraphChunkRow.result_status,
                ).where(
                    IndexGraphChunkRow.workspace_id == self._workspace_id,
                    IndexGraphChunkRow.kb_id == config.knowledge_base_id,
                    IndexGraphChunkRow.build_id == config.build_id,
                    IndexGraphChunkRow.index_chunk_id == chunk.id,
                    IndexGraphChunkRow.content_hash == chunk.content_hash,
                    IndexGraphChunkRow.extractor_version == config.extractor_version,
                )
            )
            existing_row = existing.one_or_none()
            if existing_row is None:
                return _chunk_source(config, chunk, target, document_version, document)
        return None

    async def _snapshot(
        self, row: KnowledgeBaseGraphConfigRow
    ) -> GraphConfigSnapshot:
        eligible_statement = _eligible_chunk_statement(self._workspace_id, row.kb_id)
        eligible_count = int(
            await self._session.scalar(
                select(func.count()).select_from(eligible_statement.subquery())
            )
            or 0
        )
        eligible_ids = eligible_statement.with_only_columns(IndexChunkRow.id).subquery()
        grouped = (
            await self._session.execute(
                select(
                    IndexGraphChunkRow.result_status,
                    func.count(IndexGraphChunkRow.id),
                )
                .join(
                    IndexChunkRow,
                    and_(
                        IndexChunkRow.id == IndexGraphChunkRow.index_chunk_id,
                        IndexChunkRow.workspace_id
                        == IndexGraphChunkRow.workspace_id,
                        IndexChunkRow.kb_id == IndexGraphChunkRow.kb_id,
                    ),
                )
                .where(
                    IndexGraphChunkRow.workspace_id == self._workspace_id,
                    IndexGraphChunkRow.kb_id == row.kb_id,
                    IndexGraphChunkRow.build_id == row.build_id,
                    IndexGraphChunkRow.extractor_version == row.extractor_version,
                    IndexGraphChunkRow.index_chunk_id.in_(select(eligible_ids.c.id)),
                    IndexGraphChunkRow.content_hash == IndexChunkRow.content_hash,
                )
                .group_by(IndexGraphChunkRow.result_status)
            )
        ).all()
        counts = {str(status): int(count) for status, count in grouped}
        processed = sum(counts.values())
        return GraphConfigSnapshot(
            workspace_id=row.workspace_id,
            knowledge_base_id=row.kb_id,
            status=GraphConfigStatus(row.status),
            build_id=row.build_id,
            chat_profile_revision_id=row.chat_profile_revision_id,
            extractor_version=row.extractor_version,
            preflight_extractor_version=row.preflight_extractor_version,
            last_error_code=row.last_error_code,
            eligible_chunk_count=eligible_count,
            processed_chunk_count=processed,
            extracted_chunk_count=counts.get(GraphChunkResultStatus.EXTRACTED.value, 0),
            empty_chunk_count=counts.get(GraphChunkResultStatus.EMPTY.value, 0),
            protocol_skipped_count=counts.get(
                GraphChunkResultStatus.SKIPPED_PROTOCOL.value, 0
            ),
            resource_skipped_count=counts.get(
                GraphChunkResultStatus.SKIPPED_RESOURCE.value, 0
            ),
        )


def _eligible_chunk_statement(workspace_id: UUID, kb_id: UUID):
    return (
        select(
            IndexChunkRow,
            IndexedDocumentVersionRow,
            DocumentVersionRow,
            DocumentRow,
            KnowledgeBaseRow,
        )
        .join(
            IndexedDocumentVersionRow,
            and_(
                IndexedDocumentVersionRow.id
                == IndexChunkRow.indexed_document_version_id,
                IndexedDocumentVersionRow.workspace_id == IndexChunkRow.workspace_id,
                IndexedDocumentVersionRow.kb_id == IndexChunkRow.kb_id,
                IndexedDocumentVersionRow.build_status == IndexBuildStatus.READY,
                IndexedDocumentVersionRow.serving_status == IndexServingStatus.SERVING,
            ),
        )
        .join(
            IndexRevisionRow,
            and_(
                IndexRevisionRow.id == IndexedDocumentVersionRow.index_revision_id,
                IndexRevisionRow.workspace_id == IndexedDocumentVersionRow.workspace_id,
                IndexRevisionRow.status == IndexRevisionStatus.ACTIVE,
            ),
        )
        .join(
            KnowledgeBaseRow,
            and_(
                KnowledgeBaseRow.id == IndexChunkRow.kb_id,
                KnowledgeBaseRow.workspace_id == IndexChunkRow.workspace_id,
                KnowledgeBaseRow.active_index_revision_id
                == IndexedDocumentVersionRow.index_revision_id,
                KnowledgeBaseRow.deleted_at.is_(None),
            ),
        )
        .join(
            DocumentRow,
            and_(
                DocumentRow.id == IndexedDocumentVersionRow.document_id,
                DocumentRow.kb_id == IndexedDocumentVersionRow.kb_id,
                DocumentRow.workspace_id == IndexedDocumentVersionRow.workspace_id,
                DocumentRow.deleted_at.is_(None),
            ),
        )
        .join(
            DocumentVersionRow,
            and_(
                DocumentVersionRow.id == IndexedDocumentVersionRow.document_version_id,
                DocumentVersionRow.document_id == IndexedDocumentVersionRow.document_id,
                DocumentVersionRow.kb_id == IndexedDocumentVersionRow.kb_id,
                DocumentVersionRow.workspace_id == IndexedDocumentVersionRow.workspace_id,
                DocumentVersionRow.source_status == DocumentSourceStatus.AVAILABLE,
            ),
        )
        .where(
            IndexChunkRow.workspace_id == workspace_id,
            IndexChunkRow.kb_id == kb_id,
            IndexChunkRow.modality.in_(("text", "table")),
        )
    )


def _chunk_source(config, chunk, target, document_version, document):
    from rag_kb.domain import GraphChunkSource

    return GraphChunkSource(
        workspace_id=config.workspace_id,
        knowledge_base_id=config.knowledge_base_id,
        build_id=config.build_id,
        index_chunk_id=chunk.id,
        index_revision_id=target.index_revision_id,
        indexed_document_version_id=target.id,
        document_id=target.document_id,
        document_version_id=document_version.id,
        ordinal=chunk.ordinal,
        modality=chunk.modality,
        content=chunk.content,
        content_hash=chunk.content_hash,
        source_location=chunk.source_location,
        hierarchy=chunk.hierarchy,
        source_metadata=chunk.source_metadata,
        excluded=chunk.excluded_at is not None,
    )
