"""SQLAlchemy persistence for Graphiti build orchestration."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
import hashlib
from uuid import UUID, uuid4

from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_kb.document_processing.profiles import SEMANTIC_CHUNKING_CONFIG
from rag_kb.db.models import (
    Document as DocumentRow,
    DocumentSourceStatus,
    DocumentVersion as DocumentVersionRow,
    EmbeddingSpace as EmbeddingSpaceRow,
    GraphitiEpisodeChunk as GraphitiEpisodeChunkRow,
    GraphitiGraphBuild as GraphitiGraphBuildRow,
    IndexBuildStatus,
    IndexChunk as IndexChunkRow,
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
    GraphConfigSnapshot,
    GraphConfigStatus,
    GraphitiBuildSnapshot,
    GraphitiBuildStatus,
    GraphWorkItem,
    GraphWorkKind,
    ResourceNotFoundError,
    ResourceStateConflictError,
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
        self._retired_builds: list[GraphitiBuildSnapshot] = []

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
        if extractor_version != GRAPH_EXTRACTOR_VERSION:
            raise ValueError("only the Graphiti extractor is supported")
        row = await self._locked_config(kb_id)
        if row is None:
            await self.ensure_config(kb_id)
            row = await self._locked_config(kb_id)
        assert row is not None
        if not enabled:
            await self._retire_builds(row.kb_id)
            row.status = GraphConfigStatus.DISABLED.value
            row.chat_profile_revision_id = None
            row.active_build_id = None
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
        await self._ensure_graphiti_build(row)
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
        if extractor_version != GRAPH_EXTRACTOR_VERSION:
            raise ValueError("only the Graphiti extractor is supported")
        version_changed = row.extractor_version != extractor_version
        snapshot = await self._snapshot(row)
        if (
            not force_rebuild
            and not version_changed
            and snapshot.status is GraphConfigStatus.READY
        ):
            raise ResourceStateConflictError(
                "Graph retry of a ready build requires force_rebuild"
            )
        row.build_id = uuid4()
        row.preflight_extractor_version = None
        row.extractor_version = extractor_version
        row.status = GraphConfigStatus.BUILDING.value
        row.last_error_code = None
        await self._ensure_graphiti_build(row)
        await self._session.flush()
        return await self._snapshot(row)

    async def invalidate_for_serving_change(self, kb_id: UUID) -> bool:
        self._ensure_active()
        row = await self._locked_config(kb_id)
        if row is None or row.status == GraphConfigStatus.DISABLED.value:
            return False
        row.build_id = uuid4()
        row.status = GraphConfigStatus.BUILDING.value
        row.preflight_extractor_version = None
        row.last_error_code = None
        row.updated_at = datetime.now(UTC)
        await self._ensure_graphiti_build(row)
        await self._session.flush()
        return True

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
            kb = await self._session.scalar(
                select(KnowledgeBaseRow).where(
                    KnowledgeBaseRow.workspace_id == self._workspace_id,
                    KnowledgeBaseRow.id == row.kb_id,
                )
            )
            if kb is None or kb.deleted_at is not None:
                await self._retire_builds(row.kb_id)
                row.status = GraphConfigStatus.DISABLED.value
                row.chat_profile_revision_id = None
                row.active_build_id = None
                row.last_error_code = None
                row.updated_at = datetime.now(UTC)
                await self._session.flush()
                continue
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

    async def get_graphiti_build(
        self, kb_id: UUID, *, build_id: UUID
    ) -> GraphitiBuildSnapshot | None:
        self._ensure_active()
        row = await self._session.scalar(
            select(GraphitiGraphBuildRow).where(
                GraphitiGraphBuildRow.workspace_id == self._workspace_id,
                GraphitiGraphBuildRow.kb_id == kb_id,
                GraphitiGraphBuildRow.build_id == build_id,
            )
        )
        return _graphiti_build_snapshot(row) if row is not None else None

    async def save_graphiti_episode(
        self,
        *,
        kb_id: UUID,
        build_id: UUID,
        index_chunk_id: UUID,
        content_hash: str,
        episode_uuid: str,
    ) -> bool:
        self._ensure_active()
        build = await self._session.scalar(
            select(GraphitiGraphBuildRow)
            .where(
                GraphitiGraphBuildRow.workspace_id == self._workspace_id,
                GraphitiGraphBuildRow.kb_id == kb_id,
                GraphitiGraphBuildRow.build_id == build_id,
                GraphitiGraphBuildRow.status == GraphitiBuildStatus.BUILDING.value,
            )
            .with_for_update()
        )
        if build is None:
            return False
        chunk = await self._session.scalar(
            select(IndexChunkRow).where(
                IndexChunkRow.workspace_id == self._workspace_id,
                IndexChunkRow.kb_id == kb_id,
                IndexChunkRow.id == index_chunk_id,
                IndexChunkRow.content_hash == content_hash,
                IndexChunkRow.excluded_at.is_(None),
            )
        )
        if chunk is None:
            return False
        existing = await self._session.scalar(
            select(GraphitiEpisodeChunkRow).where(
                GraphitiEpisodeChunkRow.workspace_id == self._workspace_id,
                GraphitiEpisodeChunkRow.kb_id == kb_id,
                GraphitiEpisodeChunkRow.build_id == build_id,
                GraphitiEpisodeChunkRow.index_chunk_id == index_chunk_id,
            )
        )
        if existing is not None:
            if existing.episode_uuid != episode_uuid:
                raise ResourceStateConflictError("Graphiti episode mapping conflict")
            return True
        self._session.add(
            GraphitiEpisodeChunkRow(
                workspace_id=self._workspace_id,
                kb_id=kb_id,
                build_id=build_id,
                group_id=build.group_id,
                episode_uuid=episode_uuid,
                index_revision_id=build.index_revision_id,
                index_chunk_id=index_chunk_id,
                content_hash=content_hash,
                status="completed",
            )
        )
        await self._session.flush()
        return True

    async def finalize_graphiti_if_complete(
        self, kb_id: UUID, *, build_id: UUID
    ) -> GraphConfigSnapshot | None:
        self._ensure_active()
        config = await self._locked_config(kb_id)
        build = await self._session.scalar(
            select(GraphitiGraphBuildRow)
            .where(
                GraphitiGraphBuildRow.workspace_id == self._workspace_id,
                GraphitiGraphBuildRow.kb_id == kb_id,
                GraphitiGraphBuildRow.build_id == build_id,
            )
            .with_for_update()
        )
        if (
            config is None
            or build is None
            or config.build_id != build_id
            or config.status != GraphConfigStatus.BUILDING.value
            or build.status != GraphitiBuildStatus.BUILDING.value
            or build.superseded_by is not None
        ):
            return None
        count = int(
            await self._session.scalar(
                select(func.count(GraphitiEpisodeChunkRow.id)).where(
                    GraphitiEpisodeChunkRow.workspace_id == self._workspace_id,
                    GraphitiEpisodeChunkRow.kb_id == kb_id,
                    GraphitiEpisodeChunkRow.build_id == build_id,
                )
            )
            or 0
        )
        revision_id, digest, expected, profile_id, model, dimension = (
            await self._graphiti_frozen_input(kb_id)
        )
        if count < build.expected_episode_count:
            return await self._snapshot(config)
        if (
            revision_id != build.index_revision_id
            or digest != build.serving_chunk_digest
            or expected != build.expected_episode_count
            or count != expected
            or config.chat_profile_revision_id != build.chat_profile_revision_id
            or config.extractor_version != build.extractor_version
            or profile_id != build.embedding_profile_revision_id
            or model != build.embedding_model
            or dimension != build.embedding_dimension
        ):
            self._remember_retired(build)
            build.status = GraphitiBuildStatus.SUPERSEDED.value
            config.build_id = uuid4()
            build.superseded_by = config.build_id
            await self._ensure_graphiti_build(config)
            return await self._snapshot(config)
        build.status = GraphitiBuildStatus.READY.value
        build.completed_at = datetime.now(UTC)
        if config.active_build_id is not None and config.active_build_id != build_id:
            await self._supersede_build(config.active_build_id, successor_id=build_id)
        config.active_build_id = build_id
        config.status = GraphConfigStatus.READY.value
        config.last_error_code = None
        config.updated_at = datetime.now(UTC)
        await self._session.flush()
        return await self._snapshot(config)

    async def first_graphiti_episode_uuid(
        self, kb_id: UUID, *, build_id: UUID
    ) -> str | None:
        self._ensure_active()
        return await self._session.scalar(
            select(GraphitiEpisodeChunkRow.episode_uuid)
            .where(
                GraphitiEpisodeChunkRow.workspace_id == self._workspace_id,
                GraphitiEpisodeChunkRow.kb_id == kb_id,
                GraphitiEpisodeChunkRow.build_id == build_id,
            )
            .order_by(GraphitiEpisodeChunkRow.created_at, GraphitiEpisodeChunkRow.id)
            .limit(1)
        )

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
        await self._session.execute(
            update(GraphitiGraphBuildRow)
            .where(
                GraphitiGraphBuildRow.workspace_id == self._workspace_id,
                GraphitiGraphBuildRow.kb_id == kb_id,
                GraphitiGraphBuildRow.build_id == build_id,
                GraphitiGraphBuildRow.status == GraphitiBuildStatus.BUILDING.value,
            )
            .values(
                status=GraphitiBuildStatus.FAILED.value,
                last_error_code=error_code,
                completed_at=datetime.now(UTC),
            )
        )
        failed = await self._session.get(GraphitiGraphBuildRow, build_id)
        if failed is not None:
            self._remember_retired(failed)
        return result.rowcount > 0

    def take_retired_graphiti_builds(self) -> tuple[GraphitiBuildSnapshot, ...]:
        retired = tuple(self._retired_builds)
        self._retired_builds.clear()
        return retired

    def _remember_retired(self, row: GraphitiGraphBuildRow) -> None:
        self._retired_builds.append(_graphiti_build_snapshot(row))

    async def _retire_builds(self, kb_id: UUID) -> None:
        rows = (
            await self._session.scalars(
                select(GraphitiGraphBuildRow).where(
                    GraphitiGraphBuildRow.workspace_id == self._workspace_id,
                    GraphitiGraphBuildRow.kb_id == kb_id,
                )
            )
        ).all()
        now = datetime.now(UTC)
        for build in rows:
            self._remember_retired(build)
            if build.status in {
                GraphitiBuildStatus.BUILDING.value,
                GraphitiBuildStatus.READY.value,
            }:
                build.status = GraphitiBuildStatus.SUPERSEDED.value
                build.completed_at = now
        await self._session.flush()

    async def _supersede_build(
        self, build_id: UUID, *, successor_id: UUID
    ) -> None:
        row = await self._session.get(GraphitiGraphBuildRow, build_id)
        if row is None or row.status == GraphitiBuildStatus.SUPERSEDED.value:
            return
        self._remember_retired(row)
        row.status = GraphitiBuildStatus.SUPERSEDED.value
        row.superseded_by = successor_id
        row.completed_at = datetime.now(UTC)

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
            mapped = await self._session.scalar(
                select(GraphitiEpisodeChunkRow.id).where(
                    GraphitiEpisodeChunkRow.workspace_id == self._workspace_id,
                    GraphitiEpisodeChunkRow.kb_id == config.knowledge_base_id,
                    GraphitiEpisodeChunkRow.build_id == config.build_id,
                    GraphitiEpisodeChunkRow.index_chunk_id == chunk.id,
                    GraphitiEpisodeChunkRow.content_hash == chunk.content_hash,
                )
            )
            if mapped is None:
                return _chunk_source(config, chunk, target, document_version, document)
        return None

    async def _ensure_graphiti_build(
        self, row: KnowledgeBaseGraphConfigRow
    ) -> GraphitiGraphBuildRow:
        existing = await self._session.get(GraphitiGraphBuildRow, row.build_id)
        if existing is not None:
            return existing
        previous = await self._session.scalar(
            select(GraphitiGraphBuildRow)
            .where(
                GraphitiGraphBuildRow.workspace_id == self._workspace_id,
                GraphitiGraphBuildRow.kb_id == row.kb_id,
                GraphitiGraphBuildRow.status == GraphitiBuildStatus.BUILDING.value,
            )
            .order_by(GraphitiGraphBuildRow.started_at.desc())
            .limit(1)
            .with_for_update()
        )
        if previous is not None and previous.build_id != row.build_id:
            self._remember_retired(previous)
            previous.status = GraphitiBuildStatus.SUPERSEDED.value
            previous.superseded_by = row.build_id
            previous.completed_at = datetime.now(UTC)
        revision_id, digest, expected, profile_id, model, dimension = (
            await self._graphiti_frozen_input(row.kb_id)
        )
        if row.chat_profile_revision_id is None:
            raise ResourceStateConflictError("Graphiti Chat Profile Revision is missing")
        build = GraphitiGraphBuildRow(
            build_id=row.build_id,
            workspace_id=self._workspace_id,
            kb_id=row.kb_id,
            group_id=f"ws_{self._workspace_id}_kb_{row.kb_id}_b_{row.build_id}",
            status=GraphitiBuildStatus.BUILDING.value,
            index_revision_id=revision_id,
            serving_chunk_digest=digest,
            expected_episode_count=expected,
            chat_profile_revision_id=row.chat_profile_revision_id,
            embedding_profile_revision_id=profile_id,
            embedding_model=model,
            embedding_dimension=dimension,
            extractor_version=row.extractor_version,
        )
        self._session.add(build)
        await self._session.flush()
        return build

    async def _graphiti_frozen_input(
        self, kb_id: UUID
    ) -> tuple[UUID, str, int, UUID, str, int]:
        revision = await self._session.scalar(
            select(IndexRevisionRow).where(
                IndexRevisionRow.workspace_id == self._workspace_id,
                IndexRevisionRow.kb_id == kb_id,
                IndexRevisionRow.status == IndexRevisionStatus.ACTIVE,
            )
        )
        if revision is None:
            raise ResourceStateConflictError("Graphiti build requires an active index revision")
        if not _graph_chunking_profile_compatible(revision.chunking_config):
            raise ResourceStateConflictError(
                "Graphiti build requires reindexing with the current semantic chunking profile"
            )
        space = await self._session.scalar(
            select(EmbeddingSpaceRow).where(
                EmbeddingSpaceRow.workspace_id == self._workspace_id,
                EmbeddingSpaceRow.id == revision.embedding_space_id,
            )
        )
        if space is None or space.model_profile_revision_id is None:
            raise ResourceStateConflictError("Graphiti build requires a revision-bound embedding space")
        rows = (
            await self._session.execute(
                _eligible_chunk_statement(self._workspace_id, kb_id).order_by(IndexChunkRow.id)
            )
        ).all()
        digest = hashlib.sha256()
        for chunk, *_ in rows:
            digest.update(chunk.id.bytes)
            digest.update(chunk.content_hash.encode("ascii"))
        return (
            revision.id,
            digest.hexdigest(),
            len(rows),
            space.model_profile_revision_id,
            space.resolved_model,
            space.dimension,
        )

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
        eligible_keys = eligible_statement.with_only_columns(
            IndexChunkRow.id, IndexChunkRow.content_hash
        ).subquery()
        processed = int(
            await self._session.scalar(
                select(func.count())
                .select_from(GraphitiEpisodeChunkRow)
                .join(
                    eligible_keys,
                    and_(
                        GraphitiEpisodeChunkRow.index_chunk_id == eligible_keys.c.id,
                        GraphitiEpisodeChunkRow.content_hash
                        == eligible_keys.c.content_hash,
                    ),
                )
                .where(
                    GraphitiEpisodeChunkRow.workspace_id == self._workspace_id,
                    GraphitiEpisodeChunkRow.kb_id == row.kb_id,
                    GraphitiEpisodeChunkRow.build_id == row.build_id,
                )
            )
            or 0
        )
        build = await self._session.get(GraphitiGraphBuildRow, row.build_id)
        return GraphConfigSnapshot(
            workspace_id=row.workspace_id,
            knowledge_base_id=row.kb_id,
            status=GraphConfigStatus(row.status),
            build_id=row.build_id,
            active_build_id=row.active_build_id,
            chat_profile_revision_id=row.chat_profile_revision_id,
            extractor_version=row.extractor_version,
            preflight_extractor_version=row.preflight_extractor_version,
            last_error_code=row.last_error_code,
            eligible_chunk_count=eligible_count,
            processed_chunk_count=processed,
            extracted_chunk_count=processed,
            group_id=build.group_id if build is not None else None,
            index_revision_id=build.index_revision_id if build is not None else None,
            embedding_profile_revision_id=(
                build.embedding_profile_revision_id if build is not None else None
            ),
            embedding_model=build.embedding_model if build is not None else None,
            embedding_dimension=build.embedding_dimension if build is not None else None,
        )


def _graph_chunking_profile_compatible(chunking_config: object) -> bool:
    if not isinstance(chunking_config, dict):
        return False
    if chunking_config.get("strategy") != "semantic_breakpoint":
        return True
    return chunking_config == SEMANTIC_CHUNKING_CONFIG


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
            IndexChunkRow.excluded_at.is_(None),
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
        reference_time=document_version.created_at,
        excluded=chunk.excluded_at is not None,
    )


def _graphiti_build_snapshot(row: GraphitiGraphBuildRow) -> GraphitiBuildSnapshot:
    return GraphitiBuildSnapshot(
        workspace_id=row.workspace_id,
        knowledge_base_id=row.kb_id,
        build_id=row.build_id,
        group_id=row.group_id,
        status=GraphitiBuildStatus(row.status),
        index_revision_id=row.index_revision_id,
        serving_chunk_digest=row.serving_chunk_digest,
        expected_episode_count=row.expected_episode_count,
        chat_profile_revision_id=row.chat_profile_revision_id,
        embedding_profile_revision_id=row.embedding_profile_revision_id,
        embedding_model=row.embedding_model,
        embedding_dimension=row.embedding_dimension,
        extractor_version=row.extractor_version,
        superseded_by=row.superseded_by,
    )
