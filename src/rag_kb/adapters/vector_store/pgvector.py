"""One-statement exact pgvector retrieval for a frozen embedding space."""

from __future__ import annotations

from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Integer,
    and_,
    bindparam,
    column,
    func,
    or_,
    select,
    true,
    values,
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import aliased

from rag_kb.db.models import (
    Document,
    DocumentSourceStatus,
    DocumentVersion,
    EmbeddingSpace,
    IndexAsset,
    IndexBuildStatus,
    IndexChunk,
    IndexedDocumentVersion,
    IndexRevision,
    IndexRevisionEmbeddingSpace,
    IndexRevisionStatus,
    IndexServingStatus,
    KnowledgeBase,
    ModelProfileRevision,
    VectorRecord,
)
from rag_kb.domain import (
    AdjacentChunkHit,
    AdjacentChunkQuery,
    AdjacentChunkResult,
    ErrorCode,
    RetrievalExecutionError,
    RetrievalQueryPlan,
    RetrievalStrategy,
    VectorSearchHit,
    VectorSearchResult,
    EmbeddingSpaceDefinition,
)


class PgVectorStore:
    """Resolve the active selector and exact hits in one snapshot statement."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        configured_space: EmbeddingSpaceDefinition,
    ) -> None:
        if (
            configured_space.distance_metric != "cosine"
            or configured_space.vector_data_type != "float32"
        ):
            raise ValueError("configured embedding space is not supported")
        self._sessions = sessions
        self._configured_space = configured_space

    async def has_space_role(
        self, plan: RetrievalQueryPlan, space_role: str
    ) -> bool:
        statement = (
            select(IndexRevisionEmbeddingSpace.index_revision_id)
            .join(
                KnowledgeBase,
                and_(
                    KnowledgeBase.active_index_revision_id
                    == IndexRevisionEmbeddingSpace.index_revision_id,
                    KnowledgeBase.workspace_id
                    == IndexRevisionEmbeddingSpace.workspace_id,
                ),
            )
            .where(
                KnowledgeBase.workspace_id == plan.workspace_id,
                KnowledgeBase.id == plan.knowledge_base_id,
                KnowledgeBase.deleted_at.is_(None),
                IndexRevisionEmbeddingSpace.role == space_role,
            )
            .limit(1)
        )
        async with self._sessions() as session:
            return (await session.scalar(statement)) is not None

    async def adjacent_chunks(
        self,
        query: AdjacentChunkQuery,
    ) -> AdjacentChunkResult | None:
        statement = self._adjacent_statement(len(query.anchors))
        parameters: dict[str, Any] = {
            "workspace_id": query.workspace_id,
            "knowledge_base_id": query.knowledge_base_id,
            "index_revision_id": query.index_revision_id,
        }
        for index, anchor in enumerate(query.anchors):
            parameters.update(
                {
                    f"adj_input_rank_{index}": index + 1,
                    f"adj_input_chunk_id_{index}": anchor.index_chunk_id,
                    f"adj_input_target_id_{index}": (
                        anchor.indexed_document_version_id
                    ),
                    f"adj_input_ordinal_{index}": anchor.ordinal,
                }
            )
        async with self._sessions() as session:
            rows = (
                await session.execute(statement, parameters)
            ).mappings().all()
        if not rows:
            return None
        first = rows[0]
        return AdjacentChunkResult(
            resolved_active_revision_id=first["active_revision_id"],
            validated_anchor_count=first["validated_anchor_count"],
            hits=tuple(
                self._adjacent_hit(row)
                for row in rows
                if row["index_chunk_id"] is not None
            ),
        )

    async def resolve_space(
        self,
        plan: RetrievalQueryPlan,
        space_role: str,
    ) -> EmbeddingSpaceDefinition | None:
        return (await self.resolve_spaces(plan)).get(space_role)

    async def resolve_spaces(
        self,
        plan: RetrievalQueryPlan,
    ) -> dict[str, EmbeddingSpaceDefinition]:
        statement = (
            select(
                IndexRevisionEmbeddingSpace.role,
                EmbeddingSpace,
                ModelProfileRevision.validation_snapshot,
            )
            .join(
                IndexRevisionEmbeddingSpace,
                and_(
                    IndexRevisionEmbeddingSpace.embedding_space_id
                    == EmbeddingSpace.id,
                    IndexRevisionEmbeddingSpace.workspace_id
                    == EmbeddingSpace.workspace_id,
                ),
            )
            .outerjoin(
                ModelProfileRevision,
                ModelProfileRevision.id == EmbeddingSpace.model_profile_revision_id,
            )
            .join(
                KnowledgeBase,
                and_(
                    KnowledgeBase.active_index_revision_id
                    == IndexRevisionEmbeddingSpace.index_revision_id,
                    KnowledgeBase.workspace_id
                    == IndexRevisionEmbeddingSpace.workspace_id,
                ),
            )
            .where(
                KnowledgeBase.workspace_id == plan.workspace_id,
                KnowledgeBase.id == plan.knowledge_base_id,
                KnowledgeBase.deleted_at.is_(None),
            )
            .order_by(IndexRevisionEmbeddingSpace.role)
        )
        async with self._sessions() as session:
            rows = (await session.execute(statement)).all()
        return {
            role: self._definition(space, validation_snapshot)
            for role, space, validation_snapshot in rows
        }

    @staticmethod
    def _definition(
        space: EmbeddingSpace,
        validation_snapshot: dict[str, Any] | None,
    ) -> EmbeddingSpaceDefinition:
        return EmbeddingSpaceDefinition(
            provider_identity=space.provider_identity,
            endpoint_identity=space.endpoint_identity,
            requested_model=space.requested_model,
            resolved_model=space.resolved_model,
            model_version=space.model_version,
            deployment_revision=space.deployment_revision,
            dimension=space.dimension,
            distance_metric=space.distance_metric,
            vector_data_type=space.vector_data_type,
            normalization=space.normalization,
            configuration_fingerprint=space.configuration_fingerprint,
            tokenizer_fingerprint=space.tokenizer_fingerprint,
            compatibility_fingerprint=space.compatibility_fingerprint,
            model_profile_revision_id=space.model_profile_revision_id,
            dimension_request_mode=(
                str(validation_snapshot.get("dimension_request_mode", "explicit"))
                if validation_snapshot is not None
                else "explicit"
            ),
        )

    async def search(
        self,
        plan: RetrievalQueryPlan,
        query_embedding: tuple[float, ...],
    ) -> VectorSearchResult | None:
        return await self.search_space(
            plan,
            query_embedding,
            space_role="text_retrieval",
            representation_kinds=("text", "caption_text", "ocr_text", "table_text"),
            expected_space=self._configured_space,
        )

    async def search_space(
        self,
        plan: RetrievalQueryPlan,
        query_embedding: tuple[float, ...],
        *,
        space_role: str,
        representation_kinds: tuple[str, ...],
        expected_space: EmbeddingSpaceDefinition,
    ) -> VectorSearchResult | None:
        self._require_exact_plan(plan, query_embedding, expected_space)
        if space_role not in {
            "text_retrieval",
            "cross_modal_retrieval",
            "semantic_analysis",
        } or not representation_kinds:
            raise ValueError("space role and representation allowlist are required")
        statement = self._statement(
            expected_space.dimension,
            document_ids=plan.document_ids,
        )
        parameters = {
            "workspace_id": plan.workspace_id,
            "knowledge_base_id": plan.knowledge_base_id,
            "query_embedding": list(query_embedding),
            "top_k": plan.candidate_count or plan.top_k,
            "space_role": space_role,
            "representation_kinds": list(representation_kinds),
            "expected_dimension": expected_space.dimension,
        }
        async with self._sessions() as session:
            result = await session.execute(statement, parameters)
            rows = result.mappings().all()
        if not rows:
            return None

        first = rows[0]
        expected_fingerprint = expected_space.compatibility_fingerprint
        if first["compatibility_fingerprint"] != expected_fingerprint:
            raise RetrievalExecutionError(
                ErrorCode.EMBEDDING_SPACE_MISMATCH,
                diagnostic={"check": "active_revision_embedding_space"},
            )
        hits = tuple(
            self._hit(row)
            for row in rows
            if row["index_chunk_id"] is not None
        )
        return VectorSearchResult(
            resolved_active_revision_id=first["active_revision_id"],
            hits=hits,
            embedding_space_id=first["embedding_space_id"],
            compatibility_fingerprint=first["compatibility_fingerprint"],
            space_role=space_role,
        )

    def _require_exact_plan(
        self,
        plan: RetrievalQueryPlan,
        query_embedding: tuple[float, ...],
        expected_space: EmbeddingSpaceDefinition | None = None,
    ) -> None:
        configured = expected_space or self._configured_space
        if (
            plan.strategy
            not in {RetrievalStrategy.EXACT_VECTOR, RetrievalStrategy.HYBRID}
            or plan.distance_metric != "cosine"
        ):
            raise RetrievalExecutionError(
                ErrorCode.CAPABILITY_NOT_ENABLED,
                diagnostic={"capability": "non_exact_vector_plan"},
            )
        if len(query_embedding) != configured.dimension:
            raise RetrievalExecutionError(
                ErrorCode.EMBEDDING_RESPONSE_INVALID,
                diagnostic={"check": "query_dimension"},
            )

    @staticmethod
    def _statement(
        dimension: int = 1024,
        *,
        document_ids: tuple | list = (),
    ):
        vector_record = VectorRecord
        query_vector = bindparam(
            "query_embedding",
            type_=Vector(dimension),
        )
        distance = vector_record.embedding.cosine_distance(query_vector).label(
            "cosine_distance"
        )
        scope_filter = (
            IndexedDocumentVersion.document_id.in_(document_ids)
            if document_ids
            else true()
        )
        hits = (
            select(
                IndexChunk.workspace_id.label("hit_workspace_id"),
                IndexChunk.kb_id.label("hit_knowledge_base_id"),
                IndexedDocumentVersion.index_revision_id.label(
                    "hit_index_revision_id"
                ),
                IndexChunk.id.label("index_chunk_id"),
                IndexedDocumentVersion.id.label("indexed_document_version_id"),
                Document.id.label("document_id"),
                DocumentVersion.id.label("document_version_id"),
                Document.display_name.label("document_display_name"),
                DocumentVersion.original_filename.label(
                    "document_original_filename"
                ),
                IndexChunk.ordinal.label("ordinal"),
                IndexChunk.content.label("content"),
                IndexChunk.source_location.label("source_location"),
                IndexChunk.hierarchy.label("hierarchy"),
                IndexChunk.source_metadata.label("source_metadata"),
                IndexChunk.modality.label("modality"),
                IndexChunk.evidence_group_key.label("evidence_group_key"),
                vector_record.representation_kind.label("representation_kind"),
                IndexAsset.id.label("index_asset_id"),
                IndexAsset.media_type.label("asset_media_type"),
                IndexAsset.checksum_sha256.label("asset_checksum_sha256"),
                IndexAsset.width.label("asset_width"),
                IndexAsset.height.label("asset_height"),
                distance,
                IndexedDocumentVersion.build_status.label("build_status"),
                IndexedDocumentVersion.serving_status.label("serving_status"),
            )
            .select_from(IndexedDocumentVersion)
            .join(
                Document,
                and_(
                    Document.id == IndexedDocumentVersion.document_id,
                    Document.kb_id == IndexedDocumentVersion.kb_id,
                    Document.workspace_id == IndexedDocumentVersion.workspace_id,
                ),
            )
            .join(
                DocumentVersion,
                and_(
                    DocumentVersion.id
                    == IndexedDocumentVersion.document_version_id,
                    DocumentVersion.document_id
                    == IndexedDocumentVersion.document_id,
                    DocumentVersion.kb_id == IndexedDocumentVersion.kb_id,
                    DocumentVersion.workspace_id
                    == IndexedDocumentVersion.workspace_id,
                ),
            )
            .join(
                IndexChunk,
                and_(
                    IndexChunk.indexed_document_version_id
                    == IndexedDocumentVersion.id,
                    IndexChunk.kb_id == IndexedDocumentVersion.kb_id,
                    IndexChunk.workspace_id == IndexedDocumentVersion.workspace_id,
                ),
            )
            .join(
                vector_record,
                and_(
                    vector_record.index_chunk_id == IndexChunk.id,
                    vector_record.kb_id == IndexChunk.kb_id,
                    vector_record.workspace_id == IndexChunk.workspace_id,
                ),
            )
            .outerjoin(
                IndexAsset,
                and_(
                    IndexAsset.id == IndexChunk.index_asset_id,
                    IndexAsset.indexed_document_version_id
                    == IndexChunk.indexed_document_version_id,
                ),
            )
            .where(
                IndexedDocumentVersion.workspace_id == KnowledgeBase.workspace_id,
                IndexedDocumentVersion.kb_id == KnowledgeBase.id,
                IndexedDocumentVersion.index_revision_id == IndexRevision.id,
                IndexedDocumentVersion.build_status == IndexBuildStatus.READY,
                IndexedDocumentVersion.serving_status == IndexServingStatus.SERVING,
                Document.deleted_at.is_(None),
                IndexChunk.excluded_at.is_(None),
                DocumentVersion.source_status == DocumentSourceStatus.AVAILABLE,
                scope_filter,
                vector_record.embedding_space_id
                == IndexRevisionEmbeddingSpace.embedding_space_id,
                vector_record.embedding_dimension
                == bindparam("expected_dimension", type_=Integer),
                vector_record.representation_kind.in_(
                    bindparam("representation_kinds", expanding=True)
                ),
            )
            .order_by(distance.asc(), IndexChunk.id.asc())
            .limit(bindparam("top_k", type_=Integer))
            .lateral("retrieval_hits")
        )
        return (
            select(
                KnowledgeBase.active_index_revision_id.label("active_revision_id"),
                EmbeddingSpace.compatibility_fingerprint.label(
                    "compatibility_fingerprint"
                ),
                EmbeddingSpace.id.label("embedding_space_id"),
                hits.c.hit_workspace_id,
                hits.c.hit_knowledge_base_id,
                hits.c.hit_index_revision_id,
                hits.c.index_chunk_id,
                hits.c.indexed_document_version_id,
                hits.c.document_id,
                hits.c.document_version_id,
                hits.c.document_display_name,
                hits.c.document_original_filename,
                hits.c.ordinal,
                hits.c.content,
                hits.c.source_location,
                hits.c.hierarchy,
                hits.c.source_metadata,
                hits.c.modality,
                hits.c.evidence_group_key,
                hits.c.representation_kind,
                hits.c.index_asset_id,
                hits.c.asset_media_type,
                hits.c.asset_checksum_sha256,
                hits.c.asset_width,
                hits.c.asset_height,
                hits.c.cosine_distance,
                hits.c.build_status,
                hits.c.serving_status,
            )
            .select_from(KnowledgeBase)
            .join(
                IndexRevision,
                and_(
                    IndexRevision.id == KnowledgeBase.active_index_revision_id,
                    IndexRevision.kb_id == KnowledgeBase.id,
                    IndexRevision.workspace_id == KnowledgeBase.workspace_id,
                    IndexRevision.status == IndexRevisionStatus.ACTIVE,
                ),
            )
            .join(
                IndexRevisionEmbeddingSpace,
                and_(
                    IndexRevisionEmbeddingSpace.index_revision_id == IndexRevision.id,
                    IndexRevisionEmbeddingSpace.workspace_id == IndexRevision.workspace_id,
                    IndexRevisionEmbeddingSpace.role == bindparam("space_role"),
                ),
            )
            .join(
                EmbeddingSpace,
                and_(
                    EmbeddingSpace.id
                    == IndexRevisionEmbeddingSpace.embedding_space_id,
                    EmbeddingSpace.workspace_id
                    == IndexRevisionEmbeddingSpace.workspace_id,
                ),
            )
            .outerjoin(hits, true())
            .where(
                KnowledgeBase.workspace_id == bindparam("workspace_id"),
                KnowledgeBase.id == bindparam("knowledge_base_id"),
                KnowledgeBase.deleted_at.is_(None),
            )
            .order_by(
                hits.c.cosine_distance.asc().nulls_last(),
                hits.c.index_chunk_id.asc().nulls_last(),
            )
        )

    @staticmethod
    def _adjacent_statement(anchor_count: int):
        if not 1 <= anchor_count <= 2:
            raise ValueError("adjacency query requires one or two anchors")

        anchor_rows = values(
            column("anchor_rank", Integer),
            column("anchor_chunk_id", PostgreSQLUUID(as_uuid=True)),
            column("anchor_target_id", PostgreSQLUUID(as_uuid=True)),
            column("anchor_ordinal", Integer),
            name="adjacency_anchor_input",
        ).data(
            tuple(
                (
                    bindparam(f"adj_input_rank_{index}", type_=Integer),
                    bindparam(
                        f"adj_input_chunk_id_{index}",
                        type_=PostgreSQLUUID(as_uuid=True),
                    ),
                    bindparam(
                        f"adj_input_target_id_{index}",
                        type_=PostgreSQLUUID(as_uuid=True),
                    ),
                    bindparam(f"adj_input_ordinal_{index}", type_=Integer),
                )
                for index in range(anchor_count)
            )
        ).cte("adjacency_anchor_input")

        anchor_chunk = aliased(IndexChunk, name="adjacency_anchor_chunk")
        target = aliased(
            IndexedDocumentVersion,
            name="adjacency_indexed_document_version",
        )
        document = aliased(Document, name="adjacency_document")
        document_version = aliased(
            DocumentVersion,
            name="adjacency_document_version",
        )
        revision = aliased(IndexRevision, name="adjacency_index_revision")
        knowledge_base = aliased(KnowledgeBase, name="adjacency_knowledge_base")
        valid_anchors = (
            select(
                anchor_rows.c.anchor_rank,
                anchor_rows.c.anchor_chunk_id,
                anchor_rows.c.anchor_ordinal,
                target.workspace_id.label("hit_workspace_id"),
                target.kb_id.label("hit_knowledge_base_id"),
                target.index_revision_id.label("hit_index_revision_id"),
                target.id.label("indexed_document_version_id"),
                target.document_id,
                target.document_version_id,
                target.build_status,
                target.serving_status,
                document.display_name.label("document_display_name"),
                document_version.original_filename.label(
                    "document_original_filename"
                ),
            )
            .select_from(anchor_rows)
            .join(
                anchor_chunk,
                and_(
                    anchor_chunk.id == anchor_rows.c.anchor_chunk_id,
                    anchor_chunk.indexed_document_version_id
                    == anchor_rows.c.anchor_target_id,
                    anchor_chunk.ordinal == anchor_rows.c.anchor_ordinal,
                    anchor_chunk.workspace_id
                    == bindparam(
                        "workspace_id", type_=PostgreSQLUUID(as_uuid=True)
                    ),
                    anchor_chunk.kb_id
                    == bindparam(
                        "knowledge_base_id", type_=PostgreSQLUUID(as_uuid=True)
                    ),
                    anchor_chunk.excluded_at.is_(None),
                    anchor_chunk.modality.in_(("text", "table")),
                ),
            )
            .join(
                target,
                and_(
                    target.id == anchor_chunk.indexed_document_version_id,
                    target.workspace_id == anchor_chunk.workspace_id,
                    target.kb_id == anchor_chunk.kb_id,
                    target.index_revision_id
                    == bindparam(
                        "index_revision_id", type_=PostgreSQLUUID(as_uuid=True)
                    ),
                    target.build_status == IndexBuildStatus.READY,
                    target.serving_status == IndexServingStatus.SERVING,
                ),
            )
            .join(
                revision,
                and_(
                    revision.id == target.index_revision_id,
                    revision.workspace_id == target.workspace_id,
                    revision.kb_id == target.kb_id,
                    revision.status == IndexRevisionStatus.ACTIVE,
                ),
            )
            .join(
                knowledge_base,
                and_(
                    knowledge_base.id == target.kb_id,
                    knowledge_base.workspace_id == target.workspace_id,
                    knowledge_base.active_index_revision_id == revision.id,
                    knowledge_base.deleted_at.is_(None),
                ),
            )
            .join(
                document,
                and_(
                    document.id == target.document_id,
                    document.workspace_id == target.workspace_id,
                    document.kb_id == target.kb_id,
                    document.deleted_at.is_(None),
                ),
            )
            .join(
                document_version,
                and_(
                    document_version.id == target.document_version_id,
                    document_version.document_id == target.document_id,
                    document_version.workspace_id == target.workspace_id,
                    document_version.kb_id == target.kb_id,
                    document_version.source_status
                    == DocumentSourceStatus.AVAILABLE,
                ),
            )
            .cte("valid_adjacency_anchors")
        )

        neighbor = aliased(IndexChunk, name="adjacency_neighbor_chunk")
        asset = aliased(IndexAsset, name="adjacency_neighbor_asset")
        offset = (neighbor.ordinal - valid_anchors.c.anchor_ordinal).label(
            "adjacency_offset"
        )
        hits = (
            select(
                valid_anchors.c.anchor_rank,
                valid_anchors.c.anchor_chunk_id,
                valid_anchors.c.hit_workspace_id,
                valid_anchors.c.hit_knowledge_base_id,
                valid_anchors.c.hit_index_revision_id,
                neighbor.id.label("index_chunk_id"),
                valid_anchors.c.indexed_document_version_id,
                valid_anchors.c.document_id,
                valid_anchors.c.document_version_id,
                valid_anchors.c.document_display_name,
                valid_anchors.c.document_original_filename,
                neighbor.ordinal,
                neighbor.content,
                neighbor.source_location,
                neighbor.hierarchy,
                neighbor.source_metadata,
                neighbor.modality,
                neighbor.evidence_group_key,
                asset.id.label("index_asset_id"),
                asset.media_type.label("asset_media_type"),
                asset.checksum_sha256.label("asset_checksum_sha256"),
                asset.width.label("asset_width"),
                asset.height.label("asset_height"),
                offset,
                valid_anchors.c.build_status,
                valid_anchors.c.serving_status,
            )
            .select_from(valid_anchors)
            .join(
                neighbor,
                and_(
                    neighbor.indexed_document_version_id
                    == valid_anchors.c.indexed_document_version_id,
                    neighbor.workspace_id == valid_anchors.c.hit_workspace_id,
                    neighbor.kb_id == valid_anchors.c.hit_knowledge_base_id,
                    or_(
                        neighbor.ordinal
                        == valid_anchors.c.anchor_ordinal - 1,
                        neighbor.ordinal
                        == valid_anchors.c.anchor_ordinal + 1,
                    ),
                    neighbor.excluded_at.is_(None),
                    neighbor.modality.in_(("text", "table")),
                ),
            )
            .outerjoin(
                asset,
                and_(
                    asset.id == neighbor.index_asset_id,
                    asset.indexed_document_version_id
                    == neighbor.indexed_document_version_id,
                ),
            )
            .order_by(
                valid_anchors.c.anchor_rank,
                offset,
                neighbor.id,
            )
            .lateral("adjacent_hits")
        )
        validated_anchor_count = (
            select(func.count())
            .select_from(valid_anchors)
            .scalar_subquery()
            .label("validated_anchor_count")
        )
        return (
            select(
                KnowledgeBase.active_index_revision_id.label(
                    "active_revision_id"
                ),
                validated_anchor_count,
                hits.c.anchor_rank,
                hits.c.anchor_chunk_id,
                hits.c.hit_workspace_id,
                hits.c.hit_knowledge_base_id,
                hits.c.hit_index_revision_id,
                hits.c.index_chunk_id,
                hits.c.indexed_document_version_id,
                hits.c.document_id,
                hits.c.document_version_id,
                hits.c.document_display_name,
                hits.c.document_original_filename,
                hits.c.ordinal,
                hits.c.content,
                hits.c.source_location,
                hits.c.hierarchy,
                hits.c.source_metadata,
                hits.c.modality,
                hits.c.evidence_group_key,
                hits.c.index_asset_id,
                hits.c.asset_media_type,
                hits.c.asset_checksum_sha256,
                hits.c.asset_width,
                hits.c.asset_height,
                hits.c.adjacency_offset,
                hits.c.build_status,
                hits.c.serving_status,
            )
            .select_from(KnowledgeBase)
            .join(
                IndexRevision,
                and_(
                    IndexRevision.id == KnowledgeBase.active_index_revision_id,
                    IndexRevision.workspace_id == KnowledgeBase.workspace_id,
                    IndexRevision.kb_id == KnowledgeBase.id,
                    IndexRevision.status == IndexRevisionStatus.ACTIVE,
                ),
            )
            .outerjoin(hits, true())
            .where(
                KnowledgeBase.workspace_id
                == bindparam(
                    "workspace_id", type_=PostgreSQLUUID(as_uuid=True)
                ),
                KnowledgeBase.id
                == bindparam(
                    "knowledge_base_id", type_=PostgreSQLUUID(as_uuid=True)
                ),
                KnowledgeBase.deleted_at.is_(None),
            )
            .order_by(
                hits.c.anchor_rank.asc().nulls_last(),
                hits.c.adjacency_offset.asc().nulls_last(),
                hits.c.index_chunk_id.asc().nulls_last(),
            )
        )

    @staticmethod
    def _hit(row: Any) -> VectorSearchHit:
        return VectorSearchHit(
            workspace_id=row["hit_workspace_id"],
            knowledge_base_id=row["hit_knowledge_base_id"],
            index_revision_id=row["hit_index_revision_id"],
            index_chunk_id=row["index_chunk_id"],
            indexed_document_version_id=row["indexed_document_version_id"],
            document_id=row["document_id"],
            document_version_id=row["document_version_id"],
            document_display_name=row["document_display_name"],
            document_original_filename=row["document_original_filename"],
            ordinal=row["ordinal"],
            text=row["content"],
            source_location=dict(row["source_location"]),
            hierarchy=dict(row["hierarchy"]),
            source_metadata=dict(row["source_metadata"]),
            cosine_distance=float(row["cosine_distance"]),
            build_status=row["build_status"].value,
            serving_status=row["serving_status"].value,
            is_current_serving_version=True,
            modality=row["modality"],
            evidence_group_key=row["evidence_group_key"],
            representation_kind=row["representation_kind"],
            index_asset_id=row["index_asset_id"],
            asset_media_type=row["asset_media_type"],
            asset_checksum_sha256=row["asset_checksum_sha256"],
            asset_width=row["asset_width"],
            asset_height=row["asset_height"],
        )

    @staticmethod
    def _adjacent_hit(row: Any) -> AdjacentChunkHit:
        build_status = row["build_status"]
        serving_status = row["serving_status"]
        return AdjacentChunkHit(
            workspace_id=row["hit_workspace_id"],
            knowledge_base_id=row["hit_knowledge_base_id"],
            index_revision_id=row["hit_index_revision_id"],
            index_chunk_id=row["index_chunk_id"],
            indexed_document_version_id=row["indexed_document_version_id"],
            document_id=row["document_id"],
            document_version_id=row["document_version_id"],
            document_display_name=row["document_display_name"],
            document_original_filename=row["document_original_filename"],
            ordinal=row["ordinal"],
            text=row["content"],
            source_location=dict(row["source_location"]),
            hierarchy=dict(row["hierarchy"]),
            source_metadata=dict(row["source_metadata"]),
            anchor_index_chunk_id=row["anchor_chunk_id"],
            anchor_rank=row["anchor_rank"],
            offset=row["adjacency_offset"],
            build_status=(
                build_status.value
                if hasattr(build_status, "value")
                else str(build_status)
            ),
            serving_status=(
                serving_status.value
                if hasattr(serving_status, "value")
                else str(serving_status)
            ),
            is_current_serving_version=True,
            modality=row["modality"],
            evidence_group_key=row["evidence_group_key"],
            index_asset_id=row["index_asset_id"],
            asset_media_type=row["asset_media_type"],
            asset_checksum_sha256=row["asset_checksum_sha256"],
            asset_width=row["asset_width"],
            asset_height=row["asset_height"],
        )
