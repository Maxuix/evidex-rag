"""One-statement exact pgvector retrieval for the fixed P1A space."""

from __future__ import annotations

from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, and_, bindparam, select, true
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_kb.adapters.vector_store.fixed_pgvector import FixedPgVectorSpace
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
    VectorRecord,
)
from rag_kb.domain import (
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
        vector_space: FixedPgVectorSpace,
    ) -> None:
        self._sessions = sessions
        self._vector_space = vector_space

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
                IndexRevisionEmbeddingSpace.role == space_role,
            )
            .limit(1)
        )
        async with self._sessions() as session:
            return (await session.scalar(statement)) is not None

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
            expected_space=self._vector_space.configured_space,
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
        statement = self._statement()
        parameters = {
            "workspace_id": plan.workspace_id,
            "knowledge_base_id": plan.knowledge_base_id,
            "query_embedding": list(query_embedding),
            "top_k": plan.candidate_count or plan.top_k,
            "space_role": space_role,
            "representation_kinds": list(representation_kinds),
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
        configured = expected_space or self._vector_space.configured_space
        if (
            plan.strategy is not RetrievalStrategy.EXACT_VECTOR
            or plan.distance_metric != "cosine"
            or plan.ef_search is not None
            or plan.iterative_scan.value != "disabled"
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
    def _statement():
        query_vector = bindparam(
            "query_embedding",
            type_=Vector(FixedPgVectorSpace.dimension),
        )
        distance = VectorRecord.embedding.cosine_distance(query_vector).label(
            "cosine_distance"
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
                IndexChunk.ordinal.label("ordinal"),
                IndexChunk.content.label("content"),
                IndexChunk.source_location.label("source_location"),
                IndexChunk.hierarchy.label("hierarchy"),
                IndexChunk.source_metadata.label("source_metadata"),
                IndexChunk.modality.label("modality"),
                IndexChunk.evidence_group_key.label("evidence_group_key"),
                VectorRecord.representation_kind.label("representation_kind"),
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
                VectorRecord,
                and_(
                    VectorRecord.index_chunk_id == IndexChunk.id,
                    VectorRecord.kb_id == IndexChunk.kb_id,
                    VectorRecord.workspace_id == IndexChunk.workspace_id,
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
                DocumentVersion.source_status == DocumentSourceStatus.AVAILABLE,
                VectorRecord.embedding_space_id
                == IndexRevisionEmbeddingSpace.embedding_space_id,
                VectorRecord.representation_kind.in_(
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
            )
            .order_by(
                hits.c.cosine_distance.asc().nulls_last(),
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
