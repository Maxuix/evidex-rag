"""PostgreSQL FTS retrieval with target-completeness validation."""

from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_kb.db.models import IndexBuildStatus, IndexServingStatus
from rag_kb.document_processing.lexical import (
    LEXICAL_ANALYZER_VERSION,
    LEXICAL_QUERY_VERSION,
    analyze_query,
    build_or_tsquery,
    lexical_manifest_hash,
)
from rag_kb.domain import (
    ErrorCode,
    LexicalManifestStatus,
    LexicalSearchResult,
    RetrievalExecutionError,
    RetrievalQueryPlan,
    ServingScopeQuery,
    VectorSearchHit,
)


class PgLexicalStore:
    """Serve only complete lexical targets from one repeatable-read snapshot."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def search(
        self,
        plan: RetrievalQueryPlan,
        query: str,
        query_embedding: tuple[float, ...],
        *,
        analyzer_version: str = LEXICAL_ANALYZER_VERSION,
        query_version: str = LEXICAL_QUERY_VERSION,
        candidate_count: int,
    ) -> LexicalSearchResult | None:
        if (
            analyzer_version != LEXICAL_ANALYZER_VERSION
            or query_version != LEXICAL_QUERY_VERSION
        ):
            raise RetrievalExecutionError(
                ErrorCode.INDEX_REVISION_INCOMPATIBLE,
                diagnostic={"check": "lexical_version"},
            )
        if not 1 <= candidate_count <= 100:
            raise ValueError("lexical candidate count must be between 1 and 100")
        lexemes = analyze_query(query)
        tsquery = build_or_tsquery(lexemes)
        async with self._sessions() as session:
            async with session.begin():
                await session.execute(
                    text(
                        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
                    )
                )
                resolved = await self._serving_scope(
                    session,
                    workspace_id=plan.workspace_id,
                    knowledge_base_id=plan.knowledge_base_id,
                )
                if resolved is None:
                    return None
                active_revision_id, target_ids = resolved
                await self._validate_manifests(
                    session,
                    target_ids,
                    analyzer_version=analyzer_version,
                )
                if tsquery is None or not target_ids:
                    return LexicalSearchResult(
                        resolved_active_revision_id=active_revision_id,
                        analyzer_version=analyzer_version,
                        manifest_target_count=len(target_ids),
                    )
                dimension = len(query_embedding)
                statement = self._statement().bindparams(
                    bindparam("query_embedding", type_=Vector(dimension))
                )
                rows = (
                    await session.execute(
                        statement,
                        {
                            "workspace_id": plan.workspace_id,
                            "kb_id": plan.knowledge_base_id,
                            "revision_id": active_revision_id,
                            "target_ids": list(target_ids),
                            "analyzer_version": analyzer_version,
                            "tsquery": tsquery,
                            "query_embedding": list(query_embedding),
                            "embedding_dimension": dimension,
                            "candidate_count": candidate_count,
                        },
                    )
                ).mappings().all()
        hits = tuple(
            self._hit(row, lexical_rank=rank)
            for rank, row in enumerate(rows, start=1)
        )
        return LexicalSearchResult(
            resolved_active_revision_id=active_revision_id,
            analyzer_version=analyzer_version,
            manifest_target_count=len(target_ids),
            hits=hits,
        )

    async def manifest_status(
        self, query: ServingScopeQuery
    ) -> LexicalManifestStatus | None:
        async with self._sessions() as session:
            async with session.begin():
                await session.execute(
                    text(
                        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
                    )
                )
                resolved = await self._serving_scope(
                    session,
                    workspace_id=query.workspace_id,
                    knowledge_base_id=query.knowledge_base_id,
                )
                if resolved is None:
                    return None
                active_revision_id, target_ids = resolved
                if not target_ids:
                    return LexicalManifestStatus(
                        resolved_active_revision_id=active_revision_id,
                        serving_target_count=0,
                        manifested_target_count=0,
                    )
                manifested = (
                    await session.execute(
                        text(
                            "SELECT indexed_document_version_id "
                            "FROM index_lexical_manifest "
                            "WHERE indexed_document_version_id IN :target_ids "
                            "AND analyzer_version = :analyzer_version"
                        ).bindparams(bindparam("target_ids", expanding=True)),
                        {
                            "target_ids": list(target_ids),
                            "analyzer_version": LEXICAL_ANALYZER_VERSION,
                        },
                    )
                ).scalars().all()
        return LexicalManifestStatus(
            resolved_active_revision_id=active_revision_id,
            serving_target_count=len(target_ids),
            manifested_target_count=len(set(manifested)),
        )

    @staticmethod
    async def _serving_scope(
        session: AsyncSession,
        *,
        workspace_id,
        knowledge_base_id,
    ) -> tuple | None:
        active_revision_id = await session.scalar(
            text(
                "SELECT kb.active_index_revision_id "
                "FROM knowledge_base kb "
                "JOIN index_revision revision "
                " ON revision.id = kb.active_index_revision_id "
                " AND revision.kb_id = kb.id "
                " AND revision.workspace_id = kb.workspace_id "
                " AND revision.status = 'active' "
                "WHERE kb.workspace_id = :workspace_id "
                " AND kb.id = :kb_id "
                " AND kb.deleted_at IS NULL"
            ),
            {
                "workspace_id": workspace_id,
                "kb_id": knowledge_base_id,
            },
        )
        if active_revision_id is None:
            return None
        target_ids = tuple(
            (
                await session.execute(
                    text(
                        "SELECT target.id "
                        "FROM indexed_document_version target "
                        "JOIN document doc ON doc.id = target.document_id "
                        " AND doc.kb_id = target.kb_id "
                        " AND doc.workspace_id = target.workspace_id "
                        "JOIN document_version version "
                        " ON version.id = target.document_version_id "
                        " AND version.document_id = target.document_id "
                        " AND version.kb_id = target.kb_id "
                        " AND version.workspace_id = target.workspace_id "
                        "WHERE target.workspace_id = :workspace_id "
                        " AND target.kb_id = :kb_id "
                        " AND target.index_revision_id = :revision_id "
                        " AND target.build_status = 'ready' "
                        " AND target.serving_status = 'serving' "
                        " AND doc.deleted_at IS NULL "
                        " AND version.source_status = 'available' "
                        "ORDER BY target.id"
                    ),
                    {
                        "workspace_id": workspace_id,
                        "kb_id": knowledge_base_id,
                        "revision_id": active_revision_id,
                    },
                )
            ).scalars().all()
        )
        return active_revision_id, target_ids

    @staticmethod
    async def _validate_manifests(
        session: AsyncSession,
        target_ids: tuple,
        *,
        analyzer_version: str,
    ) -> None:
        if not target_ids:
            return
        manifests = (
            await session.execute(
                text(
                    "SELECT indexed_document_version_id, lexical_chunk_count, "
                    "lexical_manifest_hash FROM index_lexical_manifest "
                    "WHERE indexed_document_version_id IN :target_ids "
                    "AND analyzer_version = :analyzer_version"
                ).bindparams(bindparam("target_ids", expanding=True)),
                {
                    "target_ids": list(target_ids),
                    "analyzer_version": analyzer_version,
                },
            )
        ).mappings().all()
        by_target = {
            item["indexed_document_version_id"]: item for item in manifests
        }
        if set(by_target) != set(target_ids):
            raise RetrievalExecutionError(
                ErrorCode.INDEX_REVISION_INCOMPATIBLE,
                diagnostic={"check": "lexical_manifest_coverage"},
            )
        hashes = (
            await session.execute(
                text(
                    "SELECT indexed_document_version_id, index_chunk_id, "
                    "lexical_text_hash FROM index_chunk_lexical "
                    "WHERE indexed_document_version_id IN :target_ids "
                    "AND analyzer_version = :analyzer_version "
                    "ORDER BY indexed_document_version_id, index_chunk_id"
                ).bindparams(bindparam("target_ids", expanding=True)),
                {
                    "target_ids": list(target_ids),
                    "analyzer_version": analyzer_version,
                },
            )
        ).mappings().all()
        rows_by_target = {target_id: [] for target_id in target_ids}
        for item in hashes:
            rows_by_target[item["indexed_document_version_id"]].append(
                (item["index_chunk_id"], item["lexical_text_hash"])
            )
        for target_id in target_ids:
            manifest = by_target[target_id]
            rows = rows_by_target[target_id]
            if (
                manifest["lexical_chunk_count"] != len(rows)
                or manifest["lexical_manifest_hash"]
                != lexical_manifest_hash(analyzer_version, rows)
            ):
                raise RetrievalExecutionError(
                    ErrorCode.INDEX_REVISION_INCOMPATIBLE,
                    diagnostic={"check": "lexical_manifest_hash"},
                )

    @staticmethod
    def _statement():
        return text(
            """
            WITH lexical_candidates AS MATERIALIZED (
              SELECT lexical.index_chunk_id,
                     ts_rank_cd(
                       lexical.lexical_tsv,
                       to_tsquery('simple', :tsquery),
                       32
                     ) AS lexical_score
              FROM index_chunk_lexical lexical
              JOIN index_chunk admitted_chunk
                ON admitted_chunk.id = lexical.index_chunk_id
                AND admitted_chunk.workspace_id = lexical.workspace_id
                AND admitted_chunk.kb_id = lexical.kb_id
              WHERE lexical.workspace_id = :workspace_id
                AND lexical.kb_id = :kb_id
                AND lexical.indexed_document_version_id IN :target_ids
                AND lexical.analyzer_version = :analyzer_version
                AND admitted_chunk.excluded_at IS NULL
                AND lexical.lexical_tsv @@ to_tsquery('simple', :tsquery)
              ORDER BY lexical_score DESC, lexical.index_chunk_id
              LIMIT :candidate_count
            ),
            best_vectors AS (
              SELECT DISTINCT ON (chunk.id)
                     chunk.workspace_id AS hit_workspace_id,
                     chunk.kb_id AS hit_knowledge_base_id,
                     target.index_revision_id AS hit_index_revision_id,
                     chunk.id AS index_chunk_id,
                     target.id AS indexed_document_version_id,
                     doc.id AS document_id,
                     version.id AS document_version_id,
                     doc.display_name AS document_display_name,
                     version.original_filename AS document_original_filename,
                     chunk.ordinal,
                     chunk.content,
                     chunk.source_location,
                     chunk.hierarchy,
                     chunk.source_metadata,
                     chunk.modality,
                     chunk.evidence_group_key,
                     vector.representation_kind,
                     asset.id AS index_asset_id,
                     asset.media_type AS asset_media_type,
                     asset.checksum_sha256 AS asset_checksum_sha256,
                     asset.width AS asset_width,
                     asset.height AS asset_height,
                     vector.embedding <=> :query_embedding AS cosine_distance,
                     target.build_status,
                     target.serving_status,
                     candidate.lexical_score
              FROM lexical_candidates candidate
              JOIN index_chunk chunk ON chunk.id = candidate.index_chunk_id
              JOIN indexed_document_version target
                ON target.id = chunk.indexed_document_version_id
              JOIN document doc ON doc.id = target.document_id
                AND doc.kb_id = target.kb_id
                AND doc.workspace_id = target.workspace_id
              JOIN document_version version ON version.id = target.document_version_id
                AND version.document_id = target.document_id
                AND version.kb_id = target.kb_id
                AND version.workspace_id = target.workspace_id
              JOIN index_revision_embedding_space binding
                ON binding.index_revision_id = target.index_revision_id
                AND binding.workspace_id = target.workspace_id
                AND binding.role = 'text_retrieval'
              JOIN vector_record vector ON vector.index_chunk_id = chunk.id
                AND vector.kb_id = chunk.kb_id
                AND vector.workspace_id = chunk.workspace_id
                AND vector.embedding_space_id = binding.embedding_space_id
                AND vector.embedding_dimension = :embedding_dimension
                AND vector.representation_kind IN
                    ('text', 'caption_text', 'ocr_text', 'table_text')
              LEFT JOIN index_asset asset ON asset.id = chunk.index_asset_id
                AND asset.indexed_document_version_id = chunk.indexed_document_version_id
              WHERE target.workspace_id = :workspace_id
                AND target.kb_id = :kb_id
                AND target.index_revision_id = :revision_id
                AND target.id IN :target_ids
                AND target.build_status = 'ready'
                AND target.serving_status = 'serving'
                AND doc.deleted_at IS NULL
                AND chunk.excluded_at IS NULL
                AND version.source_status = 'available'
              ORDER BY chunk.id, cosine_distance,
                       CASE vector.representation_kind
                         WHEN 'text' THEN 0 WHEN 'caption_text' THEN 1
                         WHEN 'ocr_text' THEN 2 ELSE 3 END
            )
            SELECT * FROM best_vectors
            ORDER BY lexical_score DESC, index_chunk_id
            LIMIT :candidate_count
            """
        ).bindparams(bindparam("target_ids", expanding=True))

    @staticmethod
    def _hit(row, *, lexical_rank: int) -> VectorSearchHit:
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
            build_status=_enum_value(row["build_status"], IndexBuildStatus),
            serving_status=_enum_value(row["serving_status"], IndexServingStatus),
            is_current_serving_version=True,
            modality=row["modality"],
            evidence_group_key=row["evidence_group_key"],
            representation_kind=row["representation_kind"],
            index_asset_id=row["index_asset_id"],
            asset_media_type=row["asset_media_type"],
            asset_checksum_sha256=row["asset_checksum_sha256"],
            asset_width=row["asset_width"],
            asset_height=row["asset_height"],
            lexical_rank=lexical_rank,
            lexical_score=float(row["lexical_score"]),
        )


def _enum_value(value, enum_type) -> str:
    try:
        return enum_type(value).value
    except ValueError:
        return value.value
