"""Recoverable lexical-index backfill for active serving targets."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import and_, exists, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_kb.db.models import (
    Document,
    DocumentSourceStatus,
    DocumentVersion,
    IndexBuildStatus,
    IndexChunk,
    IndexChunkLexical,
    IndexedDocumentVersion,
    IndexLexicalManifest,
    IndexRevisionEmbeddingSpace,
    IndexServingStatus,
    KnowledgeBase,
    VectorRecord,
)
from rag_kb.document_processing.lexical import (
    LEXICAL_ANALYZER_VERSION,
    analyze_document,
    lexical_manifest_hash,
)

MAX_TARGET_CHUNKS = 20_000


@dataclass(frozen=True, slots=True)
class LexicalBackfillTargetResult:
    indexed_document_version_id: UUID
    lexical_chunk_count: int
    status: str
    error_code: str | None = None


class LexicalBackfillService:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        workspace_id: UUID,
    ) -> None:
        self._sessions = sessions
        self._workspace_id = workspace_id

    async def run(
        self, *, kb_id: UUID | None = None
    ) -> tuple[LexicalBackfillTargetResult, ...]:
        async with self._sessions() as session:
            targets = (
                await session.execute(self._target_statement(kb_id))
            ).scalars().all()
        results: list[LexicalBackfillTargetResult] = []
        for target_id in targets:
            try:
                result = await self._backfill_target(target_id)
            except Exception:
                result = LexicalBackfillTargetResult(
                    target_id, 0, "failed", "INDEX_PERSISTENCE_FAILED"
                )
            results.append(result)
        return tuple(results)

    def _target_statement(self, kb_id: UUID | None):
        statement = (
            select(IndexedDocumentVersion.id)
            .join(
                KnowledgeBase,
                and_(
                    KnowledgeBase.workspace_id
                    == IndexedDocumentVersion.workspace_id,
                    KnowledgeBase.id == IndexedDocumentVersion.kb_id,
                    KnowledgeBase.active_index_revision_id
                    == IndexedDocumentVersion.index_revision_id,
                ),
            )
            .join(
                Document,
                and_(
                    Document.id == IndexedDocumentVersion.document_id,
                    Document.kb_id == IndexedDocumentVersion.kb_id,
                    Document.workspace_id
                    == IndexedDocumentVersion.workspace_id,
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
            .where(
                IndexedDocumentVersion.workspace_id == self._workspace_id,
                IndexedDocumentVersion.build_status == IndexBuildStatus.READY,
                IndexedDocumentVersion.serving_status
                == IndexServingStatus.SERVING,
                Document.deleted_at.is_(None),
                DocumentVersion.source_status
                == DocumentSourceStatus.AVAILABLE,
            )
            .order_by(IndexedDocumentVersion.id)
        )
        if kb_id is not None:
            statement = statement.where(IndexedDocumentVersion.kb_id == kb_id)
        return statement

    async def _backfill_target(
        self, target_id: UUID
    ) -> LexicalBackfillTargetResult:
        async with self._sessions() as session:
            active_target_id = await session.scalar(
                select(IndexedDocumentVersion.id)
                .join(
                    KnowledgeBase,
                    and_(
                        KnowledgeBase.workspace_id
                        == IndexedDocumentVersion.workspace_id,
                        KnowledgeBase.id == IndexedDocumentVersion.kb_id,
                        KnowledgeBase.active_index_revision_id
                        == IndexedDocumentVersion.index_revision_id,
                    ),
                )
                .join(
                    Document,
                    and_(
                        Document.id == IndexedDocumentVersion.document_id,
                        Document.kb_id == IndexedDocumentVersion.kb_id,
                        Document.workspace_id
                        == IndexedDocumentVersion.workspace_id,
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
                .where(
                    IndexedDocumentVersion.workspace_id == self._workspace_id,
                    IndexedDocumentVersion.id == target_id,
                    IndexedDocumentVersion.build_status
                    == IndexBuildStatus.READY,
                    IndexedDocumentVersion.serving_status
                    == IndexServingStatus.SERVING,
                    Document.deleted_at.is_(None),
                    DocumentVersion.source_status
                    == DocumentSourceStatus.AVAILABLE,
                )
            )
            if active_target_id is None:
                return LexicalBackfillTargetResult(
                    target_id, 0, "skipped", "INDEX_TARGET_INVALID"
                )
            existing = await session.get(
                IndexLexicalManifest,
                (target_id, LEXICAL_ANALYZER_VERSION),
            )
            if existing is not None:
                stored = (
                    await session.execute(
                        select(
                            IndexChunkLexical.index_chunk_id,
                            IndexChunkLexical.lexical_text_hash,
                        )
                        .where(
                            IndexChunkLexical.indexed_document_version_id
                            == target_id,
                            IndexChunkLexical.analyzer_version
                            == LEXICAL_ANALYZER_VERSION,
                        )
                        .order_by(IndexChunkLexical.index_chunk_id)
                    )
                ).all()
                stored_hash = lexical_manifest_hash(
                    LEXICAL_ANALYZER_VERSION,
                    (
                        (item.index_chunk_id, item.lexical_text_hash)
                        for item in stored
                    ),
                )
                if (
                    existing.lexical_chunk_count != len(stored)
                    or existing.lexical_manifest_hash != stored_hash
                ):
                    raise RuntimeError("existing lexical manifest differs")
                return LexicalBackfillTargetResult(
                    target_id, existing.lexical_chunk_count, "skipped"
                )
            chunks = (
                await session.execute(
                    select(
                        IndexChunk.id,
                        IndexChunk.embedding_text,
                        IndexChunk.content,
                    )
                    .join(
                        IndexedDocumentVersion,
                        IndexedDocumentVersion.id
                        == IndexChunk.indexed_document_version_id,
                    )
                    .where(
                        IndexedDocumentVersion.workspace_id
                        == self._workspace_id,
                        IndexedDocumentVersion.id == target_id,
                        exists(
                            select(VectorRecord.id)
                            .join(
                                IndexRevisionEmbeddingSpace,
                                and_(
                                    IndexRevisionEmbeddingSpace.index_revision_id
                                    == IndexedDocumentVersion.index_revision_id,
                                    IndexRevisionEmbeddingSpace.embedding_space_id
                                    == VectorRecord.embedding_space_id,
                                    IndexRevisionEmbeddingSpace.role
                                    == "text_retrieval",
                                ),
                            )
                            .where(
                                VectorRecord.index_chunk_id == IndexChunk.id,
                                VectorRecord.representation_kind.in_(
                                    (
                                        "text",
                                        "caption_text",
                                        "ocr_text",
                                        "table_text",
                                    )
                                ),
                            )
                        ),
                    )
                    .order_by(IndexChunk.id)
                    .limit(MAX_TARGET_CHUNKS + 1)
                )
            ).all()
        if len(chunks) > MAX_TARGET_CHUNKS:
            raise RuntimeError("lexical backfill target exceeds chunk limit")
        rows = await asyncio.to_thread(_analyze_rows, chunks)
        manifest_hash = lexical_manifest_hash(
            LEXICAL_ANALYZER_VERSION,
            ((item["index_chunk_id"], item["lexical_text_hash"]) for item in rows),
        )
        async with self._sessions() as session:
            async with session.begin():
                target = await session.scalar(
                    select(IndexedDocumentVersion)
                    .join(
                        KnowledgeBase,
                        and_(
                            KnowledgeBase.workspace_id
                            == IndexedDocumentVersion.workspace_id,
                            KnowledgeBase.id == IndexedDocumentVersion.kb_id,
                            KnowledgeBase.active_index_revision_id
                            == IndexedDocumentVersion.index_revision_id,
                        ),
                    )
                    .join(
                        Document,
                        and_(
                            Document.id
                            == IndexedDocumentVersion.document_id,
                            Document.kb_id == IndexedDocumentVersion.kb_id,
                            Document.workspace_id
                            == IndexedDocumentVersion.workspace_id,
                        ),
                    )
                    .join(
                        DocumentVersion,
                        and_(
                            DocumentVersion.id
                            == IndexedDocumentVersion.document_version_id,
                            DocumentVersion.document_id
                            == IndexedDocumentVersion.document_id,
                            DocumentVersion.kb_id
                            == IndexedDocumentVersion.kb_id,
                            DocumentVersion.workspace_id
                            == IndexedDocumentVersion.workspace_id,
                        ),
                    )
                    .where(
                        IndexedDocumentVersion.workspace_id
                        == self._workspace_id,
                        IndexedDocumentVersion.id == target_id,
                        IndexedDocumentVersion.build_status
                        == IndexBuildStatus.READY,
                        IndexedDocumentVersion.serving_status
                        == IndexServingStatus.SERVING,
                        Document.deleted_at.is_(None),
                        DocumentVersion.source_status
                        == DocumentSourceStatus.AVAILABLE,
                    )
                    .with_for_update()
                )
                if target is None:
                    return LexicalBackfillTargetResult(
                        target_id, 0, "skipped", "INDEX_TARGET_INVALID"
                    )
                if rows:
                    await session.execute(
                        pg_insert(IndexChunkLexical)
                        .values(
                            [
                                {
                                    **item,
                                    "workspace_id": self._workspace_id,
                                    "kb_id": target.kb_id,
                                    "indexed_document_version_id": target.id,
                                    "analyzer_version": LEXICAL_ANALYZER_VERSION,
                                }
                                for item in rows
                            ]
                        )
                        .on_conflict_do_nothing(
                            index_elements=[
                                "index_chunk_id",
                                "analyzer_version",
                            ]
                        )
                    )
                stored = (
                    await session.execute(
                        select(
                            IndexChunkLexical.index_chunk_id,
                            IndexChunkLexical.lexical_text_hash,
                        )
                        .where(
                            IndexChunkLexical.indexed_document_version_id
                            == target.id,
                            IndexChunkLexical.analyzer_version
                            == LEXICAL_ANALYZER_VERSION,
                        )
                        .order_by(IndexChunkLexical.index_chunk_id)
                    )
                ).all()
                if (
                    len(stored) != len(rows)
                    or lexical_manifest_hash(
                        LEXICAL_ANALYZER_VERSION,
                        (
                            (item.index_chunk_id, item.lexical_text_hash)
                            for item in stored
                        ),
                    )
                    != manifest_hash
                ):
                    raise RuntimeError("lexical backfill content differs")
                await session.execute(
                    pg_insert(IndexLexicalManifest)
                    .values(
                        indexed_document_version_id=target.id,
                        analyzer_version=LEXICAL_ANALYZER_VERSION,
                        workspace_id=self._workspace_id,
                        kb_id=target.kb_id,
                        lexical_chunk_count=len(rows),
                        lexical_manifest_hash=manifest_hash,
                    )
                    .on_conflict_do_nothing(
                        index_elements=[
                            "indexed_document_version_id",
                            "analyzer_version",
                        ]
                    )
                )
        return LexicalBackfillTargetResult(target_id, len(rows), "completed")


def _analyze_rows(chunks) -> tuple[dict, ...]:
    rows: list[dict] = []
    for chunk in chunks:
        analyzed = analyze_document(chunk.embedding_text or chunk.content)
        if analyzed is None:
            continue
        rows.append(
            {
                "index_chunk_id": chunk.id,
                "lexical_text": analyzed.lexical_text,
                "lexical_text_hash": analyzed.lexical_text_hash,
            }
        )
    return tuple(rows)
