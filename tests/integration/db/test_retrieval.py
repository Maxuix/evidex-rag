from __future__ import annotations

import asyncio
import json
import os
import unittest
from dataclasses import dataclass, replace
from uuid import UUID, uuid4

import asyncpg
from sqlalchemy import event

from tests.integration.db import require_database_test_dsns
from rag_kb.adapters.lexical_store.postgres import PgLexicalStore
from rag_kb.adapters.vector_store.pgvector import PgVectorStore
from rag_kb.db import DatabaseProcess, create_database_resources
from rag_kb.domain import (
    EmbeddingSpaceDefinition,
    ErrorCode,
    Evidence,
    EvidenceScoreKind,
    ResourceNotFoundError,
    RetrievalExecutionError,
    RetrievalRequest,
    RerankMode,
    RetrievalStrategy,
    SERVING_DOCUMENT_LIST_LIMIT,
)
from rag_kb.document_processing.lexical import (
    LEXICAL_ANALYZER_VERSION,
    analyze_document,
    lexical_manifest_hash,
)
from rag_kb.retrieval.service import RetrievalService
from rag_kb.services.composite_evidence import CompositeEvidenceHydrationService
from rag_kb.services.content import DocumentService
from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWorkFactory


MIGRATION_DSN = os.environ.get("RAG_KB_TEST_MIGRATION_DSN")
RUNTIME_SQLALCHEMY_DSN = os.environ.get("RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN")
WORKSPACE = UUID("01900000-0000-7000-8000-000000001001")
OTHER_WORKSPACE = UUID("01900000-0000-7000-8000-000000001002")
EXPECTED_FINGERPRINT = "sha256:retrieval-compatible"

require_database_test_dsns(
    "RAG_KB_TEST_MIGRATION_DSN",
    "RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN",
)
class ExactRetrievalDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            await connection.execute("TRUNCATE TABLE workspace CASCADE")
        finally:
            await connection.close()
        self.database = create_database_resources(
            RUNTIME_SQLALCHEMY_DSN,
            pool_size=3,
            max_overflow=0,
            process=DatabaseProcess.API,
        )
        self.definition = _embedding_space()
        self.provider = _Provider(self.definition, _axis_vector(0))
        self.vector_store = PgVectorStore(
            self.database.sessions,
            self.definition,
        )
        self.hydrator = CompositeEvidenceHydrationService(
            SqlAlchemyUnitOfWorkFactory(self.database.sessions, WORKSPACE)
        )
        self.documents = DocumentService(
            SqlAlchemyUnitOfWorkFactory(self.database.sessions, WORKSPACE),
        )
        self.service = RetrievalService(
            WORKSPACE,
            self.provider,
            self.vector_store,
        )

    async def asyncTearDown(self) -> None:
        await self.database.close()

    async def test_role_probe_and_exact_search_keep_stable_top_k(self) -> None:
        foundation = await self._foundation()
        first = UUID("01900000-0000-7000-8000-000000001111")
        second = UUID("01900000-0000-7000-8000-000000001112")
        third = UUID("01900000-0000-7000-8000-000000001113")
        await self._target(foundation, chunk_id=second, vector=_axis_vector(0))
        await self._target(foundation, chunk_id=first, vector=_axis_vector(0))
        await self._target(foundation, chunk_id=third, vector=_axis_vector(1))
        statements: list[str] = []

        def capture_statement(*args) -> None:
            statements.append(args[2])

        event.listen(
            self.database.engine.sync_engine,
            "before_cursor_execute",
            capture_statement,
        )
        try:
            pack = await self.service.retrieve(
                RetrievalRequest(
                    foundation.kb_id,
                    "query",
                    top_k=2,
                    include_debug=True,
                ),
            )
        finally:
            event.remove(
                self.database.engine.sync_engine,
                "before_cursor_execute",
                capture_statement,
            )

        self.assertEqual(len(statements), 2)
        role_probe = next(
            item for item in statements if "index_revision_embedding_space" in item
            and "<=>" not in item
        )
        exact_search = next(item for item in statements if "<=>" in item)
        self.assertIn("knowledge_base", role_probe)
        self.assertIn("LEFT OUTER JOIN LATERAL", exact_search)
        self.assertIn("workspace_id", exact_search)
        self.assertIn("knowledge_base_id", exact_search)
        self.assertEqual(pack.index_revision_id, foundation.revision_id)
        self.assertEqual(
            tuple(item.index_chunk_id for item in pack.evidence),
            (first, second),
        )
        self.assertEqual(tuple(item.rank for item in pack.evidence), (1, 2))
        self.assertTrue(all(item.score == 1.0 for item in pack.evidence))
        self.assertIsNotNone(pack.debug)
        assert pack.debug is not None
        self.assertEqual(pack.debug.result_count, 2)

    async def test_adjacency_is_one_scoped_read_with_shared_neighbor_links(
        self,
    ) -> None:
        foundation = await self._foundation()
        target = await self._target(
            foundation,
            chunk_id=uuid4(),
            vector=_axis_vector(0),
        )
        chunk_ids = {ordinal: uuid4() for ordinal in range(1, 6)}
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            async with connection.transaction():
                for ordinal, modality, excluded in (
                    (1, "text", True),
                    (2, "text", False),
                    (3, "table", False),
                    (4, "table", False),
                    (5, "image", False),
                ):
                    chunk_id = chunk_ids[ordinal]
                    await connection.execute(
                        """
                        INSERT INTO index_chunk (
                            id, workspace_id, kb_id,
                            indexed_document_version_id, ordinal, content,
                            content_hash, token_count, source_location,
                            hierarchy, source_metadata, unit_key, modality,
                            excluded_at
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, 2,
                            jsonb_build_object('ordinal', $5::integer),
                            '{}'::jsonb, '{}'::jsonb, $8, $9,
                            CASE WHEN $10 THEN now() ELSE NULL END
                        )
                        """,
                        chunk_id,
                        foundation.workspace_id,
                        foundation.kb_id,
                        target.indexed_document_version_id,
                        ordinal,
                        "" if modality == "image" else f"chunk-{ordinal}",
                        chunk_id.hex.ljust(64, "0")[:64],
                        f"adjacency:{chunk_id}",
                        modality,
                        excluded,
                    )
        finally:
            await connection.close()

        anchors = tuple(
            Evidence(
                rank=rank,
                index_chunk_id=chunk_ids[ordinal],
                indexed_document_version_id=target.indexed_document_version_id,
                document_id=target.document_id,
                document_version_id=target.document_version_id,
                index_revision_id=foundation.revision_id,
                ordinal=ordinal,
                text=f"chunk-{ordinal}",
                source_location={"ordinal": ordinal},
                hierarchy={},
                source_metadata={},
                score=0.9,
                modality="text" if ordinal == 2 else "table",
            )
            for rank, ordinal in enumerate((2, 4), start=1)
        )
        statements: list[str] = []

        def capture_statement(*args) -> None:
            statements.append(args[2])

        event.listen(
            self.database.engine.sync_engine,
            "before_cursor_execute",
            capture_statement,
        )
        try:
            evidence = await self.service.retrieve_adjacent_evidence(
                knowledge_base_id=foundation.kb_id,
                index_revision_id=foundation.revision_id,
                anchors=anchors,
            )
        finally:
            event.remove(
                self.database.engine.sync_engine,
                "before_cursor_execute",
                capture_statement,
            )

        self.assertEqual(len(statements), 1)
        self.assertNotIn("vector_record", statements[0])
        self.assertEqual(len(evidence), 2)
        self.assertEqual(
            {item.index_chunk_id for item in evidence},
            {chunk_ids[3]},
        )
        self.assertEqual(
            tuple(item.adjacency_anchor_index_chunk_id for item in evidence),
            (chunk_ids[2], chunk_ids[4]),
        )
        self.assertEqual(tuple(item.adjacency_offset for item in evidence), (1, -1))
        self.assertTrue(
            all(
                item.score_kind is EvidenceScoreKind.ADJACENCY
                and item.score == 0.0
                and item.modality == "table"
                and item.matched_representations == ("table_text",)
                for item in evidence
            )
        )

    async def test_hybrid_fts_validates_manifest_and_returns_lane_rank(
        self,
    ) -> None:
        foundation = await self._foundation()
        target = await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001121"),
            vector=_axis_vector(0),
        )
        await self._lexical_target(foundation, target)
        lexical_store = PgLexicalStore(self.database.sessions)
        service = RetrievalService(
            WORKSPACE,
            self.provider,
            self.vector_store,
            lexical_store=lexical_store,
            hybrid_enabled=True,
        )

        pack = await service.retrieve(
            RetrievalRequest(
                foundation.kb_id,
                "evidence",
                top_k=3,
                strategy=RetrievalStrategy.HYBRID,
                rerank_mode=RerankMode.CLASSIC,
                include_debug=True,
            ),
        )

        self.assertEqual(len(pack.evidence), 1)
        self.assertEqual(pack.evidence[0].index_chunk_id, target.chunk_id)
        self.assertEqual(pack.evidence[0].lexical_rank, 1)
        self.assertEqual(pack.evidence[0].text_space_rank, 1)
        assert pack.debug is not None
        self.assertEqual(pack.debug.lexical_candidate_count, 1)
        self.assertEqual(pack.debug.lexical_manifest_target_count, 1)
        self.assertEqual(
            pack.debug.lexical_analyzer_version,
            LEXICAL_ANALYZER_VERSION,
        )

    async def test_excluded_chunk_remains_manifest_complete_but_is_not_retrieved(
        self,
    ) -> None:
        foundation = await self._foundation()
        target = await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001120"),
            vector=_axis_vector(0),
        )
        await self._lexical_target(foundation, target)
        excluded_at = await self.documents.exclude_chunk(
            document_id=target.document_id,
            chunk_id=target.chunk_id,
        )
        self.assertIsNotNone(excluded_at)

        exact = await self.service.retrieve(
            RetrievalRequest(foundation.kb_id, "evidence", top_k=3),
        )
        hybrid_service = RetrievalService(
            WORKSPACE,
            self.provider,
            self.vector_store,
            lexical_store=PgLexicalStore(self.database.sessions),
            hybrid_enabled=True,
        )
        hybrid = await hybrid_service.retrieve(
            RetrievalRequest(
                foundation.kb_id,
                "evidence",
                top_k=3,
                strategy=RetrievalStrategy.HYBRID,
                rerank_mode=RerankMode.CLASSIC,
                include_debug=True,
            ),
        )

        self.assertEqual(exact.evidence, ())
        self.assertEqual(hybrid.evidence, ())
        assert hybrid.debug is not None
        self.assertEqual(hybrid.debug.lexical_manifest_target_count, 1)

    async def test_hybrid_fts_rejects_partial_target_backfill(self) -> None:
        foundation = await self._foundation()
        target = await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001122"),
            vector=_axis_vector(0),
        )
        await self._lexical_target(
            foundation,
            target,
            create_manifest=False,
        )
        service = RetrievalService(
            WORKSPACE,
            self.provider,
            self.vector_store,
            lexical_store=PgLexicalStore(self.database.sessions),
            hybrid_enabled=True,
        )

        with self.assertRaises(RetrievalExecutionError) as failure:
            await service.retrieve(
                RetrievalRequest(
                    foundation.kb_id,
                    "evidence",
                    strategy=RetrievalStrategy.HYBRID,
                    rerank_mode=RerankMode.CLASSIC,
                ),
            )

        self.assertEqual(
            failure.exception.code,
            ErrorCode.INDEX_REVISION_INCOMPATIBLE,
        )
        self.assertEqual(
            failure.exception.diagnostic,
            {"check": "lexical_manifest_coverage"},
        )

    async def test_lexical_only_hits_and_skips_excluded_chunks(self) -> None:
        foundation = await self._foundation()
        kept = await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001123"),
            vector=_axis_vector(0),
        )
        excluded = await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001124"),
            vector=_axis_vector(1),
        )
        await self._lexical_target(foundation, kept)
        await self._lexical_target(foundation, excluded)
        await self.documents.exclude_chunk(
            document_id=excluded.document_id,
            chunk_id=excluded.chunk_id,
        )
        service = RetrievalService(
            WORKSPACE,
            self.provider,
            self.vector_store,
            lexical_store=PgLexicalStore(self.database.sessions),
            hybrid_enabled=True,
        )

        pack = await service.retrieve_lexical_only(
            RetrievalRequest(
                foundation.kb_id,
                "evidence",
                top_k=3,
                include_debug=True,
            ),
        )

        self.assertEqual(
            [item.index_chunk_id for item in pack.evidence],
            [kept.chunk_id],
        )
        self.assertIs(pack.evidence[0].score_kind, EvidenceScoreKind.LEXICAL)
        self.assertEqual(pack.evidence[0].lexical_rank, 1)
        self.assertIsNone(pack.evidence[0].vector_similarity)
        assert pack.debug is not None
        self.assertEqual(pack.debug.lexical_candidate_count, 1)

    async def test_manifest_status_reports_coverage_without_hash(self) -> None:
        foundation = await self._foundation()
        complete = await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001125"),
            vector=_axis_vector(0),
        )
        missing = await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001126"),
            vector=_axis_vector(1),
        )
        await self._lexical_target(foundation, complete)
        await self._lexical_target(foundation, missing, create_manifest=False)
        service = RetrievalService(
            WORKSPACE,
            self.provider,
            self.vector_store,
            lexical_store=PgLexicalStore(self.database.sessions),
            hybrid_enabled=True,
        )

        status = await service.lexical_manifest_status(foundation.kb_id)

        self.assertIsNotNone(status)
        assert status is not None
        self.assertEqual(status.resolved_active_revision_id, foundation.revision_id)
        self.assertEqual(status.serving_target_count, 2)
        self.assertEqual(status.manifested_target_count, 1)
        self.assertFalse(status.complete)

    async def test_list_serving_documents_counts_outline_and_truncation(
        self,
    ) -> None:
        foundation = await self._foundation()
        first = await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001130"),
            vector=_axis_vector(0),
        )
        await self._set_chunk_hierarchy(
            first.chunk_id,
            {
                "titles": [
                    {"depth": 1, "text": "Section"},
                    {"depth": 0, "text": "Root"},
                    {"depth": 0, "text": "Root"},
                    {"depth": 0, "text": "Also root"},
                ]
            },
        )
        listed = await self.service.list_serving_documents(foundation.kb_id)
        self.assertIsNotNone(listed)
        assert listed is not None
        self.assertEqual(listed.resolved_active_revision_id, foundation.revision_id)
        self.assertFalse(listed.truncated)
        self.assertEqual(len(listed.entries), 1)
        outlined = listed.entries[0]
        self.assertEqual(outlined.document_id, first.document_id)
        self.assertEqual(outlined.chunk_count, 1)
        self.assertEqual(outlined.outline, ("Root", "Also root"))
        self.assertEqual(outlined.version_number, 1)

        for index in range(SERVING_DOCUMENT_LIST_LIMIT):
            await self._target(
                foundation,
                chunk_id=UUID(f"01900000-0000-7000-8000-0000000012{index:02d}"),
                vector=_axis_vector(index % 8),
            )
        truncated = await self.service.list_serving_documents(foundation.kb_id)
        self.assertIsNotNone(truncated)
        assert truncated is not None
        self.assertTrue(truncated.truncated)
        self.assertEqual(len(truncated.entries), SERVING_DOCUMENT_LIST_LIMIT)

    async def test_empty_result_keeps_revision_and_missing_scope_is_not_found(self) -> None:
        foundation = await self._foundation()

        empty = await self.service.retrieve(
            RetrievalRequest(foundation.kb_id, "no matches", top_k=10),
        )

        self.assertEqual(empty.index_revision_id, foundation.revision_id)
        self.assertEqual(empty.evidence, ())
        with self.assertRaises(ResourceNotFoundError):
            await self.service.retrieve(
                RetrievalRequest(uuid4(), "missing knowledge base"),
            )

        other = await self._foundation(
            workspace_id=OTHER_WORKSPACE,
            compatibility_fingerprint="sha256:other-workspace-space",
        )
        with self.assertRaises(ResourceNotFoundError):
            await self.service.retrieve(
                RetrievalRequest(other.kb_id, "cross workspace"),
            )

    async def test_only_available_ready_serving_active_content_is_returned(self) -> None:
        foundation = await self._foundation()
        valid = UUID("01900000-0000-7000-8000-000000001201")
        await self._target(foundation, chunk_id=valid, vector=_axis_vector(0))
        await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001202"),
            vector=_axis_vector(0),
            serving_status="candidate",
        )
        await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001203"),
            vector=_axis_vector(0),
            build_status="failed",
            serving_status="candidate",
        )
        await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001204"),
            vector=_axis_vector(0),
            serving_status="retired",
        )
        await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001205"),
            vector=_axis_vector(0),
            source_status="unavailable",
        )
        await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001206"),
            vector=_axis_vector(0),
            deleted=True,
        )
        retired_revision = await self._revision(
            foundation,
            status="retired",
        )
        await self._target(
            replace(foundation, revision_id=retired_revision),
            chunk_id=UUID("01900000-0000-7000-8000-000000001208"),
            vector=_axis_vector(0),
        )

        pack = await self.service.retrieve(
            RetrievalRequest(foundation.kb_id, "query", top_k=100),
        )

        self.assertEqual(
            tuple(item.index_chunk_id for item in pack.evidence),
            (valid,),
        )

    async def test_revision_activation_reads_are_complete_old_or_new_snapshots(self) -> None:
        foundation = await self._foundation()
        old_first = await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001301"),
            vector=_axis_vector(0),
        )
        old_second = await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001302"),
            vector=_axis_vector(0),
        )
        new_revision_id = await self._revision(foundation, status="ready")
        new_foundation = replace(foundation, revision_id=new_revision_id)
        new_first = await self._copy_target_to_revision(
            new_foundation,
            old_first,
            chunk_id=UUID("01900000-0000-7000-8000-000000001311"),
            vector=_axis_vector(0),
        )
        new_second = await self._copy_target_to_revision(
            new_foundation,
            old_second,
            chunk_id=UUID("01900000-0000-7000-8000-000000001312"),
            vector=_axis_vector(0),
        )
        old_snapshot = (
            foundation.revision_id,
            frozenset(
                (
                    (old_first.document_version_id, old_first.chunk_id),
                    (old_second.document_version_id, old_second.chunk_id),
                )
            ),
        )
        new_snapshot = (
            new_revision_id,
            frozenset(
                (
                    (new_first.document_version_id, new_first.chunk_id),
                    (new_second.document_version_id, new_second.chunk_id),
                )
            ),
        )

        writer = await asyncpg.connect(MIGRATION_DSN)
        transaction = writer.transaction()
        await transaction.start()
        try:
            await writer.execute(
                """
                UPDATE indexed_document_version
                   SET serving_status = 'retired'
                 WHERE index_revision_id = $1
                """,
                foundation.revision_id,
            )
            await writer.execute(
                """
                UPDATE indexed_document_version
                   SET serving_status = 'serving'
                 WHERE index_revision_id = $1
                """,
                new_revision_id,
            )
            await writer.execute(
                "UPDATE index_revision SET status = 'retired' WHERE id = $1",
                foundation.revision_id,
            )
            await writer.execute(
                "UPDATE index_revision SET status = 'active' WHERE id = $1",
                new_revision_id,
            )
            await writer.execute(
                """
                UPDATE knowledge_base
                   SET active_index_revision_id = $1
                 WHERE id = $2
                """,
                new_revision_id,
                foundation.kb_id,
            )
            observations = await self._race_reads_through_commit(
                foundation.kb_id,
                transaction,
            )
        finally:
            await writer.close()

        self.assertEqual(set(observations), {old_snapshot, new_snapshot})

    async def test_update_keeps_old_serving_until_atomic_promotion(self) -> None:
        foundation = await self._foundation()
        old = await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001401"),
            vector=_axis_vector(0),
        )
        new = await self._append_version_target(
            foundation,
            old,
            chunk_id=UUID("01900000-0000-7000-8000-000000001402"),
            vector=_axis_vector(0),
        )
        old_snapshot = (
            foundation.revision_id,
            frozenset(((old.document_version_id, old.chunk_id),)),
        )
        new_snapshot = (
            foundation.revision_id,
            frozenset(((new.document_version_id, new.chunk_id),)),
        )

        before_promotion = await self.service.retrieve(
            RetrievalRequest(foundation.kb_id, "during update", top_k=10),
        )
        self.assertEqual(
            (
                before_promotion.index_revision_id,
                frozenset(
                    (item.document_version_id, item.index_chunk_id)
                    for item in before_promotion.evidence
                ),
            ),
            old_snapshot,
        )

        writer = await asyncpg.connect(MIGRATION_DSN)
        transaction = writer.transaction()
        await transaction.start()
        try:
            await writer.execute(
                """
                UPDATE indexed_document_version
                   SET serving_status = 'retired'
                 WHERE id = $1
                """,
                old.indexed_document_version_id,
            )
            await writer.execute(
                """
                UPDATE indexed_document_version
                   SET serving_status = 'serving'
                 WHERE id = $1
                """,
                new.indexed_document_version_id,
            )
            observations = await self._race_reads_through_commit(
                foundation.kb_id,
                transaction,
            )
        finally:
            await writer.close()

        self.assertEqual(set(observations), {old_snapshot, new_snapshot})

    async def test_relation_hydration_keeps_old_serving_version_during_update(
        self,
    ) -> None:
        foundation = await self._foundation()
        old = await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001421"),
            vector=_axis_vector(0),
        )
        old_asset_id = await self._relation(
            foundation,
            old,
            visual_chunk_id=UUID("01900000-0000-7000-8000-000000001422"),
            asset_id=UUID("01900000-0000-7000-8000-000000001423"),
        )
        candidate = await self._append_version_target(
            foundation,
            old,
            chunk_id=UUID("01900000-0000-7000-8000-000000001424"),
            vector=_axis_vector(0),
        )
        candidate_asset_id = await self._relation(
            foundation,
            candidate,
            visual_chunk_id=UUID("01900000-0000-7000-8000-000000001425"),
            asset_id=UUID("01900000-0000-7000-8000-000000001426"),
        )

        during_update = await self.hydrator.hydrate(
            kb_id=foundation.kb_id,
            index_revision_id=foundation.revision_id,
            chunk_ids=(old.chunk_id, candidate.chunk_id),
            asset_ids=(old_asset_id, candidate_asset_id),
        )

        self.assertEqual(len(during_update), 1)
        self.assertEqual(
            during_update[0].indexed_document_version_id,
            old.indexed_document_version_id,
        )
        self.assertEqual(during_update[0].chunk_id, old.chunk_id)

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            await connection.execute(
                """
                UPDATE document_version
                   SET source_status = 'unavailable'
                 WHERE id = $1
                """,
                old.document_version_id,
            )
        finally:
            await connection.close()

        unavailable = await self.hydrator.hydrate(
            kb_id=foundation.kb_id,
            index_revision_id=foundation.revision_id,
            chunk_ids=(old.chunk_id,),
            asset_ids=(old_asset_id,),
        )
        self.assertEqual(unavailable, ())

    async def test_delete_commit_changes_visible_content_to_empty_atomically(self) -> None:
        foundation = await self._foundation()
        target = await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001501"),
            vector=_axis_vector(0),
        )
        visible_snapshot = (
            foundation.revision_id,
            frozenset(((target.document_version_id, target.chunk_id),)),
        )
        empty_snapshot = (foundation.revision_id, frozenset())

        writer = await asyncpg.connect(MIGRATION_DSN)
        transaction = writer.transaction()
        await transaction.start()
        try:
            await writer.execute(
                "UPDATE document SET deleted_at = now() WHERE id = $1",
                target.document_id,
            )
            await writer.execute(
                """
                UPDATE document_version
                   SET source_status = 'deleted'
                 WHERE document_id = $1
                """,
                target.document_id,
            )
            await writer.execute(
                """
                UPDATE indexed_document_version
                   SET serving_status = 'retired'
                 WHERE document_id = $1
                """,
                target.document_id,
            )
            observations = await self._race_reads_through_commit(
                foundation.kb_id,
                transaction,
            )
        finally:
            await writer.close()

        self.assertEqual(set(observations), {visible_snapshot, empty_snapshot})

    async def test_active_embedding_space_mismatch_fails_closed(self) -> None:
        foundation = await self._foundation(
            compatibility_fingerprint="sha256:different-space"
        )

        with self.assertRaises(RetrievalExecutionError) as failure:
            await self.service.retrieve(
                RetrievalRequest(foundation.kb_id, "query"),
            )

        self.assertEqual(failure.exception.code, ErrorCode.EMBEDDING_SPACE_MISMATCH)

    async def _foundation(
        self,
        *,
        workspace_id: UUID = WORKSPACE,
        compatibility_fingerprint: str = EXPECTED_FINGERPRINT,
    ) -> "_Foundation":
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            async with connection.transaction():
                await connection.execute(
                    "INSERT INTO workspace (id, name) VALUES ($1, $2)",
                    workspace_id,
                    f"workspace-{workspace_id}",
                )
                embedding_space_id = await connection.fetchval(
                    """
                    INSERT INTO embedding_space (
                        workspace_id, provider_identity, endpoint_identity,
                        requested_model, resolved_model, model_version, dimension,
                        distance_metric, vector_data_type, normalization,
                        configuration_fingerprint, compatibility_fingerprint
                    ) VALUES (
                        $1, 'test', 'test-endpoint', 'test-embedding',
                        'test-embedding', 'v1', 1024, 'cosine', 'float32', 'l2',
                        'sha256:configuration', $2
                    ) RETURNING id
                    """,
                    workspace_id,
                    compatibility_fingerprint,
                )
                kb_id = await connection.fetchval(
                    """
                    INSERT INTO knowledge_base (workspace_id, name)
                    VALUES ($1, $2) RETURNING id
                    """,
                    workspace_id,
                    f"kb-{uuid4()}",
                )
                revision_id = await connection.fetchval(
                    """
                    INSERT INTO index_revision (
                        workspace_id, kb_id, embedding_space_id, status,
                        source_snapshot_seq, parser_config, chunking_config
                    ) VALUES ($1, $2, $3, 'active', 0, '{}', '{}')
                    RETURNING id
                    """,
                    workspace_id,
                    kb_id,
                    embedding_space_id,
                )
                await connection.execute(
                    """
                    INSERT INTO index_revision_embedding_space (
                        workspace_id, index_revision_id, role,
                        embedding_space_id, required, retrieval_weight_micros
                    ) VALUES ($1, $2, 'text_retrieval', $3, true, 1000000)
                    """,
                    workspace_id,
                    revision_id,
                    embedding_space_id,
                )
                await connection.execute(
                    """
                    UPDATE knowledge_base
                       SET active_index_revision_id = $1, provisioned_at = now()
                     WHERE id = $2
                    """,
                    revision_id,
                    kb_id,
                )
        finally:
            await connection.close()
        return _Foundation(
            workspace_id,
            kb_id,
            revision_id,
            embedding_space_id,
        )

    async def _revision(
        self,
        foundation: "_Foundation",
        *,
        status: str,
    ) -> UUID:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            async with connection.transaction():
                revision_id = await connection.fetchval(
                    """
                    INSERT INTO index_revision (
                        workspace_id, kb_id, embedding_space_id, status,
                        source_snapshot_seq, parser_config, chunking_config
                    ) VALUES ($1, $2, $3, $4, 0, '{}', '{}') RETURNING id
                    """,
                    foundation.workspace_id,
                    foundation.kb_id,
                    foundation.embedding_space_id,
                    status,
                )
                await connection.execute(
                    """
                    INSERT INTO index_revision_embedding_space (
                        workspace_id, index_revision_id, role,
                        embedding_space_id, required, retrieval_weight_micros
                    ) VALUES ($1, $2, 'text_retrieval', $3, true, 1000000)
                    """,
                    foundation.workspace_id,
                    revision_id,
                    foundation.embedding_space_id,
                )
            return revision_id
        finally:
            await connection.close()

    async def _target(
        self,
        foundation: "_Foundation",
        *,
        chunk_id: UUID,
        vector: tuple[float, ...],
        build_status: str = "ready",
        serving_status: str = "serving",
        source_status: str = "available",
        deleted: bool = False,
    ) -> "_Target":
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            async with connection.transaction():
                document_id = await connection.fetchval(
                    """
                    INSERT INTO document (
                        workspace_id, kb_id, display_name, deleted_at
                    ) VALUES ($1, $2, $3, CASE WHEN $4 THEN now() ELSE NULL END)
                    RETURNING id
                    """,
                    foundation.workspace_id,
                    foundation.kb_id,
                    f"document-{chunk_id}",
                    deleted,
                )
                version_id = await self._document_version(
                    connection,
                    foundation,
                    document_id,
                    version_number=1,
                    source_status=source_status,
                )
                await connection.execute(
                    "UPDATE document SET current_version_id = $1 WHERE id = $2",
                    version_id,
                    document_id,
                )
                target = await self._indexed_target(
                    connection,
                    foundation,
                    document_id=document_id,
                    document_version_id=version_id,
                    chunk_id=chunk_id,
                    vector=vector,
                    source_change_seq=1,
                    build_status=build_status,
                    serving_status=serving_status,
                )
        finally:
            await connection.close()
        return target

    async def _lexical_target(
        self,
        foundation: "_Foundation",
        target: "_Target",
        *,
        create_manifest: bool = True,
    ) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            async with connection.transaction():
                content = await connection.fetchval(
                    "SELECT content FROM index_chunk WHERE id = $1",
                    target.chunk_id,
                )
                analyzed = analyze_document(content)
                assert analyzed is not None
                await connection.execute(
                    """
                    INSERT INTO index_chunk_lexical (
                        index_chunk_id, analyzer_version, workspace_id, kb_id,
                        indexed_document_version_id, lexical_text,
                        lexical_text_hash
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    target.chunk_id,
                    LEXICAL_ANALYZER_VERSION,
                    foundation.workspace_id,
                    foundation.kb_id,
                    target.indexed_document_version_id,
                    analyzed.lexical_text,
                    analyzed.lexical_text_hash,
                )
                if create_manifest:
                    await connection.execute(
                        """
                        INSERT INTO index_lexical_manifest (
                            indexed_document_version_id, analyzer_version,
                            workspace_id, kb_id, lexical_chunk_count,
                            lexical_manifest_hash
                        ) VALUES ($1, $2, $3, $4, 1, $5)
                        """,
                        target.indexed_document_version_id,
                        LEXICAL_ANALYZER_VERSION,
                        foundation.workspace_id,
                        foundation.kb_id,
                        lexical_manifest_hash(
                            LEXICAL_ANALYZER_VERSION,
                            ((target.chunk_id, analyzed.lexical_text_hash),),
                        ),
                    )
        finally:
            await connection.close()

    async def _append_version_target(
        self,
        foundation: "_Foundation",
        previous: "_Target",
        *,
        chunk_id: UUID,
        vector: tuple[float, ...],
        serving_status: str = "candidate",
    ) -> "_Target":
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            async with connection.transaction():
                version_id = await self._document_version(
                    connection,
                    foundation,
                    previous.document_id,
                    version_number=2,
                    source_status="available",
                )
                await connection.execute(
                    "UPDATE document SET current_version_id = $1 WHERE id = $2",
                    version_id,
                    previous.document_id,
                )
                return await self._indexed_target(
                    connection,
                    foundation,
                    document_id=previous.document_id,
                    document_version_id=version_id,
                    chunk_id=chunk_id,
                    vector=vector,
                    source_change_seq=2,
                    build_status="ready",
                    serving_status=serving_status,
                )
        finally:
            await connection.close()

    async def _copy_target_to_revision(
        self,
        foundation: "_Foundation",
        source: "_Target",
        *,
        chunk_id: UUID,
        vector: tuple[float, ...],
        serving_status: str = "candidate",
    ) -> "_Target":
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            async with connection.transaction():
                return await self._indexed_target(
                    connection,
                    foundation,
                    document_id=source.document_id,
                    document_version_id=source.document_version_id,
                    chunk_id=chunk_id,
                    vector=vector,
                    source_change_seq=1,
                    build_status="ready",
                    serving_status=serving_status,
                )
        finally:
            await connection.close()

    async def _indexed_target(
        self,
        connection: asyncpg.Connection,
        foundation: "_Foundation",
        *,
        document_id: UUID,
        document_version_id: UUID,
        chunk_id: UUID,
        vector: tuple[float, ...],
        source_change_seq: int,
        build_status: str,
        serving_status: str,
    ) -> "_Target":
        indexed_id = await connection.fetchval(
            """
            INSERT INTO indexed_document_version (
                workspace_id, kb_id, document_id, document_version_id,
                index_revision_id, source_change_seq, build_status,
                serving_status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING id
            """,
            foundation.workspace_id,
            foundation.kb_id,
            document_id,
            document_version_id,
            foundation.revision_id,
            source_change_seq,
            build_status,
            serving_status,
        )
        await connection.execute(
            """
            INSERT INTO index_chunk (
                id, workspace_id, kb_id, indexed_document_version_id,
                ordinal, content, content_hash, token_count,
                source_location, hierarchy, source_metadata, unit_key, modality
            ) VALUES (
                $1, $2, $3, $4, 0, $5, $6, 2,
                '{"line_start": 1, "line_end": 1}', '{}',
                '{"filename": "fixture.txt"}', $7, 'text'
            )
            """,
            chunk_id,
            foundation.workspace_id,
            foundation.kb_id,
            indexed_id,
            f"evidence-{chunk_id}",
            chunk_id.hex.ljust(64, "0")[:64],
            f"retrieval:{chunk_id}",
        )
        await connection.execute(
            """
            INSERT INTO vector_record (
                workspace_id, kb_id, index_chunk_id,
                embedding_space_id, embedding_dimension,
                representation_kind, embedding
            ) VALUES ($1, $2, $3, $4, 1024, 'text', $5::vector)
            """,
            foundation.workspace_id,
            foundation.kb_id,
            chunk_id,
            foundation.embedding_space_id,
            _vector_literal(vector),
        )
        return _Target(document_id, document_version_id, indexed_id, chunk_id)

    async def _relation(
        self,
        foundation: "_Foundation",
        target: "_Target",
        *,
        visual_chunk_id: UUID,
        asset_id: UUID,
    ) -> UUID:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO index_asset (
                        id, workspace_id, kb_id, document_id,
                        document_version_id, indexed_document_version_id,
                        asset_key, kind, storage_uri, media_type,
                        checksum_sha256, width, height, source_location
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, 'image',
                        $8, 'image/png', $9, 64, 64, '{}'::jsonb
                    )
                    """,
                    asset_id,
                    foundation.workspace_id,
                    foundation.kb_id,
                    target.document_id,
                    target.document_version_id,
                    target.indexed_document_version_id,
                    f"asset:{asset_id}",
                    f"local://asset/{asset_id}",
                    asset_id.hex.ljust(64, "0")[:64],
                )
                await connection.execute(
                    """
                    INSERT INTO index_chunk (
                        id, workspace_id, kb_id,
                        indexed_document_version_id, ordinal, content,
                        content_hash, token_count, source_location, hierarchy,
                        source_metadata, unit_key, modality, index_asset_id,
                        evidence_group_key
                    ) VALUES (
                        $1, $2, $3, $4, 1, '', $5, 0,
                        '{"page_number": 1}', '{}', '{}', $6, 'image', $7, $8
                    )
                    """,
                    visual_chunk_id,
                    foundation.workspace_id,
                    foundation.kb_id,
                    target.indexed_document_version_id,
                    visual_chunk_id.hex.ljust(64, "0")[:64],
                    f"visual:{visual_chunk_id}",
                    asset_id,
                    f"group:{target.indexed_document_version_id}",
                )
                await connection.execute(
                    """
                    INSERT INTO index_chunk_asset_relation (
                        workspace_id, kb_id, indexed_document_version_id,
                        chunk_id, visual_unit_id, asset_id, relation_type,
                        confidence_micros, ordinal, provenance,
                        evidence_group_key
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, 'caption_of',
                        1000000, 0, 'author_caption_v2', $7
                    )
                    """,
                    foundation.workspace_id,
                    foundation.kb_id,
                    target.indexed_document_version_id,
                    target.chunk_id,
                    visual_chunk_id,
                    asset_id,
                    f"group:{target.indexed_document_version_id}",
                )
        finally:
            await connection.close()
        return asset_id

    async def _race_reads_through_commit(
        self,
        knowledge_base_id: UUID,
        transaction: asyncpg.Transaction,
    ) -> list[tuple[UUID, frozenset[tuple[UUID, UUID]]]]:
        observations: list[tuple[UUID, frozenset[tuple[UUID, UUID]]]] = []
        first_read = asyncio.Event()

        async def read_repeatedly() -> None:
            for _ in range(8):
                pack = await self.service.retrieve(
                    RetrievalRequest(knowledge_base_id, "race", top_k=100),
                )
                observations.append(
                    (
                        pack.index_revision_id,
                        frozenset(
                            (item.document_version_id, item.index_chunk_id)
                            for item in pack.evidence
                        ),
                    )
                )
                first_read.set()
                await asyncio.sleep(0)

        readers = [asyncio.create_task(read_repeatedly()) for _ in range(3)]
        await first_read.wait()
        await transaction.commit()
        await asyncio.gather(*readers)
        final = await self.service.retrieve(
            RetrievalRequest(knowledge_base_id, "after commit", top_k=100),
        )
        observations.append(
            (
                final.index_revision_id,
                frozenset(
                    (item.document_version_id, item.index_chunk_id)
                    for item in final.evidence
                ),
            )
        )
        return observations

    async def _set_chunk_hierarchy(
        self,
        chunk_id: UUID,
        hierarchy: dict,
    ) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            await connection.execute(
                "UPDATE index_chunk SET hierarchy = $1::jsonb WHERE id = $2",
                json.dumps(hierarchy),
                chunk_id,
            )
        finally:
            await connection.close()

    async def _document_version(
        self,
        connection: asyncpg.Connection,
        foundation: "_Foundation",
        document_id: UUID,
        *,
        version_number: int,
        source_status: str,
    ) -> UUID:
        return await connection.fetchval(
            """
            INSERT INTO document_version (
                workspace_id, kb_id, document_id, version_number, source_status,
                checksum_sha256, storage_uri, original_filename, media_type,
                size_bytes
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, 'fixture.txt', 'text/plain', 1
            ) RETURNING id
            """,
            foundation.workspace_id,
            foundation.kb_id,
            document_id,
            version_number,
            source_status,
            f"{version_number:064d}",
            f"file:///fixture/{uuid4()}.txt",
        )


@dataclass(frozen=True, slots=True)
class _Foundation:
    workspace_id: UUID
    kb_id: UUID
    revision_id: UUID
    embedding_space_id: UUID


@dataclass(frozen=True, slots=True)
class _Target:
    document_id: UUID
    document_version_id: UUID
    indexed_document_version_id: UUID
    chunk_id: UUID


class _Provider:
    max_batch_size = 10

    def __init__(
        self,
        embedding_space: EmbeddingSpaceDefinition,
        vector: tuple[float, ...],
    ) -> None:
        self.embedding_space = embedding_space
        self.vector = vector

    async def embed_query(self, text: str) -> tuple[float, ...]:
        if not text:
            raise AssertionError("retrieval embeds one non-empty query")
        return self.vector


def _embedding_space() -> EmbeddingSpaceDefinition:
    return EmbeddingSpaceDefinition(
        provider_identity="test",
        endpoint_identity="test-endpoint",
        requested_model="test-embedding",
        resolved_model="test-embedding",
        model_version="v1",
        deployment_revision=None,
        dimension=1024,
        distance_metric="cosine",
        vector_data_type="float32",
        normalization="l2",
        configuration_fingerprint="sha256:configuration",
        tokenizer_fingerprint=None,
        compatibility_fingerprint=EXPECTED_FINGERPRINT,
    )


def _axis_vector(index: int) -> tuple[float, ...]:
    values = [0.0] * 1024
    values[index] = 1.0
    return tuple(values)


def _vector_literal(vector: tuple[float, ...]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"
