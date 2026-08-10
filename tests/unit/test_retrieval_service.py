from __future__ import annotations

import asyncio
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

from sqlalchemy.dialects import postgresql

from rag_kb.adapters.lexical_store.postgres import PgLexicalStore
from rag_kb.adapters.vector_store.pgvector import PgVectorStore
from rag_kb.auth import (
    AccessDeniedError,
    AuthContext,
    MetadataFilter,
    SingleWorkspaceAccessPolicy,
)
from rag_kb.domain import (
    AdjacentChunkHit,
    AdjacentChunkResult,
    EmbeddingSpaceDefinition,
    EmbeddingBatch,
    ErrorCode,
    Evidence,
    EvidenceScoreKind,
    IndexChunkAssetRelationSnapshot,
    LexicalSearchResult,
    ModelRerankScore,
    RerankMode,
    RetrievalExecutionError,
    RetrievalQueryPlan,
    RetrievalRequest,
    RetrievalStrategy,
    VectorSearchHit,
    VectorSearchResult,
)
from rag_kb.repositories.sqlalchemy_indexing import SqlAlchemyIndexingRepository
from rag_kb.retrieval.service import RetrievalService
from rag_kb.retrieval.reranker import rerank_hits


WORKSPACE = UUID("01900000-0000-7000-8000-000000000801")
OTHER_WORKSPACE = UUID("01900000-0000-7000-8000-000000000802")
KB_ID = UUID("01900000-0000-7000-8000-000000000803")
REVISION_ID = UUID("01900000-0000-7000-8000-000000000804")
CHUNK_1 = UUID("01900000-0000-7000-8000-000000000811")
CHUNK_2 = UUID("01900000-0000-7000-8000-000000000812")
VISUAL_1 = UUID("01900000-0000-7000-8000-000000000815")
ASSET_1 = UUID("01900000-0000-7000-8000-000000000816")


class RetrievalContractTests(unittest.TestCase):
    def test_request_normalizes_query_and_rejects_invalid_top_k(self) -> None:
        request = RetrievalRequest(KB_ID, "  查询 ABC-42  ", top_k=5)
        self.assertEqual(request.query, "查询 ABC-42")
        with self.assertRaises(ValueError):
            RetrievalRequest(KB_ID, "   ")
        with self.assertRaises(ValueError):
            RetrievalRequest(KB_ID, "query", top_k=101)

    def test_pgvector_statement_is_one_exact_filtered_snapshot_shape(self) -> None:
        statement = PgVectorStore._statement()  # noqa: SLF001 - SQL contract
        sql = str(statement.compile(dialect=postgresql.dialect()))

        self.assertIn("LEFT OUTER JOIN LATERAL", sql)
        self.assertIn("vector_record.embedding <=>", sql)
        self.assertIn("knowledge_base.active_index_revision_id", sql)
        self.assertIn("indexed_document_version.build_status", sql)
        self.assertIn("indexed_document_version.serving_status", sql)
        self.assertNotIn("document.current_version_id", sql)
        self.assertIn("document.deleted_at IS NULL", sql)
        self.assertIn("knowledge_base.deleted_at IS NULL", sql)
        self.assertIn("index_chunk.excluded_at IS NULL", sql)
        self.assertIn("document_version.source_status", sql)
        self.assertIn("vector_record.embedding_space_id", sql)
        self.assertIn("vector_record.embedding_dimension", sql)
        self.assertIn("ORDER BY cosine_distance ASC, index_chunk.id ASC", sql)
        self.assertIn("LIMIT %(top_k)s", sql)
        self.assertNotIn("hnsw", sql.lower())

        cross_modal = str(
            PgVectorStore._statement(768).compile(dialect=postgresql.dialect())
        )
        self.assertIn("vector_record.embedding <=>", cross_modal)
        self.assertIn("vector_record.embedding_space_id", cross_modal)
        self.assertIn("vector_record.embedding_dimension", cross_modal)

    def test_adjacency_statement_is_one_bounded_frozen_scope_read(self) -> None:
        statement = PgVectorStore._adjacent_statement(2)  # noqa: SLF001
        sql = str(statement.compile(dialect=postgresql.dialect()))

        self.assertIn("VALUES (", sql)
        self.assertIn("valid_adjacency_anchors", sql)
        self.assertIn("LEFT OUTER JOIN LATERAL", sql)
        self.assertIn("active_index_revision_id", sql)
        self.assertIn("adjacency_anchor_chunk.id =", sql)
        self.assertIn("adjacency_anchor_chunk.ordinal =", sql)
        self.assertIn("adjacency_neighbor_chunk.ordinal =", sql)
        self.assertIn("adjacency_neighbor_chunk.excluded_at IS NULL", sql)
        self.assertIn("adjacency_document.deleted_at IS NULL", sql)
        self.assertIn("adjacency_document_version.source_status", sql)
        self.assertIn("build_status", sql)
        self.assertIn("serving_status", sql)
        self.assertNotIn("vector_record", sql)

    def test_lexical_statement_uses_gin_predicate_and_real_cosine(self) -> None:
        statement = PgLexicalStore._statement()  # noqa: SLF001 - SQL contract
        sql = str(statement.compile(dialect=postgresql.dialect()))

        self.assertIn("AS MATERIALIZED", sql)
        self.assertIn("lexical_tsv @@ to_tsquery('simple'", sql)
        self.assertIn("ts_rank_cd(", sql)
        self.assertIn("vector_record", sql)
        self.assertIn("vector.embedding <=>", sql)
        self.assertIn("vector.embedding_dimension =", sql)
        self.assertIn("target.build_status = 'ready'", sql)
        self.assertIn("target.serving_status = 'serving'", sql)
        self.assertIn("doc.deleted_at IS NULL", sql)
        self.assertIn("admitted_chunk.excluded_at IS NULL", sql)
        self.assertIn("chunk.excluded_at IS NULL", sql)
        self.assertIn("version.source_status = 'available'", sql)
        self.assertNotIn("hnsw", sql.lower())

    def test_pgvector_adapter_rejects_wrong_dimension(self) -> None:
        definition = replace(_embedding_space(), dimension=1024)
        store = PgVectorStore(None, definition)  # type: ignore[arg-type]
        exact = _plan()

        with self.assertRaises(RetrievalExecutionError) as dimension:
            store._require_exact_plan(exact, (1.0, 0.0))  # noqa: SLF001
        self.assertEqual(dimension.exception.code, ErrorCode.EMBEDDING_RESPONSE_INVALID)

    def test_query_plan_contains_only_executable_controls(self) -> None:
        plan = _plan()
        relaxed = (
            {"distance_metric": "inner_product"},
            {"candidate_count": 50},
        )

        for changes in relaxed:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(plan, **changes)

        reranked = replace(plan, rerank_mode=RerankMode.CLASSIC)
        self.assertEqual(reranked.candidate_count, 20)


class RelationHydrationRepositoryQueryTests(unittest.IsolatedAsyncioTestCase):
    async def test_relation_hydration_uses_serving_source_predicate(self) -> None:
        session = MagicMock()
        session.execute = AsyncMock(
            return_value=MagicMock(all=MagicMock(return_value=[]))
        )
        repository = SqlAlchemyIndexingRepository(
            session,
            WORKSPACE,
            lambda: None,
        )

        relations = await repository.list_relations(
            kb_id=KB_ID,
            index_revision_id=REVISION_ID,
            chunk_ids=(CHUNK_1,),
        )

        self.assertEqual(relations, ())
        statement = session.execute.await_args.args[0]
        sql = str(statement.compile(dialect=postgresql.dialect()))
        self.assertNotIn("document.current_version_id", sql)
        self.assertIn("JOIN document_version", sql)
        self.assertIn("document_version.source_status", sql)
        self.assertIn("indexed_document_version.build_status", sql)
        self.assertIn("indexed_document_version.serving_status", sql)
        self.assertIn("knowledge_base.active_index_revision_id", sql)
        self.assertIn("document.deleted_at IS NULL", sql)
        self.assertIn("knowledge_base.deleted_at IS NULL", sql)
        self.assertGreaterEqual(sql.count("excluded_at IS NULL"), 2)


class RetrievalServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_capability_snapshot_is_pure_and_deterministic(self) -> None:
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            _Provider(),
            _Store(VectorSearchResult(REVISION_ID)),
            hybrid_enabled=True,
        )
        snapshot = service.capabilities_snapshot()
        self.assertEqual(snapshot.default_mode, "vector")
        self.assertEqual(
            [(item.mode, item.strategy, item.profile_version, item.enabled) for item in snapshot.modes],
            [
                ("vector", "exact_vector", "exact_vector_v2", True),
                ("hybrid", "hybrid", "hybrid_fts_rrf_v2", False),
            ],
        )
        self.assertFalse(service.hybrid_request_enabled())

        lexical = _LexicalStore(
            LexicalSearchResult(
                REVISION_ID,
                analyzer_version="lexical_simple_cjk_bigram_v1",
                manifest_target_count=1,
                hits=(),
            )
        )
        enabled = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            _Provider(),
            _Store(VectorSearchResult(REVISION_ID)),
            lexical_store=lexical,
            hybrid_enabled=True,
        )
        self.assertTrue(enabled.hybrid_request_enabled())
        self.assertTrue(enabled.capabilities_snapshot().modes[1].enabled)

    def test_reranker_fuses_query_terms_and_removes_duplicate_chunks(self) -> None:
        generic = replace(
            _hit(CHUNK_1, distance=0.05, ordinal=1),
            text="general handbook introduction",
        )
        matching = replace(
            _hit(CHUNK_2, distance=0.20, ordinal=2),
            text="policy deadline is Friday",
        )
        duplicate = replace(
            _hit(UUID("01900000-0000-7000-8000-000000000813"), distance=0.10, ordinal=3),
            text="policy deadline is Friday with more detail",
        )
        distinct = replace(
            _hit(UUID("01900000-0000-7000-8000-000000000814"), distance=0.35, ordinal=4),
            text="policy escalation contact and owner",
        )

        ranked = rerank_hits(
            "policy deadline",
            (generic, matching, duplicate, distinct),
            top_k=2,
        )

        self.assertEqual(ranked[0].hit.index_chunk_id, CHUNK_2)
        self.assertNotEqual(ranked[1].hit.index_chunk_id, duplicate.index_chunk_id)
        self.assertGreater(ranked[0].lexical_coverage, 0.9)

    async def test_adjacent_evidence_has_independent_score_and_no_embedding(
        self,
    ) -> None:
        anchor = _evidence(CHUNK_1, ordinal=4)
        store = _Store(
            None,
            adjacent_result=AdjacentChunkResult(
                resolved_active_revision_id=REVISION_ID,
                validated_anchor_count=1,
                hits=(
                    AdjacentChunkHit(
                        workspace_id=WORKSPACE,
                        knowledge_base_id=KB_ID,
                        index_revision_id=REVISION_ID,
                        index_chunk_id=CHUNK_2,
                        indexed_document_version_id=(
                            anchor.indexed_document_version_id
                        ),
                        document_id=anchor.document_id,
                        document_version_id=anchor.document_version_id,
                        ordinal=5,
                        text="continued definition",
                        source_location={"line_start": 5},
                        hierarchy={"section": "test"},
                        source_metadata={},
                        anchor_index_chunk_id=CHUNK_1,
                        anchor_rank=1,
                        offset=1,
                        build_status="ready",
                        serving_status="serving",
                        is_current_serving_version=True,
                        modality="table",
                    ),
                ),
            ),
        )
        provider = _Provider()
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            provider,
            store,
        )

        evidence = await service.retrieve_adjacent_evidence(
            _context(),
            knowledge_base_id=KB_ID,
            index_revision_id=REVISION_ID,
            anchors=(anchor,),
        )

        self.assertEqual(provider.queries, [])
        self.assertEqual(len(store.adjacent_queries), 1)
        self.assertEqual(len(evidence), 1)
        self.assertIs(evidence[0].score_kind, EvidenceScoreKind.ADJACENCY)
        self.assertEqual(evidence[0].score, 0.0)
        self.assertIsNone(evidence[0].vector_similarity)
        self.assertEqual(evidence[0].adjacency_anchor_index_chunk_id, CHUNK_1)
        self.assertEqual(evidence[0].adjacency_offset, 1)

    async def test_adjacent_evidence_rejects_partial_anchor_validation(self) -> None:
        store = _Store(
            None,
            adjacent_result=AdjacentChunkResult(
                resolved_active_revision_id=REVISION_ID,
                validated_anchor_count=0,
            ),
        )
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            _Provider(),
            store,
        )

        with self.assertRaises(RetrievalExecutionError) as raised:
            await service.retrieve_adjacent_evidence(
                _context(),
                knowledge_base_id=KB_ID,
                index_revision_id=REVISION_ID,
                anchors=(_evidence(CHUNK_1, ordinal=4),),
            )

        self.assertEqual(raised.exception.code, ErrorCode.INTERNAL_SERVER_ERROR)
        self.assertEqual(
            raised.exception.diagnostic,
            {"check": "adjacency_anchor_scope"},
        )

    async def test_builds_mandatory_plan_and_returns_deterministic_evidence(self) -> None:
        provider = _Provider()
        store = _Store(
            VectorSearchResult(
                REVISION_ID,
                (
                    _hit(CHUNK_2, distance=0.2, ordinal=2),
                    _hit(CHUNK_1, distance=0.2, ordinal=1),
                ),
            )
        )
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE), provider, store
        )

        pack = await service.retrieve(
            _context(),
            RetrievalRequest(KB_ID, " query ", top_k=5, include_debug=True),
        )

        self.assertEqual(provider.queries, ["query"])
        self.assertEqual(store.embeddings, [(0.6, 0.8)])
        plan = store.plans[0]
        self.assertEqual((plan.workspace_id, plan.knowledge_base_id), (WORKSPACE, KB_ID))
        self.assertEqual((plan.strategy.value, plan.distance_metric), ("exact_vector", "cosine"))
        self.assertIsNone(plan.candidate_count)
        self.assertEqual([item.index_chunk_id for item in pack.evidence], [CHUNK_1, CHUNK_2])
        self.assertEqual([item.rank for item in pack.evidence], [1, 2])
        self.assertAlmostEqual(pack.evidence[0].score, 0.8)
        self.assertIsNotNone(pack.debug)
        assert pack.debug is not None
        self.assertEqual(pack.debug.resolved_active_revision_id, REVISION_ID)
        self.assertEqual(pack.debug.result_count, 2)

    async def test_rerank_retrieval_uses_service_weights_and_returns_hybrid_scores(
        self,
    ) -> None:
        generic = replace(
            _hit(CHUNK_1, distance=0.05, ordinal=1),
            text="general handbook introduction",
        )
        matching = replace(
            _hit(CHUNK_2, distance=0.20, ordinal=2),
            text="policy deadline is Friday",
        )
        store = _Store(VectorSearchResult(REVISION_ID, (generic, matching)))
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            _Provider(),
            store,
        )

        pack = await service.retrieve(
            _context(),
            RetrievalRequest(
                KB_ID,
                "policy deadline",
                top_k=2,
                rerank_mode=RerankMode.CLASSIC,
                include_debug=True,
            ),
        )

        self.assertTrue(store.plans[0].rerank)
        self.assertEqual(store.plans[0].candidate_count, 8)
        self.assertEqual(len(pack.evidence), 2)
        self.assertEqual(pack.evidence[0].index_chunk_id, CHUNK_2)
        self.assertEqual(
            pack.evidence[0].score_kind,
            EvidenceScoreKind.HYBRID_RERANK,
        )
        self.assertGreater(pack.evidence[0].lexical_score or 0.0, 0.0)

    async def test_local_model_reorders_classic_candidates_and_keeps_base_scores(
        self,
    ) -> None:
        first = replace(
            _hit(CHUNK_1, distance=0.05, ordinal=1),
            text="general handbook introduction",
        )
        second = replace(
            _hit(CHUNK_2, distance=0.20, ordinal=2),
            text="specific policy deadline",
        )
        store = _Store(VectorSearchResult(REVISION_ID, (first, second)))
        reranker = _LocalReranker(
            {
                CHUNK_1: (0.9, 2.0, 2, 1),
                CHUNK_2: (0.2, -1.0, 3, 0),
            }
        )
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            _Provider(),
            store,
            text_reranker=reranker,
        )

        pack = await service.retrieve(
            _context(),
            RetrievalRequest(
                KB_ID,
                "policy deadline",
                top_k=2,
                rerank_mode=RerankMode.LOCAL_MINILM_V1,
                include_debug=True,
            ),
        )

        self.assertEqual(
            [item.index_chunk_id for item in pack.evidence],
            [CHUNK_1, CHUNK_2],
        )
        self.assertEqual(reranker.queries, ["policy deadline"])
        self.assertEqual(
            [item.index_chunk_id for item in reranker.documents[0]],
            [CHUNK_2, CHUNK_1],
        )
        self.assertEqual(pack.evidence[0].model_rerank_score, 0.9)
        self.assertEqual(pack.evidence[0].model_rerank_rank, 1)
        self.assertEqual(pack.evidence[0].model_rerank_window_count, 2)
        self.assertEqual(pack.evidence[0].model_rerank_winning_window_index, 1)
        self.assertIs(pack.evidence[0].score_kind, EvidenceScoreKind.HYBRID_RERANK)
        assert pack.debug is not None
        self.assertEqual(pack.debug.model_rerank_candidate_count, 2)
        self.assertEqual(pack.debug.model_rerank_window_count, 5)

    async def test_selected_local_model_fails_explicitly_when_not_configured(
        self,
    ) -> None:
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            _Provider(),
            _Store(VectorSearchResult(REVISION_ID, (_hit(CHUNK_1),))),
        )

        with self.assertRaises(RetrievalExecutionError) as failure:
            await service.retrieve(
                _context(),
                RetrievalRequest(
                    KB_ID,
                    "query",
                    rerank_mode=RerankMode.LOCAL_MINILM_V1,
                ),
            )

        self.assertEqual(
            failure.exception.code,
            ErrorCode.LOCAL_RERANKER_UNAVAILABLE,
        )

    async def test_empty_and_underfilled_results_are_valid_without_filter_relaxation(self) -> None:
        store = _Store(VectorSearchResult(REVISION_ID))
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE), _Provider(), store
        )

        pack = await service.retrieve(
            _context(), RetrievalRequest(KB_ID, "no match", top_k=10)
        )

        self.assertEqual(pack.evidence, ())
        self.assertIsNone(pack.debug)
        self.assertEqual(store.plans[0].top_k, 10)

    async def test_unsupported_strategy_and_rerank_fail_before_external_io(self) -> None:
        for request in (
            RetrievalRequest(KB_ID, "query", strategy=RetrievalStrategy.HYBRID),
        ):
            provider = _Provider()
            store = _Store(VectorSearchResult(REVISION_ID))
            service = RetrievalService(
                SingleWorkspaceAccessPolicy(WORKSPACE), provider, store
            )
            with self.assertRaises(RetrievalExecutionError) as failure:
                await service.retrieve(_context(), request)
            self.assertEqual(failure.exception.code, ErrorCode.CAPABILITY_NOT_ENABLED)
            self.assertEqual(provider.queries, [])
            self.assertEqual(store.plans, [])

    async def test_hybrid_admits_lexical_candidate_outside_dense_lane(self) -> None:
        dense = replace(
            _hit(CHUNK_1, distance=0.05),
            text="general handbook introduction",
        )
        lexical = replace(
            _hit(CHUNK_2, distance=0.20, ordinal=2),
            text="policy deadline is Friday",
            lexical_rank=1,
            lexical_score=0.9,
        )
        vector_store = _HybridVectorStore(
            VectorSearchResult(REVISION_ID, (dense,))
        )
        lexical_store = _LexicalStore(
            LexicalSearchResult(
                REVISION_ID,
                analyzer_version="lexical_simple_cjk_bigram_v1",
                manifest_target_count=1,
                hits=(lexical,),
            )
        )
        provider = _Provider()
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            provider,
            vector_store,
            lexical_store=lexical_store,
            hybrid_enabled=True,
        )

        pack = await service.retrieve(
            _context(),
            RetrievalRequest(
                KB_ID,
                "policy deadline",
                top_k=2,
                strategy=RetrievalStrategy.HYBRID,
                rerank_mode=RerankMode.CLASSIC,
                include_debug=True,
            ),
        )

        self.assertEqual(
            {item.index_chunk_id for item in pack.evidence},
            {CHUNK_1, CHUNK_2},
        )
        lexical_evidence = next(
            item for item in pack.evidence if item.index_chunk_id == CHUNK_2
        )
        self.assertEqual(lexical_evidence.lexical_rank, 1)
        self.assertIsNone(lexical_evidence.text_space_rank)
        self.assertEqual(provider.queries, ["policy deadline"])
        self.assertEqual(lexical_store.queries, ["policy deadline"])
        self.assertEqual(lexical_store.embeddings, [(0.6, 0.8)])
        assert pack.debug is not None
        self.assertEqual(pack.debug.text_candidate_count, 1)
        self.assertEqual(pack.debug.lexical_candidate_count, 1)
        self.assertEqual(pack.debug.lexical_manifest_target_count, 1)

    async def test_hybrid_rejects_low_cosine_lexical_candidate(self) -> None:
        dense = replace(
            _hit(CHUNK_1, distance=0.10),
            text="policy deadline handbook",
        )
        lexical = replace(
            _hit(CHUNK_2, distance=0.90, ordinal=2),
            text="policy deadline is Friday",
            lexical_rank=1,
            lexical_score=0.9,
        )
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            _Provider(),
            _HybridVectorStore(VectorSearchResult(REVISION_ID, (dense,))),
            lexical_store=_LexicalStore(
                LexicalSearchResult(
                    REVISION_ID,
                    analyzer_version="lexical_simple_cjk_bigram_v1",
                    manifest_target_count=1,
                    hits=(lexical,),
                )
            ),
            hybrid_enabled=True,
        )

        pack = await service.retrieve(
            _context(),
            RetrievalRequest(
                KB_ID,
                "policy deadline",
                top_k=2,
                strategy=RetrievalStrategy.HYBRID,
                rerank_mode=RerankMode.CLASSIC,
            ),
        )

        self.assertEqual(
            [item.index_chunk_id for item in pack.evidence], [CHUNK_1]
        )

    async def test_hybrid_lane_failure_cancels_and_settles_dense_sibling(
        self,
    ) -> None:
        vector_store = _CancellableHybridVectorStore()
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            _Provider(),
            vector_store,
            lexical_store=_FailingLexicalStore(),
            hybrid_enabled=True,
        )

        with self.assertRaises(RetrievalExecutionError) as failure:
            await asyncio.wait_for(
                service.retrieve(
                    _context(),
                    RetrievalRequest(
                        KB_ID,
                        "query",
                        strategy=RetrievalStrategy.HYBRID,
                        rerank_mode=RerankMode.CLASSIC,
                    ),
                ),
                timeout=1.0,
            )

        self.assertEqual(
            failure.exception.code,
            ErrorCode.INDEX_REVISION_INCOMPATIBLE,
        )
        self.assertTrue(vector_store.cancelled)

    async def test_workspace_and_debug_authorization_fail_closed(self) -> None:
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            _Provider(),
            _Store(VectorSearchResult(REVISION_ID)),
        )
        with self.assertRaises(AccessDeniedError):
            await service.retrieve(
                _context(OTHER_WORKSPACE), RetrievalRequest(KB_ID, "query")
            )

        provider = _Provider()
        store = _Store(VectorSearchResult(REVISION_ID))
        denied = RetrievalService(_DebugDeniedPolicy(), provider, store)
        with self.assertRaises(AccessDeniedError):
            await denied.retrieve(
                _context(), RetrievalRequest(KB_ID, "query", include_debug=True)
            )
        self.assertEqual(provider.queries, [])
        self.assertEqual(store.plans, [])

    async def test_wrong_scope_status_revision_and_duplicates_are_rejected(self) -> None:
        invalid_results = (
            VectorSearchResult(
                REVISION_ID,
                (replace(_hit(CHUNK_1), workspace_id=OTHER_WORKSPACE),),
            ),
            VectorSearchResult(
                REVISION_ID,
                (replace(_hit(CHUNK_1), serving_status="candidate"),),
            ),
            VectorSearchResult(
                REVISION_ID,
                (replace(_hit(CHUNK_1), index_revision_id=CHUNK_2),),
            ),
            VectorSearchResult(
                REVISION_ID,
                (replace(_hit(CHUNK_1), is_current_serving_version=False),),
            ),
            VectorSearchResult(
                REVISION_ID,
                (_hit(CHUNK_1), _hit(CHUNK_1, ordinal=2)),
            ),
        )
        for result in invalid_results:
            with self.subTest(result=result):
                service = RetrievalService(
                    SingleWorkspaceAccessPolicy(WORKSPACE),
                    _Provider(),
                    _Store(result),
                )
                with self.assertRaises(RetrievalExecutionError) as failure:
                    await service.retrieve(_context(), RetrievalRequest(KB_ID, "query"))
                self.assertEqual(
                    failure.exception.code, ErrorCode.INTERNAL_SERVER_ERROR
                )

    async def test_invalid_query_embedding_dimension_is_rejected(self) -> None:
        provider = _Provider(vector=())
        store = _Store(VectorSearchResult(REVISION_ID))
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE), provider, store
        )

        with self.assertRaises(RetrievalExecutionError) as failure:
            await service.retrieve(_context(), RetrievalRequest(KB_ID, "query"))

        self.assertEqual(failure.exception.code, ErrorCode.EMBEDDING_RESPONSE_INVALID)
        self.assertEqual(store.plans, [])

    async def test_absolute_deadline_cancels_hanging_retrieval(self) -> None:
        provider = _HangingProvider()
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            provider,
            _Store(VectorSearchResult(REVISION_ID)),
            deadline_seconds=0.01,
        )

        with self.assertRaises(RetrievalExecutionError) as failure:
            await service.retrieve(_context(), RetrievalRequest(KB_ID, "query"))

        self.assertEqual(
            failure.exception.code,
            ErrorCode.RETRIEVAL_DEADLINE_EXCEEDED,
        )
        self.assertEqual(
            failure.exception.diagnostic,
            {"check": "absolute_deadline"},
        )
        self.assertTrue(provider.cancelled)

    async def test_retrieval_deadline_must_be_positive(self) -> None:
        for deadline_seconds in (0, float("inf"), float("nan")):
            with self.subTest(deadline_seconds=deadline_seconds), self.assertRaises(
                ValueError
            ):
                RetrievalService(
                    SingleWorkspaceAccessPolicy(WORKSPACE),
                    _Provider(),
                    _Store(VectorSearchResult(REVISION_ID)),
                    deadline_seconds=deadline_seconds,
                )

    async def test_composite_retrieval_runs_lanes_in_parallel_and_hydrates_strong_visual(
        self,
    ) -> None:
        gates = _ParallelGates()
        text_hit = _hit(CHUNK_1, distance=0.20)
        visual_hit = replace(
            _hit(VISUAL_1, distance=0.90),
            text="",
            modality="image",
            representation_kind="native_image",
            index_asset_id=ASSET_1,
        )
        store = _MultimodalStore(
            VectorSearchResult(REVISION_ID, (text_hit,)),
            VectorSearchResult(REVISION_ID, (visual_hit,), space_role="cross_modal_retrieval"),
            gates,
        )
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            _ParallelTextProvider(gates),
            store,
            multimodal_embedding_provider=_ParallelMultimodalProvider(gates),
            relation_hydrator=_Hydrator(
                (
                    _relation("inline_figure"),
                    replace(
                        _relation("explicit_figure_reference"),
                        id=UUID("01900000-0000-7000-8000-000000000818"),
                        ordinal=1,
                    ),
                )
            ),
        )

        pack = await asyncio.wait_for(
            service.retrieve(
                _context(),
                RetrievalRequest(KB_ID, "Figure 1", top_k=3, include_debug=True),
            ),
            timeout=1.0,
        )

        self.assertEqual(len(pack.evidence), 1)
        self.assertEqual(pack.evidence[0].index_chunk_id, CHUNK_1)
        self.assertEqual(pack.evidence[0].related_visuals[0].asset.id, ASSET_1)
        self.assertEqual(
            pack.evidence[0].related_visuals[0].relation_type,
            "explicit_figure_reference",
        )
        self.assertEqual(len(pack.evidence[0].related_visuals), 1)
        self.assertIsNone(pack.evidence[0].related_visuals[0].cross_modal_rank)
        assert pack.debug is not None
        self.assertEqual(pack.debug.hydrated_relation_count, 2)

    async def test_unified_exact_reuses_one_query_vector_for_both_lanes(self) -> None:
        gates = _ParallelGates()
        provider = _UnifiedProvider()
        store = _MultimodalStore(
            VectorSearchResult(REVISION_ID),
            VectorSearchResult(
                REVISION_ID,
                space_role="cross_modal_retrieval",
            ),
            gates,
        )
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            provider,
            store,
            multimodal_embedding_provider=provider,
        )

        await service.retrieve(
            _context(), RetrievalRequest(KB_ID, "unified query")
        )

        self.assertEqual(provider.queries, ["unified query"])
        self.assertEqual(store.text_embeddings, [(0.6, 0.8)])
        self.assertEqual(store.cross_embeddings, [(0.6, 0.8)])

    async def test_unified_hybrid_reuses_one_vector_across_three_lanes(self) -> None:
        gates = _ParallelGates()
        provider = _UnifiedProvider()
        store = _MultimodalStore(
            VectorSearchResult(REVISION_ID),
            VectorSearchResult(
                REVISION_ID,
                space_role="cross_modal_retrieval",
            ),
            gates,
        )
        lexical = _LexicalStore(
            LexicalSearchResult(
                REVISION_ID,
                analyzer_version="lexical_simple_cjk_bigram_v1",
                manifest_target_count=0,
            )
        )
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            provider,
            store,
            multimodal_embedding_provider=provider,
            lexical_store=lexical,
            hybrid_enabled=True,
        )

        await service.retrieve(
            _context(),
            RetrievalRequest(
                KB_ID,
                "unified hybrid",
                strategy=RetrievalStrategy.HYBRID,
                rerank_mode=RerankMode.CLASSIC,
            ),
        )

        self.assertEqual(provider.queries, ["unified hybrid"])
        self.assertEqual(store.text_embeddings, [(0.6, 0.8)])
        self.assertEqual(store.cross_embeddings, [(0.6, 0.8)])
        self.assertEqual(lexical.embeddings, [(0.6, 0.8)])

    async def test_native_image_hit_reverse_expands_to_parent_text(self) -> None:
        gates = _ParallelGates()
        visual_hit = replace(
            _hit(VISUAL_1, distance=0.10),
            text="",
            modality="image",
            representation_kind="native_image",
            index_asset_id=ASSET_1,
        )
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            _ParallelTextProvider(gates),
            _MultimodalStore(
                VectorSearchResult(REVISION_ID),
                VectorSearchResult(REVISION_ID, (visual_hit,), space_role="cross_modal_retrieval"),
                gates,
            ),
            multimodal_embedding_provider=_ParallelMultimodalProvider(gates),
            relation_hydrator=_Hydrator((_relation("inline_figure"),)),
        )

        pack = await service.retrieve(
            _context(), RetrievalRequest(KB_ID, "diagram", top_k=2)
        )

        self.assertEqual(pack.evidence[0].index_chunk_id, CHUNK_1)
        self.assertEqual(pack.evidence[0].text, "parent narrative")
        self.assertEqual(pack.evidence[0].modality, "text")
        self.assertEqual(pack.evidence[0].cross_modal_rank, 1)

    async def test_same_raw_table_group_is_scoped_to_each_index_target(self) -> None:
        gates = _ParallelGates()
        shared_group = "docling-table-group"
        first = replace(
            _hit(CHUNK_1, distance=0.10),
            text="Alpha budget table",
            modality="table",
            evidence_group_key=shared_group,
            representation_kind="table_text",
        )
        second_target = UUID("01900000-0000-7000-8000-000000000824")
        second = replace(
            _hit(CHUNK_2, distance=0.12, ordinal=1),
            indexed_document_version_id=second_target,
            document_id=UUID("01900000-0000-7000-8000-000000000825"),
            document_version_id=UUID("01900000-0000-7000-8000-000000000826"),
            text="Beta headcount table",
            modality="table",
            evidence_group_key=shared_group,
            representation_kind="table_text",
        )
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            _ParallelTextProvider(gates),
            _MultimodalStore(
                VectorSearchResult(REVISION_ID, (first, second)),
                VectorSearchResult(
                    REVISION_ID,
                    space_role="cross_modal_retrieval",
                ),
                gates,
            ),
            multimodal_embedding_provider=_ParallelMultimodalProvider(gates),
        )

        pack = await service.retrieve(
            _context(),
            RetrievalRequest(
                KB_ID,
                "budget headcount",
                top_k=2,
                include_debug=True,
            ),
        )

        self.assertEqual(len(pack.evidence), 2)
        self.assertEqual(
            {item.indexed_document_version_id for item in pack.evidence},
            {
                first.indexed_document_version_id,
                second.indexed_document_version_id,
            },
        )
        assert pack.debug is not None
        self.assertEqual(pack.debug.evidence_group_count, 2)

    async def test_weak_relation_does_not_expand_visual(self) -> None:
        gates = _ParallelGates()
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            _ParallelTextProvider(gates),
            _MultimodalStore(
                VectorSearchResult(REVISION_ID, (_hit(CHUNK_1, distance=0.2),)),
                VectorSearchResult(REVISION_ID, space_role="cross_modal_retrieval"),
                gates,
            ),
            multimodal_embedding_provider=_ParallelMultimodalProvider(gates),
            relation_hydrator=_Hydrator((_relation("same_page"),)),
        )

        pack = await service.retrieve(
            _context(), RetrievalRequest(KB_ID, "page", top_k=2)
        )

        self.assertEqual(pack.evidence[0].related_visuals, ())


class _Provider:
    max_batch_size = 10

    def __init__(self, *, vector: tuple[float, ...] = (0.6, 0.8)) -> None:
        self.embedding_space = _embedding_space()
        self.vector = vector
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> tuple[float, ...]:
        self.queries.append(text)
        return self.vector


class _LocalReranker:
    profile = RerankMode.LOCAL_MINILM_V1
    max_documents = 20

    def __init__(
        self,
        values: dict[UUID, tuple[float, float, int, int]],
    ) -> None:
        self._values = values
        self.queries: list[str] = []
        self.documents = []

    async def score(self, query, documents):
        self.queries.append(query)
        self.documents.append(documents)
        return tuple(
            ModelRerankScore(
                index_chunk_id=item.index_chunk_id,
                score=self._values[item.index_chunk_id][0],
                raw_logit=self._values[item.index_chunk_id][1],
                window_count=self._values[item.index_chunk_id][2],
                winning_window_index=self._values[item.index_chunk_id][3],
            )
            for item in documents
        )


class _HangingProvider(_Provider):
    def __init__(self) -> None:
        super().__init__()
        self.cancelled = False

    async def embed_query(self, text: str) -> tuple[float, ...]:
        self.queries.append(text)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("unreachable")


class _Store:
    def __init__(
        self,
        result: VectorSearchResult | None,
        *,
        adjacent_result: AdjacentChunkResult | None = None,
    ) -> None:
        self.result = result
        self.adjacent_result = adjacent_result
        self.plans = []
        self.embeddings = []
        self.adjacent_queries = []

    async def adjacent_chunks(self, query):
        self.adjacent_queries.append(query)
        return self.adjacent_result

    async def search(self, plan, query_embedding):
        self.plans.append(plan)
        self.embeddings.append(query_embedding)
        return self.result


class _HybridVectorStore(_Store):
    async def has_space_role(self, plan, role):
        del plan, role
        return False


class _LexicalStore:
    def __init__(self, result: LexicalSearchResult | None) -> None:
        self.result = result
        self.queries: list[str] = []
        self.embeddings: list[tuple[float, ...]] = []

    async def search(self, plan, query, query_embedding, **kwargs):
        del plan, kwargs
        self.queries.append(query)
        self.embeddings.append(query_embedding)
        return self.result


class _FailingLexicalStore:
    async def search(self, plan, query, query_embedding, **kwargs):
        del plan, query, query_embedding, kwargs
        await asyncio.sleep(0)
        raise RetrievalExecutionError(
            ErrorCode.INDEX_REVISION_INCOMPATIBLE,
            diagnostic={"check": "lexical_manifest_coverage"},
        )


class _CancellableHybridVectorStore:
    def __init__(self) -> None:
        self.cancelled = False

    async def has_space_role(self, plan, role):
        del plan, role
        return False

    async def search(self, plan, query_embedding):
        del plan, query_embedding
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("unreachable")


class _ParallelGates:
    def __init__(self) -> None:
        self.text_embedding_started = asyncio.Event()
        self.image_embedding_started = asyncio.Event()
        self.text_search_started = asyncio.Event()
        self.image_search_started = asyncio.Event()


class _ParallelTextProvider(_Provider):
    def __init__(self, gates: _ParallelGates) -> None:
        super().__init__()
        self._gates = gates

    async def embed_query(self, text: str) -> tuple[float, ...]:
        self.queries.append(text)
        self._gates.text_embedding_started.set()
        await self._gates.image_embedding_started.wait()
        return self.vector


class _ParallelMultimodalProvider:
    max_batch_size = 10

    def __init__(self, gates: _ParallelGates) -> None:
        self.embedding_space = _embedding_space()
        self._gates = gates

    async def embed_texts(self, texts: tuple[str, ...]) -> EmbeddingBatch:
        del texts
        self._gates.image_embedding_started.set()
        await self._gates.text_embedding_started.wait()
        return EmbeddingBatch(((0.6, 0.8),))


class _UnifiedProvider:
    max_batch_size = 10

    def __init__(self) -> None:
        revision_id = UUID("01900000-0000-7000-8000-000000000899")
        self.embedding_space = replace(
            _embedding_space(),
            model_profile_revision_id=revision_id,
        )
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> tuple[float, ...]:
        self.queries.append(text)
        return (0.6, 0.8)


class _MultimodalStore:
    def __init__(self, text_result, cross_result, gates: _ParallelGates) -> None:
        self.text_result = text_result
        self.cross_result = cross_result
        self._gates = gates
        self.text_embeddings: list[tuple[float, ...]] = []
        self.cross_embeddings: list[tuple[float, ...]] = []

    async def has_space_role(self, plan, role):
        del plan, role
        return True

    async def search(self, plan, query_embedding):
        del plan
        self.text_embeddings.append(query_embedding)
        self._gates.text_search_started.set()
        await self._gates.image_search_started.wait()
        return self.text_result

    async def search_space(self, plan, query_embedding, **kwargs):
        del plan, kwargs
        self.cross_embeddings.append(query_embedding)
        self._gates.image_search_started.set()
        await self._gates.text_search_started.wait()
        return self.cross_result


class _Hydrator:
    def __init__(self, relations) -> None:
        self.relations = relations

    async def hydrate(self, context, **kwargs):
        del context, kwargs
        return self.relations


class _DebugDeniedPolicy:
    def metadata_filter(self, context: AuthContext) -> MetadataFilter:
        if context.workspace_id != WORKSPACE:
            raise AccessDeniedError("wrong workspace")
        return MetadataFilter(WORKSPACE)

    def authorize_retrieval_debug(self, context: AuthContext) -> None:
        del context
        raise AccessDeniedError("retrieval debug is not authorized")


def _context(workspace_id: UUID = WORKSPACE) -> AuthContext:
    return AuthContext("principal", "client", workspace_id)


def _plan() -> RetrievalQueryPlan:
    return RetrievalQueryPlan(
        workspace_id=WORKSPACE,
        knowledge_base_id=KB_ID,
        strategy=RetrievalStrategy.EXACT_VECTOR,
        top_k=5,
    )


def _embedding_space() -> EmbeddingSpaceDefinition:
    return EmbeddingSpaceDefinition(
        provider_identity="test",
        endpoint_identity="test-endpoint",
        requested_model="test-embedding",
        resolved_model="test-embedding",
        model_version="v1",
        deployment_revision=None,
        dimension=2,
        distance_metric="cosine",
        vector_data_type="float32",
        normalization="l2",
        configuration_fingerprint="sha256:configuration",
        tokenizer_fingerprint=None,
        compatibility_fingerprint="sha256:compatibility",
    )


def _relation(relation_type: str) -> IndexChunkAssetRelationSnapshot:
    return IndexChunkAssetRelationSnapshot(
        id=UUID("01900000-0000-7000-8000-000000000817"),
        workspace_id=WORKSPACE,
        kb_id=KB_ID,
        indexed_document_version_id=UUID(
            "01900000-0000-7000-8000-000000000821"
        ),
        index_revision_id=REVISION_ID,
        chunk_id=CHUNK_1,
        visual_unit_id=VISUAL_1,
        asset_id=ASSET_1,
        relation_type=relation_type,
        confidence_micros=900_000,
        figure_label="Figure 1",
        ordinal=0,
        provenance="parser_structure",
        evidence_group_key="figure:1",
        document_id=UUID("01900000-0000-7000-8000-000000000822"),
        document_version_id=UUID("01900000-0000-7000-8000-000000000823"),
        chunk_ordinal=0,
        chunk_content="parent narrative",
        chunk_modality="text",
        chunk_source_location={"page": 1},
        chunk_hierarchy={"section": "test"},
        chunk_source_metadata={"filename": "safe.pdf"},
        visual_ordinal=1,
        visual_content="",
        visual_modality="image",
        visual_source_location={"page": 1},
        visual_hierarchy={"section": "test"},
        visual_source_metadata={"filename": "safe.pdf"},
        asset_media_type="image/png",
        asset_checksum_sha256="a" * 64,
        asset_width=640,
        asset_height=480,
    )


def _hit(
    chunk_id: UUID,
    *,
    distance: float = 0.1,
    ordinal: int = 0,
) -> VectorSearchHit:
    return VectorSearchHit(
        workspace_id=WORKSPACE,
        knowledge_base_id=KB_ID,
        index_revision_id=REVISION_ID,
        index_chunk_id=chunk_id,
        indexed_document_version_id=UUID(
            "01900000-0000-7000-8000-000000000821"
        ),
        document_id=UUID("01900000-0000-7000-8000-000000000822"),
        document_version_id=UUID("01900000-0000-7000-8000-000000000823"),
        ordinal=ordinal,
        text=f"evidence-{ordinal}",
        source_location={"line_start": ordinal + 1, "line_end": ordinal + 1},
        hierarchy={"section": "test"},
        source_metadata={"filename": "safe.txt"},
        cosine_distance=distance,
        build_status="ready",
        serving_status="serving",
        is_current_serving_version=True,
    )


def _evidence(chunk_id: UUID, *, ordinal: int) -> Evidence:
    hit = _hit(chunk_id, ordinal=ordinal)
    return Evidence(
        rank=1,
        index_chunk_id=hit.index_chunk_id,
        indexed_document_version_id=hit.indexed_document_version_id,
        document_id=hit.document_id,
        document_version_id=hit.document_version_id,
        index_revision_id=hit.index_revision_id,
        ordinal=hit.ordinal,
        text=hit.text,
        source_location=hit.source_location,
        hierarchy=hit.hierarchy,
        source_metadata=hit.source_metadata,
        score=0.9,
    )
