from __future__ import annotations

import unittest
from dataclasses import replace
from uuid import UUID

from sqlalchemy.dialects import postgresql

from rag_kb.adapters import FixedPgVectorSpace, PgVectorStore
from rag_kb.auth import (
    AccessDeniedError,
    AuthContext,
    MetadataFilter,
    SingleWorkspaceAccessPolicy,
)
from rag_kb.domain import (
    EmbeddingBatch,
    EmbeddingSpaceDefinition,
    ErrorCode,
    RetrievalExecutionError,
    RetrievalQueryPlan,
    RetrievalRequest,
    RetrievalStrategy,
    VectorSearchHit,
    VectorSearchResult,
)
from rag_kb.retrieval import RetrievalService


WORKSPACE = UUID("01900000-0000-7000-8000-000000000801")
OTHER_WORKSPACE = UUID("01900000-0000-7000-8000-000000000802")
KB_ID = UUID("01900000-0000-7000-8000-000000000803")
REVISION_ID = UUID("01900000-0000-7000-8000-000000000804")
CHUNK_1 = UUID("01900000-0000-7000-8000-000000000811")
CHUNK_2 = UUID("01900000-0000-7000-8000-000000000812")


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
        self.assertIn("vector_record_1024.embedding <=>", sql)
        self.assertIn("knowledge_base.active_index_revision_id", sql)
        self.assertIn("indexed_document_version.build_status", sql)
        self.assertIn("indexed_document_version.serving_status", sql)
        self.assertNotIn("document.current_version_id", sql)
        self.assertIn("document.deleted_at IS NULL", sql)
        self.assertIn("document_version.source_status", sql)
        self.assertIn("vector_record_1024.embedding_space_id", sql)
        self.assertIn("ORDER BY cosine_distance ASC, index_chunk.id ASC", sql)
        self.assertIn("LIMIT %(top_k)s", sql)
        self.assertNotIn("hnsw", sql.lower())

    def test_pgvector_adapter_rejects_non_exact_or_wrong_dimension(self) -> None:
        definition = replace(_embedding_space(), dimension=1024)
        store = PgVectorStore(None, FixedPgVectorSpace(definition))  # type: ignore[arg-type]
        exact = _plan()

        with self.assertRaises(RetrievalExecutionError) as dimension:
            store._require_exact_plan(exact, (1.0, 0.0))  # noqa: SLF001
        self.assertEqual(dimension.exception.code, ErrorCode.EMBEDDING_RESPONSE_INVALID)

        with self.assertRaises(RetrievalExecutionError) as capability:
            store._require_exact_plan(  # noqa: SLF001
                replace(exact, strategy=RetrievalStrategy.HYBRID),
                tuple(0.0 for _ in range(1024)),
            )
        self.assertEqual(capability.exception.code, ErrorCode.CAPABILITY_NOT_ENABLED)

    def test_query_plan_rejects_relaxed_server_owned_filters(self) -> None:
        plan = _plan()
        relaxed = (
            {"revision_selector": "retired"},
            {"current_document_version_only": False},
            {"build_status": "processing"},
            {"serving_status": "candidate"},
            {"distance_metric": "inner_product"},
            {"candidate_count": 50},
            {"ef_search": 100},
            {"rerank": True},
        )

        for changes in relaxed:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(plan, **changes)


class RetrievalServiceTests(unittest.IsolatedAsyncioTestCase):
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

        self.assertEqual(provider.queries, [("query",)])
        self.assertEqual(store.embeddings, [(0.6, 0.8)])
        plan = store.plans[0]
        self.assertEqual((plan.workspace_id, plan.knowledge_base_id), (WORKSPACE, KB_ID))
        self.assertEqual(plan.revision_selector.value, "active")
        self.assertTrue(plan.current_document_version_only)
        self.assertEqual((plan.build_status, plan.serving_status), ("ready", "serving"))
        self.assertEqual((plan.strategy.value, plan.distance_metric), ("exact_vector", "cosine"))
        self.assertIsNone(plan.candidate_count)
        self.assertIsNone(plan.ef_search)
        self.assertEqual([item.index_chunk_id for item in pack.evidence], [CHUNK_1, CHUNK_2])
        self.assertEqual([item.rank for item in pack.evidence], [1, 2])
        self.assertAlmostEqual(pack.evidence[0].score, 0.8)
        self.assertIsNotNone(pack.debug)
        assert pack.debug is not None
        self.assertEqual(pack.debug.resolved_active_revision_id, REVISION_ID)
        self.assertEqual(pack.debug.result_count, 2)

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
        self.assertEqual(store.plans[0].build_status, "ready")

    async def test_unsupported_strategy_and_rerank_fail_before_external_io(self) -> None:
        for request in (
            RetrievalRequest(KB_ID, "query", strategy=RetrievalStrategy.HYBRID),
            RetrievalRequest(KB_ID, "query", rerank=True),
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

    async def test_invalid_query_embedding_cardinality_is_rejected(self) -> None:
        provider = _Provider(vectors=())
        store = _Store(VectorSearchResult(REVISION_ID))
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE), provider, store
        )

        with self.assertRaises(RetrievalExecutionError) as failure:
            await service.retrieve(_context(), RetrievalRequest(KB_ID, "query"))

        self.assertEqual(failure.exception.code, ErrorCode.EMBEDDING_RESPONSE_INVALID)
        self.assertEqual(store.plans, [])


class _Provider:
    max_batch_size = 10

    def __init__(self, *, vectors: tuple[tuple[float, ...], ...] = ((0.6, 0.8),)) -> None:
        self.embedding_space = _embedding_space()
        self.vectors = vectors
        self.queries: list[tuple[str, ...]] = []

    async def embed(self, texts: tuple[str, ...]) -> EmbeddingBatch:
        self.queries.append(texts)
        return EmbeddingBatch(model="test-embedding", vectors=self.vectors)


class _Store:
    def __init__(self, result: VectorSearchResult | None) -> None:
        self.result = result
        self.plans = []
        self.embeddings = []

    async def search(self, plan, query_embedding):
        self.plans.append(plan)
        self.embeddings.append(query_embedding)
        return self.result


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
