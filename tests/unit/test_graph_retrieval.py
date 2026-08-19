from __future__ import annotations

import unittest
from dataclasses import replace
from uuid import UUID

from rag_kb.auth import SingleWorkspaceAccessPolicy
from rag_kb.domain import (
    GRAPH_EXTRACTOR_VERSION,
    GraphChunkEvidence,
    GraphConfigSnapshot,
    GraphConfigStatus,
    GraphitiBuildSnapshot,
    GraphitiBuildStatus,
    GraphitiEdgeResult,
    GraphitiSearchQuery,
    GraphPathCandidate,
    GraphPathHop,
    GraphRetrievalRequest,
    GraphTraversalResult,
    RetrievalExecutionError,
    RerankMode,
    RetrievalStrategy,
    VectorSearchResult,
)
from sqlalchemy import bindparam, text

from rag_kb.adapters.graph_store.postgres import _bounded_graphiti_paths
from rag_kb.domain.graph import GRAPH_MAX_PATHS
from rag_kb.retrieval.service import (
    RetrievalService,
    _graph_evidence_from_chunk,
    _pack_graph_evidence,
)

from tests.unit.test_retrieval_service import (
    CHUNK_1,
    KB_ID,
    REVISION_ID,
    WORKSPACE,
    _Provider,
    _LocalReranker,
    _Store,
    _context,
    _hit,
)
from rag_kb.domain import Evidence
from rag_kb.document_processing.lexical import LEXICAL_ANALYZER_VERSION
from rag_kb.domain import LexicalSearchResult


BUILD_ID = UUID("01900000-0000-7000-8000-000000000931")
CHUNK_3 = UUID("01900000-0000-7000-8000-000000000933")
TARGET_ID = UUID("01900000-0000-7000-8000-000000000934")
DOCUMENT_ID = UUID("01900000-0000-7000-8000-000000000935")
VERSION_ID = UUID("01900000-0000-7000-8000-000000000936")
RELATION_ID = UUID("01900000-0000-7000-8000-000000000937")
ENTITY_A = "a" * 64
ENTITY_B = "b" * 64


class GraphRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_exception_fails_closed_without_rebuild(self) -> None:
        graph_store = _GraphStore(_ready_config(), _path_result())
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            _Provider(),
            _Store(VectorSearchResult(REVISION_ID, (_hit(CHUNK_3, ordinal=0),))),
            lexical_store=_LexicalStore(),
            graph_store=graph_store,
            graphiti_graph=_Graphiti(search_error=RuntimeError("embed timeout")),
        )

        with self.assertRaises(RetrievalExecutionError) as raised:
            await service.retrieve_graph(
                _context(), GraphRetrievalRequest(KB_ID, "Atlas", top_k=4)
            )

        self.assertEqual(raised.exception.code.value, "GRAPH_NOT_READY")
        self.assertEqual(graph_store.rebuilds, [])

    async def test_probe_exception_fails_closed_without_rebuild(self) -> None:
        graph_store = _GraphStore(_ready_config(), _path_result())
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            _Provider(),
            _Store(VectorSearchResult(REVISION_ID, (_hit(CHUNK_3, ordinal=0),))),
            lexical_store=_LexicalStore(),
            graph_store=graph_store,
            graphiti_graph=_Graphiti(probe_error=RuntimeError("falkordb down")),
        )

        with self.assertRaises(RetrievalExecutionError) as raised:
            await service.retrieve_graph(
                _context(), GraphRetrievalRequest(KB_ID, "Atlas", top_k=4)
            )

        self.assertEqual(raised.exception.code.value, "GRAPH_NOT_READY")
        self.assertEqual(graph_store.rebuilds, [])

    async def test_runtime_probe_failure_fails_closed_without_scheduling_rebuild(self) -> None:
        graph_store = _GraphStore(_ready_config(), _path_result())
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            _Provider(),
            _Store(VectorSearchResult(REVISION_ID, (_hit(CHUNK_3, ordinal=0),))),
            lexical_store=_LexicalStore(),
            graph_store=graph_store,
            graphiti_graph=_Graphiti(probe_success=False),
        )

        with self.assertRaises(RetrievalExecutionError) as raised:
            await service.retrieve_graph(
                _context(), GraphRetrievalRequest(KB_ID, "Atlas", top_k=4)
            )

        self.assertEqual(raised.exception.code.value, "GRAPH_NOT_READY")
        self.assertEqual(graph_store.rebuilds, [])

    async def test_graph_ignores_ordinary_hybrid_gate_and_preserves_hybrid_seed_strategy(
        self,
    ) -> None:
        vector_store = _Store(
            VectorSearchResult(REVISION_ID, (_hit(CHUNK_3, ordinal=0),))
        )
        lexical_store = _LexicalStore()
        graph_store = _GraphStore(_ready_config(), _path_result())
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            _Provider(),
            vector_store,
            lexical_store=lexical_store,
            hybrid_enabled=False,
            graph_store=graph_store,
            graphiti_graph=_Graphiti(),
        )

        pack = await service.retrieve_graph(
            _context(), GraphRetrievalRequest(KB_ID, "Atlas", top_k=4, include_debug=True)
        )

        self.assertIs(pack.strategy, RetrievalStrategy.HYBRID)
        self.assertEqual(
            {item.index_chunk_id for item in pack.evidence},
            {CHUNK_1, CHUNK_3},
        )
        self.assertIsNotNone(pack.debug)
        assert pack.debug is not None
        self.assertIsNotNone(pack.debug.graph)
        self.assertEqual(pack.debug.graph.bundle_count, 1)

    async def test_graph_building_fails_closed_before_embedding_or_graph_traversal(
        self,
    ) -> None:
        vector_store = _Store(VectorSearchResult(REVISION_ID, ()))
        graph_store = _GraphStore(replace(_ready_config(), status=GraphConfigStatus.BUILDING), None)
        provider = _Provider()
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            provider,
            vector_store,
            lexical_store=_LexicalStore(),
            graph_store=graph_store,
            graphiti_graph=_Graphiti(),
        )

        with self.assertRaises(RetrievalExecutionError) as raised:
            await service.retrieve_graph(
                _context(), GraphRetrievalRequest(KB_ID, "Atlas", top_k=4)
            )

        self.assertEqual(raised.exception.code.value, "GRAPH_NOT_READY")
        self.assertEqual(provider.queries, [])
        self.assertEqual(graph_store.traversal_queries, [])

    async def test_adaptive_supplement_skips_simple_seed_and_excludes_existing_chunks(
        self,
    ) -> None:
        provider = _Provider()
        graph_store = _GraphStore(_ready_config(), _path_result())
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            provider,
            _Store(VectorSearchResult(REVISION_ID, ())),
            lexical_store=_LexicalStore(),
            graph_store=graph_store,
            graphiti_graph=_Graphiti(),
        )

        result = await service.retrieve_graphiti_supplement(
            _context(),
            knowledge_base_id=KB_ID,
            index_revision_id=REVISION_ID,
            query="Atlas relation",
            rerank_mode=RerankMode.CLASSIC,
            excluded_index_chunk_ids=(CHUNK_3,),
        )

        self.assertEqual(result.route_result_code, "admitted")
        self.assertEqual([item.index_chunk_id for item in result.evidence], [CHUNK_1])
        self.assertEqual(provider.queries, [])
        self.assertEqual(graph_store.traversal_queries, [])

    async def test_adaptive_supplement_reports_not_ready_without_active_build(
        self,
    ) -> None:
        graph_store = _GraphStore(
            replace(_ready_config(), status=GraphConfigStatus.BUILDING), None
        )
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            _Provider(),
            _Store(VectorSearchResult(REVISION_ID, ())),
            lexical_store=_LexicalStore(),
            graph_store=graph_store,
            graphiti_graph=_Graphiti(),
        )

        result = await service.retrieve_graphiti_supplement(
            _context(),
            knowledge_base_id=KB_ID,
            index_revision_id=REVISION_ID,
            query="Atlas relation",
            rerank_mode=RerankMode.CLASSIC,
            excluded_index_chunk_ids=(),
        )

        self.assertEqual(result.route_result_code, "not_ready")

    async def test_building_config_serves_the_existing_active_ready_build(self) -> None:
        graph_store = _GraphStore(
            replace(_ready_config(), status=GraphConfigStatus.BUILDING),
            _path_result(),
        )
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            _Provider(),
            _Store(VectorSearchResult(REVISION_ID, ())),
            lexical_store=_LexicalStore(),
            graph_store=graph_store,
            graphiti_graph=_Graphiti(),
        )

        result = await service.retrieve_graphiti_supplement(
            _context(),
            knowledge_base_id=KB_ID,
            index_revision_id=REVISION_ID,
            query="Atlas relation",
            rerank_mode=RerankMode.CLASSIC,
            excluded_index_chunk_ids=(),
        )

        self.assertEqual(result.route_result_code, "admitted")

    async def test_graph_classic_uses_native_order_without_local_reranker(self) -> None:
        traversal = _path_result()
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            _Provider(),
            _Store(VectorSearchResult(REVISION_ID, ())),
        )

        result = await service._rerank_graphiti_candidates(  # noqa: SLF001
            "Atlas",
            traversal,
            rerank_mode=RerankMode.CLASSIC,
        )

        self.assertEqual(result, traversal)

    async def test_graph_minilm_reorders_without_deleting_low_scores(self) -> None:
        traversal = _path_result()
        reranker = _LocalReranker(
            {
                CHUNK_1: (0.10, -2.0, 1, 0),
                CHUNK_3: (0.05, -3.0, 1, 0),
            }
        )
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            _Provider(),
            _Store(VectorSearchResult(REVISION_ID, ())),
            text_reranker=reranker,
        )

        result = await service._rerank_graphiti_candidates(  # noqa: SLF001
            "Atlas",
            traversal,
            rerank_mode=RerankMode.LOCAL_MINILM_V1,
        )

        self.assertEqual(result.chunks, traversal.chunks)
        self.assertEqual(len(result.paths), len(traversal.paths))

    async def test_stale_ready_graph_fails_closed_before_model_or_traversal(self) -> None:
        graph_store = _GraphStore(
            replace(
                _ready_config(),
                extractor_version="entity_graph_v1",
                preflight_extractor_version="entity_graph_v1",
            ),
            None,
        )
        provider = _Provider()
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            provider,
            _Store(VectorSearchResult(REVISION_ID, ())),
            lexical_store=_LexicalStore(),
            graph_store=graph_store,
            graphiti_graph=_Graphiti(),
        )

        with self.assertRaises(RetrievalExecutionError):
            await service.retrieve_graph(
                _context(), GraphRetrievalRequest(KB_ID, "Atlas", top_k=4)
            )

        self.assertEqual(provider.queries, [])
        self.assertEqual(graph_store.traversal_queries, [])

    def test_episode_uuid_lookup_uses_an_expanding_bind(self) -> None:
        statement = text(
            "SELECT 1 FROM graphiti_episode_chunk mapping "
            "WHERE mapping.episode_uuid IN :episode_uuids"
        ).bindparams(bindparam("episode_uuids", expanding=True))
        compiled = str(
            statement.bindparams(episode_uuids=["episode-1", "episode-2"]).compile(
                compile_kwargs={"render_postcompile": True}
            )
        )
        self.assertIn("IN (", compiled)
        self.assertNotIn("ANY(", compiled)

    def test_hydration_caps_paths_at_the_domain_bound(self) -> None:
        edges = tuple(
            GraphitiEdgeResult(f"edge-{index}", "fact", (f"episode-{index}",), index)
            for index in range(1, GRAPH_MAX_PATHS + 4)
        )
        chunks = {
            f"episode-{index}": _graph_chunk(UUID(int=index))
            for index in range(1, GRAPH_MAX_PATHS + 4)
        }

        paths, admitted, rejected = _bounded_graphiti_paths(edges, chunks)

        self.assertEqual(len(paths), GRAPH_MAX_PATHS)
        self.assertEqual(len(admitted), GRAPH_MAX_PATHS)
        self.assertEqual(rejected, 3)

        GraphTraversalResult(
            resolved_active_revision_id=REVISION_ID,
            paths=paths,
            chunks=admitted,
            rejected_path_count=rejected,
        )

    def test_graphiti_search_accepts_evaluator_limit_64(self) -> None:
        query = GraphitiSearchQuery(
            workspace_id=WORKSPACE,
            knowledge_base_id=KB_ID,
            build_id=BUILD_ID,
            group_id="evaluation",
            query="bounded evaluator probe",
            limit=64,
        )

        self.assertEqual(query.limit, 64)

    def test_hydration_keeps_distinct_edges_for_the_same_chunk(self) -> None:
        edges = (
            GraphitiEdgeResult("edge-1", "first fact", ("episode-1",), 1),
            GraphitiEdgeResult("edge-2", "second fact", ("episode-1",), 2),
        )
        chunk = _graph_chunk(CHUNK_1)

        paths, admitted, rejected = _bounded_graphiti_paths(
            edges,
            {"episode-1": chunk},
        )

        self.assertEqual(len(paths), 2)
        self.assertEqual({path.hops[0].object_entity_key for path in paths}, {"edge-1", "edge-2"})
        self.assertEqual(admitted, (chunk,))
        self.assertEqual(rejected, 0)

    def test_seed_first_packing_keeps_a_path_whole(self) -> None:
        seed = _seed_evidence(CHUNK_1)
        path = _path_result()

        evidence, bundles = _pack_graph_evidence((seed,), path, top_k=4)

        self.assertEqual([item.index_chunk_id for item in evidence], [seed.index_chunk_id, CHUNK_3])
        self.assertEqual(len(bundles), 1)
        self.assertEqual(bundles[0].chunk_ids, (seed.index_chunk_id, CHUNK_3))

    def test_graph_chunk_hydration_keeps_graph_and_text_representations(self) -> None:
        traversal = _path_result()
        text_chunk = traversal.chunks[0]
        table_chunk = replace(text_chunk, modality="table")

        text_evidence = _graph_evidence_from_chunk(text_chunk, traversal.paths[0])
        table_evidence = _graph_evidence_from_chunk(table_chunk, traversal.paths[0])

        self.assertEqual(text_evidence.matched_representations, ("graph_path", "text"))
        self.assertEqual(
            table_evidence.matched_representations,
            ("graph_path", "table_text"),
        )


class _LexicalStore:
    async def search(self, plan, query, query_embedding, **kwargs):
        del plan, query, query_embedding, kwargs
        return LexicalSearchResult(
            REVISION_ID,
            analyzer_version=LEXICAL_ANALYZER_VERSION,
            manifest_target_count=1,
            hits=(),
        )


class _GraphStore:
    def __init__(self, config, traversal) -> None:
        self.config = config
        self.traversal = traversal
        self.traversal_queries = []
        self.rebuilds = []

    async def get_config(self, workspace_id, knowledge_base_id):
        assert workspace_id == WORKSPACE
        assert knowledge_base_id == KB_ID
        return self.config

    async def get_active_graphiti_build(self, workspace_id, knowledge_base_id):
        assert workspace_id == WORKSPACE
        assert knowledge_base_id == KB_ID
        return _ready_build() if self.traversal is not None else None

    async def first_graphiti_episode_uuid(
        self, workspace_id, knowledge_base_id, build_id
    ):
        del workspace_id, knowledge_base_id, build_id
        return "episode-1"

    async def hydrate_graphiti_edges(self, **kwargs):
        del kwargs
        return self.traversal

    async def schedule_graphiti_rebuild(
        self, workspace_id, knowledge_base_id, failed_build_id
    ):
        self.rebuilds.append((workspace_id, knowledge_base_id, failed_build_id))
        return True


class _Graphiti:
    def __init__(self, *, probe_success=True, probe_error=None, search_error=None):
        self.probe_success = probe_success
        self.probe_error = probe_error
        self.search_error = search_error

    async def probe(self, build, *, episode_uuid=None, require_complete=False):
        del build, episode_uuid, require_complete
        if self.probe_error is not None:
            raise self.probe_error
        return self.probe_success

    async def search(self, build, query):
        del build, query
        if self.search_error is not None:
            raise self.search_error
        return (GraphitiEdgeResult("edge-1", "released", ("episode-1",), 1),)

    async def add_episode(self, build, chunk):
        raise AssertionError((build, chunk))

    async def delete_graph(self, build):
        raise AssertionError(build)


def _ready_config() -> GraphConfigSnapshot:
    return GraphConfigSnapshot(
        workspace_id=WORKSPACE,
        knowledge_base_id=KB_ID,
        status=GraphConfigStatus.READY,
        build_id=BUILD_ID,
        active_build_id=BUILD_ID,
        chat_profile_revision_id=UUID("01900000-0000-7000-8000-000000000932"),
        extractor_version=GRAPH_EXTRACTOR_VERSION,
        preflight_extractor_version=GRAPH_EXTRACTOR_VERSION,
        last_error_code=None,
        eligible_chunk_count=1,
        processed_chunk_count=1,
        extracted_chunk_count=1,
        group_id=f"ws_{WORKSPACE}_kb_{KB_ID}_b_{BUILD_ID}",
    )


def _ready_build() -> GraphitiBuildSnapshot:
    return GraphitiBuildSnapshot(
        workspace_id=WORKSPACE,
        knowledge_base_id=KB_ID,
        build_id=BUILD_ID,
        group_id=f"ws_{WORKSPACE}_kb_{KB_ID}_b_{BUILD_ID}",
        status=GraphitiBuildStatus.READY,
        index_revision_id=REVISION_ID,
        serving_chunk_digest="a" * 64,
        expected_episode_count=1,
        chat_profile_revision_id=UUID("01900000-0000-7000-8000-000000000932"),
        embedding_profile_revision_id=UUID("01900000-0000-7000-8000-000000000940"),
        embedding_model="fake-embedding",
        embedding_dimension=768,
        extractor_version=GRAPH_EXTRACTOR_VERSION,
    )


def _seed_evidence(chunk_id: UUID = CHUNK_3) -> Evidence:
    hit = _hit(chunk_id, ordinal=0)
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


def _graph_chunk(chunk_id: UUID) -> GraphChunkEvidence:
    return GraphChunkEvidence(
        workspace_id=WORKSPACE,
        knowledge_base_id=KB_ID,
        index_revision_id=REVISION_ID,
        index_chunk_id=chunk_id,
        indexed_document_version_id=TARGET_ID,
        document_id=DOCUMENT_ID,
        document_version_id=VERSION_ID,
        ordinal=0,
        text=f"graph evidence {chunk_id}",
        source_location={"paragraph": 1},
        hierarchy={},
        source_metadata={},
        modality="text",
        evidence_group_key=None,
        document_display_name="document",
        document_original_filename="document.txt",
    )


def _path_result() -> GraphTraversalResult:
    hop = GraphPathHop(
        subject_entity_key=ENTITY_A,
        object_entity_key=ENTITY_B,
        predicate="released",
        normalized_predicate="released",
        relation_id=RELATION_ID,
        source_chunk_id=CHUNK_3,
        source_index_revision_id=REVISION_ID,
        source_location={"paragraph": 1},
    )
    path = GraphPathCandidate(
        path_id="path-1",
        entry_entity_key=ENTITY_A,
        hops=(hop,),
        anchor_chunk_id=CHUNK_1,
        rank=1,
        seed_entry=True,
    )
    return GraphTraversalResult(
        resolved_active_revision_id=REVISION_ID,
        paths=(path,),
        chunks=(_graph_chunk(CHUNK_1), _graph_chunk(CHUNK_3)),
    )


if __name__ == "__main__":
    unittest.main()
