from __future__ import annotations

import unittest
from dataclasses import replace
from uuid import UUID

from rag_kb.auth import SingleWorkspaceAccessPolicy
from rag_kb.adapters.graph_store.postgres import (
    PgGraphStore,
    _GraphEdge,
    _aggregate_edges,
    _build_paths,
)
from rag_kb.domain import (
    GRAPH_EXTRACTOR_VERSION,
    GraphChunkEvidence,
    GraphConfigSnapshot,
    GraphConfigStatus,
    GraphEntityCandidate,
    GraphPathCandidate,
    GraphPathHop,
    GraphRetrievalRequest,
    GraphTraversalResult,
    RetrievalExecutionError,
    RetrievalStrategy,
    VectorSearchResult,
)
from rag_kb.retrieval.service import RetrievalService, _pack_graph_evidence

from tests.unit.test_retrieval_service import (
    CHUNK_1,
    KB_ID,
    REVISION_ID,
    WORKSPACE,
    _Provider,
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
        )

        with self.assertRaises(RetrievalExecutionError) as raised:
            await service.retrieve_graph(
                _context(), GraphRetrievalRequest(KB_ID, "Atlas", top_k=4)
            )

        self.assertEqual(raised.exception.code.value, "GRAPH_NOT_READY")
        self.assertEqual(provider.queries, [])
        self.assertEqual(graph_store.traversal_queries, [])

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
        )

        with self.assertRaises(RetrievalExecutionError):
            await service.retrieve_graph(
                _context(), GraphRetrievalRequest(KB_ID, "Atlas", top_k=4)
            )

        self.assertEqual(provider.queries, [])
        self.assertEqual(graph_store.traversal_queries, [])

    def test_seed_first_packing_keeps_a_path_whole(self) -> None:
        seed = _seed_evidence(CHUNK_1)
        path = _path_result()

        evidence, bundles = _pack_graph_evidence((seed,), path, top_k=4)

        self.assertEqual([item.index_chunk_id for item in evidence], [seed.index_chunk_id, CHUNK_3])
        self.assertEqual(len(bundles), 1)
        self.assertEqual(bundles[0].chunk_ids, (seed.index_chunk_id, CHUNK_3))


class GraphStoreAlgorithmTests(unittest.TestCase):
    def test_seed_chunk_can_ground_paths_without_query_entity_keys(self) -> None:
        edge = _GraphEdge(
            relation_id=RELATION_ID,
            subject_entity_key=ENTITY_A,
            object_entity_key=ENTITY_B,
            predicate="released",
            normalized_predicate="released",
            source_chunk=_graph_chunk(CHUNK_1),
        )

        paths, chunks, rejected = _build_paths(
            (),
            (edge,),
            (),
            {CHUNK_1},
            max_paths=20,
        )

        self.assertEqual(rejected, 0)
        self.assertEqual(len(paths), 2)
        self.assertEqual({path.entry_entity_key for path in paths}, {ENTITY_A, ENTITY_B})
        self.assertEqual(chunks, (_graph_chunk(CHUNK_1),))

    def test_graph_sql_is_scoped_and_does_not_require_pg_trgm(self) -> None:
        statements = (
            PgGraphStore._config_statement(),
            PgGraphStore._candidate_statement(False),
            PgGraphStore._candidate_statement(True),
            PgGraphStore._one_hop_statement(),
            PgGraphStore._two_hop_statement(),
        )

        for statement in statements:
            self.assertNotIn("pg_trgm", statement.text)
            self.assertIn(":workspace_id", statement.text)
            self.assertIn(":kb_id", statement.text)
        self.assertIn("rel.subject_entity_key <> rel.object_entity_key", statements[-1].text)

    def test_two_hop_path_aggregates_duplicate_support_and_keeps_sources(self) -> None:
        first = _GraphEdge(
            relation_id=RELATION_ID,
            subject_entity_key=ENTITY_A,
            object_entity_key=ENTITY_B,
            predicate="released",
            normalized_predicate="released",
            source_chunk=_graph_chunk(CHUNK_1),
        )
        duplicate = replace(
            first,
            relation_id=UUID("01900000-0000-7000-8000-000000000938"),
            source_chunk=_graph_chunk(CHUNK_3),
        )
        final = _GraphEdge(
            relation_id=UUID("01900000-0000-7000-8000-000000000939"),
            subject_entity_key=ENTITY_B,
            object_entity_key="c" * 64,
            predicate="supports",
            normalized_predicate="supports",
            source_chunk=_graph_chunk(CHUNK_3),
        )
        aggregated = _aggregate_edges((first, duplicate), {CHUNK_1})
        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0].support_count, 2)
        self.assertEqual(aggregated[0].source_chunk.index_chunk_id, CHUNK_1)

        paths, chunks, rejected = _build_paths(
            (ENTITY_A,),
            (first,),
            ((first, final),),
            {CHUNK_1},
            max_paths=20,
        )

        self.assertEqual(rejected, 0)
        self.assertTrue(any(path.hop_count == 2 for path in paths))
        self.assertEqual({item.index_chunk_id for item in chunks}, {CHUNK_1, CHUNK_3})

    def test_two_hop_intermediate_degree_limit_rejects_high_degree_expansion(self) -> None:
        intermediate = ENTITY_B
        first = _GraphEdge(
            relation_id=RELATION_ID,
            subject_entity_key=ENTITY_A,
            object_entity_key=intermediate,
            predicate="relates",
            normalized_predicate="relates",
            source_chunk=_graph_chunk(CHUNK_1),
        )
        neighbors = tuple(
            _GraphEdge(
                relation_id=UUID(f"01900000-0000-7000-8000-0000000009{i:02d}"),
                subject_entity_key=intermediate,
                object_entity_key=f"{i:064x}",
                predicate="relates",
                normalized_predicate="relates",
                source_chunk=_graph_chunk(CHUNK_3),
            )
            for i in range(1, 18)
        )

        paths, _, _ = _build_paths(
            (ENTITY_A,),
            (first, *neighbors),
            tuple((first, neighbor) for neighbor in neighbors),
            {CHUNK_1},
            max_paths=20,
        )

        self.assertFalse(any(path.hop_count == 2 for path in paths))


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

    async def get_config(self, workspace_id, knowledge_base_id):
        assert workspace_id == WORKSPACE
        assert knowledge_base_id == KB_ID
        return self.config

    async def find_entity_candidates(self, query):
        return (
            GraphEntityCandidate(
                entity_key=ENTITY_A,
                entity_type="organization",
                surface="Atlas",
                normalized_surface="atlas",
                index_chunk_id=CHUNK_3,
                indexed_document_version_id=TARGET_ID,
                index_revision_id=REVISION_ID,
            ),
        )

    async def traverse(self, query):
        self.traversal_queries.append(query)
        return self.traversal


def _ready_config() -> GraphConfigSnapshot:
    return GraphConfigSnapshot(
        workspace_id=WORKSPACE,
        knowledge_base_id=KB_ID,
        status=GraphConfigStatus.READY,
        build_id=BUILD_ID,
        chat_profile_revision_id=UUID("01900000-0000-7000-8000-000000000932"),
        extractor_version=GRAPH_EXTRACTOR_VERSION,
        preflight_extractor_version=GRAPH_EXTRACTOR_VERSION,
        last_error_code=None,
        eligible_chunk_count=1,
        processed_chunk_count=1,
        extracted_chunk_count=1,
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
