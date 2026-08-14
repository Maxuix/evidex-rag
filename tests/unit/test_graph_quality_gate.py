from __future__ import annotations

import unittest
from dataclasses import dataclass
import math
from uuid import UUID

from rag_kb.auth import SingleWorkspaceAccessPolicy
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
    LexicalSearchResult,
    RetrievalRequest,
    RetrievalStrategy,
    RerankMode,
    VectorSearchResult,
)
from rag_kb.document_processing.lexical import LEXICAL_ANALYZER_VERSION
from rag_kb.retrieval.service import RetrievalService

from tests.unit.test_retrieval_service import (
    CHUNK_1,
    CHUNK_2,
    KB_ID,
    REVISION_ID,
    WORKSPACE,
    _Provider,
    _context,
    _hit,
)


BUILD_ID = UUID("01900000-0000-7000-8000-000000000951")
TARGET_ID = UUID("01900000-0000-7000-8000-000000000952")
DOCUMENT_ID = UUID("01900000-0000-7000-8000-000000000953")
VERSION_ID = UUID("01900000-0000-7000-8000-000000000954")


@dataclass(frozen=True, slots=True)
class _HardCase:
    query: str
    entry_key: str
    seed_chunk_id: UUID
    path_chunk_ids: tuple[UUID, ...]
    traversal: GraphTraversalResult

    @property
    def gold_chunk_ids(self) -> frozenset[UUID]:
        return frozenset((self.seed_chunk_id, *self.path_chunk_ids))


class GraphQualityGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_graph_improves_complete_path_recall_without_reordering_seed(self) -> None:
        cases = _hard_cases()
        hits = {case.query: _hit(case.seed_chunk_id, ordinal=0) for case in cases}
        provider = _QualityProvider(hits)
        vector_store = _QualityVectorStore(provider, hits)
        graph_store = _QualityGraphStore(cases)
        service = RetrievalService(
            SingleWorkspaceAccessPolicy(WORKSPACE),
            provider,
            vector_store,
            lexical_store=_QualityLexicalStore(),
            hybrid_enabled=True,
            graph_store=graph_store,
        )

        hybrid_packs = []
        graph_packs = []
        for case in cases:
            hybrid_packs.append(
                await service.retrieve(
                    _context(),
                    RetrievalRequest(
                        KB_ID,
                        case.query,
                        top_k=6,
                        strategy=RetrievalStrategy.HYBRID,
                        rerank_mode=RerankMode.CLASSIC,
                    ),
                )
            )
            graph_packs.append(
                await service.retrieve_graph(
                    _context(),
                    GraphRetrievalRequest(
                        KB_ID, case.query, top_k=6, include_debug=True
                    ),
                )
            )

        hybrid_recall = _complete_path_recall(hybrid_packs, cases)
        graph_recall = _complete_path_recall(graph_packs, cases)
        hybrid_coverage = _evidence_coverage(hybrid_packs, cases)
        graph_coverage = _evidence_coverage(graph_packs, cases)

        self.assertEqual(hybrid_recall, 0.0)
        self.assertEqual(graph_recall, 1.0)
        self.assertLess(hybrid_coverage, graph_coverage)
        self.assertEqual(graph_coverage, 1.0)
        for hybrid, graph in zip(hybrid_packs, graph_packs):
            self.assertEqual(
                graph.evidence[0].index_chunk_id,
                hybrid.evidence[0].index_chunk_id,
            )
            self.assertIsNotNone(graph.debug)
            assert graph.debug is not None
            self.assertIsNotNone(graph.debug.graph)
            assert graph.debug.graph is not None
            self.assertEqual(len(graph.debug.graph.bundles), 1)


class _QualityProvider(_Provider):
    def __init__(self, hits: dict[str, object]) -> None:
        super().__init__()
        self._vectors = {}
        for index, query in enumerate(hits):
            first = 0.6 + index * 0.01
            norm = math.sqrt(first * first + 0.8 * 0.8)
            self._vectors[query] = (first / norm, 0.8 / norm)
        self._query_by_vector = {
            vector: query for query, vector in self._vectors.items()
        }

    async def embed_query(self, text: str) -> tuple[float, ...]:
        self.queries.append(text)
        return self._vectors[text]


class _QualityVectorStore:
    def __init__(self, provider: _QualityProvider, hits: dict[str, object]) -> None:
        self._hits = {
            vector: hits[query]
            for vector, query in provider._query_by_vector.items()
        }

    async def search(self, plan, query_embedding):
        del plan
        return VectorSearchResult(
            REVISION_ID,
            (self._hits[query_embedding],),
        )


class _QualityLexicalStore:
    async def search(self, plan, query, query_embedding, **kwargs):
        del plan, query, query_embedding, kwargs
        return LexicalSearchResult(
            REVISION_ID,
            analyzer_version=LEXICAL_ANALYZER_VERSION,
            manifest_target_count=1,
            hits=(),
        )


class _QualityGraphStore:
    def __init__(self, cases: tuple[_HardCase, ...]) -> None:
        self._cases = {case.query: case for case in cases}
        self._by_key = {case.entry_key: case for case in cases}

    async def get_config(self, workspace_id, knowledge_base_id):
        assert workspace_id == WORKSPACE
        assert knowledge_base_id == KB_ID
        return GraphConfigSnapshot(
            workspace_id=WORKSPACE,
            knowledge_base_id=KB_ID,
            status=GraphConfigStatus.READY,
            build_id=BUILD_ID,
            chat_profile_revision_id=UUID("01900000-0000-7000-8000-000000000955"),
            extractor_version=GRAPH_EXTRACTOR_VERSION,
            preflight_extractor_version=GRAPH_EXTRACTOR_VERSION,
            last_error_code=None,
            eligible_chunk_count=6,
            processed_chunk_count=6,
            extracted_chunk_count=6,
        )

    async def find_entity_candidates(self, query):
        for case in self._by_key.values():
            if query.normalized_surface == case.query:
                return (
                    GraphEntityCandidate(
                        entity_key=case.entry_key,
                        entity_type="organization",
                        surface=case.query,
                        normalized_surface=case.query,
                        index_chunk_id=case.seed_chunk_id,
                        indexed_document_version_id=TARGET_ID,
                        index_revision_id=REVISION_ID,
                    ),
                )
        return ()

    async def traverse(self, query):
        for case in self._by_key.values():
            if case.entry_key in query.entry_entity_keys:
                return case.traversal
        return GraphTraversalResult(resolved_active_revision_id=REVISION_ID)


def _complete_path_recall(packs, cases: tuple[_HardCase, ...]) -> float:
    complete = 0
    for pack, case in zip(packs, cases):
        if case.gold_chunk_ids.issubset({item.index_chunk_id for item in pack.evidence}):
            complete += 1
    return complete / len(cases)


def _evidence_coverage(packs, cases: tuple[_HardCase, ...]) -> float:
    coverage = 0.0
    for pack, case in zip(packs, cases):
        coverage += len(
            case.gold_chunk_ids & {item.index_chunk_id for item in pack.evidence}
        ) / len(case.gold_chunk_ids)
    return coverage / len(cases)


def _hard_cases() -> tuple[_HardCase, ...]:
    one_hop = _path_case(
        "atlas",
        CHUNK_1,
        (CHUNK_2,),
        relation_ids=(UUID("01900000-0000-7000-8000-000000000961"),),
        endpoints=(("atlas", "orion"),),
    )
    two_hop = _path_case(
        "mercury",
        UUID("01900000-0000-7000-8000-000000000962"),
        (
            UUID("01900000-0000-7000-8000-000000000963"),
            UUID("01900000-0000-7000-8000-000000000964"),
        ),
        relation_ids=(
            UUID("01900000-0000-7000-8000-000000000965"),
            UUID("01900000-0000-7000-8000-000000000966"),
        ),
        endpoints=(("mercury", "project-x"), ("project-x", "orion")),
    )
    cross_document = _path_case(
        "apollo",
        UUID("01900000-0000-7000-8000-000000000967"),
        (UUID("01900000-0000-7000-8000-000000000968"),),
        relation_ids=(UUID("01900000-0000-7000-8000-000000000969"),),
        endpoints=(("apollo", "orion"),),
    )
    return one_hop, two_hop, cross_document


def _path_case(
    entry_key: str,
    seed_chunk_id: UUID,
    path_chunk_ids: tuple[UUID, ...],
    *,
    relation_ids: tuple[UUID, ...],
    endpoints: tuple[tuple[str, str], ...],
) -> _HardCase:
    hops = tuple(
        GraphPathHop(
            subject_entity_key=subject,
            object_entity_key=object_,
            predicate="relates-to",
            normalized_predicate="relates-to",
            relation_id=relation_id,
            source_chunk_id=chunk_id,
            source_index_revision_id=REVISION_ID,
            source_location={"paragraph": index + 1},
        )
        for index, (relation_id, chunk_id, (subject, object_)) in enumerate(
            zip(relation_ids, path_chunk_ids, endpoints)
        )
    )
    path = GraphPathCandidate(
        path_id=f"quality-{entry_key}",
        entry_entity_key=entry_key,
        hops=hops,
        anchor_chunk_id=seed_chunk_id,
        rank=1,
        seed_entry=True,
    )
    chunks = tuple(
        _graph_chunk(chunk_id)
        for chunk_id in (seed_chunk_id, *path_chunk_ids)
    )
    return _HardCase(
        query=entry_key,
        entry_key=entry_key,
        seed_chunk_id=seed_chunk_id,
        path_chunk_ids=path_chunk_ids,
        traversal=GraphTraversalResult(
            resolved_active_revision_id=REVISION_ID,
            paths=(path,),
            chunks=chunks,
        ),
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
        text=f"grounded graph fact {chunk_id}",
        source_location={"paragraph": 1},
        hierarchy={},
        source_metadata={},
        modality="text",
        evidence_group_key=None,
        document_display_name="quality fixture",
        document_original_filename="quality.txt",
    )


if __name__ == "__main__":
    unittest.main()
