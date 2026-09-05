from __future__ import annotations

import unittest
from dataclasses import replace
from uuid import UUID

from rag_kb.domain import RetrievalExecutionError, RetrievalRequest, RerankMode, VectorSearchResult
from rag_kb.retrieval.reranker import score_hits
from rag_kb.retrieval.service import RetrievalService
from rag_kb.retrieval.source_context import rank_with_source_context, table_neighbor_compatible
from tests.unit.test_retrieval_service import WORKSPACE, KB_ID, REVISION_ID, _Provider, _Store, _hit


def source(number, *, text='source body', distance=.2, ordinal=None, modality='text'):
    return replace(_hit(UUID(int=number), distance=distance, ordinal=number if ordinal is None else ordinal),
                   text=text, modality=modality,
                   source_location={'surface_type': 'page', 'surface_start': 1}, hierarchy={})


class SourceContextRankingTests(unittest.TestCase):
    def test_fixed_statistics_preserve_every_core_score(self):
        core = (source(1, text='cash assets revenue'), source(2, text='cash flow'),
                source(3, text='capital and equity income'))
        extras = tuple(source(i, text='cash '*i) for i in range(4, 24))
        before = score_hits('cash assets', core)
        after = score_hits('cash assets', (*core, *extras), reference_hits=iter(core))
        self.assertEqual(after[:len(core)], before)
        self.assertNotEqual(score_hits('cash assets', (*core, *extras))[:len(core)], before)
        with self.assertRaises(ValueError):
            score_hits('cash', extras, reference_hits=())

    def test_prefix_is_protected_and_supplement_keeps_own_evidence_and_distance(self):
        core = tuple(source(i, text='ordinary revenue', distance=.1+i*.03) for i in range(1, 9))
        extra = source(100, text='cash assets balance sheet', distance=.01, modality='table')
        baseline = rank_with_source_context('cash assets', core, (), top_k=6)
        actual = rank_with_source_context('cash assets', core, (extra,), top_k=6)
        self.assertEqual(actual[:3], baseline[:3])
        self.assertEqual(actual[3].hit, extra)
        self.assertAlmostEqual(actual[3].vector_similarity, .99)
        self.assertEqual(len({s.hit.index_chunk_id for s in actual}), 6)

    def test_empty_and_duplicate_pools_are_checked(self):
        self.assertEqual(rank_with_source_context('q', (), (), top_k=3), ())
        for core, extra, k in (((), (source(1),), 3), ((source(1),), (source(1),), 3),
                               ((source(1),), (), 0),
                               ((source(1),), tuple(source(i) for i in range(2, 203)), 3)):
            with self.subTest(k=k, size=len(extra)), self.assertRaises(ValueError):
                rank_with_source_context('q', core, extra, top_k=k)

    def test_table_neighbor_respects_document_position_page_and_section(self):
        anchor = source(1, ordinal=10)
        neighbor = source(2, ordinal=11, modality='table')
        self.assertTrue(table_neighbor_compatible(anchor, neighbor))
        for changed in (
            replace(neighbor, indexed_document_version_id=UUID(int=999)),
            replace(neighbor, ordinal=12), replace(neighbor, modality='text'),
            replace(neighbor, source_location={'surface_type': 'page', 'surface_start': 3}),
            replace(neighbor, source_location={'surface_type': 'page', 'surface_start': True}),
            replace(neighbor, source_location={'surface_type': 'line', 'surface_start': 1}),
        ):
            with self.subTest(neighbor=changed):
                self.assertFalse(table_neighbor_compatible(anchor, changed))
        heading = {'titles': [{'text': 'Cash flows'}]}
        self.assertTrue(table_neighbor_compatible(replace(anchor, hierarchy=heading), neighbor))
        self.assertFalse(table_neighbor_compatible(replace(anchor, hierarchy=heading),
            replace(neighbor, hierarchy={'titles': [{'text': 'Equity'}]})))


class SourceContextServiceTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self):
        core = tuple(source(i, ordinal=i*10, distance=.1+i*.01) for i in range(1, 21))
        anchor = source(30, ordinal=300, distance=.4)
        table = source(31, ordinal=301, distance=.45, text='cash assets', modality='table')
        other_text = source(32, ordinal=299, distance=.42, text='cash assets')
        result = VectorSearchResult(REVISION_ID, core)

        class Store(_Store):
            wide = replace(result, hits=(*core, anchor))
            neighbors = replace(result, hits=(table, other_text))
            lookups = []

            async def search(self, plan, query_embedding):
                await super().search(plan, query_embedding)
                return self.wide if plan.candidate_count == 100 else self.result

            async def source_neighbors(self, plan, query_embedding, **kwargs):
                self.lookups.append(kwargs)
                return self.neighbors

        store = Store(result)
        provider = _Provider()
        return store, provider, RetrievalService(WORKSPACE, provider, store), table

    async def test_lookahead_supplies_only_table_neighbors_with_one_embedding(self):
        store, provider, service, table = self.fixture()
        pack = await service.retrieve(RetrievalRequest(KB_ID, 'cash assets', top_k=5,
            rerank_mode=RerankMode.CLASSIC, include_debug=True))
        self.assertEqual(provider.queries, ['cash assets'])
        self.assertEqual([p.candidate_count for p in store.plans], [20, 100])
        self.assertEqual(store.plans[1].auto_qa_candidate_count, 0)
        self.assertFalse(store.plans[1].allow_unverified_auto_qa)
        self.assertEqual(pack.debug.source_context_candidate_count, 1)
        self.assertIn(table.index_chunk_id, [e.index_chunk_id for e in pack.evidence])
        self.assertEqual(store.lookups[0]['index_revision_id'], REVISION_ID)
        self.assertEqual(len(store.lookups[0]['anchors']), 21)

    async def test_scope_drift_duplicates_and_nonadjacent_rows_fail_closed(self):
        for fault in ('changed_core', 'revision', 'duplicate_wide', 'foreign', 'duplicate_neighbor',
                      'nonadjacent', 'fingerprint'):
            store, _, service, table = self.fixture()
            if fault == 'changed_core':
                store.wide = replace(store.wide, hits=(replace(store.wide.hits[0], cosine_distance=.11+1e-4), *store.wide.hits[1:]))
            elif fault == 'revision':
                store.wide = replace(store.wide, resolved_active_revision_id=UUID(int=999))
            elif fault == 'duplicate_wide':
                store.wide = replace(store.wide, hits=(*store.wide.hits, store.wide.hits[-1]))
            elif fault == 'foreign':
                store.neighbors = replace(store.neighbors, hits=(replace(table, knowledge_base_id=UUID(int=999)),))
            elif fault == 'duplicate_neighbor':
                store.neighbors = replace(store.neighbors, hits=(table, table))
            elif fault == 'nonadjacent':
                store.neighbors = replace(store.neighbors, hits=(replace(table, ordinal=500),))
            else:
                store.neighbors = replace(store.neighbors, compatibility_fingerprint='sha256:changed')
            with self.subTest(fault=fault), self.assertRaises(RetrievalExecutionError):
                await service.retrieve(RetrievalRequest(KB_ID, 'cash assets', top_k=5, rerank_mode=RerankMode.CLASSIC))

    async def test_no_reranker_does_not_activate_context_policy(self):
        store, _, service, _ = self.fixture()
        # NONE accepts only top_k raw hits from the adapter.
        store.result = replace(store.result, hits=store.result.hits[:5])
        await service.retrieve(RetrievalRequest(KB_ID, 'cash assets', top_k=5))
        self.assertEqual(len(store.plans), 1)
        self.assertEqual(store.lookups, [])
