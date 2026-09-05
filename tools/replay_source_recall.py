#!/usr/bin/env python3
"""Replay the real source-only retrieval service against frozen query vectors, read-only."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from contextlib import asynccontextmanager
from urllib.parse import urlsplit
from uuid import UUID

from sqlalchemy import text

from rag_kb.adapters.vector_store.pgvector import PgVectorStore
from rag_kb.config import load_settings
from rag_kb.db import DatabaseProcess, create_database_resources
from rag_kb.domain import RetrievalQueryPlan, RetrievalRequest, RetrievalStrategy, RerankMode
from rag_kb.retrieval.service import RetrievalService
from tools.analyze_source_recall import ROOT, inputs, read, write, require, digest, STATE, REPLAY, OUTPUT as RANK_ROOT
from tools.evaluate_source_recall import OUTPUT
from tools.evaluation_runtime import load_evaluation_runtime


async def replay():
    from apps.api.dependencies import build_api_dependencies

    corpus, _, baseline, _, cases = inputs()
    runtime = load_evaluation_runtime()
    settings = load_settings(env_file=runtime.env_file)
    url = urlsplit(settings.database.runtime_dsn.get_secret_value())
    require(url.hostname in {'127.0.0.1', 'localhost'} and url.port == runtime.ports['postgres'],
            'Only the isolated host evaluation database is allowed')
    state = read(STATE)
    require(state['status'] == 'indexed' and state['corpus_sha256'] == corpus['dataset_sha256'], 'State changed')
    arm = state['arms']['on']
    meta = read(RANK_ROOT/'exact-ranks.json')
    expected = {r['case_id']: r for r in read(OUTPUT/'comparison.json')['arms']['protected_table_context']['cases']}
    expected_baseline = {r['case_id']: r for r in baseline['arms']['classic_source']['cases']}
    query_cache = read(REPLAY/'query-vectors.json')
    deps = build_api_dependencies(settings=settings)
    try:
        plan = RetrievalQueryPlan(settings.identity.workspace_id, UUID(arm['knowledge_base_id']),
            RetrievalStrategy.EXACT_VECTOR, top_k=10, candidate_count=40,
            rerank_mode=RerankMode.CLASSIC, auto_qa_candidate_count=0)
        configured = await deps.retrieval_service._text_embedding_provider(plan)
        space = configured.embedding_space
        require(space.requested_model == 'qwen3.7-text-embedding' and
                space.compatibility_fingerprint == meta['compatibility_fingerprint'], 'Embedding space changed')
    finally:
        await deps.close()

    class CachedProvider:
        embedding_space = space
        max_batch_size = 10
        calls = 0

        async def embed_query(self, query):
            key = hashlib.sha256((space.compatibility_fingerprint+'\n'+query).encode()).hexdigest()
            require(key in query_cache, 'Cached query missing; no model fallback')
            self.calls += 1
            return tuple(query_cache[key])

    class CapturedService(RetrievalService):
        candidates = ()

        async def _source_context_candidates(self, plan, result, *args):
            additions = await super()._source_context_candidates(plan, result, *args)
            self.candidates = (*result.hits, *additions)
            return additions

    class BaselineService(RetrievalService):
        async def _source_context_candidates(self, *args):
            return ()

    records = []
    async with create_database_resources(settings.database.runtime_dsn.get_secret_value(),
            pool_size=1, max_overflow=0, process=DatabaseProcess.MAINTENANCE) as db:
        @asynccontextmanager
        async def readonly_session():
            async with db.sessions() as session, session.begin():
                await session.execute(text('SET TRANSACTION READ ONLY'))
                yield session

        provider = CachedProvider()
        store = PgVectorStore(readonly_session, space)
        service = CapturedService(settings.identity.workspace_id, provider, store)
        baseline_service = BaselineService(settings.identity.workspace_id, provider, store)
        for position, case in enumerate(cases, 1):
            request = RetrievalRequest(plan.knowledge_base_id, case['question'], top_k=10,
                                       rerank_mode=RerankMode.CLASSIC, include_debug=True)
            start = time.monotonic()
            previous = await baseline_service.retrieve(request)
            baseline_ms = (time.monotonic()-start)*1000
            start = time.monotonic()
            pack = await service.retrieve(request)
            cid = case['evaluation_case_id']
            ids = [str(e.index_chunk_id) for e in pack.evidence]
            candidates = {str(h.index_chunk_id) for h in service.candidates}
            require(pack.index_revision_id == UUID(arm['index_revision_id']), 'Active revision changed')
            require([str(e.index_chunk_id) for e in previous.evidence] == expected_baseline[cid]['final_ids'],
                    f'Real baseline rank drift: {cid}')
            require(ids == expected[cid]['final_ids'], f'Real service rank drift: {cid}')
            require(candidates == set(expected[cid]['candidate_ids']), f'Real service candidate drift: {cid}')
            require(all(h.source_candidate and not h.matched_question for h in service.candidates), 'Unexpected QA')
            records.append({'case_id': cid, 'final_ids': ids, 'candidate_ids': sorted(candidates),
                'candidate_count': len(candidates),
                'source_context_candidate_count': pack.debug.source_context_candidate_count,
                'baseline_elapsed_ms': round(baseline_ms, 2),
                'elapsed_ms': round((time.monotonic()-start)*1000, 2)})
            if position % 10 == 0 or position == len(cases):
                print(json.dumps({'service_queries': position, 'total': len(cases)}), flush=True)
        require(provider.calls == 2*len(cases), 'Unexpected query-embedding lookup count')
    result = {'status': 'complete', 'cases': records, 'model_calls': 0, 'qa_generation_calls': 0,
              'database_writes': 0, 'query_cache_lookups': provider.calls,
              'corpus_sha256': corpus['dataset_sha256'], 'comparison_sha256': digest(OUTPUT/'comparison.json'),
              'query_cache_sha256': digest(REPLAY/'query-vectors.json'),
              'code_sha256': {p: digest(ROOT/p) for p in (
                  'src/rag_kb/retrieval/source_context.py', 'src/rag_kb/retrieval/service.py',
                  'src/rag_kb/retrieval/reranker.py', 'src/rag_kb/adapters/vector_store/pgvector.py',
                  'tools/evaluate_source_recall.py', 'tools/replay_source_recall.py')}}
    write(OUTPUT/'service-replay.json', result)
    print(json.dumps({'status': 'complete', 'cases': len(records), 'model_calls': 0}), flush=True)


if __name__ == '__main__':
    asyncio.run(replay())
