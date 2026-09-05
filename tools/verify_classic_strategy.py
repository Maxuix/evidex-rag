#!/usr/bin/env python3
"""Replay baseline/selected source pools through actual RetrievalService normalization.

Source data/distances are real frozen inputs; scope/provider fixtures are local.
This checks evidence projection, scope validation and ranking, not live DB I/O,
Agent reasoning, embedding calls, or table-neighbor discovery.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

from rag_kb.domain import RerankMode, VectorSearchResult
from rag_kb.retrieval.profile import exact_profile
import rag_kb.retrieval.reranker as runtime_reranker
from rag_kb.retrieval.service import RetrievalService
from tests.unit.test_retrieval_service import WORKSPACE, REVISION_ID, _Provider, _Store, _hit, _plan
from tools.classic_strategy import POLICIES
from tools.evaluate_classic_strategy import external_records, financial_records, validate
from tools.prepare_auto_strategy import read, require, sha, write


async def run(args):
    validate(args)
    choice = read(args.output/'selection.json')
    policy = next(p for p in POLICIES if p.name == choice['selected'])
    require(policy.lexical == 'classic' and policy.depth == 40 and policy.mmr == 1
            and not policy.append and not policy.protect, 'Service replay supports only the selected weight change')
    dev, fresh = read(args.output/'development.json'), read(args.output/'validation.json')
    expected = {name: {r['case_id']: r['ids'] for source in (dev, fresh) for r in source['arms'][name]}
                for name in ('classic10', policy.name)}
    plan = replace(_plan(), top_k=10, candidate_count=40, rerank_mode=RerankMode.CLASSIC)
    provider = _Provider()
    service = RetrievalService(WORKSPACE, provider, _Store(None))
    records = []
    with patch.object(runtime_reranker, '_terms', lru_cache(maxsize=20000)(runtime_reranker._terms)):
        for dataset in (financial_records(args), external_records(args, 'hotpot'),
                        external_records(args, 'musique'), external_records(args, 'hotpot', True)):
            for case, hits, core, extras, _, _ in dataset:
                by_id = {str(h.index_chunk_id): h for h in hits}
                def project(i):
                    h = by_id[i]
                    return replace(_hit(h.index_chunk_id, distance=h.cosine_distance), text=h.text,
                        hierarchy=h.hierarchy, source_location=h.source_location, source_metadata=h.source_metadata,
                        modality=h.modality)
                result = VectorSearchResult(REVISION_ID, tuple(project(i) for i in core))
                extra_hits = tuple(project(i) for i in extras)
                for name, weight in [('classic10', .65), (policy.name, policy.vector_weight)]:
                    profile = replace(exact_profile(top_k=10), rerank_vector_weight=weight, rerank_lexical_weight=1-weight)
                    evidence, _, _ = await service._normalize(plan, result, query=case['query'],
                        profile=profile, source_context_hits=extra_hits)
                    actual = [str(e.index_chunk_id) for e in evidence]
                    require(actual == expected[name][case['case_id']], 'Service ordering differs from evaluator')
                    require(all(e.text == by_id[str(e.index_chunk_id)].text
                        and e.source_metadata == by_id[str(e.index_chunk_id)].source_metadata
                        and e.source_location == by_id[str(e.index_chunk_id)].source_location
                        and e.vector_similarity == 1-by_id[str(e.index_chunk_id)].cosine_distance
                        for e in evidence), 'Service evidence projection changed source values')
                    records.append({'case_id': case['case_id'], 'policy': name, 'ids': actual})
                if len(records) % 100 == 0:
                    print({'verified_service_rankings': len(records)}, flush=True)
    require(not provider.queries, 'Service replay unexpectedly called an embedding provider')
    write(args.output/'service-verification.json', {'status': 'complete', 'rankings': len(records),
        'records': records, 'model_calls': 0, 'database_access': False,
        'development_sha256': sha(args.output/'development.json'),
        'validation_sha256': sha(args.output/'validation.json'),
        'service_sha256': sha(Path(__file__).resolve().parents[1]/'src/rag_kb/retrieval/service.py'),
        'verifier_sha256': sha(Path(__file__))})
    print({'verified_service_rankings': len(records), 'model_calls': 0}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
