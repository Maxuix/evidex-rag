#!/usr/bin/env python3
"""Fixed Classic/MiniLM collaboration comparisons over cached and external corpora."""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

import numpy as np
from rag_kb.tokenizer import get_cl100k_base_encoding

from rag_kb.adapters.local_reranker import _sigmoid
import rag_kb.retrieval.reranker as runtime_reranker
from rag_kb.retrieval.service import rerank_document_from_evidence
from tools.analyze_auto_qa_ranking import classic_order, rank_with_scores
from tools.auto_strategy_metrics import evidence_row, paired_changes, policies, summarize
from tools.evaluate_minilm_source import ReferenceScorer, verify_service_case
from tools.prepare_auto_strategy import ROOT, OUTPUT, read, write, require, sha, jsonl


def costs_for(documents):
    tokenizer = get_cl100k_base_encoding()
    return {i: max(1, len(tokenizer.encode(text, disallowed_special=()))) for i, text in documents.items()}


def record(arms, case, classic, raw, mmr, costs):
    for name, selected in policies(classic, raw, mmr, costs).items():
        arms[name].append(evidence_row(case, selected, classic, costs))


def finish(arms):
    return {name: {'summary': summarize(rows), 'cases': rows,
                  'vs_classic10': paired_changes(arms['classic10'], rows)} for name, rows in arms.items()}


def develop():
    from tools.analyze_source_recall import inputs, DIAGNOSIS
    from tools.evaluate_minilm_completed import OUTPUT as PRIOR, PROBES
    _, snapshot, _, _, definitions = inputs()
    prior = read(PRIOR/'results.json')
    require(prior['status'] == 'complete', 'Historical reranker cache incomplete')
    for name, expected in prior['input_sha256'].items():
        require(sha(ROOT/name) == expected, 'Historical input changed')
    cases = {c['evaluation_case_id']: c for c in definitions}
    source_cases = {c['case_id']: c for c in jsonl(ROOT/'evaluation/document-qa-v1/cases.jsonl')}
    paraphrases = {c['case_id']: c['base_case_id'] for c in jsonl(ROOT/'evaluation/document-qa-v1/auto-qa-paraphrases.jsonl')}
    base = {c['case_id']: c for c in prior['arms']['classic_protected']['cases']}
    mmrs = {c['case_id']: c for c in prior['arms']['minilm_completed_mmr']['cases']}
    raws = {c['case_id']: c for c in prior['arms']['minilm_completed_raw']['cases']}
    probes = {c['case_id']: [x['chunk_id'] for x in c['required']] for c in read(PROBES)['cases']}
    all_costs = costs_for({i: c['content'] for i, c in snapshot['chunks'].items()})
    arms = defaultdict(list)
    for d in prior['diagnostics']:
        cid = d['case_id']
        initial = [c['id'] for c in d['candidates']]
        raw = sorted(initial, key=lambda i: (-_sigmoid(max(next(c['logits'] for c in d['candidates'] if c['id'] == i))), initial.index(i)))
        require(raw[:10] == raws[cid]['final_ids'], 'Historical raw order not reproduced')
        classic = base[cid]['final_ids'] + [i for i in initial if i not in base[cid]['final_ids']]
        parent = paraphrases.get(cid, cid)
        family = source_cases[parent]['source_dataset']
        case = {'case_id': cid, 'family': family, 'group': family+':'+d['group'], 'cluster': parent,
            'query': d['query'], 'answerable': True, 'required_paths': [['historical_label']],
            'labels': {c['id']: ['historical_label'] if c['label'] else [] for c in d['candidates']},
            'probe_ids': probes.get(cid, [])}
        record(arms, case, classic, raw, mmrs[cid]['final_ids'], {i: all_costs[i] for i in initial})
    result = {'status': 'complete', 'stage': 'development', 'qa_generation_calls': 0, 'new_model_calls': 0,
        'input_sha256': {str((PRIOR/'results.json').relative_to(ROOT)): sha(PRIOR/'results.json'),
                         str((DIAGNOSIS/'sources.json').relative_to(ROOT)): sha(DIAGNOSIS/'sources.json')},
        'arms': finish(arms)}
    write(OUTPUT/'development.json', result)
    for name, arm in result['arms'].items():
        rows = arm['cases']
        print(name, {'any': sum(r['any_evidence'] for r in rows), 'hit1': sum(r['hit1'] for r in rows),
                     'hit10': sum(r['hit10'] for r in rows), 'mean_tokens': round(sum(r['tokens'] for r in rows)/len(rows)),
                     'complete_probes': sum(bool(r['probe_complete']) for r in rows)}, flush=True)


def new_hit(doc, distance):
    return SimpleNamespace(index_chunk_id=UUID(doc['id']), text=doc['text'], cosine_distance=distance,
        modality='text', hierarchy={'titles': [{'text': doc['title']}]}, source_location={},
        source_metadata={'original_filename': doc['filename'], 'document_id': doc['id'], 'document_version_id': doc['id']})


def external(family, cache_only=False):
    started = time.perf_counter()
    require((OUTPUT/'selection.json').exists(), 'Freeze development selection before external scoring')
    selection = read(OUTPUT/'selection.json')
    require(selection['development_sha256'] == sha(OUTPUT/'development.json'), 'Development selection changed')
    require(selection['policy_sha256'] == sha(ROOT/'tools/auto_strategy_metrics.py'), 'Frozen comparison policies changed')
    frozen_path, vectors_path = OUTPUT/(family+'-frozen.json'), OUTPUT/(family+'-vectors.npz')
    frozen, vector_binding = read(frozen_path), read(OUTPUT/(family+'-vectors.json'))
    require(vector_binding['input_sha256'] == sha(frozen_path) and vector_binding['vectors_sha256'] == sha(vectors_path), 'Frozen external vector inputs changed')
    data = np.load(vectors_path, allow_pickle=False)
    matrix, queries = data['documents'], data['queries']
    require(matrix.shape == (len(frozen['documents']), 1024) and queries.shape == (len(frozen['cases']), 1024), 'External vector shape differs')
    matrix = matrix/np.linalg.norm(matrix, axis=1, keepdims=True)
    queries = queries/np.linalg.norm(queries, axis=1, keepdims=True)
    similarities = queries@matrix.T
    scorer = ReferenceScorer(SimpleNamespace(backend='int8', output_root=OUTPUT, variant=family,
        reference=ROOT/'.runtime/model-assets/minilm-reference-1427fd6',
        tokenizer=ROOT/'.runtime/model-assets/local-reranker', cache_only=cache_only, window_batch_size=8))
    by_id = {d['id']: d for d in frozen['documents']}
    costs = costs_for({i: d['text'] for i, d in by_id.items()})
    contexts = {i: d['text'][:2048] for i, d in by_id.items()}
    arms, diagnostics, verified = defaultdict(list), [], []
    result = {'status': 'running', 'family': family, 'qa_generation_calls': 0,
        'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (frozen_path, vectors_path, OUTPUT/'selection.json')},
        'model_sha256': scorer.model_sha256, 'window_code_sha256': scorer.window_code}
    for index, (original_case, similarities_row) in enumerate(zip(frozen['cases'], similarities, strict=True), 1):
        order = sorted(range(len(by_id)), key=lambda n: (-similarities_row[n], UUID(frozen['documents'][n]['id']).int))
        indices = [n for n in order[:40] if similarities_row[n] >= .35]
        hits = [new_hit(frozen['documents'][n], float(1-similarities_row[n])) for n in indices]
        scored = classic_order(original_case['query'], hits)
        initial = [s.hit for s in scored]
        classic = [str(h.index_chunk_id) for h in initial]
        documents = tuple(rerank_document_from_evidence(h, contexts[str(h.index_chunk_id)]) for h in initial)
        scores = scorer.score(original_case['query'], documents) if documents else {}
        probabilities = {i: _sigmoid(max(v['logits'])) for i, v in scores.items()}
        raw = [str(h.index_chunk_id) for h in rank_with_scores(initial, probabilities, mmr=False)]
        mmr = [str(h.index_chunk_id) for h in rank_with_scores(initial, probabilities, mmr=True)]
        case = {**original_case, 'labels': {i: [by_id[i]['document_id']] for i in classic}}
        record(arms, case, classic, raw, mmr, {i: costs[i] for i in classic})
        if initial:
            with patch.object(runtime_reranker, '_text_similarity', lru_cache(maxsize=8192)(runtime_reranker._text_similarity)):
                asyncio.run(verify_service_case(case['query'], initial, documents, scores, contexts, mmr[:10]))
        verified.append(case['case_id'])
        required = set().union(*(set(p) for p in case['required_paths']))
        exact_ranks = {frozen['documents'][n]['document_id']: rank for rank, n in enumerate(order, 1)
                       if frozen['documents'][n]['document_id'] in required}
        diagnostics.append({'case_id': case['case_id'], 'candidate_ids': classic,
            'raw_order': raw, 'mmr_order': mmr, 'required_full_corpus_ranks': exact_ranks,
            'candidate_documents': {i: by_id[i]['document_id'] for i in classic},
            'scores': {str(i): value for i, value in scores.items()}})
        result.update(completed_cases=index, new_inference_pairs=scorer.new_pairs,
            new_inference_windows=scorer.new_windows, inference_seconds=scorer.seconds,
            runtime_verified_case_ids=verified, diagnostics=diagnostics, arms={k: {'cases': v} for k, v in arms.items()})
        write(OUTPUT/(family+'-results.json'), result)
        print({'family': family, 'case': index, 'total': len(frozen['cases']), 'candidates': len(classic),
               'new_pairs': scorer.new_pairs, 'inference_seconds': round(scorer.seconds, 2)}, flush=True)
    result.update(status='complete', arms=finish(arms), elapsed_seconds=time.perf_counter()-started,
        code_sha256={str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), ROOT/'tools/auto_strategy_metrics.py',
            ROOT/'tools/evaluate_minilm_source.py', ROOT/'src/rag_kb/retrieval/reranker.py', ROOT/'src/rag_kb/retrieval/service.py')})
    target = OUTPUT/(family+'-results.json')
    write(target, result)
    if cache_only:
        live = read(OUTPUT/(family+'-inference.json'))
        require(result['arms'] == live['arms'] and result['diagnostics'] == live['diagnostics'] and
                result['new_inference_pairs'] == result['new_inference_windows'] == 0, 'Cache-only reproduction differs')
        write(OUTPUT/(family+'-verification.json'), {'status': 'complete', 'rankings': len(arms)*len(verified),
            'runtime_verified': len(verified), 'new_inference_pairs': 0, 'results_sha256': sha(target),
            'live_sha256': sha(OUTPUT/(family+'-inference.json'))})
    elif not (OUTPUT/(family+'-inference.json')).exists():
        write(OUTPUT/(family+'-inference.json'), result)
    print({name: arm['summary'] for name, arm in result['arms'].items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('develop', 'hotpot', 'musique'))
    parser.add_argument('--cache-only', action='store_true')
    args = parser.parse_args()
    if args.stage == 'develop':
        develop()
    else:
        external(args.stage, args.cache_only)
