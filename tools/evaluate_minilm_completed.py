#!/usr/bin/env python3
"""Compare frozen Classic and deployed MiniLM on identical completed source pools.

Uses only existing questions/source snapshots and pinned local model assets. There
is no database, provider, QA generation, model download or production mutation.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from contextlib import nullcontext
from functools import lru_cache
import json
import re
import time
from types import SimpleNamespace
from pathlib import Path
from uuid import UUID
from unittest.mock import patch

from rag_kb.adapters.local_reranker import _sigmoid
import rag_kb.retrieval.reranker as runtime_reranker
from rag_kb.retrieval.reranker import score_hits
from rag_kb.retrieval.source_context import rank_with_source_context
from tools.analyze_auto_qa_ranking import classic_order, rank_with_scores, row_result, summarize
from tools.analyze_source_recall import ROOT, DIAGNOSIS, inputs, hit, read, digest, write, require
from tools.evaluate_minilm_source import ReferenceScorer, project_document, verify_service_case
from tools.evaluate_source_recall import OUTPUT as RECALL_ROOT, PROBES

OUTPUT = ROOT / '.runtime/evaluations/minilm-completed-20260905'
OLD_MODEL_ROOT = ROOT / '.runtime/evaluations/minilm-repair-20260905'
RANK_ROOT = ROOT / '.runtime/evaluations/source-recall-20260905'
METRICS = ('hit_1', 'hit_5', 'hit_10', 'mrr_10')


def protected_order(baseline, proposed, top_k=10):
    """Diagnostic control: keep the same Classic prefix, then use model order."""
    prefix = list(baseline[:(top_k+1)//2])
    protected_ids = {h.index_chunk_id for h in prefix}
    return (prefix + [h for h in proposed if h.index_chunk_id not in protected_ids])[:top_k]


def compare_rows(baseline, rows):
    old = {r['case_id']: r for r in baseline}
    require(len(old) == len(baseline) == len(rows) and
            set(old) == {r['case_id'] for r in rows}, 'Comparison case identity differs')
    groups = {}
    for group in ('direct', 'paraphrase'):
        selected = [r for r in rows if r['group'] == group]
        result = {}
        for metric in ('hit_1', 'recall_10'):
            result['lost_'+metric] = [r['case_id'] for r in selected if old[r['case_id']][metric] and not r[metric]]
            result['gained_'+metric] = [r['case_id'] for r in selected if r[metric] and not old[r['case_id']][metric]]
        result['rank_improved'] = [r['case_id'] for r in selected
            if (r['relevant_rank'] or 11) < (old[r['case_id']]['relevant_rank'] or 11)]
        result['rank_worsened'] = [r['case_id'] for r in selected
            if (r['relevant_rank'] or 11) > (old[r['case_id']]['relevant_rank'] or 11)]
        result['rank_unchanged'] = sum(r['relevant_rank'] == old[r['case_id']]['relevant_rank'] for r in selected)
        groups[group] = result
    return groups


def probe_results(rows, probes):
    results = []
    for row in rows:
        required = probes.get(row['case_id'])
        if required is None:
            continue
        found = required & set(row['final_ids'])
        results.append({'case_id': row['case_id'], 'required': len(required),
            'found': len(found), 'complete': found == required,
            'found_ids': sorted(found), 'missing_ids': sorted(required-found)})
    return results


def replacement_gate(baseline, rows, probes):
    changes = compare_rows(baseline, rows)
    old_summary, new_summary = summarize(baseline), summarize(rows)
    regressions = [f'{g}.{metric}' for g in ('direct', 'paraphrase') for metric in METRICS
                   if new_summary[g][metric] < old_summary[g][metric]]
    old_probes = {r['case_id']: r for r in probe_results(baseline, probes)}
    lost_complete = [r['case_id'] for r in probe_results(rows, probes)
                     if old_probes[r['case_id']]['complete'] and not r['complete']]
    passed = not regressions and not lost_complete and all(
        not changes[g]['lost_hit_1'] and not changes[g]['lost_recall_10']
        for g in ('direct', 'paraphrase'))
    return {'passed': passed, 'decreased_metrics': regressions, 'lost_complete_probes': lost_complete,
            'scope': 'historical development regression gate; not unseen-data or answer-quality proof'}


def frozen_cases():
    corpus, snapshot, _, _, cases = inputs()
    comparison = read(RECALL_ROOT/'comparison.json')
    require(comparison['status'] == 'complete' and comparison['source_only'] and
            comparison['corpus_sha256'] == corpus['dataset_sha256'], 'Completed source pool changed')
    pools = {r['case_id']: r for r in comparison['arms']['protected_table_context']['cases']}
    cores = {r['case_id']: r for r in comparison['arms']['dense40']['cases']}
    ranks = {r['case_id']: dict(r['ranks']) for r in read(RANK_ROOT/'exact-ranks.json')['cases']}
    runtime = read(RECALL_ROOT/'service-replay.json')
    require(runtime['status'] == 'complete' and runtime['comparison_sha256'] == digest(RECALL_ROOT/'comparison.json'),
            'Recall runtime verification changed')
    for record in runtime['cases']:
        require(set(record['candidate_ids']) == set(pools[record['case_id']]['candidate_ids']) and
                record['final_ids'] == pools[record['case_id']]['final_ids'], 'Unverified candidate pool')
    require(len(pools) == len(cores) == len(runtime['cases']) == len(cases) == 85, 'Incomplete fixed pool')
    chunks = snapshot['chunks']
    contexts = {}
    for chunk in sorted(chunks.values(), key=lambda c: c['ordinal']):
        if chunk['modality'] in {'text', 'table'} and chunk['content'].strip():
            contexts.setdefault(chunk['source_metadata']['document_id'], chunk['content'][:2048])
    probes = {}
    for probe in read(PROBES)['cases']:
        probes[probe['case_id']] = {r['chunk_id'] for r in probe['required']}
        for item in probe['required']:
            require(re.sub(r'\s+', ' ', item['quote']) in
                    re.sub(r'\s+', ' ', chunks[item['chunk_id']]['content']), 'Probe source quote changed')
    prepared = []
    for case in cases:
        cid = case['evaluation_case_id']
        pool_ids, core_ids = pools[cid]['candidate_ids'], cores[cid]['candidate_ids']
        require(len(set(pool_ids)) == len(pool_ids) <= 240 and set(core_ids) <= set(pool_ids), 'Invalid expansion')
        core = tuple(hit(i, ranks[cid][i], chunks, case) for i in core_ids)
        all_hits = tuple(hit(i, ranks[cid][i], chunks, case) for i in pool_ids)
        require(all(h.modality in {'text', 'table'} for h in all_hits), 'Unexpected visual candidate')
        extras = tuple(h for h in all_hits if str(h.index_chunk_id) not in set(core_ids))
        original = [s.hit for s in classic_order(case['question'], core)]
        protected = [s.hit for s in rank_with_source_context(case['question'], core, extras, top_k=10)]
        require(row_result(case, original)['final_ids'] == cores[cid]['final_ids'], 'Original Classic drift')
        require(row_result(case, protected)['final_ids'] == pools[cid]['final_ids'], 'Protected Classic drift')
        # The model receives the whole expanded pool, in the same deterministic
        # score/tie order as the Classic control, never only Classic's final ten.
        scored = sorted(score_hits(case['question'], all_hits, reference_hits=core),
                        key=lambda s: (-s.score, s.hit.cosine_distance, s.hit.index_chunk_id.int))
        initial = [s.hit for s in scored]
        prepared.append((case, original, protected, scored, initial))
    return corpus, contexts, probes, prepared


def scorer_arguments(output_root, *, cache_only, variant):
    return SimpleNamespace(backend='int8', variant=variant, output_root=output_root, cache_only=cache_only,
        reference=ROOT/'.runtime/model-assets/minilm-reference-1427fd6',
        tokenizer=ROOT/'.runtime/model-assets/local-reranker', window_batch_size=8)


def run(args):
    started = time.perf_counter()
    corpus, contexts, probes, prepared = frozen_cases()
    old_scorer = ReferenceScorer(scorer_arguments(OLD_MODEL_ROOT, cache_only=True, variant='repaired'))
    old_rows = read(OLD_MODEL_ROOT/'repaired-int8.json')['arms']
    old_expected = {name: {r['case_id']: r for r in old_rows[name]['cases']} for name in ('max_raw', 'max_mmr')}
    arms = defaultdict(list)
    for case, original, protected, scored, initial in prepared:
        docs = tuple(project_document(h, contexts, 'repaired') for h in original)
        scores = old_scorer.score(case['question'], docs)
        probabilities = {i: _sigmoid(max(v['logits'])) for i, v in scores.items()}
        for mmr in (False, True):
            name = 'max_mmr' if mmr else 'max_raw'
            row = row_result(case, rank_with_scores(original, probabilities, mmr=mmr))
            require(row['final_ids'] == old_expected[name][case['evaluation_case_id']]['final_ids'], 'Old MiniLM cache drift')
            arms['minilm_core_'+('mmr' if mmr else 'raw')].append(row)
        arms['classic_core'].append(row_result(case, original))
        arms['classic_completed'].append(row_result(case, initial))
        arms['classic_protected'].append(row_result(case, protected))
    require(old_scorer.new_pairs == 0, 'Original pool unexpectedly ran inference')
    preflight = {'status': 'complete', 'cases': len(prepared), 'candidate_pairs': sum(len(p[-1]) for p in prepared),
        'max_candidates': max(len(p[-1]) for p in prepared), 'core_cache_verified_rankings': len(prepared)*2,
        'new_inference_pairs': old_scorer.new_pairs, 'qa_generation_calls': 0,
        'model_sha256': old_scorer.model_sha256, 'window_code_sha256': old_scorer.window_code}
    write(OUTPUT/'preflight.json', preflight)
    print(json.dumps({'preflight': preflight}), flush=True)
    if args.preflight_only:
        return

    scorer = ReferenceScorer(scorer_arguments(OUTPUT, cache_only=args.cache_only, variant='completed'))
    if not args.cache_only and not scorer.cache_path.exists():
        # Keys bind the complete ordered request; only exactly unchanged requests
        # can reuse original INT8 batches. New pools cannot reuse partial scores.
        scorer.cache.update(old_scorer.cache)
    diagnostics = []
    result = {'status': 'running', 'schema_version': 'minilm_completed_v1', 'backend': 'int8',
        'source_only': True, 'qa_generation_calls': 0, 'remote_model_calls': 0, 'database_access': False,
        'corpus_sha256': corpus['dataset_sha256'], 'model_sha256': scorer.model_sha256,
        'window_code_sha256': scorer.window_code, 'completed_cases': 0,
        'input_sha256': {str(path.relative_to(ROOT)): digest(path) for path in (
            RECALL_ROOT/'comparison.json', RECALL_ROOT/'service-replay.json',
            DIAGNOSIS/'sources.json', RANK_ROOT/'exact-ranks.json', PROBES)},
        'service_verified_case_ids': [], 'arms': {}, 'diagnostics': diagnostics}
    for position, (case, original, protected, scored, initial) in enumerate(prepared, 1):
        query, cid = case['question'], case['evaluation_case_id']
        docs = tuple(project_document(h, contexts, 'repaired') for h in initial)
        require(len({d.index_chunk_id for d in docs}) == len(initial), 'Duplicate model inputs')
        scores = scorer.score(query, docs)
        probabilities = {i: _sigmoid(max(v['logits'])) for i, v in scores.items()}
        model_raw = rank_with_scores(initial, probabilities, mmr=False)
        model_mmr = rank_with_scores(initial, probabilities, mmr=True)
        arms['minilm_completed_raw'].append(row_result(case, model_raw))
        arms['minilm_completed_mmr'].append(row_result(case, model_mmr))
        arms['minilm_protected_raw'].append(row_result(case, protected_order(original, model_raw)))
        for weight in (.1, .2):
            fused = {s.hit.index_chunk_id: (1-weight)*s.score + weight*probabilities[s.hit.index_chunk_id] for s in scored}
            arms[f'fusion_{int(weight*100)}'].append(row_result(case, rank_with_scores(initial, fused, mmr=False)))
        # Live inference exercises the unchanged runtime implementation. During
        # cache-only replay, memoize its pure text similarity to avoid repeating
        # expensive tokenization inside cubic MMR loops. Values/order stay exact.
        similarity_cache = (
            patch.object(runtime_reranker, '_text_similarity',
                         lru_cache(maxsize=8192)(runtime_reranker._text_similarity))
            if args.cache_only else nullcontext()
        )
        with similarity_cache:
            asyncio.run(verify_service_case(query, initial, docs, scores, contexts,
                                           arms['minilm_completed_mmr'][-1]['final_ids']))
        result['service_verified_case_ids'].append(cid)
        diagnostics.append({'case_id': cid, 'group': case['group'], 'query': query,
            'candidate_count': len(initial), 'candidate_label_present': any(h.label_match for h in initial),
            'candidates': [{'id': str(s.hit.index_chunk_id), 'label': s.hit.label_match,
                'modality': s.hit.modality, 'classic_score': s.score, 'vector_similarity': s.vector_similarity,
                'core': s.hit.index_chunk_id in {h.index_chunk_id for h in original},
                'logits': scores[s.hit.index_chunk_id]['logits']} for s in scored]})
        result.update(completed_cases=position, new_inference_pairs=scorer.new_pairs,
                      new_inference_windows=scorer.new_windows, inference_seconds=scorer.seconds,
                      elapsed_seconds=time.perf_counter()-started)
        result['arms'] = {name: {'cases': rows} for name, rows in arms.items()}
        write(OUTPUT/'results.json', result)
        print(json.dumps({'case': position, 'total': len(prepared), 'id': cid, 'candidate_count': len(initial),
            'new_pairs': scorer.new_pairs, 'new_windows': scorer.new_windows,
            'inference_seconds': round(scorer.seconds, 2)}), flush=True)
    baseline = arms['classic_protected']
    for name, rows in arms.items():
        result['arms'][name].update(summary=summarize(rows), vs_deployed_classic=compare_rows(baseline, rows),
            vs_unprotected_classic=compare_rows(arms['classic_completed'], rows),
            probes=probe_results(rows, probes), replacement_gate=replacement_gate(baseline, rows, probes))
    result['pool_effect_on_minilm'] = compare_rows(arms['minilm_core_mmr'], arms['minilm_completed_mmr'])
    result.update(status='complete', code_sha256={str(path.relative_to(ROOT)): digest(path) for path in (
        Path(__file__), ROOT/'tools/evaluate_minilm_source.py', ROOT/'src/rag_kb/retrieval/service.py',
        ROOT/'src/rag_kb/retrieval/reranker.py')})
    write(OUTPUT/'results.json', result)
    print(json.dumps({name: {'summary': arm['summary'], 'gate': arm['replacement_gate']}
                      for name, arm in result['arms'].items()}, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-only', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
