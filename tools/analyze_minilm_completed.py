#!/usr/bin/env python3
"""Verify completed-pool replay and separate candidate competition from batch effects.

This is score-only diagnosis. The core-subset counterfactual never becomes a new
production policy, and the gold-document check is not used to filter candidates.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from rag_kb.adapters.local_reranker import _sigmoid
from tools.analyze_source_recall import ROOT, DIAGNOSIS, inputs, read, write, digest, require
from tools.analyze_auto_qa_ranking import rank_with_scores, row_result, summarize
from tools.evaluate_minilm_completed import OUTPUT, OLD_MODEL_ROOT, compare_rows


def analyze():
    current = read(OUTPUT/'results.json')
    live = read(OUTPUT/'inference-result.json')
    require(current['status'] == live['status'] == 'complete' and current['completed_cases'] == 85,
            'A complete paired replay is required')
    require(current['new_inference_pairs'] == current['new_inference_windows'] == 0,
            'Verification must come from a cache-only run')
    for path, sha in current['input_sha256'].items():
        require(digest(ROOT/path) == sha, f'Input changed: {path}')
    for path, sha in current['code_sha256'].items():
        require(digest(ROOT/path) == sha, f'Replay code changed: {path}')
    require(current['model_sha256'] == live['model_sha256'] and
            current['window_code_sha256'] == live['window_code_sha256'], 'Inference version changed')
    require(current['arms'] == live['arms'] and current['diagnostics'] == live['diagnostics'],
            'Cached scores or final rankings differ from live inference')
    require(len(set(current['service_verified_case_ids'])) == 85, 'Runtime-stage verification incomplete')
    _, snapshot, _, _, cases = inputs()
    definitions = {c['evaluation_case_id']: c for c in cases}
    chunks = snapshot['chunks']
    prior = {r['case_id']: {c['id']: c for c in r['candidates']}
             for r in read(OLD_MODEL_ROOT/'repaired-int8.json')['diagnostics']}
    fixed_score_core = []
    shifts = []
    for diagnostic in current['diagnostics']:
        cid = diagnostic['case_id']
        core = [c for c in diagnostic['candidates'] if c['core']]
        require(set(prior[cid]) == {c['id'] for c in core}, 'Core-score comparison differs')
        hits = [SimpleNamespace(index_chunk_id=UUID(c['id']), label_match=c['label'],
                                text=chunks[c['id']]['content']) for c in core]
        probabilities = {UUID(c['id']): _sigmoid(max(c['logits'])) for c in core}
        fixed_score_core.append(row_result(definitions[cid], rank_with_scores(hits, probabilities, mmr=True)))
        for c in core:
            old = prior[cid][c['id']]['logits']
            require(len(old) == len(c['logits']), 'Source window coverage changed')
            shifts.append(abs(max(old)-max(c['logits'])))
    require(len(shifts) == 3400, 'Unexpected number of core pairs')
    arms = current['arms']
    document_mismatches = {}
    for name, arm in arms.items():
        document_mismatches[name] = {}
        for group in ('direct', 'paraphrase'):
            document_mismatches[name][group] = [r['case_id'] for r in arm['cases']
                if r['group'] == group and r['final_ids'] and
                chunks[r['final_ids'][0]]['source_metadata']['original_filename']
                != Path(definitions[r['case_id']]['document_path']).name]
    import json
    pair_path = ROOT/'evaluation/document-qa-v1/auto-qa-paraphrases.jsonl'
    pairs = [json.loads(line) for line in pair_path.read_text().splitlines() if line.strip()]
    require(len(pairs) == len({p['base_case_id'] for p in pairs}) == 24, 'Paired query identity changed')
    paired_results = {}
    for name, arm in arms.items():
        by_id = {r['case_id']: r for r in arm['cases']}
        original = [by_id[p['base_case_id']] for p in pairs]
        rewritten = [by_id[p['case_id']] for p in pairs]
        paired_results[name] = {
            'summary': summarize(original + rewritten),
            'paraphrase_vs_own_original': compare_rows(original, [
                {**r, 'case_id': p['base_case_id'], 'group': 'direct'}
                for p, r in zip(pairs, rewritten, strict=True)])['direct'],
        }
    result = {
        'status': 'complete', 'qa_generation_calls': 0, 'new_inference_pairs': 0,
        'verified_rankings': sum(len(a['cases']) for a in arms.values()),
        'verified_runtime_cases': len(current['service_verified_case_ids']),
        'original_core_with_expanded_batch_scores': {
            'diagnostic_only': True, 'summary': summarize(fixed_score_core), 'cases': fixed_score_core,
            'vs_original_core_scores': compare_rows(arms['minilm_core_mmr']['cases'], fixed_score_core),
        },
        'candidate_addition_effect_at_fixed_scores': compare_rows(
            fixed_score_core, arms['minilm_completed_mmr']['cases']),
        'core_max_logit_shifts': {
            'count': len(shifts), 'above_1e_6': sum(v > 1e-6 for v in shifts),
            'above_0_1': sum(v > .1 for v in shifts),
            'p50': sorted(shifts)[len(shifts)//2], 'p95': sorted(shifts)[int(len(shifts)*.95)],
            'max': max(shifts),
        },
        'gold_document_top1_mismatches': document_mismatches,
        'gold_document_warning': 'Diagnostic reference only; an underspecified query may admit multiple documents. No gold filtering.',
        'paired_24': paired_results,
        'paired_warning': 'Same underlying questions, but rewriting can change query meaning and retrieval membership. Not a pure phrasing-only causal estimate.',
        'files_sha256': {str(p.relative_to(ROOT)): digest(p) for p in (
            OUTPUT/'results.json', OUTPUT/'inference-result.json', pair_path, Path(__file__))},
    }
    write(OUTPUT/'verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k in {
        'status', 'verified_rankings', 'verified_runtime_cases', 'core_max_logit_shifts'}}, indent=2))


if __name__ == '__main__':
    analyze()
