#!/usr/bin/env python3
"""Analyze frozen external comparisons without adjusting policies or model calls."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.auto_strategy_metrics import paired_changes, summarize
from tools.prepare_auto_strategy import ROOT, OUTPUT, read, write, require, sha


def paired_interval(baseline, proposed, metric, *, repetitions=10000):
    """Paired cluster bootstrap; keep original/rewritten or shared MuSiQue parents together."""
    old = {r['case_id']: r for r in baseline}
    require(len(old) == len(baseline) == len(proposed) and set(old) == {r['case_id'] for r in proposed}, 'Interval case identity differs')
    clusters = defaultdict(list)
    for row in proposed:
        if row[metric] is not None:
            clusters[row['cluster']].append(float(row[metric])-float(old[row['case_id']][metric]))
    if not clusters:
        return {'cases': 0, 'clusters': 0, 'difference': None, 'ci95': None}
    sums = np.array([sum(v) for v in clusters.values()])
    counts = np.array([len(v) for v in clusters.values()])
    rng = np.random.default_rng(20260905)
    draws = rng.integers(0, len(sums), size=(repetitions, len(sums)))
    values = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    return {'cases': int(counts.sum()), 'clusters': len(sums), 'difference': float(sums.sum()/counts.sum()),
            'ci95': [float(x) for x in np.quantile(values, [.025, .975])], 'repetitions': repetitions}


def compact(rows):
    grouped = [{**r, 'group': 'all'} for r in rows]
    return summarize(grouped)['all']


def quota_probe():
    """Explicit post-hoc diagnosis on already seen data; not another held-out result."""
    from rag_kb.adapters.local_reranker import _sigmoid
    from tools.evaluate_auto_strategy import costs_for
    from tools.auto_strategy_metrics import supplement, evidence_row
    out = {}
    for family in ('development', 'musique', 'hotpot'):
        result = read(OUTPUT/('development.json' if family == 'development' else family+'-results.json'))
        base = {r['case_id']: r for r in result['arms']['classic10']['cases']}
        if family == 'development':
            old = read(ROOT/'.runtime/evaluations/minilm-completed-20260905/results.json')
            sources = read(ROOT/'.runtime/evaluations/auto-qa-diagnosis-20260905/sources.json')['chunks']
            costs = costs_for({i: c['content'] for i, c in sources.items()})
            records = []
            for d in old['diagnostics']:
                cid = d['case_id']
                initial = [c['id'] for c in d['candidates']]
                logits = {c['id']: max(c['logits']) for c in d['candidates']}
                raw = sorted(initial, key=lambda i: (-_sigmoid(logits[i]), initial.index(i)))
                classic = base[cid]['ids'] + [i for i in initial if i not in base[cid]['ids']]
                case = {**base[cid], 'required_paths': [['old']],
                    'labels': {c['id']: ['old'] if c['label'] else [] for c in d['candidates']}}
                records.append((case, classic, raw))
        else:
            frozen = read(OUTPUT/(family+'-frozen.json'))
            docs = {d['id']: d for d in frozen['documents']}
            costs = costs_for({i: d['text'] for i, d in docs.items()})
            definitions = {c['case_id']: c for c in frozen['cases']}
            records = [({**definitions[d['case_id']],
                'labels': {i: [docs[i]['document_id']] for i in d['candidate_ids']}},
                d['candidate_ids'], d['raw_order']) for d in result['diagnostics']]
        arms = defaultdict(list)
        for case, classic, raw in records:
            baseline = classic[:10]
            added = supplement(baseline, raw[:15], costs)
            used = sum(costs[i] for i in added[len(baseline):])
            strategies = {
                'raw_unique5': added,
                'classic_cap2048': supplement(baseline, classic, costs),
                'classic_actual_budget': supplement(baseline, classic, costs, token_budget=used),
            }
            for name, ids in strategies.items():
                arms[name].append(evidence_row(case, ids, classic, costs))
        out[family] = {'diagnostic_only_seen_data': True,
            'hypothesis': 'Consider raw MiniLM top15, deduplicate against all Classic top10, then admit at most 5 new whole items / 2048 text tokens.',
            'arms': dict(arms), 'summary': {name: compact(rows) for name, rows in arms.items()},
            'vs_classic_actual_budget': paired_changes(arms['classic_actual_budget'], arms['raw_unique5'])}
    write(OUTPUT/'quota-probe.json', out)
    return {family: {'summary': data['summary'], 'changes': data['vs_classic_actual_budget']} for family, data in out.items()}


def analyze():
    selection = read(OUTPUT/'selection.json')
    require(selection['development_sha256'] == sha(OUTPUT/'development.json') and
            selection['policy_sha256'] == sha(ROOT/'tools/auto_strategy_metrics.py'), 'Frozen selection changed')
    selected = selection['selected_policy']
    output = {'status': 'complete', 'selected_before_external_scoring': selected,
              'qa_generation_calls': 0, 'new_model_calls': 0, 'families': {}}
    for family in ('development', 'musique', 'hotpot'):
        path = OUTPUT/('development.json' if family == 'development' else family+'-results.json')
        result = read(path)
        require(result['status'] == 'complete', 'Comparison incomplete')
        for p, value in result['input_sha256'].items():
            require(sha(ROOT/p) == value, 'Comparison input changed')
        if family != 'development':
            require(result['new_inference_pairs'] == 0, 'External final result must be cache-only')
            verification = read(OUTPUT/(family+'-verification.json'))
            require(verification['status'] == 'complete' and verification['results_sha256'] == sha(path), 'Unverified external replay')
            for p, value in result['code_sha256'].items():
                require(sha(ROOT/p) == value, 'Comparison code changed')
        arms = result['arms']
        current = arms[selected]['cases']
        comparisons = {}
        for control in ('classic10', 'classic15', 'classic_mmr_same_count', 'classic_mmr_same_tokens'):
            baseline = arms[control]['cases']
            comparisons[control] = {'changes': paired_changes(baseline, current),
                'intervals': {m: paired_interval(baseline, current, m) for m in ('any_evidence', 'complete_evidence', 'required_fraction')}}
        control = arms['classic_mmr_same_tokens']['cases']
        selected_by_group = {}
        for group in sorted({r['group'] for r in current}):
            proposed = [r for r in current if r['group'] == group]
            baseline = [r for r in control if r['group'] == group]
            selected_by_group[group] = {'changes_vs_same_tokens': paired_changes(baseline, proposed),
                'complete_interval_vs_same_tokens': paired_interval(baseline, proposed, 'complete_evidence')}
        item = {'arms': {name: compact(arm['cases']) for name, arm in arms.items()},
                'selected_comparisons': comparisons, 'selected_by_group': selected_by_group,
                'group_summaries': {name: arm['summary'] for name, arm in arms.items()}}
        if family != 'development':
            frozen = read(OUTPUT/(family+'-frozen.json'))
            by_case = {c['case_id']: c for c in frozen['cases']}
            item['by_hop_count'] = {name: summarize([{**r, 'group': 'hop'+str(by_case[r['case_id']]['hop_count'])}
                for r in arm['cases'] if r['answerable']]) for name, arm in arms.items()}
            inference = read(OUTPUT/(family+'-inference.json'))
            item['inference_cost'] = {k: inference[k] for k in ('new_inference_pairs', 'new_inference_windows', 'inference_seconds', 'elapsed_seconds')}
            item['verified_rankings'] = verification['rankings']
            item['negative_case_ids'] = [r['case_id'] for r in current if not r['answerable']]
        output['families'][family] = item
    external = [output['families'][name] for name in ('musique', 'hotpot')]
    output['external_gate'] = {
        'baseline_evidence_preserved': all(not f['selected_comparisons']['classic10']['changes']['baseline_evidence_removed'] for f in external),
        'complete_coverage_beats_same_tokens_on_both_datasets': all(
            f['selected_comparisons']['classic_mmr_same_tokens']['intervals']['complete_evidence']['difference'] > 0 for f in external),
        'paired_ci_lower_bound_positive_on_both_datasets': all(
            f['selected_comparisons']['classic_mmr_same_tokens']['intervals']['complete_evidence']['ci95'][0] > 0 for f in external),
        'old_complete_evidence_lost_vs_same_tokens': {name:
            output['families'][name]['selected_comparisons']['classic_mmr_same_tokens']['changes']['complete_evidence']['lost']
            for name in ('musique', 'hotpot')},
    }
    output['limits'] = [
        'Historical development families are not independent unseen test sets.',
        'HotpotQA 100 sampled questions use all 9793 shared paragraphs and exclude the 20 pilot questions; not the official fullwiki protocol.',
        'Archived MuSiQue was previously used for Agent/Graph evaluation; this is transfer outside current reranker selection, not never-evaluated material.',
        'Shared source documents can correlate cases beyond the clustered parent IDs; intervals are approximate, not universal guarantees.',
        'External items retain one complete source paragraph document; this isolates retrieval/ranking, not current PDF parsing/chunking or end-to-end Agent behavior.',
        'Token budgets use exact local cl100k text counts, not Mimo billing or full prompt overhead.',
        'Unanswerable rows are diagnostics with no vacuous retrieval success; no refusal accuracy is claimed.',
        'All supplements were exercised offline. An autonomous Agent trigger and resulting answer quality are not measured here.',
    ]
    output['posthoc_quota_diagnosis'] = quota_probe()
    output['artifact_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in (
        OUTPUT/'selection.json', OUTPUT/'development.json', OUTPUT/'musique-results.json', OUTPUT/'hotpot-results.json', Path(__file__))}
    write(OUTPUT/'analysis.json', output)
    for name, item in output['families'].items():
        print(name)
        for arm in ('classic10', 'classic15', 'minilm_raw10', 'minilm_mmr10', 'rrf10', 'protected_rrf10', 'append_raw', 'append_mmr', 'classic_mmr_same_tokens'):
            print(arm, item['arms'][arm])
        print('selected vs same tokens', item['selected_comparisons']['classic_mmr_same_tokens'])
    print('External gate:', output['external_gate'])


if __name__ == '__main__':
    analyze()
