"""Offline ranking policies and evidence metrics; no models, labels in policy inputs, or I/O."""
from __future__ import annotations

import math
from collections import defaultdict


def supplement(base, proposed, costs, *, max_items=5, token_budget=2048):
    """Keep the complete baseline, adding whole unique chunks within both budgets."""
    result, used, added = list(base), 0, 0
    for item in proposed:
        if item in result:
            continue
        if added >= max_items:
            break
        if costs[item] <= token_budget - used:
            result.append(item)
            used += costs[item]
            added += 1
    return result


def rrf_order(classic, model, *, k=60):
    if len(set(classic)) != len(classic) or set(classic) != set(model) or len(classic) != len(model):
        raise ValueError('RRF requires identical unique candidate membership')
    ranks = {item: rank for rank, item in enumerate(model, 1)}
    return sorted(classic, key=lambda item: (
        -(1/(k+classic.index(item)+1) + 1/(k+ranks[item])), classic.index(item)))


def policies(classic, raw, mmr, costs):
    """The only routing inputs are rankings and whole-chunk token costs."""
    if len(set(classic)) != len(classic) or set(classic) != set(raw):
        raise ValueError('Ranking candidate membership differs')
    if len(set(mmr)) != len(mmr) or not set(mmr) <= set(classic):
        raise ValueError('Invalid MMR prefix')
    if set(costs) != set(classic) or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in costs.values()):
        raise ValueError('Token cost identity or value differs')
    base = classic[:10]
    fused = rrf_order(classic, raw)
    prefix = base[:5]
    out = {
        'classic10': base,
        'classic15': classic[:15],
        'minilm_raw10': raw[:10],
        'minilm_mmr10': mmr[:10],
        'rrf10': fused[:10],
        'protected_rrf10': (prefix + [i for i in fused if i not in prefix])[:10],
    }
    for name, order in (('raw', raw), ('mmr', mmr)):
        proposed = order[:5]
        out[f'append_{name}_unbounded'] = supplement(base, proposed, costs, token_budget=math.inf)
        selected = supplement(base, proposed, costs)
        out[f'append_{name}'] = selected
        actual_cost = sum(costs[i] for i in selected[len(base):])
        out[f'classic_{name}_same_count'] = supplement(base, classic, costs,
            max_items=len(selected)-len(base), token_budget=2048)
        out[f'classic_{name}_same_tokens'] = supplement(base, classic, costs,
            max_items=5, token_budget=actual_cost)
    return out


def evidence_row(case, selected, candidate_ids, costs):
    if len(selected) != len(set(selected)) or not set(selected) <= set(candidate_ids):
        raise ValueError('Invalid selected evidence')
    labels = case['labels']
    paths = [set(p) for p in case['required_paths']]
    answerable = case['answerable']
    if answerable and (not paths or any(not p for p in paths)):
        raise ValueError('Answerable case requires a nonempty evidence path')
    def units(ids):
        return set().union(*(set(labels.get(i, [])) for i in ids))
    required = set().union(*paths) if paths else set()
    found, available = units(selected), units(candidate_ids)
    ranks = [n for n, i in enumerate(selected, 1) if required & set(labels.get(i, []))]
    top_rank = min(ranks, default=None)
    complete = any(path <= found for path in paths) if answerable else None
    coverage = max((len(path & found)/len(path) for path in paths), default=None) if answerable else None
    return {
        'case_id': case['case_id'], 'group': case['group'], 'family': case['family'],
        'cluster': case.get('cluster', case['case_id']), 'answerable': answerable,
        'ids': selected, 'items': len(selected), 'tokens': sum(costs[i] for i in selected),
        'hit1': top_rank == 1 if answerable else None,
        'hit10': top_rank is not None and top_rank <= 10 if answerable else None,
        'any_evidence': bool(required & found) if answerable else None,
        'complete_evidence': complete, 'required_fraction': coverage,
        'mrr10': 1/top_rank if top_rank is not None and top_rank <= 10 else 0 if answerable else None,
        'candidate_complete': any(p <= available for p in paths) if answerable else None,
        'probe_complete': set(case['probe_ids']) <= set(selected) if case.get('probe_ids') else None,
    }


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['group']].append(row)
    result = {}
    for group, all_rows in groups.items():
        values = [r for r in all_rows if r['answerable']]
        entry = {'cases': len(all_rows), 'answerable': len(values),
                 'mean_items': sum(r['items'] for r in all_rows)/len(all_rows),
                 'mean_tokens': sum(r['tokens'] for r in all_rows)/len(all_rows)}
        for metric in ('hit1', 'hit10', 'any_evidence', 'complete_evidence', 'candidate_complete'):
            entry[metric] = sum(r[metric] for r in values) if values else None
        for metric in ('mrr10', 'required_fraction'):
            entry[metric] = sum(r[metric] for r in values)/len(values) if values else None
        result[group] = entry
    return result


def paired_changes(baseline, proposed):
    old = {r['case_id']: r for r in baseline}
    if len(old) != len(baseline) or len(old) != len(proposed) or set(old) != {r['case_id'] for r in proposed}:
        raise ValueError('Paired case identities differ')
    out = {}
    for metric in ('hit1', 'hit10', 'any_evidence', 'complete_evidence', 'probe_complete'):
        eligible = [r for r in proposed if r[metric] is not None]
        out[metric] = {
            'gained': [r['case_id'] for r in eligible if r[metric] and not old[r['case_id']][metric]],
            'lost': [r['case_id'] for r in eligible if old[r['case_id']][metric] and not r[metric]],
        }
    out['baseline_evidence_removed'] = [r['case_id'] for r in proposed if not set(old[r['case_id']]['ids']) <= set(r['ids'])]
    return out
