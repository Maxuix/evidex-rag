"""Explicit experimental Classic policies. No labels, model calls, or persistence."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
import math

from rag_kb.retrieval.reranker import _normalized_query, _normalized_text, _terms, score_hits
from tools.auto_strategy_metrics import supplement


@dataclass(frozen=True)
class Policy:
    name: str
    vector_weight: float = .65
    lexical: str = 'classic'
    depth: int = 40
    freeze_statistics: bool = True
    protect: int = 0
    mmr: float = 1.
    append: bool = False


# This finite grid is frozen before development scoring. Append controls cannot
# win a fixed-10 ranking comparison. No case-specific routes or gold inputs.
POLICIES = (
    Policy('classic10'),
    *(Policy(f'weight{int(w*100)}', vector_weight=w) for w in (.45, .55, .75, .85, 1.)),
    Policy('bm25_bound65', lexical='bounded_bm25'),
    Policy('bm25_bound50', lexical='bounded_bm25', vector_weight=.5),
    Policy('idf_coverage65', lexical='idf_coverage'),
    Policy('idf_coverage75', lexical='idf_coverage', vector_weight=.75),
    Policy('rrf60', lexical='rrf'),
    Policy('mmr95', mmr=.95),
    Policy('mmr85', mmr=.85),
    Policy('depth80', depth=80, freeze_statistics=False),
    Policy('depth100', depth=100, freeze_statistics=False),
    Policy('depth80_fixed', depth=80),
    Policy('depth100_fixed', depth=100),
    Policy('depth80_protect5', depth=80, protect=5),
    Policy('depth100_protect5', depth=100, protect=5),
    Policy('weight75_protect5', vector_weight=.75, protect=5),
    Policy('classic_append5', append=True),
)


def policy_grid():
    return [asdict(p) for p in POLICIES]


@lru_cache(maxsize=20000)
def terms(text):
    return _terms(text)


def order(query, hits, reference, policy):
    """Score source candidates against an explicit reference pool; never gold."""
    if not hits:
        return []
    if policy.lexical in {'classic', 'rrf'}:
        scored = score_hits(query, hits, vector_weight=policy.vector_weight,
                            lexical_weight=1-policy.vector_weight, reference_hits=reference)
        values = {s.hit.index_chunk_id: s.score for s in scored}
        if policy.lexical == 'rrf':
            dense = sorted(hits, key=lambda h: (h.cosine_distance, h.index_chunk_id.int))
            lexical = sorted(scored, key=lambda s: (-s.lexical_score, s.hit.cosine_distance, s.hit.index_chunk_id.int))
            values = {h.index_chunk_id: 1/(60+n) for n, h in enumerate(dense, 1)}
            for n, s in enumerate(lexical, 1):
                values[s.hit.index_chunk_id] += 1/(60+n)
    else:
        query_terms = tuple(dict.fromkeys(terms(query)))
        docs = [terms(h.text) for h in reference]
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        average = sum(map(len, docs))/max(1, n)
        idfs = {t: math.log1p((n-df[t]+.5)/(df[t]+.5)) for t in query_terms}
        total = sum(idfs.values())
        values = {}
        for hit in hits:
            counts = Counter(terms(hit.text))
            if policy.lexical == 'idf_coverage':
                lexical = sum(v for t, v in idfs.items() if counts[t])/total if total else 0.
            elif policy.lexical == 'bounded_bm25':
                norm = .25 + .75*max(1, sum(counts.values()))/max(1., average)
                raw = sum(v*counts[t]*2.2/(counts[t]+1.2*norm) for t, v in idfs.items())
                # The query-wide asymptotic maximum is independent of this
                # document's length. Unlike Classic, length penalty survives.
                lexical = raw/(2.2*total) if total else 0.
            else:
                raise ValueError('unknown lexical policy')
            semantic = (max(-1., min(1., 1-hit.cosine_distance))+1)/2
            value = policy.vector_weight*semantic + (1-policy.vector_weight)*lexical
            if query_terms and _normalized_query(terms(query)) in _normalized_text(hit.text):
                value = min(1., value+.08)
            values[hit.index_chunk_id] = value
    ranked = sorted(hits, key=lambda h: (-values[h.index_chunk_id], h.cosine_distance, h.index_chunk_id.int))
    if policy.mmr == 1:
        return ranked
    sets = {h.index_chunk_id: set(terms(h.text)) for h in hits}
    remaining, selected, redundancy = list(ranked), [], {h.index_chunk_id: 0. for h in hits}
    while remaining:
        best = max(remaining, key=lambda h: (
            policy.mmr*values[h.index_chunk_id]-(1-policy.mmr)*redundancy[h.index_chunk_id]
            if selected else values[h.index_chunk_id], values[h.index_chunk_id],
            -h.cosine_distance, -h.index_chunk_id.int))
        selected.append(best)
        remaining.remove(best)
        a = sets[best.index_chunk_id]
        for h in remaining:
            b = sets[h.index_chunk_id]
            similarity = len(a & b)/len(a | b) if a | b else 0.
            redundancy[h.index_chunk_id] = max(redundancy[h.index_chunk_id], similarity)
    return selected


def ids(hits):
    return [str(h.index_chunk_id) for h in hits]


def rank(query, full_hits, core_ids, extra_ids, costs, policy):
    """Reproduce source-context protection, then apply a declared experiment.

    full_hits are exact-distance ordered authorized source hits, with at least
    the first 100 plus current table supplements. Threshold stays fixed at .35.
    Existing table supplements keep their original admission rules.
    """
    by_id = {str(h.index_chunk_id): h for h in full_hits}
    if len(by_id) != len(full_hits) or set(core_ids) & set(extra_ids):
        raise ValueError('candidate identities must be unique and disjoint')
    original_core = [by_id[i] for i in core_ids]
    base_policy = POLICIES[0]
    base_core_order = order(query, original_core, original_core, base_policy)
    original_pool = original_core + [by_id[i] for i in extra_ids]
    base_order = order(query, original_pool, original_core, base_policy)
    if extra_ids:
        prefix = base_core_order[:5]
        kept = set(ids(prefix))
        base_order = prefix + [h for h in base_order if str(h.index_chunk_id) not in kept]
    baseline = ids(base_order[:10])
    if policy.name == 'classic10':
        return baseline, ids(original_pool)
    if policy.append:
        return supplement(baseline, ids(base_order), costs, max_items=5, token_budget=2048), ids(original_pool)
    core = original_core if policy.depth == 40 else [h for h in full_hits[:policy.depth] if 1-h.cosine_distance >= .35]
    core_set = set(ids(core))
    extras = [by_id[i] for i in extra_ids if i not in core_set]
    pool = core + extras
    reference = original_core if policy.freeze_statistics else core
    if not reference:
        return [], ids(pool)
    ranked = order(query, pool, reference, policy)
    if extras:
        prefix = order(query, core, reference, policy)[:5]
        kept = set(ids(prefix))
        ranked = prefix + [h for h in ranked if str(h.index_chunk_id) not in kept]
    if policy.protect:
        prefix = [by_id[i] for i in baseline[:policy.protect]]
        kept = set(ids(prefix))
        ranked = prefix + [h for h in ranked if str(h.index_chunk_id) not in kept]
    return ids(ranked[:10]), ids(pool)


def choose(arms):
    """Select fixed-10 policy by macro-family evidence coverage, then Top-1/cost.

    Reject a policy if any family's coverage or complete-probe count decreases
    or its macro average whole-source token count grows more than ten percent.
    Individual tradeoffs are reported, not hidden by this aggregate gate.
    """
    baseline = arms['classic10']
    families = sorted({r['family'] for r in baseline if r['answerable']})
    def metrics(rows):
        grouped = {f: [r for r in rows if r['family'] == f and r['answerable']] for f in families}
        return {f: {'coverage': sum(r['complete_evidence'] for r in rr)/len(rr),
                    'hit1': sum(r['hit1'] for r in rr)/len(rr),
                    'tokens': sum(r['tokens'] for r in rr)/len(rr),
                    'probes': sum(bool(r['probe_complete']) for r in rr)} for f, rr in grouped.items()}
    base = metrics(baseline)
    candidates = []
    for p in POLICIES:
        if p.append:
            continue
        values = metrics(arms[p.name])
        eligible = all(values[f]['coverage'] >= base[f]['coverage'] and values[f]['probes'] >= base[f]['probes'] for f in families)
        mean_growth = sum(values[f]['tokens']/base[f]['tokens'] for f in families)/len(families)
        eligible = eligible and mean_growth <= 1.10
        candidates.append({'name': p.name, 'eligible': eligible, 'families': values,
            'macro_coverage': sum(v['coverage'] for v in values.values())/len(values),
            'macro_hit1': sum(v['hit1'] for v in values.values())/len(values),
            'macro_token_ratio': mean_growth, 'depth': p.depth})
    eligible = [c for c in candidates if c['eligible']]
    best = max(eligible, key=lambda c: (c['macro_coverage'], c['macro_hit1'], -c['macro_token_ratio'], -c['depth'], c['name'] == 'classic10'))
    return {'selected': best['name'], 'candidates': candidates,
            'rule': 'nondecreasing family evidence/probes; macro token ratio <=1.10; maximize macro complete, Top-1, then minimize tokens/depth'}
