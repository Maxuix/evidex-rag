"""Integrity and behavioral checks for the bounded Classic experiment."""
from types import SimpleNamespace
from uuid import UUID

import pytest

from rag_kb.retrieval.source_context import rank_with_source_context
from tools.classic_strategy import POLICIES, Policy, choose, ids, order, rank
from tools.evaluate_classic_strategy import immutable


def hit(i, text='cash revenue assets', distance=.2):
    return SimpleNamespace(index_chunk_id=UUID(int=i), text=text, cosine_distance=distance)


def test_baseline_replays_runtime_with_frozen_source_supplements():
    core = [hit(i, text='cash and revenue '+str(i), distance=.1+i*.015) for i in range(1, 15)]
    extra = hit(100, text='cash assets liquidity', distance=.7)
    full = sorted([*core, extra], key=lambda h: h.cosine_distance)
    price = {i: 20 for i in ids(full)}
    selected, candidates = rank('cash assets', full, ids(core), ids([extra]), price, POLICIES[0])
    expected = ids(s.hit for s in rank_with_source_context('cash assets', tuple(core), (extra,), top_k=10))
    assert selected == expected
    assert candidates == ids([*core, extra])


def test_length_normalization_retains_length_penalty():
    short = hit(2, text='revenue')
    long = hit(1, text='revenue '+('unrelated '*100))
    # Classic's per-document denominator cancels the BM25 length penalty when
    # the matched term occurs once, leaving the UUID tie-break in control.
    assert order('revenue growth', [short, long], [short, long], POLICIES[0])[0] is long
    assert order('revenue growth', [short, long], [short, long], Policy('bounded', lexical='bounded_bm25'))[0] is short


def test_protected_expansion_keeps_head_without_gold_or_extra_slots():
    full = [hit(i, text='revenue' if i > 40 else 'ordinary', distance=.1+i*.004) for i in range(1, 101)]
    price = {i: 20 for i in ids(full)}
    base, _ = rank('revenue', full, ids(full[:40]), [], price, POLICIES[0])
    chosen, candidates = rank('revenue', full, ids(full[:40]), [], price, Policy('expanded', depth=80, protect=5))
    assert chosen[:5] == base[:5]
    assert len(chosen) == len(set(chosen)) == 10
    assert len(candidates) == 80 and set(chosen) <= set(candidates)
    assert any(UUID(i).int > 40 for i in chosen)


def test_depth_expansion_keeps_fixed_cosine_threshold():
    full = [hit(i, distance=.2 if i <= 45 else .9) for i in range(1, 101)]
    selected, candidates = rank('query', full, ids(full[:40]), [], {i: 10 for i in ids(full)}, Policy('expanded', depth=100))
    assert len(candidates) == 45
    assert all(UUID(i).int <= 45 for i in selected)


def test_append_preserves_whole_baseline_and_both_budgets():
    full = [hit(i, distance=.1+i*.01) for i in range(1, 41)]
    price = {i: 500 for i in ids(full)}
    base, _ = rank('cash', full, ids(full), [], price, POLICIES[0])
    chosen, _ = rank('cash', full, ids(full), [], price, Policy('append', append=True))
    assert chosen[:10] == base
    assert len(chosen) == 14
    assert sum(price[i] for i in chosen[10:]) == 2000


def test_duplicate_candidates_and_changed_freeze_rejected(tmp_path):
    a = hit(1)
    with pytest.raises(ValueError, match='unique'):
        rank('q', [a, a], ids([a]), [], {str(a.index_chunk_id): 1}, POLICIES[0])
    path = tmp_path/'frozen.json'
    immutable(path, {'selection': 'a'})
    immutable(path, {'selection': 'a'})
    with pytest.raises(RuntimeError, match='Frozen'):
        immutable(path, {'selection': 'b'})


def test_selection_does_not_hide_family_regression_or_reward_more_slots():
    rows = [{'family': family, 'answerable': True, 'complete_evidence': True, 'hit1': False,
             'tokens': 100, 'probe_complete': None} for family in ('a', 'b')]
    arms = {p.name: [dict(r) for r in rows] for p in POLICIES}
    arms['weight75'][1]['complete_evidence'] = False
    arms['weight75'][0]['hit1'] = True
    arms['weight85'][0]['hit1'] = True
    arms['weight85'][0]['tokens'] = 200
    arms['classic_append5'][0]['hit1'] = True
    result = choose(arms)
    assert result['selected'] == 'classic10'
    eligible = {c['name']: c['eligible'] for c in result['candidates']}
    assert not eligible['weight75'] and not eligible['weight85']
    assert 'classic_append5' not in eligible


def test_empty_pool_does_not_invent_results():
    for policy in POLICIES:
        assert rank('query', [], [], [], {}, policy) == ([], [])
