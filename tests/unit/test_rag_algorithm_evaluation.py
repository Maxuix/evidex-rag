from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest

from tools.rag_algorithm_policies import CorpusIndex, balanced, pack, rrf, validate_output
from tools.rag_algorithm_runtime import ModelIO, validate_vector


def test_followup_requires_real_visible_source_and_exact_quote_in_query():
    valid = {'queries': [{'query': 'Diageo headquarters city', 'source_id': 's1', 'quote': 'Diageo'}]}
    assert validate_output('iterate', valid, {'s1': 'Owned by Diageo.'}) == valid
    for visible in ({}, {'s1': 'Owned by another company.'}):
        with pytest.raises(ValueError):
            validate_output('iterate', valid, visible)
    with pytest.raises(ValueError):
        validate_output('iterate', {'queries': [{'query': 'company headquarters', 'source_id': 's1', 'quote': 'Diageo'}]}, {'s1': 'Diageo'})


def test_generated_shapes_cannot_add_answer_or_exceed_query_budget():
    for kind, value in [('multiquery', {'queries': ['a', 'a']}), ('stepback', {'queries': ['a', 'b']}),
        ('hyde', {'passage': 'speculative', 'answer': 'invented'}), ('iterate', {'queries': [{}]*3})]:
        with pytest.raises(ValueError):
            validate_output(kind, value)
    assert validate_output('iterate', {'queries': []}) == {'queries': []}


def test_sparse_search_can_recover_source_outside_dense_candidates():
    docs = [{'id': str(UUID(int=i+1)), 'text': 'rarecode ZQ81' if i == 44 else 'ordinary text'} for i in range(50)]
    matrix = np.zeros((50, 1024)); matrix[:, 0] = 1
    query = matrix[0]
    index = CorpusIndex(docs, matrix)
    dense, _ = index.dense(query)
    sparse = index.bm25('ZQ81')
    assert docs[44]['id'] not in dense
    assert sparse == [docs[44]['id']]
    assert docs[44]['id'] in rrf([dense, sparse])


def test_fusion_identity_and_budget_controls():
    assert rrf([['a', 'b'], ['b', 'c']])[0] == 'b'
    with pytest.raises(ValueError):
        rrf([['a', 'a']])
    ordered = balanced(['a', 'b', 'c', 'd'], [['x', 'y'], ['z', 'x']], ['d'])
    assert ordered[:4] == ['a', 'b', 'c', 'd']
    assert len(ordered) == len(set(ordered))
    assert pack(['a', 'x', 'b'], {'a': 6, 'x': 11, 'b': 4}, token_budget=10) == ['a', 'b']


def test_invalid_vectors_fail_before_search_or_cache():
    for vector in ([0.]*1024, [1.]*100, [float('nan')]*1024, [True]*1024):
        with pytest.raises(RuntimeError):
            validate_vector(vector)


def test_cache_only_miss_does_not_call_provider(tmp_path):
    import asyncio
    io = ModelIO(tmp_path, {'chat': {'model': 'mimo-v2.5'}}, {'dimension': 1024})
    with pytest.raises(RuntimeError, match='Missing generation'):
        asyncio.run(io.generate('stepback', {'question': 'q'}))
    with pytest.raises(RuntimeError, match='Missing vector'):
        asyncio.run(io.embed('q', kind='query'))


def test_invalid_model_output_is_not_saved_or_used(tmp_path):
    import asyncio
    class Invalid:
        async def complete(self, _):
            return SimpleNamespace(content='{"queries":["x","x"]}', usage={'total_tokens': 7}, model='mimo-v2.5', finish_reason='stop')
    io = ModelIO(tmp_path, {'chat': {'model': 'mimo-v2.5'}}, {}, {'chat': Invalid()})
    with pytest.raises(ValueError, match='twice'):
        asyncio.run(io.generate('multiquery', {'question': 'q'}))
    assert not list((tmp_path/'calls').glob('*.json'))
    failure = next((tmp_path/'failures').glob('*.json')).read_text()
    assert '"x"' not in failure
    assert io.new_calls == 2


def test_valid_generation_survives_json_cache_roundtrip(tmp_path):
    import asyncio
    from tools.rag_algorithm_runtime import PLANNING_MAX_OUTPUT_TOKENS
    class Valid:
        async def complete(self, request):
            assert request.max_output_tokens == PLANNING_MAX_OUTPUT_TOKENS == 4096
            return SimpleNamespace(content='{"queries":["broader question"]}',
                usage={'total_tokens': 7}, model='mimo-v2.5', finish_reason='stop')
    io = ModelIO(tmp_path, {'chat': {'model': 'mimo-v2.5'}}, {}, {'chat': Valid()})
    original = asyncio.run(io.generate('stepback', {'question': 'q'}))
    replay = ModelIO(tmp_path, io.identities, {})
    assert asyncio.run(replay.generate('stepback', {'question': 'q'})) == original
    assert io.new_calls == 1 and replay.new_calls == 0


def test_reader_keeps_full_real_source_and_resolves_only_issued_refs():
    from tools.evaluate_rag_algorithm_reader import reader_input
    from rag_kb.answering.evidence import render_text_final_answer
    source_id = str(UUID(int=2))
    body = 'The headquarters is in Paris. '+('Source context. '*300)
    messages, prompts, _ = reader_input({'query': 'Where?'}, [source_id],
        {source_id: {'text': body, 'title': 'Company'}})
    assert body in next(m.content for m in messages if m.role == 'evidence')
    _, rendered, refs, observed = render_text_final_answer('Paris [ev_1][ev_99]', prompts,
        loaded_visual_refs=set(), current_query='Where?')
    assert rendered.content == 'Paris [1]'
    assert refs == ('ev_1',) and 'ev_99' in observed
    assert 'ev_99' not in prompts
