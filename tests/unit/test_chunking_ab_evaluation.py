from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from rag_kb.domain import EmbeddingBatch
from tools.evaluate_chunking_ab import EmbeddingCache, gate_pair, support_match


def row(case='case', rank=1, group='exact_support', answer=True):
    return {'case_id': case, 'rank': rank, 'group': group, 'answer_pass': answer}


def test_gate_rejects_lost_cases_hits_answers_and_group_mrr():
    assert not gate_pair([row()], [row()])
    assert gate_pair([row()], []) == ['case_coverage']
    assert 'lost_hit10:case' in gate_pair([row()], [row(rank=None)])
    assert 'lost_exact_hit1:case' in gate_pair([row()], [row(rank=2)])
    assert 'lost_answer:case' in gate_pair([row()], [row(answer=False)])
    assert 'mrr_regression:direct' in gate_pair([row(group='direct')], [row(group='direct', rank=2)])


def test_support_requires_source_identity_and_complete_literal_support():
    case = {'kind': 'exact_support', 'filename': 'table.csv', 'supports': ['Revenue', 'Item109', '1109']}
    assert support_match(case, {'filename': 'table.csv', 'text': 'Revenue\nItem109 1109'})
    assert not support_match(case, {'filename': 'other.csv', 'text': 'Revenue\nItem109 1109'})
    assert not support_match(case, {'filename': 'table.csv', 'text': 'Item109 1109'})


def test_cache_replays_without_provider_and_separates_query_from_document():
    class Provider:
        max_batch_size = 2
        calls = 0

        async def embed_documents(self, texts):
            self.calls += 1
            return EmbeddingBatch(tuple((1., 0.) for _ in texts))

        async def embed_query(self, text):
            self.calls += 1
            return (0., 1.)

    async def scenario(path):
        provider = Provider()
        first = EmbeddingCache(provider, path, {'revision': 'fixed'})
        assert await first.documents(('same', 'same')) == ((1., 0.), (1., 0.))
        assert await first.query('same') == [0., 1.]
        assert provider.calls == 2
        replay = EmbeddingCache(None, path, {'revision': 'fixed'}, cache_only=True)
        assert await replay.documents(('same',)) == ((1., 0.),)
        assert await replay.query('same') == [0., 1.]
        with pytest.raises(ValueError, match='cache missing'):
            await replay.documents(('new input',))
        changed = EmbeddingCache(None, path, {'revision': 'changed'}, cache_only=True)
        with pytest.raises(ValueError, match='cache missing'):
            await changed.query('same')
        for cache in (first, replay, changed):
            cache.database.close()
    with TemporaryDirectory() as directory:
        asyncio.run(scenario(Path(directory) / 'vectors.sqlite'))


def test_embedding_whitespace_probe_keeps_source_values_and_structured_content():
    from tools.diagnose_chunking_embedding import projection
    prose = {'embedding_text': '[body]\nSensor\nvoltage 3.14;\n\nhttps://x.example/v1.2',
             'source_location': {'item_refs': ['#/texts/0']}, 'modality': 'text'}
    original = prose['embedding_text']
    assert projection(prose, set()) == '[body]\nSensor voltage 3.14; https://x.example/v1.2'
    assert prose['embedding_text'] == original
    code = {**prose, 'embedding_text': '[body]\ndef f():\n    return 3.14'}
    assert projection(code, {'#/texts/0'}) == code['embedding_text']
    table = {**prose, 'modality': 'table',
             'embedding_text': '[table]\nRevenue\n\n| Year | USD |\n| --- | --- |\n| 2026 | 3.14 |'}
    assert projection(table, set()).endswith('| Year | USD |\n| --- | --- |\n| 2026 | 3.14 |')

    assert projection(prose, set(), preserve_paragraphs=True) == '[body]\nSensor voltage 3.14;\n\nhttps://x.example/v1.2'
