from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from uuid import UUID

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument

from tests.unit.test_docling_consumers import markdown_document, paginated_document, table_data
from rag_kb.document_processing.composite_text import with_composite_embedding_text
from rag_kb.document_processing.docling import (
    assemble_semantic_chunks, assemble_structural, composite_evidence,
    docling_semantic_units, docling_unit_sequence_hash,
)
from rag_kb.document_processing.docling.table_chunks import table_chunks
from rag_kb.document_processing.profiles import (
    SEMANTIC_CHUNKING_CONFIG, SEMANTIC_CHUNKING_CONFIG_V3, SEMANTIC_CHUNKING_CONFIG_V4,
    STRUCTURAL_CHUNKING_CONFIG, STRUCTURAL_CHUNKING_CONFIG_V4, profile_fingerprint,
    profile_for_preset, resolve,
)
from rag_kb.document_processing.semantic_boundaries import build_chunk_plan, requires_semantic_vectors
from rag_kb.document_processing.semantic_text import joined_units
from rag_kb.document_processing.tokenization import count_chunk_tokens
from rag_kb.domain import ChunkingPreset, ParserExecutionError
from rag_kb.repositories.sqlalchemy_graph import _graph_chunking_profile_compatible


def semantic(doc, config=SEMANTIC_CHUNKING_CONFIG, captions=False):
    units = docling_semantic_units(doc, chunking_config=config, include_captions=captions)
    plan = build_chunk_plan(indexed_document_version_id=UUID(int=1),
                            source_checksum_sha256='a' * 64, profile_fingerprint='b' * 64,
                            units=units, sequence_hash=docling_unit_sequence_hash(units),
                            vectors=tuple((1., 0.) for _ in units) if requires_semantic_vectors(units) else None)
    return units, plan, assemble_semantic_chunks(doc, units, plan)


@pytest.mark.parametrize('text', [
    '价格为3.14元，版本v1.2.3，地址https://example.com/a。下一句完整。',
    'Dr. Smith paid $19.95. Contact a.b@example.com; version 1.2.3 works.',
    '第一段。  第二句！\n\n另一段有完整空行。',
    ' '.join(['alpha'] * 600),
    '罕见字符𠮷😀价格3.14' * 180,
])
def test_source_text_is_not_rewritten_or_duplicated(text):
    doc = DoclingDocument(name='source')
    doc.add_text(label=DocItemLabel.TEXT, text=text)
    units, plan, chunks = semantic(doc)
    assert joined_units(units) == text
    assert all(unit.token_count <= 160 for unit in units)
    assert all(chunk.token_count <= 800 for chunk in chunks)
    if count_chunk_tokens(text) <= 800:
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert not requires_semantic_vectors(units)


def test_code_preserves_numeric_literals_newlines_and_indentation():
    body = 'def amount(x):\n    if x > 3.14:\n        return x * 1.05\n    return 0'
    doc = markdown_document('```python\n' + body + '\n```')
    assert semantic(doc)[2][0].text == body
    compile(semantic(doc)[2][0].text, '<chunk>', 'exec')


def test_explicit_legacy_keeps_its_exact_source_projection():
    text = '价格为3.14元，版本v1.2.3，地址https://example.com/a。下一句完整。'
    doc = markdown_document(text)
    old = semantic(doc, SEMANTIC_CHUNKING_CONFIG_V4)[2][0]
    assert old.text == '价格为3. 14元，版本v1. 2. 3，地址https://example. com/a。 下一句完整。'
    assert semantic(doc)[2][0].text == text
    parser = profile_for_preset(ChunkingPreset.SEMANTIC_BALANCED_V1).parser_config
    for config in (SEMANTIC_CHUNKING_CONFIG_V3, SEMANTIC_CHUNKING_CONFIG_V4, STRUCTURAL_CHUNKING_CONFIG_V4):
        assert resolve(parser, deepcopy(config))
    assert _graph_chunking_profile_compatible(SEMANTIC_CHUNKING_CONFIG_V4)
    assert _graph_chunking_profile_compatible(SEMANTIC_CHUNKING_CONFIG)
    assert not _graph_chunking_profile_compatible(SEMANTIC_CHUNKING_CONFIG_V3)


def test_text_captions_reach_final_evidence_and_embedding_only_when_requested():
    doc = paginated_document()
    for chunks in (assemble_structural(doc, include_captions=True), semantic(doc, captions=True)[2]):
        draft = composite_evidence(doc, chunks, (), (), profile='test', source_checksum_sha256='a'*64)
        units = with_composite_embedding_text(draft.units, draft.relations)
        assert any('Figure 1. A chart' in unit.content for unit in units)
        assert any('Figure 1. A chart' in unit.embedding_text for unit in units)
    assert all('Figure 1. A chart' not in chunk.text for chunk in assemble_structural(doc))
    assert all('Figure 1. A chart' not in chunk.text for chunk in assemble_structural(doc, chunking_config=STRUCTURAL_CHUNKING_CONFIG_V4, include_captions=True))


def test_table_repeats_headers_covers_rows_and_skips_semantic_calls():
    doc = DoclingDocument(name='table')
    doc.add_table(data=table_data((('Product', 'Year', 'Revenue'),
                                   *((f'Item{i:03d}', '2026', str(1000+i)) for i in range(120)))))
    units, _, chunks = semantic(doc)
    assert not requires_semantic_vectors(units)
    for result in (chunks, assemble_structural(doc)):
        assert len(result) > 1
        assert all('Revenue' in chunk.text and chunk.token_count <= 800 for chunk in result)
        text = '\n'.join(chunk.text for chunk in result)
        for i in range(120):
            assert text.count(f'Item{i:03d}') == 1


def test_wide_table_rows_keep_headers_and_multiple_header_rows_repeat():
    text = '| Name | Amount |\n|---|---|\n| unit | USD |\n' + '\n'.join('| Row%d | %s |' % (i, 'value '*900) for i in range(4))
    parts = table_chunks(text, header_rows=2, prefix='Annual revenue')
    assert len(parts) > 4
    assert all('Name' in part and 'USD' in part and 'Annual revenue' in part for part in parts)
    assert all(count_chunk_tokens(part) <= 800 for part in parts)
    assert sum(part.count('value') for part in parts) == 3600


def test_heading_does_not_become_orphan_before_long_body():
    doc = DoclingDocument(name='heading')
    doc.add_heading(text='Project Orion', level=1)
    doc.add_text(label=DocItemLabel.TEXT, text=' '.join(['alpha']*1800))
    current = assemble_structural(doc)
    legacy = assemble_structural(doc, chunking_config=STRUCTURAL_CHUNKING_CONFIG_V4)
    assert legacy[0].text == 'Project Orion'
    assert all('alpha' in chunk.text and chunk.token_count <= 800 for chunk in current)
    assert current[0].text.startswith('Project Orion\n\n')


def test_eof_and_heading_only_documents_keep_source_text():
    doc = markdown_document('# Intro\n\nBody content.\n\n## Final appendix')
    assert 'Final appendix' in '\n'.join(chunk.text for chunk in semantic(doc)[2])
    doc = markdown_document('# Only heading')
    assert semantic(doc)[2][0].text == 'Only heading'


def test_skipping_unused_vectors_preserves_complete_plan_identity():
    units = docling_semantic_units(paginated_document())
    facts = dict(indexed_document_version_id=UUID(int=1), source_checksum_sha256='a'*64,
                 profile_fingerprint='b'*64, units=units, sequence_hash=docling_unit_sequence_hash(units))
    assert not requires_semantic_vectors(units)
    assert build_chunk_plan(**facts, vectors=None) == build_chunk_plan(**facts, vectors=tuple((1., 0.) for _ in units))
    changed = (units[0], replace(units[1], separator_before=' '), *units[2:])
    assert docling_unit_sequence_hash(changed) != docling_unit_sequence_hash(units)


def test_long_code_preserves_indentation_at_final_chunk_boundaries():
    body = "def measures():\n" + "".join(f"    value_{i} = {i}.14\n" for i in range(220)) + "    return value_219"
    doc = markdown_document('```python\n' + body + '\n```')
    chunks = semantic(doc)[2]
    assert len(chunks) > 1
    assert ''.join(chunk.text for chunk in chunks) == body
    compile(''.join(chunk.text for chunk in chunks), '<chunks>', 'exec')
