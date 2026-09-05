"""Frozen source-heading context experiment for semantic chunk embeddings."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

from rag_kb.document_processing.tokenization import count_chunk_tokens, split_by_tokens
from tools.evaluate_chunking_ab import EmbeddingCache, digest, gate_pair, load_models, metrics, support_match, write


def context_projection(row: dict) -> str:
    original = row['embedding_text']
    body = ' '.join(row['text'].split()).casefold()
    titles = list(dict.fromkeys(
        title['text'].strip() for title in row['hierarchy'].get('titles', ())
        if title['text'].strip() and ' '.join(title['text'].split()).casefold() not in body
    ))
    if not titles:
        return original
    context = ' > '.join(titles)
    if count_chunk_tokens(context) > 128:
        context = split_by_tokens(context, max_tokens=128, overlap_tokens=0)[0]
    value = '[section]\n' + context + '\n' + original
    if count_chunk_tokens(value) > 1200:
        raise ValueError('context exceeds frozen embedding input budget')
    return value


def compact_table_projection(row: dict) -> str:
    if row['modality'] != 'table':
        return row['embedding_text']
    from rag_kb.document_processing.composite_text import compact_markdown_table
    return compact_markdown_table(row['embedding_text'])


def ranked_cases(frozen, rows, scores):
    evaluated = []
    for index, case in enumerate(frozen['cases']):
        ranking = np.argsort(-scores[index], kind='stable')[:max(frozen['top_k'])].tolist()
        rank = next((i+1 for i,pos in enumerate(ranking) if support_match(case, rows[pos])),None)
        evaluated.append({'case_id':case['evaluation_case_id'], 'group':case.get('group','exact_support'),
                          'rank':rank,'top10':ranking})
    return evaluated


async def run(args):
    root = args.output
    frozen = json.loads((root/'frozen.json').read_text())
    baseline = json.loads((root/'first-pass-result.json').read_text())
    rows_path = root/'semantic-fidelity-chunks.json'
    if not rows_path.exists():
        rows_path = root/'semantic_new-chunks.json'
    rows = json.loads(rows_path.read_text())
    identities = json.loads((root/'models.json').read_text())
    assert digest(frozen) == baseline['frozen_sha256']
    projection = compact_table_projection if args.compact_tables else context_projection
    stem = 'compact-table' if args.compact_tables else 'context'
    texts = tuple(projection(row) for row in rows)
    manifest = {'variant':'compact_table_padding_v1' if args.compact_tables else 'missing_source_headings_v1', 'source_rows_sha256':digest(rows),
                'embedding_inputs_sha256':digest(texts), 'frozen_sha256':digest(frozen),
                'changed_inputs':sum(a != b['embedding_text'] for a,b in zip(texts,rows,strict=True)),
                'source_evidence_unchanged':True, 'max_context_tokens':128}
    path = root/(stem+'-probe-frozen.json')
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        write(path,manifest)
    models,current = await load_models(args.primary)
    assert current == identities
    cache = EmbeddingCache(models['text_embedding'],root/'vectors.sqlite',identities['text_embedding'],args.cache_only)
    queries = np.asarray([await cache.query(case['question']) for case in frozen['cases']])
    original = ranked_cases(frozen,rows,queries @ np.asarray(await cache.documents(tuple(row['embedding_text'] for row in rows))).T)
    assert original == [{k:v for k,v in row.items() if k not in ('answer_pass','answer_key')}
                        for row in baseline['arms']['semantic_new']['cases']]
    scores = queries @ np.asarray(await cache.documents(texts)).T
    evaluated = ranked_cases(frozen,rows,scores)
    prior = [{k:v for k,v in row.items() if k not in ('answer_pass','answer_key')}
             for row in baseline['arms']['semantic_old']['cases']]
    result = {'manifest':manifest, 'cases':evaluated,'new_embedding_inputs':cache.missing_inputs,
              'summary':{g:metrics([r for r in evaluated if r['group']==g]) for g in {r['group'] for r in evaluated}},
              'retrieval_gate_failures':gate_pair(prior,evaluated),'answer_gate_evaluated':False,
              'baseline_rankings_identical':True}
    write(root/(stem+'-probe-result.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('.runtime/evaluations/chunking-ab-20260905'))
    parser.add_argument('--cache-only',action='store_true')
    parser.add_argument('--compact-tables',action='store_true')
    asyncio.run(run(parser.parse_args()))
