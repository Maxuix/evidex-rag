"""One frozen semantic embedding whitespace experiment; source evidence stays intact."""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import numpy as np

from tools.evaluate_chunking_ab import EmbeddingCache, digest, gate_pair, load_models, metrics, support_match, write


def projection(row: dict, code_refs: set[str], *, preserve_paragraphs: bool = False) -> str:
    if any(ref in code_refs for ref in row['source_location'].get('item_refs', ())):
        return row['embedding_text']
    label, body = row['embedding_text'].split('\n', 1)
    if row['modality'] == 'table':
        body = re.sub(r'\n{2,}', '\n', body)
    elif preserve_paragraphs:
        body = '\n\n'.join(re.sub(r'\s+', ' ', part).strip() for part in re.split(r'\n[ \t]*\n+', body))
    else:
        body = re.sub(r'\s+', ' ', body).strip()
    return label + '\n' + body


async def run(args):
    root = args.output
    frozen = json.loads((root/'frozen.json').read_text())
    documents = json.loads((root/'documents.json').read_text())
    baseline = json.loads((root/'first-pass-result.json').read_text())
    rows_path = root/'semantic-fidelity-chunks.json'
    if not rows_path.exists():
        rows_path = root/'semantic_new-chunks.json'
    rows = json.loads(rows_path.read_text())
    identities = json.loads((root/'models.json').read_text())
    assert digest(frozen) == baseline['frozen_sha256']
    assert digest(documents) == frozen['documents_sha256']
    refs = {name: {item['self_ref'] for item in doc['texts'] if item['label'] == 'code'}
            for name, doc in documents.items()}
    texts = tuple(projection(row, refs[row['filename']], preserve_paragraphs=args.preserve_paragraphs) for row in rows)
    stem = 'paragraph-probe' if args.preserve_paragraphs else 'whitespace-probe'
    variant = 'paragraph_prose_single_table_breaks_v1' if args.preserve_paragraphs else 'compact_prose_single_table_breaks_v1'
    manifest = {'variant': variant, 'source_rows_sha256': digest(rows),
                'embedding_inputs_sha256': digest(texts), 'frozen_sha256': digest(frozen),
                'changed_inputs': sum(a != b['embedding_text'] for a,b in zip(texts,rows,strict=True)),
                'source_evidence_unchanged': True}
    manifest_path = root/(stem+'-frozen.json')
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        write(manifest_path, manifest)
    models, current = await load_models(args.primary)
    assert current == identities
    cache = EmbeddingCache(models['text_embedding'], root/'vectors.sqlite', identities['text_embedding'],args.cache_only)
    vectors = np.asarray(await cache.documents(texts))
    queries = np.asarray([await cache.query(case['question']) for case in frozen['cases']])
    scores = queries @ vectors.T
    evaluated = []
    for index, case in enumerate(frozen['cases']):
        ranking = np.argsort(-scores[index], kind='stable')[:max(frozen['top_k'])].tolist()
        rank = next((i+1 for i,pos in enumerate(ranking) if support_match(case, rows[pos])),None)
        evaluated.append({'case_id':case['evaluation_case_id'], 'group':case.get('group','exact_support'),
                          'rank':rank,'top10':ranking})
    before = [{k:v for k,v in row.items() if k not in ('answer_pass','answer_key')}
              for row in baseline['arms']['semantic_old']['cases']]
    result = {'manifest':manifest, 'cases':evaluated,'new_embedding_inputs':cache.missing_inputs,
              'summary':{group:metrics([row for row in evaluated if row['group']==group]) for group in {row['group'] for row in evaluated}},
              'retrieval_gate_failures':gate_pair(before,evaluated),
              'answer_gate_evaluated':False}
    write(root/(stem+'-result.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k != 'cases'},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('.runtime/evaluations/chunking-ab-20260905'))
    parser.add_argument('--cache-only',action='store_true')
    parser.add_argument('--preserve-paragraphs',action='store_true')
    asyncio.run(run(parser.parse_args()))
