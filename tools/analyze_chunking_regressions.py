"""Offline source/embedding diagnostics for the frozen chunking A/B campaign."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np

from rag_kb.adapters.model_api.langchain_embeddings import _utf8_windows
from tools.evaluate_chunking_ab import EmbeddingCache, digest, metrics, support_match, write
from tools.diagnose_chunking_context import compact_table_projection, ranked_cases


def cell_projection(text):
    rows = []
    for line in text.split('\n'):
        value = line.strip()
        if value.startswith('|') and value.endswith('|') and '\\|' not in value:
            cells = tuple(cell.strip() for cell in value[1:-1].split('|'))
            if all(re.fullmatch(r':?-+:?', cell) for cell in cells):
                rows.append(tuple((cell.startswith(':'),cell.endswith(':')) for cell in cells))
            else:
                rows.append(cells)
        else:
            rows.append(line)
    return rows


def required_amd_operands(row):
    text = ' '.join(row['text'].split())
    return row['filename']=='AMD_2022_10K.pdf' and all(value in text for value in (
        'Cash and cash equivalents','4,835','Short-term investments','1,020',
        'Accounts receivable','4,126','Receivables from related parties',
        'Total current liabilities','6,369'))


async def run(root):
    frozen=json.loads((root/'frozen.json').read_text())
    first=json.loads((root/'first-pass-result.json').read_text())
    passing=json.loads((root/'passing-result.json').read_text())
    identities=json.loads((root/'models.json').read_text())
    cache=EmbeddingCache(None,root/'vectors.sqlite',identities['text_embedding'],True)
    queries=np.asarray([await cache.query(case['question']) for case in frozen['cases']])
    variants={'legacy':json.loads((root/'semantic_old-chunks.json').read_text()),
              'fidelity':json.loads((root/'semantic-fidelity-chunks.json').read_text()),
              'compact':json.loads((root/'semantic_new-chunks.json').read_text())}
    result={'frozen_sha256':digest(frozen),'new_model_calls':0,'variants':{}}
    amd_index=next(i for i,c in enumerate(frozen['cases']) if c['evaluation_case_id']=='financebench_id_00222')
    for name,rows in variants.items():
        texts=tuple(row['embedding_text'] for row in rows)
        scores=queries@np.asarray(await cache.documents(texts)).T
        cases=ranked_cases(frozen,rows,scores)
        expected=(passing if name=='compact' else first)['arms']['semantic_old' if name=='legacy' else 'semantic_new']['cases']
        assert cases==[{k:v for k,v in row.items() if k not in ('answer_pass','answer_key')} for row in expected]
        ranks=np.argsort(-scores[amd_index],kind='stable').tolist()
        support_rank=next((i+1 for i,pos in enumerate(ranks) if required_amd_operands(rows[pos])),None)
        tables=[text for text,row in zip(texts,rows,strict=True) if row['modality']=='table']
        result['variants'][name]={'baseline_rankings_identical':True,'cases':cases,
            'table_input_bytes':sum(len(text.encode()) for text in tables),
            'table_window_histogram':dict(Counter(len(_utf8_windows(text)) for text in tables)),
            'all_window_histogram':dict(Counter(len(_utf8_windows(text)) for text in texts)),
            'amd_quick_ratio':{'page_label_rank':cases[amd_index]['rank'],
                              'all_required_operands_rank':support_rank,
                              'first_page_hit_has_operands':required_amd_operands(rows[ranks[cases[amd_index]['rank']-1]])}}
    old={row['case_id']:row for row in result['variants']['legacy']['cases']}
    for name in ('fidelity','compact'):
        changes=[]
        for row in result['variants'][name]['cases']:
            prior=old[row['case_id']]
            if row['rank']!=prior['rank']:
                count=metrics([r for r in old.values() if r['group']==row['group']])['count']
                delta=(1/row['rank'] if row['rank'] else 0)-(1/prior['rank'] if prior['rank'] else 0)
                changes.append({'case_id':row['case_id'],'group':row['group'],'old_rank':prior['rank'],
                                'new_rank':row['rank'],'group_mrr_delta':delta/count})
        result['variants'][name]['rank_changes']=changes
    compact=variants['compact'];fidelity=variants['fidelity']
    assert len(compact)==len(fidelity)
    checked=0
    for before,after in zip(fidelity,compact,strict=True):
        assert {k:v for k,v in before.items() if k!='embedding_text'}=={k:v for k,v in after.items() if k!='embedding_text'}
        assert after['embedding_text']==compact_table_projection(before)
        if before['modality']=='table':
            assert cell_projection(before['embedding_text'])==cell_projection(after['embedding_text'])
            checked+=1
    result['table_cell_preservation_checks']=checked
    result['source_evidence_unchanged']=True
    write(root/'root-cause-analysis.json',result)
    cache.database.close()
    print(json.dumps({'table_cell_preservation_checks':checked,'source_evidence_unchanged':True,
        'variants':{k:{n:v for n,v in values.items() if n not in ('cases','rank_changes')} for k,values in result['variants'].items()}}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('.runtime/evaluations/chunking-ab-20260905'))
    asyncio.run(run(parser.parse_args().output))
