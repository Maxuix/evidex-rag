#!/usr/bin/env python3
"""One source-context embedding ablation, reusing every old query and QA."""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

import numpy as np

from tools.analyze_source_recall import (
    ROOT, STATE, DIAGNOSIS, REPLAY, OUTPUT as RANK_ROOT,
    inputs, read, write, require, digest, hit, classic_order, row_result,
)
from tools.analyze_auto_qa_ranking import summarize
from tools.evaluate_source_recall import OUTPUT, PROBES


def projection(chunk, body, introduction):
    filename=chunk['source_metadata']['original_filename'][:256]
    titles=' > '.join(t['text'] for t in chunk['hierarchy'].get('titles',[]) if t.get('text'))[:512]
    prefix=[f'[document]\n{filename}']
    if introduction and introduction not in body:
        prefix.append(f'[document introduction]\n{introduction}')
    if titles:
        prefix.append(f'[section]\n{titles}')
    return '\n\n'.join(prefix+[body])


async def embed(args):
    from sqlalchemy import text
    from apps.api.dependencies import build_api_dependencies
    from rag_kb.config import load_settings
    from rag_kb.domain import RetrievalQueryPlan,RetrievalStrategy,RerankMode
    from tools.evaluation_runtime import load_evaluation_runtime
    corpus,snapshot,_,_,_=inputs()
    rt=load_evaluation_runtime();settings=load_settings(env_file=rt.env_file)
    url=urlsplit(settings.database.runtime_dsn.get_secret_value())
    require(url.hostname in {'127.0.0.1','localhost'} and url.port==rt.ports['postgres'],'Not isolated host runtime')
    state=read(STATE)
    requirements={'chat':'mimo-v2.5','text_embedding':'qwen3.7-text-embedding',
                  'multimodal_embedding':'tongyi-embedding-vision-flash-2026-03-06'}
    require(all(state['profiles'][k]['model']==v for k,v in requirements.items()),'Frozen model configuration differs')
    arm=state['arms']['on'];meta=read(RANK_ROOT/'exact-ranks.json')
    deps=build_api_dependencies(settings=settings)
    try:
        plan=RetrievalQueryPlan(settings.identity.workspace_id,UUID(arm['knowledge_base_id']),
            RetrievalStrategy.EXACT_VECTOR,top_k=10,candidate_count=40,rerank_mode=RerankMode.CLASSIC,auto_qa_candidate_count=0)
        provider=await deps.retrieval_service._text_embedding_provider(plan)
        require(provider.embedding_space.requested_model==requirements['text_embedding'] and
                provider.embedding_space.compatibility_fingerprint==meta['compatibility_fingerprint'],'Embedding space changed')
        async with deps.database.sessions() as session,session.begin():
            await session.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY'))
            rows=(await session.execute(text('''
                SELECT c.id,c.content,c.embedding_text FROM index_chunk c
                JOIN indexed_document_version t ON t.id=c.indexed_document_version_id
                WHERE c.workspace_id=:workspace AND c.kb_id=:kb AND t.index_revision_id=:revision
            '''),{'workspace':settings.identity.workspace_id,'kb':plan.knowledge_base_id,'revision':UUID(arm['index_revision_id'])})).mappings().all()
        require({str(r['id']) for r in rows}==set(snapshot['chunks']),'Source inventory changed')
        bodies={str(r['id']):r['embedding_text'] or r['content'] for r in rows}
        require(all(r['content']==snapshot['chunks'][str(r['id'])]['content'] for r in rows),'Source text changed')
        intros={}
        for c in sorted(snapshot['chunks'].values(),key=lambda c:c['ordinal']):
            intros.setdefault(c['source_metadata']['document_version_id'],c['content'].split('\n\n')[0][:512])
        ids=sorted(bodies)
        texts={i:projection(snapshot['chunks'][i],bodies[i],intros[snapshot['chunks'][i]['source_metadata']['document_version_id']]) for i in ids}
        write(OUTPUT/'context-inputs.json',{'source_sha256':digest(DIAGNOSIS/'sources.json'),'texts':texts,
            'fingerprint':meta['compatibility_fingerprint'],'projection_sha256':digest(Path(__file__))})
        if args.prepare_only:
            endpoint=urlsplit(provider._model_arguments['base_url'])
            print(json.dumps({'prepared_sources':len(texts),'destination':endpoint.scheme+'://'+endpoint.netloc+endpoint.path,
                'model':provider.embedding_space.requested_model,'source_filenames':sorted({c['source_metadata']['original_filename'] for c in snapshot['chunks'].values()}),
                'model_calls':0,'qa_generation_calls':0},ensure_ascii=False),flush=True)
            return
        vectors={};completed=0;new_batches=0
        cache_dir=OUTPUT/'context-batches';cache_dir.mkdir(mode=0o700,parents=True,exist_ok=True)
        semaphore=asyncio.Semaphore(4)
        async def batch(batch_ids):
            nonlocal completed,new_batches
            payload={'fingerprint':meta['compatibility_fingerprint'],'ids':batch_ids,'texts':[texts[i] for i in batch_ids]}
            key=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
            cache=cache_dir/(key+'.json')
            async with semaphore:
                if cache.exists():
                    cached=read(cache);require(cached['input_sha256']==key,'Embedding batch cache mismatch')
                    values=cached['vectors']
                else:
                    require(not args.cache_only,'Context embedding cache missing; no fallback allowed')
                    result=await provider.embed_documents(tuple(payload['texts']))
                    values=result.vectors
                    require(len(values)==len(batch_ids) and all(len(v)==meta['dimension'] and
                        all(math.isfinite(x) for x in v) and sum(x*x for x in v)>0 for v in values),'Invalid provider vectors')
                    write(cache,{'input_sha256':key,'vectors':values})
                    new_batches+=1
                require(len(values)==len(batch_ids) and all(len(v)==meta['dimension'] and
                    all(math.isfinite(x) for x in v) and sum(x*x for x in v)>0 for v in values),'Invalid cached vectors')
                vectors.update(zip(batch_ids,values,strict=True));completed+=len(batch_ids)
                if completed%100==0 or completed==len(ids):
                    print(json.dumps({'source_vectors':completed,'total':len(ids),'new_batches':new_batches,'qa_generation_calls':0}),flush=True)
        size=provider.max_batch_size
        # TaskGroup cancels pending batches on any error; no provider/model substitution.
        async with asyncio.TaskGroup() as group:
            for start in range(0,len(ids),size):group.create_task(batch(ids[start:start+size]))
        write(OUTPUT/'context-vectors.json',{'corpus_sha256':corpus['dataset_sha256'],'source_sha256':digest(DIAGNOSIS/'sources.json'),
            'inputs_sha256':digest(OUTPUT/'context-inputs.json'),'fingerprint':meta['compatibility_fingerprint'],
            'new_embedding_batches':new_batches,'qa_generation_calls':0,'vectors':vectors})
    finally:
        await deps.close()


def evaluate():
    corpus,snapshot,baseline,_,cases=inputs();stored=read(OUTPUT/'context-vectors.json')
    require(stored['source_sha256']==digest(DIAGNOSIS/'sources.json') and
            stored['corpus_sha256']==corpus['dataset_sha256'],'Source snapshot changed')
    projected=read(OUTPUT/'context-inputs.json')
    require(projected['source_sha256']==stored['source_sha256'] and
            projected['fingerprint']==stored['fingerprint'],'Projection snapshot changed')
    ids=sorted(stored['vectors']);matrix=np.array([stored['vectors'][i] for i in ids],dtype=np.float64)
    matrix/=np.linalg.norm(matrix,axis=1,keepdims=True)
    queries=read(REPLAY/'query-vectors.json');qvec=[]
    for c in cases:
        key=hashlib.sha256((stored['fingerprint']+'\n'+c['question']).encode()).hexdigest()
        require(key in queries,'Frozen query missing');qvec.append(queries[key])
    qmat=np.array(qvec,dtype=np.float64);qmat/=np.linalg.norm(qmat,axis=1,keepdims=True)
    similarities=qmat@matrix.T
    probes={p['case_id']:{r['chunk_id'] for r in p['required']} for p in read(PROBES)['cases']}
    arms={n:[] for n in ['context40','context100','context40_aligned']};details=[]
    old={r['case_id']:r for r in baseline['arms']['classic_source']['cases']}
    for c,sim in zip(cases,similarities,strict=True):
        cid=c['evaluation_case_id'];order=sorted(range(len(ids)),key=lambda n:(-sim[n],UUID(ids[n]).int))
        if cid in probes:
            ranks={ids[n]:rank for rank,n in enumerate(order,1)}
            details.append({'case_id':cid,'required_ranks':{i:ranks[i] for i in probes[cid]}})
        for name,limit in [('context40',40),('context100',100),('context40_aligned',40)]:
            hits=[hit(ids[n],float(1-sim[n]),snapshot['chunks'],c) for n in order[:limit] if sim[n]>=.35]
            if name.endswith('_aligned'):
                # label_match was computed from original source evidence above;
                # only the experimental lexical scoring input is projected.
                for h in hits:
                    h.text=projected['texts'][str(h.index_chunk_id)]
            final=[s.hit for s in classic_order(c['question'],hits)]
            row=row_result(c,final);row.update(candidate_match=any(h.label_match for h in hits),
                candidate_ids=[str(h.index_chunk_id) for h in hits])
            if cid in probes:
                row.update(probe_candidate_complete=probes[cid]<=set(row['candidate_ids']),
                           probe_final_complete=probes[cid]<=set(row['final_ids']))
            arms[name].append(row)
    result={'corpus_sha256':corpus['dataset_sha256'],'qa_generation_calls':0,'details':details,'arms':{}}
    for name,rows in arms.items():
        summary=summarize(rows)
        for g in ['direct','paraphrase']:
            sel=[r for r in rows if r['group']==g]
            summary[g].update(candidate_coverage=sum(r['candidate_match'] for r in sel),
                lost_top1=[r['case_id'] for r in sel if old[r['case_id']]['hit_1'] and not r['hit_1']],
                lost_top10=[r['case_id'] for r in sel if old[r['case_id']]['recall_10'] and not r['recall_10']])
        result['arms'][name]={'summary':summary,'cases':rows}
        print(name,json.dumps(summary,ensure_ascii=False),'probes',sum(r.get('probe_candidate_complete',False) for r in rows),sum(r.get('probe_final_complete',False) for r in rows),flush=True)
    write(OUTPUT/'context-comparison.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embed',action='store_true');parser.add_argument('--cache-only',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.embed or args.prepare_only:asyncio.run(embed(args))
    if args.prepare_only:return
    evaluate()


if __name__=='__main__':main()
