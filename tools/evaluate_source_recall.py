#!/usr/bin/env python3
"""Compare bounded source-only recall on frozen vectors; never call a model."""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import hashlib
import json
import re
import math
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import UUID

from tools.analyze_source_recall import (
    ROOT, CORPUS, DIAGNOSIS, OUTPUT as RANK_ROOT, REPLAY, STATE,
    inputs, read, digest, write, require, hit, classic_order, row_result,
)
from tools.analyze_auto_qa_ranking import summarize
from rag_kb.retrieval.fusion import reciprocal_rank_fusion_lanes
from rag_kb.retrieval.reranker import _score, _terms
from rag_kb.retrieval.source_context import (
    SOURCE_CONTEXT_ANCHOR_LIMIT, rank_with_source_context, table_neighbor_compatible,
)
from collections import Counter

OUTPUT = ROOT / ".runtime/evaluations/source-recall-repair-20260905"
PROBES = CORPUS / "source-recall-probes.json"


async def export_context_terms(path):
    """Use PostgreSQL's installed English stemmer plus existing CJK analysis, read-only."""
    from sqlalchemy import text
    from rag_kb.config import load_settings
    from rag_kb.db import DatabaseProcess, create_database_resources
    from rag_kb.document_processing.lexical import analyze_document, analyze_query
    from tools.evaluation_runtime import load_evaluation_runtime
    corpus,snapshot,_,_,cases=inputs()
    rt=load_evaluation_runtime();settings=load_settings(env_file=rt.env_file)
    url=urlsplit(settings.database.runtime_dsn.get_secret_value())
    require(url.hostname in {'127.0.0.1','localhost'} and url.port==rt.ports['postgres'],'Not isolated host DB')
    records=[]
    for i,c in snapshot['chunks'].items():
        titles='\n'.join(t['text'] for t in c['hierarchy'].get('titles',[]) if t.get('text'))
        for kind,body in [('body',c['content']),('context',titles+'\n'+c['content'])]:
            lex=analyze_document(body)
            records.append({'id':kind+':'+i,'value':lex.lexical_text if lex else ''})
    records.extend({'id':'query:'+c['evaluation_case_id'],'value':' '.join(analyze_query(c['question']))} for c in cases)
    terms={}
    async with create_database_resources(settings.database.runtime_dsn.get_secret_value(),pool_size=1,max_overflow=0,
            process=DatabaseProcess.MAINTENANCE) as db:
        async with db.sessions() as session,session.begin():
            await session.execute(text('SET TRANSACTION READ ONLY'))
            for start in range(0,len(records),200):
                rows=(await session.execute(text("""
                    SELECT x.id,tsvector_to_array(to_tsvector('english',x.value)) AS terms
                    FROM jsonb_to_recordset(CAST(:records AS jsonb)) AS x(id text,value text)
                """),{'records':json.dumps(records[start:start+200])})).mappings().all()
                terms.update({r['id']:r['terms'] for r in rows})
    write(path,{'source_sha256':digest(DIAGNOSIS/'sources.json'),'corpus_sha256':corpus['dataset_sha256'],
                'terms':terms,'model_calls':0,'database_writes':0})


def idf_lexical(terms, chunks, case_id, kind):
    docs={i:set(terms[kind+':'+i]) for i in chunks}
    frequency=Counter(t for row in docs.values() for t in row)
    average=sum(map(len,docs.values()))/len(docs)
    query=set(terms['query:'+case_id])
    scores={i:sum(math.log1p((len(docs)-frequency[t]+.5)/(frequency[t]+.5))*2.2/
                       (1+1.2*(.25+.75*len(row)/average)) for t in query & row) for i,row in docs.items()}
    return [i for i in sorted(scores,key=lambda i:(-scores[i],UUID(i).int)) if scores[i]>0]


def stable_order(query, hits, reference):
    """Diagnostic: freeze BM25 statistics to original dense-40 while scoring extras."""
    terms = [_terms(h.text) for h in reference]
    frequency = Counter(t for row in terms for t in set(row))
    average = sum(map(len, terms))/len(terms)
    scored = [_score(h, _terms(h.text), _terms(query), frequency, len(reference), average, .65, .35) for h in hits]
    return [s.hit for s in sorted(scored, key=lambda s: (-s.score,s.hit.cosine_distance,s.hit.index_chunk_id.int))]


def table_neighbor(left, right):
    if 'table' not in {left['modality'],right['modality']}:
        return False
    if left['source_metadata']['document_version_id'] != right['source_metadata']['document_version_id']:
        return False
    lt=[x.get('text','') for x in left['hierarchy'].get('titles',[]) if x.get('text')]
    rt=[x.get('text','') for x in right['hierarchy'].get('titles',[]) if x.get('text')]
    if lt and rt and lt!=rt:
        return False
    ls,rs=left['source_location'],right['source_location']
    if ls.get('surface_type')!=rs.get('surface_type'):
        return False
    if ls.get('surface_type')=='page':
        a,b=ls.get('surface_start'),rs.get('surface_start')
        if not isinstance(a,int) or not isinstance(b,int) or abs(a-b)>1:
            return False
    return True


def enrich(ids, ordered, chunks, by_ordinal):
    expanded=list(ids)
    for anchor in ordered[:20]:
        source=chunks[str(anchor.index_chunk_id)]
        for offset in [-1,1]:
            key=(source['source_metadata']['document_version_id'],source['ordinal']+offset)
            i=by_ordinal.get(key)
            if i and i not in expanded and table_neighbor(source,chunks[i]):
                expanded.append(i)
    require(len(expanded)<=len(ids)+40 and len(expanded)<=320,'Expansion budget exceeded')
    return expanded


async def export_lexical(path):
    from pgvector.sqlalchemy import Vector
    from sqlalchemy import text, bindparam
    from rag_kb.adapters.lexical_store.postgres import PgLexicalStore
    from rag_kb.config import load_settings
    from rag_kb.db import DatabaseProcess, create_database_resources
    from rag_kb.document_processing.lexical import (
        LEXICAL_ANALYZER_VERSION, analyze_document, analyze_query, build_or_tsquery,
    )
    from tools.evaluation_runtime import load_evaluation_runtime

    corpus, snapshot, _, _, cases = inputs()
    state = read(STATE)
    require(state["status"] == "indexed" and state["corpus_sha256"] == corpus["dataset_sha256"], "State changed")
    rt = load_evaluation_runtime()
    settings = load_settings(env_file=rt.env_file)
    url = urlsplit(settings.database.runtime_dsn.get_secret_value())
    require(url.hostname in {"127.0.0.1", "localhost"} and url.port == rt.ports["postgres"], "Not isolated host DB")
    arm = state["arms"]["off"]
    params = {"workspace_id": settings.identity.workspace_id, "kb_id": UUID(arm["knowledge_base_id"]),
              "revision_id": UUID(arm["index_revision_id"]), "analyzer_version": LEXICAL_ANALYZER_VERSION}
    require(not arm["auto_qa"]["enabled"], "Source-only lexical index required")
    on_by_key = {(c["source_metadata"]["checksum_sha256"], c["ordinal"]): (i, c)
                 for i, c in snapshot["chunks"].items()}
    query_cache = read(REPLAY / "query-vectors.json")
    dense_meta = read(RANK_ROOT / "exact-ranks.json")
    result = {"source_only": True, "model_calls": 0, "qa_generation_calls": 0, "database_writes": 0,
              "corpus_sha256": corpus["dataset_sha256"], "source_sha256": digest(DIAGNOSIS / "sources.json"),
              "query_cache_sha256": digest(REPLAY / "query-vectors.json"), "cases": []}
    async with create_database_resources(settings.database.runtime_dsn.get_secret_value(), pool_size=1,
            max_overflow=0, process=DatabaseProcess.MAINTENANCE) as db:
        async with db.sessions() as session, session.begin():
            await session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
            await session.execute(text("SET LOCAL statement_timeout='30s'"))
            resolved = await PgLexicalStore._serving_scope(session, workspace_id=params["workspace_id"],
                                                         knowledge_base_id=params["kb_id"])
            require(resolved is not None and resolved[0] == params["revision_id"], "Off source revision changed")
            await PgLexicalStore._validate_manifests(session, resolved[1], analyzer_version=LEXICAL_ANALYZER_VERSION)
            source_rows = (await session.execute(text("""
                SELECT c.id,c.ordinal,c.content,c.embedding_text,v.checksum_sha256,l.lexical_text_hash
                FROM index_chunk c JOIN indexed_document_version t ON t.id=c.indexed_document_version_id
                JOIN document_version v ON v.id=t.document_version_id
                JOIN index_chunk_lexical l ON l.index_chunk_id=c.id
                  AND l.workspace_id=c.workspace_id AND l.kb_id=c.kb_id
                WHERE c.workspace_id=:workspace_id AND c.kb_id=:kb_id
                  AND t.index_revision_id=:revision_id AND l.analyzer_version=:analyzer_version
            """), params)).mappings().all()
            require(len(source_rows) == len(on_by_key) == 2480, "Lexical source inventory changed")
            mapped = {}
            for row in source_rows:
                i, source = on_by_key[(row["checksum_sha256"], row["ordinal"])]
                require(row["content"] == source["content"], "Off/on source differs")
                lexical = analyze_document(row["embedding_text"] or row["content"])
                require(lexical and lexical.lexical_text_hash == row["lexical_text_hash"], "QA-contaminated lexical row")
                mapped[str(row["id"])] = i
            statement = PgLexicalStore._statement().bindparams(
                bindparam("query_embedding", type_=Vector(dense_meta["dimension"])))
            for position, case in enumerate(cases, 1):
                query = case["question"]
                key = hashlib.sha256((dense_meta["compatibility_fingerprint"] + "\n" + query).encode()).hexdigest()
                require(key in query_cache, "Query cache missing; no model fallback")
                tsquery = build_or_tsquery(analyze_query(query))
                rows = [] if tsquery is None else (await session.execute(statement, {**params,
                    "target_ids": list(resolved[1]), "tsquery": tsquery, "query_embedding": query_cache[key],
                    "embedding_dimension": dense_meta["dimension"], "candidate_count": 2481})).mappings().all()
                # Export every match before mapping IDs: UUID ties at a quota boundary must
                # use the on-arm identities used by the fixed dense snapshot, not off-arm IDs.
                ranks = sorted([[mapped[str(r["index_chunk_id"])], float(r["lexical_score"])] for r in rows],
                               key=lambda r: (-r[1], UUID(r[0]).int))
                result["cases"].append({"case_id": case["evaluation_case_id"], "ranks": ranks})
                if position % 10 == 0 or position == len(cases):
                    print(json.dumps({"lexical_queries": position, "total": len(cases)}), flush=True)
    result["status"] = "complete"
    write(path, result)


def evaluate(args):
    corpus, snapshot, baseline, _, cases = inputs()
    lexical = read(args.lexical)
    require(lexical["status"] == "complete" and lexical["source_only"]
            and lexical["source_sha256"] == digest(DIAGNOSIS / "sources.json")
            and lexical["corpus_sha256"] == corpus["dataset_sha256"], "Invalid lexical snapshot")
    dense = {r["case_id"]: r["ranks"] for r in read(RANK_ROOT / "exact-ranks.json")["cases"]}
    lex = {r["case_id"]: r["ranks"] for r in lexical["cases"]}
    chunks = snapshot["chunks"]
    by_ordinal={(c['source_metadata']['document_version_id'],c['ordinal']):i for i,c in chunks.items()}
    probes={c['case_id']:c['required'] for c in read(PROBES)['cases']}
    for cid,required in probes.items():
        require(cid in dense,'Probe query missing')
        for p in required:
            require(p['chunk_id'] in chunks and re.sub(r'\s+',' ',p['quote']) in
                    re.sub(r'\s+',' ',chunks[p['chunk_id']]['content']),'Unverified evidence probe')
    arms = defaultdict(list)
    term_path=OUTPUT/'context-terms.json'
    term_snapshot=read(term_path) if term_path.exists() else None
    if term_snapshot:
        require(term_snapshot['source_sha256']==digest(DIAGNOSIS/'sources.json'),'Context terms changed')
    for case in cases:
        cid = case["evaluation_case_id"]
        distances = dict(dense[cid])
        reference=[hit(i,d,chunks,case) for i,d in dense[cid][:40] if 1-d>=.35]
        def source_hit(i):
            value = hit(i, distances[i], chunks, case)
            value.indexed_document_version_id = UUID(chunks[i]['source_metadata']['document_version_id'])
            value.ordinal = chunks[i]['ordinal']
            return value
        core = tuple(source_hit(str(h.index_chunk_id)) for h in reference)
        selected_ids = {str(h.index_chunk_id) for h in core}
        supplements = []
        for i, distance in dense[cid][:SOURCE_CONTEXT_ANCHOR_LIMIT]:
            if 1-distance < .35 or chunks[i]['modality'] not in {'text', 'table'}:
                continue
            anchor = source_hit(i)
            for offset in (-1, 1):
                neighbor_id = by_ordinal.get((chunks[i]['source_metadata']['document_version_id'], anchor.ordinal+offset))
                if neighbor_id and neighbor_id not in selected_ids:
                    neighbor = source_hit(neighbor_id)
                    if table_neighbor_compatible(anchor, neighbor):
                        supplements.append(neighbor)
                        selected_ids.add(neighbor_id)
        ranked = rank_with_source_context(case['question'], core, tuple(supplements), top_k=10)
        selected = row_result(case, [s.hit for s in ranked])
        selected.update(candidate_ids=[str(h.index_chunk_id) for h in (*core, *supplements)],
                        candidate_count=len(core)+len(supplements),
                        candidate_match=any(h.label_match for h in (*core, *supplements)))
        arms['protected_table_context'].append(selected)
        for name, dcount, lcount in [("dense40",40,0),("dense80",80,0),("dense100",100,0),
                                    ("dense40_lex40",40,40),("dense100_lex40",100,40)]:
            ids = list(dict.fromkeys([i for i,d in dense[cid][:dcount] if 1-d>=.35]
                                    + [i for i,_ in lex[cid][:lcount]]))
            hits = [hit(i, distances[i], chunks, case) for i in ids]
            order = [s.hit for s in classic_order(case["question"], hits)]
            row = row_result(case, order)
            row.update(candidate_ids=ids,candidate_count=len(ids),candidate_match=any(h.label_match for h in hits))
            arms[name].append(row)
            if name != 'dense40':
                # Apply the same prefix and statistics protection to competing
                # pool strategies, so table expansion is not favored by the guard.
                reference_ids = {h.index_chunk_id for h in reference}
                extras = tuple(h for h in hits if h.index_chunk_id not in reference_ids)
                guarded = rank_with_source_context(case['question'], tuple(reference), extras, top_k=10)
                guarded_row = row_result(case, [s.hit for s in guarded])
                guarded_row.update(candidate_ids=ids, candidate_count=len(ids),
                                   candidate_match=any(h.label_match for h in hits))
                arms['protected_'+name].append(guarded_row)
            if name in {'dense40','dense100','dense40_lex40','dense100_lex40'}:
                expanded=enrich(ids,order,chunks,by_ordinal)
                extra_hits=[hit(i,distances[i],chunks,case) for i in expanded]
                for suffix,selected in [('adj20',[s.hit for s in classic_order(case['question'],extra_hits)]),
                                        ('adj20_stable',stable_order(case['question'],extra_hits,reference))]:
                    extra=row_result(case,selected)
                    extra.update(candidate_ids=expanded,candidate_count=len(expanded),candidate_match=any(h.label_match for h in extra_hits))
                    arms[name+'_'+suffix].append(extra)
        for dcount in [40,100]:
            dids=[i for i,d in dense[cid][:dcount] if 1-d>=.35];lids=[i for i,_ in lex[cid][:40]]
            def make(i):
                h=hit(i,distances[i],chunks,case)
                h.indexed_document_version_id=UUID(chunks[i]['source_metadata']['document_version_id'])
                h.evidence_group_key=None
                h.representation_labels=lambda: ('text',)
                return h
            fused=reciprocal_rank_fusion_lanes((('dense_text',tuple(make(i) for i in dids),1000000),
                ('lexical',tuple(make(i) for i in lids),1000000)),top_k=len(dids)+len(lids))
            ids=[str(f.hit.index_chunk_id) for f in fused]
            order=[hit(i,distances[i],chunks,case) for i in ids]
            row=row_result(case,order);row.update(candidate_ids=ids,candidate_count=len(ids),candidate_match=any(h.label_match for h in order))
            arms[f"existing_rrf{dcount}_40"].append(row)
        if term_snapshot:
            for kind in ['body','context']:
                lids=idf_lexical(term_snapshot['terms'],chunks,cid,kind)[:40]
                ids=list(dict.fromkeys([str(h.index_chunk_id) for h in reference]+lids))
                hits=[hit(i,distances[i],chunks,case) for i in ids]
                initial=[s.hit for s in classic_order(case['question'],hits)]
                expanded=enrich(ids,initial,chunks,by_ordinal)
                for suffix,selected_ids in [('','' ),('_adj20',expanded)]:
                    selected_ids=ids if not suffix else selected_ids
                    eh=[hit(i,distances[i],chunks,case) for i in selected_ids]
                    order=[s.hit for s in classic_order(case['question'],eh)]
                    r=row_result(case,order);r.update(candidate_ids=selected_ids,candidate_count=len(eh),candidate_match=any(h.label_match for h in eh))
                    arms['dense40_idf_'+kind+suffix].append(r)
    result={"status":"complete","model_calls":0,"qa_generation_calls":0,"source_only":True,
            "corpus_sha256":corpus["dataset_sha256"],"probes_sha256":digest(PROBES),"arms":{}}
    old={r['case_id']:r for r in baseline['arms']['classic_source']['cases']}
    for name,rows in arms.items():
        summary=summarize(rows)
        checked=[]
        for r in rows:
            if r['case_id'] in probes:
                required={p['chunk_id'] for p in probes[r['case_id']]}
                check={'case_id':r['case_id'],'required':len(required),'candidate_found':len(required & set(r['candidate_ids'])),
                       'final_found':len(required & set(r['final_ids']))}
                check.update(candidate_complete=check['candidate_found']==len(required),final_complete=check['final_found']==len(required))
                checked.append(check)
        for group in ["direct","paraphrase"]:
            selected=[r for r in rows if r['group']==group]
            summary[group].update(candidate_coverage=sum(r['candidate_match'] for r in selected),
                mean_candidates=round(sum(r['candidate_count'] for r in selected)/len(selected),1),
                lost_top1=[r['case_id'] for r in selected if old[r['case_id']]['hit_1'] and not r['hit_1']],
                lost_top10=[r['case_id'] for r in selected if old[r['case_id']]['recall_10'] and not r['recall_10']])
        result['arms'][name]={'summary':summary,'cases':rows,'probes':checked}
        print(name,json.dumps({'summary':summary,'probe_candidates':sum(p['candidate_complete'] for p in checked),
                              'probe_final':sum(p['final_complete'] for p in checked)},ensure_ascii=False),flush=True)
    require([r['final_ids'] for r in arms['dense40']] == [old[c['evaluation_case_id']]['final_ids'] for c in cases],
            "Classic baseline does not reproduce")
    write(args.output,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-lexical',action='store_true')
    parser.add_argument('--export-context-terms',action='store_true')
    parser.add_argument('--lexical',type=Path,default=OUTPUT/'lexical-ranks.json')
    parser.add_argument('--output',type=Path,default=OUTPUT/'comparison.json')
    args=parser.parse_args()
    if args.export_lexical:asyncio.run(export_lexical(args.lexical))
    if args.export_context_terms:asyncio.run(export_context_terms(OUTPUT/'context-terms.json'))
    evaluate(args)


if __name__=='__main__':main()
