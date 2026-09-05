#!/usr/bin/env python3
"""Frozen algorithm comparison on full shared corpora; host-only, no new QA/index."""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

import rag_kb.retrieval.reranker as reranker
from tools.analyze_auto_strategy import paired_interval
from tools.auto_strategy_metrics import evidence_row, paired_changes, summarize
from tools.evaluate_classic_strategy import costs, immutable
from tools.prepare_auto_strategy import digest, jsonl, read, require, sha, write
from tools.rag_algorithm_policies import ARMS, CONTROLS, GENERATIVE, PROMPTS, SYSTEM, CorpusIndex, algorithms, pack
from tools.rag_algorithm_runtime import load_io

ROOT = Path(__file__).resolve().parents[1]
PRIOR = '.runtime/evaluations/auto-strategy-20260905'


def freeze(args):
    old = read(args.primary/PRIOR/'hotpot-frozen.json')
    prior300 = read(args.primary/'.runtime/evaluations/classic-strategy-20260905/fresh-cases.json')
    excluded = {c['original_case_id'] for c in old['cases']+prior300}
    excluded |= set(read(args.primary/'evaluation/hotpotqa-1000-v1/selection.json')['pilot_case_ids'])
    source = args.primary/'evaluation/hotpotqa-1000-v1/cases.jsonl'
    require(sha(source) == read(source.parent/'manifest.json')['artifacts']['cases.jsonl'], 'Hotpot manifest changed')
    cases, docs = jsonl(source), {d['document_id']: d for d in old['documents']}
    fresh = []
    for kind, count in [('bridge', 160), ('comparison', 40)]:
        eligible = [c for c in cases if c['type'] == kind and c['case_id'] not in excluded]
        for c in sorted(eligible, key=lambda c: digest(['rag-algorithms-v1', c['case_id']]))[:count]:
            for fact in c['supporting_facts']:
                require(docs[fact['document_id']]['text'][fact['char_start']:fact['char_end']] == fact['quote'], 'Source quote changed')
            fresh.append({'case_id': 'new:'+c['case_id'], 'original_case_id': c['case_id'],
                'family': 'hotpot-new', 'group': 'hotpot-new:'+c['type'], 'cluster': c['case_id'],
                'query': c['question'], 'answer': c['answer'], 'answerable': c['answerable'],
                'required_paths': [c['required_document_ids']]})
    require(len(fresh) == len({c['case_id'] for c in fresh}) == 200, 'Fresh split incomplete')
    require(not excluded & {c['original_case_id'] for c in fresh}, 'Split leakage')
    reader_ids = [c['case_id'] for c in sorted(fresh, key=lambda c: digest(['rag-algorithms-reader-v1', c['case_id']]))[:80]]
    inputs = [args.primary/PRIOR/f'{f}-{suffix}' for f in ('hotpot', 'musique')
              for suffix in ('frozen.json', 'vectors.npz', 'vectors.json', 'results.json')]
    inputs += [source, args.primary/'.runtime/evaluations/classic-strategy-20260905/fresh-cases.json']
    protocol = {'schema': 'rag_algorithms_v1', 'base_revision': 'bd2bc5f3', 'arms': ARMS,
        'source_sha256': {str(p.relative_to(args.primary)): sha(p) for p in inputs},
        'policy_sha256': sha(ROOT/'tools/rag_algorithm_policies.py'),
        'runtime_sha256': sha(ROOT/'tools/rag_algorithm_runtime.py'),
        'prompts': PROMPTS, 'system': SYSTEM, 'max_output_tokens': 1024, 'schema_repair_attempts': 1,
        'candidate_k_per_query': 40, 'min_cosine': .35, 'bm25_k1': 1.2, 'bm25_b': .75, 'rrf_k': 60,
        'read_k': 10, 'iterative_rounds': 2, 'iterative_queries_per_round': 2,
        'matching': 'whole-source selection with max10 and each query Classic10 text-token cap; no truncation',
        'selection': 'fixed10 and token-capped macro complete-evidence; no family complete/Top1 decrease; macro text ratio <=1.10; prefer cheaper tie',
        'validation': 'frozen selected policy and non-generative controls; no retuning',
        'fresh_sha256': digest(fresh), 'reader_case_ids': reader_ids, 'excluded_ids': sorted(excluded),
        'qa_generation_calls': 0, 'product_changes': False}
    # JSON canonicalization turns tuples into arrays before immutable comparison.
    import json
    immutable(args.output/'protocol.json', json.loads(json.dumps(protocol)))
    immutable(args.output/'fresh-cases.json', fresh)
    print({'frozen': True, 'algorithms': len(ARMS), 'development': 176, 'fresh': 200, 'reader': 80}, flush=True)


def validate(args):
    p = read(args.output/'protocol.json')
    require(p['policy_sha256'] == sha(ROOT/'tools/rag_algorithm_policies.py')
        and p['runtime_sha256'] == sha(ROOT/'tools/rag_algorithm_runtime.py'), 'Frozen algorithm changed')
    require(p['fresh_sha256'] == digest(read(args.output/'fresh-cases.json')), 'Fresh cases changed')
    for name, expected in p['source_sha256'].items():
        require(sha(args.primary/name) == expected, 'Source artifact changed: '+name)
    return p


def load_corpus(args, family):
    base = args.primary/PRIOR
    frozen = read(base/f'{family}-frozen.json')
    binding = read(base/f'{family}-vectors.json')
    require(binding['input_sha256'] == sha(base/f'{family}-frozen.json')
        and binding['vectors_sha256'] == sha(base/f'{family}-vectors.npz'), 'Vector binding differs')
    with np.load(base/f'{family}-vectors.npz', allow_pickle=False) as data:
        index = CorpusIndex(frozen['documents'], data['documents'])
        queries = data['queries'].copy()
    return frozen, index, queries


def compact(rows):
    return summarize([{**r, 'group': r['family']} for r in rows])


def choose(arms):
    baseline = compact(arms['classic65'])
    candidates = []
    for name in ARMS:
        if name == 'classic15':
            continue
        sums, matched = compact(arms[name]), compact(arms[name+'__tokens'])
        valid = all(sums[f]['complete_evidence'] >= b['complete_evidence']
            and matched[f]['complete_evidence'] >= b['complete_evidence']
            and sums[f]['hit1'] >= b['hit1'] for f, b in baseline.items())
        ratio = np.mean([sums[f]['mean_tokens']/b['mean_tokens'] for f, b in baseline.items()])
        score = np.mean([(sums[f]['complete_evidence']+matched[f]['complete_evidence'])/(2*b['answerable']) for f, b in baseline.items()])
        overhead = np.mean([r['overhead']['llm_total_tokens'] for r in arms[name]])
        candidates.append({'name': name, 'eligible': bool(valid and ratio <= 1.10),
            'macro_complete': float(score), 'text_ratio': float(ratio), 'mean_llm_tokens': float(overhead)})
    selected = max([c for c in candidates if c['eligible']], key=lambda c: (c['macro_complete'], -c['mean_llm_tokens'], -c['text_ratio'], c['name'] == 'classic65'))
    return {'selected': selected['name'], 'candidates': candidates}


async def evaluate(args, *, fresh=False):
    validate(args)
    io = await load_io(args, live=not args.cache_only)
    selected = None
    if fresh:
        selection = read(args.output/'selection.json')
        require(selection['development_sha256'] == sha(args.output/'development.json')
            and selection['protocol_sha256'] == sha(args.output/'protocol.json'), 'Selection changed')
        selected = selection['selected']
        requested = tuple(dict.fromkeys([*CONTROLS, selected]))
    else:
        requested = ARMS
    start = time.perf_counter()
    all_records = []
    for family in (('hotpot',) if fresh else ('hotpot', 'musique')):
        frozen, index, vectors = load_corpus(args, family)
        cases = read(args.output/'fresh-cases.json') if fresh else frozen['cases']
        prices = costs({d['id']: d['text'] for d in frozen['documents']})
        expected = {} if fresh else {r['case_id']: r['ids'] for r in read(args.primary/PRIOR/f'{family}-results.json')['arms']['classic10']['cases']}
        case_sem = asyncio.Semaphore(4)
        done = 0
        async def one(n, case):
            nonlocal done
            async with case_sem:
                vector, original_key = (await io.embed(case['query'], kind='query')) if fresh else (vectors[n], None)
                ordered, requests, traces = await algorithms(case['query'], vector, index, io, requested=requested)
                require(case['case_id'] not in expected or ordered['classic65'][:10] == expected[case['case_id']], 'Classic baseline drift')
                gold = {**case, 'labels': {d['id']: [d['document_id']] for d in frozen['documents']}}
                token_cap = sum(prices[i] for i in ordered['classic65'][:10])
                rows = {}
                for name, order in ordered.items():
                    # classic15 is the original top10 plus five whole source
                    # chunks with a 2048-token added budget, matching report 88.
                    choices = pack(order, prices)
                    if name == 'classic15':
                        choices = order[:10]+pack(order[10:], prices, max_items=5, token_budget=2048)
                    for suffix, chosen in [('', choices), ('__tokens', pack(order, prices, token_budget=token_cap))]:
                        row = evidence_row(gold, chosen, order, prices)
                        row.update(overhead=io.cost(requests[name]), candidate_count=len(order))
                        rows[name+suffix] = row
                record = {'case_id': case['case_id'], 'rows': rows, 'traces': traces,
                    'requests': requests, 'original_embedding_key': original_key}
                dest = args.output/'records'/('fresh' if fresh else family)/f'{digest(case["case_id"])}.json'
                if args.cache_only:
                    require(read(dest) == record, 'Cached algorithm replay changed')
                else:
                    immutable(dest, record)
                done += 1
                if done % 10 == 0:
                    print({'family': family, 'cases': done, 'total': len(cases), 'new_llm_calls': io.new_calls,
                        'new_vectors': io.new_vectors, 'seconds': round(time.perf_counter()-start)}, flush=True)
                return record
        with patch.object(reranker, '_terms', lru_cache(maxsize=20000)(reranker._terms)):
            records = await asyncio.gather(*(one(n, c) for n, c in enumerate(cases)))
        all_records.extend(records)
    arms = defaultdict(list)
    for record in all_records:
        for name, row in record['rows'].items():
            arms[name].append(row)
    result = {'status': 'complete', 'arms': dict(arms), 'summaries': {n: compact(r) for n, r in arms.items()},
        'protocol_sha256': sha(args.output/'protocol.json'), 'new_llm_calls': io.new_calls,
        'new_vectors': io.new_vectors, 'qa_generation_calls': 0, 'seconds': time.perf_counter()-start,
        'evaluator_sha256': sha(Path(__file__))}
    stage = 'validation' if fresh else 'development'
    target = args.output/f'{stage}.json'
    if args.cache_only:
        require(read(target)['arms'] == dict(arms), 'Evaluation replay drift')
        write(args.output/f'{stage}-verification.json', {'status': 'complete', 'rankings': sum(map(len, arms.values())),
            'new_llm_calls': io.new_calls, 'new_vectors': io.new_vectors, 'result_sha256': sha(target)})
    else:
        require(not target.exists(), 'Completed evaluation exists; use --cache-only')
        write(target, result)
    if not fresh:
        selection = choose(arms)
        selection.update(development_sha256=sha(target), protocol_sha256=sha(args.output/'protocol.json'))
        immutable(args.output/'selection.json', selection)
        print({'selected': selection['selected']}, flush=True)
    for name in requested:
        print({'policy': name, 'summary': result['summaries'][name]}, flush=True)


def analyze(args):
    validate(args)
    selection = read(args.output/'selection.json')['selected']
    result = {'selected': selection, 'stages': {}}
    for stage in ('development', 'validation'):
        data = read(args.output/f'{stage}.json')
        old = data['arms']['classic65']
        analyses = {}
        for name, rows in data['arms'].items():
            groups = {}
            for family in sorted({r['family'] for r in rows}):
                base = [r for r in old if r['family'] == family]
                current = [r for r in rows if r['family'] == family]
                groups[family] = {'summary': compact(current)[family], 'paired': paired_changes(base, current),
                    'intervals': {m: paired_interval(base, current, m) for m in ('complete_evidence', 'hit1', 'required_fraction')},
                    'mean_overhead': {key: sum(r['overhead'][key] for r in current)/len(current) for key in current[0]['overhead']}}
            analyses[name] = groups
        result['stages'][stage] = analyses
    write(args.output/'analysis.json', result)
    print({'selected': selection, 'fresh': result['stages']['validation'][selection]}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'develop', 'validate', 'analyze'))
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    if args.stage == 'freeze':
        freeze(args)
    elif args.stage == 'analyze':
        analyze(args)
    else:
        asyncio.run(evaluate(args, fresh=args.stage == 'validate'))


if __name__ == '__main__':
    main()
