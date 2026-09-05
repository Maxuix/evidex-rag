#!/usr/bin/env python3
"""Classic search with historical development and separately frozen Hotpot validation.

All writes use --output. --primary supplies read-only source caches and the already
isolated host model configuration. No Docker, indexing, QA generation, or installs.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

import numpy as np

import rag_kb.retrieval.reranker as runtime_reranker
from rag_kb.retrieval.source_context import rank_with_source_context
from rag_kb.tokenizer import get_cl100k_base_encoding
from tools.auto_strategy_metrics import evidence_row, paired_changes, summarize
from tools.analyze_auto_strategy import paired_interval
from tools.classic_strategy import POLICIES, choose, ids, policy_grid, rank
from tools.prepare_auto_strategy import digest, jsonl, read, require, sha, write

ROOT = Path(__file__).resolve().parents[1]
PRIOR = '.runtime/evaluations/auto-strategy-20260905'


def immutable(path, value):
    if path.exists():
        require(read(path) == value, f'Frozen artifact differs: {path.name}')
    else:
        write(path, value)


def costs(documents):
    encoder = get_cl100k_base_encoding()
    return {i: max(1, len(encoder.encode(t, disallowed_special=()))) for i, t in documents.items()}


def freeze(args):
    primary, out = args.primary, args.output
    old = read(primary/PRIOR/'hotpot-frozen.json')
    selection = read(primary/'evaluation/hotpotqa-1000-v1/selection.json')
    excluded = set(selection['pilot_case_ids']) | {c['original_case_id'] for c in old['cases']}
    source = primary/'evaluation/hotpotqa-1000-v1/cases.jsonl'
    manifest = read(source.parent/'manifest.json')
    require(sha(source) == manifest['artifacts']['cases.jsonl'], 'Hotpot source manifest changed')
    all_cases = jsonl(source)
    selected = []
    for kind, count in [('bridge', 240), ('comparison', 60)]:
        available = [c for c in all_cases if c['type'] == kind and c['case_id'] not in excluded]
        selected.extend(sorted(available, key=lambda c: digest(['classic-strategy-v1', c['case_id']]))[:count])
    require(len(selected) == len({c['case_id'] for c in selected}) == 300, 'Fresh split incomplete')
    require(not excluded & {c['case_id'] for c in selected}, 'Fresh split overlaps development/pilot')
    docs = {d['document_id']: d for d in old['documents']}
    fresh = []
    for c in selected:
        require(c['answerable'] and set(c['required_document_ids']) <= docs.keys(), 'Missing gold documents')
        for fact in c['supporting_facts']:
            text = docs[fact['document_id']]['text']
            require(text[fact['char_start']:fact['char_end']] == fact['quote'], 'Evidence quote mismatch')
        fresh.append({'case_id': 'hotpot-fresh:'+c['case_id'], 'cluster': c['case_id'],
            'original_case_id': c['case_id'], 'query': c['question'], 'family': 'hotpot-fresh',
            'group': 'hotpot-fresh:'+c['type'], 'answerable': True,
            'required_paths': [c['required_document_ids']], 'answer': c['answer'],
            'supporting_facts': c['supporting_facts']})
    input_paths = [primary/PRIOR/f'{f}-{suffix}' for f in ('hotpot', 'musique')
                   for suffix in ('frozen.json', 'vectors.json', 'vectors.npz', 'results.json')]
    input_paths += [source, primary/PRIOR/'embedding-identity.json',
        primary/'.runtime/evaluations/auto-qa-diagnosis-20260905/sources.json',
        primary/'.runtime/evaluations/source-recall-20260905/exact-ranks.json',
        primary/'.runtime/evaluations/source-recall-repair-20260905/comparison.json',
        primary/'evaluation/document-qa-v1/source-recall-probes.json',
        primary/'evaluation/document-qa-v1/cases.jsonl',
        primary/'evaluation/document-qa-v1/auto-qa-paraphrases.jsonl']
    protocol = {'schema': 'classic_strategy_v1', 'policies': policy_grid(),
        'policy_sha256': sha(ROOT/'tools/classic_strategy.py'),
        'ranking_sha256': sha(ROOT/'src/rag_kb/retrieval/reranker.py'),
        'input_sha256': {str(p.relative_to(primary)): sha(p) for p in input_paths},
        'development': '85 historical financial/contract + 100 prior Hotpot + 76 prior MuSiQue',
        'validation': '300 additional Hotpot questions; same complete 9793-document shared corpus',
        'excluded_ids': sorted(excluded), 'fresh_ids': [c['case_id'] for c in fresh],
        'selection': 'family nondecrease in complete evidence/probes, <=10% macro token growth; maximize macro coverage, Top-1, then minimize cost/depth',
        'validation_gate': 'positive paired 95% complete-evidence difference, no aggregate Top-1 decrease, <=10% mean token growth; report individual losses',
        'answer_stage': 'only if retrieval selection passes validation; same existing answer pipeline and specified models',
        'scope': 'exact text Classic; expanded table source pool stays frozen; no claims for hybrid/multimodal',
        'qa_generation_calls': 0}
    immutable(out/'protocol.json', protocol)
    immutable(out/'fresh-cases.json', fresh)
    print({'frozen': True, 'development_cases': 261, 'fresh_cases': len(fresh), 'policies': len(POLICIES)}, flush=True)


def validate(args):
    protocol = read(args.output/'protocol.json')
    require(protocol['policies'] == policy_grid() and protocol['policy_sha256'] == sha(ROOT/'tools/classic_strategy.py'), 'Frozen policy changed')
    require(protocol['ranking_sha256'] == sha(ROOT/'src/rag_kb/retrieval/reranker.py'), 'Baseline runtime changed')
    for name, expected in protocol['input_sha256'].items():
        require(sha(args.primary/name) == expected, f'Frozen input changed: {name}')
    require(protocol['fresh_ids'] == [c['case_id'] for c in read(args.output/'fresh-cases.json')], 'Fresh identities changed')
    return protocol


async def embed(args):
    validate(args)
    from tools.evaluate_chunking_ab import load_models, preflight
    models, identities = await load_models(args.primary)
    checks = await preflight(models)
    write(args.output/'provider-preflight.json', {'identities': identities, 'checks': checks})
    require(all(c['ok'] for c in checks.values()), 'Required model unavailable; stop for user decision')
    provider = models['text_embedding']
    identity = read(args.primary/PRIOR/'embedding-identity.json')
    require(identity['model'] == provider.embedding_space.requested_model == 'qwen3.7-text-embedding'
        and identity['dimension'] == provider.embedding_space.dimension == 1024
        and identity['fingerprint'] == provider.embedding_space.compatibility_fingerprint,
        'Embedding identity differs; no fallback allowed')
    fresh = read(args.output/'fresh-cases.json')
    path = args.output/'fresh-query-cache.json'
    cache = read(path) if path.exists() else {}
    semaphore = asyncio.Semaphore(4)
    calls = 0
    started = time.perf_counter()
    async def one(c):
        nonlocal calls
        key = digest([identity, 'query', c['query']])
        if key in cache:
            return
        async with semaphore:
            vector = list(await provider.embed_query(c['query']))
            require(len(vector) == 1024 and all(np.isfinite(vector)) and np.linalg.norm(vector) > 0, 'Invalid query vector')
            cache[key] = vector
            calls += 1
            write(path, cache)
            if calls % 25 == 0:
                print({'embedded': calls, 'total': len(fresh)}, flush=True)
    async with asyncio.TaskGroup() as group:
        for c in fresh:
            group.create_task(one(c))
    vectors = np.array([cache[digest([identity, 'query', c['query']])] for c in fresh], dtype=np.float64)
    require(vectors.shape == (300, 1024) and np.isfinite(vectors).all(), 'Incomplete query vectors')
    dest = args.output/'fresh-queries.npz'
    if dest.exists():
        require(np.array_equal(np.load(dest, allow_pickle=False)['queries'], vectors), 'Query vectors changed')
    else:
        np.savez_compressed(dest, queries=vectors)
        dest.chmod(0o600)
    immutable(args.output/'fresh-vectors.json', {'identity': identity, 'cases_sha256': sha(args.output/'fresh-cases.json'),
        'vectors_sha256': sha(dest), 'documents_npz_sha256': sha(args.primary/PRIOR/'hotpot-vectors.npz')})
    write(args.output/'embedding-run.json', {'new_query_calls': calls, 'new_document_calls': 0,
        'qa_generation_calls': 0, 'seconds': time.perf_counter()-started})
    print({'embedding_complete': True, 'new_queries': calls}, flush=True)


def external_records(args, family, fresh=False):
    base = args.primary/PRIOR
    frozen = read(base/f'{family}-frozen.json')
    binding = read(base/f'{family}-vectors.json')
    require(binding['input_sha256'] == sha(base/f'{family}-frozen.json')
            and binding['vectors_sha256'] == sha(base/f'{family}-vectors.npz'), 'Vector binding changed')
    data = np.load(base/f'{family}-vectors.npz', allow_pickle=False)
    documents, queries = data['documents'], data['queries']
    cases = frozen['cases']
    if fresh:
        b = read(args.output/'fresh-vectors.json')
        require(b['cases_sha256'] == sha(args.output/'fresh-cases.json')
            and b['vectors_sha256'] == sha(args.output/'fresh-queries.npz')
            and b['documents_npz_sha256'] == sha(base/'hotpot-vectors.npz')
            and b['identity'] == read(base/'embedding-identity.json'), 'Fresh vectors changed')
        queries = np.load(args.output/'fresh-queries.npz', allow_pickle=False)['queries']
        cases = read(args.output/'fresh-cases.json')
    require(documents.shape == (len(frozen['documents']), 1024) and queries.shape == (len(cases), 1024)
        and np.isfinite(documents).all() and np.isfinite(queries).all(), 'Invalid vector shape/value')
    require((np.linalg.norm(documents, axis=1) > 0).all() and (np.linalg.norm(queries, axis=1) > 0).all(), 'Zero vector')
    similarities = (queries/np.linalg.norm(queries, axis=1, keepdims=True)) @ (documents/np.linalg.norm(documents, axis=1, keepdims=True)).T
    docs = frozen['documents']
    price = costs({d['id']: d['text'] for d in docs})
    expected = {} if fresh else {r['case_id']: r['ids'] for r in read(base/f'{family}-results.json')['arms']['classic10']['cases']}
    for c, row in zip(cases, similarities, strict=True):
        ordering = sorted(range(len(docs)), key=lambda n: (-row[n], UUID(docs[n]['id']).int))
        # Retain full shared-space distances for candidate-ceiling diagnostics.
        hits = [SimpleNamespace(index_chunk_id=UUID(docs[n]['id']), text=docs[n]['text'], cosine_distance=float(1-row[n]),
                    modality='text', hierarchy={'titles': [{'text': docs[n]['title']}]}, source_location={},
                    source_metadata={'document_id': docs[n]['id'], 'document_version_id': docs[n]['id']}) for n in ordering]
        core = [str(h.index_chunk_id) for h in hits[:40] if 1-h.cosine_distance >= .35]
        case = {**c, 'labels': {d['id']: [d['document_id']] for d in docs}}
        yield case, hits, core, [], price, expected.get(c['case_id'])


def financial_records(args):
    from tools.evaluate_auto_qa_retrieval import _evaluation_cases, evidence_matches
    primary = args.primary
    chunks = read(primary/'.runtime/evaluations/auto-qa-diagnosis-20260905/sources.json')['chunks']
    full = read(primary/'.runtime/evaluations/source-recall-20260905/exact-ranks.json')
    require(full['embedding_model'] == 'qwen3.7-text-embedding' and full['source_sha256'] == sha(
        primary/'.runtime/evaluations/auto-qa-diagnosis-20260905/sources.json'), 'Financial source/vector identity changed')
    ranks = {r['case_id']: r['ranks'] for r in full['cases']}
    comparison = read(primary/'.runtime/evaluations/source-recall-repair-20260905/comparison.json')['arms']
    cores = {r['case_id']: r for r in comparison['dense40']['cases']}
    pools = {r['case_id']: r for r in comparison['protected_table_context']['cases']}
    probes = {c['case_id']: [p['chunk_id'] for p in c['required']] for c in read(primary/'evaluation/document-qa-v1/source-recall-probes.json')['cases']}
    price = costs({i: c['content'] for i, c in chunks.items()})
    for c in _evaluation_cases(primary/'evaluation/document-qa-v1'):
        if c['group'] not in {'direct', 'paraphrase'}:
            continue
        cid = c['evaluation_case_id']
        hits = [SimpleNamespace(index_chunk_id=UUID(i), text=chunks[i]['content'], cosine_distance=d,
            modality=chunks[i]['modality'], hierarchy=chunks[i]['hierarchy'], source_location=chunks[i]['source_location'],
            source_metadata=chunks[i]['source_metadata']) for i, d in ranks[cid]]
        case = {'case_id': cid, 'query': c['question'], 'family': c['source_dataset'],
            'group': c['source_dataset']+':'+c['group'], 'cluster': c.get('base_case_id', cid),
            'answerable': True, 'required_paths': [['historical_label']],
            'labels': {str(h.index_chunk_id): ['historical_label'] if evidence_matches(c, h) else [] for h in hits},
            'probe_ids': probes.get(cid, [])}
        core = cores[cid]['candidate_ids']
        extras = [i for i in pools[cid]['candidate_ids'] if i not in core]
        yield case, hits, core, extras, price, pools[cid]['final_ids']


def summary(rows):
    return summarize([{**r, 'group': r['family']} for r in rows])


def run(args, fresh=False):
    validate(args)
    policies = POLICIES
    if fresh:
        selection = read(args.output/'selection.json')
        require(selection['development_sha256'] == sha(args.output/'development.json')
            and selection['protocol_sha256'] == sha(args.output/'protocol.json'), 'Frozen selection changed')
        names = {'classic10', selection['selected'], 'classic_append5'}
        policies = tuple(p for p in POLICIES if p.name in names)
        datasets = [('fresh', external_records(args, 'hotpot', True))]
    else:
        datasets = [('finance', financial_records(args)), ('hotpot', external_records(args, 'hotpot')), ('musique', external_records(args, 'musique'))]
    arms, timings = defaultdict(list), defaultdict(float)
    diagnostics, verified = [], 0
    started = time.perf_counter()
    with patch.object(runtime_reranker, '_terms', lru_cache(maxsize=20000)(runtime_reranker._terms)):
        for family, records in datasets:
            for n, (case, hits, core, extras, price, expected) in enumerate(records, 1):
                by_id = {str(h.index_chunk_id): h for h in hits}
                runtime = ids(s.hit for s in rank_with_source_context(case['query'], tuple(by_id[i] for i in core),
                    tuple(by_id[i] for i in extras), top_k=10))
                require(expected is None or expected == runtime, 'Historical Classic baseline drift')
                for p in policies:
                    clock = time.perf_counter()
                    selected, candidates = rank(case['query'], hits, core, extras, price, p)
                    timings[p.name] += time.perf_counter()-clock
                    if p.name == 'classic10':
                        require(selected == runtime, 'Evaluator does not reproduce runtime Classic')
                        verified += 1
                    row = evidence_row(case, selected, candidates, price)
                    row['candidate_count'] = len(candidates)
                    arms[p.name].append(row)
                required = set().union(*(set(path) for path in case['required_paths']))
                gold_ranks = [{'id': str(h.index_chunk_id), 'rank': r, 'similarity': 1-h.cosine_distance}
                    for r, h in enumerate(hits, 1) if set(case['labels'].get(str(h.index_chunk_id), [])) & required]
                diagnostics.append({'case_id': case['case_id'], 'required_full_ranks': gold_ranks,
                    'core_count': len(core), 'supplement_count': len(extras)})
                if n % 25 == 0:
                    print({'family': family, 'cases': n, 'elapsed_seconds': round(time.perf_counter()-started, 1)}, flush=True)
            print({'finished': family, 'cases': n}, flush=True)
    result = {'status': 'complete', 'protocol_sha256': sha(args.output/'protocol.json'),
        'scope': 'exact text retrieval replay; full source corpus, immutable table supplements, no model generation',
        'qa_generation_calls': 0, 'model_calls': 0, 'runtime_verified_baselines': verified,
        'arms': dict(arms), 'summaries': {k: summary(rows) for k, rows in arms.items()},
        'ranking_seconds': dict(timings), 'elapsed_seconds': time.perf_counter()-started, 'diagnostics': diagnostics,
        'evaluator_sha256': sha(Path(__file__))}
    name = 'validation' if fresh else 'development'
    target = args.output/f'{name}.json'
    if args.replay:
        prior = read(target)
        require(prior['arms'] == result['arms'] and prior['diagnostics'] == diagnostics, 'Replay result changed')
        write(args.output/f'{name}-verification.json', {'identical_rankings': sum(map(len, arms.values())),
            'runtime_verified_baselines': verified, 'new_model_calls': 0, 'result_sha256': sha(target)})
    else:
        require(not target.exists(), 'Refusing to overwrite a completed evaluation; use --replay')
        write(target, result)
    if not fresh:
        choice = choose(arms)
        choice.update(development_sha256=sha(target), protocol_sha256=sha(args.output/'protocol.json'))
        immutable(args.output/'selection.json', choice)
        print({'selected': choice['selected']}, flush=True)
    else:
        selected = selection['selected']
        old, new = arms['classic10'], arms[selected]
        interval = paired_interval(old, new, 'complete_evidence')
        analyses = {p.name: {'summary': summary(arms[p.name]), 'paired': paired_changes(old, arms[p.name]),
            'complete_interval': paired_interval(old, arms[p.name], 'complete_evidence'),
            'fraction_interval': paired_interval(old, arms[p.name], 'required_fraction')} for p in policies}
        gate = {'complete_ci_positive': interval['ci95'][0] > 0,
                'top1_nondecrease': sum(r['hit1'] for r in new) >= sum(r['hit1'] for r in old),
                'token_ratio_at_most_1_10': sum(r['tokens'] for r in new) <= 1.10*sum(r['tokens'] for r in old)}
        write(args.output/'validation-analysis.json', {'selected': selected, 'gate': gate,
            'passed': all(gate.values()), 'arms': analyses, 'validation_sha256': sha(target)})
        print({'validation_gate': gate}, flush=True)
    for policy_name, values in result['summaries'].items():
        print({'policy': policy_name, 'complete': {f: v['complete_evidence'] for f, v in values.items()}}, flush=True)


def analyze(args):
    validate(args)
    selected = read(args.output/'selection.json')['selected']
    result = {'selected': selected, 'stages': {}, 'qa_generation_calls': 0, 'new_model_calls': 0}
    for stage in ('development', 'validation'):
        data = read(args.output/f'{stage}.json')
        groups = {'all': lambda r: True}
        groups.update({f: lambda r, f=f: r['family'] == f for f in sorted({r['family'] for r in data['arms']['classic10']})})
        groups['financial_all'] = lambda r: r['family'] not in {'hotpot', 'musique', 'hotpot-fresh'}
        if stage == 'development':
            groups['musique_multihop'] = lambda r: r['family'] == 'musique' and 'graph' in r['group']
        analyses = {}
        for group, predicate in groups.items():
            base = [r for r in data['arms']['classic10'] if predicate(r)]
            if not base:
                continue
            values = {}
            for policy_name in ('classic10', selected, 'classic_append5'):
                rows = [r for r in data['arms'][policy_name] if predicate(r)]
                values[policy_name] = {'summary': summarize([{**r, 'group': 'all'} for r in rows])['all'],
                    'changes': paired_changes(base, rows),
                    'intervals': {m: paired_interval(base, rows, m) for m in ('hit1', 'complete_evidence', 'required_fraction')},
                    'token_ratio': sum(r['tokens'] for r in rows)/sum(r['tokens'] for r in base)}
            analyses[group] = values
        result['stages'][stage] = analyses
    fresh = read(args.output/'fresh-cases.json')
    development = read(args.primary/PRIOR/'hotpot-frozen.json')
    validation = read(args.output/'validation.json')
    old = {r['case_id']: r for r in validation['arms']['classic10']}
    new = {r['case_id']: r for r in validation['arms'][selected]}
    docs = {d['document_id']: d for d in development['documents']}
    examples = []
    for c in fresh:
        before, after = old[c['case_id']], new[c['case_id']]
        if before['complete_evidence'] == after['complete_evidence']:
            continue
        examples.append({'case_id': c['case_id'], 'query': c['query'],
            'change': 'gain' if after['complete_evidence'] else 'loss',
            'required_sources': [{'title': docs[i]['title'], 'id': docs[i]['id'],
                'classic_rank': before['ids'].index(docs[i]['id'])+1 if docs[i]['id'] in before['ids'] else None,
                'selected_rank': after['ids'].index(docs[i]['id'])+1 if docs[i]['id'] in after['ids'] else None}
                for i in c['required_paths'][0]]})
    result['fresh_complete_changes'] = examples
    old_gold = {i for c in development['cases'] for path in c['required_paths'] for i in path}
    new_gold = {i for c in fresh for path in c['required_paths'] for i in path}
    result['split_overlap'] = {'question_ids': len({c['original_case_id'] for c in fresh} & {c['original_case_id'] for c in development['cases']}),
        'exact_query_texts': len({c['query'] for c in fresh} & {c['query'] for c in development['cases']}),
        'shared_gold_documents': len(old_gold & new_gold), 'fresh_gold_documents': len(new_gold),
        'retrieval_corpus_shared': True}
    result['artifact_sha256'] = {name: sha(args.output/name) for name in ('protocol.json', 'selection.json',
        'development.json', 'validation.json', 'service-verification.json', 'development-verification.json', 'validation-verification.json')}
    write(args.output/'analysis.json', result)
    print({'selected': selected, 'fresh_changes': examples, 'split_overlap': result['split_overlap']}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'develop', 'embed', 'validate', 'analyze'))
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    if args.stage == 'freeze':
        freeze(args)
    elif args.stage == 'embed':
        asyncio.run(embed(args))
    elif args.stage == 'analyze':
        analyze(args)
    else:
        run(args, fresh=args.stage == 'validate')


if __name__ == '__main__':
    main()
