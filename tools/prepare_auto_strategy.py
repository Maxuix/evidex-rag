#!/usr/bin/env python3
"""Freeze existing external QA and encode sources using the required host profiles.

No QA generation, application indexing, database writes, workers, or Docker.
Gold stays in evaluator records and is never included in embedding inputs.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / '.runtime/evaluations/auto-strategy-20260905'
MUSIQUE = ROOT / 'archive/evaluations/01-0901-completed-campaigns/corpora/routing-rag-musique-expanded-v1'
HOTPOT = ROOT / 'evaluation/hotpotqa-1000-v1'


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n')
    tmp.chmod(0o600)
    tmp.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def select_hotpot(cases, pilot_ids):
    selected = []
    for kind, count in (('bridge', 80), ('comparison', 20)):
        eligible = [c for c in cases if c['type'] == kind and c['case_id'] not in pilot_ids]
        selected.extend(sorted(eligible, key=lambda c: digest(['auto-strategy-v1', c['case_id']]))[:count])
    require(len(selected) == len({c['case_id'] for c in selected}) == 100, 'Hotpot selection incomplete')
    require(not set(pilot_ids) & {c['case_id'] for c in selected}, 'Pilot overlaps held-out selection')
    return selected


def freeze():
    OUTPUT.mkdir(mode=0o700, parents=True, exist_ok=True)
    from tools.build_hotpot_benchmark import validate
    validate(HOTPOT)
    for family, root in (('hotpot', HOTPOT), ('musique', MUSIQUE)):
        manifest = read(root/'manifest.json')
        for name in ('cases.jsonl', 'documents.jsonl'):
            require(sha(root/name) == manifest['artifacts'][name], 'QA/catalog manifest hash differs')
        records = jsonl(root/'documents.jsonl')
        require(len(records) == (9793 if family == 'hotpot' else 578), 'Unexpected corpus size')
        documents = []
        for record in records:
            path = root/record['path'] if family == 'hotpot' else root/'documents'/record['filename']
            expected = record['sha256'] if family == 'hotpot' else record['artifact_sha256']
            require(sha(path) == expected, 'Source artifact hash differs')
            body = path.read_text()
            require(bool(body.strip()), 'Empty source document')
            documents.append({'id': str(uuid5(NAMESPACE_URL, family+':'+record['document_id'])),
                'document_id': record['document_id'], 'filename': path.name, 'text': body,
                'title': record['title'], 'source_sha256': expected, 'source_path': str(path.relative_to(ROOT))})
        by_doc = {d['document_id']: d for d in documents}
        source_cases = jsonl(root/'cases.jsonl')
        if family == 'hotpot':
            selection = read(root/'selection.json')
            selected = select_hotpot(source_cases, set(selection['pilot_case_ids']))
        else:
            require(len(source_cases) == 76, 'MuSiQue cases changed')
            selected = source_cases
        cases = []
        for c in selected:
            paths = [c['required_document_ids']] if family == 'hotpot' else c['required_paths']
            require(all(set(p) <= set(by_doc) for p in paths), 'Gold source missing from complete corpus')
            if family == 'hotpot':
                for fact in c['supporting_facts']:
                    body = by_doc[fact['document_id']]['text']
                    require(body[fact['char_start']:fact['char_end']] == fact['quote'], 'Source evidence span differs')
            cases.append({'case_id': family+':'+c['case_id'], 'original_case_id': c['case_id'],
                'family': family, 'group': family+':'+(c['type'] if family == 'hotpot' else c['route_label']),
                'cluster': family+':'+(c['case_id'] if family == 'hotpot' else c['upstream_id']),
                'query': c['question'], 'answerable': c['answerable'], 'required_paths': paths,
                'answer': c['answer'] if family == 'hotpot' else c['expected_answer'],
                'supporting_facts': c.get('supporting_facts', []), 'hop_count': c.get('hop_count', 2)})
        frozen = {'schema': 'auto_strategy_external_v1', 'family': family, 'documents': documents,
            'cases': cases, 'qa_generation_calls': 0,
            'protocol': 'full shared source corpus; one original paragraph document per evaluation item; exact vector core40; no gold insertion',
            'input_sha256': {str((root/name).relative_to(ROOT)): sha(root/name)
                             for name in ('manifest.json', 'cases.jsonl', 'documents.jsonl')},
            'prior_use': ('100 questions exclude all 20 cost-pilot cases; full 9793-document corpus retained'
                         if family == 'hotpot' else 'archived MuSiQue routing benchmark; previously used for Agent/Graph evaluation, not current reranker selection')}
        dest = OUTPUT/(family+'-frozen.json')
        if dest.exists():
            require(read(dest) == frozen, 'Frozen external inputs changed; refusing overwrite')
        else:
            write(dest, frozen)
        print(json.dumps({'frozen': family, 'questions': len(cases), 'answerable': sum(c['answerable'] for c in cases),
                          'documents': len(documents), 'input_sha256': sha(dest)}), flush=True)


class VectorCache:
    def __init__(self, identity):
        self.identity = identity
        self.path = OUTPUT/'vectors.sqlite'
        self.db = sqlite3.connect(self.path)
        self.db.execute('CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, value TEXT NOT NULL, origin TEXT NOT NULL)')
        self.path.chmod(0o600)

    def key(self, text, kind):
        return digest([self.identity, kind, text])

    def get(self, text, kind):
        row = self.db.execute('SELECT value FROM vectors WHERE key=?', (self.key(text, kind),)).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        self.validate(value)
        return value

    def validate(self, vector):
        require(len(vector) == self.identity['dimension'] and all(
            isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x) for x in vector)
            and sum(x*x for x in vector) > 0, 'Invalid embedding vector')

    def put(self, text, kind, vector, origin):
        vector = list(vector)
        self.validate(vector)
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO vectors VALUES (?,?,?)',
                (self.key(text, kind), json.dumps(vector), origin))


async def seed_host_vectors(cache):
    """Read exact known evaluation targets only; reuse text with identical space."""
    import asyncpg
    from tools.evaluation_runtime import load_evaluation_runtime
    from rag_kb.config import load_settings
    runtime = load_evaluation_runtime()
    settings = load_settings(env_file=runtime.env_file)
    url = urlsplit(settings.database.runtime_dsn.get_secret_value())
    require(url.hostname in {'127.0.0.1', 'localhost'} and url.port == runtime.ports['postgres'], 'Non-isolated database')
    ids = [UUID('01a0714f-c797-7560-9580-8d35ce783ef7'), UUID('01a03e45-acc1-7462-a4cf-26bc61e39bea')]
    con = await asyncpg.connect(host=url.hostname, port=url.port, user=url.username, password=url.password,
                                database=url.path.lstrip('/'), timeout=10)
    try:
        async with con.transaction(readonly=True):
            rows = await con.fetch('''SELECT c.content,c.embedding_text,v.embedding::text AS vector,
                e.compatibility_fingerprint,v.representation_kind
                FROM vector_record v JOIN index_chunk c ON c.id=v.index_chunk_id
                JOIN indexed_document_version t ON t.id=c.indexed_document_version_id
                JOIN knowledge_base k ON k.id=c.kb_id JOIN embedding_space e ON e.id=v.embedding_space_id
                WHERE k.id=ANY($1::uuid[]) AND k.deleted_at IS NULL AND c.excluded_at IS NULL
                AND t.index_revision_id=k.active_index_revision_id
                AND t.build_status='ready' AND t.serving_status='serving'
                AND v.representation_kind IN ('text','table_text')
                AND e.compatibility_fingerprint=$2''', ids, cache.identity['fingerprint'])
    finally:
        await con.close()
    for row in rows:
        cache.put(row['embedding_text'] or row['content'], 'document', json.loads(row['vector']), 'existing_host_source')
    write(OUTPUT/'source-vector-reuse.json', {'rows': len(rows), 'database_writes': 0,
                                           'fingerprint': cache.identity['fingerprint']})
    print(json.dumps({'existing_compatible_host_source_vectors': len(rows)}), flush=True)


async def embed(cache_only=False):
    from tools.evaluate_chunking_ab import load_models
    started = time.perf_counter()
    if cache_only:
        identity = read(OUTPUT/'embedding-identity.json')
        provider = None
    else:
        models, _ = await load_models(ROOT)
        provider = models['text_embedding']
        identity = {'model': provider.embedding_space.requested_model,
            'fingerprint': provider.embedding_space.compatibility_fingerprint,
            'dimension': provider.embedding_space.dimension,
            'adapter_sha256': sha(ROOT/'src/rag_kb/adapters/model_api/langchain_embeddings.py')}
        require(identity['model'] == 'qwen3.7-text-embedding', 'Required embedding model differs')
        if (OUTPUT/'embedding-identity.json').exists():
            require(read(OUTPUT/'embedding-identity.json') == identity, 'Embedding identity changed')
        else:
            write(OUTPUT/'embedding-identity.json', identity)
    cache = VectorCache(identity)
    if not cache_only:
        await seed_host_vectors(cache)
    counts = {'new_document_vectors': 0, 'new_query_vectors': 0, 'qa_generation_calls': 0}
    semaphore = asyncio.Semaphore(4)
    async def encode_documents(batch):
        async with semaphore:
            result = await provider.embed_documents(tuple(batch))
            require(len(result.vectors) == len(batch), 'Embedding count differs')
            for text, vector in zip(batch, result.vectors, strict=True):
                cache.put(text, 'document', vector, 'required_qwen')
            counts['new_document_vectors'] += len(batch)
            if counts['new_document_vectors'] % 100 == 0:
                print(json.dumps(counts), flush=True)
    async def encode_query(query):
        async with semaphore:
            result = await provider.embed_query(query)
            cache.put(query, 'query', result, 'required_qwen')
            counts['new_query_vectors'] += 1
    for family in ('musique', 'hotpot'):
        frozen = read(OUTPUT/(family+'-frozen.json'))
        texts = list(dict.fromkeys(d['text'] for d in frozen['documents']))
        missing = [text for text in texts if cache.get(text, 'document') is None]
        queries = list(dict.fromkeys(c['query'] for c in frozen['cases']))
        missing_queries = [q for q in queries if cache.get(q, 'query') is None]
        require(not cache_only or not missing and not missing_queries, 'Vector cache missing; no fallback allowed')
        print(json.dumps({'embedding_family': family, 'missing_sources': len(missing), 'missing_queries': len(missing_queries)}), flush=True)
        if not cache_only:
            async with asyncio.TaskGroup() as group:
                for offset in range(0, len(missing), provider.max_batch_size):
                    group.create_task(encode_documents(missing[offset:offset+provider.max_batch_size]))
                for query in missing_queries:
                    group.create_task(encode_query(query))
        document_vectors = np.array([cache.get(d['text'], 'document') for d in frozen['documents']], dtype=np.float64)
        query_vectors = np.array([cache.get(c['query'], 'query') for c in frozen['cases']], dtype=np.float64)
        path = OUTPUT/(family+'-vectors.npz')
        if cache_only:
            with np.load(path, allow_pickle=False) as prior:
                require(np.array_equal(prior['documents'], document_vectors) and np.array_equal(prior['queries'], query_vectors), 'Frozen vectors changed')
        else:
            np.savez_compressed(path, documents=document_vectors, queries=query_vectors)
            path.chmod(0o600)
        write(OUTPUT/(family+'-vectors.json'), {'input_sha256': sha(OUTPUT/(family+'-frozen.json')),
            'vectors_sha256': sha(path), 'identity': identity, 'documents': len(document_vectors), 'queries': len(query_vectors)})
        print(json.dumps({'completed_embedding': family, **counts}), flush=True)
    cache.db.close()
    write(OUTPUT/('embedding-cache-replay.json' if cache_only else 'embedding-run.json'),
          {**counts, 'status': 'complete', 'seconds': time.perf_counter()-started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'embed'))
    parser.add_argument('--cache-only', action='store_true')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze()
    else:
        asyncio.run(embed(args.cache_only))
