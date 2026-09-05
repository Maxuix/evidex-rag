"""Validated private request caches for the host-only algorithm experiment."""
from __future__ import annotations

import asyncio
import json
import time

import numpy as np

from rag_kb.domain import ChatModelMessage, ChatModelRequest
from tools.prepare_auto_strategy import digest, read, require, write
from tools.rag_algorithm_policies import PROMPTS, SYSTEM, validate_output

PLANNING_MAX_OUTPUT_TOKENS = 4096


class ModelIO:
    def __init__(self, output, identities, embedding_identity, models=None):
        self.output, self.identities, self.embedding_identity, self.models = output, identities, embedding_identity, models
        self.semaphore = asyncio.Semaphore(4)
        self.new_calls = self.new_vectors = 0

    async def generate(self, kind, payload, visible=None):
        messages = [ChatModelMessage('system', SYSTEM+'\n'+PROMPTS[kind]),
                    ChatModelMessage('user', json.dumps(payload, ensure_ascii=False, sort_keys=True))]
        identity = {'model': self.identities['chat'], 'kind': kind, 'schema': 'rag_algorithms_v1',
            'messages': [[m.role, m.content] for m in messages], 'max_output_tokens': PLANNING_MAX_OUTPUT_TOKENS}
        key = digest(identity)
        path = self.output/'calls'/f'{key}.json'
        if path.exists():
            record = read(path)
            require(record['identity'] == identity, 'Cached request identity changed')
            return validate_output(kind, record['value'], visible), 'g:'+key
        require(self.models is not None, 'Missing generation cache; real call is not allowed in replay')
        attempts = []
        async with self.semaphore:
            for attempt in range(2):
                start = time.perf_counter()
                # One schema-only repair is part of the frozen protocol. Invalid
                # output content is never persisted or used for subsequent search.
                response = await self.models['chat'].complete(ChatModelRequest(messages=tuple(messages), max_output_tokens=PLANNING_MAX_OUTPUT_TOKENS))
                self.new_calls += 1
                attempts.append({'usage': dict(response.usage), 'seconds': time.perf_counter()-start,
                                 'finish_reason': response.finish_reason, 'model': response.model})
                try:
                    raw = response.content.strip()
                    if raw.startswith('```') and raw.endswith('```'):
                        raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]
                    value = validate_output(kind, json.loads(raw), visible)
                except (ValueError, TypeError) as error:
                    if attempt:
                        write(self.output/'failures'/f'{key}.json', {'identity': identity,
                            'error_type': type(error).__name__, 'attempts': attempts})
                        raise ValueError('Required algorithm output failed validation twice') from error
                    messages.append(ChatModelMessage('system', 'The previous response failed schema or exact-source-anchor validation. Return a fresh valid JSON object obeying every constraint.'))
                    continue
                write(path, {'identity': identity, 'value': value, 'attempts': attempts,
                    'validation': 'schema; query limits; exact visible source/quote anchors for follow-up queries',
                    'trust': 'retrieval_probe_only'})
                return value, 'g:'+key
        raise AssertionError('unreachable')

    async def embed(self, text, *, kind):
        require(kind in {'query', 'document'}, 'Unknown embedding input kind')
        identity = {'space': self.embedding_identity, 'kind': kind, 'text': text}
        key = digest(identity)
        path = self.output/'vectors'/f'{key}.json'
        if path.exists():
            record = read(path)
            require(record['identity'] == identity, 'Cached embedding identity changed')
            vector = record['vector']
        else:
            require(self.models is not None, 'Missing vector cache; real call is not allowed in replay')
            async with self.semaphore:
                start = time.perf_counter()
                provider = self.models['text_embedding']
                vector = (await provider.embed_documents((text,))).vectors[0] if kind == 'document' else await provider.embed_query(text)
                vector = list(vector)
                validate_vector(vector)
                self.new_vectors += 1
                write(path, {'identity': identity, 'vector': vector, 'seconds': time.perf_counter()-start})
        validate_vector(vector)
        return vector, 'e:'+key

    def cost(self, keys):
        values = {'llm_calls': 0, 'embedding_calls': 0, 'llm_total_tokens': 0, 'model_seconds_sum': 0.}
        for ref in dict.fromkeys(keys):
            kind, key = ref.split(':', 1)
            record = read(self.output/('calls' if kind == 'g' else 'vectors')/f'{key}.json')
            if kind == 'g':
                values['llm_calls'] += len(record['attempts'])
                values['llm_total_tokens'] += sum(a['usage'].get('total_tokens', 0) for a in record['attempts'])
                values['model_seconds_sum'] += sum(a['seconds'] for a in record['attempts'])
            else:
                values['embedding_calls'] += 1
                values['model_seconds_sum'] += record['seconds']
        return values


def validate_vector(vector):
    require(len(vector) == 1024 and all(isinstance(x, (float, int)) and not isinstance(x, bool)
        and np.isfinite(x) for x in vector) and np.linalg.norm(vector) > 0, 'Invalid Qwen vector')


async def load_io(args, *, live):
    manifest = args.output/'model-identity.json'
    if live:
        from tools.evaluate_chunking_ab import load_models, preflight
        models, identities = await load_models(args.primary)
        if not manifest.exists():
            checks = await preflight(models)
            write(args.output/'execution-preflight.json', {'checks': checks, 'identities': identities})
            require(all(v['ok'] for v in checks.values()), 'Required model unavailable; stop for user decision')
        space = models['text_embedding'].embedding_space
        expected = read(args.primary/'.runtime/evaluations/auto-strategy-20260905/embedding-identity.json')
        require(space.requested_model == expected['model'] == 'qwen3.7-text-embedding'
                and space.dimension == expected['dimension'] == 1024
                and space.compatibility_fingerprint == expected['fingerprint'], 'Required Qwen space differs')
        value = {'identities': identities, 'embedding_identity': expected}
        if manifest.exists():
            require(read(manifest) == value, 'Frozen provider configuration changed; user decision required')
        else:
            write(manifest, value)
    else:
        value, models = read(manifest), None
    return ModelIO(args.output, value['identities'], value['embedding_identity'], models)
