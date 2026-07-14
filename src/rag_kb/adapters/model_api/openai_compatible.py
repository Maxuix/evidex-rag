"""Bounded OpenAI-compatible embeddings adapter without SDK coupling."""

from __future__ import annotations

import asyncio
import json
import random
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rag_kb.domain import (
    EmbeddingBatch,
    EmbeddingSpaceDefinition,
    ErrorCode,
    IndexingExecutionError,
    IndexingPhase,
)


_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        embedding_space: EmbeddingSpaceDefinition,
        max_batch_size: int,
        timeout_seconds: float,
        max_retries: int,
        max_concurrency: int,
        retryable_statuses: frozenset[int] = frozenset({429, 500, 503}),
    ) -> None:
        if not api_key:
            raise ValueError("embedding API key is required")
        if max_batch_size <= 0 or max_concurrency <= 0 or timeout_seconds <= 0:
            raise ValueError("embedding provider limits must be positive")
        if max_retries < 0:
            raise ValueError("embedding provider retries must be non-negative")
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._api_key = api_key
        self._embedding_space = embedding_space
        self._max_batch_size = max_batch_size
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retryable_statuses = retryable_statuses
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @property
    def embedding_space(self) -> EmbeddingSpaceDefinition:
        return self._embedding_space

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    async def embed(self, texts: tuple[str, ...]) -> EmbeddingBatch:
        if not texts or len(texts) > self._max_batch_size:
            raise ValueError("embedding batch size is outside the configured bound")
        async with self._semaphore:
            return await self._embed_with_retries(texts)

    async def _embed_with_retries(self, texts: tuple[str, ...]) -> EmbeddingBatch:
        for attempt in range(self._max_retries + 1):
            try:
                return await asyncio.to_thread(self._request, texts)
            except _RetryableProviderError:
                if attempt >= self._max_retries:
                    break
                delay = 0.2 * (2**attempt) * random.uniform(0.8, 1.2)
                await asyncio.sleep(delay)
        raise IndexingExecutionError(
            ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE,
            phase=IndexingPhase.EMBEDDING,
            diagnostic={"retry_exhausted": True},
        )

    def _request(self, texts: tuple[str, ...]) -> EmbeddingBatch:
        payload = json.dumps(
            {
                "model": self._embedding_space.requested_model,
                "input": texts,
                "dimensions": self._embedding_space.dimension,
                "encoding_format": "float",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            self._url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                body = response.read(_MAX_RESPONSE_BYTES + 1)
        except HTTPError as error:
            if error.code in self._retryable_statuses:
                raise _RetryableProviderError from error
            raise IndexingExecutionError(
                ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE,
                phase=IndexingPhase.EMBEDDING,
                diagnostic={"http_status": error.code, "retryable": False},
            ) from error
        except (URLError, TimeoutError, HTTPException, OSError) as error:
            raise _RetryableProviderError from error
        if len(body) > _MAX_RESPONSE_BYTES:
            raise IndexingExecutionError(
                ErrorCode.EMBEDDING_RESPONSE_INVALID,
                phase=IndexingPhase.EMBEDDING,
                diagnostic={"check": "response_size"},
            )
        return self._decode(body, expected_count=len(texts))

    def _decode(self, body: bytes, *, expected_count: int) -> EmbeddingBatch:
        try:
            payload: Any = json.loads(body)
            model = payload["model"]
            data = payload["data"]
            if not isinstance(model, str) or not isinstance(data, list):
                raise TypeError
            ordered = sorted(data, key=lambda item: item["index"])
            if [item["index"] for item in ordered] != list(range(expected_count)):
                raise ValueError
            vectors = tuple(_numeric_vector(item["embedding"]) for item in ordered)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise IndexingExecutionError(
                ErrorCode.EMBEDDING_RESPONSE_INVALID,
                phase=IndexingPhase.EMBEDDING,
                diagnostic={"check": "wire_shape"},
            ) from error
        return EmbeddingBatch(model=model, vectors=vectors)


class _RetryableProviderError(RuntimeError):
    pass


def _numeric_vector(value: Any) -> tuple[float, ...]:
    if not isinstance(value, list) or not all(
        not isinstance(item, bool) and isinstance(item, (int, float))
        for item in value
    ):
        raise TypeError
    return tuple(float(item) for item in value)
