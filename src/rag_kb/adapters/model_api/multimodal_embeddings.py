"""Direct bounded adapter for Model Studio independent multimodal embeddings."""

from __future__ import annotations

import asyncio
import base64
import math
from typing import Any

import httpx

from rag_kb.domain import (
    EmbeddingBatch,
    EmbeddingSpaceDefinition,
    ErrorCode,
    ImageEmbeddingInput,
    IndexingExecutionError,
    IndexingPhase,
)


_RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})


class QwenMultimodalEmbeddingAdapter:
    """Embed text and local raster images in one qwen3-vl shared space."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        embedding_space: EmbeddingSpaceDefinition,
        max_batch_size: int,
        timeout_seconds: float,
        max_retries: int,
        max_concurrency: int,
        text_query_template: str = "query: {text}",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not endpoint or not api_key or not embedding_space.requested_model:
            raise ValueError("multimodal endpoint, key, and model are required")
        if embedding_space.dimension != 1024:
            raise ValueError("multimodal physical vector space must be 1024-dimensional")
        if max_batch_size < 1 or timeout_seconds <= 0 or max_concurrency < 1:
            raise ValueError("multimodal provider limits must be positive")
        if max_retries < 0:
            raise ValueError("multimodal retries must be non-negative")
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._embedding_space = embedding_space
        self._max_batch_size = max_batch_size
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._text_query_template = text_query_template
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client = client

    @property
    def embedding_space(self) -> EmbeddingSpaceDefinition:
        return self._embedding_space

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    async def embed_texts(self, texts: tuple[str, ...]) -> EmbeddingBatch:
        if not texts or len(texts) > self._max_batch_size or any(not text for text in texts):
            raise ValueError("multimodal text batch is outside the configured bound")
        contents = [
            {"text": self._text_query_template.format(text=text)} for text in texts
        ]
        return EmbeddingBatch(await self._invoke(contents))

    async def embed_images(
        self, images: tuple[ImageEmbeddingInput, ...]
    ) -> EmbeddingBatch:
        if not images or len(images) > self._max_batch_size:
            raise ValueError("multimodal image batch is outside the configured bound")
        supported = {"image/png", "image/jpeg", "image/webp", "image/bmp", "image/tiff"}
        if any(image.media_type not in supported or not image.content for image in images):
            raise ValueError("multimodal image input is unsupported")
        contents = [
            {
                "image": (
                    f"data:{image.media_type};base64,"
                    f"{base64.b64encode(image.content).decode('ascii')}"
                )
            }
            for image in images
        ]
        return EmbeddingBatch(await self._invoke(contents))

    async def _invoke(self, contents: list[dict[str, str]]) -> tuple[tuple[float, ...], ...]:
        payload = {
            "model": self._embedding_space.requested_model,
            "input": {"contents": contents},
            "parameters": {"dimension": self._embedding_space.dimension, "enable_fusion": False},
        }
        async with self._semaphore:
            for attempt in range(self._max_retries + 1):
                try:
                    async with asyncio.timeout(self._timeout_seconds):
                        if self._client is not None:
                            response = await self._client.post(
                                self._endpoint,
                                headers={"Authorization": f"Bearer {self._api_key}"},
                                json=payload,
                            )
                        else:
                            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                                response = await client.post(
                                    self._endpoint,
                                    headers={"Authorization": f"Bearer {self._api_key}"},
                                    json=payload,
                                )
                    if response.status_code >= 400:
                        if response.status_code in _RETRYABLE_STATUSES and attempt < self._max_retries:
                            continue
                        raise _unavailable(response.status_code)
                    return _vectors(response.json(), len(contents), self._embedding_space.dimension)
                except (httpx.TimeoutException, httpx.TransportError, TimeoutError) as error:
                    if attempt < self._max_retries:
                        continue
                    raise _unavailable(None) from error
                except (ValueError, TypeError, KeyError, IndexError) as error:
                    raise _invalid("provider_result") from error
        raise _unavailable(None)


def _vectors(payload: Any, expected: int, dimension: int) -> tuple[tuple[float, ...], ...]:
    if not isinstance(payload, dict):
        raise ValueError("response is not an object")
    output = payload.get("output")
    values = output.get("embeddings") if isinstance(output, dict) else None
    if not isinstance(values, list) or len(values) != expected:
        raise ValueError("embedding count differs")
    ordered = sorted(values, key=lambda item: item.get("index", 0) if isinstance(item, dict) else -1)
    vectors: list[tuple[float, ...]] = []
    for item in ordered:
        vector = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(vector, list) or len(vector) != dimension:
            raise ValueError("embedding dimension differs")
        normalized = tuple(float(value) for value in vector)
        if not all(math.isfinite(value) for value in normalized):
            raise ValueError("embedding contains non-finite values")
        norm = math.sqrt(sum(value * value for value in normalized))
        if abs(norm - 1.0) > 0.001:
            raise ValueError("embedding is not L2 normalized")
        vectors.append(normalized)
    return tuple(vectors)


def _unavailable(status: int | None) -> IndexingExecutionError:
    diagnostic: dict[str, object] = {"retryable": True}
    if status is not None:
        diagnostic["http_status"] = status
    return IndexingExecutionError(
        ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE,
        phase=IndexingPhase.MULTIMODAL_EMBEDDING,
        diagnostic=diagnostic,
    )


def _invalid(check: str) -> IndexingExecutionError:
    return IndexingExecutionError(
        ErrorCode.EMBEDDING_RESPONSE_INVALID,
        phase=IndexingPhase.MULTIMODAL_EMBEDDING,
        diagnostic={"check": check},
    )
