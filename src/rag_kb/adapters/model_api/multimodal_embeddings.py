"""Direct bounded adapter for Model Studio independent multimodal embeddings."""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import httpx

from rag_kb.domain import (
    EmbeddingBatch,
    EmbeddingSpaceDefinition,
    ErrorCode,
    ImageEmbeddingInput,
    IndexingExecutionError,
    IndexingPhase,
    MAX_EMBEDDING_DIMENSION,
    MIN_EMBEDDING_DIMENSION,
    normalize_embedding_vector,
)


_RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
DOCUMENT_TRANSFORMATION_VERSION = "tongyi_document_text_v1"
QUERY_TRANSFORMATION_VERSION = "tongyi_query_prefix_v1"
IMAGE_PREPROCESSING_VERSION = "tongyi_data_url_res1_v1"


class TongyiVisionEmbeddingAdapter:
    """Embed text and local raster images in one Tongyi Vision shared space."""

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
        if not MIN_EMBEDDING_DIMENSION <= embedding_space.dimension <= MAX_EMBEDDING_DIMENSION:
            raise ValueError("multimodal embedding dimension is unsupported")
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

    async def embed_documents(self, texts: tuple[str, ...]) -> EmbeddingBatch:
        if not texts or len(texts) > self._max_batch_size or any(not text for text in texts):
            raise ValueError("multimodal text batch is outside the configured bound")
        contents = [{"text": text} for text in texts]
        return EmbeddingBatch(await self._invoke(contents))

    async def embed_query(self, text: str) -> tuple[float, ...]:
        if not text:
            raise ValueError("multimodal query must not be empty")
        vectors = await self._invoke(
            [{"text": self._text_query_template.format(text=text)}]
        )
        return vectors[0]

    async def embed_texts(self, texts: tuple[str, ...]) -> EmbeddingBatch:
        if not texts or len(texts) > self._max_batch_size or any(not text for text in texts):
            raise ValueError("multimodal text batch is outside the configured bound")
        vectors = []
        for text in texts:
            vectors.append(await self.embed_query(text))
        return EmbeddingBatch(tuple(vectors))

    async def embed_images(
        self, images: tuple[ImageEmbeddingInput, ...]
    ) -> EmbeddingBatch:
        if not images or len(images) > self._max_batch_size:
            raise ValueError("multimodal image batch is outside the configured bound")
        supported = {"image/png", "image/jpeg", "image/webp", "image/bmp", "image/tiff"}
        if any(
            image.media_type not in supported
            or not image.content
            or len(image.content) > 10_000_000
            for image in images
        ):
            raise _invalid("image_input")
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
        parameters: dict[str, object] = {
            "output_type": "dense",
            "res_level": 1,
        }
        if self._embedding_space.dimension_request_mode == "explicit":
            parameters["dimension"] = self._embedding_space.dimension
        elif self._embedding_space.dimension_request_mode != "omitted":
            raise ValueError("unsupported embedding dimension request mode")
        payload = {
            "model": self._embedding_space.requested_model,
            "input": {"contents": contents},
            "parameters": parameters,
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
                    return _vectors(
                        response.json(), len(contents), self._embedding_space
                    )
                except (httpx.TimeoutException, httpx.TransportError, TimeoutError) as error:
                    if attempt < self._max_retries:
                        continue
                    raise _unavailable(None) from error
                except (ValueError, TypeError, KeyError, IndexError) as error:
                    raise _invalid("provider_result") from error
        raise _unavailable(None)


def _vectors(
    payload: Any,
    expected: int,
    definition: EmbeddingSpaceDefinition,
) -> tuple[tuple[float, ...], ...]:
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
        vectors.append(normalize_embedding_vector(vector, definition))
    return tuple(vectors)


async def probe_tongyi_embedding_dimension(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    requested_dimension: int | None,
    timeout_seconds: float,
    max_retries: int,
    client: httpx.AsyncClient | None = None,
) -> int:
    parameters: dict[str, object] = {
        "output_type": "dense",
        "res_level": 1,
    }
    if requested_dimension is not None:
        parameters["dimension"] = requested_dimension
    payload = {
        "model": model,
        "input": {"contents": [{"text": "model validation"}]},
        "parameters": parameters,
    }
    for attempt in range(max_retries + 1):
        try:
            async with asyncio.timeout(timeout_seconds):
                if client is not None:
                    response = await client.post(
                        endpoint.rstrip("/"),
                        headers={"Authorization": f"Bearer {api_key}"},
                        json=payload,
                    )
                else:
                    async with httpx.AsyncClient(timeout=timeout_seconds) as owned:
                        response = await owned.post(
                            endpoint.rstrip("/"),
                            headers={"Authorization": f"Bearer {api_key}"},
                            json=payload,
                        )
            if response.status_code >= 400:
                if response.status_code in _RETRYABLE_STATUSES and attempt < max_retries:
                    continue
                raise ValueError("embedding provider rejected validation")
            response_payload = response.json()
            output = response_payload.get("output") if isinstance(response_payload, dict) else None
            values = output.get("embeddings") if isinstance(output, dict) else None
            item = values[0] if isinstance(values, list) and len(values) == 1 else None
            vector = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vector, list):
                raise ValueError("embedding provider returned an invalid vector")
            dimension = len(vector)
            if not MIN_EMBEDDING_DIMENSION <= dimension <= MAX_EMBEDDING_DIMENSION:
                raise ValueError("embedding dimension is outside the supported range")
            definition = EmbeddingSpaceDefinition(
                provider_identity="validation",
                endpoint_identity="validation",
                requested_model=model,
                resolved_model=model,
                model_version=model,
                deployment_revision=None,
                dimension=dimension,
                distance_metric="cosine",
                vector_data_type="float32",
                normalization="client_l2_v1",
                configuration_fingerprint="validation",
                tokenizer_fingerprint=None,
                compatibility_fingerprint="validation",
            )
            normalize_embedding_vector(vector, definition)
            return dimension
        except (httpx.TimeoutException, httpx.TransportError, TimeoutError):
            if attempt >= max_retries:
                raise ValueError("embedding provider validation was unavailable")
    raise ValueError("embedding provider validation was unavailable")


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
