"""LangChain-backed implementation of the application embedding-model contract."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
import openai

from rag_kb.domain import (
    EmbeddingBatch,
    EmbeddingSpaceDefinition,
    ErrorCode,
    IndexingExecutionError,
    IndexingPhase,
)


_RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
_MAX_RETRY_AFTER_SECONDS = 60
_TIMEOUT_SCHEDULING_MARGIN_SECONDS = 1


class LangChainEmbeddingModelAdapter:
    """Use OpenAIEmbeddings asynchronously while preserving domain invariants."""

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
        embedding_model: Embeddings | None = None,
    ) -> None:
        if not api_key or not embedding_space.requested_model:
            raise ValueError("embedding API key and model are required")
        if max_batch_size <= 0 or timeout_seconds <= 0 or max_concurrency <= 0:
            raise ValueError("embedding provider limits must be positive")
        if max_retries < 0:
            raise ValueError("embedding provider retries must be non-negative")
        self._embedding_space = embedding_space
        self._max_batch_size = max_batch_size
        self._total_timeout_seconds = (
            timeout_seconds * (max_retries + 1)
            + _MAX_RETRY_AFTER_SECONDS * max_retries
            + _TIMEOUT_SCHEDULING_MARGIN_SECONDS
        )
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._model = embedding_model or OpenAIEmbeddings(
            model=embedding_space.requested_model,
            dimensions=embedding_space.dimension,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
            chunk_size=max_batch_size,
            check_embedding_ctx_length=False,
            model_kwargs={"encoding_format": "float"},
        )

    @property
    def embedding_space(self) -> EmbeddingSpaceDefinition:
        return self._embedding_space

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    async def embed_documents(self, texts: tuple[str, ...]) -> EmbeddingBatch:
        if not texts or len(texts) > self._max_batch_size:
            raise ValueError("embedding batch size is outside the configured bound")
        vectors = await self._invoke(
            lambda: self._model.aembed_documents(list(texts))
        )
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise _invalid_response(
                "batch_count",
                expected=len(texts),
                observed=len(vectors) if isinstance(vectors, list) else None,
            )
        return EmbeddingBatch(
            vectors=tuple(
                _numeric_vector(
                    vector,
                    check="document_vector",
                    dimension=self._embedding_space.dimension,
                    normalization=self._embedding_space.normalization,
                )
                for vector in vectors
            )
        )

    async def embed_query(self, text: str) -> tuple[float, ...]:
        if not text:
            raise ValueError("embedding query must not be empty")
        vector = await self._invoke(lambda: self._model.aembed_query(text))
        return _numeric_vector(
            vector,
            check="query_vector",
            dimension=self._embedding_space.dimension,
            normalization=self._embedding_space.normalization,
        )

    async def _invoke(
        self,
        operation: Callable[[], Awaitable[object]],
    ) -> object:
        async with self._semaphore:
            try:
                async with asyncio.timeout(self._total_timeout_seconds):
                    return await operation()
            except TimeoutError as error:
                raise _provider_unavailable({"check": "total_timeout"}) from error
            except openai.APIStatusError as error:
                status = error.status_code
                raise _provider_unavailable(
                    {
                        "http_status": status,
                        "retryable": (
                            status in _RETRYABLE_STATUSES or status >= 500
                        ),
                    }
                ) from error
            except (openai.APITimeoutError, openai.APIConnectionError) as error:
                raise _provider_unavailable(
                    {"check": "transport", "retryable": True}
                ) from error
            except openai.OpenAIError as error:
                raise _provider_unavailable({"check": "provider_sdk"}) from error
            except IndexingExecutionError:
                raise
            except (KeyError, TypeError, ValueError, IndexError) as error:
                raise _invalid_response("provider_result") from error


def _numeric_vector(
    value: object,
    *,
    check: str,
    dimension: int,
    normalization: str,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        not isinstance(item, bool) and isinstance(item, (int, float))
        for item in value
    ):
        raise _invalid_response(check)
    vector = tuple(float(item) for item in value)
    if len(vector) != dimension:
        raise _invalid_response(
            f"{check}_dimension",
            expected=dimension,
            observed=len(vector),
        )
    if not all(math.isfinite(item) for item in vector):
        raise _invalid_response(f"{check}_finite")
    if normalization == "l2":
        norm = math.sqrt(sum(item * item for item in vector))
        if abs(norm - 1.0) > 0.001:
            raise _invalid_response(f"{check}_normalization")
    return vector


def _provider_unavailable(diagnostic: dict[str, object]) -> IndexingExecutionError:
    return IndexingExecutionError(
        ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE,
        phase=IndexingPhase.EMBEDDING,
        diagnostic=diagnostic,
    )


def _invalid_response(
    check: str,
    *,
    expected: int | None = None,
    observed: int | None = None,
) -> IndexingExecutionError:
    diagnostic: dict[str, object] = {"check": check}
    if expected is not None:
        diagnostic["expected"] = expected
    if observed is not None:
        diagnostic["observed"] = observed
    return IndexingExecutionError(
        ErrorCode.EMBEDDING_RESPONSE_INVALID,
        phase=IndexingPhase.EMBEDDING,
        diagnostic=diagnostic,
    )
