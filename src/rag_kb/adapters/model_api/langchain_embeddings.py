"""LangChain-backed implementation of the application embedding-model contract."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import httpx
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
import openai

from rag_kb.config.settings import provider_retry_budget_seconds
from rag_kb.domain import (
    EmbeddingBatch,
    EmbeddingSpaceDefinition,
    ErrorCode,
    IndexingExecutionError,
    IndexingPhase,
    MAX_EMBEDDING_DIMENSION,
    MIN_EMBEDDING_DIMENSION,
    normalize_embedding_vector,
)


_RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
_NON_PERSISTENT_CONNECTION_HEADERS = {"Connection": "close"}


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
        self._total_timeout_seconds = provider_retry_budget_seconds(
            timeout_seconds,
            max_retries,
        )
        self._semaphore = asyncio.Semaphore(max_concurrency)
        model_arguments: dict[str, object] = {
            "model": embedding_space.requested_model,
            "api_key": api_key,
            "base_url": base_url,
            "timeout": timeout_seconds,
            "max_retries": max_retries,
            "chunk_size": max_batch_size,
            "check_embedding_ctx_length": False,
            "model_kwargs": {"encoding_format": "float"},
            # Some OpenAI-compatible gateways leave an idle HTTP/1.1 connection
            # open even though it can no longer serve another request.  Semantic
            # chunking performs several embedding calls in sequence, so reusing
            # that connection turns an otherwise healthy provider into repeated
            # read timeouts.  Embedding calls are coarse-grained and bounded; a
            # fresh connection is preferable to a stuck indexing job here.
            "default_headers": _NON_PERSISTENT_CONNECTION_HEADERS,
        }
        if embedding_space.dimension_request_mode == "explicit":
            model_arguments["dimensions"] = embedding_space.dimension
        elif embedding_space.dimension_request_mode != "omitted":
            raise ValueError("unsupported embedding dimension request mode")
        if embedding_model is None:
            model_arguments["http_async_client"] = _non_persistent_http_client(
                max_connections=max_concurrency,
            )
            self._model = OpenAIEmbeddings(**model_arguments)
        else:
            self._model = embedding_model

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
    definition = EmbeddingSpaceDefinition(
        provider_identity="validation",
        endpoint_identity="validation",
        requested_model="validation",
        resolved_model="validation",
        model_version="validation",
        deployment_revision=None,
        dimension=dimension,
        distance_metric="cosine",
        vector_data_type="float32",
        normalization=normalization,
        configuration_fingerprint="validation",
        tokenizer_fingerprint=None,
        compatibility_fingerprint="validation",
    )
    try:
        return normalize_embedding_vector(value, definition)
    except IndexingExecutionError as error:
        diagnostic = dict(error.diagnostic)
        suffix = {
            "numeric_sequence": "",
            "dimension": "dimension",
            "finite_float32": "finite",
            "l2_normalization": "normalization",
            "nonzero_norm": "normalization",
        }.get(str(diagnostic.get("check")), "invalid")
        diagnostic["check"] = f"{check}_{suffix}" if suffix else check
        raise IndexingExecutionError(
            ErrorCode.EMBEDDING_RESPONSE_INVALID,
            phase=IndexingPhase.EMBEDDING,
            diagnostic=diagnostic,
        ) from error


async def probe_openai_embedding_dimension(
    *,
    base_url: str,
    api_key: str,
    model: str,
    requested_dimension: int | None,
    timeout_seconds: float,
    max_retries: int,
) -> int:
    arguments: dict[str, object] = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "timeout": timeout_seconds,
        "max_retries": max_retries,
        "check_embedding_ctx_length": False,
        "model_kwargs": {"encoding_format": "float"},
        "default_headers": _NON_PERSISTENT_CONNECTION_HEADERS,
    }
    if requested_dimension is not None:
        arguments["dimensions"] = requested_dimension
    client = _non_persistent_http_client(max_connections=1)
    arguments["http_async_client"] = client
    adapter = OpenAIEmbeddings(**arguments)
    try:
        value = await adapter.aembed_query("model validation")
    finally:
        await client.aclose()
    if not isinstance(value, (list, tuple)):
        raise ValueError("embedding response is not a vector")
    dimension = len(value)
    if not MIN_EMBEDDING_DIMENSION <= dimension <= MAX_EMBEDDING_DIMENSION:
        raise ValueError("embedding dimension is outside the supported range")
    probe_definition = EmbeddingSpaceDefinition(
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
    normalize_embedding_vector(value, probe_definition)
    return dimension


def _non_persistent_http_client(*, max_connections: int) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers=_NON_PERSISTENT_CONNECTION_HEADERS,
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=0,
        ),
    )


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
