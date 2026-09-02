"""LangChain-backed implementation of the application embedding-model contract."""

from __future__ import annotations

import asyncio
import hashlib
import math
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
    MAX_EMBEDDING_INPUT_UTF8_BYTES,
    MIN_EMBEDDING_DIMENSION,
    normalize_embedding_vector,
)


_RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
_BATCH_PROBE_TIMEOUT_SECONDS = 10.0


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
        self._timeout_seconds = timeout_seconds
        self._total_timeout_seconds = provider_retry_budget_seconds(
            timeout_seconds,
            max_retries,
        )
        self._max_retries = max_retries
        self._max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._model_arguments: dict[str, object] | None = None
        model_arguments: dict[str, object] = {
            "model": embedding_space.requested_model,
            "api_key": api_key,
            "base_url": base_url,
            "timeout": timeout_seconds,
            # Retries are performed below so a transport failure can replace the
            # entire client instead of reusing the same unresponsive route.
            "max_retries": 0,
            "chunk_size": max_batch_size,
            "check_embedding_ctx_length": False,
            "model_kwargs": {"encoding_format": "float"},
        }
        if embedding_space.dimension_request_mode == "explicit":
            model_arguments["dimensions"] = embedding_space.dimension
        elif embedding_space.dimension_request_mode != "omitted":
            raise ValueError("unsupported embedding dimension request mode")
        if embedding_model is None:
            self._model_arguments = model_arguments
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
        windows_by_input = tuple(_utf8_windows(text) for text in texts)
        flattened = tuple(window for windows in windows_by_input for window in windows)
        vectors: list[object] = []
        for offset in range(0, len(flattened), self._max_batch_size):
            batch = flattened[offset : offset + self._max_batch_size]
            vectors.extend(await self._embed_document_vectors(batch, offset=offset))
        normalized: list[tuple[float, ...]] = []
        for index, (text, vector) in enumerate(zip(flattened, vectors, strict=True)):
            try:
                normalized.append(
                    _numeric_vector(
                        vector,
                        check="document_vector",
                        dimension=self._embedding_space.dimension,
                        normalization=self._embedding_space.normalization,
                    )
                )
            except IndexingExecutionError as error:
                raise _with_input_diagnostic(
                    error,
                    text=text,
                    input_index=index,
                ) from error
        collapsed: list[tuple[float, ...]] = []
        vector_offset = 0
        for windows in windows_by_input:
            window_vectors = normalized[vector_offset : vector_offset + len(windows)]
            vector_offset += len(windows)
            collapsed.append(
                _pool_vectors(
                    window_vectors,
                    tuple(len(window.encode("utf-8")) for window in windows),
                )
            )
        return EmbeddingBatch(vectors=tuple(collapsed))

    async def _embed_document_vectors(
        self,
        texts: tuple[str, ...],
        *,
        offset: int,
    ) -> list[object]:
        try:
            vectors = await self._invoke(
                lambda model: model.aembed_documents(list(texts)),
                timeout_seconds=(
                    min(self._timeout_seconds, _BATCH_PROBE_TIMEOUT_SECONDS)
                    if len(texts) > 1
                    else None
                ),
                max_retries=0 if len(texts) > 1 else None,
            )
        except IndexingExecutionError as error:
            if len(texts) > 1 and _is_batch_isolatable(error):
                midpoint = len(texts) // 2
                left = await self._embed_document_vectors(
                    texts[:midpoint],
                    offset=offset,
                )
                right = await self._embed_document_vectors(
                    texts[midpoint:],
                    offset=offset + midpoint,
                )
                return [*left, *right]
            if len(texts) == 1:
                raise _with_input_diagnostic(
                    error,
                    text=texts[0],
                    input_index=offset,
                ) from error
            raise
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            error = _invalid_response(
                "batch_count",
                expected=len(texts),
                observed=len(vectors) if isinstance(vectors, list) else None,
            )
            if len(texts) > 1:
                midpoint = len(texts) // 2
                left = await self._embed_document_vectors(
                    texts[:midpoint],
                    offset=offset,
                )
                right = await self._embed_document_vectors(
                    texts[midpoint:],
                    offset=offset + midpoint,
                )
                return [*left, *right]
            raise _with_input_diagnostic(
                error,
                text=texts[0],
                input_index=offset,
            )
        return list(vectors)

    async def embed_query(self, text: str) -> tuple[float, ...]:
        if not text:
            raise ValueError("embedding query must not be empty")
        windows = _utf8_windows(text)
        vectors: list[tuple[float, ...]] = []
        for window in windows:
            raw_vector = await self._invoke(
                lambda model, value=window: model.aembed_query(value)
            )
            vectors.append(
                _numeric_vector(
                    raw_vector,
                    check="query_vector",
                    dimension=self._embedding_space.dimension,
                    normalization=self._embedding_space.normalization,
                )
            )
        return _pool_vectors(
            vectors,
            tuple(len(window.encode("utf-8")) for window in windows),
        )

    async def _invoke(
        self,
        operation: Callable[[Embeddings], Awaitable[object]],
        *,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ) -> object:
        retry_count = self._max_retries if max_retries is None else max_retries
        total_timeout_seconds = (
            self._total_timeout_seconds
            if timeout_seconds is None and max_retries is None
            else provider_retry_budget_seconds(
                timeout_seconds or self._timeout_seconds,
                retry_count,
            )
        )
        async with self._semaphore:
            try:
                async with asyncio.timeout(total_timeout_seconds):
                    attempt = 0
                    while True:
                        client: httpx.AsyncClient | None = None
                        if self._model_arguments is None:
                            model = self._model
                            if model is None:
                                raise RuntimeError("embedding model is unavailable")
                        else:
                            client, model = self._new_model()
                        try:
                            return await operation(model)
                        except (
                            openai.APITimeoutError,
                            openai.APIConnectionError,
                        ) as error:
                            if (
                                self._model_arguments is None
                                or attempt >= retry_count
                            ):
                                raise _provider_unavailable(
                                    {"check": "transport", "retryable": True}
                                ) from error
                            attempt += 1
                        except openai.APIStatusError as error:
                            status = error.status_code
                            retryable = status in _RETRYABLE_STATUSES or status >= 500
                            if (
                                self._model_arguments is None
                                or not retryable
                                or attempt >= retry_count
                            ):
                                raise _provider_unavailable(
                                    {
                                        "http_status": status,
                                        "retryable": retryable,
                                    }
                                ) from error
                            attempt += 1
                            await asyncio.sleep(min(0.5 * (2**attempt), 5.0))
                        finally:
                            if client is not None:
                                await client.aclose()
            except TimeoutError as error:
                raise _provider_unavailable({"check": "total_timeout"}) from error
            except openai.OpenAIError as error:
                raise _provider_unavailable({"check": "provider_sdk"}) from error
            except IndexingExecutionError:
                raise
            except (KeyError, TypeError, ValueError, IndexError) as error:
                raise _invalid_response("provider_result") from error

    def _new_model(self) -> tuple[httpx.AsyncClient, OpenAIEmbeddings]:
        if self._model_arguments is None:
            raise RuntimeError("embedding model factory is unavailable")
        # Some otherwise OpenAI-compatible gateways close or poison an idle
        # HTTP/1.1 connection without signalling it correctly.  Disabling idle
        # reuse is a provider-neutral correctness trade-off: every logical call
        # gets a clean transport route while in-flight concurrency is preserved.
        client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=self._max_concurrency,
                max_keepalive_connections=0,
            )
        )
        return client, OpenAIEmbeddings(
            **self._model_arguments,
            http_async_client=client,
        )


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


def _utf8_windows(text: str) -> tuple[str, ...]:
    if len(text.encode("utf-8")) <= MAX_EMBEDDING_INPUT_UTF8_BYTES:
        return (text,)
    windows: list[str] = []
    current: list[str] = []
    current_bytes = 0
    for character in text:
        character_bytes = len(character.encode("utf-8"))
        if current and current_bytes + character_bytes > MAX_EMBEDDING_INPUT_UTF8_BYTES:
            windows.append("".join(current))
            current = []
            current_bytes = 0
        current.append(character)
        current_bytes += character_bytes
    if current:
        windows.append("".join(current))
    return tuple(windows)


def _pool_vectors(
    vectors: list[tuple[float, ...]] | tuple[tuple[float, ...], ...],
    weights: tuple[int, ...],
) -> tuple[float, ...]:
    if len(vectors) == 1:
        return vectors[0]
    total_weight = sum(weights)
    pooled = tuple(
        math.fsum(
            vector[index] * weight
            for vector, weight in zip(vectors, weights, strict=True)
        )
        / total_weight
        for index in range(len(vectors[0]))
    )
    norm = math.sqrt(math.fsum(value * value for value in pooled))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise _invalid_response("document_vector_window_pooling")
    return tuple(value / norm for value in pooled)


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
    }
    if requested_dimension is not None:
        arguments["dimensions"] = requested_dimension
    adapter = OpenAIEmbeddings(**arguments)
    value = await adapter.aembed_query("model validation")
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


def _provider_unavailable(diagnostic: dict[str, object]) -> IndexingExecutionError:
    return IndexingExecutionError(
        ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE,
        phase=IndexingPhase.EMBEDDING,
        diagnostic=diagnostic,
    )


def _is_batch_isolatable(error: IndexingExecutionError) -> bool:
    if error.code == ErrorCode.EMBEDDING_RESPONSE_INVALID:
        return True
    if error.code != ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE:
        return False
    check = error.diagnostic.get("check")
    if check in {"transport", "total_timeout", "provider_result"}:
        return True
    return error.diagnostic.get("http_status") in {400, 408, 409, 413, 422}


def _with_input_diagnostic(
    error: IndexingExecutionError,
    *,
    text: str,
    input_index: int,
) -> IndexingExecutionError:
    diagnostic = dict(error.diagnostic)
    diagnostic.update(
        {
            "input_index": input_index,
            "input_characters": len(text),
            "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    )
    return IndexingExecutionError(
        error.code,
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
