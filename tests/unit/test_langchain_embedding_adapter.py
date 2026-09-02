from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import openai
from langchain_openai import OpenAIEmbeddings

from rag_kb.adapters.model_api.langchain_embeddings import (
    LangChainEmbeddingModelAdapter,
)
from rag_kb.domain import (
    EmbeddingSpaceDefinition,
    ErrorCode,
    IndexingExecutionError,
)


class _FakeEmbeddings:
    def __init__(
        self,
        *,
        documents: object = None,
        query: object = None,
        error: Exception | None = None,
        delay: float = 0,
    ) -> None:
        self.documents = documents
        self.query = query
        self.error = error
        self.delay = delay
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    async def aembed_documents(self, texts: list[str]) -> object:
        self.document_calls.append(texts)
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.documents

    async def aembed_query(self, text: str) -> object:
        self.query_calls.append(text)
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.query


def _space(
    *,
    dimension: int = 2,
    normalization: str = "l2",
    dimension_request_mode: str = "explicit",
) -> EmbeddingSpaceDefinition:
    return EmbeddingSpaceDefinition(
        provider_identity="alibaba-cloud-model-studio-qwen",
        endpoint_identity="alibaba-model-studio-beijing-embedding",
        requested_model="qwen3.7-text-embedding",
        resolved_model="qwen3.7-text-embedding",
        model_version="qwen3.7-text-embedding",
        deployment_revision=None,
        dimension=dimension,
        distance_metric="cosine",
        vector_data_type="float32",
        normalization=normalization,
        configuration_fingerprint="sha256:configuration",
        tokenizer_fingerprint=None,
        compatibility_fingerprint="sha256:compatibility",
        dimension_request_mode=dimension_request_mode,
    )


def _adapter(
    model: object,
    *,
    timeout_seconds: float = 1.0,
    max_retries: int = 2,
    max_concurrency: int = 1,
    max_batch_size: int = 10,
) -> LangChainEmbeddingModelAdapter:
    return LangChainEmbeddingModelAdapter(
        base_url="https://provider.invalid/v1",
        api_key="secret",
        embedding_space=_space(),
        max_batch_size=max_batch_size,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_concurrency=max_concurrency,
        embedding_model=model,  # type: ignore[arg-type]
    )


class LangChainEmbeddingAdapterTests(unittest.IsolatedAsyncioTestCase):
    def test_constructor_configures_fixed_provider_request(self) -> None:
        model = _FakeEmbeddings()
        http_client = object()
        with (
            patch(
                "rag_kb.adapters.model_api.langchain_embeddings.OpenAIEmbeddings",
                return_value=model,
            ) as constructor,
            patch(
                "rag_kb.adapters.model_api.langchain_embeddings.httpx.AsyncClient",
                return_value=http_client,
            ) as client_constructor,
        ):
            LangChainEmbeddingModelAdapter(
                base_url="https://provider.invalid/v1",
                api_key="secret",
                embedding_space=_space(dimension=1024),
                max_batch_size=10,
                timeout_seconds=30,
                max_retries=2,
                max_concurrency=2,
            )
            LangChainEmbeddingModelAdapter(
                base_url="https://provider.invalid/v1",
                api_key="secret",
                embedding_space=_space(dimension=1024),
                max_batch_size=10,
                timeout_seconds=30,
                max_retries=0,
                max_concurrency=2,
            )

        arguments = constructor.call_args_list[0].kwargs
        self.assertEqual(arguments["model"], "qwen3.7-text-embedding")
        self.assertEqual(arguments["dimensions"], 1024)
        self.assertEqual(arguments["chunk_size"], 10)
        self.assertEqual(arguments["timeout"], 30)
        self.assertEqual(arguments["max_retries"], 0)
        self.assertFalse(arguments["check_embedding_ctx_length"])
        self.assertEqual(
            arguments["model_kwargs"],
            {"encoding_format": "float"},
        )
        self.assertIs(arguments["http_async_client"], http_client)
        self.assertEqual(client_constructor.call_count, 2)
        zero_retry_arguments = constructor.call_args_list[1].kwargs
        self.assertEqual(zero_retry_arguments["timeout"], 30)
        self.assertEqual(zero_retry_arguments["max_retries"], 0)

    async def test_transport_retry_replaces_the_entire_http_client(self) -> None:
        request = httpx.Request("POST", "https://provider.invalid/v1/embeddings")
        first_model = _FakeEmbeddings(error=openai.APITimeoutError(request))
        second_model = _FakeEmbeddings(query=[0.6, 0.8])
        first_client = AsyncMock(spec=httpx.AsyncClient)
        second_client = AsyncMock(spec=httpx.AsyncClient)

        with (
            patch(
                "rag_kb.adapters.model_api.langchain_embeddings.OpenAIEmbeddings",
                side_effect=(first_model, second_model),
            ) as constructor,
            patch(
                "rag_kb.adapters.model_api.langchain_embeddings.httpx.AsyncClient",
                side_effect=(first_client, second_client),
            ),
        ):
            adapter = LangChainEmbeddingModelAdapter(
                base_url="https://provider.invalid/v1",
                api_key="secret",
                embedding_space=_space(),
                max_batch_size=10,
                timeout_seconds=1,
                max_retries=1,
                max_concurrency=1,
            )
            vector = await adapter.embed_query("query")

        self.assertEqual(vector, (0.6, 0.8))
        self.assertEqual(constructor.call_count, 2)
        first_client.aclose.assert_awaited_once()
        self.assertEqual(first_model.query_calls, ["query"])
        self.assertEqual(second_model.query_calls, ["query"])

    def test_constructor_omits_dimension_for_fixed_provider_default(self) -> None:
        with patch(
            "rag_kb.adapters.model_api.langchain_embeddings.OpenAIEmbeddings",
            return_value=_FakeEmbeddings(),
        ) as constructor:
            LangChainEmbeddingModelAdapter(
                base_url="https://provider.invalid/v1",
                api_key="secret",
                embedding_space=_space(
                    dimension=724,
                    dimension_request_mode="omitted",
                ),
                max_batch_size=10,
                timeout_seconds=30,
                max_retries=0,
                max_concurrency=1,
            )

        self.assertNotIn("dimensions", constructor.call_args.kwargs)

    async def test_documents_and_query_use_distinct_langchain_async_methods(self) -> None:
        model = _FakeEmbeddings(
            documents=[[0.6, 0.8], [0.0, 1.0]],
            query=[0.8, 0.6],
        )
        adapter = _adapter(model)

        documents = await adapter.embed_documents(("first", "second"))
        query = await adapter.embed_query("question")

        self.assertEqual(
            documents.vectors,
            ((0.6, 0.8), (0.0, 1.0)),
        )
        self.assertEqual(query, (0.8, 0.6))
        self.assertEqual(model.document_calls, [["first", "second"]])
        self.assertEqual(model.query_calls, ["question"])

    async def test_real_openai_embeddings_round_trip_uses_public_api(self) -> None:
        payloads: list[dict[str, object]] = []
        connection_headers: list[str | None] = []

        def respond(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            payloads.append(payload)
            connection_headers.append(request.headers.get("connection"))
            inputs = payload["input"]
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "qwen3.7-text-embedding",
                    "data": [
                        {
                            "object": "embedding",
                            "index": index,
                            "embedding": [0.6, 0.8],
                        }
                        for index, _ in enumerate(inputs)
                    ],
                    "usage": {
                        "prompt_tokens": len(inputs),
                        "total_tokens": len(inputs),
                    },
                },
            )

        async_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            model = OpenAIEmbeddings(
                model="qwen3.7-text-embedding",
                dimensions=2,
                api_key="probe-key",
                base_url="https://provider.invalid/v1",
                timeout=1,
                max_retries=0,
                chunk_size=10,
                check_embedding_ctx_length=False,
                model_kwargs={"encoding_format": "float"},
                default_headers={"Connection": "close"},
                http_async_client=async_client,
            )
            adapter = _adapter(model)
            documents = await adapter.embed_documents(("first", "second"))
            query = await adapter.embed_query("question")
        finally:
            await async_client.aclose()

        self.assertEqual(documents.vectors, ((0.6, 0.8), (0.6, 0.8)))
        self.assertEqual(query, (0.6, 0.8))
        self.assertEqual(connection_headers, ["close", "close"])
        self.assertEqual(
            payloads,
            [
                {
                    "input": ["first", "second"],
                    "model": "qwen3.7-text-embedding",
                    "dimensions": 2,
                    "encoding_format": "float",
                },
                {
                    "input": ["question"],
                    "model": "qwen3.7-text-embedding",
                    "dimensions": 2,
                    "encoding_format": "float",
                },
            ],
        )

    async def test_batch_bounds_and_invalid_results_fail_closed(self) -> None:
        model = _FakeEmbeddings(documents=[], query="not-a-vector")
        adapter = _adapter(model, max_batch_size=1)

        for texts in ((), ("one", "two")):
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                await adapter.embed_documents(texts)
        self.assertEqual(model.document_calls, [])

        with self.assertRaises(IndexingExecutionError) as count:
            await adapter.embed_documents(("one",))
        self.assertEqual(count.exception.code, ErrorCode.EMBEDDING_RESPONSE_INVALID)
        self.assertEqual(count.exception.diagnostic["check"], "batch_count")

        with self.assertRaises(IndexingExecutionError) as vector:
            await adapter.embed_query("query")
        self.assertEqual(vector.exception.code, ErrorCode.EMBEDDING_RESPONSE_INVALID)
        self.assertEqual(vector.exception.diagnostic["check"], "query_vector")

    async def test_dimension_finite_and_normalization_fail_closed(self) -> None:
        cases = (
            ([1.0], "query_vector_dimension"),
            ([float("nan"), 0.0], "query_vector_finite"),
            ([1.0, 1.0], "query_vector_normalization"),
        )
        for query, expected_check in cases:
            with self.subTest(expected_check=expected_check), self.assertRaises(
                IndexingExecutionError
            ) as raised:
                await _adapter(_FakeEmbeddings(query=query)).embed_query("query")
            self.assertEqual(
                raised.exception.code,
                ErrorCode.EMBEDDING_RESPONSE_INVALID,
            )
            self.assertEqual(
                raised.exception.diagnostic["check"],
                expected_check,
            )

    async def test_client_l2_normalizes_document_and_query_vectors(self) -> None:
        model = _FakeEmbeddings(documents=[[3.0, 4.0]], query=[0.0, 5.0])
        adapter = LangChainEmbeddingModelAdapter(
            base_url="https://provider.invalid/v1",
            api_key="secret",
            embedding_space=_space(normalization="client_l2_v1"),
            max_batch_size=10,
            timeout_seconds=1,
            max_retries=0,
            max_concurrency=1,
            embedding_model=model,
        )

        documents = await adapter.embed_documents(("document",))
        query = await adapter.embed_query("query")

        self.assertEqual(documents.vectors, ((0.6, 0.8),))
        self.assertEqual(query, (0.0, 1.0))

    async def test_timeout_and_status_errors_are_stable_and_content_safe(self) -> None:
        with (
            patch(
                "rag_kb.config.settings."
                "_MAX_PROVIDER_RETRY_AFTER_SECONDS",
                0.001,
            ),
            patch(
                "rag_kb.config.settings."
                "_PROVIDER_TIMEOUT_SCHEDULING_MARGIN_SECONDS",
                0.001,
            ),
            self.assertRaises(IndexingExecutionError) as timeout,
        ):
            await _adapter(
                _FakeEmbeddings(query=[0.6, 0.8], delay=0.02),
                timeout_seconds=0.001,
                max_retries=1,
            ).embed_query("query")
        self.assertEqual(timeout.exception.code, ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE)
        self.assertEqual(timeout.exception.diagnostic, {"check": "total_timeout"})

        for status, retryable in ((429, True), (400, False)):
            request = httpx.Request("POST", "https://secret.invalid/v1")
            response = httpx.Response(status, request=request)
            provider_error = openai.APIStatusError(
                "sensitive provider response",
                response=response,
                body={"secret": "must-not-leak"},
            )
            with self.subTest(status=status), self.assertRaises(
                IndexingExecutionError
            ) as raised:
                await _adapter(
                    _FakeEmbeddings(error=provider_error)
                ).embed_query("sensitive query")
            self.assertEqual(
                raised.exception.diagnostic,
                {"http_status": status, "retryable": retryable},
            )
            rendered = str(raised.exception.diagnostic)
            self.assertNotIn("sensitive", rendered)
            self.assertNotIn("secret.invalid", rendered)

    async def test_outer_timeout_covers_sdk_retry_budget(self) -> None:
        model = _FakeEmbeddings(
            documents=[[0.6, 0.8]],
            query=[0.6, 0.8],
            delay=0.02,
        )
        adapter = _adapter(
            model,
            timeout_seconds=0.01,
            max_retries=1,
        )

        documents = await adapter.embed_documents(("document",))
        query = await adapter.embed_query("query")

        self.assertEqual(documents.vectors, ((0.6, 0.8),))
        self.assertEqual(query, (0.6, 0.8))

    async def test_outer_timeout_uses_exact_derived_budget(self) -> None:
        real_timeout = asyncio.timeout
        observed_budgets: list[float | None] = []

        def recording_timeout(delay: float | None) -> asyncio.Timeout:
            observed_budgets.append(delay)
            return real_timeout(delay)

        with patch(
            "rag_kb.adapters.model_api.langchain_embeddings.asyncio.timeout",
            side_effect=recording_timeout,
        ):
            await _adapter(
                _FakeEmbeddings(query=[0.6, 0.8]),
                timeout_seconds=30,
                max_retries=2,
            ).embed_query("query")
            await _adapter(
                _FakeEmbeddings(query=[0.6, 0.8]),
                timeout_seconds=30,
                max_retries=0,
            ).embed_query("query")

        self.assertEqual(observed_budgets, [211, 31])

    async def test_semaphore_wait_does_not_consume_provider_timeout(self) -> None:
        class _QueuedEmbeddings(_FakeEmbeddings):
            def __init__(self) -> None:
                super().__init__(query=[0.6, 0.8])
                self.first_started = asyncio.Event()
                self.calls = 0

            async def aembed_query(self, text: str) -> object:
                self.query_calls.append(text)
                self.calls += 1
                if self.calls == 1:
                    self.first_started.set()
                    await asyncio.sleep(0.02)
                return self.query

        model = _QueuedEmbeddings()
        adapter = _adapter(model, max_retries=0)
        adapter._total_timeout_seconds = 0.01
        first = asyncio.create_task(adapter.embed_query("first"))
        await model.first_started.wait()
        second = asyncio.create_task(adapter.embed_query("second"))

        first_result, second_result = await asyncio.gather(
            first,
            second,
            return_exceptions=True,
        )

        self.assertIsInstance(first_result, IndexingExecutionError)
        self.assertEqual(
            first_result.diagnostic,  # type: ignore[union-attr]
            {"check": "total_timeout"},
        )
        self.assertEqual(second_result, (0.6, 0.8))

    async def test_concurrency_limit_wraps_both_embedding_operations(self) -> None:
        class _ConcurrentEmbeddings(_FakeEmbeddings):
            def __init__(self) -> None:
                super().__init__(documents=[[0.6, 0.8]], query=[0.6, 0.8])
                self.active = 0
                self.maximum = 0

            async def _run(self, result: object) -> object:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
                await asyncio.sleep(0.01)
                self.active -= 1
                return result

            async def aembed_documents(self, texts: list[str]) -> object:
                self.document_calls.append(texts)
                return await self._run(self.documents)

            async def aembed_query(self, text: str) -> object:
                self.query_calls.append(text)
                return await self._run(self.query)

        model = _ConcurrentEmbeddings()
        adapter = _adapter(model, max_concurrency=1)
        await asyncio.gather(
            adapter.embed_documents(("document",)),
            adapter.embed_query("query"),
        )
        self.assertEqual(model.maximum, 1)


if __name__ == "__main__":
    unittest.main()
