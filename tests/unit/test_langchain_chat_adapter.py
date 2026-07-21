from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
import openai
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from rag_kb.adapters import LangChainChatModelAdapter
from rag_kb.answering import WireAnswer
from rag_kb.domain import (
    ChatModelExecutionError,
    ChatModelMessage,
    ChatModelRequest,
    ChatOutputSchema,
    ErrorCode,
)


class _FakeChatModel:
    def __init__(self, response: object = None, *, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[list[object]] = []

    async def ainvoke(self, messages: list[object]) -> object:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.response


def _request(content: str = "hello") -> ChatModelRequest:
    return ChatModelRequest(
        (
            ChatModelMessage("system", "system instruction"),
            ChatModelMessage("user", content),
            ChatModelMessage("assistant", "prior answer"),
        )
    )


def _message(
    *,
    content: str = '{"answer":"ok"}',
    metadata: dict[str, object] | None = None,
    usage: dict[str, int] | None = None,
) -> AIMessage:
    return AIMessage(
        content=content,
        response_metadata=metadata
        or {
            "model_name": "resolved-model",
            "finish_reason": "stop",
            "headers": {"x-request-id": "request-1"},
        },
        usage_metadata=usage,
    )


def _adapter(
    model: object,
    *,
    timeout_seconds: float = 1.0,
    max_concurrency: int = 1,
) -> LangChainChatModelAdapter:
    return LangChainChatModelAdapter(
        base_url="https://provider.invalid/v1",
        api_key="secret",
        model="configured-model",
        timeout_seconds=timeout_seconds,
        max_retries=2,
        max_concurrency=max_concurrency,
        chat_model=model,  # type: ignore[arg-type]
    )


class LangChainChatAdapterTests(unittest.IsolatedAsyncioTestCase):
    def test_constructor_configures_knowledge_base_generation_limits(self) -> None:
        model = _FakeChatModel(_message())
        with patch(
            "rag_kb.adapters.model_api.langchain_chat.ChatOpenAI",
            return_value=model,
        ) as constructor:
            LangChainChatModelAdapter(
                base_url="https://provider.invalid/v1",
                api_key="secret",
                model="configured-model",
                timeout_seconds=30,
                max_retries=2,
                max_concurrency=2,
                temperature=0.1,
                max_tokens=2048,
            )

        arguments = constructor.call_args.kwargs
        self.assertEqual(arguments["temperature"], 0.1)
        self.assertEqual(arguments["extra_body"], {"max_tokens": 2048})

    async def test_real_chatopenai_json_mode_round_trip_uses_public_api(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            self.assertEqual(payload["max_tokens"], 256)
            return httpx.Response(
                200,
                headers={"x-request-id": "structured-request"},
                json={
                    "id": "completion-1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "resolved-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": '{"outcome":"answered","claims":[],"missing_aspects":[]}',
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                },
            )

        async_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            model = ChatOpenAI(
                model="configured-model",
                api_key="probe-key",
                base_url="https://provider.invalid/v1",
                max_retries=0,
                include_response_headers=True,
                use_responses_api=False,
                http_async_client=async_client,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
            response = await _adapter(model).complete(
                ChatModelRequest(
                    (ChatModelMessage("user", "answer"),),
                    output_schema=ChatOutputSchema.ANSWER_V1,
                    max_output_tokens=256,
                )
            )
        finally:
            await async_client.aclose()

        self.assertEqual(response.model, "resolved-model")
        self.assertEqual(response.provider_request_id, "structured-request")
        self.assertEqual(
            response.content,
            '{"claims":[],"missing_aspects":[],"outcome":"answered"}',
        )

    async def test_length_finish_is_returned_as_invalid_wire_not_provider_outage(
        self,
    ) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            del request
            return httpx.Response(
                200,
                json={
                    "id": "truncated-completion",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "resolved-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": ""},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 256,
                        "total_tokens": 356,
                    },
                },
            )

        async_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            model = ChatOpenAI(
                model="configured-model",
                api_key="probe-key",
                base_url="https://provider.invalid/v1",
                max_retries=0,
                include_response_headers=True,
                use_responses_api=False,
                http_async_client=async_client,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
            response = await _adapter(model).complete(
                ChatModelRequest(
                    (ChatModelMessage("user", "acknowledged"),),
                    output_schema=ChatOutputSchema.CONTEXTUAL_QUERY_V2,
                    max_output_tokens=256,
                )
            )
        finally:
            await async_client.aclose()

        self.assertEqual(response.content, '{"_response_truncated":true}')
        self.assertEqual(response.finish_reason, "length")
        self.assertEqual(response.provider_request_id, "truncated-completion")
        self.assertEqual(response.usage["completion_tokens"], 256)

    async def test_empty_structured_message_is_returned_for_business_repair(
        self,
    ) -> None:
        raw = _message(
            content="",
            metadata={
                "model_name": "resolved-model",
                "finish_reason": "length",
                "headers": {"x-request-id": "empty-structured-request"},
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 2048,
                    "total_tokens": 2148,
                },
            },
        )

        class _Runnable:
            async def ainvoke(self, messages: list[object]) -> dict[str, object]:
                del messages
                return {"raw": raw, "parsed": None, "parsing_error": ValueError()}

        class _StructuredModel:
            def with_structured_output(self, *args, **kwargs) -> _Runnable:
                del args, kwargs
                return _Runnable()

        response = await _adapter(_StructuredModel()).complete(
            ChatModelRequest(
                (ChatModelMessage("user", "answer"),),
                output_schema=ChatOutputSchema.ANSWER_V1,
            )
        )

        self.assertEqual(response.content, '{"_response_truncated":true}')
        self.assertEqual(response.finish_reason, "length")
        self.assertEqual(response.provider_request_id, "empty-structured-request")
        self.assertEqual(response.usage["completion_tokens"], 2048)

    async def test_structured_output_maps_schema_and_preserves_raw_metadata(self) -> None:
        raw = _message(content="provider formatting is replaced")
        parsed = WireAnswer(
            outcome="answered",
            claims=[{"text": "Policy applies.", "citation_ids": ["cite_1"]}],
            missing_aspects=[],
        )

        class _Runnable:
            async def ainvoke(self, messages: list[object]) -> dict[str, object]:
                return {"raw": raw, "parsed": parsed, "parsing_error": None}

        class _StructuredModel:
            def __init__(self) -> None:
                self.bindings: list[tuple[object, str, bool]] = []

            def with_structured_output(
                self, schema: object, *, method: str, include_raw: bool
            ) -> _Runnable:
                self.bindings.append((schema, method, include_raw))
                return _Runnable()

            async def ainvoke(self, messages: list[object]) -> object:
                raise AssertionError("structured requests must use the bound runnable")

        model = _StructuredModel()
        adapter = _adapter(model)
        request = ChatModelRequest(
            (ChatModelMessage("user", "answer"),),
            output_schema=ChatOutputSchema.ANSWER_V1,
        )

        first = await adapter.complete(request)
        second = await adapter.complete(request)

        self.assertEqual(
            first.content,
            '{"claims":[{"citation_ids":["cite_1"],"text":"Policy applies."}],'
            '"missing_aspects":[],"outcome":"answered"}',
        )
        self.assertEqual(first.model, "resolved-model")
        self.assertEqual(first.provider_request_id, "request-1")
        self.assertEqual(second.content, first.content)
        self.assertEqual(
            model.bindings,
            [(WireAnswer, "json_mode", True)],
        )

    async def test_structured_request_passes_per_call_output_limit(self) -> None:
        raw = _message(content='{"standalone_query":"expanded topic"}')

        class _Runnable:
            def __init__(self) -> None:
                self.arguments: list[dict[str, object]] = []

            async def ainvoke(
                self, messages: list[object], **kwargs: object
            ) -> dict[str, object]:
                del messages
                self.arguments.append(kwargs)
                return {
                    "raw": raw,
                    "parsed": None,
                    "parsing_error": ValueError(),
                }

        class _StructuredModel:
            def __init__(self) -> None:
                self.runnable = _Runnable()
                self.bindings: list[dict[str, object]] = []

            def bind(self, **kwargs: object) -> _StructuredModel:
                self.bindings.append(kwargs)
                return self

            def with_structured_output(self, *args, **kwargs) -> _Runnable:
                del args, kwargs
                return self.runnable

        model = _StructuredModel()
        adapter = _adapter(model)

        response = await adapter.complete(
            ChatModelRequest(
                (ChatModelMessage("user", "expand"),),
                output_schema=ChatOutputSchema.CONTEXTUAL_QUERY_V2,
                max_output_tokens=256,
            )
        )

        self.assertEqual(model.bindings, [{"max_tokens": 256}])
        self.assertEqual(model.runnable.arguments, [{}])
        self.assertEqual(response.content, '{"standalone_query":"expanded topic"}')

    async def test_structured_parse_failure_preserves_raw_for_business_repair(self) -> None:
        raw = _message(content="not-json")

        class _Runnable:
            async def ainvoke(self, messages: list[object]) -> dict[str, object]:
                return {"raw": raw, "parsed": None, "parsing_error": ValueError()}

        class _StructuredModel:
            def with_structured_output(self, *args, **kwargs) -> _Runnable:
                return _Runnable()

        response = await _adapter(_StructuredModel()).complete(
            ChatModelRequest(
                (ChatModelMessage("user", "answer"),),
                output_schema=ChatOutputSchema.ANSWER_V1,
            )
        )

        self.assertEqual(response.content, "not-json")
        self.assertEqual(response.model, "resolved-model")

    async def test_maps_domain_messages_and_complete_metadata(self) -> None:
        model = _FakeChatModel(
            _message(
                usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
            )
        )

        response = await _adapter(model).complete(_request())

        self.assertEqual(response.content, '{"answer":"ok"}')
        self.assertEqual(response.model, "resolved-model")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.provider_request_id, "request-1")
        self.assertEqual(
            dict(response.usage),
            {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        )
        self.assertEqual(
            [message.type for message in model.calls[0]],
            ["system", "human", "ai"],
        )

    async def test_usage_may_be_missing_or_partial(self) -> None:
        missing = await _adapter(_FakeChatModel(_message())).complete(_request())
        self.assertEqual(dict(missing.usage), {})

        partial_message = _message(
            metadata={
                "model_name": "resolved-model",
                "token_usage": {"prompt_tokens": 7},
            }
        )
        partial = await _adapter(_FakeChatModel(partial_message)).complete(_request())
        self.assertEqual(dict(partial.usage), {"prompt_tokens": 7})

    async def test_reported_model_is_preserved_for_downstream_drift_check(self) -> None:
        response = await _adapter(
            _FakeChatModel(
                _message(metadata={"model_name": "different-provider-model"})
            )
        ).complete(_request())

        self.assertEqual(response.model, "different-provider-model")

    async def test_timeout_and_status_errors_are_stable_and_content_safe(self) -> None:
        class _SlowModel:
            async def ainvoke(self, messages: list[object]) -> AIMessage:
                await asyncio.sleep(0.02)
                return _message()

        with self.assertRaises(ChatModelExecutionError) as timeout:
            await _adapter(_SlowModel(), timeout_seconds=0.001).complete(_request())
        self.assertEqual(timeout.exception.code, ErrorCode.CHAT_PROVIDER_UNAVAILABLE)
        self.assertEqual(timeout.exception.diagnostic, {"check": "total_timeout"})

        for status, retryable in ((429, True), (400, False)):
            request = httpx.Request("POST", "https://secret.invalid/v1")
            response = httpx.Response(status, request=request)
            provider_error = openai.APIStatusError(
                "sensitive provider response",
                response=response,
                body={"secret": "must-not-leak"},
            )
            model = _FakeChatModel(error=provider_error)
            with self.subTest(status=status), self.assertRaises(
                ChatModelExecutionError
            ) as raised:
                await _adapter(model).complete(_request("sensitive prompt"))
            self.assertEqual(raised.exception.code, ErrorCode.CHAT_PROVIDER_UNAVAILABLE)
            self.assertEqual(
                raised.exception.diagnostic,
                {"http_status": status, "retryable": retryable},
            )
            rendered = str(raised.exception.diagnostic)
            self.assertNotIn("sensitive", rendered)
            self.assertNotIn("secret.invalid", rendered)
            self.assertEqual(len(model.calls), 1)

    async def test_invalid_visible_response_metadata_fails_closed(self) -> None:
        cases = (
            AIMessage(content="", response_metadata={}),
            AIMessage(content="{}", response_metadata={}),
            AIMessage(
                content="{}",
                response_metadata={
                    "model_name": "model",
                    "token_usage": {"prompt_tokens": -1},
                },
            ),
            AIMessage(
                content="{}",
                response_metadata={"model_name": "model", "finish_reason": 7},
            ),
            HumanMessage(content="{}"),
        )
        for message in cases:
            with self.subTest(message=message), self.assertRaises(
                ChatModelExecutionError
            ) as raised:
                await _adapter(_FakeChatModel(message)).complete(_request())
            self.assertEqual(raised.exception.code, ErrorCode.CHAT_RESPONSE_INVALID)

    async def test_concurrency_limit_wraps_async_invoke(self) -> None:
        class _ConcurrentModel:
            def __init__(self) -> None:
                self.active = 0
                self.maximum = 0

            async def ainvoke(self, messages: list[object]) -> AIMessage:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
                await asyncio.sleep(0.01)
                self.active -= 1
                return _message()

        model = _ConcurrentModel()
        adapter = _adapter(model, max_concurrency=1)
        await asyncio.gather(adapter.complete(_request()), adapter.complete(_request()))
        self.assertEqual(model.maximum, 1)

    async def test_content_size_limits_apply_without_claiming_wire_size(self) -> None:
        model = _FakeChatModel(_message())
        with self.assertRaises(ChatModelExecutionError) as request_error:
            await _adapter(model).complete(_request("x" * (1024 * 1024)))
        self.assertEqual(
            request_error.exception.diagnostic,
            {"check": "request_content_size"},
        )
        self.assertEqual(model.calls, [])

        oversized = _message(content="x" * (2 * 1024 * 1024 + 1))
        with self.assertRaises(ChatModelExecutionError) as response_error:
            await _adapter(_FakeChatModel(oversized)).complete(_request())
        self.assertEqual(
            response_error.exception.diagnostic,
            {"check": "response_content_size"},
        )


if __name__ == "__main__":
    unittest.main()
