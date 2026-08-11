"""LangChain-backed implementation of the application chat-model contract."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
import json
from typing import Any

import openai
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from rag_kb.adapters.model_api.langchain_mapping import (
    from_langchain_message,
    to_langchain_messages,
)
from rag_kb.answering.wire_schemas import OUTPUT_SCHEMAS
from rag_kb.config.settings import provider_retry_budget_seconds
from rag_kb.domain import (
    ChatModelExecutionError,
    ChatModelRequest,
    ChatModelResponse,
    ErrorCode,
)
from rag_kb.ports.model_api import ChatModelContentDeltaHandler


_MAX_REQUEST_CONTENT_BYTES = 1024 * 1024
_MAX_RESPONSE_CONTENT_BYTES = 2 * 1024 * 1024
_RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})


class LangChainChatModelAdapter:
    """Use ChatOpenAI asynchronously while preserving the domain contract."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        max_concurrency: int,
        temperature: float = 0.1,
        top_p: float | None = None,
        sampling_top_k: int | None = None,
        max_tokens: int = 2048,
        structured_output_mode: str = "json_object",
        thinking_enabled: bool = False,
        reasoning_effort: str = "off",
        max_visual_images: int = 4,
        max_visual_image_bytes: int = 5 * 1024 * 1024,
        max_visual_total_bytes: int = 12 * 1024 * 1024,
        chat_model: BaseChatModel | None = None,
    ) -> None:
        if not api_key or not model:
            raise ValueError("chat API key and model are required")
        if timeout_seconds <= 0 or max_concurrency <= 0 or max_retries < 0:
            raise ValueError("chat provider limits are invalid")
        if not 0.0 <= temperature <= 2.0 or max_tokens <= 0:
            raise ValueError("chat generation limits are invalid")
        if top_p is not None and not 0.0 < top_p <= 1.0:
            raise ValueError("chat top_p is invalid")
        if sampling_top_k is not None and sampling_top_k < 1:
            raise ValueError("chat sampling top_k is invalid")
        if reasoning_effort not in {"off", "low", "medium", "high"}:
            raise ValueError("chat reasoning effort is invalid")
        if structured_output_mode not in {"json_object", "json_schema"}:
            raise ValueError("unsupported structured output mode")
        if (
            max_visual_images < 1
            or max_visual_image_bytes < 1
            or max_visual_total_bytes < max_visual_image_bytes
        ):
            raise ValueError("chat visual input limits are invalid")
        self._timeout_seconds = timeout_seconds
        self._total_budget_seconds = provider_retry_budget_seconds(
            timeout_seconds,
            max_retries,
        )
        self._max_tokens = max_tokens
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._structured_output_method = (
            "json_schema" if structured_output_mode == "json_schema" else "json_mode"
        )
        self._structured_models: dict[
            tuple[object, int | None, bool | None], Any
        ] = {}
        self._max_visual_images = max_visual_images
        self._max_visual_image_bytes = max_visual_image_bytes
        self._max_visual_total_bytes = max_visual_total_bytes
        extra_body: dict[str, Any] = {
            "max_tokens": max_tokens,
            "enable_thinking": thinking_enabled or reasoning_effort != "off",
        }
        if sampling_top_k is not None:
            extra_body["top_k"] = sampling_top_k
        if reasoning_effort != "off":
            extra_body["reasoning_effort"] = reasoning_effort
        self._model = chat_model or ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
            temperature=temperature,
            top_p=top_p,
            include_response_headers=True,
            use_responses_api=False,
            extra_body=extra_body,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    async def complete(self, request: ChatModelRequest) -> ChatModelResponse:
        self._validate_request(request)
        response = await self._execute(
            lambda: self._invoke(
                request,
                to_langchain_messages(request.messages),
            )
        )
        return _validated_response(response)

    async def complete_streaming(
        self,
        request: ChatModelRequest,
        *,
        on_content_delta: ChatModelContentDeltaHandler,
    ) -> ChatModelResponse:
        self._validate_request(request)
        response = await self._execute(
            lambda: self._stream(
                request,
                to_langchain_messages(request.messages),
                on_content_delta,
            )
        )
        return _validated_response(response)

    def _validate_request(self, request: ChatModelRequest) -> None:
        if _request_content_bytes(request) > _MAX_REQUEST_CONTENT_BYTES:
            raise ChatModelExecutionError(
                ErrorCode.CHAT_RESPONSE_INVALID,
                diagnostic={"check": "request_content_size"},
            )
        visual_content = tuple(
            item
            for message in request.messages
            for item in message.visual_content
        )
        if (
            len(visual_content) > self._max_visual_images
            or any(
                len(item.content) > self._max_visual_image_bytes
                for item in visual_content
            )
            or sum(len(item.content) for item in visual_content)
            > self._max_visual_total_bytes
        ):
            raise ChatModelExecutionError(
                ErrorCode.CHAT_RESPONSE_INVALID,
                diagnostic={"check": "request_visual_size"},
            )

    async def _execute(
        self,
        operation: Callable[[], Awaitable[Any]],
    ) -> Any:
        async with self._semaphore:
            try:
                async with asyncio.timeout(self._total_budget_seconds):
                    return await operation()
            except TimeoutError as error:
                raise ChatModelExecutionError(
                    ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
                    diagnostic={"check": "total_timeout"},
                ) from error
            except openai.LengthFinishReasonError as error:
                return _truncated_response(error)
            except openai.APIStatusError as error:
                status = error.status_code
                raise ChatModelExecutionError(
                    ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
                    diagnostic={
                        "http_status": status,
                        "retryable": status in _RETRYABLE_STATUSES or status >= 500,
                    },
                ) from error
            except (openai.APITimeoutError, openai.APIConnectionError) as error:
                raise ChatModelExecutionError(
                    ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
                    diagnostic={"check": "transport", "retryable": True},
                ) from error
            except openai.OpenAIError as error:
                raise ChatModelExecutionError(
                    ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
                    diagnostic={"check": "provider_sdk"},
                ) from error

    async def _invoke(self, request: ChatModelRequest, messages: list[Any]) -> Any:
        output_limit = self._output_limit(request)
        model = _model_with_request_options(
            self._model,
            max_tokens=output_limit,
            thinking_enabled=request.thinking_enabled,
        )
        if request.output_schema is None:
            return await model.ainvoke(messages)
        schema = OUTPUT_SCHEMAS.get(request.output_schema)
        if schema is None:
            raise ChatModelExecutionError(
                ErrorCode.CHAT_RESPONSE_INVALID,
                diagnostic={"check": "output_schema"},
            )
        cache_key = (
            request.output_schema,
            output_limit,
            request.thinking_enabled,
        )
        runnable = self._structured_models.get(cache_key)
        if runnable is None:
            runnable = model.with_structured_output(
                schema,
                method=self._structured_output_method,
                include_raw=True,
            )
            self._structured_models[cache_key] = runnable
        result = await runnable.ainvoke(messages)
        return _structured_message(result, schema)

    async def _stream(
        self,
        request: ChatModelRequest,
        messages: list[Any],
        on_content_delta: ChatModelContentDeltaHandler,
    ) -> ChatModelResponse:
        schema = None
        if request.output_schema is not None:
            schema = OUTPUT_SCHEMAS.get(request.output_schema)
            if schema is None:
                raise ChatModelExecutionError(
                    ErrorCode.CHAT_RESPONSE_INVALID,
                    diagnostic={"check": "output_schema"},
                )
        output_limit = self._output_limit(request)
        model = _model_with_request_options(
            self._model,
            max_tokens=output_limit,
            thinking_enabled=request.thinking_enabled,
        )
        combined = None
        content_bytes = 0
        async for chunk in model.astream(messages, stream_usage=True):
            combined = chunk if combined is None else combined + chunk
            content = chunk.content
            if isinstance(content, str) and content:
                content_bytes += len(content.encode("utf-8"))
                if content_bytes > _MAX_RESPONSE_CONTENT_BYTES:
                    raise ChatModelExecutionError(
                        ErrorCode.CHAT_RESPONSE_INVALID,
                        diagnostic={"check": "response_content_size"},
                    )
                try:
                    await on_content_delta(content)
                except Exception:
                    pass
        if combined is None:
            raise ChatModelExecutionError(
                ErrorCode.CHAT_RESPONSE_INVALID,
                diagnostic={"check": "stream_empty"},
            )
        mapped = from_langchain_message(combined)
        if schema is None:
            return mapped
        try:
            parsed = schema.model_validate_json(mapped.content)
        except ValueError:
            return mapped
        canonical = json.dumps(
            parsed.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return replace(mapped, content=canonical)

    def _output_limit(self, request: ChatModelRequest) -> int | None:
        if request.max_output_tokens is None:
            return None
        return min(request.max_output_tokens, self._max_tokens)


def _model_with_request_options(
    model: Any,
    *,
    max_tokens: int | None,
    thinking_enabled: bool | None,
) -> Any:
    if max_tokens is None and thinking_enabled is None:
        return model
    model_copy = getattr(model, "model_copy", None)
    if callable(model_copy):
        extra_body = dict(getattr(model, "extra_body", None) or {})
        if max_tokens is not None:
            extra_body["max_tokens"] = max_tokens
        if thinking_enabled is not None:
            extra_body["enable_thinking"] = thinking_enabled
            if not thinking_enabled:
                extra_body.pop("reasoning_effort", None)
        return model_copy(update={"extra_body": extra_body})
    arguments: dict[str, Any] = {}
    if max_tokens is not None:
        arguments["max_tokens"] = max_tokens
    if thinking_enabled is not None:
        extra_body = dict(getattr(model, "extra_body", None) or {})
        extra_body["enable_thinking"] = thinking_enabled
        if not thinking_enabled:
            extra_body.pop("reasoning_effort", None)
        arguments["extra_body"] = extra_body
    return model.bind(**arguments)


def _truncated_response(error: openai.LengthFinishReasonError) -> ChatModelResponse:
    completion = error.completion
    if not completion.choices or not completion.model:
        raise ChatModelExecutionError(
            ErrorCode.CHAT_RESPONSE_INVALID,
            diagnostic={"check": "truncated_completion"},
        ) from error
    usage: dict[str, int] = {}
    if completion.usage is not None:
        for source, target in (
            ("prompt_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            value = getattr(completion.usage, source, None)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                usage[target] = value
    return ChatModelResponse(
        content='{"_response_truncated":true}',
        model=completion.model,
        finish_reason=completion.choices[0].finish_reason,
        provider_request_id=completion.id or None,
        usage=usage,
    )


def _request_content_bytes(request: ChatModelRequest) -> int:
    return sum(
        len(message.role.encode("utf-8")) + len(message.content.encode("utf-8"))
        for message in request.messages
    )


def _validated_response(response: Any) -> ChatModelResponse:
    mapped = (
        response
        if isinstance(response, ChatModelResponse)
        else from_langchain_message(response)
    )
    if len(mapped.content.encode("utf-8")) > _MAX_RESPONSE_CONTENT_BYTES:
        raise ChatModelExecutionError(
            ErrorCode.CHAT_RESPONSE_INVALID,
            diagnostic={"check": "response_content_size"},
        )
    return mapped


def _structured_message(value: object, schema: type[BaseModel]) -> ChatModelResponse:
    if not isinstance(value, dict):
        raise ChatModelExecutionError(
            ErrorCode.CHAT_RESPONSE_INVALID,
            diagnostic={"check": "structured_result"},
        )
    raw = value.get("raw")
    parsed = value.get("parsed")
    mapped = from_langchain_message(raw)
    if parsed is None:
        return mapped
    if not isinstance(parsed, schema):
        raise ChatModelExecutionError(
            ErrorCode.CHAT_RESPONSE_INVALID,
            diagnostic={"check": "structured_schema"},
        )
    canonical = json.dumps(
        parsed.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return replace(mapped, content=canonical)
