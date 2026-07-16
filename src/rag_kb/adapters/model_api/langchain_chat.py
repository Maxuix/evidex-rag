"""LangChain-backed implementation of the application chat-model contract."""

from __future__ import annotations

import asyncio
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
from rag_kb.domain import (
    ChatModelExecutionError,
    ChatModelRequest,
    ChatModelResponse,
    ErrorCode,
)


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
        structured_output_mode: str = "json_object",
        chat_model: BaseChatModel | None = None,
    ) -> None:
        if not api_key or not model:
            raise ValueError("chat API key and model are required")
        if timeout_seconds <= 0 or max_concurrency <= 0 or max_retries < 0:
            raise ValueError("chat provider limits are invalid")
        if structured_output_mode not in {"json_object", "json_schema"}:
            raise ValueError("unsupported structured output mode")
        self._timeout_seconds = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._structured_output_method = (
            "json_schema" if structured_output_mode == "json_schema" else "json_mode"
        )
        self._structured_models: dict[object, Any] = {}
        self._model = chat_model or ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
            include_response_headers=True,
            use_responses_api=False,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    async def complete(self, request: ChatModelRequest) -> ChatModelResponse:
        if _request_content_bytes(request) > _MAX_REQUEST_CONTENT_BYTES:
            raise ChatModelExecutionError(
                ErrorCode.CHAT_RESPONSE_INVALID,
                diagnostic={"check": "request_content_size"},
            )
        async with self._semaphore:
            try:
                async with asyncio.timeout(self._timeout_seconds):
                    response = await self._invoke(
                        request,
                        to_langchain_messages(request.messages),
                    )
            except TimeoutError as error:
                raise ChatModelExecutionError(
                    ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
                    diagnostic={"check": "total_timeout"},
                ) from error
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

    async def _invoke(self, request: ChatModelRequest, messages: list[Any]) -> Any:
        if request.output_schema is None:
            return await self._model.ainvoke(messages)
        schema = OUTPUT_SCHEMAS.get(request.output_schema)
        if schema is None:
            raise ChatModelExecutionError(
                ErrorCode.CHAT_RESPONSE_INVALID,
                diagnostic={"check": "output_schema"},
            )
        runnable = self._structured_models.get(request.output_schema)
        if runnable is None:
            runnable = self._model.with_structured_output(
                schema,
                method=self._structured_output_method,
                include_raw=True,
            )
            self._structured_models[request.output_schema] = runnable
        result = await runnable.ainvoke(messages)
        return _structured_message(result, schema)


def _request_content_bytes(request: ChatModelRequest) -> int:
    return sum(
        len(message.role.encode("utf-8")) + len(message.content.encode("utf-8"))
        for message in request.messages
    )


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
