"""Bounded OpenAI-compatible JSON chat adapter without SDK coupling."""

from __future__ import annotations

import asyncio
import json
import random
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rag_kb.domain import (
    ChatModelExecutionError,
    ChatModelRequest,
    ChatModelResponse,
    ErrorCode,
)


_MAX_REQUEST_BYTES = 1024 * 1024
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class OpenAICompatibleChatModelAdapter:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        max_concurrency: int,
        retryable_statuses: frozenset[int] = frozenset({429, 500, 503}),
    ) -> None:
        if not api_key or not model:
            raise ValueError("chat API key and model are required")
        if timeout_seconds <= 0 or max_concurrency <= 0 or max_retries < 0:
            raise ValueError("chat provider limits are invalid")
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retryable_statuses = retryable_statuses
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def complete(self, request: ChatModelRequest) -> ChatModelResponse:
        async with self._semaphore:
            try:
                async with asyncio.timeout(self._timeout_seconds):
                    return await self._complete_with_retries(request)
            except TimeoutError as error:
                raise ChatModelExecutionError(
                    ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
                    diagnostic={"check": "total_timeout"},
                ) from error

    async def _complete_with_retries(
        self, request: ChatModelRequest
    ) -> ChatModelResponse:
        for attempt in range(self._max_retries + 1):
            try:
                return await asyncio.to_thread(self._request, request)
            except _RetryableProviderError:
                if attempt >= self._max_retries:
                    break
                delay = 0.2 * (2**attempt) * random.uniform(0.8, 1.2)
                await asyncio.sleep(delay)
        raise ChatModelExecutionError(
            ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
            diagnostic={"retry_exhausted": True},
        )

    def _request(self, request_value: ChatModelRequest) -> ChatModelResponse:
        payload = json.dumps(
            {
                "model": self._model,
                "messages": [
                    {"role": item.role, "content": item.content}
                    for item in request_value.messages
                ],
                "response_format": {"type": "json_object"},
                "stream": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(payload) > _MAX_REQUEST_BYTES:
            raise ChatModelExecutionError(
                ErrorCode.CHAT_RESPONSE_INVALID,
                diagnostic={"check": "request_size"},
            )
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
                request_id = response.headers.get("x-request-id")
        except HTTPError as error:
            if error.code in self._retryable_statuses:
                raise _RetryableProviderError from error
            raise ChatModelExecutionError(
                ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
                diagnostic={"http_status": error.code, "retryable": False},
            ) from error
        except (URLError, TimeoutError, HTTPException, OSError) as error:
            raise _RetryableProviderError from error
        if len(body) > _MAX_RESPONSE_BYTES:
            raise ChatModelExecutionError(
                ErrorCode.CHAT_RESPONSE_INVALID,
                diagnostic={"check": "response_size"},
            )
        return self._decode(body, provider_request_id=request_id)

    @staticmethod
    def _decode(body: bytes, *, provider_request_id: str | None) -> ChatModelResponse:
        try:
            payload: Any = json.loads(body)
            model = payload["model"]
            choices = payload["choices"]
            usage_value = payload.get("usage", {})
            if (
                not isinstance(model, str)
                or not isinstance(choices, list)
                or len(choices) != 1
                or not isinstance(usage_value, dict)
            ):
                raise TypeError
            choice = choices[0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            if (
                not isinstance(content, str)
                or not content
                or (finish_reason is not None and not isinstance(finish_reason, str))
            ):
                raise TypeError
            usage = {
                key: value
                for key, value in usage_value.items()
                if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
            }
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in usage.values()
            ):
                raise TypeError
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ChatModelExecutionError(
                ErrorCode.CHAT_RESPONSE_INVALID,
                diagnostic={"check": "wire_shape"},
            ) from error
        return ChatModelResponse(
            content=content,
            model=model,
            finish_reason=finish_reason,
            provider_request_id=provider_request_id,
            usage=usage,
        )


class _RetryableProviderError(RuntimeError):
    pass
