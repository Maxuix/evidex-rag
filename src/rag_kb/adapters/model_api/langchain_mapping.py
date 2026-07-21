"""Framework mappings kept inside the LangChain adapter boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from rag_kb.domain import (
    ChatModelExecutionError,
    ChatModelMessage,
    ChatModelResponse,
    ErrorCode,
)


_INVALID_WIRE_RESPONSE = '{"_response_truncated":true}'


def to_langchain_messages(
    messages: tuple[ChatModelMessage, ...],
) -> list[BaseMessage]:
    mapped: list[BaseMessage] = []
    for message in messages:
        if message.role == "system":
            mapped.append(SystemMessage(content=message.content))
        elif message.role == "user":
            mapped.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            mapped.append(AIMessage(content=message.content))
        else:  # The domain validates roles; retain a fail-closed adapter boundary.
            raise ChatModelExecutionError(
                ErrorCode.CHAT_RESPONSE_INVALID,
                diagnostic={"check": "message_role"},
            )
    return mapped


def from_langchain_message(message: BaseMessage) -> ChatModelResponse:
    if not isinstance(message, AIMessage):
        raise _invalid("message_type")
    if not isinstance(message.content, str):
        raise _invalid("content")

    metadata = message.response_metadata
    model = _metadata_string(metadata, "model_name") or _metadata_string(
        metadata, "model"
    )
    if model is None:
        raise _invalid("resolved_model")
    finish_reason = _metadata_string(metadata, "finish_reason")
    request_id = _request_id(metadata, message.additional_kwargs)
    usage = _usage(metadata.get("token_usage"), message.usage_metadata)
    content = message.content or _INVALID_WIRE_RESPONSE

    try:
        return ChatModelResponse(
            content=content,
            model=model,
            finish_reason=finish_reason,
            provider_request_id=request_id,
            usage=usage,
        )
    except ValueError as error:
        raise _invalid("message_metadata") from error


def _request_id(
    metadata: Mapping[str, Any], additional_kwargs: Mapping[str, Any]
) -> str | None:
    direct = _metadata_string(metadata, "request_id") or _metadata_string(
        additional_kwargs, "request_id"
    )
    if direct is not None:
        return direct
    headers = metadata.get("headers")
    if isinstance(headers, Mapping):
        return _metadata_string(headers, "x-request-id") or _metadata_string(
            headers, "request-id"
        )
    return None


def _usage(
    token_usage: object,
    usage_metadata: Mapping[str, Any] | None,
) -> dict[str, int]:
    usage: dict[str, int] = {}
    if isinstance(token_usage, Mapping):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            _copy_usage_value(usage, key, token_usage.get(key))
    elif token_usage is not None:
        raise _invalid("usage")

    if usage_metadata is not None:
        for source, target in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            if target not in usage:
                _copy_usage_value(usage, target, usage_metadata.get(source))
    return usage


def _copy_usage_value(target: dict[str, int], key: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _invalid("usage")
    target[key] = value


def _metadata_string(values: Mapping[str, Any], key: str) -> str | None:
    value = values.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise _invalid("message_metadata")
    return value


def _invalid(check: str) -> ChatModelExecutionError:
    return ChatModelExecutionError(
        ErrorCode.CHAT_RESPONSE_INVALID,
        diagnostic={"check": check},
    )
