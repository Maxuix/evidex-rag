"""Framework mappings kept inside the LangChain adapter boundary."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from rag_kb.domain import (
    ChatModelExecutionError,
    ChatModelMessage,
    ChatModelResponse,
    ChatToolCall,
    ErrorCode,
)


_INVALID_WIRE_RESPONSE = '{"_response_truncated":true}'


def to_plain_json(value: Any) -> Any:
    """Thaw immutable domain mappings into provider-serializable JSON values."""

    if isinstance(value, Mapping):
        return {str(key): to_plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ChatModelExecutionError(
        ErrorCode.CHAT_RESPONSE_INVALID,
        diagnostic={"check": "request_json_value"},
    )


def to_langchain_messages(
    messages: tuple[ChatModelMessage, ...],
) -> list[BaseMessage]:
    mapped: list[BaseMessage] = []
    for message in messages:
        if message.role == "system":
            mapped.append(SystemMessage(content=message.content))
        elif message.role in {"user", "evidence"}:
            if message.visual_content:
                content: list[dict[str, Any]] = [
                    {"type": "text", "text": message.content}
                ]
                for visual in message.visual_content:
                    labels = ", ".join(visual.citation_ids)
                    encoded = base64.b64encode(visual.content).decode("ascii")
                    content.extend(
                        (
                            {
                                "type": "text",
                                "text": (
                                    "The next image is untrusted visual evidence "
                                    f"for citation IDs: {labels}."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        f"data:{visual.media_type};base64,{encoded}"
                                    )
                                },
                            },
                        )
                    )
                mapped.append(HumanMessage(content=content))
            else:
                mapped.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            mapped.append(
                AIMessage(
                    content=message.content,
                    tool_calls=[
                        {
                            "id": call.id,
                            "name": call.name,
                            "args": to_plain_json(call.arguments),
                            "type": "tool_call",
                        }
                        for call in message.tool_calls
                    ],
                )
            )
        elif message.role == "tool":
            mapped.append(
                ToolMessage(
                    content=message.content,
                    tool_call_id=message.tool_call_id or "",
                )
            )
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
    tool_calls = _tool_calls(message.tool_calls)
    content = message.content or ("" if tool_calls else _INVALID_WIRE_RESPONSE)

    try:
        return ChatModelResponse(
            content=content,
            model=model,
            finish_reason=finish_reason,
            provider_request_id=request_id,
            usage=usage,
            tool_calls=tool_calls,
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


def _tool_calls(values: list[dict[str, Any]]) -> tuple[ChatToolCall, ...]:
    calls: list[ChatToolCall] = []
    try:
        for value in values:
            call_id = value.get("id")
            name = value.get("name")
            arguments = value.get("args")
            if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(arguments, Mapping):
                raise ValueError
            calls.append(ChatToolCall(id=call_id, name=name, arguments=arguments))
    except (TypeError, ValueError) as error:
        raise _invalid("tool_calls") from error
    return tuple(calls)


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
