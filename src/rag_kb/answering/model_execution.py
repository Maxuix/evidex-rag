"""Shared phase-aware chat-model execution helpers."""

from __future__ import annotations

from rag_kb.domain import (
    ChatExecutionContext,
    ChatModelCallRecord,
    ChatModelExecutionError,
    ChatModelOperation,
    ChatModelRequest,
    ChatModelResponse,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ErrorCode,
)
from rag_kb.ports.model_api import ChatModelAdapter


async def complete_model(
    model: ChatModelAdapter,
    request: ChatModelRequest,
    *,
    phase: ChatPipelinePhase,
) -> ChatModelResponse:
    try:
        return await model.complete(request)
    except ChatModelExecutionError as error:
        raise ChatPipelineExecutionError(
            error.code,
            phase=phase,
            diagnostic=error.diagnostic,
        ) from error


def require_frozen_model(
    context: ChatExecutionContext,
    response: ChatModelResponse,
    *,
    phase: ChatPipelinePhase,
) -> None:
    expected = context.model_configuration.get("resolved_model")
    if not isinstance(expected, str):
        raise ChatPipelineExecutionError(
            ErrorCode.CHAT_CONTEXT_INVALID,
            phase=phase,
            diagnostic={"check": "model_snapshot"},
        )
    if response.model != expected:
        raise ChatPipelineExecutionError(
            ErrorCode.CHAT_RESPONSE_INVALID,
            phase=phase,
            diagnostic={"check": "resolved_model"},
        )


def model_call_record(
    operation: ChatModelOperation, response: ChatModelResponse
) -> ChatModelCallRecord:
    return ChatModelCallRecord(
        operation=operation,
        model=response.model,
        provider_request_id=response.provider_request_id,
        usage=response.usage,
    )
