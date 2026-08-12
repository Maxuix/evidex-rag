"""Application services for atomic chat success and failure settlement."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from rag_kb.domain import (
    ChatFailureSettlementCommand,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatPipelineState,
    ChatRunLease,
    ChatTerminalSuccessCommand,
    ChatTerminalWriteStatus,
    ErrorCode,
)
from rag_kb.answering.agent import AGENT_TRACE_ARTIFACT
from rag_kb.uow import UnitOfWork, UnitOfWorkFactory, execute_in_transaction


class ChatResultPersistenceStep:
    """Persist a W05-safe answer and all terminal facts in one transaction."""

    def __init__(
        self,
        unit_of_work: UnitOfWorkFactory,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._clock = clock or (lambda: datetime.now(UTC))

    async def run(self, state: ChatPipelineState) -> ChatPipelineState:
        context = state.context
        answering = state.answering
        if (
            context is None
            or answering is None
            or answering.validated is None
            or answering.rendered is None
            or answering.validation is None
        ):
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_CONTEXT_INVALID,
                phase=ChatPipelinePhase.PERSIST_RESULT,
                diagnostic={"check": "validated_answer_state"},
                model_calls=answering.model_calls if answering is not None else (),
            )
        command = ChatTerminalSuccessCommand(
            lease=context.lease,
            assistant_message_id=context.assistant_message_id,
            rendered=answering.rendered,
            validation=answering.validation,
            model_calls=answering.model_calls,
            finished_at=self._clock(),
            retrieval_diagnostics=_retrieval_diagnostics(state),
            visual_decisions=answering.visual_decisions,
            visual_image_count=len(answering.visual_content),
            visual_total_bytes=answering.visual_total_bytes,
            final_llm_context=_final_llm_context(state),
            agent_trace=_agent_trace(state),
        )

        async def persist(uow: UnitOfWork) -> ChatTerminalWriteStatus:
            return await uow.chat.complete_owned_run(command)

        try:
            result = await execute_in_transaction(self._unit_of_work, persist)
        except ChatPipelineExecutionError:
            raise
        except Exception as error:
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_PERSISTENCE_FAILED,
                phase=ChatPipelinePhase.PERSIST_RESULT,
                diagnostic={"operation": "terminal_success"},
                model_calls=answering.model_calls,
            ) from error
        if result is ChatTerminalWriteStatus.STALE:
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_CONTEXT_INVALID,
                phase=ChatPipelinePhase.PERSIST_RESULT,
                diagnostic={"check": "terminal_lease"},
                model_calls=answering.model_calls,
            )
        return state


def _agent_trace(state: ChatPipelineState) -> dict[str, Any] | None:
    value = state.artifacts.get(AGENT_TRACE_ARTIFACT)
    if value is None:
        return None
    as_dict = getattr(value, "as_dict", None)
    if not callable(as_dict):
        raise ChatPipelineExecutionError(
            ErrorCode.CHAT_CONTEXT_INVALID,
            phase=ChatPipelinePhase.PERSIST_RESULT,
            diagnostic={"check": "agent_trace"},
        )
    return as_dict()


class ChatFailureSettlementService:
    """Classify and atomically requeue or terminally fail one owned attempt."""

    def __init__(
        self,
        unit_of_work: UnitOfWorkFactory,
        *,
        max_attempts: int,
        base_delay_seconds: float,
        max_delay_seconds: float,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if max_attempts < 1 or base_delay_seconds <= 0 or max_delay_seconds <= 0:
            raise ValueError("chat retry settings must be positive")
        if base_delay_seconds > max_delay_seconds:
            raise ValueError("chat retry base delay cannot exceed maximum")
        self._unit_of_work = unit_of_work
        self._max_attempts = max_attempts
        self._base_delay_seconds = base_delay_seconds
        self._max_delay_seconds = max_delay_seconds
        self._clock = clock or (lambda: datetime.now(UTC))

    async def settle(
        self, lease: ChatRunLease, error: ChatPipelineExecutionError
    ) -> ChatTerminalWriteStatus:
        finished_at = self._clock()
        retryable = _is_retryable(error)
        exhausted = retryable and lease.attempt >= self._max_attempts
        next_attempt_at = None
        if retryable and not exhausted:
            delay = min(
                self._max_delay_seconds,
                self._base_delay_seconds * (2 ** (lease.attempt - 1)),
            )
            next_attempt_at = finished_at + timedelta(seconds=delay)
        command = ChatFailureSettlementCommand(
            lease=lease,
            phase=error.phase,
            code=error.code,
            diagnostic=_safe_diagnostic(error.diagnostic),
            model_calls=error.model_calls,
            retryable=retryable,
            exhausted=exhausted,
            finished_at=finished_at,
            next_attempt_at=next_attempt_at,
        )

        async def persist(uow: UnitOfWork) -> ChatTerminalWriteStatus:
            return await uow.chat.settle_owned_failure(command)

        return await execute_in_transaction(self._unit_of_work, persist)


_RETRYABLE_CODES = frozenset(
    {
        ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
        ErrorCode.CHAT_ASSESSMENT_INVALID,
        ErrorCode.CHAT_PIPELINE_DEADLINE_EXCEEDED,
        ErrorCode.CHAT_PERSISTENCE_FAILED,
        ErrorCode.CHAT_STALE_WORKER,
        ErrorCode.CHAT_WORKER_STOPPED,
    }
)
_SAFE_DIAGNOSTIC_KEYS = frozenset(
    {
        "check",
        "operation",
        "http_status",
        "retryable",
        "retry_exhausted",
        "limit",
    }
)


def _is_retryable(error: ChatPipelineExecutionError) -> bool:
    if (
        error.code is ErrorCode.CHAT_PROVIDER_UNAVAILABLE
        and error.diagnostic.get("retryable") is False
    ):
        return False
    if error.code in _RETRYABLE_CODES:
        return True
    return bool(
        error.code is ErrorCode.CHAT_RESPONSE_INVALID
        and error.diagnostic.get("check") in {"wire_shape", "response_wire"}
    )


def _safe_diagnostic(value: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in _SAFE_DIAGNOSTIC_KEYS:
        item = value.get(key)
        if isinstance(item, (str, int, float, bool)) and not isinstance(item, bytes):
            safe[key] = item
    return safe


def _retrieval_diagnostics(state: ChatPipelineState) -> dict[str, int]:
    pack = state.evidence_pack
    debug = getattr(pack, "debug", None) if pack is not None else None
    if debug is None:
        return {}
    values = {
        "result_count": debug.result_count,
        "text_candidate_count": debug.text_candidate_count,
        "cross_modal_candidate_count": debug.cross_modal_candidate_count,
        "hydrated_relation_count": debug.hydrated_relation_count,
        "evidence_group_count": debug.evidence_group_count,
        "model_rerank_candidate_count": getattr(
            debug, "model_rerank_candidate_count", None
        ),
        "model_rerank_window_count": getattr(
            debug, "model_rerank_window_count", None
        ),
    }
    return {key: value for key, value in values.items() if value is not None}


def _final_llm_context(state: ChatPipelineState) -> dict[str, Any] | None:
    value = state.artifacts.get("final_llm_context")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ChatPipelineExecutionError(
            ErrorCode.CHAT_CONTEXT_INVALID,
            phase=ChatPipelinePhase.PERSIST_RESULT,
            diagnostic={"check": "final_llm_context"},
        )
    return _plain_json_object(value)


def _plain_json_object(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _plain_json_value(item) for key, item in value.items()}


def _plain_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _plain_json_object(value)
    if isinstance(value, (tuple, list)):
        return [_plain_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ChatPipelineExecutionError(
        ErrorCode.CHAT_CONTEXT_INVALID,
        phase=ChatPipelinePhase.PERSIST_RESULT,
        diagnostic={"check": "final_llm_context"},
    )
