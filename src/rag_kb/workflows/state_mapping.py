"""Small, execution-only state shared by the fixed chat graph nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired, TypedDict

from rag_kb.domain import (
    ChatExecutionCommand,
    ChatExecutionContext,
    ChatModelCallRecord,
    ChatPipelinePhase,
    ChatPipelineState,
    EvidencePack,
    ContextualizedQuery,
    ChatWorkflowState,
)
from rag_kb.services.chat_progress import ChatProgressReporter


@dataclass(slots=True)
class ChatGraphProgress:
    phase: ChatPipelinePhase = ChatPipelinePhase.LOAD_CONTEXT
    model_calls: tuple[ChatModelCallRecord, ...] = ()
    reporter: ChatProgressReporter | None = None


class ChatGraphState(TypedDict):
    command: ChatExecutionCommand
    progress: ChatGraphProgress
    context: NotRequired[ChatExecutionContext]
    query_context: NotRequired[ContextualizedQuery]
    workflow_state: NotRequired[ChatWorkflowState]
    workflow_model_calls: NotRequired[tuple[ChatModelCallRecord, ...]]
    evidence_pack: NotRequired[EvidencePack]
    pipeline_state: NotRequired[ChatPipelineState]


def retain_progress(
    progress: ChatGraphProgress,
    state: ChatPipelineState,
) -> None:
    if state.answering is not None:
        progress.model_calls = state.answering.model_calls
