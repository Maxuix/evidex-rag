"""Framework-independent values for best-effort live Chat delivery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from rag_kb.domain.chat_workflow import (
    ChatResolvedMode,
    ChatRouteReason,
    ChatRouteStatus,
    ChatWorkflowMode,
    ResearchStatus,
)


CHAT_PREVIEW_VERSION = "chat_preview_v1"
CHAT_PROGRESS_VERSION = "chat_progress_v1"
MAX_PROGRESS_TEXT_LENGTH = 160
MAX_PROGRESS_LIST_ITEMS = 6


class ChatPreviewResetReason(StrEnum):
    GENERATION_FAILED = "generation_failed"
    VALIDATION_REPAIR = "validation_repair"
    PREVIEW_INVALID = "preview_invalid"


class ChatProgressStage(StrEnum):
    UNDERSTAND_QUERY = "understand_query"
    SELECT_WORKFLOW = "select_workflow"
    RETRIEVE_EVIDENCE = "retrieve_evidence"
    ASSESS_EVIDENCE = "assess_evidence"
    PREPARE_VISUAL_EVIDENCE = "prepare_visual_evidence"
    GENERATE_ANSWER = "generate_answer"
    VALIDATE_ANSWER = "validate_answer"
    PERSIST_RESULT = "persist_result"


class ChatProgressActivity(StrEnum):
    LOAD_CONTEXT = "load_context"
    CONTEXTUALIZE_QUERY = "contextualize_query"
    ROUTE_DECISION = "route_decision"
    SIMPLE_SEARCH = "simple_search"
    AGENT_DECISION = "agent_decision"
    AGENT_SEARCH = "agent_search"
    RETRIEVAL_COMPLETE = "retrieval_complete"
    VERIFY_COVERAGE = "verify_coverage"
    RESEARCH_COMPLETE = "research_complete"
    ASSESS_EVIDENCE = "assess_evidence"
    PREPARE_VISUAL_EVIDENCE = "prepare_visual_evidence"
    GENERATE_ANSWER = "generate_answer"
    VALIDATE_ANSWER = "validate_answer"
    PERSIST_RESULT = "persist_result"


class ChatProgressDecision(StrEnum):
    SELECT_SIMPLE = "select_simple"
    SELECT_AGENT = "select_agent"
    SEARCH_EVIDENCE = "search_evidence"
    CONTINUE_SEARCH = "continue_search"
    FINISH_RESEARCH = "finish_research"


class ChatProgressStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class ChatProgressFacts:
    """Small allowlisted facts; never evidence text or free-form reasoning."""

    objective: str | None = None
    queries: tuple[str, ...] = ()
    evidence_count: int | None = None
    new_evidence_count: int | None = None
    retrieval_calls: int | None = None
    route_status: ChatRouteStatus | None = None
    route_reason_codes: tuple[ChatRouteReason, ...] = ()
    research_status: ResearchStatus | None = None
    covered_aspects: tuple[str, ...] = ()
    missing_aspects: tuple[str, ...] = ()
    conflict_count: int | None = None
    decision: ChatProgressDecision | None = None

    def __post_init__(self) -> None:
        if self.objective is not None:
            _validate_progress_text(self.objective, "progress objective")
        for name, values, maximum in (
            ("progress queries", self.queries, 3),
            ("covered progress aspects", self.covered_aspects, MAX_PROGRESS_LIST_ITEMS),
            ("missing progress aspects", self.missing_aspects, MAX_PROGRESS_LIST_ITEMS),
        ):
            if len(values) > maximum or len(values) != len(set(values)):
                raise ValueError(f"{name} exceed their bound or are duplicated")
            for value in values:
                _validate_progress_text(value, name)
        if (
            len(self.route_reason_codes) > MAX_PROGRESS_LIST_ITEMS
            or len(self.route_reason_codes) != len(set(self.route_reason_codes))
        ):
            raise ValueError("progress route reasons are invalid")
        for name, value in (
            ("evidence_count", self.evidence_count),
            ("new_evidence_count", self.new_evidence_count),
            ("retrieval_calls", self.retrieval_calls),
            ("conflict_count", self.conflict_count),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 1_000
            ):
                raise ValueError(f"progress {name} is invalid")


@dataclass(frozen=True, slots=True)
class ChatProgressUpdate:
    active_stage: ChatProgressStage
    activity: ChatProgressActivity
    completed_stages: tuple[ChatProgressStage, ...] = ()
    status: ChatProgressStatus = ChatProgressStatus.ACTIVE
    requested_mode: ChatWorkflowMode | None = None
    resolved_mode: ChatResolvedMode = ChatResolvedMode.PENDING
    facts: ChatProgressFacts = ChatProgressFacts()

    def __post_init__(self) -> None:
        if (
            len(self.completed_stages) > len(ChatProgressStage)
            or len(self.completed_stages) != len(set(self.completed_stages))
        ):
            raise ValueError("completed progress stages are invalid")
        if (
            self.status is ChatProgressStatus.COMPLETED
            and self.active_stage not in self.completed_stages
        ):
            raise ValueError("completed progress must include its active stage")


@dataclass(frozen=True, slots=True)
class ChatProgressSnapshot:
    run_id: UUID
    attempt: int
    seq: int
    update: ChatProgressUpdate

    def __post_init__(self) -> None:
        if self.attempt < 1 or self.seq < 1:
            raise ValueError("chat progress attempt and sequence must be positive")


@dataclass(frozen=True, slots=True)
class ChatPreviewDelta:
    run_id: UUID
    attempt: int
    seq: int
    delta: str

    def __post_init__(self) -> None:
        if self.attempt < 1 or self.seq < 1:
            raise ValueError("chat preview attempt and sequence must be positive")
        if not self.delta:
            raise ValueError("chat preview delta must not be empty")


@dataclass(frozen=True, slots=True)
class ChatPreviewReset:
    run_id: UUID
    attempt: int
    seq: int
    reason: ChatPreviewResetReason

    def __post_init__(self) -> None:
        if self.attempt < 1 or self.seq < 1:
            raise ValueError("chat preview attempt and sequence must be positive")


ChatLiveEvent = ChatPreviewDelta | ChatPreviewReset | ChatProgressSnapshot
ChatPreviewEvent = ChatLiveEvent


def _validate_progress_text(value: str, field: str) -> None:
    if not value.strip() or len(value) > MAX_PROGRESS_TEXT_LENGTH:
        raise ValueError(f"{field} is empty or too long")
