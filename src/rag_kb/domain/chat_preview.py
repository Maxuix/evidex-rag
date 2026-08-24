"""Framework-independent values for best-effort live Chat delivery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID



CHAT_PROGRESS_VERSION = "chat_progress_v1"
MAX_PROGRESS_TEXT_LENGTH = 160
MAX_PROGRESS_LIST_ITEMS = 6


class ChatProgressStage(StrEnum):
    UNDERSTAND_QUERY = "understand_query"
    RETRIEVE_EVIDENCE = "retrieve_evidence"
    PREPARE_VISUAL_EVIDENCE = "prepare_visual_evidence"
    GENERATE_ANSWER = "generate_answer"
    VALIDATE_ANSWER = "validate_answer"
    PERSIST_RESULT = "persist_result"


class ChatProgressActivity(StrEnum):
    LOAD_CONTEXT = "load_context"
    TOOL_DECISION = "tool_decision"
    SEARCH_KNOWLEDGE_BASE = "search_knowledge_base"
    SEARCH_GRAPH_RELATIONS = "search_graph_relations"
    CALCULATE = "calculate"
    SUBMIT_ANSWER = "submit_answer"
    RETRIEVAL_COMPLETE = "retrieval_complete"
    PREPARE_VISUAL_EVIDENCE = "prepare_visual_evidence"
    GENERATE_ANSWER = "generate_answer"
    VALIDATE_ANSWER = "validate_answer"
    PERSIST_RESULT = "persist_result"


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
    covered_aspects: tuple[str, ...] = ()
    missing_aspects: tuple[str, ...] = ()
    conflict_count: int | None = None

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


ChatLiveEvent = ChatProgressSnapshot
ChatPreviewEvent = ChatLiveEvent


def _validate_progress_text(value: str, field: str) -> None:
    if not value.strip() or len(value) > MAX_PROGRESS_TEXT_LENGTH:
        raise ValueError(f"{field} is empty or too long")
