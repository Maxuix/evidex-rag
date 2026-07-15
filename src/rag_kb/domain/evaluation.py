"""Framework-independent evaluation persistence facts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID


class EvaluationRunState(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationConflictError(RuntimeError):
    """A stable evaluation identity was reused with different immutable input."""


@dataclass(frozen=True, slots=True)
class EvaluationCaseDefinition:
    case_key: str
    question: str
    expected: dict[str, Any]
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.case_key or not self.question.strip():
            raise ValueError("evaluation cases require a key and question")
        if len(self.tags) != len(set(self.tags)):
            raise ValueError("evaluation case tags must be unique")
        object.__setattr__(self, "question", self.question.strip())
        object.__setattr__(self, "expected", dict(self.expected))


@dataclass(frozen=True, slots=True)
class EvaluationDatasetDefinition:
    name: str
    version: str
    manifest_hash: str
    metadata: dict[str, Any]
    cases: tuple[EvaluationCaseDefinition, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.version or len(self.manifest_hash) != 64:
            raise ValueError("evaluation dataset identity is invalid")
        case_keys = [case.case_key for case in self.cases]
        if not self.cases or len(case_keys) != len(set(case_keys)):
            raise ValueError("evaluation dataset case keys must be non-empty and unique")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class EvaluationRunDefinition:
    run_id: UUID
    knowledge_base_id: UUID
    index_revision_id: UUID
    dataset: EvaluationDatasetDefinition
    run_config: dict[str, Any]
    started_at: datetime

    def __post_init__(self) -> None:
        if not self.run_config or self.started_at.tzinfo is None:
            raise ValueError("evaluation run config and timezone-aware start are required")
        object.__setattr__(self, "run_config", dict(self.run_config))


@dataclass(frozen=True, slots=True)
class EvaluationCaseResult:
    case_key: str
    evidence: dict[str, Any] | None
    metrics: dict[str, Any]
    error_code: str | None = None
    error_detail: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.case_key or not self.metrics:
            raise ValueError("evaluation result requires a case key and metrics")
        if (self.error_code is None) != (self.error_detail is None):
            raise ValueError("evaluation result error code and detail must appear together")
        if self.evidence is not None:
            object.__setattr__(self, "evidence", dict(self.evidence))
        object.__setattr__(self, "metrics", dict(self.metrics))
        if self.error_detail is not None:
            object.__setattr__(self, "error_detail", dict(self.error_detail))


@dataclass(frozen=True, slots=True)
class EvaluationRunSnapshot:
    run_id: UUID
    dataset_id: UUID
    state: EvaluationRunState
    result_count: int
    error_code: str | None
