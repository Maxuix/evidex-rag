"""Framework-independent evidence assessment and answer-draft contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any
from uuid import UUID


class EvidenceCoverage(StrEnum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    NONE = "none"
    AMBIGUOUS = "ambiguous"


class AnswerOutcome(StrEnum):
    ANSWERED = "answered"
    PARTIAL = "partial"
    REFUSED = "refused"


class AnswerDraftSource(StrEnum):
    PROVIDER = "provider"
    DETERMINISTIC = "deterministic"


class AnswerControlReason(StrEnum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NO_USABLE_EVIDENCE = "no_usable_evidence"
    AMBIGUOUS_QUESTION = "ambiguous_question"


class ChatModelOperation(StrEnum):
    ASSESS_EVIDENCE = "assess_evidence"
    GENERATE_ANSWER = "generate_answer"


@dataclass(frozen=True, slots=True)
class PromptEvidence:
    citation_id: str
    rank: int
    index_chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    excerpt: str
    source_location: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.citation_id != f"cite_{self.rank}" or self.rank < 1:
            raise ValueError("prompt evidence citation identifier must match its rank")
        if not self.excerpt:
            raise ValueError("prompt evidence excerpt must not be empty")
        object.__setattr__(
            self, "source_location", MappingProxyType(dict(self.source_location))
        )


@dataclass(frozen=True, slots=True)
class EvidenceEnvelope:
    knowledge_base_id: UUID
    index_revision_id: UUID
    items: tuple[PromptEvidence, ...]

    def __post_init__(self) -> None:
        expected = [f"cite_{rank}" for rank in range(1, len(self.items) + 1)]
        if [item.citation_id for item in self.items] != expected:
            raise ValueError("prompt evidence must have contiguous citation identifiers")
        if len(self.items) > 100:
            raise ValueError("prompt evidence exceeds the retrieval limit")

    @property
    def citation_ids(self) -> frozenset[str]:
        return frozenset(item.citation_id for item in self.items)


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    coverage: EvidenceCoverage
    usable_citation_ids: tuple[str, ...]
    supported_aspects: tuple[str, ...]
    missing_aspects: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_bounded_unique_strings(
            self.usable_citation_ids, field="usable citation identifiers"
        )
        _require_bounded_unique_strings(
            self.supported_aspects, field="supported aspects"
        )
        _require_bounded_unique_strings(
            self.missing_aspects, field="missing aspects"
        )
        if self.coverage is EvidenceCoverage.SUFFICIENT and (
            not self.usable_citation_ids
            or not self.supported_aspects
            or self.missing_aspects
        ):
            raise ValueError("sufficient evidence has support and no missing aspects")
        if self.coverage is EvidenceCoverage.PARTIAL and (
            not self.usable_citation_ids
            or not self.supported_aspects
            or not self.missing_aspects
        ):
            raise ValueError("partial evidence requires support and missing aspects")
        if self.coverage is EvidenceCoverage.NONE and (
            self.usable_citation_ids or self.supported_aspects
        ):
            raise ValueError("no usable evidence cannot declare supported aspects")
        if self.coverage is EvidenceCoverage.AMBIGUOUS and (
            self.usable_citation_ids
            or self.supported_aspects
            or not self.missing_aspects
        ):
            raise ValueError("ambiguous questions require only missing clarification")


@dataclass(frozen=True, slots=True)
class ChatModelCallRecord:
    operation: ChatModelOperation
    model: str
    provider_request_id: str | None
    usage: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model call record requires a model")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.usage.values()
        ):
            raise ValueError("model call usage values must be non-negative integers")
        object.__setattr__(self, "usage", MappingProxyType(dict(self.usage)))


@dataclass(frozen=True, slots=True)
class AnswerDraftCandidate:
    raw_json: str
    expected_outcome: AnswerOutcome
    source: AnswerDraftSource
    control_reason: AnswerControlReason | None = None

    def __post_init__(self) -> None:
        if not self.raw_json:
            raise ValueError("answer draft must not be empty")
        if self.source is AnswerDraftSource.DETERMINISTIC:
            if (
                self.expected_outcome is not AnswerOutcome.REFUSED
                or self.control_reason is None
            ):
                raise ValueError("deterministic drafts must be controlled refusals")
        elif self.control_reason is not None:
            raise ValueError("provider drafts cannot declare a control reason")


@dataclass(frozen=True, slots=True)
class ChatAnsweringState:
    evidence: EvidenceEnvelope
    assessment: EvidenceAssessment
    draft: AnswerDraftCandidate | None = None
    model_calls: tuple[ChatModelCallRecord, ...] = ()


def _require_bounded_unique_strings(values: tuple[str, ...], *, field: str) -> None:
    if len(values) > 100 or len(values) != len(set(values)):
        raise ValueError(f"{field} must be unique and bounded")
    if any(not value.strip() or len(value) > 1000 for value in values):
        raise ValueError(f"{field} contain invalid values")
