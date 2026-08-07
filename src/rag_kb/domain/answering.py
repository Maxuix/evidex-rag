"""Framework-independent evidence assessment and answer-draft contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import re
from types import MappingProxyType
from typing import Any
from uuid import UUID

from rag_kb.domain.composite import VisualEvidenceDecision
from rag_kb.domain.chat_workflow import ResearchAspect, ResearchStatus


class EvidenceCoverage(StrEnum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    NONE = "none"
    AMBIGUOUS = "ambiguous"
    CONFLICT = "conflict"


class AnswerOutcome(StrEnum):
    ANSWERED = "answered"
    PARTIAL = "partial"
    ACKNOWLEDGED = "acknowledged"
    REFUSED = "refused"


class AnswerDraftSource(StrEnum):
    PROVIDER = "provider"
    DETERMINISTIC = "deterministic"


class AnswerControlReason(StrEnum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NO_USABLE_EVIDENCE = "no_usable_evidence"
    AMBIGUOUS_QUESTION = "ambiguous_question"
    CONFLICT_UNRESOLVED = "conflict_unresolved"
    STRUCTURE_VALIDATION_FAILED = "structure_validation_failed"


class ChatModelOperation(StrEnum):
    CONTEXTUALIZE_QUERY = "contextualize_query"
    ASSESS_EVIDENCE = "assess_evidence"
    GENERATE_ANSWER = "generate_answer"
    REPAIR_ANSWER = "repair_answer"
    RETRIEVAL_AGENT = "retrieval_agent"
    REPAIR_RETRIEVAL_AGENT = "repair_retrieval_agent"
    VERIFY_RESEARCH_RESULT = "verify_research_result"
    REPAIR_RESEARCH_RESULT = "repair_research_result"
    AUTO_ROUTE = "auto_route"
    REPAIR_AUTO_ROUTE = "repair_auto_route"


class RetrievalAgentActionKind(StrEnum):
    SEARCH = "search"
    FINISH = "finish"


class RetrievalAgentProposedReason(StrEnum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    NO_EVIDENCE = "no_evidence"
    NO_PROGRESS = "no_progress"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CONFLICT_UNRESOLVED = "conflict_unresolved"
    PREMISE_UNSUPPORTED = "premise_unsupported"


@dataclass(frozen=True, slots=True)
class RetrievalAgentQuery:
    query: str
    based_on_observation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.query.strip() or len(self.query) > 2048:
            raise ValueError("Agent query is invalid")
        _require_bounded_unique_strings(
            self.based_on_observation_ids,
            field="Agent observation references",
        )


@dataclass(frozen=True, slots=True)
class RetrievalAgentAction:
    action: RetrievalAgentActionKind
    objective: str | None = None
    queries: tuple[RetrievalAgentQuery, ...] = ()
    proposed_reason: RetrievalAgentProposedReason | None = None
    selected_evidence_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.action is RetrievalAgentActionKind.SEARCH:
            if (
                self.objective is None
                or not self.objective.strip()
                or len(self.objective) > 1024
                or not 1 <= len(self.queries) <= 3
                or self.proposed_reason is not None
                or self.selected_evidence_keys
            ):
                raise ValueError("Agent search action is invalid")
        elif (
            self.objective is not None
            or self.queries
            or self.proposed_reason is None
        ):
            raise ValueError("Agent finish action is invalid")
        _require_bounded_unique_strings(
            self.selected_evidence_keys,
            field="Agent selected evidence",
        )


@dataclass(frozen=True, slots=True)
class RetrievalToolObservation:
    observation_id: str
    objective: str
    queries: tuple[str, ...]
    result: str
    new_evidence_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.observation_id.strip() or len(self.observation_id) > 128:
            raise ValueError("retrieval observation identifier is invalid")
        if not self.objective.strip() or len(self.objective) > 1024:
            raise ValueError("retrieval observation objective is invalid")
        _require_bounded_unique_strings(self.queries, field="observation queries")
        _require_bounded_unique_strings(
            self.new_evidence_keys, field="observation evidence"
        )
        if self.result not in {"evidence_found", "no_evidence", "verification_gap"}:
            raise ValueError("retrieval observation result is invalid")


@dataclass(frozen=True, slots=True)
class ResearchResultVerification:
    status: ResearchStatus
    aspects: tuple[ResearchAspect, ...]
    missing_aspects: tuple[str, ...]
    conflicts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.aspects or len(self.aspects) > 100:
            raise ValueError("research verification aspects are invalid")
        _require_bounded_unique_strings(
            self.missing_aspects, field="verification missing aspects"
        )
        _require_bounded_unique_strings(
            self.conflicts, field="verification conflicts"
        )


class AnswerValidationIssue(StrEnum):
    JSON_INVALID = "json_invalid"
    SCHEMA_INVALID = "schema_invalid"
    OUTCOME_MISMATCH = "outcome_mismatch"
    CLAIMS_REQUIRED = "claims_required"
    CLAIMS_FORBIDDEN = "claims_forbidden"
    MISSING_ASPECTS_FORBIDDEN = "missing_aspects_forbidden"
    MISSING_ASPECTS_REQUIRED = "missing_aspects_required"
    MISSING_ASPECTS_MISMATCH = "missing_aspects_mismatch"
    CITATIONS_REQUIRED = "citations_required"
    CITATION_DUPLICATE = "citation_duplicate"
    CITATION_NOT_ALLOWED = "citation_not_allowed"
    RENDER_LIMIT_EXCEEDED = "render_limit_exceeded"


@dataclass(frozen=True, slots=True)
class PromptEvidence:
    citation_id: str
    rank: int
    index_chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    document_display_name: str
    document_original_filename: str
    excerpt: str
    source_location: Mapping[str, Any]
    score: float | None = None
    modality: str = "text"
    asset_snapshot: Mapping[str, Any] | None = None
    matched_representations: tuple[str, ...] = ("text",)

    def __post_init__(self) -> None:
        if self.citation_id != f"cite_{self.rank}" or self.rank < 1:
            raise ValueError("prompt evidence citation identifier must match its rank")
        if (
            not self.excerpt
            or not self.document_display_name.strip()
            or not self.document_original_filename.strip()
        ):
            raise ValueError("prompt evidence excerpt must not be empty")
        object.__setattr__(
            self, "source_location", MappingProxyType(dict(self.source_location))
        )
        if self.asset_snapshot is not None:
            object.__setattr__(
                self,
                "asset_snapshot",
                MappingProxyType(dict(self.asset_snapshot)),
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
        if len(self.items) > 104:
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
class ChatModelVisualContent:
    citation_ids: tuple[str, ...]
    asset_id: UUID
    media_type: str
    checksum_sha256: str
    content: bytes
    width: int
    height: int

    def __post_init__(self) -> None:
        if (
            not self.citation_ids
            or len(self.citation_ids) != len(set(self.citation_ids))
            or any(
                re.fullmatch(r"cite_[1-9][0-9]*", value) is None
                for value in self.citation_ids
            )
        ):
            raise ValueError("visual content citation identifiers are invalid")
        if self.media_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("visual content media type is unsupported")
        if (
            re.fullmatch(r"[0-9a-f]{64}", self.checksum_sha256) is None
            or not self.content
            or hashlib.sha256(self.content).hexdigest() != self.checksum_sha256
        ):
            raise ValueError("visual content checksum is invalid")
        if self.width < 1 or self.height < 1:
            raise ValueError("visual content dimensions must be positive")


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
        elif (
            self.control_reason is not None
            or self.expected_outcome is AnswerOutcome.REFUSED
        ):
            raise ValueError("provider drafts must be substantive without control reasons")


@dataclass(frozen=True, slots=True)
class AnswerClaim:
    text: str
    citation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.text.strip() or len(self.text) > 4000:
            raise ValueError("answer claim text is invalid")
        if (
            not self.citation_ids
            or len(self.citation_ids) > 100
            or len(self.citation_ids) != len(set(self.citation_ids))
            or any(not value.strip() for value in self.citation_ids)
        ):
            raise ValueError("answer claim citations are invalid")


@dataclass(frozen=True, slots=True)
class ValidatedAnswer:
    outcome: AnswerOutcome
    claims: tuple[AnswerClaim, ...]
    missing_aspects: tuple[str, ...]
    source: AnswerDraftSource
    control_reason: AnswerControlReason | None = None

    def __post_init__(self) -> None:
        if len(self.claims) > 100:
            raise ValueError("validated answer has too many claims")
        _require_bounded_unique_strings(
            self.missing_aspects, field="validated missing aspects"
        )
        if self.outcome is AnswerOutcome.ANSWERED:
            if not self.claims or self.missing_aspects:
                raise ValueError("answered results require claims and no gaps")
        elif self.outcome is AnswerOutcome.PARTIAL:
            if not self.claims or not self.missing_aspects:
                raise ValueError("partial results require claims and gaps")
        elif self.claims or self.missing_aspects:
            raise ValueError("non-substantive results cannot contain answer content")
        if self.source is AnswerDraftSource.DETERMINISTIC:
            if (
                self.outcome is not AnswerOutcome.REFUSED
                or self.control_reason is None
            ):
                raise ValueError("deterministic validated results must be refusals")
        elif self.outcome is AnswerOutcome.REFUSED:
            if self.control_reason is not AnswerControlReason.INSUFFICIENT_EVIDENCE:
                raise ValueError("provider refusals require a safe control reason")
        elif self.control_reason is not None:
            raise ValueError("provider answers cannot have a control reason")


@dataclass(frozen=True, slots=True)
class RenderedCitation:
    ordinal: int
    citation_id: str
    index_chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    document_display_name: str
    document_original_filename: str
    quoted_text: str
    source_location: Mapping[str, Any]
    score: float | None
    modality: str = "text"
    asset_snapshot: Mapping[str, Any] | None = None
    matched_representations: tuple[str, ...] = ("text",)

    def __post_init__(self) -> None:
        if (
            self.ordinal < 0
            or not self.citation_id
            or not self.quoted_text
            or not self.document_display_name.strip()
            or not self.document_original_filename.strip()
        ):
            raise ValueError("rendered citation identity is invalid")
        object.__setattr__(
            self, "source_location", MappingProxyType(dict(self.source_location))
        )
        if self.asset_snapshot is not None:
            object.__setattr__(
                self,
                "asset_snapshot",
                MappingProxyType(dict(self.asset_snapshot)),
            )


@dataclass(frozen=True, slots=True)
class RenderedAnswer:
    outcome: AnswerOutcome
    content: str
    citations: tuple[RenderedCitation, ...]
    control_reason: AnswerControlReason | None = None

    def __post_init__(self) -> None:
        if (
            not self.content.strip()
            or len(self.content.encode("utf-8")) > 2 * 1024 * 1024
        ):
            raise ValueError("rendered answer content is invalid")
        if [item.ordinal for item in self.citations] != list(
            range(len(self.citations))
        ):
            raise ValueError("rendered citations must be contiguous")
        if len({item.citation_id for item in self.citations}) != len(self.citations):
            raise ValueError("rendered citations must be unique")
        if self.outcome in {
            AnswerOutcome.REFUSED,
            AnswerOutcome.ACKNOWLEDGED,
        } and self.citations:
            raise ValueError("non-substantive results cannot contain citations")
        if self.outcome is not AnswerOutcome.REFUSED and self.control_reason is not None:
            raise ValueError("only refusals can carry a control reason")
        if self.outcome in {
            AnswerOutcome.ANSWERED,
            AnswerOutcome.PARTIAL,
        } and not self.citations:
            raise ValueError("substantive results require citations")


@dataclass(frozen=True, slots=True)
class AnswerValidationRecord:
    initial_issues: tuple[AnswerValidationIssue, ...]
    repair_issues: tuple[AnswerValidationIssue, ...] = ()
    repair_attempted: bool = False
    repair_succeeded: bool = False
    safe_fallback: bool = False

    def __post_init__(self) -> None:
        if len(self.initial_issues) != len(set(self.initial_issues)):
            raise ValueError("initial validation issues must be unique")
        if len(self.repair_issues) != len(set(self.repair_issues)):
            raise ValueError("repair validation issues must be unique")
        if not self.initial_issues:
            if (
                self.repair_issues
                or self.repair_attempted
                or self.repair_succeeded
                or self.safe_fallback
            ):
                raise ValueError("successful initial validation cannot repair or fallback")
            return
        if not self.repair_attempted:
            if self.repair_issues or self.repair_succeeded or not self.safe_fallback:
                raise ValueError("unrepaired invalid structure must safely fallback")
            return
        if self.repair_succeeded:
            if self.repair_issues or self.safe_fallback:
                raise ValueError("successful repair state is inconsistent")
        elif not self.repair_issues or not self.safe_fallback:
            raise ValueError("failed repair must record issues and safely fallback")


@dataclass(frozen=True, slots=True)
class ChatAnsweringState:
    evidence: EvidenceEnvelope
    assessment: EvidenceAssessment
    draft: AnswerDraftCandidate | None = None
    model_calls: tuple[ChatModelCallRecord, ...] = ()
    visual_content: tuple[ChatModelVisualContent, ...] = ()
    visual_decisions: tuple[VisualEvidenceDecision, ...] = ()
    visual_total_bytes: int = 0
    validated: ValidatedAnswer | None = None
    rendered: RenderedAnswer | None = None
    validation: AnswerValidationRecord | None = None

    def __post_init__(self) -> None:
        if self.visual_total_bytes < 0:
            raise ValueError("visual evidence bytes must be non-negative")
        if len(self.visual_decisions) > 400:
            raise ValueError("visual evidence decisions must be bounded")
        if sum(len(item.content) for item in self.visual_content) != self.visual_total_bytes:
            raise ValueError("visual evidence bytes must match attached content")
        visual_citations = [
            citation_id
            for item in self.visual_content
            for citation_id in item.citation_ids
        ]
        visual_assets = [item.asset_id for item in self.visual_content]
        if len(visual_assets) != len(set(visual_assets)):
            raise ValueError("visual evidence assets must be unique")
        if any(
            citation_id not in self.assessment.usable_citation_ids
            for citation_id in visual_citations
        ):
            raise ValueError("visual evidence must be admitted before model input")
        completed = (self.validated, self.rendered, self.validation)
        if any(value is not None for value in completed) and any(
            value is None for value in completed
        ):
            raise ValueError("validated answer state must be complete")
        if self.validated is not None and self.draft is None:
            raise ValueError("validated answer state requires its original draft")
        if self.rendered is not None and self.validated is not None:
            if self.rendered.outcome is not self.validated.outcome:
                raise ValueError("validated and rendered outcomes must match")
        if (
            self.validation is not None
            and self.validation.safe_fallback
            and self.validated is not None
            and (
                self.validated.source is not AnswerDraftSource.DETERMINISTIC
                or self.validated.control_reason
                not in {
                    AnswerControlReason.INSUFFICIENT_EVIDENCE,
                    AnswerControlReason.STRUCTURE_VALIDATION_FAILED,
                }
            )
        ):
            raise ValueError("safe fallback must use a deterministic refusal")
        if (
            self.validation is not None
            and self.validation.repair_succeeded
            and self.validated is not None
            and self.validated.source is not AnswerDraftSource.PROVIDER
        ):
            raise ValueError("successful repair must produce a provider result")


def _require_bounded_unique_strings(values: tuple[str, ...], *, field: str) -> None:
    if len(values) > 100 or len(values) != len(set(values)):
        raise ValueError(f"{field} must be unique and bounded")
    if any(not value.strip() or len(value) > 1000 for value in values):
        raise ValueError(f"{field} contain invalid values")
