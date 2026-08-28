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


class AnswerOutcome(StrEnum):
    ANSWERED = "answered"
    PARTIAL = "partial"
    REFUSED = "refused"
    CLARIFY = "clarify"


class AnswerDraftSource(StrEnum):
    PROVIDER = "provider"
    DETERMINISTIC = "deterministic"


class AnswerControlReason(StrEnum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NO_USABLE_EVIDENCE = "no_usable_evidence"


class AnswerConflictType(StrEnum):
    TEMPORAL = "temporal"
    VERSION = "version"
    OPINION = "opinion"
    MISINFORMATION = "misinformation"
    UNKNOWN = "unknown"


class AnswerConflictAdjudication(StrEnum):
    RESOLVABLE = "resolvable"
    UNRESOLVABLE = "unresolvable"


class ChatModelOperation(StrEnum):
    AGENT_ROUND = "agent_round"
    AGENT_VERIFIER = "agent_verifier"
    CONTEXTUALIZE_QUERY = "contextualize_query"


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
    graph_path_id: str | None = None
    graph_anchor_index_chunk_id: UUID | None = None
    graph_hop_count: int | None = None
    graph_path_rank: int | None = None

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
        graph_values = (
            self.graph_path_id,
            self.graph_anchor_index_chunk_id,
            self.graph_hop_count,
            self.graph_path_rank,
        )
        if any(value is not None for value in graph_values) and (
            self.graph_path_id is None
            or self.graph_anchor_index_chunk_id is None
            or self.graph_hop_count not in {1, 2, 3}
            or self.graph_path_rank is None
            or self.graph_path_rank < 1
        ):
            raise ValueError("prompt graph path metadata is incomplete")


@dataclass(frozen=True, slots=True)
class EvidenceEnvelope:
    knowledge_base_id: UUID
    index_revision_id: UUID
    items: tuple[PromptEvidence, ...]

    def __post_init__(self) -> None:
        expected = [f"cite_{rank}" for rank in range(1, len(self.items) + 1)]
        if [item.citation_id for item in self.items] != expected:
            raise ValueError("prompt evidence must have contiguous citation identifiers")

    @property
    def citation_ids(self) -> frozenset[str]:
        return frozenset(item.citation_id for item in self.items)


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
class AnswerConflict:
    supporting_citation_ids: tuple[str, ...]
    conflicting_citation_ids: tuple[str, ...]
    conflict_type: AnswerConflictType
    adjudication: AnswerConflictAdjudication

    def __post_init__(self) -> None:
        _require_unique_strings(
            self.supporting_citation_ids, field="answer conflict supporting citations"
        )
        _require_unique_strings(
            self.conflicting_citation_ids, field="answer conflict conflicting citations"
        )
        if not self.supporting_citation_ids or not self.conflicting_citation_ids:
            raise ValueError("answer conflict sides must be non-empty")
        if set(self.supporting_citation_ids) & set(self.conflicting_citation_ids):
            raise ValueError("answer conflict sides must be disjoint")


@dataclass(frozen=True, slots=True)
class AnswerClaim:
    text: str
    citation_ids: tuple[str, ...]
    conflict: AnswerConflict | None = None

    def __post_init__(self) -> None:
        if not self.text.strip() or len(self.text) > 4000:
            raise ValueError("answer claim text is invalid")
        if (
            not self.citation_ids
            or len(self.citation_ids) != len(set(self.citation_ids))
            or any(not value.strip() for value in self.citation_ids)
        ):
            raise ValueError("answer claim citations are invalid")
        if self.conflict is not None:
            conflict_ids = set(self.conflict.supporting_citation_ids) | set(
                self.conflict.conflicting_citation_ids
            )
            if not conflict_ids.issubset(self.citation_ids):
                raise ValueError("answer conflict citations must belong to the claim")


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
        elif self.outcome is AnswerOutcome.CLARIFY:
            if self.claims or not self.missing_aspects:
                raise ValueError("clarify results require questions and no claims")
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
        if self.outcome in {AnswerOutcome.REFUSED, AnswerOutcome.CLARIFY} and self.citations:
            raise ValueError("non-substantive results cannot contain citations")
        if self.outcome is not AnswerOutcome.REFUSED and self.control_reason is not None:
            raise ValueError("only refusals can carry a control reason")
        if self.outcome in {
            AnswerOutcome.ANSWERED,
            AnswerOutcome.PARTIAL,
        } and not self.citations:
            raise ValueError("substantive results require citations")


@dataclass(frozen=True, slots=True)
class ChatAnsweringState:
    evidence: EvidenceEnvelope
    usable_citation_ids: tuple[str, ...]
    draft: AnswerDraftCandidate | None = None
    model_calls: tuple[ChatModelCallRecord, ...] = ()
    visual_content: tuple[ChatModelVisualContent, ...] = ()
    visual_decisions: tuple[VisualEvidenceDecision, ...] = ()
    visual_total_bytes: int = 0
    validated: ValidatedAnswer | None = None
    rendered: RenderedAnswer | None = None

    def __post_init__(self) -> None:
        if self.visual_total_bytes < 0:
            raise ValueError("visual evidence bytes must be non-negative")
        if len(self.visual_decisions) > 400:
            raise ValueError("visual evidence decisions must be bounded")
        _require_unique_strings(
            self.usable_citation_ids, field="usable citation identifiers"
        )
        if not set(self.usable_citation_ids).issubset(self.evidence.citation_ids):
            raise ValueError("usable citations must belong to the evidence envelope")
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
            citation_id not in self.usable_citation_ids
            for citation_id in visual_citations
        ):
            raise ValueError("visual evidence must be admitted before model input")
        completed = (self.validated, self.rendered)
        if any(value is not None for value in completed) and any(
            value is None for value in completed
        ):
            raise ValueError("validated answer state must be complete")
        if self.validated is not None and self.draft is None:
            raise ValueError("validated answer state requires its original draft")
        if self.rendered is not None and self.validated is not None:
            if self.rendered.outcome is not self.validated.outcome:
                raise ValueError("validated and rendered outcomes must match")


def _require_bounded_unique_strings(values: tuple[str, ...], *, field: str) -> None:
    if len(values) > 100 or len(values) != len(set(values)):
        raise ValueError(f"{field} must be unique and bounded")
    if any(not value.strip() or len(value) > 1000 for value in values):
        raise ValueError(f"{field} contain invalid values")


def _require_unique_strings(values: tuple[str, ...], *, field: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field} must be unique")
    if any(not value.strip() or len(value) > 1000 for value in values):
        raise ValueError(f"{field} contain invalid values")
