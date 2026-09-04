"""Framework-independent evidence and answer types."""

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


class ChatModelOperation(StrEnum):
    AGENT_ROUND = "agent_round"
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
class AnswerClaim:
    """Internal normalized answer text and its resolved citation identifiers."""

    text: str
    citation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ValidatedAnswer:
    """Normalized answer result; invariants belong to the active answer boundary."""

    outcome: AnswerOutcome
    claims: tuple[AnswerClaim, ...]
    missing_aspects: tuple[str, ...]
    source: AnswerDraftSource
    control_reason: AnswerControlReason | None = None


@dataclass(frozen=True, slots=True)
class RenderedCitation:
    """Display order over already-admitted evidence; metadata is not copied."""

    ordinal: int
    evidence: PromptEvidence


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


@dataclass(frozen=True, slots=True)
class ChatAnsweringState:
    evidence: EvidenceEnvelope
    usable_citation_ids: tuple[str, ...]
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


def _require_unique_strings(values: tuple[str, ...], *, field: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field} must be unique")
    if any(not value.strip() or len(value) > 1000 for value in values):
        raise ValueError(f"{field} contain invalid values")
