"""Immutable domain contracts for composite multimodal evidence v2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID


class ChunkAssetRelationType(StrEnum):
    EXPLICIT_FIGURE_REFERENCE = "explicit_figure_reference"
    CAPTION_OF = "caption_of"
    INLINE_FIGURE = "inline_figure"
    OCR_OF = "ocr_of"
    TABLE_OF = "table_of"
    SPATIAL_NEIGHBOR = "spatial_neighbor"
    SAME_PAGE = "same_page"

    @property
    def is_strong(self) -> bool:
        return self in {
            self.EXPLICIT_FIGURE_REFERENCE,
            self.CAPTION_OF,
            self.INLINE_FIGURE,
            self.OCR_OF,
            self.TABLE_OF,
        }


class ChunkAssetRelationProvenance(StrEnum):
    AUTHOR_REFERENCE_V2 = "author_reference_v2"
    AUTHOR_CAPTION_V2 = "author_caption_v2"
    OOXML_RELATIONSHIP_V2 = "ooxml_relationship_v2"
    OCR_ASSET_IDENTITY_V2 = "ocr_asset_identity_v2"
    TABLE_IDENTITY_V2 = "table_identity_v2"
    BOUNDED_GEOMETRY_V2 = "bounded_geometry_v2"
    PAGE_IDENTITY_V2 = "page_identity_v2"


class VisualEvidenceReason(StrEnum):
    SELECTED_EXPLICIT_REFERENCE = "selected_explicit_reference"
    SELECTED_STRONG_RELATION = "selected_strong_relation"
    SELECTED_DUAL_LANE = "selected_dual_lane"
    SELECTED_IMAGE_ONLY = "selected_image_only"
    REJECTED_LOW_SIMILARITY = "rejected_low_similarity"
    REJECTED_WEAK_RELATION = "rejected_weak_relation"
    REJECTED_PARENT_NOT_ADMITTED = "rejected_parent_not_admitted"
    REJECTED_DUPLICATE = "rejected_duplicate"
    REJECTED_VISUAL_BUDGET = "rejected_visual_budget"
    REJECTED_ASSET_INTEGRITY = "rejected_asset_integrity"
    REJECTED_UNAUTHORIZED = "rejected_unauthorized"
    REJECTED_UNSUPPORTED_MEDIA = "rejected_unsupported_media"

    @property
    def selected(self) -> bool:
        return self.value.startswith("selected_")


@dataclass(frozen=True, slots=True)
class CompositeChunkDraft:
    unit_key: str
    ordinal: int
    content: str
    embedding_text: str
    embedding_text_hash: str
    token_count: int
    source_location: Mapping[str, Any]
    hierarchy: Mapping[str, Any]
    processing_metadata: Mapping[str, Any]
    evidence_group_key: str

    def __post_init__(self) -> None:
        if not self.unit_key or not self.evidence_group_key:
            raise ValueError("composite chunk identities must not be empty")
        if self.ordinal < 0 or self.token_count < 0:
            raise ValueError("composite chunk counters must be non-negative")
        if not self.content.strip() or not self.embedding_text.strip():
            raise ValueError("composite chunk text must not be empty")
        if len(self.embedding_text_hash) != 64:
            raise ValueError("embedding text hash must be sha256 hex")
        try:
            bytes.fromhex(self.embedding_text_hash)
        except ValueError as error:
            raise ValueError("embedding text hash must be sha256 hex") from error
        object.__setattr__(self, "source_location", _frozen_mapping(self.source_location))
        object.__setattr__(self, "hierarchy", _frozen_mapping(self.hierarchy))
        object.__setattr__(
            self, "processing_metadata", _frozen_mapping(self.processing_metadata)
        )


@dataclass(frozen=True, slots=True)
class ChunkAssetRelationDraft:
    chunk_unit_key: str
    visual_unit_key: str
    asset_key: str
    relation_type: ChunkAssetRelationType
    confidence_micros: int
    ordinal: int
    provenance: ChunkAssetRelationProvenance
    evidence_group_key: str
    figure_label: str | None = None

    def __post_init__(self) -> None:
        for value in (
            self.chunk_unit_key,
            self.visual_unit_key,
            self.asset_key,
            self.evidence_group_key,
        ):
            if not value:
                raise ValueError("relation identities must not be empty")
        if not 0 <= self.confidence_micros <= 1_000_000:
            raise ValueError("relation confidence must be integer micros")
        if isinstance(self.confidence_micros, bool):
            raise ValueError("relation confidence must be integer micros")
        if self.ordinal < 0:
            raise ValueError("relation ordinal must be non-negative")
        object.__setattr__(self, "relation_type", ChunkAssetRelationType(self.relation_type))
        object.__setattr__(
            self, "provenance", ChunkAssetRelationProvenance(self.provenance)
        )
        if self.figure_label is not None:
            normalized = self.figure_label.strip()
            object.__setattr__(self, "figure_label", normalized or None)


@dataclass(frozen=True, slots=True)
class VisualEvidenceDecision:
    visual_unit_id: UUID
    asset_id: UUID
    reason_code: VisualEvidenceReason
    parent_text_citation_ids: tuple[str, ...] = ()
    relation_type: ChunkAssetRelationType | None = None
    text_rank: int | None = None
    cross_modal_rank: int | None = None
    priority_micros: int = 0

    def __post_init__(self) -> None:
        reason = VisualEvidenceReason(self.reason_code)
        object.__setattr__(self, "reason_code", reason)
        if len(self.parent_text_citation_ids) != len(
            set(self.parent_text_citation_ids)
        ):
            raise ValueError("parent citation identifiers must be unique")
        if any(not item for item in self.parent_text_citation_ids):
            raise ValueError("parent citation identifiers must not be empty")
        for rank in (self.text_rank, self.cross_modal_rank):
            if rank is not None and (isinstance(rank, bool) or rank < 1):
                raise ValueError("lane ranks must be positive when provided")
        if isinstance(self.priority_micros, bool) or not isinstance(
            self.priority_micros, int
        ):
            raise ValueError("visual priority must be integer micros")
        if self.relation_type is not None:
            object.__setattr__(
                self, "relation_type", ChunkAssetRelationType(self.relation_type)
            )

    @property
    def selected(self) -> bool:
        return self.reason_code.selected


def quantize_score_micros(value: float) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score must be numeric")
    if not math.isfinite(value):
        raise ValueError("score must be finite")
    return int(round(float(value) * 1_000_000))


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            key: _frozen_mapping(item) if isinstance(item, Mapping) else item
            for key, item in value.items()
        }
    )
