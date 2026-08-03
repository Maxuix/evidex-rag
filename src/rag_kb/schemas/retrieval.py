"""Public retrieval request, evidence, and authorized-debug DTOs."""

from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import ConfigDict, Field, field_validator

from rag_kb.domain import (
    EvidencePack,
    EvidenceScoreKind,
    IterativeScanMode,
    RetrievalStrategy,
    RevisionSelector,
)
from rag_kb.schemas.common import PublicSchema


class RetrievalPublicSchema(PublicSchema):
    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)


class RetrievalQueryRequest(RetrievalPublicSchema):
    knowledge_base_id: UUID
    query: Annotated[str, Field(min_length=1)]
    top_k: Annotated[int, Field(ge=1, le=100)] = 10
    strategy: RetrievalStrategy = RetrievalStrategy.EXACT_VECTOR
    rerank: bool = False
    include_debug: bool = False

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must contain non-whitespace characters")
        return normalized


class RetrievalCapabilityResponse(RetrievalPublicSchema):
    mode: Literal["vector", "hybrid"]
    strategy: Literal["exact_vector", "hybrid"]
    profile_version: Literal["exact_vector_v1", "hybrid_fts_rrf_v1"]
    enabled: bool


class RetrievalCapabilitiesResponse(RetrievalPublicSchema):
    default_mode: Literal["vector"]
    modes: tuple[RetrievalCapabilityResponse, ...]


class RetrievalQueryPlanResponse(RetrievalPublicSchema):
    workspace_id: UUID
    knowledge_base_id: UUID
    strategy: RetrievalStrategy
    top_k: int
    revision_selector: RevisionSelector
    current_document_version_only: bool
    build_status: str
    serving_status: str
    distance_metric: str
    candidate_count: int | None
    ef_search: int | None
    iterative_scan: IterativeScanMode
    rerank: bool


class EvidenceResponse(RetrievalPublicSchema):
    rank: int
    index_chunk_id: UUID
    indexed_document_version_id: UUID
    document_id: UUID
    document_version_id: UUID
    index_revision_id: UUID
    ordinal: int
    text: str
    source_location: dict[str, Any]
    hierarchy: dict[str, Any]
    source_metadata: dict[str, Any]
    score: float
    score_kind: EvidenceScoreKind
    vector_similarity: float | None = None
    lexical_score: float = 0.0
    lexical_coverage: float = 0.0
    modality: str = "text"
    asset: "EvidenceAssetResponse | None" = None
    evidence_group_key: str | None = None
    matched_representations: tuple[str, ...] = ("text",)
    text_space_rank: int | None = None
    lexical_rank: int | None = None
    cross_modal_rank: int | None = None
    fusion_score: float | None = None
    related_visuals: tuple["RelatedVisualEvidenceResponse", ...] = ()


class EvidenceAssetResponse(RetrievalPublicSchema):
    id: UUID
    media_type: str
    checksum_sha256: str
    content_url: str
    width: int | None = None
    height: int | None = None


class RelatedVisualEvidenceResponse(RetrievalPublicSchema):
    visual_unit_id: UUID
    asset: EvidenceAssetResponse
    relation_type: str
    relation_confidence_micros: int
    relation_provenance: str
    evidence_group_key: str
    figure_label: str | None = None
    parent_chunk_id: UUID | None = None
    modality: str
    source_location: dict[str, Any]
    text_space_rank: int | None = None
    lexical_rank: int | None = None
    cross_modal_rank: int | None = None


class RetrievalDebugResponse(RetrievalPublicSchema):
    query_plan: RetrievalQueryPlanResponse
    resolved_active_revision_id: UUID
    result_count: int
    text_candidate_count: int | None = None
    lexical_candidate_count: int | None = None
    cross_modal_candidate_count: int | None = None
    lexical_analyzer_version: str | None = None
    lexical_manifest_target_count: int | None = None
    hydrated_relation_count: int | None = None
    evidence_group_count: int | None = None


class EvidencePackResponse(RetrievalPublicSchema):
    knowledge_base_id: UUID
    index_revision_id: UUID
    strategy: RetrievalStrategy
    evidence: tuple[EvidenceResponse, ...]
    debug: RetrievalDebugResponse | None = None

    @classmethod
    def from_domain(cls, value: EvidencePack) -> "EvidencePackResponse":
        return cls.model_validate(value)
