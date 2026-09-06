"""Public retrieval request, evidence, and authorized-debug DTOs."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self
from uuid import UUID

from pydantic import ConfigDict, Field, field_validator, model_validator

from rag_kb.domain import (
    EvidencePack,
    EvidenceScoreKind,
    RerankMode,
    RetrievalStrategy,
)
from rag_kb.schemas.common import PublicSchema


class RetrievalPublicSchema(PublicSchema):
    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)


class RetrievalQueryRequest(RetrievalPublicSchema):
    knowledge_base_id: UUID
    query: Annotated[str, Field(min_length=1)]
    top_k: Annotated[int, Field(ge=1, le=100)] = 10
    strategy: RetrievalStrategy = RetrievalStrategy.EXACT_VECTOR
    rerank_mode: RerankMode = RerankMode.NONE
    include_debug: bool = False
    mode: Literal["graph"] | None = None

    @model_validator(mode="after")
    def require_supported_rerank_combination(self) -> Self:
        if self.mode == "graph":
            if not 4 <= self.top_k <= 20:
                raise ValueError("graph retrieval top_k must be between 4 and 20")
            if self.rerank_mode not in {
                RerankMode.CLASSIC,
                RerankMode.LOCAL_MINILM_V1,
            }:
                raise ValueError("graph retrieval requires an enabled reranker")
            return self
        if (
            self.strategy is RetrievalStrategy.HYBRID
            and self.rerank_mode is RerankMode.NONE
        ):
            raise ValueError("hybrid retrieval requires reranking")
        if (
            self.strategy is RetrievalStrategy.ITERATIVE_BALANCED
            and self.rerank_mode is RerankMode.NONE
        ):
            raise ValueError("iterative balanced retrieval requires reranking")
        if (
            self.rerank_mode is RerankMode.LOCAL_MINILM_V1
            and self.top_k > 20
        ):
            raise ValueError("local reranking supports top_k up to 20")
        return self

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must contain non-whitespace characters")
        return normalized


class RetrievalQueryPlanResponse(RetrievalPublicSchema):
    workspace_id: UUID
    knowledge_base_id: UUID
    strategy: RetrievalStrategy
    top_k: int
    distance_metric: str
    candidate_count: int | None
    rerank_mode: RerankMode


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
    model_rerank_score: float | None = None
    model_rerank_rank: int | None = None
    model_rerank_window_count: int | None = None
    model_rerank_winning_window_index: int | None = None
    related_visuals: tuple["RelatedVisualEvidenceResponse", ...] = ()
    graph_path_id: str | None = None
    graph_anchor_index_chunk_id: UUID | None = None
    graph_hop_count: Literal[1, 2, 3] | None = None
    graph_path_rank: int | None = None


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
    model_rerank_candidate_count: int | None = None
    model_rerank_window_count: int | None = None
    graph: "GraphDebugResponse | None" = None
    matched_questions: tuple["RetrievalMatchedQuestionResponse", ...] = ()


class RetrievalMatchedQuestionResponse(RetrievalPublicSchema):
    index_chunk_id: UUID
    ordinal: int
    question: str


class GraphPathDebugResponse(RetrievalPublicSchema):
    path_id: str
    entry_entity_key: str
    hop_count: Literal[1, 2, 3]
    seed_entry: bool
    anchor_chunk_id: UUID
    rank: int
    support_counts: tuple[int, ...]
    source_chunk_ids: tuple[UUID, ...]


class GraphBundleDebugResponse(RetrievalPublicSchema):
    path_id: str
    chunk_ids: tuple[UUID, ...]


class GraphDebugResponse(RetrievalPublicSchema):
    dense_seed_count: int
    lexical_seed_count: int
    fused_seed_count: int
    query_entity_count: int
    one_hop_path_count: int
    two_hop_path_count: int
    three_hop_path_count: int
    rejected_path_count: int
    bundle_count: int
    protocol_skipped_count: int
    resource_skipped_count: int
    paths: tuple[GraphPathDebugResponse, ...] = ()
    bundles: tuple[GraphBundleDebugResponse, ...] = ()


class EvidencePackResponse(RetrievalPublicSchema):
    knowledge_base_id: UUID
    index_revision_id: UUID
    strategy: RetrievalStrategy
    evidence: tuple[EvidenceResponse, ...]
    debug: RetrievalDebugResponse | None = None

    @classmethod
    def from_domain(cls, value: EvidencePack) -> "EvidencePackResponse":
        return cls.model_validate(value)
