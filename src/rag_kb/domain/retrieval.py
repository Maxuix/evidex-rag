"""Framework-independent retrieval requests, plans, and evidence facts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import UUID

from rag_kb.domain.errors import ErrorCode


class RetrievalStrategy(StrEnum):
    EXACT_VECTOR = "exact_vector"
    ANN_VECTOR = "ann_vector"
    LEXICAL = "lexical"
    HYBRID = "hybrid"


class RevisionSelector(StrEnum):
    ACTIVE = "active"


class IterativeScanMode(StrEnum):
    DISABLED = "disabled"


class EvidenceScoreKind(StrEnum):
    COSINE_SIMILARITY = "cosine_similarity"
    HYBRID_RERANK = "hybrid_rerank"
    RECIPROCAL_RANK_FUSION = "reciprocal_rank_fusion"


@dataclass(frozen=True, slots=True)
class EvidenceAsset:
    id: UUID
    media_type: str
    checksum_sha256: str
    content_url: str
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True, slots=True)
class RelatedVisualEvidence:
    visual_unit_id: UUID
    asset: EvidenceAsset
    relation_type: str
    relation_confidence_micros: int
    relation_provenance: str
    evidence_group_key: str
    figure_label: str | None = None
    parent_chunk_id: UUID | None = None
    modality: str = "image"
    source_location: dict[str, Any] | None = None
    text_space_rank: int | None = None
    cross_modal_rank: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.relation_confidence_micros <= 1_000_000:
            raise ValueError("relation confidence must be integer micros")
        for rank in (self.text_space_rank, self.cross_modal_rank):
            if rank is not None and rank < 1:
                raise ValueError("lane rank must be positive")
        if self.modality not in {"image", "table"}:
            raise ValueError("related visual modality is unsupported")
        object.__setattr__(self, "source_location", dict(self.source_location or {}))


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    knowledge_base_id: UUID
    query: str
    top_k: int = 10
    strategy: RetrievalStrategy = RetrievalStrategy.EXACT_VECTOR
    rerank: bool = False
    include_debug: bool = False

    def __post_init__(self) -> None:
        normalized = self.query.strip()
        if not normalized:
            raise ValueError("retrieval query must not be empty")
        if not 1 <= self.top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        object.__setattr__(self, "query", normalized)


@dataclass(frozen=True, slots=True)
class RetrievalQueryPlan:
    workspace_id: UUID
    knowledge_base_id: UUID
    strategy: RetrievalStrategy
    top_k: int
    revision_selector: RevisionSelector = RevisionSelector.ACTIVE
    current_document_version_only: bool = True
    build_status: str = "ready"
    serving_status: str = "serving"
    distance_metric: str = "cosine"
    candidate_count: int | None = None
    ef_search: int | None = None
    iterative_scan: IterativeScanMode = IterativeScanMode.DISABLED
    rerank: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        if self.revision_selector is not RevisionSelector.ACTIVE:
            raise ValueError("the active revision selector is mandatory")
        if not self.current_document_version_only:
            raise ValueError("the current retrievable version filter is mandatory")
        if self.build_status != "ready" or self.serving_status != "serving":
            raise ValueError("ready + serving filters are mandatory")
        if self.strategy is RetrievalStrategy.EXACT_VECTOR:
            if self.distance_metric != "cosine":
                raise ValueError("exact-vector retrieval uses cosine distance")
            if self.ef_search is not None:
                raise ValueError("exact-vector retrieval has no ANN parameters")
            if self.iterative_scan is not IterativeScanMode.DISABLED:
                raise ValueError("exact-vector retrieval has no iterative scan")
            if self.rerank:
                candidate_count = self.candidate_count
                if candidate_count is None:
                    candidate_count = max(self.top_k, min(self.top_k * 4, 40))
                    object.__setattr__(self, "candidate_count", candidate_count)
                if not self.top_k <= candidate_count <= 100:
                    raise ValueError(
                        "rerank candidate_count must be between top_k and 100"
                    )
            elif self.candidate_count is not None:
                raise ValueError("candidate_count requires reranking")


@dataclass(frozen=True, slots=True)
class VectorSearchHit:
    workspace_id: UUID
    knowledge_base_id: UUID
    index_revision_id: UUID
    index_chunk_id: UUID
    indexed_document_version_id: UUID
    document_id: UUID
    document_version_id: UUID
    ordinal: int
    text: str
    source_location: dict[str, Any]
    hierarchy: dict[str, Any]
    source_metadata: dict[str, Any]
    cosine_distance: float
    build_status: str
    serving_status: str
    is_current_serving_version: bool
    modality: str = "text"
    evidence_group_key: str | None = None
    representation_kind: str = "text"
    index_asset_id: UUID | None = None
    asset_media_type: str | None = None
    asset_checksum_sha256: str | None = None
    asset_width: int | None = None
    asset_height: int | None = None

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("chunk ordinal must be non-negative")
        if not self.text and self.modality == "text":
            raise ValueError("text evidence must not be empty")
        if self.modality not in {"text", "image", "table"}:
            raise ValueError("unsupported evidence modality")
        if (
            isinstance(self.cosine_distance, bool)
            or not isinstance(self.cosine_distance, (int, float))
            or not math.isfinite(self.cosine_distance)
            or not 0.0 <= float(self.cosine_distance) <= 2.0
        ):
            raise ValueError("cosine distance must be finite and between 0 and 2")
        object.__setattr__(self, "cosine_distance", float(self.cosine_distance))
        object.__setattr__(self, "source_location", dict(self.source_location))
        object.__setattr__(self, "hierarchy", dict(self.hierarchy))
        object.__setattr__(self, "source_metadata", dict(self.source_metadata))


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    resolved_active_revision_id: UUID
    hits: tuple[VectorSearchHit, ...] = ()
    embedding_space_id: UUID | None = None
    compatibility_fingerprint: str | None = None
    space_role: str = "text_retrieval"


@dataclass(frozen=True, slots=True)
class Evidence:
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
    score_kind: EvidenceScoreKind = EvidenceScoreKind.COSINE_SIMILARITY
    vector_similarity: float | None = None
    lexical_score: float = 0.0
    lexical_coverage: float = 0.0
    modality: str = "text"
    asset: EvidenceAsset | None = None
    evidence_group_key: str | None = None
    matched_representations: tuple[str, ...] = ("text",)
    text_space_rank: int | None = None
    cross_modal_rank: int | None = None
    fusion_score: float | None = None
    related_visuals: tuple[RelatedVisualEvidence, ...] = ()

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("evidence rank must be positive")
        if self.ordinal < 0:
            raise ValueError("chunk ordinal must be non-negative")
        if not math.isfinite(self.score):
            raise ValueError("evidence score must be finite")
        try:
            object.__setattr__(self, "score_kind", EvidenceScoreKind(self.score_kind))
        except ValueError as error:
            raise ValueError("unsupported evidence score kind") from error
        for name, value in (
            ("vector_similarity", self.vector_similarity),
            ("lexical_score", self.lexical_score),
            ("lexical_coverage", self.lexical_coverage),
        ):
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite when provided")
        if self.lexical_score < 0.0 or self.lexical_coverage < 0.0:
            raise ValueError("lexical scores must not be negative")
        if self.vector_similarity is not None and not -1.0 <= self.vector_similarity <= 1.0:
            raise ValueError("vector similarity must be between -1 and 1")
        if self.lexical_score > 1.0 or self.lexical_coverage > 1.0:
            raise ValueError("lexical scores must be at most one")
        if self.modality not in {"text", "image", "table"}:
            raise ValueError("unsupported evidence modality")
        if not self.text and self.modality == "text":
            raise ValueError("text evidence must not be empty")
        object.__setattr__(self, "source_location", dict(self.source_location))
        object.__setattr__(self, "hierarchy", dict(self.hierarchy))
        object.__setattr__(self, "source_metadata", dict(self.source_metadata))
        asset_ids = [item.asset.id for item in self.related_visuals]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("related visual assets must be unique")


@dataclass(frozen=True, slots=True)
class RetrievalDebug:
    query_plan: RetrievalQueryPlan
    resolved_active_revision_id: UUID
    result_count: int
    text_candidate_count: int | None = None
    cross_modal_candidate_count: int | None = None
    hydrated_relation_count: int | None = None
    evidence_group_count: int | None = None

    def __post_init__(self) -> None:
        if self.result_count < 0 or self.result_count > self.query_plan.top_k:
            raise ValueError("debug result_count must be within the plan limit")
        for value in (
            self.text_candidate_count,
            self.cross_modal_candidate_count,
            self.hydrated_relation_count,
            self.evidence_group_count,
        ):
            if value is not None and value < 0:
                raise ValueError("debug counts must be non-negative")


@dataclass(frozen=True, slots=True)
class EvidencePack:
    knowledge_base_id: UUID
    index_revision_id: UUID
    strategy: RetrievalStrategy
    evidence: tuple[Evidence, ...] = ()
    debug: RetrievalDebug | None = None

    def __post_init__(self) -> None:
        chunk_ids: set[UUID] = set()
        for expected_rank, item in enumerate(self.evidence, start=1):
            if item.rank != expected_rank:
                raise ValueError("evidence ranks must be contiguous and ordered")
            if item.index_revision_id != self.index_revision_id:
                raise ValueError("all evidence must belong to one index revision")
            if item.index_chunk_id in chunk_ids:
                raise ValueError("evidence chunk identifiers must be unique")
            chunk_ids.add(item.index_chunk_id)
        if self.debug is not None:
            if self.debug.resolved_active_revision_id != self.index_revision_id:
                raise ValueError("debug and evidence revision identifiers must match")
            if self.debug.result_count != len(self.evidence):
                raise ValueError("debug result_count must match evidence cardinality")


class RetrievalExecutionError(RuntimeError):
    """Stable, content-safe retrieval failure at a capability boundary."""

    def __init__(
        self,
        code: ErrorCode,
        *,
        diagnostic: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code.value)
        self.code = code
        self.diagnostic = dict(diagnostic or {})
