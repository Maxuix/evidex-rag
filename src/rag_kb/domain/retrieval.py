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

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("chunk ordinal must be non-negative")
        if not self.text:
            raise ValueError("evidence text must not be empty")
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
        object.__setattr__(self, "source_location", dict(self.source_location))
        object.__setattr__(self, "hierarchy", dict(self.hierarchy))
        object.__setattr__(self, "source_metadata", dict(self.source_metadata))


@dataclass(frozen=True, slots=True)
class RetrievalDebug:
    query_plan: RetrievalQueryPlan
    resolved_active_revision_id: UUID
    result_count: int

    def __post_init__(self) -> None:
        if self.result_count < 0 or self.result_count > self.query_plan.top_k:
            raise ValueError("debug result_count must be within the plan limit")


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
