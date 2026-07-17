"""Public retrieval request, evidence, and authorized-debug DTOs."""

from __future__ import annotations

from typing import Annotated, Any
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


class RetrievalDebugResponse(RetrievalPublicSchema):
    query_plan: RetrievalQueryPlanResponse
    resolved_active_revision_id: UUID
    result_count: int


class EvidencePackResponse(RetrievalPublicSchema):
    knowledge_base_id: UUID
    index_revision_id: UUID
    strategy: RetrievalStrategy
    evidence: tuple[EvidenceResponse, ...]
    debug: RetrievalDebugResponse | None = None

    @classmethod
    def from_domain(cls, value: EvidencePack) -> "EvidencePackResponse":
        return cls.model_validate(value)
