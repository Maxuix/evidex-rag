"""Framework-independent retrieval requests, plans, and evidence facts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import UUID

from rag_kb.domain.errors import ErrorCode


EvidenceGroupIdentity = tuple[UUID, str]


def evidence_group_identity(
    indexed_document_version_id: UUID,
    group_key: str,
) -> EvidenceGroupIdentity:
    """Scope a raw evidence-group key to one indexed document target."""

    return indexed_document_version_id, group_key


class RetrievalStrategy(StrEnum):
    EXACT_VECTOR = "exact_vector"
    HYBRID = "hybrid"


class RerankMode(StrEnum):
    NONE = "none"
    CLASSIC = "classic"
    LOCAL_MINILM_V1 = "local_minilm_v1"


class EvidenceScoreKind(StrEnum):
    COSINE_SIMILARITY = "cosine_similarity"
    HYBRID_RERANK = "hybrid_rerank"
    RECIPROCAL_RANK_FUSION = "reciprocal_rank_fusion"
    ADJACENCY = "adjacency"
    GRAPH_PATH = "graph_path"
    LEXICAL = "lexical"


SERVING_DOCUMENT_LIST_LIMIT = 50
SERVING_DOCUMENT_OUTLINE_LIMIT = 8


@dataclass(frozen=True, slots=True)
class ServingScopeQuery:
    workspace_id: UUID
    knowledge_base_id: UUID
    after_document_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class ServingDocumentEntry:
    document_id: UUID
    document_version_id: UUID
    indexed_document_version_id: UUID
    display_name: str
    original_filename: str
    version_number: int
    chunk_count: int
    outline: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        display_name = self.display_name.strip()
        original_filename = self.original_filename.strip()
        if not display_name:
            raise ValueError("serving document display_name must not be empty")
        if not original_filename:
            raise ValueError("serving document original_filename must not be empty")
        if self.version_number < 1:
            raise ValueError("serving document version_number must be at least 1")
        if self.chunk_count < 0:
            raise ValueError("serving document chunk_count must be non-negative")
        outline: list[str] = []
        seen: set[str] = set()
        for item in self.outline:
            title = item.strip()
            if not 1 <= len(title) <= 256:
                raise ValueError("serving document outline title is invalid")
            if title in seen:
                raise ValueError("serving document outline titles must be unique")
            seen.add(title)
            outline.append(title)
        if len(outline) > SERVING_DOCUMENT_OUTLINE_LIMIT:
            raise ValueError("serving document outline exceeds the bound")
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "original_filename", original_filename)
        object.__setattr__(self, "outline", tuple(outline))


@dataclass(frozen=True, slots=True)
class ServingDocumentList:
    resolved_active_revision_id: UUID
    entries: tuple[ServingDocumentEntry, ...] = ()
    truncated: bool = False
    next_document_id: UUID | None = None

    def __post_init__(self) -> None:
        if len(self.entries) > SERVING_DOCUMENT_LIST_LIMIT:
            raise ValueError("serving document list exceeds the bound")
        document_ids = [item.document_id for item in self.entries]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("serving document list entries must be unique")


@dataclass(frozen=True, slots=True)
class LexicalManifestStatus:
    resolved_active_revision_id: UUID
    serving_target_count: int
    manifested_target_count: int

    def __post_init__(self) -> None:
        if self.serving_target_count < 0 or self.manifested_target_count < 0:
            raise ValueError("lexical manifest counts must be non-negative")
        if self.manifested_target_count > self.serving_target_count:
            raise ValueError("manifested targets cannot exceed serving targets")

    @property
    def complete(self) -> bool:
        return (
            self.serving_target_count > 0
            and self.manifested_target_count == self.serving_target_count
        )


@dataclass(frozen=True, slots=True)
class RerankDocument:
    index_chunk_id: UUID
    text: str
    hierarchy: dict[str, Any]
    modality: str = "text"
    document_context: str = ""

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("rerank document text must not be empty")
        if self.modality not in {"text", "table"}:
            raise ValueError("rerank document modality is unsupported")
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "hierarchy", dict(self.hierarchy))
        object.__setattr__(self, "document_context", self.document_context.strip())


@dataclass(frozen=True, slots=True)
class ModelRerankScore:
    index_chunk_id: UUID
    score: float
    raw_logit: float
    window_count: int
    winning_window_index: int

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.score)
            or not 0.0 <= self.score <= 1.0
            or not math.isfinite(self.raw_logit)
        ):
            raise ValueError("model rerank scores must be finite")
        if self.window_count < 1:
            raise ValueError("model rerank window count must be positive")
        if not 0 <= self.winning_window_index < self.window_count:
            raise ValueError("winning rerank window index is invalid")


@dataclass(frozen=True, slots=True)
class AdjacentChunkAnchor:
    index_chunk_id: UUID
    indexed_document_version_id: UUID
    ordinal: int

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("adjacency anchor ordinal must be non-negative")


@dataclass(frozen=True, slots=True)
class AdjacentChunkQuery:
    workspace_id: UUID
    knowledge_base_id: UUID
    index_revision_id: UUID
    anchors: tuple[AdjacentChunkAnchor, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.anchors) <= 2:
            raise ValueError("adjacency query requires one or two anchors")
        chunk_ids = [item.index_chunk_id for item in self.anchors]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("adjacency anchors must be unique")


@dataclass(frozen=True, slots=True)
class AdjacentChunkHit:
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
    anchor_index_chunk_id: UUID
    anchor_rank: int
    offset: int
    build_status: str
    serving_status: str
    is_current_serving_version: bool
    modality: str = "text"
    evidence_group_key: str | None = None
    index_asset_id: UUID | None = None
    asset_media_type: str | None = None
    asset_checksum_sha256: str | None = None
    asset_width: int | None = None
    asset_height: int | None = None
    document_display_name: str | None = None
    document_original_filename: str | None = None

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("adjacent chunk ordinal must be non-negative")
        if self.anchor_rank < 1:
            raise ValueError("adjacency anchor rank must be positive")
        if self.offset not in {-1, 1}:
            raise ValueError("adjacency offset must be minus or plus one")
        if self.modality not in {"text", "table"}:
            raise ValueError("adjacent chunk modality is unsupported")
        if not self.text and self.modality == "text":
            raise ValueError("adjacent text evidence must not be empty")
        object.__setattr__(self, "source_location", dict(self.source_location))
        object.__setattr__(self, "hierarchy", dict(self.hierarchy))
        object.__setattr__(self, "source_metadata", dict(self.source_metadata))


@dataclass(frozen=True, slots=True)
class AdjacentChunkResult:
    resolved_active_revision_id: UUID
    validated_anchor_count: int
    hits: tuple[AdjacentChunkHit, ...] = ()

    def __post_init__(self) -> None:
        if not 0 <= self.validated_anchor_count <= 2:
            raise ValueError("validated adjacency anchor count is invalid")
        if len(self.hits) > 4:
            raise ValueError("adjacency result exceeds the fixed neighbor bound")


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
    lexical_rank: int | None = None
    cross_modal_rank: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.relation_confidence_micros <= 1_000_000:
            raise ValueError("relation confidence must be integer micros")
        for rank in (self.text_space_rank, self.lexical_rank, self.cross_modal_rank):
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
    rerank_mode: RerankMode = RerankMode.NONE
    include_debug: bool = False

    def __post_init__(self) -> None:
        normalized = self.query.strip()
        if not normalized:
            raise ValueError("retrieval query must not be empty")
        if not 1 <= self.top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        try:
            object.__setattr__(
                self,
                "rerank_mode",
                RerankMode(self.rerank_mode),
            )
        except ValueError as error:
            raise ValueError("unsupported rerank mode") from error
        if (
            self.rerank_mode is RerankMode.LOCAL_MINILM_V1
            and self.top_k > 20
        ):
            raise ValueError("local reranking supports top_k up to 20")
        object.__setattr__(self, "query", normalized)

    @property
    def rerank(self) -> bool:
        return self.rerank_mode is not RerankMode.NONE


@dataclass(frozen=True, slots=True)
class GraphRetrievalRequest:
    knowledge_base_id: UUID
    query: str
    top_k: int = 10
    rerank_mode: RerankMode = RerankMode.CLASSIC
    include_debug: bool = False

    def __post_init__(self) -> None:
        normalized = self.query.strip()
        if not normalized:
            raise ValueError("graph retrieval query must not be empty")
        if not 4 <= self.top_k <= 20:
            raise ValueError("graph retrieval top_k must be between 4 and 20")
        try:
            object.__setattr__(self, "rerank_mode", RerankMode(self.rerank_mode))
        except ValueError as error:
            raise ValueError("unsupported graph rerank mode") from error
        object.__setattr__(self, "query", normalized)


@dataclass(frozen=True, slots=True)
class RetrievalQueryPlan:
    workspace_id: UUID
    knowledge_base_id: UUID
    strategy: RetrievalStrategy
    top_k: int
    distance_metric: str = "cosine"
    candidate_count: int | None = None
    rerank_mode: RerankMode = RerankMode.NONE
    auto_qa_candidate_count: int = 20
    allow_unverified_auto_qa: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.allow_unverified_auto_qa, bool):
            raise ValueError("unverified auto-qa opt-in must be boolean")
        if (
            not isinstance(self.auto_qa_candidate_count, int)
            or isinstance(self.auto_qa_candidate_count, bool)
            or not 0 <= self.auto_qa_candidate_count <= 20
        ):
            raise ValueError("auto-qa candidate count must be between zero and twenty")
        if not 1 <= self.top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        try:
            object.__setattr__(
                self,
                "rerank_mode",
                RerankMode(self.rerank_mode),
            )
        except ValueError as error:
            raise ValueError("unsupported rerank mode") from error
        if (
            self.rerank_mode is RerankMode.LOCAL_MINILM_V1
            and self.top_k > 20
        ):
            raise ValueError("local reranking supports top_k up to 20")
        if self.strategy is RetrievalStrategy.EXACT_VECTOR:
            if self.distance_metric != "cosine":
                raise ValueError("exact-vector retrieval uses cosine distance")
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
        elif self.strategy is RetrievalStrategy.HYBRID:
            if self.distance_metric != "cosine":
                raise ValueError("hybrid dense companion uses cosine distance")
            if (
                not self.rerank
                or self.candidate_count is None
                or not self.top_k <= self.candidate_count <= 100
            ):
                raise ValueError(
                    "hybrid retrieval requires a bounded scored candidate set"
                )

    @property
    def rerank(self) -> bool:
        return self.rerank_mode is not RerankMode.NONE


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
    document_display_name: str | None = None
    document_original_filename: str | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    matched_question: str | None = None
    matched_question_ordinal: int | None = None
    question_cosine_distance: float | None = None
    source_candidate: bool = True

    def representation_labels(self) -> tuple[str, ...]:
        if self.matched_question and self.representation_kind != "auto_qa_question":
            return (self.representation_kind, "auto_qa_question")
        if self.representation_kind != "auto_qa_question":
            return (self.representation_kind,)
        body = {"table": "table_text", "image": "caption_text"}.get(
            self.modality, "text"
        )
        return ("auto_qa_question", body)

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
        if self.question_cosine_distance is not None and (
            isinstance(self.question_cosine_distance, bool)
            or not isinstance(self.question_cosine_distance, (int, float))
            or not math.isfinite(self.question_cosine_distance)
            or not 0.0 <= self.question_cosine_distance <= 2.0
        ):
            raise ValueError("question cosine distance must be finite and between 0 and 2")
        if not self.source_candidate and not self.matched_question:
            raise ValueError("supplementary candidates require question provenance")
        object.__setattr__(self, "source_location", dict(self.source_location))
        object.__setattr__(self, "hierarchy", dict(self.hierarchy))
        object.__setattr__(self, "source_metadata", dict(self.source_metadata))
        if self.lexical_rank is not None and self.lexical_rank < 1:
            raise ValueError("lexical rank must be positive")
        if self.lexical_score is not None and (
            isinstance(self.lexical_score, bool)
            or not isinstance(self.lexical_score, (int, float))
            or not math.isfinite(self.lexical_score)
            or self.lexical_score < 0.0
        ):
            raise ValueError("lexical FTS rank must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    resolved_active_revision_id: UUID
    hits: tuple[VectorSearchHit, ...] = ()
    embedding_space_id: UUID | None = None
    compatibility_fingerprint: str | None = None
    space_role: str = "text_retrieval"


@dataclass(frozen=True, slots=True)
class LexicalSearchResult:
    resolved_active_revision_id: UUID
    analyzer_version: str
    manifest_target_count: int
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
    modality: str = "text"
    asset: EvidenceAsset | None = None
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
    related_visuals: tuple[RelatedVisualEvidence, ...] = ()
    document_display_name: str | None = None
    document_original_filename: str | None = None
    adjacency_anchor_index_chunk_id: UUID | None = None
    adjacency_offset: int | None = None
    graph_path_id: str | None = None
    graph_anchor_index_chunk_id: UUID | None = None
    graph_hop_count: int | None = None
    graph_path_rank: int | None = None

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
        if self.score_kind is EvidenceScoreKind.ADJACENCY:
            if (
                self.score != 0.0
                or self.lexical_score != 0.0
                or self.lexical_coverage != 0.0
                or self.modality not in {"text", "table"}
                or self.adjacency_anchor_index_chunk_id is None
                or self.adjacency_offset not in {-1, 1}
                or self.vector_similarity is not None
                or self.text_space_rank is not None
                or self.lexical_rank is not None
                or self.cross_modal_rank is not None
                or self.fusion_score is not None
                or self.model_rerank_score is not None
                or self.model_rerank_rank is not None
                or self.model_rerank_window_count is not None
                or self.model_rerank_winning_window_index is not None
            ):
                raise ValueError("adjacency evidence metadata is invalid")
        elif self.score_kind is EvidenceScoreKind.GRAPH_PATH:
            if (
                self.score <= 0.0
                or self.vector_similarity is not None
                or self.lexical_score != 0.0
                or self.lexical_coverage != 0.0
                or self.text_space_rank is not None
                or self.lexical_rank is not None
                or self.cross_modal_rank is not None
                or self.fusion_score is not None
                or any(value is not None for value in (
                    self.model_rerank_score,
                    self.model_rerank_rank,
                    self.model_rerank_window_count,
                    self.model_rerank_winning_window_index,
                ))
                or self.graph_path_id is None
                or self.graph_anchor_index_chunk_id is None
                or self.graph_hop_count not in {1, 2, 3}
                or self.graph_path_rank is None
                or self.graph_path_rank < 1
                or not math.isclose(
                    self.score, 1.0 / self.graph_path_rank, rel_tol=1e-9
                )
            ):
                raise ValueError("graph path evidence metadata is invalid")
        elif self.score_kind is EvidenceScoreKind.LEXICAL:
            if (
                self.lexical_rank is None
                or self.lexical_rank < 1
                or self.vector_similarity is not None
                or self.lexical_score != 0.0
                or self.lexical_coverage != 0.0
                or any(
                    value is not None
                    for value in (
                        self.model_rerank_score,
                        self.model_rerank_rank,
                        self.model_rerank_window_count,
                        self.model_rerank_winning_window_index,
                    )
                )
                or not math.isclose(
                    self.score, 1.0 / self.lexical_rank, rel_tol=1e-9
                )
            ):
                raise ValueError("lexical evidence metadata is invalid")
        if self.score_kind is not EvidenceScoreKind.ADJACENCY and (
            self.adjacency_anchor_index_chunk_id is not None
            or self.adjacency_offset is not None
        ):
            raise ValueError("non-adjacency evidence cannot reference an anchor")
        if self.score_kind is not EvidenceScoreKind.GRAPH_PATH and any(
            value is not None
            for value in (
                self.graph_path_id,
                self.graph_anchor_index_chunk_id,
                self.graph_hop_count,
                self.graph_path_rank,
            )
        ):
            raise ValueError("non-graph evidence cannot reference a graph path")
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
        model_values = (
            self.model_rerank_score,
            self.model_rerank_rank,
            self.model_rerank_window_count,
            self.model_rerank_winning_window_index,
        )
        if any(value is not None for value in model_values):
            if any(value is None for value in model_values):
                raise ValueError("model rerank evidence metadata is incomplete")
            assert self.model_rerank_score is not None
            assert self.model_rerank_rank is not None
            assert self.model_rerank_window_count is not None
            assert self.model_rerank_winning_window_index is not None
            if (
                not math.isfinite(self.model_rerank_score)
                or not 0.0 <= self.model_rerank_score <= 1.0
                or self.model_rerank_rank < 1
                or self.model_rerank_window_count < 1
                or not 0
                <= self.model_rerank_winning_window_index
                < self.model_rerank_window_count
                or self.modality not in {"text", "table"}
            ):
                raise ValueError("model rerank evidence metadata is invalid")
        if not self.text and self.modality == "text":
            raise ValueError("text evidence must not be empty")
        object.__setattr__(self, "source_location", dict(self.source_location))
        object.__setattr__(self, "hierarchy", dict(self.hierarchy))
        object.__setattr__(self, "source_metadata", dict(self.source_metadata))
        asset_ids = [item.asset.id for item in self.related_visuals]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("related visual assets must be unique")


@dataclass(frozen=True, slots=True)
class RetrievalMatchedQuestion:
    index_chunk_id: UUID
    ordinal: int
    question: str

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("matched question ordinal must be non-negative")
        if not self.question.strip():
            raise ValueError("matched question must not be empty")
        if len(self.question) > 200:
            object.__setattr__(self, "question", self.question[:200])


@dataclass(frozen=True, slots=True)
class RetrievalDebug:
    query_plan: RetrievalQueryPlan
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
    source_context_candidate_count: int = 0
    graph: Any | None = None
    matched_questions: tuple[RetrievalMatchedQuestion, ...] = ()

    def __post_init__(self) -> None:
        if self.result_count < 0 or self.result_count > self.query_plan.top_k:
            raise ValueError("debug result_count must be within the plan limit")
        for value in (
            self.text_candidate_count,
            self.lexical_candidate_count,
            self.cross_modal_candidate_count,
            self.lexical_manifest_target_count,
            self.hydrated_relation_count,
            self.evidence_group_count,
            self.model_rerank_candidate_count,
            self.model_rerank_window_count,
            self.source_context_candidate_count,
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


GRAPH_SEARCH_RESULT_CODES = frozenset(
    {
        "admitted",
        "no_evidence",
        "not_ready",
        "timeout",
        "unavailable",
        "rejected",
    }
)
# One Graph tool call may return at most this many source chunks.
GRAPH_SEARCH_SOURCE_CHUNK_LIMIT = 16


@dataclass(frozen=True, slots=True)
class GraphSearchResult:
    """Bounded, source-only evidence returned by one first-class Graph search."""

    route_result_code: str
    evidence: tuple[Evidence, ...] = ()
    new_index_chunk_ids: tuple[UUID, ...] | None = None
    candidate_count: int | None = None
    path_count: int | None = None
    hydrated_chunk_count: int | None = None
    hop1_count: int | None = None
    hop2_count: int | None = None
    hop3_count: int | None = None

    def __post_init__(self) -> None:
        if self.route_result_code not in GRAPH_SEARCH_RESULT_CODES:
            raise ValueError("Graph search result code is invalid")
        if len(self.evidence) > GRAPH_SEARCH_SOURCE_CHUNK_LIMIT:
            raise ValueError("Graph search evidence is unbounded")
        chunk_ids: set[UUID] = set()
        for expected_rank, item in enumerate(self.evidence, start=1):
            if item.rank != expected_rank:
                raise ValueError("Graph search evidence ranks are invalid")
            if item.score_kind is not EvidenceScoreKind.GRAPH_PATH:
                raise ValueError("Graph search evidence must be graph grounded")
            if item.index_chunk_id in chunk_ids:
                raise ValueError("Graph search evidence chunks must be unique")
            chunk_ids.add(item.index_chunk_id)
        new_ids = (
            tuple(item.index_chunk_id for item in self.evidence)
            if self.new_index_chunk_ids is None
            else tuple(dict.fromkeys(self.new_index_chunk_ids))
        )
        if any(item not in chunk_ids for item in new_ids):
            raise ValueError("Graph search new evidence is not in its paths")
        object.__setattr__(self, "new_index_chunk_ids", new_ids)
        hop_counts = (self.hop1_count, self.hop2_count, self.hop3_count)
        if any(value is not None for value in hop_counts) and not all(
            value is not None for value in hop_counts
        ):
            raise ValueError("Graph search hop counts must be reported together")
        if all(value is not None for value in hop_counts):
            if sum(hop_counts) != len(self.evidence):  # type: ignore[arg-type]
                raise ValueError("Graph search hop counts must match evidence")
        if self.route_result_code == "admitted":
            if not self.evidence or not new_ids:
                raise ValueError("admitted Graph search requires new evidence")
        elif self.route_result_code != "no_evidence":
            if self.evidence or new_ids:
                raise ValueError("unavailable Graph search cannot carry evidence")

    @property
    def new_evidence_count(self) -> int:
        return len(self.new_index_chunk_ids or ())


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
