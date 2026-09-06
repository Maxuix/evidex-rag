"""Small persisted retrieval presets and current runtime execution settings."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping

from rag_kb.document_processing.lexical import (
    LEXICAL_ANALYZER_VERSION,
    LEXICAL_QUERY_VERSION,
)
from rag_kb.domain import (
    GRAPH_AUGMENTATION_VERSION,
    GRAPH_RETRIEVAL_PROFILE_VERSION,
    RerankMode,
    RetrievalStrategy,
)


EXACT_PROFILE_VERSION = "exact_vector_v2"
HYBRID_PROFILE_VERSION = "hybrid_fts_rrf_v2"
ITERATIVE_BALANCED_PROFILE_VERSION = "iterative_balanced_v1"
ADAPTIVE_GRAPHITI_PROFILE_VERSION = "adaptive_graphiti_v3"
ADAPTIVE_GRAPHITI_ROUTER_VERSION = "native_agent_graph_tool_v1"

# Frozen Graph Tool parameters added to the v3 adaptive Chat profile.
GRAPH_EDGE_LIMIT_DEFAULT = 16
GRAPH_SOURCE_CHUNK_TARGET_DEFAULT = 12
GRAPH_SOURCE_CHUNK_LIMIT_DEFAULT = 16
GRAPH_CALL_TIMEOUT_SECONDS_DEFAULT = 90

RetrievalExecutionType = Literal[
    "simple", "iterative_balanced", "manual_graph", "adaptive_graphiti"
]


@dataclass(frozen=True, slots=True)
class GraphRetrievalProfile:
    """Small outer profile; dense/lexical execution remains the hybrid profile."""

    profile_version: str
    strategy: RetrievalStrategy
    top_k: int
    rerank_mode: RerankMode
    augmentation: str

    def __post_init__(self) -> None:
        if self.profile_version != GRAPH_RETRIEVAL_PROFILE_VERSION:
            raise ValueError("unsupported graph retrieval profile version")
        if self.strategy is not RetrievalStrategy.HYBRID:
            raise ValueError("graph retrieval uses hybrid seeds")
        if not 4 <= self.top_k <= 20:
            raise ValueError("graph retrieval top_k is invalid")
        if self.rerank_mode not in {
            RerankMode.CLASSIC,
            RerankMode.LOCAL_MINILM_V1,
        }:
            raise ValueError("graph retrieval requires an enabled reranker")
        if self.augmentation != GRAPH_AUGMENTATION_VERSION:
            raise ValueError("graph augmentation version is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_version": self.profile_version,
            "strategy": self.strategy.value,
            "top_k": self.top_k,
            "rerank_mode": self.rerank_mode.value,
            "augmentation": self.augmentation,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveGraphitiRetrievalProfile:
    """Chat-only profile with one first-class bounded Graph Relations tool."""

    profile_version: str
    strategy: RetrievalStrategy
    top_k: int
    rerank_mode: RerankMode
    router: str
    augmentation: str
    graph_edge_limit: int = GRAPH_EDGE_LIMIT_DEFAULT
    graph_source_chunk_target: int = GRAPH_SOURCE_CHUNK_TARGET_DEFAULT
    graph_source_chunk_limit: int = GRAPH_SOURCE_CHUNK_LIMIT_DEFAULT
    graph_call_timeout_seconds: int = GRAPH_CALL_TIMEOUT_SECONDS_DEFAULT

    def __post_init__(self) -> None:
        if self.profile_version != ADAPTIVE_GRAPHITI_PROFILE_VERSION:
            raise ValueError("unsupported adaptive Graphiti profile version")
        if self.strategy is not RetrievalStrategy.EXACT_VECTOR:
            raise ValueError("adaptive Graphiti uses exact vector Simple retrieval")
        if not 1 <= self.top_k <= 100:
            raise ValueError("adaptive Graphiti top_k is invalid")
        if self.rerank_mode is RerankMode.LOCAL_MINILM_V1 and self.top_k > 20:
            raise ValueError("local reranking supports top_k up to 20")
        if self.router != ADAPTIVE_GRAPHITI_ROUTER_VERSION:
            raise ValueError("adaptive Graphiti router version is invalid")
        if self.augmentation != GRAPH_AUGMENTATION_VERSION:
            raise ValueError("adaptive Graphiti augmentation version is invalid")
        for name, value in (
            ("graph_edge_limit", self.graph_edge_limit),
            ("graph_source_chunk_target", self.graph_source_chunk_target),
            ("graph_source_chunk_limit", self.graph_source_chunk_limit),
            ("graph_call_timeout_seconds", self.graph_call_timeout_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("adaptive Graphiti Graph parameter is invalid")
        if not 8 <= self.graph_edge_limit <= 32:
            raise ValueError("adaptive Graphiti edge limit is invalid")
        if not (
            1
            <= self.graph_source_chunk_target
            <= self.graph_source_chunk_limit
            <= 32
        ):
            raise ValueError("adaptive Graphiti source chunk bounds are invalid")
        if not 1 <= self.graph_call_timeout_seconds <= 120:
            raise ValueError("adaptive Graphiti call timeout is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_version": self.profile_version,
            "strategy": self.strategy.value,
            "top_k": self.top_k,
            "rerank_mode": self.rerank_mode.value,
            "router": self.router,
            "augmentation": self.augmentation,
            "graph_edge_limit": self.graph_edge_limit,
            "graph_source_chunk_target": self.graph_source_chunk_target,
            "graph_source_chunk_limit": self.graph_source_chunk_limit,
            "graph_call_timeout_seconds": self.graph_call_timeout_seconds,
        }


@dataclass(frozen=True, slots=True)
class RetrievalExecutionProfile:
    """Current process settings used for one retrieval execution."""

    profile_version: str
    strategy: RetrievalStrategy
    top_k: int
    rerank_mode: RerankMode
    dense_candidate_count: int
    lexical_candidate_count: int
    cross_modal_candidate_count: int
    lexical_analyzer_version: str | None
    lexical_query_version: str | None
    rrf_k: int
    dense_weight_micros: int
    lexical_weight_micros: int
    cross_modal_weight_micros: int
    min_cosine_similarity: float
    min_rerank_score: float
    cross_modal_min_cosine_similarity: float
    rerank_vector_weight: float
    rerank_lexical_weight: float
    mmr_lambda: float

    def __post_init__(self) -> None:
        expected_version = _profile_version(self.strategy)
        if self.profile_version != expected_version:
            raise ValueError("retrieval profile version and strategy differ")
        if not 1 <= self.top_k <= 100:
            raise ValueError("profile top_k is invalid")
        try:
            object.__setattr__(
                self,
                "rerank_mode",
                RerankMode(self.rerank_mode),
            )
        except ValueError as error:
            raise ValueError("profile rerank mode is invalid") from error
        if (
            self.strategy is RetrievalStrategy.HYBRID
            and self.rerank_mode is RerankMode.NONE
        ):
            raise ValueError("hybrid profile requires reranking")
        if (
            self.strategy is RetrievalStrategy.ITERATIVE_BALANCED
            and self.rerank_mode is RerankMode.NONE
        ):
            raise ValueError("iterative balanced profile requires reranking")
        if (
            self.rerank_mode is RerankMode.LOCAL_MINILM_V1
            and self.top_k > 20
        ):
            raise ValueError("local reranking supports top_k up to 20")
        for count in (
            self.dense_candidate_count,
            self.lexical_candidate_count,
            self.cross_modal_candidate_count,
        ):
            if not self.top_k <= count <= 100:
                raise ValueError("profile candidate count is invalid")
        for weight in (
            self.dense_weight_micros,
            self.lexical_weight_micros,
            self.cross_modal_weight_micros,
        ):
            if not 1 <= weight <= 10_000_000:
                raise ValueError("profile lane weight is invalid")
        if not 1 <= self.rrf_k <= 1000:
            raise ValueError("profile rrf_k is invalid")
        for threshold in (
            self.min_cosine_similarity,
            self.cross_modal_min_cosine_similarity,
        ):
            if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
                raise ValueError("profile cosine threshold is invalid")
        if (
            not math.isfinite(self.min_rerank_score)
            or not 0.0 <= self.min_rerank_score <= 1.0
        ):
            raise ValueError("profile rerank threshold is invalid")
        if not math.isfinite(self.mmr_lambda) or not 0.0 < self.mmr_lambda <= 1.0:
            raise ValueError("profile MMR lambda is invalid")
        if (
            not math.isfinite(self.rerank_vector_weight)
            or not math.isfinite(self.rerank_lexical_weight)
            or not 0.0 <= self.rerank_vector_weight <= 1.0
            or not 0.0 <= self.rerank_lexical_weight <= 1.0
            or abs(
                self.rerank_vector_weight + self.rerank_lexical_weight - 1.0
            )
            > 1e-9
        ):
            raise ValueError("profile rerank weights are invalid")
        if self.strategy is RetrievalStrategy.HYBRID:
            if (
                self.lexical_analyzer_version != LEXICAL_ANALYZER_VERSION
                or self.lexical_query_version != LEXICAL_QUERY_VERSION
            ):
                raise ValueError("unsupported lexical profile version")
        elif (
            self.lexical_analyzer_version is not None
            or self.lexical_query_version is not None
        ):
            raise ValueError("dense profile must not declare lexical versions")

    def as_dict(self) -> dict[str, Any]:
        """Persist only the preset and user-selected retrieval controls."""

        return {
            "profile_version": self.profile_version,
            "strategy": self.strategy.value,
            "top_k": self.top_k,
            "rerank_mode": self.rerank_mode.value,
        }

    @property
    def rerank(self) -> bool:
        return self.rerank_mode is not RerankMode.NONE


def parse_retrieval_snapshot(
    value: Mapping[str, Any],
) -> tuple[RetrievalStrategy, int, RerankMode]:
    """Validate one persisted local ChatRun retrieval preset."""

    if set(value) != {
        "profile_version",
        "strategy",
        "top_k",
        "rerank_mode",
    }:
        raise ValueError("retrieval snapshot fields are invalid")
    strategy = RetrievalStrategy(value["strategy"])
    expected_version = _profile_version(strategy)
    if value["profile_version"] != expected_version:
        raise ValueError("retrieval profile version and strategy differ")
    raw_top_k = value["top_k"]
    if isinstance(raw_top_k, bool) or not isinstance(raw_top_k, int):
        raise ValueError("retrieval snapshot top_k is invalid")
    top_k = raw_top_k
    if not 1 <= top_k <= 100:
        raise ValueError("retrieval snapshot top_k is invalid")
    rerank_mode = RerankMode(value["rerank_mode"])
    if strategy is RetrievalStrategy.HYBRID and rerank_mode is RerankMode.NONE:
        raise ValueError("hybrid snapshot requires reranking")
    if (
        strategy is RetrievalStrategy.ITERATIVE_BALANCED
        and rerank_mode is RerankMode.NONE
    ):
        raise ValueError("iterative balanced snapshot requires reranking")
    if rerank_mode is RerankMode.LOCAL_MINILM_V1 and top_k > 20:
        raise ValueError("local reranking supports top_k up to 20")
    return strategy, top_k, rerank_mode


def parse_chat_retrieval_snapshot(
    value: Mapping[str, Any],
) -> tuple[RetrievalStrategy, int, RerankMode, RetrievalExecutionType]:
    """Parse a persisted ChatRun and return its explicit execution type."""

    if value.get("profile_version") == ADAPTIVE_GRAPHITI_PROFILE_VERSION:
        profile = parse_adaptive_graphiti_snapshot(value)
        return (
            profile.strategy,
            profile.top_k,
            profile.rerank_mode,
            "adaptive_graphiti",
        )

    if value.get("augmentation") is not None:
        expected_fields = {
            "profile_version",
            "strategy",
            "top_k",
            "rerank_mode",
            "augmentation",
        }
        if set(value) != expected_fields:
            raise ValueError("graph retrieval snapshot fields are invalid")
        profile = GraphRetrievalProfile(
            profile_version=value["profile_version"],
            strategy=RetrievalStrategy(value["strategy"]),
            top_k=value["top_k"],
            rerank_mode=RerankMode(value["rerank_mode"]),
            augmentation=value["augmentation"],
        )
        return (
            profile.strategy,
            profile.top_k,
            profile.rerank_mode,
            "manual_graph",
        )
    strategy, top_k, rerank_mode = parse_retrieval_snapshot(value)
    execution_type: RetrievalExecutionType = (
        "iterative_balanced"
        if strategy is RetrievalStrategy.ITERATIVE_BALANCED
        else "simple"
    )
    return strategy, top_k, rerank_mode, execution_type


def parse_adaptive_graphiti_snapshot(
    value: Mapping[str, Any],
) -> AdaptiveGraphitiRetrievalProfile:
    """Strictly parse the independent Chat-only adaptive snapshot."""

    expected_fields = {
        "profile_version",
        "strategy",
        "top_k",
        "rerank_mode",
        "router",
        "augmentation",
        "graph_edge_limit",
        "graph_source_chunk_target",
        "graph_source_chunk_limit",
        "graph_call_timeout_seconds",
    }
    if set(value) != expected_fields:
        raise ValueError("adaptive Graphiti retrieval snapshot fields are invalid")
    raw_top_k = value["top_k"]
    if isinstance(raw_top_k, bool) or not isinstance(raw_top_k, int):
        raise ValueError("adaptive Graphiti snapshot top_k is invalid")
    return AdaptiveGraphitiRetrievalProfile(
        profile_version=value["profile_version"],
        strategy=RetrievalStrategy(value["strategy"]),
        top_k=raw_top_k,
        rerank_mode=RerankMode(value["rerank_mode"]),
        router=value["router"],
        augmentation=value["augmentation"],
        graph_edge_limit=value["graph_edge_limit"],
        graph_source_chunk_target=value["graph_source_chunk_target"],
        graph_source_chunk_limit=value["graph_source_chunk_limit"],
        graph_call_timeout_seconds=value["graph_call_timeout_seconds"],
    )


def graph_profile(
    *,
    top_k: int = 10,
    rerank_mode: RerankMode = RerankMode.CLASSIC,
) -> GraphRetrievalProfile:
    return GraphRetrievalProfile(
        profile_version=GRAPH_RETRIEVAL_PROFILE_VERSION,
        strategy=RetrievalStrategy.HYBRID,
        top_k=top_k,
        rerank_mode=rerank_mode,
        augmentation=GRAPH_AUGMENTATION_VERSION,
    )


def adaptive_graphiti_profile(
    *, top_k: int = 10, rerank_mode: RerankMode = RerankMode.NONE
) -> AdaptiveGraphitiRetrievalProfile:
    return AdaptiveGraphitiRetrievalProfile(
        profile_version=ADAPTIVE_GRAPHITI_PROFILE_VERSION,
        strategy=RetrievalStrategy.EXACT_VECTOR,
        top_k=top_k,
        rerank_mode=rerank_mode,
        router=ADAPTIVE_GRAPHITI_ROUTER_VERSION,
        augmentation=GRAPH_AUGMENTATION_VERSION,
    )


def exact_profile(
    *,
    top_k: int = 10,
    rerank_mode: RerankMode = RerankMode.CLASSIC,
) -> RetrievalExecutionProfile:
    return RetrievalExecutionProfile(
        profile_version=EXACT_PROFILE_VERSION,
        strategy=RetrievalStrategy.EXACT_VECTOR,
        top_k=top_k,
        rerank_mode=rerank_mode,
        dense_candidate_count=max(top_k, min(top_k * 4, 40)),
        lexical_candidate_count=max(top_k, 40),
        cross_modal_candidate_count=max(top_k, 20),
        lexical_analyzer_version=None,
        lexical_query_version=None,
        rrf_k=60,
        dense_weight_micros=1_000_000,
        lexical_weight_micros=1_000_000,
        cross_modal_weight_micros=1_000_000,
        min_cosine_similarity=0.35,
        min_rerank_score=0.45,
        cross_modal_min_cosine_similarity=0.25,
        rerank_vector_weight=0.65,
        rerank_lexical_weight=0.35,
        mmr_lambda=0.75,
    )


def iterative_balanced_profile(
    *,
    top_k: int = 10,
    rerank_mode: RerankMode = RerankMode.CLASSIC,
) -> RetrievalExecutionProfile:
    """Build the dense base profile used by the bounded Agent policy."""

    return RetrievalExecutionProfile(
        profile_version=ITERATIVE_BALANCED_PROFILE_VERSION,
        strategy=RetrievalStrategy.ITERATIVE_BALANCED,
        top_k=top_k,
        rerank_mode=rerank_mode,
        dense_candidate_count=max(top_k, min(top_k * 4, 40)),
        lexical_candidate_count=max(top_k, 40),
        cross_modal_candidate_count=max(top_k, 20),
        lexical_analyzer_version=None,
        lexical_query_version=None,
        rrf_k=60,
        dense_weight_micros=1_000_000,
        lexical_weight_micros=1_000_000,
        cross_modal_weight_micros=1_000_000,
        min_cosine_similarity=0.35,
        min_rerank_score=0.45,
        cross_modal_min_cosine_similarity=0.25,
        rerank_vector_weight=0.65,
        rerank_lexical_weight=0.35,
        mmr_lambda=0.75,
    )


def _profile_version(strategy: RetrievalStrategy) -> str:
    if strategy is RetrievalStrategy.EXACT_VECTOR:
        return EXACT_PROFILE_VERSION
    if strategy is RetrievalStrategy.HYBRID:
        return HYBRID_PROFILE_VERSION
    if strategy is RetrievalStrategy.ITERATIVE_BALANCED:
        return ITERATIVE_BALANCED_PROFILE_VERSION
    raise ValueError("unsupported retrieval strategy")
