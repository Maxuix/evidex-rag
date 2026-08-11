"""Small persisted retrieval presets and current runtime execution settings."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from rag_kb.document_processing.lexical import (
    LEXICAL_ANALYZER_VERSION,
    LEXICAL_QUERY_VERSION,
)
from rag_kb.domain import RerankMode, RetrievalStrategy


EXACT_PROFILE_VERSION = "exact_vector_v2"
HYBRID_PROFILE_VERSION = "hybrid_fts_rrf_v2"
LEGACY_EXACT_PROFILE_VERSION = "exact_vector_v1"
LEGACY_HYBRID_PROFILE_VERSION = "hybrid_fts_rrf_v1"


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
        expected_version = (
            EXACT_PROFILE_VERSION
            if self.strategy is RetrievalStrategy.EXACT_VECTOR
            else HYBRID_PROFILE_VERSION
        )
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
            raise ValueError("exact profile must not declare lexical versions")

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

    legacy_fields = {
        "profile_version",
        "strategy",
        "top_k",
        "rerank",
    }
    current_fields = {
        "profile_version",
        "strategy",
        "top_k",
        "rerank_mode",
    }
    metadata_fields = {"document_scope"}
    snapshot_fields = set(value)
    legacy = legacy_fields <= snapshot_fields and not (
        snapshot_fields - legacy_fields - metadata_fields
    )
    current = current_fields <= snapshot_fields and not (
        snapshot_fields - current_fields - metadata_fields
    )
    if not legacy and not current:
        raise ValueError("retrieval snapshot fields are invalid")
    strategy = RetrievalStrategy(value["strategy"])
    if legacy:
        expected_version = (
            LEGACY_EXACT_PROFILE_VERSION
            if strategy is RetrievalStrategy.EXACT_VECTOR
            else LEGACY_HYBRID_PROFILE_VERSION
        )
    else:
        expected_version = (
            EXACT_PROFILE_VERSION
            if strategy is RetrievalStrategy.EXACT_VECTOR
            else HYBRID_PROFILE_VERSION
        )
    if value["profile_version"] != expected_version:
        raise ValueError("retrieval profile version and strategy differ")
    raw_top_k = value["top_k"]
    if isinstance(raw_top_k, bool) or not isinstance(raw_top_k, int):
        raise ValueError("retrieval snapshot top_k is invalid")
    top_k = raw_top_k
    if not 1 <= top_k <= 100:
        raise ValueError("retrieval snapshot top_k is invalid")
    if legacy:
        rerank_mode = (
            RerankMode.CLASSIC
            if _require_bool(value["rerank"])
            else RerankMode.NONE
        )
    else:
        rerank_mode = RerankMode(value["rerank_mode"])
    if strategy is RetrievalStrategy.HYBRID and rerank_mode is RerankMode.NONE:
        raise ValueError("hybrid snapshot requires reranking")
    if rerank_mode is RerankMode.LOCAL_MINILM_V1 and top_k > 20:
        raise ValueError("local reranking supports top_k up to 20")
    return strategy, top_k, rerank_mode


def exact_profile(
    *,
    top_k: int = 10,
    rerank_mode: RerankMode | None = None,
    rerank: bool | None = None,
) -> RetrievalExecutionProfile:
    if rerank_mode is not None and rerank is not None:
        raise ValueError("choose rerank_mode or legacy rerank, not both")
    resolved_mode = (
        rerank_mode
        if rerank_mode is not None
        else RerankMode.CLASSIC
        if rerank is None or rerank
        else RerankMode.NONE
    )
    return RetrievalExecutionProfile(
        profile_version=EXACT_PROFILE_VERSION,
        strategy=RetrievalStrategy.EXACT_VECTOR,
        top_k=top_k,
        rerank_mode=resolved_mode,
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


def _require_bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("snapshot boolean is invalid")
    return value
