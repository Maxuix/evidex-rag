"""Strict, versioned retrieval execution snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from rag_kb.document_processing.lexical import (
    LEXICAL_ANALYZER_VERSION,
    LEXICAL_QUERY_VERSION,
)
from rag_kb.domain import RetrievalStrategy


EXACT_PROFILE_VERSION = "exact_vector_v1"
HYBRID_PROFILE_VERSION = "hybrid_fts_rrf_v1"


@dataclass(frozen=True, slots=True)
class RetrievalExecutionProfile:
    profile_version: str
    strategy: RetrievalStrategy
    top_k: int
    rerank: bool
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
        if self.strategy not in {
            RetrievalStrategy.EXACT_VECTOR,
            RetrievalStrategy.HYBRID,
        }:
            raise ValueError("retrieval profile strategy is unsupported")
        expected = (
            EXACT_PROFILE_VERSION
            if self.strategy is RetrievalStrategy.EXACT_VECTOR
            else HYBRID_PROFILE_VERSION
        )
        if self.profile_version != expected:
            raise ValueError("retrieval profile version and strategy differ")
        if not 1 <= self.top_k <= 100:
            raise ValueError("profile top_k is invalid")
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
        if (
            not math.isfinite(self.mmr_lambda)
            or not 0.0 < self.mmr_lambda <= 1.0
        ):
            raise ValueError("profile MMR lambda is invalid")
        if (
            not math.isfinite(self.rerank_vector_weight)
            or not math.isfinite(self.rerank_lexical_weight)
            or not 0.0 <= self.rerank_vector_weight <= 1.0
            or not 0.0 <= self.rerank_lexical_weight <= 1.0
        ):
            raise ValueError("profile rerank weight is invalid")
        if abs(
            self.rerank_vector_weight + self.rerank_lexical_weight - 1.0
        ) > 1e-9:
            raise ValueError("profile rerank weights must sum to one")
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
        value = asdict(self)
        value["strategy"] = self.strategy.value
        return value

    @classmethod
    def from_snapshot(
        cls,
        value: Mapping[str, Any],
    ) -> "RetrievalExecutionProfile":
        expected_fields = {
            "profile_version",
            "strategy",
            "top_k",
            "rerank",
            "dense_candidate_count",
            "lexical_candidate_count",
            "cross_modal_candidate_count",
            "lexical_analyzer_version",
            "lexical_query_version",
            "rrf_k",
            "dense_weight_micros",
            "lexical_weight_micros",
            "cross_modal_weight_micros",
            "min_cosine_similarity",
            "min_rerank_score",
            "cross_modal_min_cosine_similarity",
            "rerank_vector_weight",
            "rerank_lexical_weight",
            "mmr_lambda",
        }
        if set(value) != expected_fields:
            raise ValueError("retrieval snapshot fields are invalid")
        return cls(
            profile_version=str(value["profile_version"]),
            strategy=RetrievalStrategy(value["strategy"]),
            top_k=int(value["top_k"]),
            rerank=_require_bool(value["rerank"]),
            dense_candidate_count=int(value["dense_candidate_count"]),
            lexical_candidate_count=int(value["lexical_candidate_count"]),
            cross_modal_candidate_count=int(
                value["cross_modal_candidate_count"]
            ),
            lexical_analyzer_version=_optional_str(
                value["lexical_analyzer_version"]
            ),
            lexical_query_version=_optional_str(
                value["lexical_query_version"]
            ),
            rrf_k=int(value["rrf_k"]),
            dense_weight_micros=int(value["dense_weight_micros"]),
            lexical_weight_micros=int(value["lexical_weight_micros"]),
            cross_modal_weight_micros=int(
                value["cross_modal_weight_micros"]
            ),
            min_cosine_similarity=float(value["min_cosine_similarity"]),
            min_rerank_score=float(value["min_rerank_score"]),
            cross_modal_min_cosine_similarity=float(
                value["cross_modal_min_cosine_similarity"]
            ),
            rerank_vector_weight=float(value["rerank_vector_weight"]),
            rerank_lexical_weight=float(value["rerank_lexical_weight"]),
            mmr_lambda=float(value["mmr_lambda"]),
        )


def replace_profile(
    value: RetrievalExecutionProfile,
    *,
    top_k: int,
    rerank: bool,
) -> RetrievalExecutionProfile:
    fields = value.as_dict()
    fields["strategy"] = value.strategy
    fields["top_k"] = top_k
    fields["rerank"] = rerank
    fields["dense_candidate_count"] = max(
        top_k, min(top_k * 4, value.dense_candidate_count)
    )
    fields["lexical_candidate_count"] = max(
        top_k, value.lexical_candidate_count
    )
    fields["cross_modal_candidate_count"] = max(
        top_k, value.cross_modal_candidate_count
    )
    return RetrievalExecutionProfile(
        **fields,
    )


def exact_profile(
    *, top_k: int = 10, rerank: bool = True
) -> RetrievalExecutionProfile:
    return RetrievalExecutionProfile(
        profile_version=EXACT_PROFILE_VERSION,
        strategy=RetrievalStrategy.EXACT_VECTOR,
        top_k=top_k,
        rerank=rerank,
        dense_candidate_count=max(top_k, min(top_k * 4, 40)),
        lexical_candidate_count=40,
        cross_modal_candidate_count=20,
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


def _optional_str(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise ValueError("snapshot version is invalid")
