"""Shared deterministic evidence eligibility used before and after fusion."""

from __future__ import annotations

from dataclasses import dataclass

from rag_kb.domain import Evidence, EvidenceScoreKind
from rag_kb.retrieval.reranker import RerankedHit


@dataclass(frozen=True, slots=True)
class EvidenceEligibilityPolicy:
    min_cosine_similarity: float
    min_rerank_score: float = 0.45
    cross_modal_min_cosine_similarity: float = 0.25

    def __post_init__(self) -> None:
        if not -1.0 <= self.min_cosine_similarity <= 1.0:
            raise ValueError("min_cosine_similarity must be between -1 and 1")
        if not 0.0 <= self.min_rerank_score <= 1.0:
            raise ValueError("min_rerank_score must be between 0 and 1")
        if not -1.0 <= self.cross_modal_min_cosine_similarity <= 1.0:
            raise ValueError(
                "cross_modal_min_cosine_similarity must be between -1 and 1"
            )

    def usable(self, item: Evidence) -> bool:
        if item.score is None:
            return False
        if item.score_kind is EvidenceScoreKind.RECIPROCAL_RANK_FUSION:
            if item.vector_similarity is None:
                return False
            if item.text_space_rank is not None or item.lexical_rank is not None:
                return item.vector_similarity >= self.min_cosine_similarity
            return (
                item.cross_modal_rank is not None
                and item.vector_similarity
                >= self.cross_modal_min_cosine_similarity
            )
        if item.score_kind is not EvidenceScoreKind.HYBRID_RERANK:
            return item.score >= self.min_cosine_similarity
        return self.usable_text_candidate_values(
            vector_similarity=item.vector_similarity,
            score=item.score,
            lexical_coverage=item.lexical_coverage,
        )

    def usable_text_candidate(self, item: RerankedHit) -> bool:
        return self.usable_text_candidate_values(
            vector_similarity=item.vector_similarity,
            score=item.score,
            lexical_coverage=item.lexical_coverage,
        )

    def usable_text_candidate_values(
        self,
        *,
        vector_similarity: float | None,
        score: float,
        lexical_coverage: float,
    ) -> bool:
        return (
            vector_similarity is not None
            and vector_similarity >= self.min_cosine_similarity
            and score >= self.min_rerank_score
            and (
                lexical_coverage > 0.0
                or score >= max(self.min_rerank_score + 0.10, 0.55)
            )
        )
