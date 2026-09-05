"""Deterministic second-stage ranking for an already authorized candidate set."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping
from uuid import UUID

from rag_kb.domain import Evidence, VectorSearchHit


_TOKEN = re.compile(r"[0-9A-Za-z_]+|[\u3400-\u9fff]")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "by",
        "for",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "与",
        "和",
        "在",
        "是",
        "的",
        "了",
        "或",
        "及",
    }
)


@dataclass(frozen=True, slots=True)
class RerankedHit:
    hit: VectorSearchHit
    score: float
    vector_similarity: float
    lexical_score: float
    lexical_coverage: float


def rerank_hits(
    query: str,
    hits: Iterable[VectorSearchHit],
    *,
    top_k: int,
    vector_weight: float = 0.65,
    lexical_weight: float = 0.35,
    mmr_lambda: float = 0.75,
) -> tuple[RerankedHit, ...]:
    """Fuse semantic and lexical signals, then select diverse final evidence.

    This is deliberately local and deterministic. It does not claim to be a
    cross-encoder; its job is to repair obvious candidate-ordering failures
    while keeping retrieval available when a separate reranker provider is not.
    """

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not math.isclose(vector_weight + lexical_weight, 1.0, abs_tol=1e-9):
        raise ValueError("rerank weights must sum to one")
    if not 0.0 < mmr_lambda <= 1.0:
        raise ValueError("mmr_lambda must be between zero and one")

    scored = score_hits(
        query,
        hits,
        vector_weight=vector_weight,
        lexical_weight=lexical_weight,
    )
    if not scored:
        return ()

    remaining = list(scored)
    selected: list[RerankedHit] = []
    while remaining and len(selected) < top_k:
        best = max(
            remaining,
            key=lambda item: (
                _mmr_value(item, selected, mmr_lambda),
                item.score,
                -item.hit.index_chunk_id.int,
            ),
        )
        if selected and max(
            _text_similarity(best.hit.text, prior.hit.text) for prior in selected
        ) >= 0.45:
            remaining.remove(best)
            continue
        selected.append(best)
        remaining.remove(best)
    return tuple(selected)


def score_hits(
    query: str,
    hits: Iterable[VectorSearchHit],
    *,
    vector_weight: float = 0.65,
    lexical_weight: float = 0.35,
    reference_hits: Iterable[VectorSearchHit] | None = None,
) -> tuple[RerankedHit, ...]:
    """Score candidates, optionally freezing lexical statistics to the original pool.

    Source-context supplements must not change scores of already retrieved chunks.
    The reference pool contains original candidates only, never generated questions.
    """

    if not math.isclose(vector_weight + lexical_weight, 1.0, abs_tol=1e-9):
        raise ValueError("rerank weights must sum to one")
    candidates = tuple(hits)
    if not candidates:
        return ()
    query_terms = _terms(query)
    document_terms = tuple(_terms(hit.text) for hit in candidates)
    statistics_terms = (
        document_terms if reference_hits is None
        else tuple(_terms(hit.text) for hit in reference_hits)
    )
    if not statistics_terms:
        raise ValueError("reference hits must not be empty when scoring candidates")
    document_frequency = Counter(
        term for terms in statistics_terms for term in set(terms)
    )
    average_length = sum(len(terms) for terms in statistics_terms) / len(statistics_terms)
    return tuple(
        _score(
            hit,
            terms,
            query_terms,
            document_frequency,
            len(statistics_terms),
            average_length,
            vector_weight,
            lexical_weight,
        )
        for hit, terms in zip(candidates, document_terms, strict=True)
    )


def order_model_scored_evidence(
    evidence: Iterable[Evidence],
    scores: Mapping[UUID, float],
    *,
    mmr_lambda: float = 0.75,
) -> tuple[Evidence, ...]:
    """Order model-scored evidence with the existing diversity penalty."""

    if not 0.0 < mmr_lambda <= 1.0:
        raise ValueError("mmr_lambda must be between zero and one")
    candidates = tuple(evidence)
    if set(scores) != {item.index_chunk_id for item in candidates}:
        raise ValueError("model score identities do not match evidence")
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("model scores must be finite")
    source_order = {
        item.index_chunk_id: index for index, item in enumerate(candidates)
    }
    remaining = list(candidates)
    selected: list[Evidence] = []
    while remaining:
        best = max(
            remaining,
            key=lambda item: (
                _model_mmr_value(item, selected, scores, mmr_lambda),
                scores[item.index_chunk_id],
                -source_order[item.index_chunk_id],
                -item.index_chunk_id.int,
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return tuple(selected)


def _score(
    hit: VectorSearchHit,
    terms: tuple[str, ...],
    query_terms: tuple[str, ...],
    document_frequency: Counter[str],
    document_count: int,
    average_length: float,
    vector_weight: float,
    lexical_weight: float,
) -> RerankedHit:
    vector_similarity = max(-1.0, min(1.0, 1.0 - hit.cosine_distance))
    lexical_score, coverage = _bm25_score(
        terms,
        query_terms,
        document_frequency,
        document_count,
        average_length,
    )
    score = vector_weight * ((vector_similarity + 1.0) / 2.0) + lexical_weight * lexical_score
    if query_terms and _normalized_query(query_terms) in _normalized_text(hit.text):
        score = min(1.0, score + 0.08)
    return RerankedHit(
        hit=hit,
        score=score,
        vector_similarity=vector_similarity,
        lexical_score=lexical_score,
        lexical_coverage=coverage,
    )


def _bm25_score(
    document_terms: tuple[str, ...],
    query_terms: tuple[str, ...],
    document_frequency: Counter[str],
    document_count: int,
    average_length: float,
) -> tuple[float, float]:
    if not query_terms:
        return 0.0, 0.0
    counts = Counter(document_terms)
    unique_query_terms = tuple(dict.fromkeys(query_terms))
    k1 = 1.2
    b = 0.75
    length = max(1, len(document_terms))
    normalization = 1.0 - b + b * length / max(1.0, average_length)
    raw = 0.0
    maximum = 0.0
    matched = 0
    for term in unique_query_terms:
        idf = math.log1p(
            (document_count - document_frequency[term] + 0.5)
            / (document_frequency[term] + 0.5)
        )
        maximum += idf * (k1 + 1.0) / (k1 * normalization + 1.0)
        frequency = counts[term]
        if frequency:
            matched += 1
            raw += idf * frequency * (k1 + 1.0) / (
                frequency + k1 * normalization
            )
    lexical_score = raw / maximum if maximum else 0.0
    return min(1.0, lexical_score), matched / len(unique_query_terms)


def _mmr_value(item: RerankedHit, selected: list[RerankedHit], mmr_lambda: float) -> float:
    if not selected:
        return item.score
    redundancy = max(
        _text_similarity(item.hit.text, prior.hit.text) for prior in selected
    )
    return mmr_lambda * item.score - (1.0 - mmr_lambda) * redundancy


def _model_mmr_value(
    item: Evidence,
    selected: list[Evidence],
    scores: Mapping[UUID, float],
    mmr_lambda: float,
) -> float:
    score = scores[item.index_chunk_id]
    if not selected:
        return score
    redundancy = max(
        _text_similarity(item.text, prior.text) for prior in selected
    )
    return mmr_lambda * score - (1.0 - mmr_lambda) * redundancy


def _text_similarity(left: str, right: str) -> float:
    left_terms = set(_terms(left))
    right_terms = set(_terms(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def _terms(value: str) -> tuple[str, ...]:
    return tuple(
        token.lower()
        for token in _TOKEN.findall(value)
        if token.lower() not in _STOPWORDS
    )


def _normalized_query(terms: tuple[str, ...]) -> str:
    return "".join(terms)


def _normalized_text(value: str) -> str:
    return "".join(_terms(value))
