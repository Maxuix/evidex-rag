"""Deterministic reciprocal-rank fusion across incomparable vector spaces."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN

from rag_kb.domain import VectorSearchHit


@dataclass(frozen=True, slots=True)
class FusedHit:
    hit: VectorSearchHit
    score: float
    text_rank: int | None
    cross_modal_rank: int | None
    matched_representations: tuple[str, ...]


def reciprocal_rank_fusion(
    text_hits: tuple[VectorSearchHit, ...],
    cross_modal_hits: tuple[VectorSearchHit, ...],
    *,
    rrf_k: int = 60,
    cross_modal_weight_micros: int = 1_000_000,
    top_k: int,
) -> tuple[FusedHit, ...]:
    if rrf_k < 1 or cross_modal_weight_micros < 1 or top_k < 1:
        raise ValueError("RRF settings must be positive")
    grouped: dict[str, dict] = {}
    for lane, hits, weight in (
        ("text", text_hits, Decimal(1)),
        (
            "cross",
            cross_modal_hits,
            Decimal(cross_modal_weight_micros) / Decimal(1_000_000),
        ),
    ):
        seen_groups: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            group = hit.evidence_group_key or str(hit.index_chunk_id)
            if group in seen_groups:
                entry = grouped.get(group)
                if entry is not None:
                    entry["representations"].add(hit.representation_kind)
                continue
            seen_groups.add(group)
            entry = grouped.setdefault(
                group,
                {
                    "hit": hit,
                    "score": Decimal(0),
                    "text_rank": None,
                    "cross_rank": None,
                    "representations": set(),
                },
            )
            entry["score"] += weight / Decimal(rrf_k + rank)
            entry[f"{lane}_rank"] = rank
            entry["representations"].add(hit.representation_kind)
            current = entry["hit"]
            if _preferred(hit, current):
                entry["hit"] = hit
    ordered = sorted(
        grouped.values(),
        key=lambda item: (
            -item["score"],
            min(item["text_rank"] or 2**31, item["cross_rank"] or 2**31),
            _modality_priority(item["hit"].modality),
            item["hit"].index_chunk_id.int,
        ),
    )[:top_k]
    return tuple(
        FusedHit(
            hit=item["hit"],
            score=float(item["score"].quantize(Decimal("0.000000000001"), rounding=ROUND_HALF_EVEN)),
            text_rank=item["text_rank"],
            cross_modal_rank=item["cross_rank"],
            matched_representations=tuple(sorted(item["representations"])),
        )
        for item in ordered
    )


def _preferred(candidate: VectorSearchHit, current: VectorSearchHit) -> bool:
    return (
        bool(candidate.text),
        -_modality_priority(candidate.modality),
        -candidate.cosine_distance,
        -candidate.index_chunk_id.int,
    ) > (
        bool(current.text),
        -_modality_priority(current.modality),
        -current.cosine_distance,
        -current.index_chunk_id.int,
    )


def _modality_priority(modality: str) -> int:
    return {"text": 0, "table": 1, "image": 2}.get(modality, 3)
