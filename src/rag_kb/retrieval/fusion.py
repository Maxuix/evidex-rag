"""Deterministic reciprocal-rank fusion across incomparable vector spaces."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Mapping
from uuid import UUID

from rag_kb.domain import (
    EvidenceGroupIdentity,
    VectorSearchHit,
    evidence_group_identity,
)


@dataclass(frozen=True, slots=True)
class FusedHit:
    hit: VectorSearchHit
    group_key: str
    score: float
    text_rank: int | None
    lexical_rank: int | None
    cross_modal_rank: int | None
    matched_representations: tuple[str, ...]


def reciprocal_rank_fusion(
    text_hits: tuple[VectorSearchHit, ...],
    cross_modal_hits: tuple[VectorSearchHit, ...],
    *,
    rrf_k: int = 60,
    cross_modal_weight_micros: int = 1_000_000,
    top_k: int,
    group_keys_by_chunk: Mapping[UUID, tuple[str, ...]] | None = None,
) -> tuple[FusedHit, ...]:
    return reciprocal_rank_fusion_lanes(
        (
            ("dense_text", text_hits, 1_000_000),
            ("cross_modal", cross_modal_hits, cross_modal_weight_micros),
        ),
        rrf_k=rrf_k,
        top_k=top_k,
        group_keys_by_chunk=group_keys_by_chunk,
    )


def reciprocal_rank_fusion_lanes(
    lanes: tuple[tuple[str, tuple[VectorSearchHit, ...], int], ...],
    *,
    rrf_k: int = 60,
    top_k: int,
    group_keys_by_chunk: Mapping[UUID, tuple[str, ...]] | None = None,
) -> tuple[FusedHit, ...]:
    supported_lanes = {"dense_text", "lexical", "cross_modal"}
    if (
        rrf_k < 1
        or top_k < 1
        or not lanes
        or any(name not in supported_lanes or weight < 1 for name, _hits, weight in lanes)
        or len({name for name, _hits, _weight in lanes}) != len(lanes)
    ):
        raise ValueError("RRF settings must be positive")
    grouped: dict[EvidenceGroupIdentity, dict] = {}
    for lane, hits, weight_micros in lanes:
        weight = Decimal(weight_micros) / Decimal(1_000_000)
        seen_groups: set[EvidenceGroupIdentity] = set()
        for rank, hit in enumerate(hits, start=1):
            groups = (
                group_keys_by_chunk.get(hit.index_chunk_id, ())
                if group_keys_by_chunk is not None
                else ()
            ) or (hit.evidence_group_key or str(hit.index_chunk_id),)
            for group in groups:
                identity = evidence_group_identity(
                    hit.indexed_document_version_id,
                    group,
                )
                if identity in seen_groups:
                    entry = grouped.get(identity)
                    if entry is not None:
                        entry["representations"].add(hit.representation_kind)
                    continue
                seen_groups.add(identity)
                entry = grouped.setdefault(
                    identity,
                    {
                        "group_key": group,
                        "hit": hit,
                        "score": Decimal(0),
                        "text_rank": None,
                        "lexical_rank": None,
                        "cross_rank": None,
                        "representations": set(),
                    },
                )
                entry["score"] += weight / Decimal(rrf_k + rank)
                rank_key = {
                    "dense_text": "text_rank",
                    "lexical": "lexical_rank",
                    "cross_modal": "cross_rank",
                }[lane]
                entry[rank_key] = rank
                entry["representations"].add(hit.representation_kind)
                current = entry["hit"]
                if _preferred(hit, current):
                    entry["hit"] = hit
    ordered = sorted(
        grouped.values(),
        key=lambda item: (
            -item["score"],
            min(
                item["text_rank"] or 2**31,
                item["lexical_rank"] or 2**31,
                item["cross_rank"] or 2**31,
            ),
            _modality_priority(item["hit"].modality),
            item["hit"].index_chunk_id.int,
        ),
    )[:top_k]
    return tuple(
        FusedHit(
            hit=item["hit"],
            group_key=item["group_key"],
            score=float(item["score"].quantize(Decimal("0.000000000001"), rounding=ROUND_HALF_EVEN)),
            text_rank=item["text_rank"],
            lexical_rank=item["lexical_rank"],
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
