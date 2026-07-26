"""Integer-quantized semantic distance smoothing and constrained selection."""

from __future__ import annotations

import hashlib
import json
from uuid import UUID

from rag_kb.document_processing.profiles import SEMANTIC_CHUNKING_CONFIG
from rag_kb.document_processing.tokenization import count_chunk_tokens
from rag_kb.domain import (
    ChunkBoundary,
    ChunkBoundaryReason,
    ErrorCode,
    IndexChunkPlan,
    IndexingExecutionError,
    IndexingPhase,
    SemanticUnit,
)


def build_chunk_plan(
    *,
    indexed_document_version_id: UUID,
    source_checksum_sha256: str,
    profile_fingerprint: str,
    units: tuple[SemanticUnit, ...],
    vectors: tuple[tuple[float, ...], ...] | None,
    sequence_hash: str,
) -> IndexChunkPlan:
    """Build a deterministic immutable plan from validated unit vectors."""

    if not units:
        raise _failed("non_empty_units")
    hard_boundaries = any(unit.hard_boundary_before for unit in units[1:])
    total_tokens = count_chunk_tokens("\n\n".join(unit.text for unit in units))
    if total_tokens <= _config_int("max_chunk_tokens") and not hard_boundaries:
        boundaries: tuple[ChunkBoundary, ...] = ()
    else:
        if vectors is None or len(vectors) != len(units):
            raise _failed("analysis_vector_count")
        scores = smoothed_distances(units, vectors)
        boundaries = _select_all_regions(units, scores)

    payload = {
        "indexed_document_version_id": str(indexed_document_version_id),
        "source_checksum_sha256": source_checksum_sha256,
        "profile_fingerprint": profile_fingerprint,
        "unit_sequence_hash": sequence_hash,
        "unit_count": len(units),
        "chunk_count": len(boundaries) + 1,
        "boundaries": [_boundary_json(item) for item in boundaries],
    }
    return IndexChunkPlan(
        indexed_document_version_id=indexed_document_version_id,
        source_checksum_sha256=source_checksum_sha256,
        profile_fingerprint=profile_fingerprint,
        unit_sequence_hash=payload["unit_sequence_hash"],
        unit_count=len(units),
        chunk_count=len(boundaries) + 1,
        boundaries=boundaries,
        plan_hash=_sha256_json(payload),
    )


def smoothed_distances(
    units: tuple[SemanticUnit, ...],
    vectors: tuple[tuple[float, ...], ...],
) -> tuple[int | None, ...]:
    """Return one score after each unit except the last; hard boundaries are None."""

    if len(units) != len(vectors):
        raise ValueError("every semantic unit requires one vector")
    raw: list[int | None] = []
    quantization = _config_int("distance_quantization")
    for index in range(len(units) - 1):
        if units[index + 1].hard_boundary_before is not None:
            raw.append(None)
            continue
        dot = sum(
            float(left) * float(right)
            for left, right in zip(vectors[index], vectors[index + 1], strict=True)
        )
        raw.append(round(min(2.0, max(0.0, 1.0 - dot)) * quantization))

    result: list[int | None] = []
    for index, value in enumerate(raw):
        if value is None:
            result.append(None)
            continue
        weighted = 2 * value
        weight = 2
        for neighbor in (index - 1, index + 1):
            if 0 <= neighbor < len(raw) and raw[neighbor] is not None:
                weighted += int(raw[neighbor])
                weight += 1
        result.append(_round_ratio(weighted, weight))
    return tuple(result)


def validate_plan(
    plan: IndexChunkPlan,
    *,
    indexed_document_version_id: UUID,
    source_checksum_sha256: str,
    profile_fingerprint: str,
    units: tuple[SemanticUnit, ...],
    sequence_hash: str,
) -> None:
    expected = (
        plan.indexed_document_version_id == indexed_document_version_id
        and plan.source_checksum_sha256 == source_checksum_sha256
        and plan.profile_fingerprint == profile_fingerprint
        and plan.unit_count == len(units)
        and plan.unit_sequence_hash == sequence_hash
    )
    payload = {
        "indexed_document_version_id": str(plan.indexed_document_version_id),
        "source_checksum_sha256": plan.source_checksum_sha256,
        "profile_fingerprint": plan.profile_fingerprint,
        "unit_sequence_hash": plan.unit_sequence_hash,
        "unit_count": plan.unit_count,
        "chunk_count": plan.chunk_count,
        "boundaries": [_boundary_json(item) for item in plan.boundaries],
    }
    if not expected or plan.plan_hash != _sha256_json(payload):
        raise IndexingExecutionError(
            ErrorCode.INDEX_CHUNK_PLAN_MISMATCH,
            phase=IndexingPhase.SEMANTIC_ANALYSIS,
            diagnostic={"check": "chunk_plan_facts"},
        )


def _select_all_regions(
    units: tuple[SemanticUnit, ...],
    scores: tuple[int | None, ...],
) -> tuple[ChunkBoundary, ...]:
    boundaries: list[ChunkBoundary] = []
    start = 0
    for index in range(1, len(units) + 1):
        if index < len(units) and units[index].hard_boundary_before is None:
            continue
        boundaries.extend(_select_region(units, scores, start, index))
        if index < len(units):
            reason = ChunkBoundaryReason(units[index].hard_boundary_before)
            boundaries.append(ChunkBoundary(index - 1, reason))
        start = index
    return tuple(boundaries)


def _select_region(
    units: tuple[SemanticUnit, ...],
    scores: tuple[int | None, ...],
    start: int,
    end: int,
) -> list[ChunkBoundary]:
    if start >= end:
        return []
    if count_chunk_tokens(_joined(units, start, end)) <= _config_int("max_chunk_tokens"):
        return []

    region_scores = [
        int(scores[index])
        for index in range(start, end - 1)
        if scores[index] is not None
    ]
    threshold = _semantic_threshold(region_scores)
    gains = {
        index: max(0, int(scores[index]) - threshold)
        for index in range(start, end - 1)
        if scores[index] is not None
    }
    # end position -> (reward, current closeness, count, path)
    states: dict[int, tuple[int, int, int, tuple[int, ...]]] = {
        start: (0, 0, 0, ())
    }
    for position in range(start + 1, end + 1):
        best: tuple[int, int, int, tuple[int, ...]] | None = None
        for prior in range(position - 1, start - 1, -1):
            tokens = count_chunk_tokens(_joined(units, prior, position))
            if tokens > _config_int("max_chunk_tokens"):
                break
            prior_state = states.get(prior)
            if prior_state is None:
                continue
            region_is_small = count_chunk_tokens(_joined(units, start, end)) < _config_int(
                "min_chunk_tokens"
            )
            if tokens < _config_int("min_chunk_tokens") and not region_is_small:
                continue
            penalty = _size_penalty(tokens)
            reward = prior_state[0] - penalty
            if position < end:
                reward += gains.get(position - 1, 0)
            path = (*prior_state[3], position - 1) if position < end else prior_state[3]
            candidate = (
                reward,
                -abs(tokens - _config_int("target_chunk_tokens")),
                -(prior_state[2] + 1),
                path,
            )
            if best is None or _better(candidate, best):
                best = candidate
        if best is not None:
            states[position] = best

    selected = states.get(end)
    if selected is None:
        return _fallback_boundaries(units, start, end)
    result: list[ChunkBoundary] = []
    for ordinal in selected[3]:
        score = scores[ordinal]
        if gains.get(ordinal, 0) > 0 and score is not None:
            result.append(
                ChunkBoundary(
                    ordinal,
                    ChunkBoundaryReason.SEMANTIC,
                    int(score),
                )
            )
        else:
            result.append(
                ChunkBoundary(ordinal, ChunkBoundaryReason.MAX_TOKENS)
            )
    return result


def _fallback_boundaries(
    units: tuple[SemanticUnit, ...],
    start: int,
    end: int,
) -> list[ChunkBoundary]:
    result: list[ChunkBoundary] = []
    cursor = start
    while cursor < end:
        selected = cursor
        for position in range(cursor + 1, end + 1):
            if count_chunk_tokens(_joined(units, cursor, position)) <= _config_int(
                "max_chunk_tokens"
            ):
                selected = position
            else:
                break
        if selected == cursor:
            raise _failed("max_chunk_tokens")
        if selected < end:
            result.append(
                ChunkBoundary(selected - 1, ChunkBoundaryReason.MAX_TOKENS)
            )
        cursor = selected
    return result


def _semantic_threshold(values: list[int]) -> int:
    if not values:
        return 0
    middle = _integer_median(values)
    if len(values) < 3:
        return middle
    deviations = [abs(value - middle) for value in values]
    mad = _integer_median(deviations)
    multiplier = _config_int("semantic_mad_multiplier_micros")
    return middle + _round_ratio(mad * multiplier, 1_000_000)


def _size_penalty(tokens: int) -> int:
    target = _config_int("target_chunk_tokens")
    weight = _config_int("size_penalty_weight_micros")
    return _round_ratio(abs(tokens - target) * weight, target)


def _better(
    candidate: tuple[int, int, int, tuple[int, ...]],
    existing: tuple[int, int, int, tuple[int, ...]],
) -> bool:
    if candidate[:3] != existing[:3]:
        return candidate[:3] > existing[:3]
    return candidate[3] < existing[3]


def _joined(units: tuple[SemanticUnit, ...], start: int, end: int) -> str:
    return "\n\n".join(unit.text for unit in units[start:end])


def _boundary_json(boundary: ChunkBoundary) -> dict[str, int | str | None]:
    return {
        "after_unit_ordinal": boundary.after_unit_ordinal,
        "reason": boundary.reason.value,
        "score_micros": boundary.score_micros,
    }


def _sha256_json(value: dict) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _round_ratio(numerator: int, denominator: int) -> int:
    return (numerator + denominator // 2) // denominator


def _integer_median(values: list[int]) -> int:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return _round_ratio(ordered[middle - 1] + ordered[middle], 2)


def _config_int(name: str) -> int:
    value = SEMANTIC_CHUNKING_CONFIG[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _failed(check: str) -> IndexingExecutionError:
    return IndexingExecutionError(
        ErrorCode.SEMANTIC_CHUNKING_FAILED,
        phase=IndexingPhase.SEMANTIC_ANALYSIS,
        diagnostic={"check": check},
    )
