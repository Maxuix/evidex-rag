"""Integer-quantized semantic distance smoothing and constrained selection."""

from __future__ import annotations

import hashlib
import json
from uuid import UUID

from rag_kb.document_processing.profiles import SEMANTIC_CHUNKING_CONFIG
from rag_kb.document_processing.tokenization import count_chunk_tokens
from rag_kb.document_processing.semantic_text import joined_units
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
    total_tokens = count_chunk_tokens(joined_units(units))
    if total_tokens <= _config_int("max_chunk_tokens") and not hard_boundaries:
        boundaries: tuple[ChunkBoundary, ...] = ()
    else:
        if vectors is None and not requires_semantic_vectors(units):
            scores = tuple(None if unit.hard_boundary_before else 0 for unit in units[1:])
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
    return _merge_small_section_chunks(units, tuple(boundaries))


def _merge_small_section_chunks(
    units: tuple[SemanticUnit, ...],
    boundaries: tuple[ChunkBoundary, ...],
) -> tuple[ChunkBoundary, ...]:
    """Remove only mergeable SECTION cuts around sub-minimum spans.

    The planner has already selected all semantic and hard-boundary cuts.  This
    pass works on those immutable spans, preferring the following span and then
    the preceding span for a small span.  Boundary objects that are retained
    are returned unchanged; only SECTION objects between successfully merged
    spans are dropped.
    """

    if not boundaries:
        return boundaries

    # A boundary after ordinal ``n`` separates [start, n + 1) from the next
    # span.  Preserve the original boundary objects so score/reason identity
    # remains part of the plan hash.
    spans: list[list[int]] = []
    start = 0
    for boundary in boundaries:
        end = boundary.after_unit_ordinal + 1
        spans.append([start, end])
        start = end
    spans.append([start, len(units)])
    retained = list(boundaries)
    token_cache: dict[tuple[int, int], int] = {}

    def span_tokens(start_unit: int, end_unit: int) -> int:
        key = (start_unit, end_unit)
        if key not in token_cache:
            token_cache[key] = count_chunk_tokens(_joined(units, start_unit, end_unit))
        return token_cache[key]

    minimum = _config_int("min_chunk_tokens")
    maximum = _config_int("max_chunk_tokens")
    index = 0
    while index < len(spans):
        current_start, current_end = spans[index]
        if span_tokens(current_start, current_end) >= minimum:
            index += 1
            continue

        # Next-first is the deterministic tie-break.  A section boundary is
        # the only removable cut; all other reasons remain hard boundaries.
        if (
            index < len(retained)
            and retained[index].reason is ChunkBoundaryReason.SECTION
        ):
            next_end = spans[index + 1][1]
            if span_tokens(current_start, next_end) <= maximum:
                spans[index][1] = next_end
                spans.pop(index + 1)
                retained.pop(index)
                # Re-check the merged span for a chain or tail merge.
                continue

        if (
            index > 0
            and retained[index - 1].reason is ChunkBoundaryReason.SECTION
        ):
            previous_start = spans[index - 1][0]
            if span_tokens(previous_start, current_end) <= maximum:
                spans[index - 1][1] = current_end
                spans.pop(index)
                retained.pop(index - 1)
                index -= 1
                # Re-check the merged span against its next neighbour.
                continue

        # No legal SECTION merge at this position.  Advancing guarantees
        # termination even when every neighbour is a hard or oversized cut.
        index += 1

    return tuple(retained)


def _select_region(
    units: tuple[SemanticUnit, ...],
    scores: tuple[int | None, ...],
    start: int,
    end: int,
) -> list[ChunkBoundary]:
    if start >= end:
        return []
    maximum = _config_int("max_chunk_tokens")
    minimum = _config_int("min_chunk_tokens")
    target = _config_int("target_chunk_tokens")
    region_tokens = count_chunk_tokens(_joined(units, start, end))
    if region_tokens <= maximum:
        return []
    region_is_small = region_tokens < minimum

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
            if tokens > maximum:
                break
            prior_state = states.get(prior)
            if prior_state is None:
                continue
            if tokens < minimum and not region_is_small:
                continue
            penalty = _size_penalty(tokens)
            reward = prior_state[0] - penalty
            if position < end:
                reward += gains.get(position - 1, 0)
            path = (*prior_state[3], position - 1) if position < end else prior_state[3]
            candidate = (
                reward,
                -abs(tokens - target),
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
    return joined_units(units[start:end])


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


def requires_semantic_vectors(units: tuple[SemanticUnit, ...]) -> bool:
    """Only oversized hard regions consult distances in the current planner."""
    start = 0
    for end in range(1, len(units) + 1):
        if end == len(units) or units[end].hard_boundary_before is not None:
            if count_chunk_tokens(joined_units(units[start:end])) > _config_int("max_chunk_tokens"):
                return True
            start = end
    return False
