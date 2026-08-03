from __future__ import annotations

import unittest
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
from uuid import UUID, uuid4

from rag_kb.document_processing.profiles import (
    SEMANTIC_CHUNKING_CONFIG,
    profile_fingerprint,
    profile_for_preset,
    public_descriptor,
    resolve,
)
from rag_kb.document_processing.tokenization import count_chunk_tokens
import rag_kb.document_processing.semantic_boundaries as semantic_boundaries
from rag_kb.document_processing.semantic_boundaries import (
    build_chunk_plan,
    smoothed_distances,
    validate_plan,
)
from rag_kb.document_processing.docling.semantic import docling_unit_sequence_hash
from rag_kb.domain import (
    ChunkBoundaryReason,
    ChunkingPreset,
    ChunkingStrategyKind,
    ErrorCode,
    IndexingExecutionError,
    SemanticUnit,
)


class SemanticProfileTests(unittest.TestCase):
    def test_registry_resolves_only_complete_exact_profiles(self) -> None:
        structural = profile_for_preset(ChunkingPreset.STRUCTURAL_BALANCED_V2)
        semantic = profile_for_preset(ChunkingPreset.SEMANTIC_BALANCED_V1)

        self.assertEqual(
            resolve(structural.parser_config, structural.chunking_config),
            ChunkingStrategyKind.STRUCTURAL,
        )
        self.assertEqual(
            resolve(semantic.parser_config, semantic.chunking_config),
            ChunkingStrategyKind.SEMANTIC,
        )
        self.assertEqual(
            public_descriptor(structural.chunking_config)["preset"],
            "structural_balanced_v2",
        )
        self.assertEqual(
            semantic.chunking_config["profile"], "semantic_breakpoint_v2"
        )
        self.assertEqual(
            semantic.chunking_config["selector"],
            "constrained_dp_section_merge_v2",
        )
        self.assertEqual(
            semantic.chunking_config["section_boundary_policy"],
            "merge_under_min_adjacent_v1",
        )
        semantic.chunking_config["max_chunk_tokens"] = 799
        with self.assertRaises(ValueError):
            resolve(semantic.parser_config, semantic.chunking_config)
        self.assertEqual(SEMANTIC_CHUNKING_CONFIG["max_chunk_tokens"], 800)

    def test_unknown_semantic_profile_is_rejected(self) -> None:
        unknown = {"profile": "unknown_semantic_profile"}

        with self.assertRaises(ValueError):
            public_descriptor(unknown)
        with self.assertRaises(ValueError):
            resolve(
                profile_for_preset(
                    ChunkingPreset.STRUCTURAL_BALANCED_V2
                ).parser_config,
                unknown,
            )


class SemanticBoundaryTests(unittest.TestCase):
    def test_region_size_is_tokenized_once_outside_dynamic_programming(self) -> None:
        units = tuple(_unit(index, "bounded " * 80) for index in range(40))
        scores = tuple(0 for _ in range(len(units) - 1))
        full_region = "\n\n".join(unit.text for unit in units)
        full_region_calls = 0

        def counted(text: str) -> int:
            nonlocal full_region_calls
            if text == full_region:
                full_region_calls += 1
            return count_chunk_tokens(text)

        with patch.object(
            semantic_boundaries,
            "count_chunk_tokens",
            side_effect=counted,
        ):
            boundaries = semantic_boundaries._select_region(
                units,
                scores,
                0,
                len(units),
            )

        self.assertTrue(boundaries)
        self.assertEqual(full_region_calls, 1)

    def test_planner_optimization_has_stable_v2_plan_identity(self) -> None:
        units = tuple(
            _unit(index, ("alpha " if index < 4 else "beta ") * 120)
            for index in range(8)
        )
        vectors = tuple(
            (1.0, 0.0) if index < 4 else (0.0, 1.0)
            for index in range(8)
        )
        profile = profile_for_preset(ChunkingPreset.SEMANTIC_BALANCED_V1)

        plan = build_chunk_plan(
            indexed_document_version_id=UUID(
                "00000000-0000-0000-0000-000000000001"
            ),
            source_checksum_sha256="b" * 64,
            profile_fingerprint=profile_fingerprint(
                profile.parser_config,
                profile.chunking_config,
            ),
            units=units,
            vectors=vectors,
            sequence_hash=docling_unit_sequence_hash(units),
        )

        self.assertEqual(
            [
                (
                    boundary.after_unit_ordinal,
                    boundary.reason.value,
                    boundary.score_micros,
                )
                for boundary in plan.boundaries
            ],
            [(3, "semantic", 500000)],
        )
        self.assertEqual(
            plan.plan_hash,
            "26ff4e19d42223ebe7ff2c2cf4b5a2652082348b002967a08dc86d5d7a72ad67",
        )

    def test_no_section_boundaries_are_unchanged_but_profile_hash_changes(self) -> None:
        units = tuple(
            _unit(index, ("alpha " if index < 4 else "beta ") * 120)
            for index in range(8)
        )
        vectors = tuple(
            (1.0, 0.0) if index < 4 else (0.0, 1.0)
            for index in range(8)
        )
        semantic = profile_for_preset(ChunkingPreset.SEMANTIC_BALANCED_V1)
        old_chunking = deepcopy(semantic.chunking_config)
        old_chunking["profile"] = "semantic_breakpoint_v1"
        old_chunking["selector"] = "constrained_dp_v1"
        old_chunking.pop("section_boundary_policy", None)
        sequence_hash = docling_unit_sequence_hash(units)
        target = UUID("00000000-0000-0000-0000-000000000002")
        current = build_chunk_plan(
            indexed_document_version_id=target,
            source_checksum_sha256="b" * 64,
            profile_fingerprint=profile_fingerprint(
                semantic.parser_config, semantic.chunking_config
            ),
            units=units,
            vectors=vectors,
            sequence_hash=sequence_hash,
        )
        previous = build_chunk_plan(
            indexed_document_version_id=target,
            source_checksum_sha256="b" * 64,
            profile_fingerprint=profile_fingerprint(
                semantic.parser_config, old_chunking
            ),
            units=units,
            vectors=vectors,
            sequence_hash=sequence_hash,
        )
        self.assertEqual(current.boundaries, previous.boundaries)
        self.assertNotEqual(current.plan_hash, previous.plan_hash)

    def test_section_only_merge_matrix(self) -> None:
        units = tuple(
            _unit(index, _words(size))
            for index, size in enumerate((100, 100, 100, 100))
        )
        boundaries = tuple(
            semantic_boundaries.ChunkBoundary(index, ChunkBoundaryReason.SECTION)
            for index in range(3)
        )
        merged = semantic_boundaries._merge_small_section_chunks(units, boundaries)
        self.assertEqual(merged, ())
        text = "\n\n".join(unit.text for unit in units)
        self.assertLessEqual(count_chunk_tokens(text), 800)

    def test_small_tail_merges_with_previous_and_chain_is_rechecked(self) -> None:
        tail_units = tuple(
            _unit(index, _words(size)) for index, size in enumerate((300, 100))
        )
        tail_boundaries = (
            semantic_boundaries.ChunkBoundary(0, ChunkBoundaryReason.SECTION),
        )
        self.assertEqual(
            semantic_boundaries._merge_small_section_chunks(
                tail_units, tail_boundaries
            ),
            (),
        )

        chain_units = tuple(
            _unit(index, _words(size))
            for index, size in enumerate((50, 50, 100, 700))
        )
        chain_boundaries = tuple(
            semantic_boundaries.ChunkBoundary(index, ChunkBoundaryReason.SECTION)
            for index in range(3)
        )
        result = semantic_boundaries._merge_small_section_chunks(
            chain_units, chain_boundaries
        )
        self.assertEqual(
            tuple(item.after_unit_ordinal for item in result),
            (2,),
        )

    def test_next_first_tie_break_is_deterministic(self) -> None:
        units = tuple(
            _unit(index, _words(size)) for index, size in enumerate((300, 100, 300))
        )
        boundaries = tuple(
            semantic_boundaries.ChunkBoundary(index, ChunkBoundaryReason.SECTION)
            for index in range(2)
        )
        result = semantic_boundaries._merge_small_section_chunks(units, boundaries)
        self.assertEqual(
            tuple(item.after_unit_ordinal for item in result),
            (0,),
        )

    def test_oversized_residual_and_non_section_boundaries_are_preserved(self) -> None:
        oversized = tuple(
            _unit(index, _words(size)) for index, size in enumerate((700, 100, 700))
        )
        oversized_boundaries = (
            semantic_boundaries.ChunkBoundary(0, ChunkBoundaryReason.SECTION),
            semantic_boundaries.ChunkBoundary(1, ChunkBoundaryReason.SECTION),
        )
        result = semantic_boundaries._merge_small_section_chunks(
            oversized, oversized_boundaries
        )
        self.assertEqual(result, oversized_boundaries)

        reasons = (
            ChunkBoundaryReason.PAGE,
            ChunkBoundaryReason.TABLE,
            ChunkBoundaryReason.BLOCK,
            ChunkBoundaryReason.SEMANTIC,
            ChunkBoundaryReason.MAX_TOKENS,
        )
        protected_units = tuple(_unit(index, _words(10)) for index in range(6))
        protected = tuple(
            semantic_boundaries.ChunkBoundary(
                index,
                reason,
                1 if reason is ChunkBoundaryReason.SEMANTIC else None,
            )
            for index, reason in enumerate(reasons)
        )
        self.assertEqual(
            semantic_boundaries._merge_small_section_chunks(
                protected_units, protected
            ),
            protected,
        )

    def test_repeated_v2_planning_is_deterministic(self) -> None:
        units = tuple(_unit(index, _words(100)) for index in range(5))
        vectors = tuple((1.0, 0.0) for _ in units)
        profile = profile_for_preset(ChunkingPreset.SEMANTIC_BALANCED_V1)
        kwargs = {
            "indexed_document_version_id": UUID(
                "00000000-0000-0000-0000-000000000003"
            ),
            "source_checksum_sha256": "f" * 64,
            "profile_fingerprint": profile_fingerprint(
                profile.parser_config, profile.chunking_config
            ),
            "units": units,
            "vectors": vectors,
            "sequence_hash": docling_unit_sequence_hash(units),
        }
        first = build_chunk_plan(**kwargs)
        second = build_chunk_plan(**kwargs)
        self.assertEqual(first.boundaries, second.boundaries)
        self.assertEqual(first.plan_hash, second.plan_hash)

    def test_short_document_skips_vectors_and_assembles_one_chunk(self) -> None:
        units = (_unit(0, "short evidence"),)
        profile = profile_for_preset(ChunkingPreset.SEMANTIC_BALANCED_V1)
        plan = build_chunk_plan(
            indexed_document_version_id=uuid4(),
            source_checksum_sha256="a" * 64,
            profile_fingerprint=profile_fingerprint(
                profile.parser_config,
                profile.chunking_config,
            ),
            units=units,
            vectors=None,
            sequence_hash=docling_unit_sequence_hash(units),
        )

        self.assertEqual(plan.boundaries, ())
        self.assertEqual(plan.chunk_count, 1)
        texts = _cut(units, plan)
        self.assertEqual(texts[0], "short evidence")
        self.assertLessEqual(count_chunk_tokens(texts[0]), 800)

    def test_topic_change_uses_integer_scores_and_bounded_chunks(self) -> None:
        units = tuple(
            _unit(index, ("alpha " if index < 4 else "beta ") * 120)
            for index in range(8)
        )
        vectors = tuple(
            (1.0, 0.0) if index < 4 else (0.0, 1.0)
            for index in range(8)
        )
        scores = smoothed_distances(units, vectors)
        profile = profile_for_preset(ChunkingPreset.SEMANTIC_BALANCED_V1)
        plan = build_chunk_plan(
            indexed_document_version_id=uuid4(),
            source_checksum_sha256="b" * 64,
            profile_fingerprint=profile_fingerprint(
                profile.parser_config,
                profile.chunking_config,
            ),
            units=units,
            vectors=vectors,
            sequence_hash=docling_unit_sequence_hash(units),
        )
        texts = _cut(units, plan)

        self.assertTrue(all(value is None or isinstance(value, int) for value in scores))
        self.assertTrue(
            any(
                boundary.reason is ChunkBoundaryReason.SEMANTIC
                for boundary in plan.boundaries
            )
        )
        self.assertEqual(len(texts), plan.chunk_count)
        self.assertTrue(0 < count_chunk_tokens(text) <= 800 for text in texts)

    def test_plan_validation_detects_unit_drift(self) -> None:
        units = (_unit(0, "first"), _unit(1, "second"))
        profile = profile_for_preset(ChunkingPreset.SEMANTIC_BALANCED_V1)
        fingerprint = profile_fingerprint(
            profile.parser_config,
            profile.chunking_config,
        )
        target = uuid4()
        plan = build_chunk_plan(
            indexed_document_version_id=target,
            source_checksum_sha256="c" * 64,
            profile_fingerprint=fingerprint,
            units=units,
            vectors=None,
            sequence_hash=docling_unit_sequence_hash(units),
        )
        with self.assertRaises(IndexingExecutionError) as raised:
            validate_plan(
                plan,
                indexed_document_version_id=target,
                source_checksum_sha256="c" * 64,
                profile_fingerprint=fingerprint,
                units=(drifted := (replace(units[0], text="changed"), units[1])),
                sequence_hash=docling_unit_sequence_hash(drifted),
            )
        self.assertEqual(
            raised.exception.code,
            ErrorCode.INDEX_CHUNK_PLAN_MISMATCH,
        )

    def test_uniform_distances_choose_bounded_target_sized_chunks(self) -> None:
        units = tuple(_unit(index, "uniform " * 120) for index in range(10))
        vectors = tuple((1.0, 0.0) for _ in units)
        profile = profile_for_preset(ChunkingPreset.SEMANTIC_BALANCED_V1)

        first = build_chunk_plan(
            indexed_document_version_id=uuid4(),
            source_checksum_sha256="d" * 64,
            profile_fingerprint=profile_fingerprint(
                profile.parser_config, profile.chunking_config
            ),
            units=units,
            vectors=vectors,
            sequence_hash=docling_unit_sequence_hash(units),
        )
        texts = _cut(units, first)

        self.assertLess(first.chunk_count, len(units))
        self.assertTrue(
            all(220 <= count_chunk_tokens(text) <= 800 for text in texts)
        )

    def test_hard_boundaries_allow_small_regions_and_block_smoothing(self) -> None:
        units = (
            _unit(0, "left " * 30),
            _unit(1, "left " * 30),
            replace(_unit(2, "right " * 30), hard_boundary_before="page"),
            _unit(3, "right " * 30),
        )
        first_vectors = ((1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0))
        changed_right = ((1.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, -1.0))

        first_scores = smoothed_distances(units, first_vectors)
        changed_scores = smoothed_distances(units, changed_right)
        self.assertIsNone(first_scores[1])
        self.assertEqual(first_scores[0], changed_scores[0])

        profile = profile_for_preset(ChunkingPreset.SEMANTIC_BALANCED_V1)
        plan = build_chunk_plan(
            indexed_document_version_id=uuid4(),
            source_checksum_sha256="e" * 64,
            profile_fingerprint=profile_fingerprint(
                profile.parser_config, profile.chunking_config
            ),
            units=units,
            vectors=first_vectors,
            sequence_hash=docling_unit_sequence_hash(units),
        )
        texts = _cut(units, plan)
        self.assertEqual(plan.boundaries[0].reason, ChunkBoundaryReason.PAGE)
        self.assertTrue(all(count_chunk_tokens(text) < 220 for text in texts))


def _cut(units: tuple[SemanticUnit, ...], plan) -> list[str]:
    """Join each planned chunk's units the way assembly does."""

    cuts = (*[item.after_unit_ordinal + 1 for item in plan.boundaries], len(units))
    texts: list[str] = []
    start = 0
    for end in cuts:
        texts.append("\n\n".join(unit.text for unit in units[start:end]))
        start = end
    return texts


def _unit(ordinal: int, text: str) -> SemanticUnit:
    return SemanticUnit(
        ordinal=ordinal,
        text=text.strip(),
        token_count=count_chunk_tokens(text.strip()),
        item_refs=(f"#/texts/{ordinal}",),
        source_location={},
        hard_boundary_before=None,
    )


def _words(count: int) -> str:
    return " ".join("word" for _ in range(count))


if __name__ == "__main__":
    unittest.main()
