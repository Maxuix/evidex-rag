from __future__ import annotations

import unittest
from dataclasses import replace
from uuid import uuid4

from rag_kb.document_processing import (
    SEMANTIC_CHUNKING_CONFIG,
    UNSTRUCTURED_CHUNKING_CONFIG,
    profile_fingerprint,
    profile_for_preset,
    public_descriptor,
    resolve,
)
from rag_kb.document_processing.semantic_assembly import assemble_semantic_document
from rag_kb.document_processing.semantic_boundaries import (
    build_chunk_plan,
    smoothed_distances,
    validate_plan,
)
from rag_kb.document_processing.semantic_units import (
    semantic_units,
    unit_sequence_hash,
)
from rag_kb.domain import (
    ChunkBoundaryReason,
    ChunkingPreset,
    ChunkingStrategyKind,
    ErrorCode,
    IndexingExecutionError,
    ParsedDocument,
    ParsedElement,
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
        semantic.chunking_config["max_chunk_tokens"] = 799
        with self.assertRaises(ValueError):
            resolve(semantic.parser_config, semantic.chunking_config)
        self.assertEqual(SEMANTIC_CHUNKING_CONFIG["max_chunk_tokens"], 800)
        self.assertNotIn("preset", UNSTRUCTURED_CHUNKING_CONFIG)

    def test_retired_semantic_profile_is_described_but_never_executable(self) -> None:
        legacy = {"profile": "unstructured_title_semantic_qwen_v1"}

        self.assertEqual(
            public_descriptor(legacy),
            {
                "preset": "legacy_incompatible",
                "profile": "unstructured_title_semantic_qwen_v1",
            },
        )
        with self.assertRaises(ValueError):
            resolve(
                profile_for_preset(
                    ChunkingPreset.STRUCTURAL_BALANCED_V2
                ).parser_config,
                legacy,
            )


class SemanticUnitTests(unittest.TestCase):
    def test_title_page_table_and_sentence_rules_are_deterministic(self) -> None:
        document = ParsedDocument(
            elements=(
                _element(0, "概览", category="Title", page=1),
                _element(1, "第一句。 Second sentence! 第三句；", page=1),
                _element(2, "page two body", page=2),
                _element(3, "A | B\n1 | 2", category="Table", page=2),
            ),
            extracted_character_count=50,
        )

        first = semantic_units(document)
        second = semantic_units(document)

        self.assertEqual(first, second)
        self.assertEqual(
            [unit.ordinal for unit in first],
            list(range(len(first))),
        )
        self.assertIn("概览", first[0].text)
        self.assertEqual(first[0].hard_boundary_before, "section")
        self.assertIn("page", {unit.hard_boundary_before for unit in first})
        self.assertIn("table", {unit.hard_boundary_before for unit in first})
        self.assertEqual(unit_sequence_hash(first), unit_sequence_hash(second))
        self.assertNotEqual(
            unit_sequence_hash(first),
            unit_sequence_hash(
                tuple(
                    replace(unit, text=f"{unit.text} changed")
                    if unit.ordinal == 0
                    else unit
                    for unit in first
                )
            ),
        )

    def test_oversized_sentence_is_split_into_bounded_overlapping_units(self) -> None:
        units = semantic_units(
            ParsedDocument(
                elements=(_element(0, "token " * 500),),
                extracted_character_count=3000,
            )
        )

        self.assertGreater(len(units), 1)
        self.assertTrue(
            all(
                unit.token_count
                <= SEMANTIC_CHUNKING_CONFIG["analysis_unit_max_tokens"]
                for unit in units
            )
        )

    def test_sequence_hash_includes_location_hierarchy_and_boundary(self) -> None:
        original = _unit(0, "stable text")
        mutations = (
            replace(original, source_location={"page_start": 2, "page_end": 2}),
            replace(original, hierarchy={"titles": [{"depth": 0, "text": "T"}]}),
            replace(original, hard_boundary_before="page"),
        )

        for changed in mutations:
            with self.subTest(changed=changed):
                self.assertNotEqual(
                    unit_sequence_hash((original,)),
                    unit_sequence_hash((changed,)),
                )


class SemanticBoundaryTests(unittest.TestCase):
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
        )

        self.assertEqual(plan.boundaries, ())
        self.assertEqual(plan.chunk_count, 1)
        document = assemble_semantic_document(units, plan)
        self.assertEqual(document.chunks[0].text, "short evidence")
        self.assertLessEqual(document.chunks[0].token_count, 800)

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
        )
        document = assemble_semantic_document(units, plan)

        self.assertTrue(all(value is None or isinstance(value, int) for value in scores))
        self.assertTrue(
            any(
                boundary.reason is ChunkBoundaryReason.SEMANTIC
                for boundary in plan.boundaries
            )
        )
        self.assertEqual(
            [chunk.ordinal for chunk in document.chunks],
            list(range(plan.chunk_count)),
        )
        self.assertTrue(all(0 < chunk.token_count <= 800 for chunk in document.chunks))

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
        )
        with self.assertRaises(IndexingExecutionError) as raised:
            validate_plan(
                plan,
                indexed_document_version_id=target,
                source_checksum_sha256="c" * 64,
                profile_fingerprint=fingerprint,
                units=(replace(units[0], text="changed"), units[1]),
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
        )
        document = assemble_semantic_document(units, first)

        self.assertLess(first.chunk_count, len(units))
        self.assertTrue(
            all(220 <= chunk.token_count <= 800 for chunk in document.chunks)
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
        )
        document = assemble_semantic_document(units, plan)
        self.assertEqual(plan.boundaries[0].reason, ChunkBoundaryReason.PAGE)
        self.assertTrue(all(chunk.token_count < 220 for chunk in document.chunks))


def _element(
    ordinal: int,
    text: str,
    *,
    category: str = "NarrativeText",
    page: int | None = None,
) -> ParsedElement:
    return ParsedElement(
        ordinal=ordinal,
        text=text,
        token_count=0,
        category=category,
        source_location=(
            {"page_start": page, "page_end": page}
            if page is not None
            else {}
        ),
        hierarchy={},
        is_title=category == "Title",
        is_table=category == "Table",
    )


def _unit(ordinal: int, text: str) -> SemanticUnit:
    from rag_kb.document_processing import count_chunk_tokens

    return SemanticUnit(
        ordinal=ordinal,
        text=text.strip(),
        token_count=count_chunk_tokens(text.strip()),
        source_location={},
        hierarchy={},
        element_ordinals=(ordinal,),
        hard_boundary_before=None,
    )


if __name__ == "__main__":
    unittest.main()
