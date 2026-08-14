from __future__ import annotations

import json
import unittest
from uuid import UUID

from rag_kb.domain import (
    Evidence,
    EvidenceScoreKind,
    GraphChunkResultStatus,
    GraphProtocolError,
    GraphResourceLimitError,
    RerankMode,
    allowed_graph_skips,
    entity_key,
    first_grounded_span,
    normalize_entity_surface,
    normalize_predicate,
)
from rag_kb.graph.extraction import (
    graph_protocol_error_family,
    graph_protocol_error_is_repairable,
    parse_graph_extraction,
)
from rag_kb.retrieval.eligibility import EvidenceEligibilityPolicy


class GraphExtractionUnitTests(unittest.TestCase):
    def test_normalization_and_identity_are_deterministic(self) -> None:
        self.assertEqual(normalize_entity_surface("  ACME—  Corp  "), "acme-corp")
        self.assertEqual(normalize_predicate("  Owns—  "), "owns")
        self.assertEqual(
            entity_key("organization", "ACME Corp"),
            entity_key("organization", " acme  corp "),
        )
        self.assertNotEqual(
            entity_key("organization", "ACME Corp"),
            entity_key("organization", "ACME Corp", "US subsidiary"),
        )

    def test_relation_is_grounded_and_duplicate_predicate_is_deduplicated(self) -> None:
        text = "Acme acquired Beta. Acme acquired Beta."
        value = parse_graph_extraction(
            {
                "entities": [
                    _entity("acme", "organization", "Acme"),
                    _entity("beta", "organization", "Beta"),
                ],
                "relations": [
                    {
                        "subject": "acme",
                        "predicate": "acquired",
                        "object": "beta",
                        "support": "Acme acquired Beta",
                    },
                    {
                        "subject": "acme",
                        "predicate": "acquired",
                        "object": "beta",
                        "support": "Acme acquired Beta",
                    },
                ],
            },
            text,
        )
        self.assertIs(value.result_status, GraphChunkResultStatus.EXTRACTED)
        self.assertEqual(len(value.mentions), 2)
        self.assertEqual(len(value.relations), 1)
        self.assertEqual(value.relations[0].support_start, 0)

    def test_grounded_span_maps_display_equivalence_to_original_offsets(self) -> None:
        text = "Prefix ACME\u00a0\n  Corp — ‘Orion’ suffix"
        start, end = first_grounded_span(text, 'acme corp - \'orion\'')
        self.assertEqual(text[start:end], "ACME\u00a0\n  Corp — ‘Orion’")

    def test_nfkc_casefold_expansions_require_complete_source_tokens(self) -> None:
        for source, full, partial in (
            ("ﬁ", "fi", "f"),
            ("ß", "ss", "s"),
            ("㍿", "株式会社", "株式"),
        ):
            with self.subTest(source=source):
                self.assertEqual(first_grounded_span(source, full), (0, 1))
                with self.assertRaises(GraphProtocolError):
                    first_grounded_span(source, partial)

    def test_combining_sequence_maps_as_one_original_token(self) -> None:
        text = "Cafe\u0301 launched"
        start, end = first_grounded_span(text, "CAFÉ")
        self.assertEqual(text[start:end], "Cafe\u0301")

    def test_mentions_materialize_original_surface_and_relation_support(self) -> None:
        text = "Acme and Beta were named. Later, ACME\n acquired  BETA."
        value = parse_graph_extraction(
            {
                "entities": [
                    _entity("a", "organization", "acme"),
                    _entity("b", "organization", "BETA"),
                ],
                "relations": [
                    {
                        "subject": "a",
                        "predicate": "acquired",
                        "object": "b",
                        "support": "acme acquired beta",
                    }
                ],
            },
            text,
        )
        self.assertEqual([item.surface for item in value.mentions], ["Acme", "Beta"])
        relation = value.relations[0]
        self.assertEqual(text[relation.support_start:relation.support_end], "ACME\n acquired  BETA")

    def test_relation_without_both_literal_normalized_endpoints_is_dropped(self) -> None:
        value = parse_graph_extraction(
            {
                "entities": [
                    _entity("a", "organization", "Acme"),
                    _entity("b", "organization", "Beta"),
                ],
                "relations": [
                    {
                        "subject": "a",
                        "predicate": "acquired",
                        "object": "b",
                        "support": "Acme acquired it",
                    }
                ],
            },
            "Acme acquired it. Beta objected.",
        )
        self.assertIs(value.result_status, GraphChunkResultStatus.EXTRACTED)
        self.assertEqual([item.surface for item in value.mentions], ["Acme", "Beta"])
        self.assertEqual(value.relations, ())

    def test_quote_variants_share_entity_identity(self) -> None:
        self.assertEqual(
            entity_key("product", "‘Orion’"),
            entity_key("product", "'Orion'"),
        )

    def test_disambiguator_requires_locatable_support(self) -> None:
        with self.assertRaises(GraphProtocolError):
            parse_graph_extraction(
                {
                    "entities": [
                        {
                            "id": "mercury",
                            "type": "product",
                            "surface": "Mercury",
                            "disambiguator": "planet",
                            "disambiguator_support": None,
                        }
                    ],
                    "relations": [],
                },
                "Mercury is visible.",
            )

    def test_empty_is_distinct_from_protocol_and_resource_failures(self) -> None:
        empty = parse_graph_extraction({"entities": [], "relations": []}, "2026 12 31")
        self.assertIs(empty.result_status, GraphChunkResultStatus.EMPTY)
        with self.assertRaises(GraphProtocolError):
            parse_graph_extraction("{bad", "A sentence")
        with self.assertRaises(GraphResourceLimitError):
            parse_graph_extraction(
                {"entities": [{"id": "x", "type": "concept", "surface": "x"}]},
                "x" * 32_001,
            )

    def test_schema_errors_use_content_safe_detail_codes_and_stable_families(self) -> None:
        cases = (
            ({"entities": [], "relations": [], "extra": "secret-a"}, "schema_unexpected_key"),
            ({"entities": []}, "schema_missing"),
            ({"entities": {}, "relations": []}, "schema_type"),
            (
                {"entities": [_entity("x", "animal", "x")], "relations": []},
                "schema_entity_enum",
            ),
            (
                {
                    "entities": [
                        _entity("bad id", "concept", "x")
                    ],
                    "relations": [],
                },
                "schema_id_format",
            ),
            (
                {"entities": [_entity("x" * 65, "concept", "x")], "relations": []},
                "schema_length",
            ),
            (
                {
                    "entities": [
                        {
                            **_entity("x", "concept", "x"),
                            "disambiguator": "kind",
                            "disambiguator_support": None,
                        }
                    ],
                    "relations": [],
                },
                "schema_null_contract",
            ),
        )
        for payload, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(GraphProtocolError) as raised:
                    parse_graph_extraction(payload, "x")
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(graph_protocol_error_family(code), "schema_invalid")

        for secret in ("secret-a", "different-secret-value"):
            with self.subTest(secret=secret):
                with self.assertRaises(GraphProtocolError) as raised:
                    parse_graph_extraction(
                        {"entities": [], "relations": [], "extra": secret}, "x"
                    )
                self.assertEqual(raised.exception.code, "schema_unexpected_key")

    def test_duplicate_entity_id_keeps_the_first_mention(self) -> None:
        value = parse_graph_extraction(
            {
                "entities": [
                    _entity("x", "concept", "x"),
                    _entity("x", "concept", "x"),
                ],
                "relations": [],
            },
            "x",
        )
        self.assertIs(value.result_status, GraphChunkResultStatus.EXTRACTED)
        self.assertEqual(len(value.mentions), 1)
        self.assertEqual(value.mentions[0].mention_id, "x")

    def test_grounding_detail_codes_keep_historical_families(self) -> None:
        cases = (
            (
                {"entities": [_entity("x", "concept", "Absent")], "relations": []},
                "entity_surface_not_locatable",
                "support_text_not_locatable",
            ),
            (
                {
                    "entities": [
                        {
                            **_entity("x", "concept", "Acme"),
                            "disambiguator": "company",
                            "disambiguator_support": "Absent",
                        }
                    ],
                    "relations": [],
                },
                "disambiguator_support_not_locatable",
                "support_text_not_locatable",
            ),
        )
        for payload, code, family in cases:
            with self.subTest(code=code):
                with self.assertRaises(GraphProtocolError) as raised:
                    parse_graph_extraction(payload, "Acme owns it. Beta exists.")
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(graph_protocol_error_family(code), family)

    def test_invalid_neighbor_items_do_not_discard_grounded_relations(self) -> None:
        value = parse_graph_extraction(
            {
                "entities": [
                    _entity("acme", "organization", "Acme"),
                    _entity("beta", "organization", "Beta"),
                    _entity("ghost", "organization", "MissingCo"),
                    _entity("bad id", "organization", "Gamma"),
                ],
                "relations": [
                    {
                        "subject": "acme",
                        "predicate": "acquired",
                        "object": "beta",
                        "support": "Acme acquired Beta",
                    },
                    {
                        "subject": "acme",
                        "predicate": "owns",
                        "object": "beta",
                        "support": "Acme owns it",
                    },
                    {
                        "subject": "acme",
                        "predicate": "mentions",
                        "object": "ghost",
                        "support": "Gamma exists",
                    },
                ],
            },
            "Acme acquired Beta. Gamma exists.",
        )
        self.assertIs(value.result_status, GraphChunkResultStatus.EXTRACTED)
        self.assertEqual([item.surface for item in value.mentions], ["Acme", "Beta"])
        self.assertEqual(len(value.relations), 1)
        self.assertEqual(value.relations[0].predicate, "acquired")
        self.assertEqual(value.admission.dropped_entity_count, 2)
        self.assertEqual(value.admission.dropped_relation_count, 2)
        self.assertEqual(value.admission.dropped_relation_grounding_count, 1)
        self.assertEqual(
            value.admission.relation_drop_codes,
            (("relation_endpoint_unknown", 1), ("relation_support_not_locatable", 1)),
        )
        self.assertEqual(
            value.mentions[0].surface,
            "Acme acquired Beta. Gamma exists."[
                value.mentions[0].surface_start:value.mentions[0].surface_end
            ],
        )

    def test_relation_support_missing_endpoints_are_dropped_not_chunk_failures(self) -> None:
        for support, code in (
            ("it owns Beta", "relation_support_missing_subject"),
            ("Acme owns it", "relation_support_missing_object"),
            ("They transact", "relation_support_missing_both"),
        ):
            with self.subTest(code=code):
                text = f"Acme and Beta exist. {support}."
                value = parse_graph_extraction(
                    {
                        "entities": [
                            _entity("a", "organization", "Acme"),
                            _entity("b", "organization", "Beta"),
                        ],
                        "relations": [
                            {
                                "subject": "a",
                                "predicate": "owns",
                                "object": "b",
                                "support": support,
                            }
                        ],
                    },
                    text,
                )
                self.assertIs(value.result_status, GraphChunkResultStatus.EXTRACTED)
                self.assertEqual(len(value.mentions), 2)
                self.assertEqual(value.relations, ())
                self.assertEqual(value.admission.dropped_relation_count, 1)
                self.assertEqual(value.admission.dropped_relation_grounding_count, 1)
                self.assertEqual(value.admission.relation_drop_codes, ((code, 1),))
                self.assertEqual(
                    graph_protocol_error_family(code),
                    "relation_support_missing_endpoint",
                )
                self.assertFalse(graph_protocol_error_is_repairable(code))

    def test_unlocatable_relation_support_is_dropped_when_mentions_remain(self) -> None:
        value = parse_graph_extraction(
            {
                "entities": [
                    _entity("a", "organization", "Acme"),
                    _entity("b", "organization", "Beta"),
                ],
                "relations": [
                    {
                        "subject": "a",
                        "predicate": "owns",
                        "object": "b",
                        "support": "Absent",
                    }
                ],
            },
            "Acme owns it. Beta exists.",
        )
        self.assertIs(value.result_status, GraphChunkResultStatus.EXTRACTED)
        self.assertEqual(len(value.mentions), 2)
        self.assertEqual(value.relations, ())
        self.assertEqual(value.admission.dropped_relation_count, 1)
        self.assertEqual(value.admission.dropped_relation_grounding_count, 1)
        self.assertEqual(
            value.admission.relation_drop_codes,
            (("relation_support_not_locatable", 1),),
        )

    def test_empty_payload_has_zero_admission_drops(self) -> None:
        value = parse_graph_extraction({"entities": [], "relations": []}, "2026 12 31")
        self.assertEqual(value.admission.dropped_entity_count, 0)
        self.assertEqual(value.admission.dropped_relation_count, 0)
        self.assertEqual(value.admission.dropped_relation_grounding_count, 0)
        self.assertEqual(value.admission.relation_drop_codes, ())

    def test_all_rejected_items_preserve_admission_counts_on_the_error(self) -> None:
        with self.assertRaises(GraphProtocolError) as raised:
            parse_graph_extraction(
                {"entities": [_entity("x", "concept", "Absent")], "relations": []},
                "Nothing matches.",
            )
        self.assertEqual(raised.exception.code, "entity_surface_not_locatable")
        self.assertEqual(raised.exception.admission.dropped_entity_count, 1)
        self.assertEqual(raised.exception.admission.dropped_relation_count, 0)
        self.assertEqual(
            raised.exception.admission.entity_drop_codes,
            (("entity_surface_not_locatable", 1),),
        )

    def test_only_root_schema_failures_are_repairable(self) -> None:
        self.assertTrue(graph_protocol_error_is_repairable("json_invalid"))
        self.assertTrue(graph_protocol_error_is_repairable("schema_type"))
        self.assertFalse(graph_protocol_error_is_repairable("entity_surface_not_locatable"))
        self.assertFalse(graph_protocol_error_is_repairable("relation_support_missing_object"))
        self.assertFalse(graph_protocol_error_is_repairable("schema_id_format"))

    def test_wide_table_fixture_can_legitimately_return_empty(self) -> None:
        text = " | ".join(f"C{i}" for i in range(128)) + "\n" + " | ".join("123" for _ in range(128))
        value = parse_graph_extraction({"entities": [], "relations": []}, text)
        self.assertIs(value.result_status, GraphChunkResultStatus.EMPTY)

    def test_graph_path_eligibility_is_not_cosine_gated(self) -> None:
        evidence = Evidence(
            rank=1,
            index_chunk_id=UUID("01900000-0000-7000-8000-000000000001"),
            indexed_document_version_id=UUID("01900000-0000-7000-8000-000000000002"),
            document_id=UUID("01900000-0000-7000-8000-000000000003"),
            document_version_id=UUID("01900000-0000-7000-8000-000000000004"),
            index_revision_id=UUID("01900000-0000-7000-8000-000000000005"),
            ordinal=0,
            text="Acme acquired Beta.",
            source_location={},
            hierarchy={},
            source_metadata={},
            score=1.0,
            score_kind=EvidenceScoreKind.GRAPH_PATH,
            graph_path_id="path-1",
            graph_anchor_index_chunk_id=UUID("01900000-0000-7000-8000-000000000001"),
            graph_hop_count=1,
            graph_path_rank=1,
        )
        self.assertTrue(EvidenceEligibilityPolicy(0.95).usable(evidence))

    def test_adjacency_still_rejected(self) -> None:
        self.assertFalse(
            EvidenceEligibilityPolicy(0.0).usable(
                Evidence(
                    rank=1,
                    index_chunk_id=UUID("01900000-0000-7000-8000-000000000011"),
                    indexed_document_version_id=UUID("01900000-0000-7000-8000-000000000012"),
                    document_id=UUID("01900000-0000-7000-8000-000000000013"),
                    document_version_id=UUID("01900000-0000-7000-8000-000000000014"),
                    index_revision_id=UUID("01900000-0000-7000-8000-000000000015"),
                    ordinal=1,
                    text="neighbor",
                    source_location={},
                    hierarchy={},
                    source_metadata={},
                    score=0.0,
                    score_kind=EvidenceScoreKind.ADJACENCY,
                    adjacency_anchor_index_chunk_id=UUID("01900000-0000-7000-8000-000000000016"),
                    adjacency_offset=1,
                )
            )
        )

    def test_skip_budget_is_fixed(self) -> None:
        self.assertEqual(allowed_graph_skips(0), 0)
        self.assertEqual(allowed_graph_skips(1), 1)
        self.assertEqual(allowed_graph_skips(20), 1)
        self.assertEqual(allowed_graph_skips(100), 5)


def _entity(identifier: str, entity_type: str, surface: str) -> dict[str, object]:
    return {
        "id": identifier,
        "type": entity_type,
        "surface": surface,
        "disambiguator": None,
        "disambiguator_support": None,
    }


if __name__ == "__main__":
    unittest.main()
