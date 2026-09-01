from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

from tools import build_enterprise_profile_qualification as enterprise
from tools import build_enterprise_profile_stress as enterprise_stress
from tools import build_musique_route_candidates as musique
from tools import build_public_rag_benchmark_suite as public


ROOT = Path(__file__).resolve().parents[2]


class CollectedRagEvaluationCorporaTests(unittest.TestCase):
    def test_public_quality_suite_has_actionable_policy_denominators(self) -> None:
        self.assertEqual(
            public.validate(public.DEFAULT_OUTPUT),
            {"case_count": 281, "document_count": 1498},
        )
        manifest = json.loads(
            (public.DEFAULT_OUTPUT / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["case_counts"]["open_world_false_premise"], 40)
        self.assertEqual(manifest["case_counts"]["unanswerable"], 30)
        self.assertEqual(manifest["case_counts"]["info_not_found"], 20)
        self.assertEqual(manifest["case_counts"]["conflicting_info"], 20)
        self.assertEqual(manifest["case_counts"]["evidence_no-conflict"], 24)

    def test_public_quality_suite_remote_image_correction_is_explicit(self) -> None:
        source = (
            "Before ![Action](https://example.com/action.png) after "
            "![Page](//example.com/page.png)"
        )
        corrected = public._apply_remote_image_correction(
            "ibmcld_16257-2972-4789",
            source,
        )
        self.assertEqual(corrected, "Before  after ")
        with self.assertRaises(public.CorpusError):
            public._apply_remote_image_correction(
                "ibmcld_16257-2972-4789",
                "No remote images",
            )

    def test_expanded_musique_pool_is_large_enough_for_host_qualification(self) -> None:
        self.assertEqual(
            musique.validate(musique.DEFAULT_OUTPUT),
            {"case_count": 76, "document_count": 578},
        )
        manifest = json.loads(
            (musique.DEFAULT_OUTPUT / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["case_counts"]["graph_needed_candidate"], 48)
        self.assertEqual(
            manifest["qualification"]["minimum_qualified_graph_needed_count"], 30
        )

    def test_enterprise_profile_calibrator_covers_every_relation_type(self) -> None:
        self.assertEqual(
            enterprise.validate(enterprise.DEFAULT_OUTPUT),
            {"relation_types": 42, "positive_relations": 126, "controls": 12},
        )
        manifest = json.loads(
            (enterprise.DEFAULT_OUTPUT / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["positive_examples_per_relation"], 3)
        self.assertEqual(manifest["relation_type_count"], 42)
        entities = [
            json.loads(line)
            for line in (enterprise.DEFAULT_OUTPUT / "entities.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        controls = [
            json.loads(line)
            for line in (enterprise.DEFAULT_OUTPUT / "negative_controls.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(manifest["gold_entity_count"], len(entities))
        self.assertTrue(all(len(row["aliases"]) == 1 for row in entities))
        self.assertFalse(any(re.search(r"\d+$", row["name"]) for row in entities))
        canonical_names = {row["name"] for row in entities}
        self.assertTrue(
            all(
                control[field] in canonical_names
                for control in controls
                for field in ("source_entity", "target_entity")
            )
        )

    def test_enterprise_profile_stress_corpus_targets_compound_failures(self) -> None:
        self.assertEqual(
            enterprise_stress.validate(enterprise_stress.DEFAULT_OUTPUT),
            {
                "controls": 6,
                "documents": 12,
                "identity_controls": 1,
                "relation_assertions": 34,
                "unique_relations": 30,
            },
        )
        manifest = json.loads(
            (enterprise_stress.DEFAULT_OUTPUT / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            manifest["family_counts"],
            {
                "cross_episode_identity": 4,
                "repeated_fact": 4,
                "semantic_overlap": 4,
            },
        )
        self.assertGreater(
            manifest["relation_assertion_count"],
            manifest["unique_gold_relation_count"],
        )

    def test_enterprise_graph_contract_matches_the_three_hop_domain_limit(self) -> None:
        manifest = json.loads(
            (ROOT / "evaluation/graph-rag-v1/manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["current_graph_contract"]["expected_online_max_hops"], 3)
        self.assertEqual(manifest["supported_case_count"], 24)
        self.assertEqual(manifest["stretch_case_count"], 0)


if __name__ == "__main__":
    unittest.main()
