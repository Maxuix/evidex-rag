from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from uuid import UUID

from tools import evaluate_open_source_rag_v3 as evaluator


def _completed_observations() -> dict:
    value = evaluator.observation_template()
    value["runtime_identity"] = {
        "workspace_id": str(UUID(int=1)),
        "knowledge_base_id": str(UUID(int=2)),
        "index_revision_id": str(UUID(int=3)),
        "graph_build_id": str(UUID(int=4)),
        "embedding_profile_revision_id": str(UUID(int=5)),
        "graph_chat_profile_revision_id": str(UUID(int=6)),
        "index_configuration_sha256": "a" * 64,
        "serving_document_set_sha256": "b" * 64,
    }
    cases = {row["case_id"]: row for row in evaluator._jsonl(evaluator.CASES_PATH)}
    for row in value["case_observations"]:
        case = cases[row["case_id"]]
        row["simple_relation_ids"] = []
        row["graph_relation_ids_by_layer"] = {
            layer: [] for layer in evaluator.LAYERS
        }
        row["auto"] = {
            "attempted": False,
            "admitted": False,
            "new_source_backed_evidence_count": 0,
        }
        row["actual_outcome"] = case["expected_outcome"]
        row["forbidden_claim_hit"] = False
    value["graph_extraction"] = {
        "extracted_relation_count": 0,
        "exact_matched_extracted_relation_count": 0,
        "exact_gold_relation_ids": [],
        "topology_gold_relation_ids": [],
        "self_loop_count": 0,
    }
    return value


class OpenSourceRagV3EvaluationTests(unittest.TestCase):
    def test_template_locks_semantic_v4_profiles_and_corpus_identity(self) -> None:
        template = evaluator.observation_template()

        self.assertEqual(
            template["configuration"]["chunking_profile"],
            "semantic_breakpoint_v4",
        )
        self.assertEqual(template["configuration"]["simple"]["top_k"], 10)
        self.assertEqual(template["configuration"]["graph"]["edge_limit"], 8)
        self.assertEqual(len(template["case_observations"]), 28)
        self.assertIsNone(template["runtime_identity"])

    def test_graph_needed_and_benefit_require_one_complete_valid_path(self) -> None:
        value = _completed_observations()
        row = value["case_observations"][0]
        row["simple_relation_ids"] = ["OSR002", "OSR007"]
        row["graph_relation_ids_by_layer"] = {
            "raw": ["OSR008"],
            "hydrated": ["OSR008"],
            "reranked": ["OSR008"],
            "packed": ["OSR008"],
        }
        row["auto"] = {
            "attempted": True,
            "admitted": True,
            "new_source_backed_evidence_count": 1,
        }

        result = evaluator.evaluate(value)
        first = result["case_labels"][0]

        self.assertTrue(first["graph_needed"])
        self.assertTrue(first["graph_benefit"])
        self.assertFalse(first["simple"]["complete"])
        self.assertFalse(first["graph_only"]["packed"]["complete"])
        self.assertTrue(first["augmented"]["packed"]["complete"])
        self.assertEqual(
            first["incremental"]["packed"]["new_valid_path_relation_count"], 1
        )
        self.assertEqual(result["metrics"]["auto"]["true_positive"], 1)

    def test_graph_only_recall_never_inherits_simple_relations(self) -> None:
        value = _completed_observations()
        row = value["case_observations"][0]
        row["simple_relation_ids"] = ["OSR002", "OSR007", "OSR008"]
        row["graph_relation_ids_by_layer"] = {
            layer: ["OSR023"] for layer in evaluator.LAYERS
        }

        result = evaluator.evaluate(value)
        first = result["case_labels"][0]

        self.assertEqual(
            result["schema_version"], "open_source_rag_v3_locked_evaluation_v2"
        )
        self.assertNotIn("graph_path_recall", result["metrics"])
        self.assertTrue(first["simple"]["complete"])
        self.assertFalse(first["graph_only"]["packed"]["complete"])
        self.assertTrue(first["augmented"]["packed"]["complete"])
        self.assertEqual(
            result["metrics"]["graph_only_path_recall"]["packed"]
            ["complete_path_rate"]["numerator"],
            0,
        )
        self.assertEqual(
            result["metrics"]["augmented_path_recall"]["packed"]
            ["complete_path_rate"]["numerator"],
            1,
        )

    def test_graph_benefit_requires_a_new_valid_path_relation(self) -> None:
        value = _completed_observations()
        row = value["case_observations"][0]
        row["simple_relation_ids"] = ["OSR002", "OSR007"]
        row["graph_relation_ids_by_layer"] = {
            layer: ["OSR023"] for layer in evaluator.LAYERS
        }

        result = evaluator.evaluate(value)
        first = result["case_labels"][0]

        self.assertTrue(first["graph_needed"])
        self.assertFalse(first["graph_benefit"])
        self.assertFalse(first["augmented"]["packed"]["complete"])
        self.assertEqual(
            first["incremental"]["packed"]["new_valid_path_relation_count"], 0
        )

    def test_semantic_graph_is_not_gold_when_simple_has_complete_path(self) -> None:
        value = _completed_observations()
        row = value["case_observations"][0]
        row["simple_relation_ids"] = ["OSR002", "OSR007", "OSR008"]
        row["auto"] = {
            "attempted": True,
            "admitted": True,
            "new_source_backed_evidence_count": 1,
        }

        result = evaluator.evaluate(value)
        first = result["case_labels"][0]

        self.assertFalse(first["graph_needed"])
        self.assertFalse(first["graph_benefit"])
        self.assertEqual(result["metrics"]["auto"]["false_positive"], 1)
        self.assertFalse(
            result["route_gold_policy"]["semantic_intent_is_primary_gold"]
        )

    def test_incomplete_graph_path_is_not_benefit(self) -> None:
        value = _completed_observations()
        row = value["case_observations"][0]
        row["simple_relation_ids"] = ["OSR002"]
        row["graph_relation_ids_by_layer"] = {
            layer: ["OSR008"] for layer in evaluator.LAYERS
        }

        result = evaluator.evaluate(value)

        self.assertTrue(result["case_labels"][0]["graph_needed"])
        self.assertFalse(result["case_labels"][0]["graph_benefit"])

    def test_alternative_valid_paths_use_the_best_complete_path(self) -> None:
        observation = evaluator._path_observation(
            (("OSR001", "OSR002"), ("OSR003",)),
            ("OSR003",),
        )

        self.assertEqual(observation["path_recall"], 1.0)
        self.assertTrue(observation["complete"])

    def test_configuration_and_observation_completeness_fail_closed(self) -> None:
        value = _completed_observations()
        changed = copy.deepcopy(value)
        changed["configuration"]["simple"]["top_k"] = 20
        with self.assertRaisesRegex(evaluator.V3EvaluationError, "not frozen"):
            evaluator.evaluate(changed)

        incomplete = copy.deepcopy(value)
        incomplete["case_observations"][0]["simple_relation_ids"] = None
        with self.assertRaisesRegex(evaluator.V3EvaluationError, "relation-id list"):
            evaluator.evaluate(incomplete)

        partial = copy.deepcopy(value)
        partial["case_observations"][0]["actual_outcome"] = "partial"
        result = evaluator.evaluate(partial)
        self.assertEqual(result["status"], "computed")

    def test_relation_and_negative_control_metrics_are_separate(self) -> None:
        value = _completed_observations()
        value["graph_extraction"] = {
            "extracted_relation_count": 2,
            "exact_matched_extracted_relation_count": 1,
            "exact_gold_relation_ids": ["OSR001"],
            "topology_gold_relation_ids": ["OSR001", "OSR002"],
            "self_loop_count": 0,
        }

        result = evaluator.evaluate(value)

        extraction = result["metrics"]["relation_extraction"]
        negative = result["metrics"]["negative_controls"]
        self.assertEqual(extraction["exact_match_recall"]["numerator"], 1)
        self.assertEqual(extraction["topology_recall"]["numerator"], 2)
        self.assertEqual(negative["outcome_accuracy"]["value"], 1.0)
        self.assertEqual(negative["refusal_accuracy"]["value"], 1.0)

    def test_locked_output_is_idempotent_and_cannot_be_replaced(self) -> None:
        result = evaluator.evaluate(_completed_observations())
        with tempfile.TemporaryDirectory(prefix="rag-v3-locked-") as directory:
            output = Path(directory) / "locked.json"
            first = evaluator._write_locked(output, result)
            second = evaluator._write_locked(output, result)
            changed = copy.deepcopy(result)
            changed["status"] = "changed"
            with self.assertRaisesRegex(
                evaluator.V3EvaluationError, "already differs"
            ):
                evaluator._write_locked(output, changed)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
