from __future__ import annotations

from types import SimpleNamespace
import unittest
from uuid import uuid4

from rag_kb.domain import (
    AnswerClaim,
    AnswerControlReason,
    AnswerDraftSource,
    AnswerOutcome,
    GraphitiEdgeResult,
    ValidatedAnswer,
)
from tools.run_large_evaluation import (
    _enterprise_extraction_observation,
    _public_report,
    _routing_report,
    _score_public,
)


def _answering(*, outcome: str) -> SimpleNamespace:
    claims: tuple[AnswerClaim, ...] = ()
    missing: tuple[str, ...] = ()
    if outcome in {"answered", "partial"}:
        claims = (AnswerClaim(text="Revenue was 10.", citation_ids=("cite_1",)),)
        if outcome == "partial":
            missing = ("cause",)
    validated = ValidatedAnswer(
        outcome=AnswerOutcome(outcome),
        claims=claims,
        missing_aspects=missing,
        source=AnswerDraftSource.PROVIDER,
        control_reason=(
            AnswerControlReason.INSUFFICIENT_EVIDENCE if outcome == "refused" else None
        ),
    )
    return SimpleNamespace(validated=validated)


class PublicScorerConflictTests(unittest.TestCase):
    @staticmethod
    def _citation(document_id) -> SimpleNamespace:
        return SimpleNamespace(evidence=SimpleNamespace(document_id=document_id))

    def test_surface_evidence_conflict_requires_gold_and_two_documents(self) -> None:
        score = _score_public(
            {
                "expected_action": "surface_evidence_conflict",
                "stratum": "conflicting_info",
                "gold_answer": "12 and 10",
            }
        )
        no_evidence = score(
            "The sources conflict: one says 12 and one says 10.",
            (),
            _answering(outcome="answered"),
        )
        self.assertFalse(no_evidence["policy_correct"])
        self.assertFalse(no_evidence["multi_document_evidence"])

        first, second = uuid4(), uuid4()
        covered = score(
            "The sources report 12 and 10.",
            (self._citation(first), self._citation(second)),
            _answering(outcome="answered"),
        )
        self.assertTrue(covered["policy_correct"])
        self.assertEqual(covered["cited_document_count"], 2)

        duplicated = score(
            "The sources report 12 and 10.",
            (self._citation(first), self._citation(first)),
            _answering(outcome="answered"),
        )
        self.assertFalse(duplicated["policy_correct"])

        refused = score(
            "The sources report 12 and 10.",
            (),
            _answering(outcome="refused"),
        )
        self.assertFalse(refused["policy_correct"])

    def test_answer_without_false_conflict_uses_answer_correctness_not_a_label(self) -> None:
        score = _score_public(
            {"expected_action": "answer_without_false_conflict", "gold_answer": "Revenue was 10"}
        )
        clean = score(
            "Revenue was 10.",
            (),
            _answering(outcome="answered"),
        )
        self.assertTrue(clean["policy_correct"])

        wrong = score(
            "Revenue was 12.",
            (),
            _answering(outcome="answered"),
        )
        self.assertFalse(wrong["policy_correct"])

        refused = score(
            "Revenue was 10.",
            (),
            _answering(outcome="refused"),
        )
        self.assertFalse(refused["policy_correct"])

    def test_missing_or_unusable_gold_is_not_evaluated_even_with_two_documents(self) -> None:
        citations = (self._citation(uuid4()), self._citation(uuid4()))
        for action in ("surface_evidence_conflict", "answer_without_false_conflict"):
            for gold in (None, "", " ", "10"):
                with self.subTest(action=action, gold=gold):
                    result = _score_public({"expected_action": action, "gold_answer": gold})(
                        "Sources report different amounts.", citations, _answering(outcome="answered")
                    )
                    self.assertIsNone(result["policy_correct"])
                    self.assertEqual(result["policy_evaluation_status"], "not_evaluated_missing_gold")
                    self.assertTrue(result["multi_document_evidence"])
        # An available alias still enables scoring, even without a primary gold.
        result = _score_public({
            "expected_action": "surface_evidence_conflict",
            "gold_answer_aliases": ["Revenue was 10"],
        })("Revenue was 10.", citations, _answering(outcome="answered"))
        self.assertTrue(result["policy_correct"])
        self.assertEqual(result["policy_evaluation_status"], "evaluated")

    def test_unassessed_cases_are_excluded_from_all_public_score_denominators(self) -> None:
        base = {"expected_action": "surface_evidence_conflict", "stratum": "conflict"}
        records = [{**base, "policy_correct": value} for value in (True, False, None)]
        report = _public_report(records)
        for entry in (
            report,
            report["by_expected_action"]["surface_evidence_conflict"],
            report["by_stratum"]["conflict"],
        ):
            self.assertEqual(entry["case_count"], 3)
            self.assertEqual(entry["policy_correct"], {"numerator": 1, "denominator": 2, "value": 0.5})
            self.assertEqual(entry["policy_not_evaluated_count"], 1)
        unavailable = _public_report([{**base, "policy_correct": None}])
        self.assertEqual(unavailable["policy_correct"]["denominator"], 0)
        self.assertIsNone(unavailable["policy_correct"]["value"])



class PublicReportFalsePremiseSplitTests(unittest.TestCase):
    def test_false_premise_action_reports_a_separate_behavior_split(self) -> None:
        records = [
            {"expected_action": "decline_or_correct_false_premise", "actual_outcome": "refused", "policy_correct": True},
            {"expected_action": "decline_or_correct_false_premise", "actual_outcome": "clarify", "policy_correct": False},
            {"expected_action": "decline_or_correct_false_premise", "actual_outcome": "partial", "policy_correct": False},
            {"expected_action": "decline_or_correct_false_premise", "actual_outcome": "answered", "policy_correct": False},
            {"expected_action": "refuse_insufficient_evidence", "actual_outcome": "clarify", "policy_correct": False},
        ]
        report = _public_report(records)
        entry = report["by_expected_action"]["decline_or_correct_false_premise"]
        self.assertEqual(
            entry["false_premise_behavior"],
            {"refused": 1, "clarified": 1, "answered_anyway": 2},
        )
        # The verdict semantics are unchanged: only `refused` is policy-correct.
        self.assertEqual(entry["policy_correct"]["numerator"], 1)
        self.assertEqual(entry["policy_correct"]["denominator"], 4)
        self.assertNotIn(
            "false_premise_behavior",
            report["by_expected_action"]["refuse_insufficient_evidence"],
        )


class RoutingReportSemanticsTests(unittest.TestCase):
    def test_separates_novelty_repeatability_and_sampled_answer_impact(self) -> None:
        qualification = [
            {
                "case_id": "graph-1",
                "hop_count": 2,
                "qualified_graph_needed": True,
                "simple_complete_path_present": False,
                "graph_complete_path_present": True,
                "graph_new_source_chunk_count": 2,
                "graph_candidate_path_count": 16,
                "graph_packed_chunk_count": 4,
            },
            {
                "case_id": "graph-2",
                "hop_count": 3,
                "qualified_graph_needed": False,
                "simple_complete_path_present": True,
                "graph_complete_path_present": True,
                "graph_new_source_chunk_count": 1,
                "graph_candidate_path_count": 16,
                "graph_packed_chunk_count": 3,
            },
            {
                "case_id": "graph-3",
                "hop_count": 3,
                "qualified_graph_needed": False,
                "simple_complete_path_present": False,
                "graph_complete_path_present": False,
                "graph_new_source_chunk_count": 1,
                "graph_candidate_path_count": 8,
                "graph_packed_chunk_count": 5,
            },
        ]
        answers = [
            {
                "case_id": "graph-1",
                "lane": "simple",
                "actual_outcome": "refused",
                "lexical_answer_match_available": True,
                "lexical_answer_match": False,
                "graph_route_attempted": False,
                "graph_route_admitted": False,
                "policy_correct": False,
            },
            {
                "case_id": "graph-1",
                "lane": "auto",
                "actual_outcome": "answered",
                "lexical_answer_match_available": True,
                "lexical_answer_match": True,
                "graph_route_attempted": True,
                "graph_route_admitted": True,
                "graph_call_count": 1,
                "policy_correct": True,
            },
            {
                "case_id": "graph-1",
                "lane": "auto",
                "actual_outcome": "refused",
                "lexical_answer_match_available": True,
                "lexical_answer_match": False,
                "graph_route_attempted": False,
                "graph_route_admitted": False,
                "policy_correct": False,
            },
            {
                "case_id": "graph-1",
                "lane": "auto",
                "actual_outcome": "answered",
                "lexical_answer_match_available": True,
                "lexical_answer_match": True,
                "graph_route_attempted": True,
                "graph_route_admitted": False,
                "graph_call_count": 2,
                "policy_correct": True,
            },
        ]

        report = _routing_report(qualification, answers)

        self.assertEqual(
            report["qualification"]["novel_source_path_outcomes"],
            {
                "case_count": 3,
                "distinct_required_path_completion_count": 1,
                "simple_already_complete_count": 1,
                "required_path_still_incomplete_count": 1,
            },
        )
        self.assertEqual(
            report["qualification"]["bounds"],
            {
                "candidate_path_limit": 16,
                "candidate_path_limit_hit_count": 2,
                "source_chunk_target": 12,
                "source_chunk_target_hit_count": 0,
                "source_chunk_limit": 16,
                "source_chunk_limit_hit_count": 0,
                "maximum_packed_chunk_count": 5,
            },
        )
        answers_report = report["answers"]
        self.assertEqual(
            answers_report["attempted_run_any_admission_rate"]["value"], 0.5
        )
        self.assertEqual(answers_report["auto_graph_call_count"], 3)
        self.assertEqual(answers_report["route_precision"]["denominator"], 0)
        self.assertEqual(
            answers_report["route_precision_status"],
            "not_measured_no_auto_negative_controls",
        )
        self.assertEqual(answers_report["repeatability"]["stable_outcome"]["value"], 0.0)
        self.assertEqual(
            answers_report["repeatability"]["lexical"][
                "pass_all_repeats_correct"
            ]["value"],
            0.0,
        )
        self.assertEqual(
            answers_report["observation_route_attempt_recall"]["value"],
            0.666667,
        )
        self.assertEqual(
            answers_report["qualified_case_any_attempt_rate"]["value"], 1.0
        )
        self.assertEqual(
            answers_report["auto_by_graph_attempt"]["attempted"][
                "lexical_answer_match"
            ]["value"],
            1.0,
        )
        self.assertEqual(
            answers_report["paired_lexical_impact"]["graph_admitted"],
            {
                "pair_count": 1,
                "rescue_count": 1,
                "harm_count": 0,
                "both_correct_count": 0,
                "both_incorrect_count": 0,
                "net_rescue_count": 1,
            },
        )

    def test_route_precision_requires_auto_negative_controls(self) -> None:
        qualification = [
            {"case_id": "positive", "qualified_graph_needed": True},
            {"case_id": "negative", "qualified_graph_needed": False},
        ]
        answers = [
            {
                "case_id": "positive",
                "lane": "auto",
                "graph_route_attempted": True,
                "graph_route_admitted": False,
            },
            {
                "case_id": "negative",
                "lane": "auto",
                "expected_graph_route": False,
                "graph_route_attempted": True,
                "graph_route_admitted": True,
            },
        ]

        report = _routing_report(qualification, answers)["answers"]

        self.assertEqual(report["route_precision_status"], "measured")
        self.assertEqual(report["route_precision"]["value"], 0.5)
        self.assertEqual(report["auto_negative_control_observation_count"], 1)
        self.assertEqual(report["case_level_any_admission_rate"]["value"], 0.0)

    def test_route_precision_is_unavailable_when_labeled_controls_never_route(self) -> None:
        report = _routing_report(
            [{"case_id": "positive", "qualified_graph_needed": True}],
            [
                {
                    "case_id": "positive",
                    "lane": "auto",
                    "graph_route_attempted": False,
                    "graph_route_admitted": False,
                },
                {
                    "case_id": "negative",
                    "lane": "auto",
                    "expected_graph_route": False,
                    "graph_route_attempted": False,
                    "graph_route_admitted": False,
                },
            ],
        )["answers"]

        self.assertEqual(report["route_precision"]["value"], None)
        self.assertEqual(
            report["route_precision_status"], "not_measured_no_route_attempts"
        )

    def test_unqualified_graph_candidate_is_not_implicitly_a_negative_control(self) -> None:
        report = _routing_report(
            [{"case_id": "miss", "qualified_graph_needed": False}],
            [
                {
                    "case_id": "miss",
                    "lane": "auto",
                    "graph_route_attempted": True,
                    "graph_route_admitted": False,
                }
            ],
        )["answers"]

        self.assertEqual(report["auto_negative_control_observation_count"], 0)
        self.assertEqual(report["auto_unlabeled_observation_count"], 1)
        self.assertEqual(
            report["route_precision_status"],
            "not_measured_no_auto_negative_controls",
        )


class EnterpriseExtractionScorerTests(unittest.TestCase):
    def test_aliases_resolve_and_duplicates_require_the_same_fact(self) -> None:
        entities = [
            {
                "name": "Amber Atlas Group",
                "entity_type": "Organization",
                "aliases": ["Amber Atlas"],
            },
            {
                "name": "Blue Beacon Platform",
                "entity_type": "BusinessSystem",
                "aliases": ["Blue Beacon"],
            },
        ]
        relations = [
            {
                "relation_id": "edge-1",
                "source_entity": "Amber Atlas Group",
                "edge_type": "Provides",
                "target_entity": "Blue Beacon Platform",
            }
        ]
        edges = [
            GraphitiEdgeResult(
                edge_uuid="edge-a",
                fact="Amber Atlas provides Blue Beacon.",
                episode_uuids=("episode-1",),
                rank=1,
                source_entity_uuid="source-1",
                source_entity_name="Amber Atlas",
                target_entity_uuid="target-1",
                target_entity_name="Blue Beacon",
                relation_type="Provides",
            ),
            GraphitiEdgeResult(
                edge_uuid="edge-b",
                fact="Amber Atlas provides Blue Beacon.",
                episode_uuids=("episode-2",),
                rank=2,
                source_entity_uuid="source-1",
                source_entity_name="Amber Atlas Group",
                target_entity_uuid="target-1",
                target_entity_name="Blue Beacon Platform",
                relation_type="Provides",
            ),
            GraphitiEdgeResult(
                edge_uuid="edge-c",
                fact="A separate support agreement applies.",
                episode_uuids=("episode-3",),
                rank=3,
                source_entity_uuid="source-1",
                source_entity_name="Amber Atlas Group",
                target_entity_uuid="target-1",
                target_entity_name="Blue Beacon Platform",
                relation_type="Provides",
            ),
        ]

        report = _enterprise_extraction_observation(edges, entities, relations, ())

        self.assertEqual(report["micro_recall"]["value"], 1.0)
        self.assertEqual(report["endpoint_resolution"]["resolved"]["value"], 1.0)
        self.assertEqual(report["duplicate_observed_edge_count"], 1)
        self.assertEqual(report["endpoint_relation_instance_excess_count"], 2)
        self.assertEqual(
            report["relation_classification_given_gold_endpoints"]["value"], 1.0
        )

    def test_asserted_control_edges_are_valid_precision_hits(self) -> None:
        entities = [
            {"name": "Amber Atlas Group", "entity_type": "Organization", "aliases": []},
            {"name": "Blue Beacon Group", "entity_type": "Organization", "aliases": []},
        ]
        controls = [
            {
                "control_id": "control-1",
                "source_entity": "Amber Atlas Group",
                "asserted_edge": "InvestsIn",
                "target_entity": "Blue Beacon Group",
                "forbidden_edge": "Controls",
            }
        ]
        edges = [
            GraphitiEdgeResult(
                edge_uuid="edge-control",
                fact="Amber Atlas invested in Blue Beacon.",
                episode_uuids=("episode-control",),
                rank=1,
                source_entity_uuid="source-1",
                source_entity_name="Amber Atlas Group",
                target_entity_uuid="target-1",
                target_entity_name="Blue Beacon Group",
                relation_type="InvestsIn",
            )
        ]

        report = _enterprise_extraction_observation(edges, entities, [], controls)

        self.assertEqual(report["micro_precision"]["value"], 1.0)
        self.assertEqual(report["forbidden_edge_hit_count"], 0)

    def test_stable_entity_ids_score_alias_surfaces(self) -> None:
        entities = [
            {
                "entity_id": "org-harbor",
                "name": "Harbor Systems",
                "entity_type": "Organization",
                "aliases": ["Harbor"],
            },
            {
                "entity_id": "system-atlas",
                "name": "Atlas Platform",
                "entity_type": "BusinessSystem",
                "aliases": ["Atlas"],
            },
        ]
        relations = [
            {
                "relation_id": "edge-stable-id",
                "source_entity_id": "org-harbor",
                "edge_type": "Provides",
                "target_entity_id": "system-atlas",
            }
        ]
        edges = [
            GraphitiEdgeResult(
                edge_uuid="edge-stable-id",
                fact="Harbor provides Atlas.",
                episode_uuids=("episode-1",),
                rank=1,
                source_entity_uuid="source-1",
                source_entity_name="Harbor",
                target_entity_uuid="target-1",
                target_entity_name="Atlas",
                relation_type="Provides",
            )
        ]

        report = _enterprise_extraction_observation(edges, entities, relations, ())

        self.assertEqual(report["micro_recall"]["value"], 1.0)
        self.assertEqual(report["gold_relation_assertion_count"], 1)
        self.assertEqual(report["entity_identity"]["gold_entity_count"], 2)

    def test_identity_metrics_detect_fragmentation_and_forbidden_merges(self) -> None:
        entities = [
            {
                "entity_id": "org-left",
                "name": "Meridian Systems International",
                "entity_type": "Organization",
                "aliases": ["MSI"],
            },
            {
                "entity_id": "org-right",
                "name": "Meridian System Integration",
                "entity_type": "Organization",
                "aliases": ["MII"],
            },
            {
                "entity_id": "system-atlas",
                "name": "Atlas Platform",
                "entity_type": "BusinessSystem",
                "aliases": ["Atlas"],
            },
        ]
        edges = [
            GraphitiEdgeResult(
                edge_uuid="edge-left-1",
                fact="MSI provides Atlas.",
                episode_uuids=("episode-1",),
                rank=1,
                source_entity_uuid="merged-source",
                source_entity_name="MSI",
                target_entity_uuid="atlas",
                target_entity_name="Atlas",
                relation_type="Provides",
            ),
            GraphitiEdgeResult(
                edge_uuid="edge-left-2",
                fact="Meridian Systems International supports Atlas.",
                episode_uuids=("episode-2",),
                rank=2,
                source_entity_uuid="split-source",
                source_entity_name="Meridian Systems International",
                target_entity_uuid="atlas",
                target_entity_name="Atlas",
                relation_type="Supports",
            ),
            GraphitiEdgeResult(
                edge_uuid="edge-right",
                fact="MII supports Atlas.",
                episode_uuids=("episode-3",),
                rank=3,
                source_entity_uuid="merged-source",
                source_entity_name="MII",
                target_entity_uuid="atlas",
                target_entity_name="Atlas",
                relation_type="Supports",
            ),
        ]
        identity_controls = [
            {
                "control_id": "distinct-meridian-organizations",
                "left_entity_id": "org-left",
                "right_entity_id": "org-right",
            }
        ]

        report = _enterprise_extraction_observation(
            edges,
            entities,
            (),
            (),
            identity_controls,
        )

        self.assertEqual(report["entity_identity"]["fragmentation_excess_count"], 1)
        self.assertEqual(report["entity_identity"]["forbidden_merge_hit_count"], 1)


if __name__ == "__main__":
    unittest.main()
