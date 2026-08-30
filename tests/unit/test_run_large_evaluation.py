from __future__ import annotations

from types import SimpleNamespace
import unittest

from rag_kb.domain import (
    AnswerClaim,
    AnswerConflict,
    AnswerConflictAdjudication,
    AnswerConflictType,
    AnswerControlReason,
    AnswerDraftSource,
    AnswerOutcome,
    GraphitiEdgeResult,
    ValidatedAnswer,
)
from tools.run_large_evaluation import (
    _enterprise_extraction_observation,
    _public_report,
    _score_public,
)


def _answering(*, outcome: str, conflict: bool) -> SimpleNamespace:
    claims: tuple[AnswerClaim, ...] = ()
    missing: tuple[str, ...] = ()
    if outcome in {"answered", "partial"}:
        if conflict:
            claims = (
                AnswerClaim(
                    text="A later version reports 12; an older memo reports 10.",
                    citation_ids=("cite_1", "cite_2"),
                    conflict=AnswerConflict(
                        supporting_citation_ids=("cite_1",),
                        conflicting_citation_ids=("cite_2",),
                        conflict_type=AnswerConflictType.VERSION,
                        adjudication=AnswerConflictAdjudication.RESOLVABLE,
                    ),
                ),
            )
        else:
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
    def test_surface_evidence_conflict_requires_a_complete_conflict_claim(self) -> None:
        score = _score_public(
            {
                "expected_action": "surface_evidence_conflict",
                "stratum": "conflicting_info",
            }
        )
        keyword_only = score(
            "The sources conflict and disagree on the amount.",
            (),
            _answering(outcome="answered", conflict=False),
        )
        self.assertFalse(keyword_only["policy_correct"])
        self.assertTrue(keyword_only["conflict_marker_present"])

        structured = score(
            "The sources conflict and disagree on the amount.",
            (),
            _answering(outcome="answered", conflict=True),
        )
        self.assertTrue(structured["policy_correct"])
        self.assertTrue(structured["conflict_marker_present"])

        refused = score(
            "The sources conflict.",
            (),
            _answering(outcome="refused", conflict=False),
        )
        self.assertFalse(refused["policy_correct"])

    def test_answer_without_false_conflict_rejects_conflict_claims(self) -> None:
        score = _score_public(
            {"expected_action": "answer_without_false_conflict"}
        )
        clean = score(
            "Revenue was 10.",
            (),
            _answering(outcome="answered", conflict=False),
        )
        self.assertTrue(clean["policy_correct"])
        self.assertFalse(clean["conflict_marker_present"])

        false_conflict = score(
            "Revenue was 10.",
            (),
            _answering(outcome="answered", conflict=True),
        )
        self.assertFalse(false_conflict["policy_correct"])

        refused = score(
            "Revenue was 10.",
            (),
            _answering(outcome="refused", conflict=False),
        )
        self.assertFalse(refused["policy_correct"])


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
