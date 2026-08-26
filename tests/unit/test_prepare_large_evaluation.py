from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.evaluation_campaign_state import load_or_create
from tools.prepare_large_evaluation import build_plan


class PrepareLargeEvaluationTests(unittest.TestCase):
    def test_plan_has_actionable_denominators_and_recovery_contract(self) -> None:
        plan = build_plan()
        self.assertEqual(plan["offline_validation"]["public_answer_refusal"]["case_count"], 281)
        self.assertEqual(plan["coverage"]["routing"]["candidate_count"], 48)
        self.assertEqual(
            plan["coverage"]["routing"]["minimum_qualified_graph_needed_count"], 30
        )
        self.assertEqual(plan["coverage"]["enterprise"]["relation_types"], 42)
        self.assertEqual(plan["coverage"]["enterprise"]["online_max_hops"], 3)
        self.assertEqual(
            plan["token_reservation"]["agent_observation_range_after_routing_qualification"],
            [513, 567],
        )
        self.assertEqual(
            plan["recovery_contract"]["completed_case_policy"],
            "skip_only_durably_completed_cases",
        )

    def test_plan_binding_can_initialize_a_campaign_checkpoint(self) -> None:
        plan = build_plan()
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "campaign.json"
            state = load_or_create(checkpoint, plan["plan_binding"])
            self.assertEqual(state["binding_sha256"], plan["plan_binding_sha256"])


if __name__ == "__main__":
    unittest.main()
