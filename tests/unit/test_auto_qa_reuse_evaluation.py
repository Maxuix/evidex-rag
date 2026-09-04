from copy import deepcopy
import unittest

from tools.evaluate_auto_qa_reuse import paired_changes, retrieval_gate_passed


def _complete_result():
    summary = {"retrieval_by_group": {
        group: {"count": count, "hit_at_1": 0.5, "mrr_at_10": 0.6,
                "recall_at_5": 0.7, "recall_at_10": 0.8}
        for group, count in (("direct", 61), ("paraphrase", 24))
    }}
    return {
        "completed_cases": 103, "total_cases": 103,
        "arms": {name: {"summary": deepcopy(summary)} for name in
                 ("classic_source", "minilm_source", "minilm_augmented")},
        "paired": {name: {group: {"lost_top1": [], "lost_top10": []}
                          for group in ("direct", "paraphrase")}
                   for name in ("classic_source", "minilm_source")},
    }


class AutoQAReuseEvaluationTests(unittest.TestCase):
    def test_partial_replay_cannot_pass_gate(self):
        result = _complete_result()
        result["completed_cases"] = 8
        self.assertFalse(retrieval_gate_passed(result))

    def test_gate_rejects_case_loss_even_if_other_gains_offset_it(self):
        result = _complete_result()
        self.assertTrue(retrieval_gate_passed(result))
        result["paired"]["minilm_source"]["direct"]["lost_top10"] = ["lost-case"]
        self.assertFalse(retrieval_gate_passed(result))

    def test_gate_rejects_mrr_regression_without_top_k_loss(self):
        result = _complete_result()
        result["arms"]["minilm_augmented"]["summary"]["retrieval_by_group"]["direct"]["mrr_at_10"] = 0.59
        self.assertFalse(retrieval_gate_passed(result))

    def test_paired_changes_match_case_ids_not_order(self):
        baseline = [dict(case_id="a", group="direct", hit_1=True, recall_10=True, relevant_rank=1),
                    dict(case_id="b", group="direct", hit_1=False, recall_10=False, relevant_rank=None)]
        enhanced = [dict(baseline[1], recall_10=True, relevant_rank=4),
                    dict(baseline[0], hit_1=False, recall_10=False, relevant_rank=None)]
        changes = paired_changes(baseline, enhanced)["direct"]
        self.assertEqual(changes["lost_top1"], ["a"])
        self.assertEqual(changes["lost_top10"], ["a"])
        self.assertEqual(changes["new_top10"], ["b"])
