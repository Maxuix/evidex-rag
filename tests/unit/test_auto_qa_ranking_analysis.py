from types import SimpleNamespace
import unittest
from uuid import UUID

from rag_kb.retrieval.reranker import order_model_scored_evidence
from tools.analyze_auto_qa_ranking import rank_with_scores, row_result


class RankingAnalysisTests(unittest.TestCase):
    def test_optimized_replay_matches_production_with_ties_and_small_scores(self):
        hits = [SimpleNamespace(index_chunk_id=UUID(int=i + 1), text=text) for i, text in enumerate((
            "sales revenue data center 2022", "sales revenue data center 2021",
            "cash current liabilities 2022", "cash current liabilities 2022",
            "", "经营活动现金流 年报 2022", "经营活动现金流 年报 2021",
        ))]
        for values in ((.9, .9, .5, .5, .1, .02, .01), (.009, .008, .003, .003, .001, .0004, .0001),
                       (-1, -1, -2, -3, -4, -5, -6)):
            scores = {hit.index_chunk_id: score for hit, score in zip(hits, values, strict=True)}
            for mmr in (False, True):
                expected = order_model_scored_evidence(hits, scores, mmr_lambda=.75 if mmr else 1)
                self.assertEqual(rank_with_scores(hits, scores, mmr=mmr), list(expected))

    def test_supplement_outside_top_ten_cannot_be_reported_as_top_ten_gain(self):
        hits = [SimpleNamespace(index_chunk_id=UUID(int=i + 1), label_match=i == 10) for i in range(12)]
        row = row_result({"evaluation_case_id": "case", "group": "direct"}, hits)
        self.assertFalse(row["recall_10"])
        self.assertIsNone(row["relevant_rank"])
        self.assertEqual(len(row["final_ids"]), 10)
