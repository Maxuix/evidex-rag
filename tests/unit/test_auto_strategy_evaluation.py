import unittest

from tools.auto_strategy_metrics import evidence_row, paired_changes, policies, rrf_order, supplement
from tools.analyze_auto_strategy import paired_interval
from tools.prepare_auto_strategy import select_hotpot


class AutoStrategyEvaluationTests(unittest.TestCase):
    def test_large_supplement_cannot_evict_or_reorder_baseline(self):
        base = list(range(10))
        costs = {i: 100 for i in range(13)} | {10: 3000}
        self.assertEqual(supplement(base, [9, 10, 11, 11, 12], costs), base+[11, 12])

    def test_supplement_does_not_truncate_a_chunk_to_fit(self):
        self.assertEqual(supplement(['old'], ['large', 'small'],
            {'old': 5000, 'large': 100, 'small': 40}, token_budget=50), ['old', 'small'])

    def test_matched_controls_cannot_exceed_allowed_additional_budget(self):
        classic = list(range(20))
        raw = list(reversed(classic))
        costs = {i: 400+i*10 for i in classic}
        arms = policies(classic, raw, raw, costs)
        for name in ('raw', 'mmr'):
            append = arms['append_'+name]
            count_control = arms['classic_'+name+'_same_count']
            token_control = arms['classic_'+name+'_same_tokens']
            self.assertEqual(append[:10], classic[:10])
            self.assertLessEqual(len(count_control), len(append))
            self.assertLessEqual(sum(costs[i] for i in token_control[10:]), sum(costs[i] for i in append[10:]))
            self.assertLessEqual(sum(costs[i] for i in append[10:]), 2048)

    def test_matching_one_multihop_source_is_not_complete(self):
        case = {'case_id': 'a', 'family': 'x', 'group': 'x', 'answerable': True,
                'required_paths': [['A', 'B']], 'labels': {'1': ['A'], '2': ['B']}}
        row = evidence_row(case, ['1'], ['1', '2'], {'1': 10, '2': 10})
        self.assertTrue(row['hit1'])
        self.assertFalse(row['complete_evidence'])
        self.assertEqual(row['required_fraction'], .5)
        self.assertTrue(row['candidate_complete'])

    def test_unanswerable_case_is_not_automatic_retrieval_success(self):
        case = {'case_id': 'n', 'family': 'x', 'group': 'n', 'answerable': False,
                'required_paths': [], 'labels': {'1': ['A']}}
        row = evidence_row(case, ['1'], ['1'], {'1': 10})
        for metric in ('hit1', 'hit10', 'complete_evidence', 'required_fraction', 'candidate_complete'):
            self.assertIsNone(row[metric])

    def test_empty_positive_gold_is_rejected(self):
        case = {'case_id': 'bad', 'family': 'x', 'group': 'x', 'answerable': True,
                'required_paths': [[]], 'labels': {}}
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            evidence_row(case, [], [], {})

    def test_rankings_cannot_include_duplicate_or_unseen_candidates(self):
        for model in ([1, 1], [1, 3]):
            with self.assertRaises(ValueError):
                rrf_order([1, 2], model)
        with self.assertRaises(ValueError):
            policies([1], [1], [1], {1: -1})

    def test_regression_comparison_rejects_duplicate_cases(self):
        with self.assertRaisesRegex(ValueError, 'identities'):
            paired_changes([{'case_id': 'a'}, {'case_id': 'b'}], [{'case_id': 'a'}, {'case_id': 'a'}])

    def test_hotpot_holdout_selection_is_order_independent_and_excludes_pilot(self):
        rows = [{'case_id': str(i), 'type': 'bridge' if i < 110 else 'comparison'} for i in range(150)]
        selected = select_hotpot(rows, {'0', '110'})
        self.assertEqual(selected, select_hotpot(list(reversed(rows)), {'0', '110'}))
        self.assertEqual(len(selected), 100)
        self.assertFalse({'0', '110'} & {c['case_id'] for c in selected})
        self.assertEqual(sum(c['type'] == 'comparison' for c in selected), 20)

    def test_interval_keeps_related_questions_in_one_cluster_and_omits_negative_gold(self):
        old = [
            {'case_id': 'original', 'cluster': 'p', 'complete_evidence': False},
            {'case_id': 'rewrite', 'cluster': 'p', 'complete_evidence': False},
            {'case_id': 'negative', 'cluster': 'n', 'complete_evidence': None},
        ]
        new = [{**r, 'complete_evidence': True if r['complete_evidence'] is not None else None} for r in old]
        interval = paired_interval(old, new, 'complete_evidence', repetitions=100)
        self.assertEqual(interval['cases'], 2)
        self.assertEqual(interval['clusters'], 1)
        self.assertEqual(interval['difference'], 1)
        self.assertEqual(interval['ci95'], [1, 1])
