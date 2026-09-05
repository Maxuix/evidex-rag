from types import SimpleNamespace
import unittest
from uuid import UUID

from tools.analyze_auto_qa_ranking import row_result
from tools.evaluate_minilm_completed import compare_rows, probe_results, protected_order, replacement_gate


def row(case_id, group, rank, ids=None):
    return row_result({'evaluation_case_id': case_id, 'group': group}, [
        SimpleNamespace(index_chunk_id=UUID(int=i), label_match=position == rank)
        for position, i in enumerate(ids or range(1, 11), 1)])


class CompletedPoolEvaluationTests(unittest.TestCase):
    def test_equal_aggregate_cannot_hide_an_old_top_ten_loss(self):
        old = [row('a', 'direct', 10), row('b', 'direct', None), row('c', 'paraphrase', 1)]
        new = [row('a', 'direct', None), row('b', 'direct', 10), row('c', 'paraphrase', 1)]
        gate = replacement_gate(old, new, {})
        self.assertEqual(gate['decreased_metrics'], [])
        self.assertFalse(gate['passed'])
        changes = compare_rows(old, new)['direct']
        self.assertEqual(changes['lost_recall_10'], ['a'])
        self.assertEqual(changes['gained_recall_10'], ['b'])

    def test_page_label_success_does_not_hide_lost_continuation_evidence(self):
        old = [row('a', 'direct', 1), row('b', 'paraphrase', 1)]
        new = [row('a', 'direct', 1, ids=[*range(1, 10), 11]), row('b', 'paraphrase', 1)]
        probes = {'a': {str(UUID(int=1)), str(UUID(int=10))}}
        self.assertTrue(probe_results(old, probes)[0]['complete'])
        self.assertFalse(probe_results(new, probes)[0]['complete'])
        self.assertEqual(replacement_gate(old, new, probes)['lost_complete_probes'], ['a'])
        self.assertFalse(replacement_gate(old, new, probes)['passed'])

    def test_rank_regression_inside_top_ten_is_reported(self):
        old = [row('a', 'direct', 7), row('b', 'paraphrase', 2)]
        new = [row('a', 'direct', 9), row('b', 'paraphrase', 2)]
        changes = compare_rows(old, new)['direct']
        self.assertEqual(changes['lost_recall_10'], [])
        self.assertEqual(changes['rank_worsened'], ['a'])
        self.assertIn('direct.mrr_10', replacement_gate(old, new, {})['decreased_metrics'])

    def test_duplicate_or_missing_cases_cannot_pass(self):
        old = [row('a', 'direct', 1), row('b', 'paraphrase', 1)]
        for new in ([old[0]], [old[0], old[0]]):
            with self.assertRaisesRegex(RuntimeError, 'identity differs'):
                compare_rows(old, new)

    def test_prefix_control_preserves_positions_without_duplicate_results(self):
        baseline = [SimpleNamespace(index_chunk_id=UUID(int=i)) for i in range(1, 11)]
        model = list(reversed(baseline))
        result = protected_order(baseline, model)
        self.assertEqual(result[:5], baseline[:5])
        self.assertEqual(result[5:], model[:5])
        self.assertEqual(len({h.index_chunk_id for h in result}), 10)

    def test_identical_rankings_and_complete_evidence_pass(self):
        rows = [row('a', 'direct', 1), row('b', 'paraphrase', 3)]
        self.assertTrue(replacement_gate(rows, rows, {'a': {str(UUID(int=1))}})['passed'])
