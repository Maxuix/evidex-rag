from __future__ import annotations

import unittest

from rag_kb.retrieval.profile import (
    HYBRID_PROFILE_VERSION,
    exact_profile,
    parse_retrieval_snapshot,
)


class RetrievalExecutionProfileTests(unittest.TestCase):
    def test_incomplete_three_field_snapshot_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_retrieval_snapshot(
                {"strategy": "exact_vector", "top_k": 7, "rerank": False},
            )

    def test_snapshot_contains_only_the_selected_preset(self) -> None:
        snapshot = {
            "profile_version": HYBRID_PROFILE_VERSION,
            "strategy": "hybrid",
            "top_k": 6,
            "rerank": True,
        }

        strategy, top_k, rerank = parse_retrieval_snapshot(snapshot)

        self.assertEqual(strategy.value, "hybrid")
        self.assertEqual((top_k, rerank), (6, True))
        self.assertEqual(
            set(exact_profile(top_k=6).as_dict()),
            {"profile_version", "strategy", "top_k", "rerank"},
        )

    def test_unknown_or_incomplete_hybrid_snapshot_fails_closed(self) -> None:
        baseline = exact_profile().as_dict()
        for snapshot in (
            {**baseline, "profile_version": "future_v2"},
            {
                **baseline,
                "profile_version": HYBRID_PROFILE_VERSION,
                "strategy": "hybrid",
                "dense_candidate_count": 20,
            },
            {**baseline, "top_k": True},
            {**baseline, "rerank": 1},
            {"profile_version": HYBRID_PROFILE_VERSION},
        ):
            with self.subTest(snapshot=snapshot), self.assertRaises(
                (KeyError, ValueError)
            ):
                parse_retrieval_snapshot(snapshot)
