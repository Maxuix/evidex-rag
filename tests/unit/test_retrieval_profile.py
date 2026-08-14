from __future__ import annotations

import unittest

from rag_kb.domain import RerankMode, RetrievalStrategy
from rag_kb.retrieval.profile import (
    GRAPH_RETRIEVAL_PROFILE_VERSION,
    HYBRID_PROFILE_VERSION,
    LEGACY_EXACT_PROFILE_VERSION,
    exact_profile,
    graph_profile,
    parse_chat_retrieval_snapshot,
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
            "rerank_mode": "classic",
        }

        strategy, top_k, rerank_mode = parse_retrieval_snapshot(snapshot)

        self.assertEqual(strategy.value, "hybrid")
        self.assertEqual((top_k, rerank_mode), (6, RerankMode.CLASSIC))
        self.assertEqual(
            set(exact_profile(top_k=6).as_dict()),
            {"profile_version", "strategy", "top_k", "rerank_mode"},
        )

    def test_legacy_v1_boolean_snapshot_remains_readable(self) -> None:
        strategy, top_k, rerank_mode = parse_retrieval_snapshot(
            {
                "profile_version": LEGACY_EXACT_PROFILE_VERSION,
                "strategy": "exact_vector",
                "top_k": 7,
                "rerank": False,
            }
        )

        self.assertEqual(strategy.value, "exact_vector")
        self.assertEqual(top_k, 7)
        self.assertIs(rerank_mode, RerankMode.NONE)

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
            {**baseline, "rerank_mode": "future"},
            {"profile_version": HYBRID_PROFILE_VERSION},
        ):
            with self.subTest(snapshot=snapshot), self.assertRaises(
                (KeyError, ValueError)
            ):
                parse_retrieval_snapshot(snapshot)

    def test_graph_profile_round_trips_as_a_classic_hybrid_outer_profile(self) -> None:
        snapshot = graph_profile(top_k=8).as_dict()

        strategy, top_k, rerank_mode, augmentation = parse_chat_retrieval_snapshot(
            snapshot
        )

        self.assertEqual(snapshot["profile_version"], GRAPH_RETRIEVAL_PROFILE_VERSION)
        self.assertIs(strategy, RetrievalStrategy.HYBRID)
        self.assertEqual(top_k, 8)
        self.assertIs(rerank_mode, RerankMode.CLASSIC)
        self.assertEqual(augmentation, "entity_graph_v1")
        with self.assertRaises(ValueError):
            parse_chat_retrieval_snapshot({**snapshot, "rerank_mode": "none"})
