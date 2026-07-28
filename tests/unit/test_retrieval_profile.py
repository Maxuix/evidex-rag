from __future__ import annotations

import unittest

from rag_kb.domain import RetrievalStrategy
from rag_kb.retrieval.profile import (
    HYBRID_PROFILE_VERSION,
    RetrievalExecutionProfile,
    legacy_exact_profile,
)


class RetrievalExecutionProfileTests(unittest.TestCase):
    def test_legacy_three_field_snapshot_maps_to_exact_v1(self) -> None:
        profile = RetrievalExecutionProfile.from_snapshot(
            {"strategy": "exact_vector", "top_k": 7, "rerank": False},
            legacy_defaults=legacy_exact_profile(),
        )

        self.assertEqual(profile.profile_version, "exact_vector_v1")
        self.assertEqual(profile.strategy, RetrievalStrategy.EXACT_VECTOR)
        self.assertEqual(profile.top_k, 7)
        self.assertFalse(profile.rerank)
        self.assertIsNone(profile.lexical_analyzer_version)

    def test_hybrid_snapshot_round_trips_all_frozen_values(self) -> None:
        baseline = legacy_exact_profile(top_k=6)
        snapshot = {
            **baseline.as_dict(),
            "profile_version": HYBRID_PROFILE_VERSION,
            "strategy": "hybrid",
            "lexical_analyzer_version": "lexical_simple_cjk_bigram_v1",
            "lexical_query_version": "lexical_or_query_v1",
            "dense_candidate_count": 24,
            "lexical_candidate_count": 31,
            "dense_weight_micros": 900_000,
            "lexical_weight_micros": 1_100_000,
        }

        profile = RetrievalExecutionProfile.from_snapshot(
            snapshot,
            legacy_defaults=legacy_exact_profile(),
        )

        self.assertEqual(profile.as_dict(), snapshot)

    def test_unknown_or_incomplete_hybrid_snapshot_fails_closed(self) -> None:
        baseline = legacy_exact_profile().as_dict()
        for snapshot in (
            {**baseline, "profile_version": "future_v2"},
            {
                **baseline,
                "profile_version": HYBRID_PROFILE_VERSION,
                "strategy": "hybrid",
                "lexical_analyzer_version": "unknown",
                "lexical_query_version": "lexical_or_query_v1",
            },
            {"profile_version": HYBRID_PROFILE_VERSION},
        ):
            with self.subTest(snapshot=snapshot), self.assertRaises(
                (KeyError, ValueError)
            ):
                RetrievalExecutionProfile.from_snapshot(
                    snapshot,
                    legacy_defaults=legacy_exact_profile(),
                )
