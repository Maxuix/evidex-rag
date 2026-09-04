from __future__ import annotations

import unittest

from rag_kb.domain import RerankMode, RetrievalStrategy
from rag_kb.schemas.chat import ChatRunRetrievalResponse
from rag_kb.retrieval.profile import (
    GRAPH_RETRIEVAL_PROFILE_VERSION,
    HYBRID_PROFILE_VERSION,
    GraphRetrievalProfile,
    adaptive_graphiti_profile,
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

    def test_legacy_v1_boolean_snapshot_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_retrieval_snapshot(
                {
                    "profile_version": "exact_vector_v1",
                    "strategy": "exact_vector",
                    "top_k": 7,
                    "rerank": False,
                }
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
            {**baseline, "rerank_mode": "future"},
            {"profile_version": HYBRID_PROFILE_VERSION},
        ):
            with self.subTest(snapshot=snapshot), self.assertRaises(
                (KeyError, ValueError)
            ):
                parse_retrieval_snapshot(snapshot)

    def test_graph_profile_round_trips_as_a_classic_hybrid_outer_profile(self) -> None:
        snapshot = graph_profile(top_k=8).as_dict()

        strategy, top_k, rerank_mode, execution_type = parse_chat_retrieval_snapshot(
            snapshot
        )

        self.assertEqual(snapshot["profile_version"], GRAPH_RETRIEVAL_PROFILE_VERSION)
        self.assertIs(strategy, RetrievalStrategy.HYBRID)
        self.assertEqual(top_k, 8)
        self.assertIs(rerank_mode, RerankMode.CLASSIC)
        self.assertEqual(execution_type, "manual_graph")
        local = graph_profile(
            top_k=8, rerank_mode=RerankMode.LOCAL_MINILM_V1
        ).as_dict()
        self.assertIs(
            parse_chat_retrieval_snapshot(local)[2], RerankMode.LOCAL_MINILM_V1
        )
        with self.assertRaises(ValueError):
            parse_chat_retrieval_snapshot({**snapshot, "rerank_mode": "none"})

    def test_legacy_graph_snapshot_cannot_be_executed(self) -> None:
        snapshot = {
            "profile_version": "graph_augmented_v1",
            "strategy": "hybrid",
            "top_k": 8,
            "rerank_mode": "classic",
            "augmentation": "entity_graph_v1",
        }
        with self.assertRaises(ValueError):
            parse_chat_retrieval_snapshot(snapshot)
        with self.assertRaises(ValueError):
            GraphRetrievalProfile(
                profile_version="graph_augmented_v1",
                strategy=RetrievalStrategy.HYBRID,
                top_k=8,
                rerank_mode=RerankMode.CLASSIC,
                augmentation="entity_graph_v1",
            )
        response = ChatRunRetrievalResponse(
            mode="graph",
            profile_version=snapshot["profile_version"],
            strategy=snapshot["strategy"],
            top_k=snapshot["top_k"],
            rerank_mode=snapshot["rerank_mode"],
        )
        self.assertEqual(response.profile_version, "graph_augmented_v1")

    def test_adaptive_graphiti_snapshot_is_independent_and_strict(self) -> None:
        snapshot = adaptive_graphiti_profile(top_k=8).as_dict()

        strategy, top_k, rerank_mode, execution_type = parse_chat_retrieval_snapshot(
            snapshot
        )

        self.assertIs(strategy, RetrievalStrategy.EXACT_VECTOR)
        self.assertEqual((top_k, rerank_mode), (8, RerankMode.NONE))
        self.assertEqual(execution_type, "adaptive_graphiti")
        response = ChatRunRetrievalResponse(
            mode="auto",
            profile_version=snapshot["profile_version"],
            strategy=strategy.value,
            top_k=top_k,
            rerank_mode=rerank_mode,
        )
        self.assertEqual(response.profile_version, snapshot["profile_version"])
        for malformed in (
            {**snapshot, "router": "future_router"},
            {**snapshot, "augmentation": "entity_graph_v1"},
            {**snapshot, "extra": True},
            {key: value for key, value in snapshot.items() if key != "router"},
        ):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                parse_chat_retrieval_snapshot(malformed)
        with self.assertRaises(ValueError):
            parse_chat_retrieval_snapshot(
                {**snapshot, "profile_version": "adaptive_graphiti_v1"}
            )

    def test_previous_graphiti_profiles_cannot_be_executed(self) -> None:
        snapshots = (
            {
                "profile_version": "graphiti_edge_augmented_v1",
                "strategy": "hybrid",
                "top_k": 8,
                "rerank_mode": "classic",
                "augmentation": "graphiti_edge_v1",
            },
            {
                "profile_version": "adaptive_graphiti_v1",
                "strategy": "exact_vector",
                "top_k": 8,
                "rerank_mode": "none",
                "router": "native_agent_evidence_aware_v1",
                "augmentation": "graphiti_edge_v1",
            },
        )
        for snapshot in snapshots:
            with self.subTest(snapshot=snapshot):
                with self.assertRaises(ValueError):
                    parse_chat_retrieval_snapshot(snapshot)
