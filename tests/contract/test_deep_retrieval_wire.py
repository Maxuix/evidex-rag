from __future__ import annotations

import unittest
from uuid import uuid4

from pydantic import ValidationError

from rag_kb.schemas.deep_retrieval import (
    CanonicalEvidenceKeyWire,
    CoverageReportWire,
    DeepRetrievalCapabilitySnapshotWire,
    DeepRetrievalFailureWire,
    GoalCoverageWire,
    GoalWire,
    ModelUsageFactWire,
    PlanEnvelopeWire,
    ServerBudgetSnapshotWire,
    SingleGoalFallbackWire,
)


class DeepRetrievalWireContractTests(unittest.TestCase):
    def test_capability_and_failure_wires_are_default_off_and_content_safe(self) -> None:
        capability = DeepRetrievalCapabilitySnapshotWire.model_validate(
            {"configuration_fingerprint": "sha256:" + "a" * 64}
        )
        self.assertFalse(capability.deep_workflow_enabled)
        self.assertFalse(capability.static_multi_query_v1_enabled)
        with self.assertRaises(ValidationError):
            DeepRetrievalCapabilitySnapshotWire.model_validate(
                {
                    "configuration_fingerprint": "sha256:" + "a" * 64,
                    "static_multi_query_v1_enabled": True,
                }
            )
        failure = DeepRetrievalFailureWire.model_validate(
            {
                "code": "BUDGET_EXCEEDED",
                "status": 409,
                "retryable": False,
                "detail": {
                    "check": "budget",
                    "field": "goal_count",
                    "count": 5,
                    "limit": 4,
                },
            }
        )
        self.assertEqual(failure.to_domain().check, "budget")
        invalid = failure.model_dump(mode="json")
        invalid["detail"]["raw_provider_output"] = "secret"  # type: ignore[index]
        with self.assertRaises(ValidationError):
            DeepRetrievalFailureWire.model_validate(invalid)

    def _plan(self, **overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "plan_id": "plan-1",
            "question_ref": "question-ref",
            "workflow_depth": "deep",
            "retrieval_mode": "vector",
            "goals": (
                {
                    "goal_id": "g1",
                    "question": "one",
                    "query": "one",
                    "query_variants": ("one",),
                    "depends_on": (),
                },
            ),
            "budget": {
                "max_goals": 4,
                "max_query_variants_per_goal": 1,
                "max_adaptive_waves": 0,
                "max_repairs": 1,
                "deadline_seconds": 5.0,
                "max_parallelism": 1,
                "max_retrieval_calls": 1,
                "max_output_tokens": 256,
            },
            "capability_fingerprint": "sha256:" + "b" * 64,
        }
        payload.update(overrides)
        return payload

    def test_strict_frozen_versioned_round_trip_and_hash(self) -> None:
        wire = PlanEnvelopeWire.model_validate(self._plan())
        self.assertTrue(wire.model_config["extra"] == "forbid")
        self.assertTrue(wire.model_config["frozen"])
        self.assertEqual(wire.to_domain().canonical_hash, wire.canonical_hash)
        self.assertEqual(
            PlanEnvelopeWire.model_validate(wire.model_dump()).canonical_hash,
            wire.canonical_hash,
        )
        with self.assertRaises(ValidationError):
            wire.plan_id = "changed"  # type: ignore[misc]

    def test_unknown_server_or_execution_fields_are_rejected(self) -> None:
        forbidden = (
            {"strategy": "hybrid"},
            {"top_k": 10},
            {"filters": {}},
            {"revision_id": "rev"},
            {"provider": "provider"},
            {"tool": "search"},
            {"budget_override": {"max_goals": 4}},
        )
        for extra in forbidden:
            payload = self._plan()
            payload.update(extra)
            with self.subTest(extra=extra), self.assertRaises(ValidationError):
                PlanEnvelopeWire.model_validate(payload)

    def test_budget_wire_enforces_phase_one_bounds(self) -> None:
        for field_name, value in (
            ("max_goals", 5),
            ("max_query_variants_per_goal", 2),
            ("max_adaptive_waves", 1),
            ("max_repairs", 2),
            ("deadline_seconds", 0),
            ("deadline_seconds", 120.1),
            ("max_parallelism", 0),
            ("max_parallelism", 2),
            ("max_retrieval_calls", 0),
            ("max_retrieval_calls", 2),
        ):
            payload = {
                "max_goals": 4,
                "max_query_variants_per_goal": 1,
                "max_adaptive_waves": 0,
                "max_repairs": 1,
                "deadline_seconds": 5.0,
                "max_parallelism": 1,
                "max_retrieval_calls": 1,
            }
            payload[field_name] = value
            with self.subTest(field_name=field_name), self.assertRaises(ValidationError):
                ServerBudgetSnapshotWire.model_validate(payload)

    def test_goal_duplicate_query_and_unknown_dependency_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            GoalWire.model_validate(
                {"goal_id": "g1", "question": uuid4(), "query": "one"}
            )
        for field_name in ("question", "query"):
            payload = {"goal_id": "g1", "question": "question", "query": "one"}
            payload[field_name] = "   "
            with self.subTest(field_name=field_name), self.assertRaises(ValidationError):
                GoalWire.model_validate(payload)
        with self.assertRaises(ValidationError):
            GoalWire.model_validate(
                {
                    "goal_id": "g1",
                    "question": "question",
                    "query_variants": ("one", "two"),
                }
            )
        payload = self._plan()
        payload["goals"] = (
            {
                "goal_id": "g1",
                "question": "question",
                "query": "one",
                "depends_on": ("missing",),
            },
        )
        with self.assertRaises(ValidationError):
            PlanEnvelopeWire.model_validate(payload)

        duplicate_queries = self._plan()
        duplicate_queries["goals"] = (
            {"goal_id": "g1", "question": "one", "query": "Same query"},
            {"goal_id": "g2", "question": "two", "query": "same   query"},
        )
        with self.assertRaises(ValidationError):
            PlanEnvelopeWire.model_validate(duplicate_queries)

    def test_coverage_and_usage_cardinality_are_bounded(self) -> None:
        keys = tuple(
            {
                "revision_id": "rev",
                "target_id": "target",
                "chunk_or_group_id": f"chunk-{index}",
                "representation_id": "text",
                "plan_id": "plan-1",
            }
            for index in range(11)
        )
        with self.assertRaises(ValidationError):
            GoalCoverageWire.model_validate(
                {"goal_id": "g1", "status": "supported", "evidence_keys": keys}
            )
        with self.assertRaises(ValidationError):
            ModelUsageFactWire.model_validate({"input_tokens": 1_000_001})

    def test_evidence_key_requires_identity_and_coverage_checks_allowlist(self) -> None:
        key = {
            "revision_id": "rev",
            "target_id": "target",
            "chunk_or_group_id": "chunk",
            "representation_id": "text",
            "plan_id": "plan-1",
        }
        self.assertEqual(CanonicalEvidenceKeyWire.model_validate(key).representation_id, "text")
        with self.assertRaises(ValidationError):
            CanonicalEvidenceKeyWire.model_validate({"revision_id": "rev", "target_id": "target"})
        with self.assertRaises(ValidationError):
            CanonicalEvidenceKeyWire.model_validate(
                {
                    "revision_id": "rev with space",
                    "target_id": "target",
                    "chunk_or_group_id": "chunk",
                    "representation_id": "text",
                }
            )

        report = {
            "plan_id": "plan-1",
            "plan_goal_ids": ("g1",),
            "goals": (
                {
                    "goal_id": "g1",
                    "status": "supported",
                    "evidence_keys": (key,),
                },
            ),
            "allowed_evidence_keys": (key,),
        }
        self.assertEqual(CoverageReportWire.model_validate(report).goals[0].status.value, "supported")
        outside = dict(key)
        outside["chunk_or_group_id"] = "not-allowed"
        report["goals"] = ({"goal_id": "g1", "status": "supported", "evidence_keys": (outside,)},)
        with self.assertRaises(ValidationError):
            CoverageReportWire.model_validate(report)
        report["goals"] = ({"goal_id": "g1", "status": "supported", "evidence_keys": (key,)},)
        report["plan_goal_ids"] = ("unknown",)
        with self.assertRaises(ValidationError):
            CoverageReportWire.model_validate(report)
        report["plan_goal_ids"] = ("g1",)
        unbound = dict(key)
        unbound.pop("plan_id")
        report["goals"] = ({"goal_id": "g1", "status": "supported", "evidence_keys": (unbound,)},)
        report["allowed_evidence_keys"] = (unbound,)
        with self.assertRaises(ValidationError):
            CoverageReportWire.model_validate(report)

    def test_fallback_wire_is_not_coverage_failure(self) -> None:
        fallback = SingleGoalFallbackWire.model_validate(
            {"question_ref": "question-ref", "retrieval_mode": "hybrid"}
        )
        self.assertEqual(fallback.outcome.value, "single_goal_fallback")
        with self.assertRaises(ValidationError):
            invalid = fallback.model_dump()
            invalid["outcome"] = "assessment_failure"
            SingleGoalFallbackWire.model_validate(invalid)
        with self.assertRaises(ValidationError):
            SingleGoalFallbackWire.model_validate(
                {"question_ref": "question-ref", "reason": "raw provider output: secret"}
            )
        with self.assertRaises(ValidationError):
            SingleGoalFallbackWire.model_validate({"question_ref": "question ref"})


if __name__ == "__main__":
    unittest.main()
