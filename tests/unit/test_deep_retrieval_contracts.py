from __future__ import annotations

import unittest

from rag_kb.domain.deep_retrieval import (
    AssessmentFailure,
    CanonicalEvidenceKey,
    CoverageReport,
    CoverageStatus,
    DeepRetrievalCapabilitySnapshot,
    DeepRetrievalErrorCode,
    Goal,
    GoalCoverage,
    PlanEnvelope,
    RetrievalMode,
    ServerBudgetSnapshot,
    WorkflowDepth,
    canonical_json,
    coverage_assessment_failure,
    deep_retrieval_failure,
    single_goal_fallback,
)

_FINGERPRINT = "sha256:" + "b" * 64


def _budget(**changes: object) -> ServerBudgetSnapshot:
    values: dict[str, object] = {
        "max_goals": 4,
        "max_query_variants_per_goal": 1,
        "max_adaptive_waves": 0,
        "max_repairs": 1,
        "deadline_seconds": 5,
        "max_parallelism": 1,
        "max_retrieval_calls": 1,
    }
    values.update(changes)
    return ServerBudgetSnapshot(**values)


class DeepRetrievalDomainContractTests(unittest.TestCase):
    def test_capability_is_default_off_and_static_requires_deep(self) -> None:
        fingerprint = "sha256:" + "a" * 64
        snapshot = DeepRetrievalCapabilitySnapshot(fingerprint)
        self.assertFalse(snapshot.deep_workflow_enabled)
        self.assertFalse(snapshot.static_multi_query_v1_enabled)
        with self.assertRaises(ValueError):
            DeepRetrievalCapabilitySnapshot(
                fingerprint,
                deep_workflow_enabled=False,
                static_multi_query_v1_enabled=True,
            )

    def test_failure_facts_use_fixed_safe_policy(self) -> None:
        failure = deep_retrieval_failure(
            DeepRetrievalErrorCode.BUDGET_EXCEEDED,
            field="goal_count",
            count=5,
            limit=4,
        )
        self.assertEqual(failure.status, 409)
        self.assertFalse(failure.retryable)
        self.assertEqual(failure.check, "budget")
        with self.assertRaises(ValueError):
            deep_retrieval_failure(
                DeepRetrievalErrorCode.BUDGET_EXCEEDED,
                field="raw_provider_output",
            )

    def test_depth_and_retrieval_mode_are_independent(self) -> None:
        goal = Goal(goal_id="g1", question="one", query="one")
        vector = PlanEnvelope(
            plan_id="plan",
            question_ref="question-ref",
            workflow_depth=WorkflowDepth.DEEP,
            retrieval_mode=RetrievalMode.VECTOR,
            goals=(goal,),
            budget=_budget(),
            capability_fingerprint=_FINGERPRINT,
        )
        hybrid = PlanEnvelope(
            plan_id="plan",
            question_ref="question-ref",
            workflow_depth=WorkflowDepth.DEEP,
            retrieval_mode=RetrievalMode.HYBRID,
            goals=(goal,),
            budget=_budget(),
            capability_fingerprint=_FINGERPRINT,
        )
        self.assertEqual(vector.workflow_depth, hybrid.workflow_depth)
        self.assertNotEqual(vector.retrieval_mode, hybrid.retrieval_mode)
        self.assertNotEqual(vector.canonical_hash, hybrid.canonical_hash)

    def test_budget_hard_bounds_are_server_owned(self) -> None:
        invalid = (
            {"max_goals": 5},
            {"max_query_variants_per_goal": 2},
            {"max_query_variants_per_goal": True},
            {"max_adaptive_waves": 1},
            {"max_adaptive_waves": False},
            {"max_repairs": 2},
            {"deadline_seconds": 0},
            {"deadline_seconds": 120.1},
            {"max_parallelism": 0},
            {"max_parallelism": 2},
            {"max_parallelism": True},
            {"max_retrieval_calls": 0},
            {"max_retrieval_calls": 2},
            {"max_retrieval_calls": True},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                _budget(**changes)

    def test_plan_rejects_duplicate_unknown_and_cyclic_dependencies(self) -> None:
        first = Goal(goal_id="g1", question="one", query="one")
        duplicate = Goal(goal_id="g1", question="other", query="other")
        with self.assertRaises(ValueError):
            PlanEnvelope("p", "q", "standard", "vector", (first, duplicate), _budget(), _FINGERPRINT)
        with self.assertRaises(ValueError):
            PlanEnvelope(
                "p",
                "q",
                "standard",
                "vector",
                (Goal("g1", "one", "one", depends_on=("unknown",)),),
                _budget(),
                _FINGERPRINT,
            )
        with self.assertRaises(ValueError):
            PlanEnvelope(
                "p",
                "q",
                "standard",
                "vector",
                (
                    Goal("g1", "one", "one", depends_on=("g2",)),
                    Goal("g2", "two", "two", depends_on=("g1",)),
                ),
                _budget(),
                _FINGERPRINT,
            )
        with self.assertRaises(ValueError):
            PlanEnvelope(
                "p",
                "q",
                "deep",
                "vector",
                (
                    Goal("g1", "one", "Same query"),
                    Goal("g2", "two", "same   query"),
                ),
                _budget(),
                _FINGERPRINT,
            )

    def test_canonical_evidence_key_requires_all_identity_dimensions(self) -> None:
        key = CanonicalEvidenceKey("revision", "target", "group-1", "image", "group")
        self.assertIn('"revision_id":"revision"', key.canonical)
        self.assertIn('"target_id":"target"', key.canonical)
        self.assertIn('"chunk_or_group_id":"group-1"', key.canonical)
        self.assertIn('"representation_id":"image"', key.canonical)
        with self.assertRaises(ValueError):
            CanonicalEvidenceKey("revision", "target", "", "text")

        first = CanonicalEvidenceKey("r|target=x", "y", "c", "text")
        second = CanonicalEvidenceKey("r", "x|target=y", "c", "text")
        self.assertNotEqual(first.canonical, second.canonical)
        self.assertNotEqual(
            CanonicalEvidenceKey("revision", "target", "chunk", "text", plan_id="p1").canonical,
            CanonicalEvidenceKey("revision", "target", "chunk", "text", plan_id="p2").canonical,
        )

    def test_coverage_status_combinations_and_allowlist(self) -> None:
        key = CanonicalEvidenceKey("revision", "target", "chunk-1", "text", plan_id="p")
        supported = GoalCoverage("g1", CoverageStatus.SUPPORTED, (key,))
        report = CoverageReport(
            "p", (supported,), plan_goal_ids=("g1",), allowed_evidence_keys=(key,)
        )
        self.assertEqual(report.admitted_evidence_keys, (key,))
        with self.assertRaises(ValueError):
            GoalCoverage("g1", CoverageStatus.SUPPORTED, ())
        with self.assertRaises(ValueError):
            GoalCoverage("g1", CoverageStatus.MISSING, (key,), ("missing",))
        with self.assertRaises(ValueError):
            CoverageReport(
                "p",
                (GoalCoverage("g1", CoverageStatus.SUPPORTED, (key,)),),
                plan_goal_ids=("g1",),
                allowed_evidence_keys=(CanonicalEvidenceKey("revision", "target", "other", "text", plan_id="p"),),
            )
        with self.assertRaises(ValueError):
            CoverageReport("p", (supported,), plan_goal_ids=("g2",), allowed_evidence_keys=(key,))
        with self.assertRaises(ValueError):
            CoverageReport("p", (supported,), plan_goal_ids=("g1",))
        unbound = CanonicalEvidenceKey("revision", "target", "chunk-1", "text")
        with self.assertRaises(ValueError):
            CoverageReport(
                "p",
                (GoalCoverage("g1", CoverageStatus.SUPPORTED, (unbound,)),),
                plan_goal_ids=("g1",),
                allowed_evidence_keys=(unbound,),
            )
        other = CanonicalEvidenceKey("revision", "target", "chunk-2", "text", plan_id="p")
        with self.assertRaises(ValueError):
            GoalCoverage(
                "g1",
                CoverageStatus.CONFLICT,
                (key, other),
                conflict_keys=(key, CanonicalEvidenceKey("revision", "target", "outside", "text", plan_id="p")),
            )

    def test_hash_and_json_are_stable_and_trace_time_independent(self) -> None:
        goal = Goal("g1", "one", "one")
        first = PlanEnvelope("p", "q", "standard", "vector", (goal,), _budget(), _FINGERPRINT)
        second = PlanEnvelope(
            "p", "q", "standard", "vector", (goal,), _budget(), _FINGERPRINT,
        )
        self.assertEqual(first.canonical_hash, second.canonical_hash)
        self.assertEqual(first.canonical_json, canonical_json(first.canonical_payload))

    def test_plan_and_coverage_failures_are_distinct_versioned_facts(self) -> None:
        fallback = single_goal_fallback("question-ref")
        failure = coverage_assessment_failure()
        self.assertEqual(fallback.outcome.value, "single_goal_fallback")
        self.assertEqual(fallback.goal_count, 1)
        self.assertEqual(failure.outcome.value, "assessment_failure")
        self.assertIsInstance(failure, AssessmentFailure)
        self.assertNotEqual(fallback.wire_version, failure.wire_version)
        with self.assertRaises(ValueError):
            single_goal_fallback("question-ref", reason="raw provider output: secret")
        with self.assertRaises(ValueError):
            coverage_assessment_failure(reason="raw provider output: secret")


if __name__ == "__main__":
    unittest.main()
