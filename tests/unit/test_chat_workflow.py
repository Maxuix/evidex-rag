from __future__ import annotations

import unittest

from rag_kb.domain import (
    ChatResolvedMode,
    ChatRouteReason,
    ChatRouteStatus,
    ChatWorkflowMode,
    ChatWorkflowState,
    ResearchAspect,
    ResearchAspectStatus,
    ResearchResult,
    ResearchStatus,
    ResearchTerminationReason,
    SearchTrace,
    SearchTraceStep,
    hydrate_chat_workflow_configuration,
    hydrate_chat_workflow_state,
    initial_chat_workflow,
)


class ChatWorkflowContractTests(unittest.TestCase):
    def test_safe_research_degradation_round_trips_and_rejects_content(self) -> None:
        result = ResearchResult(
            status=ResearchStatus.NO_EVIDENCE,
            selected_evidence_keys=(),
            aspects=(
                ResearchAspect(
                    aspect="reliable_research_result",
                    status=ResearchAspectStatus.MISSING,
                ),
            ),
            covered_aspects=(),
            missing_aspects=("reliable_research_result",),
            conflicts=(),
            termination_reason=ResearchTerminationReason.NO_EVIDENCE,
            degradation_reason="retrieval_agent_action_wire_schema_invalid",
        )
        state = ChatWorkflowState(
            resolved_mode=ChatResolvedMode.AGENT,
            route_status=ChatRouteStatus.NOT_APPLICABLE,
            research_result=result,
            search_trace=SearchTrace(
                steps=(),
                decision_rounds=1,
                retrieval_calls=0,
                verifier_calls=0,
                evidence_count=0,
            ),
        )
        self.assertEqual(hydrate_chat_workflow_state(state.as_dict()), state)

        legacy = state.as_dict()
        assert legacy["research_result"] is not None
        legacy["research_result"].pop("degradation_reason")
        hydrated = hydrate_chat_workflow_state(legacy)
        assert hydrated.research_result is not None
        self.assertIsNone(hydrated.research_result.degradation_reason)

        with self.assertRaises(ValueError):
            ResearchResult(
                status=ResearchStatus.NO_EVIDENCE,
                selected_evidence_keys=(),
                aspects=result.aspects,
                covered_aspects=(),
                missing_aspects=result.missing_aspects,
                conflicts=(),
                termination_reason=ResearchTerminationReason.NO_EVIDENCE,
                degradation_reason="PRIVATE model output",
            )

    def test_initial_modes_are_deterministic_and_round_trip(self) -> None:
        expected = {
            ChatWorkflowMode.SIMPLE: (
                ChatResolvedMode.SIMPLE,
                ChatRouteStatus.NOT_APPLICABLE,
            ),
            ChatWorkflowMode.AGENT: (
                ChatResolvedMode.AGENT,
                ChatRouteStatus.NOT_APPLICABLE,
            ),
            ChatWorkflowMode.AUTO: (
                ChatResolvedMode.PENDING,
                ChatRouteStatus.PENDING,
            ),
        }
        for mode, (resolved, route_status) in expected.items():
            with self.subTest(mode=mode):
                configuration, state = initial_chat_workflow(mode)
                self.assertEqual(
                    hydrate_chat_workflow_configuration(configuration.as_dict()),
                    configuration,
                )
                self.assertEqual(
                    hydrate_chat_workflow_state(state.as_dict()), state
                )
                self.assertIs(state.resolved_mode, resolved)
                self.assertIs(state.route_status, route_status)

    def test_verified_research_and_bounded_trace_round_trip(self) -> None:
        result = ResearchResult(
            status=ResearchStatus.PARTIAL,
            selected_evidence_keys=("chunk:one",),
            aspects=(
                ResearchAspect(
                    aspect="current rule",
                    status=ResearchAspectStatus.SUPPORTED,
                    evidence_keys=("chunk:one",),
                ),
                ResearchAspect(
                    aspect="exception",
                    status=ResearchAspectStatus.MISSING,
                ),
            ),
            covered_aspects=("current rule",),
            missing_aspects=("exception",),
            conflicts=(),
            termination_reason=ResearchTerminationReason.NO_PROGRESS,
        )
        trace = SearchTrace(
            steps=(
                SearchTraceStep(
                    observation_id="observation_1",
                    objective="Find the current rule",
                    queries=("current rule",),
                    based_on_observation_ids=(),
                    result="evidence_found",
                    new_evidence_count=1,
                ),
            ),
            decision_rounds=1,
            retrieval_calls=1,
            verifier_calls=1,
            evidence_count=1,
            adjacency_loaded_count=2,
            adjacency_selected_count=1,
        )
        state = ChatWorkflowState(
            resolved_mode=ChatResolvedMode.AGENT,
            route_status=ChatRouteStatus.RESOLVED,
            route_reason_codes=(ChatRouteReason.EVIDENCE_UNCERTAIN,),
            research_result=result,
            search_trace=trace,
        )
        self.assertEqual(hydrate_chat_workflow_state(state.as_dict()), state)

        legacy = state.as_dict()
        assert legacy["search_trace"] is not None
        legacy["search_trace"].pop("adjacency_loaded_count")
        legacy["search_trace"].pop("adjacency_selected_count")
        hydrated_legacy = hydrate_chat_workflow_state(legacy)
        assert hydrated_legacy.search_trace is not None
        self.assertEqual(hydrated_legacy.search_trace.adjacency_loaded_count, 0)
        self.assertEqual(hydrated_legacy.search_trace.adjacency_selected_count, 0)

    def test_hydration_and_cross_field_invariants_reject_invalid_state(self) -> None:
        _, state = initial_chat_workflow(ChatWorkflowMode.SIMPLE)
        invalid_fields = state.as_dict()
        invalid_fields["raw_reasoning"] = "must not be stored"
        with self.assertRaises(ValueError):
            hydrate_chat_workflow_state(invalid_fields)

        inconsistent = state.as_dict()
        inconsistent["research_result"] = {
            "version": "research_result_v1",
            "status": "no_evidence",
            "selected_evidence_keys": [],
            "aspects": [],
            "covered_aspects": [],
            "missing_aspects": [],
            "conflicts": [],
            "termination_reason": "no_evidence",
        }
        with self.assertRaises(ValueError):
            hydrate_chat_workflow_state(inconsistent)

        with self.assertRaises(ValueError):
            SearchTrace(
                steps=(),
                decision_rounds=0,
                retrieval_calls=-1,
                verifier_calls=0,
                evidence_count=0,
            )

        invalid_route_pairs = (
            (ChatResolvedMode.SIMPLE, ChatRouteStatus.PENDING),
            (ChatResolvedMode.PENDING, ChatRouteStatus.NOT_APPLICABLE),
        )
        for resolved_mode, route_status in invalid_route_pairs:
            with self.subTest(
                resolved_mode=resolved_mode,
                route_status=route_status,
            ):
                with self.assertRaises(ValueError):
                    ChatWorkflowState(
                        resolved_mode=resolved_mode,
                        route_status=route_status,
                    )


if __name__ == "__main__":
    unittest.main()
