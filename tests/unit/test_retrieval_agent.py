from __future__ import annotations

import json
import unittest
from dataclasses import replace

from rag_kb.answering.pipeline_steps import AdaptiveEvidenceAssessmentStep
from rag_kb.domain import (
    ChatModelOperation,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatPipelineState,
    ChatWorkflowMode,
    ContextualizedQuery,
    EvidenceCoverage,
    ErrorCode,
    QueryContextStatus,
    QueryRewriteSource,
    ResearchStatus,
    initial_chat_workflow,
)
from rag_kb.retrieval.agent import (
    WORKFLOW_STATE_ARTIFACT,
    RetrievalAgentService,
    _agent_request,
    _repair_request,
    _verification_request,
    evidence_key,
)
from tests.unit.test_answering import _Model, _context, _pack, _response


class _Retriever:
    def __init__(self, packs) -> None:
        self.packs = list(packs)
        self.queries: list[str] = []

    async def retrieve_query(self, context, query, *, top_k_override=None):
        del context, top_k_override
        self.queries.append(query)
        if not self.packs:
            raise AssertionError("unexpected retrieval call")
        return self.packs.pop(0)


class _FailingRetriever:
    async def retrieve_query(self, context, query, *, top_k_override=None):
        del context, query, top_k_override
        raise ChatPipelineExecutionError(
            ErrorCode.CHAT_REVISION_MISMATCH,
            phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
            diagnostic={"check": "revision"},
        )


class _Progress:
    def __init__(self) -> None:
        self.events: list[tuple[object, object, object]] = []

    async def show(self, stage, activity, *, facts=None, completed=()) -> None:
        del completed
        self.events.append((stage, activity, facts))


def _agent_context(*, verifier_continuations: int = 1):
    context = _context(insufficiency="partial_answer")
    configuration, state = initial_chat_workflow(ChatWorkflowMode.AGENT)
    configuration = replace(
        configuration,
        budget=replace(
            configuration.budget,
            verifier_continuations=verifier_continuations,
        ),
    )
    return replace(
        context,
        workflow_configuration=configuration.as_dict(),
        workflow_state=state.as_dict(),
    )


def _query_context(context):
    return ContextualizedQuery(
        version="contextual_query_v2",
        status=QueryContextStatus.ORIGINAL,
        original_query=context.query,
        standalone_query=context.query,
        context_hash=context.conversation_context.content_hash,
        rewrite_source=QueryRewriteSource.ORIGINAL,
    )


def _search(objective: str, queries) -> str:
    return json.dumps(
        {
            "version": "retrieval_agent_action_v1",
            "action": "search",
            "objective": objective,
            "queries": queries,
            "proposed_reason": None,
            "selected_evidence_keys": [],
        }
    )


def _finish(keys, reason: str = "sufficient") -> str:
    return json.dumps(
        {
            "version": "retrieval_agent_action_v1",
            "action": "finish",
            "objective": None,
            "queries": [],
            "proposed_reason": reason,
            "selected_evidence_keys": list(keys),
        }
    )


def _verification(
    *, status: str, keys=(), missing=(), conflicts=()
) -> str:
    aspect_status = {
        "sufficient": "supported",
        "partial": "partial",
        "no_evidence": "missing",
        "conflict": "conflict",
        "premise_unsupported": "conflict",
    }[status]
    return json.dumps(
        {
            "version": "research_result_verification_v1",
            "status": status,
            "aspects": [
                {
                    "aspect": "question",
                    "status": aspect_status,
                    "evidence_keys": list(keys),
                }
            ],
            "missing_aspects": list(missing),
            "conflicts": list(conflicts),
        }
    )


class RetrievalAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_json_mode_requests_explicitly_require_json(self) -> None:
        context = _agent_context()
        configuration, _ = initial_chat_workflow(ChatWorkflowMode.AGENT)
        evidence = _pack(context, "policy owner").evidence
        action = _agent_request(
            context,
            _query_context(context),
            configuration,
            [],
            evidence,
            decision_rounds=1,
            retrieval_calls=0,
        )
        verification = _verification_request(context, evidence)
        repaired_action = _repair_request(
            action,
            "{}",
            schema=action.output_schema,
        )
        repaired_verification = _repair_request(
            verification,
            "{}",
            schema=verification.output_schema,
        )

        for request in (
            action,
            verification,
            repaired_action,
            repaired_verification,
        ):
            with self.subTest(schema=request.output_schema):
                self.assertIn(
                    "json",
                    "\n".join(message.content for message in request.messages).lower(),
                )
                self.assertIs(request.thinking_enabled, False)
        action_prompt = action.messages[0].content
        for field in (
            "version",
            "action",
            "objective",
            "queries",
            "proposed_reason",
            "selected_evidence_keys",
        ):
            self.assertIn(field, action_prompt)
        verification_prompt = verification.messages[0].content
        for field in (
            "version",
            "status",
            "aspects",
            "missing_aspects",
            "conflicts",
            "evidence_keys",
        ):
            self.assertIn(field, verification_prompt)
        self.assertIn("concise answer-target topic labels", verification_prompt)
        self.assertIn("same language as answer_target", verification_prompt)

    async def test_truncated_agent_action_retries_with_more_output_budget(
        self,
    ) -> None:
        context = _agent_context()
        truncated = replace(
            _response('{"_response_truncated":true}'),
            finish_reason="length",
        )
        model = _Model(truncated, truncated)

        with self.assertRaises(ChatPipelineExecutionError) as raised:
            await _service(model, _Retriever(())).research(
                context, _query_context(context)
            )

        self.assertEqual(
            raised.exception.diagnostic,
            {"check": "retrieval_agent_action_truncated"},
        )
        self.assertEqual(
            [request.max_output_tokens for request in model.requests],
            [768, 1536],
        )
        self.assertTrue(
            all(request.thinking_enabled is False for request in model.requests)
        )

    async def test_retrieval_failure_retains_completed_agent_usage(self) -> None:
        context = _agent_context()
        model = _Model(
            _response(
                _search(
                    "find owner",
                    [{"query": "owner", "based_on_observation_ids": []}],
                )
            )
        )

        with self.assertRaises(ChatPipelineExecutionError) as raised:
            await _service(model, _FailingRetriever()).research(
                context, _query_context(context)
            )

        self.assertIs(raised.exception.code, ErrorCode.CHAT_REVISION_MISMATCH)
        self.assertEqual(len(raised.exception.model_calls), 1)
        self.assertIs(
            raised.exception.model_calls[0].operation,
            ChatModelOperation.RETRIEVAL_AGENT,
        )

    async def test_multi_view_queries_run_in_parallel_and_rrf_is_bounded(self) -> None:
        context = _agent_context()
        first = _pack(context, "policy owner", "shared evidence")
        second_unique = _pack(context, "policy deadline")
        shared = replace(first.evidence[1], rank=2)
        second = replace(
            second_unique,
            evidence=(second_unique.evidence[0], shared),
        )
        expected_keys = (
            evidence_key(first.evidence[0]),
            evidence_key(first.evidence[1]),
            evidence_key(second.evidence[0]),
        )
        model = _Model(
            _response(
                _search(
                    "cover owner and deadline",
                    [
                        {"query": "policy owner", "based_on_observation_ids": []},
                        {"query": "policy deadline", "based_on_observation_ids": []},
                    ],
                )
            ),
            _response(_finish(expected_keys), request_id="finish"),
            _response(
                _verification(status="sufficient", keys=expected_keys),
                request_id="verify",
            ),
        )
        retriever = _Retriever((first, second))

        outcome = await _service(model, retriever).research(
            context, _query_context(context)
        )

        self.assertEqual(retriever.queries, ["policy owner", "policy deadline"])
        self.assertEqual(len(outcome.evidence_pack.evidence), 3)
        self.assertEqual(
            outcome.evidence_pack.evidence[0].index_chunk_id,
            first.evidence[1].index_chunk_id,
        )
        self.assertTrue(
            all(item.fusion_score is not None for item in outcome.evidence_pack.evidence)
        )
        assert outcome.workflow_state.research_result is not None
        self.assertEqual(
            outcome.workflow_state.research_result.status, ResearchStatus.SUFFICIENT
        )

    async def test_progress_exposes_decisions_but_never_evidence_text(self) -> None:
        context = _agent_context()
        pack = _pack(context, "PRIVATE EVIDENCE EXCERPT MUST STAY LOCAL")
        key = evidence_key(pack.evidence[0])
        model = _Model(
            _response(
                _search(
                    "find policy owner",
                    [{"query": "policy owner", "based_on_observation_ids": []}],
                )
            ),
            _response(_finish((key,)), request_id="finish"),
            _response(
                _verification(status="sufficient", keys=(key,)),
                request_id="verify",
            ),
        )
        progress = _Progress()

        await _service(model, _Retriever((pack,))).research(
            context,
            _query_context(context),
            progress=progress,  # type: ignore[arg-type]
        )

        rendered = repr(progress.events)
        self.assertIn("find policy owner", rendered)
        self.assertIn("FINISH_RESEARCH", rendered)
        self.assertNotIn("PRIVATE EVIDENCE EXCERPT", rendered)

    async def test_multi_hop_requires_a_known_observation_reference(self) -> None:
        context = _agent_context()
        first = _pack(context, "Acquirer was Beta")
        second = _pack(context, "Beta revenue was 10")
        keys = (
            evidence_key(first.evidence[0]),
            evidence_key(second.evidence[0]),
        )
        model = _Model(
            _response(
                _search(
                    "identify acquirer",
                    [{"query": "acquisition party", "based_on_observation_ids": []}],
                )
            ),
            _response(
                _search(
                    "find acquired-year revenue",
                    [
                        {
                            "query": "Beta revenue in acquisition year",
                            "based_on_observation_ids": ["obs_1"],
                        }
                    ],
                ),
                request_id="hop",
            ),
            _response(_finish(keys), request_id="finish"),
            _response(_verification(status="sufficient", keys=keys), request_id="verify"),
        )
        retriever = _Retriever((first, second))

        outcome = await _service(model, retriever).research(
            context, _query_context(context)
        )

        assert outcome.workflow_state.search_trace is not None
        self.assertEqual(
            outcome.workflow_state.search_trace.steps[1].based_on_observation_ids,
            ("obs_1",),
        )
        self.assertEqual(outcome.workflow_state.search_trace.retrieval_calls, 2)

    async def test_verifier_can_drive_only_one_continuation(self) -> None:
        context = _agent_context()
        first = _pack(context, "owner")
        second = _pack(context, "deadline")
        first_key = evidence_key(first.evidence[0])
        keys = (first_key, evidence_key(second.evidence[0]))
        model = _Model(
            _response(_search("owner", [{"query": "owner", "based_on_observation_ids": []}])),
            _response(_finish((first_key,)), request_id="finish-1"),
            _response(
                _verification(status="partial", keys=(first_key,), missing=("deadline",)),
                request_id="verify-1",
            ),
            _response(
                _search(
                    "deadline",
                    [
                        {
                            "query": "deadline",
                            "based_on_observation_ids": ["verification_1"],
                        }
                    ],
                ),
                request_id="continue",
            ),
            _response(_finish(keys), request_id="finish-2"),
            _response(_verification(status="sufficient", keys=keys), request_id="verify-2"),
        )
        retriever = _Retriever((first, second))

        outcome = await _service(model, retriever).research(
            context, _query_context(context)
        )

        assert outcome.workflow_state.search_trace is not None
        self.assertEqual(outcome.workflow_state.search_trace.verifier_calls, 2)
        self.assertEqual(
            sum(
                item.result == "verification_gap"
                for item in outcome.workflow_state.search_trace.steps
            ),
            1,
        )

    async def test_conflict_can_drive_one_resolution_search(self) -> None:
        context = _agent_context()
        first = _pack(context, "conflicting policy")
        second = _pack(context, "authoritative policy")
        first_key = evidence_key(first.evidence[0])
        keys = (first_key, evidence_key(second.evidence[0]))
        model = _Model(
            _response(
                _search(
                    "find policy",
                    [{"query": "policy", "based_on_observation_ids": []}],
                )
            ),
            _response(_finish((first_key,)), request_id="finish-conflict"),
            _response(
                _verification(
                    status="conflict",
                    keys=(first_key,),
                    conflicts=("policy",),
                ),
                request_id="verify-conflict",
            ),
            _response(
                _search(
                    "resolve conflict",
                    [
                        {
                            "query": "authoritative policy",
                            "based_on_observation_ids": ["verification_1"],
                        }
                    ],
                ),
                request_id="resolve",
            ),
            _response(_finish(keys), request_id="finish-resolved"),
            _response(
                _verification(status="sufficient", keys=keys),
                request_id="verify-resolved",
            ),
        )

        outcome = await _service(model, _Retriever((first, second))).research(
            context, _query_context(context)
        )

        assert outcome.workflow_state.research_result is not None
        assert outcome.workflow_state.search_trace is not None
        self.assertIs(
            outcome.workflow_state.research_result.status,
            ResearchStatus.SUFFICIENT,
        )
        self.assertEqual(outcome.workflow_state.search_trace.verifier_calls, 2)

    async def test_unknown_observation_never_reaches_the_tool(self) -> None:
        context = _agent_context()
        model = _Model(
            _response(
                _search(
                    "escape scope",
                    [
                        {
                            "query": "unknown dependency",
                            "based_on_observation_ids": ["obs_unknown"],
                        }
                    ],
                )
            ),
            _response(
                _verification(status="no_evidence", missing=("question",)),
                request_id="verify",
            ),
        )
        retriever = _Retriever(())

        outcome = await _service(model, retriever).research(
            context, _query_context(context)
        )

        self.assertEqual(retriever.queries, [])
        assert outcome.workflow_state.research_result is not None
        self.assertEqual(
            outcome.workflow_state.research_result.status,
            ResearchStatus.NO_EVIDENCE,
        )
        self.assertEqual(
            outcome.workflow_state.research_result.termination_reason.value,
            "no_evidence",
        )

    async def test_duplicate_query_stops_with_no_progress(self) -> None:
        context = _agent_context()
        pack = _pack(context, "owner evidence")
        key = evidence_key(pack.evidence[0])
        model = _Model(
            _response(
                _search(
                    "find owner",
                    [{"query": "Owner", "based_on_observation_ids": []}],
                )
            ),
            _response(
                _search(
                    "repeat owner",
                    [{"query": "  owner  ", "based_on_observation_ids": []}],
                ),
                request_id="duplicate",
            ),
            _response(
                _verification(
                    status="partial",
                    keys=(key,),
                    missing=("deadline",),
                ),
                request_id="verify",
            ),
        )
        retriever = _Retriever((pack,))

        outcome = await _service(model, retriever).research(
            context, _query_context(context)
        )

        self.assertEqual(retriever.queries, ["Owner"])
        assert outcome.workflow_state.research_result is not None
        self.assertEqual(
            outcome.workflow_state.research_result.termination_reason.value,
            "no_progress",
        )

    async def test_decision_budget_and_verifier_statuses_have_stable_endings(self) -> None:
        context = _agent_context()
        packs = tuple(_pack(context, f"evidence {index}") for index in range(4))
        # All four one-item rankings have the same RRF score while the frozen
        # top_k is three, so choose a key guaranteed to survive key tie-break.
        budget_key = min(evidence_key(pack.evidence[0]) for pack in packs)
        model = _Model(
            *(
                _response(
                    _search(
                        f"round {index}",
                        [
                            {
                                "query": f"distinct query {index}",
                                "based_on_observation_ids": [],
                            }
                        ],
                    ),
                    request_id=f"search-{index}",
                )
                for index in range(4)
            ),
            _response(
                _verification(
                    status="partial",
                    keys=(budget_key,),
                    missing=("exception",),
                ),
                request_id="verify-budget",
            ),
        )
        budget_outcome = await _service(model, _Retriever(packs)).research(
            context, _query_context(context)
        )
        assert budget_outcome.workflow_state.research_result is not None
        self.assertEqual(
            budget_outcome.workflow_state.research_result.termination_reason.value,
            "budget_exhausted",
        )
        self.assertEqual(
            budget_outcome.workflow_state.research_result.selected_evidence_keys,
            (budget_key,),
        )

        for status, termination in (
            ("conflict", "conflict_unresolved"),
            ("premise_unsupported", "premise_unsupported"),
        ):
            with self.subTest(status=status):
                context = _agent_context(verifier_continuations=0)
                pack = _pack(context, f"{status} evidence")
                key = evidence_key(pack.evidence[0])
                outcome = await _service(
                    _Model(
                        _response(
                            _search(
                                "verify premise",
                                [
                                    {
                                        "query": status,
                                        "based_on_observation_ids": [],
                                    }
                                ],
                            )
                        ),
                        _response(_finish((key,)), request_id="finish"),
                        _response(
                            _verification(
                                status=status,
                                keys=(key,),
                                conflicts=("question",),
                            ),
                            request_id="verify",
                        ),
                    ),
                    _Retriever((pack,)),
                ).research(context, _query_context(context))
                assert outcome.workflow_state.research_result is not None
                self.assertEqual(
                    outcome.workflow_state.research_result.termination_reason.value,
                    termination,
                )
                assessed = await AdaptiveEvidenceAssessmentStep(
                    0.6, 0.45, 0.25
                ).run(
                    ChatPipelineState(
                        context=context,
                        query_context=_query_context(context),
                        evidence_pack=outcome.evidence_pack,
                        artifacts={
                            WORKFLOW_STATE_ARTIFACT: outcome.workflow_state
                        },
                    )
                )
                assert assessed.answering is not None
                self.assertIs(
                    assessed.answering.assessment.coverage,
                    (
                        EvidenceCoverage.CONFLICT
                        if status == "conflict"
                        else EvidenceCoverage.NONE
                    ),
                )


def _service(model, retriever) -> RetrievalAgentService:
    return RetrievalAgentService(
        model,
        retriever,
        min_cosine_similarity=0.6,
        min_rerank_score=0.45,
        cross_modal_min_cosine_similarity=0.25,
    )
