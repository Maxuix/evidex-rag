from __future__ import annotations

import json
import unittest
from dataclasses import replace
from uuid import uuid4

from rag_kb.answering.pipeline_steps import AdaptiveEvidenceAssessmentStep
from rag_kb.domain import (
    ChatModelExecutionError,
    ChatModelOperation,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatPipelineState,
    ChatWorkflowMode,
    ContextualizedQuery,
    EvidenceCoverage,
    ResearchAspect,
    ResearchAspectStatus,
    ResearchResultVerification,
    EvidenceScoreKind,
    ErrorCode,
    QueryContextStatus,
    QueryRewriteSource,
    ResearchStatus,
    initial_chat_workflow,
)
from rag_kb.retrieval.agent import (
    WORKFLOW_STATE_ARTIFACT,
    RetrievalAgentService,
    _AgentActionFailureReason,
    _AgentActionValidationError,
    _VerificationFailureReason,
    _VerificationValidationError,
    _agent_request,
    _expand_verification_evidence,
    _fused_evidence,
    _repair_request,
    _parse_agent_action,
    _parse_verification,
    _round_robin_document_requests,
    _verification_request,
    _apply_verification_gate,
    evidence_key,
    project_agent_evidence,
    project_evidence_text,
)
from rag_kb.retrieval.calculator import evaluate_decimal_expression
from tests.unit.test_answering import _Model, _context, _pack, _response


class _Retriever:
    def __init__(self, packs, *, adjacency_results=()) -> None:
        self.packs = list(packs)
        self.adjacency_results = list(adjacency_results)
        self.queries: list[str] = []
        self.adjacency_calls: list[tuple[object, ...]] = []

    async def retrieve_query(
        self, context, query, *, top_k_override=None, document_ids=()
    ):
        del context, top_k_override
        self.queries.append(query)
        if not self.packs:
            raise AssertionError("unexpected retrieval call")
        return self.packs.pop(0)

    async def retrieve_adjacent(self, context, anchors):
        del context
        self.adjacency_calls.append(anchors)
        if self.adjacency_results:
            return self.adjacency_results.pop(0)
        return ()


class _FailingRetriever:
    async def retrieve_query(
        self, context, query, *, top_k_override=None, document_ids=()
    ):
        del context, query, top_k_override, document_ids
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


class _ProviderFailingModel:
    def __init__(self, diagnostic: dict[str, object]) -> None:
        self.diagnostic = diagnostic
        self.calls = 0

    async def complete(self, request):
        del request
        self.calls += 1
        raise ChatModelExecutionError(
            ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
            diagnostic=self.diagnostic,
        )


def _agent_context(
    *,
    verifier_continuations: int = 1,
    decision_rounds: int = 4,
):
    context = _context(insufficiency="partial_answer")
    configuration, state = initial_chat_workflow(ChatWorkflowMode.AGENT)
    configuration = replace(
        configuration,
        budget=replace(
            configuration.budget,
            verifier_continuations=verifier_continuations,
            decision_rounds=decision_rounds,
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
            "version": "retrieval_agent_action_v2",
            "action": "search",
            "objective": objective,
            "queries": queries,
            "proposed_reason": None,
            "selected_evidence_keys": [],
            "calculation": None,
        }
    )


def _finish(keys, reason: str = "sufficient") -> str:
    return json.dumps(
        {
            "version": "retrieval_agent_action_v2",
            "action": "finish",
            "objective": None,
            "queries": [],
            "proposed_reason": reason,
            "selected_evidence_keys": list(keys),
            "calculation": None,
        }
    )


def _calculate(expression: str, source_evidence_keys) -> str:
    return json.dumps(
        {
            "version": "retrieval_agent_action_v2",
            "action": "calculate",
            "objective": None,
            "queries": [],
            "proposed_reason": None,
            "selected_evidence_keys": [],
            "calculation": {
                "expression": expression,
                "source_evidence_keys": list(source_evidence_keys),
            },
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


def _adjacent(anchor, *, offset: int, text: str, chunk_id=None):
    return replace(
        anchor,
        rank=1,
        index_chunk_id=chunk_id or uuid4(),
        ordinal=anchor.ordinal + offset,
        text=text,
        source_location={"adjacent_ordinal": anchor.ordinal + offset},
        score=0.0,
        score_kind=EvidenceScoreKind.ADJACENCY,
        vector_similarity=None,
        text_space_rank=None,
        lexical_rank=None,
        cross_modal_rank=None,
        fusion_score=None,
        adjacency_anchor_index_chunk_id=anchor.index_chunk_id,
        adjacency_offset=offset,
    )


class RetrievalAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_timeout_and_403_do_not_enter_schema_degradation(
        self,
    ) -> None:
        context = _agent_context()
        for diagnostic in (
            {"check": "total_timeout"},
            {"check": "http_status", "http_status": 403, "retryable": False},
        ):
            with self.subTest(diagnostic=diagnostic):
                model = _ProviderFailingModel(diagnostic)
                with self.assertRaises(ChatPipelineExecutionError) as raised:
                    await _service(model, _Retriever(())).research(
                        context,
                        _query_context(context),
                    )
                self.assertIs(
                    raised.exception.code,
                    ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
                )
                self.assertEqual(raised.exception.diagnostic, diagnostic)
                self.assertEqual(model.calls, 1)

    def test_agent_action_failure_reasons_are_specific_and_content_safe(self) -> None:
        base = json.loads(
            _search("find fact", [{"query": "fact", "based_on_observation_ids": []}])
        )
        cases = (
            (
                {**base, "objective": None},
                _AgentActionFailureReason.RELEVANT_PAYLOAD_MISSING,
            ),
            (
                {**base, "selected_evidence_keys": ["chunk:hidden"]},
                _AgentActionFailureReason.IRRELEVANT_FIELD_NONEMPTY,
            ),
            (
                {
                    **json.loads(_calculate("1 + 1", ("chunk:source",))),
                    "calculation": {"expression": "1 + 1"},
                },
                _AgentActionFailureReason.CALCULATION_SHAPE_INVALID,
            ),
            (
                {
                    **json.loads(_calculate("1 + 1", ("chunk:source",))),
                    "calculation": {
                        "expression": "1 + 1",
                        "source_evidence_keys": ["chunk:source", "chunk:source"],
                    },
                },
                _AgentActionFailureReason.SOURCE_KEY_SHAPE_INVALID,
            ),
            (
                {**base, "version": "retrieval_agent_action_v1", "action": "calculate"},
                _AgentActionFailureReason.ACTION_VERSION_INCOMPATIBLE,
            ),
            (
                {**base, "unknown_hidden_field": "PRIVATE"},
                _AgentActionFailureReason.WIRE_SCHEMA_INVALID,
            ),
        )
        for payload, reason in cases:
            with self.subTest(reason=reason), self.assertRaises(
                _AgentActionValidationError
            ) as raised:
                _parse_agent_action(json.dumps(payload))
            self.assertIs(raised.exception.reason, reason)
            self.assertNotIn("chunk:hidden", raised.exception.validation_hint)

    def test_agent_action_normalizes_only_empty_irrelevant_fields(self) -> None:
        search = json.loads(
            _search("find fact", [{"query": "fact", "based_on_observation_ids": []}])
        )
        search.update({
            "proposed_reason": "",
            "selected_evidence_keys": None,
            "calculation": {},
        })
        self.assertEqual(_parse_agent_action(json.dumps(search)).action.value, "search")

        calculate = json.loads(_calculate("1 + 1", ("chunk:source",)))
        calculate.update({
            "objective": "",
            "queries": None,
            "proposed_reason": "",
            "selected_evidence_keys": None,
        })
        parsed_calculation = _parse_agent_action(json.dumps(calculate))
        self.assertEqual(parsed_calculation.action.value, "calculate")
        self.assertEqual(
            parsed_calculation.calculation_source_evidence_keys,
            ("chunk:source",),
        )

        finish = json.loads(_finish(("chunk:allowed",)))
        finish.update({"objective": "", "queries": None, "calculation": {}})
        self.assertEqual(_parse_agent_action(json.dumps(finish)).action.value, "finish")

    def test_agent_action_never_normalizes_nonempty_hidden_actions(self) -> None:
        search = json.loads(
            _search("find fact", [{"query": "fact", "based_on_observation_ids": []}])
        )
        search["calculation"] = {
            "expression": "1 + 1",
            "source_evidence_keys": ["chunk:hidden"],
        }
        calculate = json.loads(_calculate("1 + 1", ("chunk:source",)))
        calculate["queries"] = [
            {"query": "hidden search", "based_on_observation_ids": []}
        ]
        finish = json.loads(_finish(("chunk:allowed",)))
        finish["calculation"] = {
            "expression": "1 + 1",
            "source_evidence_keys": ["chunk:hidden"],
        }

        for payload in (search, calculate, finish):
            with self.subTest(action=payload["action"]), self.assertRaises(
                _AgentActionValidationError
            ) as raised:
                _parse_agent_action(json.dumps(payload))
            self.assertIs(
                raised.exception.reason,
                _AgentActionFailureReason.IRRELEVANT_FIELD_NONEMPTY,
            )

    def test_table_projection_preserves_bounded_identity_and_ignores_metadata(
        self,
    ) -> None:
        context = _agent_context()
        base = _pack(context, "placeholder").evidence[0]
        table = (
            "| 项目 | 2022年度 | 2021年度 |\n"
            "| --- | ---: | ---: |\n"
            "| 管理费用 | 229,129,291.07 | 200,000,000.00 |\n"
            "| 研发费用 | 262,081,206.94 | 210,000,000.00 |\n"
        )
        common = {
            "modality": "table",
            "text": table,
            "source_location": {
                "surface_type": "page",
                "surface_start": 15,
                "surface_end": 15,
                "surface_label": "15",
                "url": "PRIVATE",
            },
        }
        common_metadata = {
            "reporting_period": "2022年度",
            "secret_payload": "PRIVATE",
        }
        consolidated = replace(
            base,
            **common,
            hierarchy={"titles": [{"depth": 2, "text": "合并利润表"}]},
            source_metadata={
                **common_metadata,
                "consolidation_scope": "合并",
            },
        )
        parent = replace(
            base,
            index_chunk_id=uuid4(),
            ordinal=base.ordinal + 1,
            **common,
            hierarchy={"titles": [{"depth": 2, "text": "母公司利润表"}]},
            source_metadata={
                **common_metadata,
                "consolidation_scope": "母公司",
            },
        )

        projected = project_agent_evidence(
            (consolidated, parent),
            ("管理费用", "研发费用", "合并利润表", "2022年度"),
        )

        identities = [item["table_identity"] for item in projected]
        self.assertEqual(identities[0]["titles"], ["合并利润表"])
        self.assertEqual(identities[1]["titles"], ["母公司利润表"])
        self.assertEqual(identities[0]["reporting_period"], "2022年度")
        self.assertEqual(identities[0]["consolidation_scope"], "合并")
        self.assertEqual(identities[0]["location"]["surface_start"], 15)
        self.assertEqual(identities[0]["ordinal"], consolidated.ordinal)
        self.assertIn("| 项目 | 2022年度 | 2021年度 |", identities[0]["column_headers"])
        self.assertIn("管理费用", " ".join(identities[0]["target_row_neighbors"]))
        self.assertNotIn("PRIVATE", repr(projected))
        for item in projected:
            content = item["untrusted_excerpt"] + json.dumps(
                item["table_identity"], ensure_ascii=False
            )
            self.assertLessEqual(len(content), 2_400)

    def test_complex_05_consolidated_operands_produce_exact_decimals(self) -> None:
        context = _agent_context()
        evidence = _pack(
            context,
            "原材料 785,646,432.47",
            "合并利润表 管理费用 229,129,291.07 研发费用 262,081,206.94",
        ).evidence
        keys = tuple(evidence_key(item) for item in evidence)

        total = evaluate_decimal_expression(
            "229129291.07 + 262081206.94",
            source_evidence_keys=(keys[1],),
            evidence={key: item for key, item in zip(keys, evidence, strict=True)},
        )
        difference = evaluate_decimal_expression(
            "785646432.47 - (229129291.07 + 262081206.94)",
            source_evidence_keys=keys,
            evidence={key: item for key, item in zip(keys, evidence, strict=True)},
        )

        self.assertEqual(total.result, "491210498.01")
        self.assertEqual(difference.result, "294435934.46")

    def test_verifier_validation_classifies_safe_failure_reasons(self) -> None:
        allowed_key = "chunk:allowed"
        base = json.loads(_verification(status="sufficient", keys=(allowed_key,)))
        cases = {
            _VerificationFailureReason.WIRE_SCHEMA_INVALID: "not-json PRIVATE",
            _VerificationFailureReason.EVIDENCE_NOT_ALLOWED: json.dumps(
                {
                    **base,
                    "aspects": [
                        {
                            "aspect": "question",
                            "status": "supported",
                            "evidence_keys": ["chunk:PRIVATE"],
                        }
                    ],
                }
            ),
            _VerificationFailureReason.SUPPORTED_EVIDENCE_REQUIRED: json.dumps(
                {
                    **base,
                    "aspects": [
                        {
                            "aspect": "question",
                            "status": "supported",
                            "evidence_keys": [],
                        }
                    ],
                }
            ),
            _VerificationFailureReason.DUPLICATE_LIST_ITEMS: json.dumps(
                {
                    **base,
                    "status": "partial",
                    "missing_aspects": ["PRIVATE", "PRIVATE"],
                }
            ),
            _VerificationFailureReason.STATUS_SEMANTICS_INVALID: json.dumps(
                {**base, "missing_aspects": ["PRIVATE"]}
            ),
        }

        for expected, content in cases.items():
            with self.subTest(expected=expected):
                with self.assertRaises(_VerificationValidationError) as raised:
                    _parse_verification(content, frozenset({allowed_key}))
                self.assertIs(raised.exception.reason, expected)
                self.assertNotIn("PRIVATE", str(raised.exception))

    async def test_verifier_repair_receives_fixed_reason_hint(self) -> None:
        context = _agent_context()
        pack = _pack(context, "allowed fact")
        key = evidence_key(pack.evidence[0])
        invalid = json.loads(_verification(status="sufficient", keys=(key,)))
        invalid["aspects"][0]["evidence_keys"] = ["chunk:PRIVATE"]
        model = _Model(
            _response(
                _search("find fact", [{"query": "fact", "based_on_observation_ids": []}])
            ),
            _response(_finish((key,)), request_id="finish"),
            _response(json.dumps(invalid), request_id="verify-invalid"),
            _response(
                _verification(status="sufficient", keys=(key,)),
                request_id="verify-repaired",
            ),
        )

        outcome = await _service(model, _Retriever((pack,))).research(
            context, _query_context(context)
        )

        assert outcome.workflow_state.research_result is not None
        self.assertIs(
            outcome.workflow_state.research_result.status,
            ResearchStatus.SUFFICIENT,
        )
        repair_prompt = model.requests[-1].messages[-1].content
        self.assertIn("selected evidence allowlist", repair_prompt)
        self.assertNotIn("PRIVATE", repair_prompt)

    async def test_verifier_semantics_repair_receives_status_specific_hint(self) -> None:
        context = _agent_context()
        pack = _pack(context, "allowed fact")
        key = evidence_key(pack.evidence[0])
        invalid = json.loads(_verification(status="sufficient", keys=(key,)))
        invalid["missing_aspects"] = ["PRIVATE"]
        model = _Model(
            _response(
                _search("find fact", [{"query": "fact", "based_on_observation_ids": []}])
            ),
            _response(_finish((key,)), request_id="finish"),
            _response(json.dumps(invalid), request_id="verify-invalid"),
            _response(
                _verification(status="sufficient", keys=(key,)),
                request_id="verify-repaired",
            ),
        )

        outcome = await _service(model, _Retriever((pack,))).research(
            context, _query_context(context)
        )

        assert outcome.workflow_state.research_result is not None
        repair_prompt = model.requests[-1].messages[-1].content
        self.assertIn("For status=sufficient", repair_prompt)
        self.assertIn("missing_aspects=[]", repair_prompt)
        self.assertNotIn("PRIVATE", repair_prompt)

    async def test_verifier_second_failure_reports_safe_subreason(self) -> None:
        context = _agent_context()
        pack = _pack(context, "allowed fact")
        key = evidence_key(pack.evidence[0])
        model = _Model(
            _response(
                _search("find fact", [{"query": "fact", "based_on_observation_ids": []}])
            ),
            _response(_finish((key,)), request_id="finish"),
            _response("PRIVATE first invalid", request_id="verify-invalid"),
            _response("PRIVATE second invalid", request_id="verify-repair-invalid"),
        )

        outcome = await _service(model, _Retriever((pack,))).research(
            context, _query_context(context)
        )

        result = outcome.workflow_state.research_result
        assert result is not None
        self.assertEqual(
            result.degradation_reason,
            "research_result_verification_wire_schema_invalid",
        )
        self.assertIs(result.status, ResearchStatus.NO_EVIDENCE)
        self.assertEqual(result.selected_evidence_keys, ())
        self.assertEqual(len(outcome.model_calls), 4)
        self.assertNotIn("PRIVATE", repr(result.as_dict()))

    async def test_verifier_truncation_takes_diagnostic_priority(self) -> None:
        context = _agent_context()
        pack = _pack(context, "allowed fact")
        key = evidence_key(pack.evidence[0])
        truncated = replace(
            _response("PRIVATE invalid", request_id="verify-truncated"),
            finish_reason="length",
        )
        model = _Model(
            _response(
                _search("find fact", [{"query": "fact", "based_on_observation_ids": []}])
            ),
            _response(_finish((key,)), request_id="finish"),
            truncated,
            _response("PRIVATE invalid again", request_id="verify-repair-invalid"),
        )

        outcome = await _service(model, _Retriever((pack,))).research(
            context, _query_context(context)
        )

        result = outcome.workflow_state.research_result
        assert result is not None
        self.assertEqual(
            result.degradation_reason,
            "research_result_verification_truncated",
        )
        self.assertIs(result.status, ResearchStatus.NO_EVIDENCE)
        self.assertEqual(model.requests[-1].max_output_tokens, 2048)

    def test_scoped_search_prioritizes_uncovered_documents(self) -> None:
        first, second, third = (uuid4() for _ in range(3))

        requests = _round_robin_document_requests(
            ("q1", "q2", "q3"),
            document_ids=(first, second, third),
            covered_document_ids=frozenset({first}),
        )

        self.assertEqual(
            requests,
            (("q1", (second,)), ("q2", (third,)), ("q3", (first,))),
        )

    def test_verifier_gate_downgrades_sufficient_when_required_doc_is_missing(self) -> None:
        context = _agent_context()
        first_document = uuid4()
        second_document = uuid4()
        context = replace(
            context,
            retrieval_strategy={
                **context.retrieval_strategy,
                "document_scope": {
                    "status": "resolved",
                    "resolved": [
                        {"document_id": str(first_document)},
                        {"document_id": str(second_document)},
                    ],
                },
            },
        )
        evidence = replace(_pack(context, "one fact").evidence[0], document_id=first_document)
        verification = ResearchResultVerification(
            status=ResearchStatus.SUFFICIENT,
            aspects=(
                ResearchAspect(
                    aspect="question",
                    status=ResearchAspectStatus.SUPPORTED,
                    evidence_keys=(evidence_key(evidence),),
                ),
            ),
            missing_aspects=(),
            conflicts=(),
        )

        gated = _apply_verification_gate(context, verification, (evidence,))

        self.assertIs(gated.status, ResearchStatus.PARTIAL)
        self.assertIn("required_document_not_covered", gated.missing_aspects)

    def test_projection_keeps_late_focus_windows(self) -> None:
        text = (
            "Document title\n"
            + "\n".join(f"background line {index:03d}" for index in range(180))
            + "\nlate-target-1208 late-target-1238 late-target-1260"
            + "\n" + "x" * 500
            + "\nlate-target-1364 late-target-1385 late-target-1800"
        )
        projected = project_evidence_text(
            text,
            ("late-target-1208", "late-target-1364"),
        )
        self.assertIn("late-target-1208", projected)
        self.assertIn("late-target-1364", projected)
        self.assertLessEqual(len(projected), 2400)

    def test_projection_keeps_table_header_late_row_and_decimal_text(self) -> None:
        table = (
            "Report title\n"
            "| Metric | 2022 | 2021 |\n"
            "| --- | ---: | ---: |\n"
            + "\n".join(f"| Noise {index} | 0 | 0 |" for index in range(110))
            + "\n| Total (loan) | 828.8 | 885.3 |\n"
        )
        projected = project_evidence_text(table, ("Total (loan)", "828.8"))
        self.assertIn("| Metric | 2022 | 2021 |", projected)
        self.assertIn("| Total (loan) | 828.8 | 885.3 |", projected)
        self.assertIn("828.8", projected)
        self.assertLessEqual(len(projected), 2400)

    def test_projection_is_deterministic_and_has_total_budget(self) -> None:
        context = _agent_context()
        pack = _pack(context, *(f"target-{index}" for index in range(20)))
        evidence = tuple(
            replace(
                item,
                text=(f"Evidence {index} target-{index} " + "x" * 5000),
            )
            for index, item in enumerate(pack.evidence)
        )
        first = project_agent_evidence(evidence, ("target-12", "decimal"))
        second = project_agent_evidence(evidence, ("target-12", "decimal"))
        self.assertEqual(first, second)
        excerpts = [item["untrusted_excerpt"] for item in first]
        self.assertLessEqual(sum(len(item) for item in excerpts), 24_000)
        self.assertTrue(all(len(item) <= 2_400 for item in excerpts))
        self.assertIn("target-12", excerpts[12])

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
            "calculation",
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
        self.assertIn("table_identity title", action_prompt)
        self.assertIn("table_identity title", verification_prompt)
        self.assertIn("no_progress is server-owned", action_prompt)
        self.assertIn('"selection_limit":3', action.messages[1].content)

    async def test_truncated_agent_action_retries_with_more_output_budget(
        self,
    ) -> None:
        context = _agent_context()
        truncated = replace(
            _response('{"_response_truncated":true}'),
            finish_reason="length",
        )
        model = _Model(truncated, truncated)

        outcome = await _service(model, _Retriever(())).research(
            context, _query_context(context)
        )

        result = outcome.workflow_state.research_result
        assert result is not None
        self.assertEqual(
            result.degradation_reason,
            "retrieval_agent_action_truncated",
        )
        self.assertIs(result.status, ResearchStatus.NO_EVIDENCE)
        self.assertEqual(len(outcome.model_calls), 2)
        self.assertEqual(
            [request.max_output_tokens for request in model.requests],
            [768, 1536],
        )
        self.assertTrue(
            all(request.thinking_enabled is False for request in model.requests)
        )

    async def test_agent_action_repair_receives_action_specific_hint(self) -> None:
        context = _agent_context()
        pack = _pack(context, "allowed fact")
        key = evidence_key(pack.evidence[0])
        invalid = json.loads(
            _search("find fact", [{"query": "fact", "based_on_observation_ids": []}])
        )
        invalid["objective"] = None
        model = _Model(
            _response(json.dumps(invalid), request_id="invalid-action"),
            _response(
                _search("find fact", [{"query": "fact", "based_on_observation_ids": []}]),
                request_id="repaired-action",
            ),
            _response(_finish((key,)), request_id="finish"),
            _response(_verification(status="sufficient", keys=(key,))),
        )

        outcome = await _service(model, _Retriever((pack,))).research(
            context, _query_context(context)
        )

        assert outcome.workflow_state.research_result is not None
        repair_prompt = model.requests[1].messages[-1].content
        self.assertIn("For action=search", repair_prompt)
        self.assertIn("objective must be a non-empty string", repair_prompt)

    async def test_agent_action_second_failure_reports_safe_subreason(self) -> None:
        context = _agent_context()
        invalid = json.loads(
            _search("PRIVATE", [{"query": "PRIVATE", "based_on_observation_ids": []}])
        )
        invalid["objective"] = None
        model = _Model(_response(json.dumps(invalid)), _response(json.dumps(invalid)))

        outcome = await _service(model, _Retriever(())).research(
            context, _query_context(context)
        )

        result = outcome.workflow_state.research_result
        assert result is not None
        self.assertEqual(
            result.degradation_reason,
            (
                "retrieval_agent_action_"
                f"{_AgentActionFailureReason.RELEVANT_PAYLOAD_MISSING.value}"
            ),
        )
        self.assertIs(result.status, ResearchStatus.NO_EVIDENCE)
        self.assertEqual(result.selected_evidence_keys, ())
        self.assertNotIn("PRIVATE", repr(result.as_dict()))

    async def test_agent_action_infers_truncation_when_usage_hits_request_cap(
        self,
    ) -> None:
        context = _agent_context()
        capped = replace(
            _response('{"_response_truncated":true}'),
            finish_reason=None,
            usage={"prompt_tokens": 10, "completion_tokens": 768},
        )
        model = _Model(capped, capped)

        outcome = await _service(model, _Retriever(())).research(
            context, _query_context(context)
        )

        result = outcome.workflow_state.research_result
        assert result is not None
        self.assertEqual(
            result.degradation_reason,
            "retrieval_agent_action_truncated",
        )
        self.assertEqual(
            [request.max_output_tokens for request in model.requests],
            [768, 1536],
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

    async def test_fusion_reserves_evidence_from_each_query_before_global_fill(
        self,
    ) -> None:
        context = _agent_context()
        shared = _pack(context, "shared first", "shared second", "shared third")
        distinct = _pack(context, "distinct aspect")

        fused = _fused_evidence(
            [shared.evidence, shared.evidence, distinct.evidence],
            top_k=3,
        )

        self.assertEqual(len(fused), 3)
        self.assertIn(
            evidence_key(distinct.evidence[0]),
            {evidence_key(item) for item in fused},
        )

    async def test_decimal_action_is_bounded_and_reaches_verifier_and_answer_state(
        self,
    ) -> None:
        context = _agent_context(decision_rounds=4)
        pack = _pack(
            context,
            "Revenue 229,104,123.45 and R&D costs 262,106,374.56.",
        )
        key = evidence_key(pack.evidence[0])
        model = _Model(
            _response(
                _search(
                    "find both amounts",
                    [{"query": "amounts", "based_on_observation_ids": []}],
                )
            ),
            _response(
                _calculate("229104123.45 + 262106374.56", (key,)),
                request_id="calculate",
            ),
            _response(_finish((key,)), request_id="finish"),
            _response(_verification(status="sufficient", keys=(key,)), request_id="verify"),
        )

        outcome = await _service(model, _Retriever((pack,))).research(
            context, _query_context(context)
        )

        self.assertEqual(len(outcome.calculation_facts), 1)
        self.assertEqual(outcome.calculation_facts[0].result, "491210498.01")
        assert outcome.workflow_state.search_trace is not None
        self.assertEqual(
            outcome.workflow_state.search_trace.calculation_call_count,
            1,
        )
        self.assertEqual(
            outcome.workflow_state.search_trace.calculation_success_count,
            1,
        )
        controller_payload = json.loads(model.requests[2].messages[1].content.split("\n", 1)[1])
        self.assertEqual(
            controller_payload["validated_calculations"][0]["result"],
            "491210498.01",
        )
        verifier_payload = json.loads(model.requests[-1].messages[1].content.split("\n", 1)[1])
        self.assertEqual(
            verifier_payload["validated_calculations"][0]["source_evidence_keys"],
            [key],
        )

    async def test_finish_verifies_neighbors_and_retains_only_selected_one(
        self,
    ) -> None:
        context = _agent_context()
        pack = _pack(context, "anchor one", "anchor two", "ordinary fallback")
        first_neighbor = _adjacent(
            pack.evidence[0],
            offset=1,
            text="definition continued across the boundary",
        )
        unselected_neighbor = _adjacent(
            pack.evidence[1],
            offset=-1,
            text="unselected neighboring context",
        )
        anchor_keys = tuple(evidence_key(item) for item in pack.evidence[:2])
        neighbor_key = evidence_key(first_neighbor)
        model = _Model(
            _response(
                _search(
                    "find boundary evidence",
                    [{"query": "boundary", "based_on_observation_ids": []}],
                )
            ),
            _response(_finish(anchor_keys), request_id="finish"),
            _response(
                _verification(status="sufficient", keys=(neighbor_key,)),
                request_id="verify",
            ),
        )
        retriever = _Retriever(
            (pack,),
            adjacency_results=((first_neighbor, unselected_neighbor),),
        )

        outcome = await _service(model, retriever).research(
            context, _query_context(context)
        )

        self.assertEqual(len(retriever.adjacency_calls), 1)
        self.assertEqual(
            tuple(item.index_chunk_id for item in retriever.adjacency_calls[0]),
            tuple(item.index_chunk_id for item in pack.evidence[:2]),
        )
        verification_payload = model.requests[-1].messages[-1].content
        self.assertIn('"score_kind":"adjacency"', verification_payload)
        self.assertIn('"ordinal":1', verification_payload)
        self.assertIn('"adjacency_offset":1', verification_payload)
        self.assertIn(str(pack.evidence[0].index_chunk_id), verification_payload)

        final_keys = tuple(
            evidence_key(item) for item in outcome.evidence_pack.evidence
        )
        self.assertIn(neighbor_key, final_keys)
        self.assertNotIn(evidence_key(unselected_neighbor), final_keys)
        self.assertLessEqual(len(final_keys), 3)
        self.assertEqual(
            tuple(item.rank for item in outcome.evidence_pack.evidence),
            tuple(range(1, len(final_keys) + 1)),
        )
        assert outcome.workflow_state.search_trace is not None
        self.assertEqual(
            outcome.workflow_state.search_trace.adjacency_loaded_count,
            2,
        )
        self.assertEqual(
            outcome.workflow_state.search_trace.adjacency_selected_count,
            1,
        )

        assessed = await AdaptiveEvidenceAssessmentStep(0.6, 0.45, 0.25).run(
            ChatPipelineState(
                context=context,
                query_context=_query_context(context),
                evidence_pack=outcome.evidence_pack,
                artifacts={WORKFLOW_STATE_ARTIFACT: outcome.workflow_state},
            )
        )
        assert assessed.answering is not None
        neighbor_rank = final_keys.index(neighbor_key) + 1
        self.assertEqual(
            assessed.answering.assessment.usable_citation_ids,
            (f"cite_{neighbor_rank}",),
        )
        neighbor_prompt = assessed.answering.evidence.items[neighbor_rank - 1]
        self.assertEqual(neighbor_prompt.index_chunk_id, first_neighbor.index_chunk_id)
        self.assertEqual(
            dict(neighbor_prompt.source_location),
            first_neighbor.source_location,
        )

        ineligible_pack = replace(
            outcome.evidence_pack,
            evidence=tuple(
                replace(item, score=0.1)
                if item.index_chunk_id == pack.evidence[0].index_chunk_id
                else item
                for item in outcome.evidence_pack.evidence
            ),
        )
        rejected = await AdaptiveEvidenceAssessmentStep(0.6, 0.45, 0.25).run(
            ChatPipelineState(
                context=context,
                query_context=_query_context(context),
                evidence_pack=ineligible_pack,
                artifacts={WORKFLOW_STATE_ARTIFACT: outcome.workflow_state},
            )
        )
        assert rejected.answering is not None
        self.assertIs(rejected.answering.assessment.coverage, EvidenceCoverage.NONE)
        self.assertEqual(rejected.answering.assessment.usable_citation_ids, ())

        invalid_relation_pack = replace(
            outcome.evidence_pack,
            evidence=tuple(
                replace(item, adjacency_offset=-1)
                if item.index_chunk_id == first_neighbor.index_chunk_id
                else item
                for item in outcome.evidence_pack.evidence
            ),
        )
        invalid_relation = await AdaptiveEvidenceAssessmentStep(
            0.6, 0.45, 0.25
        ).run(
            ChatPipelineState(
                context=context,
                query_context=_query_context(context),
                evidence_pack=invalid_relation_pack,
                artifacts={WORKFLOW_STATE_ARTIFACT: outcome.workflow_state},
            )
        )
        assert invalid_relation.answering is not None
        self.assertIs(
            invalid_relation.answering.assessment.coverage,
            EvidenceCoverage.NONE,
        )
        self.assertEqual(
            invalid_relation.answering.assessment.usable_citation_ids,
            (),
        )

    async def test_repeated_finish_reuses_cached_anchor_expansion(self) -> None:
        context = _agent_context()
        pack = _pack(context, "anchor")
        anchor_key = evidence_key(pack.evidence[0])
        neighbor = _adjacent(
            pack.evidence[0],
            offset=1,
            text="cached continuation",
        )
        neighbor_key = evidence_key(neighbor)
        model = _Model(
            _response(
                _search(
                    "find anchor",
                    [{"query": "anchor", "based_on_observation_ids": []}],
                )
            ),
            _response(_finish((anchor_key,)), request_id="finish-1"),
            _response(
                _verification(
                    status="partial",
                    keys=(neighbor_key,),
                    missing=("remaining aspect",),
                ),
                request_id="verify-1",
            ),
            _response(_finish((anchor_key,)), request_id="finish-2"),
            _response(
                _verification(status="sufficient", keys=(neighbor_key,)),
                request_id="verify-2",
            ),
        )
        retriever = _Retriever(
            (pack,),
            adjacency_results=((neighbor,),),
        )

        outcome = await _service(model, retriever).research(
            context, _query_context(context)
        )

        self.assertEqual(len(retriever.adjacency_calls), 1)
        assert outcome.workflow_state.search_trace is not None
        self.assertEqual(outcome.workflow_state.search_trace.verifier_calls, 2)
        self.assertEqual(
            outcome.workflow_state.search_trace.adjacency_loaded_count,
            1,
        )

    async def test_forced_finish_expands_before_its_single_verifier(self) -> None:
        context = _agent_context(decision_rounds=1)
        pack = _pack(context, "forced anchor")
        neighbor = _adjacent(
            pack.evidence[0],
            offset=1,
            text="forced-path continuation",
        )
        neighbor_key = evidence_key(neighbor)
        model = _Model(
            _response(
                _search(
                    "use the only round",
                    [{"query": "forced", "based_on_observation_ids": []}],
                )
            ),
            _response(
                _verification(status="sufficient", keys=(neighbor_key,)),
                request_id="forced-verify",
            ),
        )
        retriever = _Retriever(
            (pack,),
            adjacency_results=((neighbor,),),
        )

        outcome = await _service(model, retriever).research(
            context, _query_context(context)
        )

        self.assertEqual(len(retriever.adjacency_calls), 1)
        self.assertEqual(
            sum(
                request.output_schema.value
                == "research_result_verification_v1"
                for request in model.requests
            ),
            1,
        )
        self.assertIn(
            neighbor_key,
            {evidence_key(item) for item in outcome.evidence_pack.evidence},
        )

    async def test_neighbor_dedupe_prefers_the_higher_ranked_anchor(self) -> None:
        context = _agent_context()
        pack = _pack(context, "higher anchor", "lower anchor", "existing")
        higher_anchor = pack.evidence[0]
        lower_anchor = replace(
            pack.evidence[1],
            indexed_document_version_id=(
                higher_anchor.indexed_document_version_id
            ),
            document_id=higher_anchor.document_id,
            document_version_id=higher_anchor.document_version_id,
            ordinal=2,
        )
        shared_id = uuid4()
        higher = _adjacent(
            higher_anchor,
            offset=1,
            text="shared neighbor",
            chunk_id=shared_id,
        )
        lower = _adjacent(
            lower_anchor,
            offset=-1,
            text="shared neighbor",
            chunk_id=shared_id,
        )
        already_in_pool = _adjacent(
            lower_anchor,
            offset=1,
            text=pack.evidence[2].text,
            chunk_id=pack.evidence[2].index_chunk_id,
        )
        retriever = _Retriever(
            (),
            adjacency_results=((higher, lower, already_in_pool),),
        )

        expanded = await _expand_verification_evidence(
            retriever,
            context,
            selected_evidence=(higher_anchor, lower_anchor),
            evidence_pool={
                evidence_key(item): item
                for item in (higher_anchor, lower_anchor, pack.evidence[2])
            },
            adjacency_cache={},
        )

        neighbors = tuple(
            item
            for item in expanded
            if item.score_kind is EvidenceScoreKind.ADJACENCY
        )
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0].index_chunk_id, shared_id)
        self.assertEqual(
            neighbors[0].adjacency_anchor_index_chunk_id,
            higher_anchor.index_chunk_id,
        )

    async def test_finish_can_select_visible_evidence_beyond_global_top_k(
        self,
    ) -> None:
        context = _agent_context()
        pack = _pack(
            context,
            "first",
            "second",
            "third",
            "fourth",
            "fifth requested aspect",
        )
        selected_key = evidence_key(pack.evidence[4])
        model = _Model(
            _response(
                _search(
                    "find all aspects",
                    [{"query": "all aspects", "based_on_observation_ids": []}],
                )
            ),
            _response(_finish((selected_key,)), request_id="finish"),
            _response(
                _verification(status="sufficient", keys=(selected_key,)),
                request_id="verify",
            ),
        )

        outcome = await _service(model, _Retriever((pack,))).research(
            context, _query_context(context)
        )

        final_keys = tuple(
            evidence_key(item) for item in outcome.evidence_pack.evidence
        )
        self.assertEqual(len(final_keys), 3)
        self.assertIn(selected_key, final_keys)
        assert outcome.workflow_state.research_result is not None
        self.assertEqual(
            outcome.workflow_state.research_result.selected_evidence_keys,
            (selected_key,),
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
        context = _agent_context(verifier_continuations=0)
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
            _response(_finish((), reason="no_evidence"), request_id="finish"),
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
        self.assertIn(
            "search_rejected_use_distinct_unexecuted_queries",
            model.requests[1].messages[1].content,
        )

    async def test_duplicate_query_is_reprompted_without_false_no_progress(
        self,
    ) -> None:
        context = _agent_context(verifier_continuations=0)
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
            _response(_finish((key,), reason="partial"), request_id="finish"),
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
            "partial",
        )
        self.assertIn(
            "search_rejected_use_distinct_unexecuted_queries",
            model.requests[2].messages[1].content,
        )

    async def test_actual_zero_new_evidence_stops_with_no_progress(self) -> None:
        context = _agent_context(verifier_continuations=0)
        pack = _pack(context, "owner evidence")
        key = evidence_key(pack.evidence[0])
        model = _Model(
            _response(
                _search(
                    "find owner",
                    [{"query": "owner", "based_on_observation_ids": []}],
                )
            ),
            _response(
                _search(
                    "find deadline",
                    [{"query": "deadline", "based_on_observation_ids": []}],
                ),
                request_id="second-search",
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

        outcome = await _service(model, _Retriever((pack, pack))).research(
            context, _query_context(context)
        )

        assert outcome.workflow_state.research_result is not None
        assert outcome.workflow_state.search_trace is not None
        self.assertEqual(
            outcome.workflow_state.research_result.termination_reason.value,
            "no_progress",
        )
        self.assertEqual(
            outcome.workflow_state.search_trace.steps[-1].result,
            "no_evidence",
        )

    async def test_model_cannot_declare_no_progress_after_finding_evidence(
        self,
    ) -> None:
        context = _agent_context(verifier_continuations=0)
        pack = _pack(context, "owner evidence")
        key = evidence_key(pack.evidence[0])
        model = _Model(
            _response(
                _search(
                    "find owner",
                    [{"query": "owner", "based_on_observation_ids": []}],
                )
            ),
            _response(_finish((key,), reason="no_progress"), request_id="invalid"),
            _response(_finish((key,), reason="partial"), request_id="finish"),
            _response(
                _verification(
                    status="partial",
                    keys=(key,),
                    missing=("deadline",),
                ),
                request_id="verify",
            ),
        )

        outcome = await _service(model, _Retriever((pack,))).research(
            context, _query_context(context)
        )

        assert outcome.workflow_state.research_result is not None
        self.assertEqual(
            outcome.workflow_state.research_result.termination_reason.value,
            "partial",
        )
        self.assertIn(
            "finish_rejected_follow_reason_and_selection_constraints",
            model.requests[2].messages[1].content,
        )

    async def test_decision_budget_and_verifier_statuses_have_stable_endings(self) -> None:
        context = _agent_context()
        packs = tuple(_pack(context, f"evidence {index}") for index in range(4))
        # The frozen top_k is three, so select a key retained by the same
        # per-query quota fusion used by the budget-exhausted verifier path.
        budget_key = evidence_key(
            _fused_evidence(
                [pack.evidence for pack in packs],
                top_k=3,
            )[0]
        )
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
