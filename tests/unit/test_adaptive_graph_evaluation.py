from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
import json

from apps.api.routers.chat import _public_agent_trace

from rag_kb.domain import (
    ChatModelMessage,
    ChatModelRequest,
    ChatModelResponse,
    ChatToolCall,
    ChatToolDefinition,
    GraphitiSupplementResult,
)
from tools.evaluate_adaptive_graph_route import (
    CapturingGraphitiSupplementRetriever,
    ForcedGraphitiSupplementChatModelPort,
    GraphitiSupplementCapture,
    GraphitiSupplementCaptureComplete,
    aggregate_graph_routing_metrics,
    aggregate_usage,
    align_chunk_layers,
    build_replay_capture_artifact,
    diagnostic_record,
    build_routing_judge_packet,
    EVALUATOR_EDGE_LIMITS,
    load_cases,
    load_manifest,
    normalize_term,
    routing_judge_cache_key,
    summarize_graph_route_trace,
    term_proxy,
    validate_evaluator_edge_limit,
    validate_evaluation_readiness,
    validate_routing_judgement,
    write_replay_capture_artifact,
)
from tools.run_adaptive_graph_r7_stage_a import (
    ANSWER_EXECUTION_LIMIT,
    JUDGE_CALL_LIMIT,
    _answer_totals,
    _assert_runtime_identity,
    _judge_cost,
    _judge_tool,
    _runtime,
    _quality_tuple,
)


class _RecordingModel:
    def __init__(self) -> None:
        self.requests: list[ChatModelRequest] = []

    async def complete(self, request: ChatModelRequest) -> ChatModelResponse:
        self.requests.append(request)
        return ChatModelResponse(
            content="ok",
            model="fake-model",
            finish_reason="stop",
            provider_request_id=None,
            usage={"total_tokens": 1},
        )


class _QueuedModel:
    def __init__(self, responses: list[ChatModelResponse]) -> None:
        self.requests: list[ChatModelRequest] = []
        self.responses = list(responses)

    async def complete(self, request: ChatModelRequest) -> ChatModelResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("queued model response was exhausted")
        return self.responses.pop(0)


class _RecordingRetriever:
    def __init__(self) -> None:
        self.simple_calls = 0
        self.supplement_calls = 0

    async def retrieve_query(self, *args, **kwargs):
        self.simple_calls += 1
        return (args, kwargs)

    async def retrieve_graphiti_supplement(self, *args, **kwargs):
        self.supplement_calls += 1
        return GraphitiSupplementResult("no_new_evidence")


def _tools() -> tuple[ChatToolDefinition, ...]:
    return (
        ChatToolDefinition("search_knowledge_base", "simple", {"type": "object"}),
        ChatToolDefinition("graphiti_supplement", "graph", {"type": "object"}),
    )


class AdaptiveGraphEvaluationTests(unittest.IsolatedAsyncioTestCase):
    def test_r7_stage_a_budget_and_paired_order_are_frozen(self) -> None:
        self.assertEqual(ANSWER_EXECUTION_LIMIT, 78)
        self.assertEqual(JUDGE_CALL_LIMIT, 78)
        self.assertGreater(
            _quality_tuple(
                {
                    "outcome_correctness": "correct",
                    "answer_correctness": "correct",
                    "claim_grounding": "supported",
                    "citation_alignment": "supported",
                }
            ),
            _quality_tuple(
                {
                    "outcome_correctness": "correct",
                    "answer_correctness": "partial",
                    "claim_grounding": "supported",
                    "citation_alignment": "supported",
                }
            ),
        )

    def test_r7_resume_identity_covers_manifest_fixture_case_order_and_runtime(self) -> None:
        manifest = load_manifest()
        runtime = _runtime(manifest)
        _assert_runtime_identity({"runtime": runtime}, runtime, artifact="answers")
        changed = dict(runtime)
        changed["case_order"] = list(runtime["case_order"])[::-1]
        with self.assertRaisesRegex(RuntimeError, "r7_answers_identity_mismatch"):
            _assert_runtime_identity(
                {"runtime": changed},
                runtime,
                artifact="answers",
            )
        legacy = dict(manifest)
        legacy["dataset_id"] = "routing-rag-v1"
        with self.assertRaisesRegex(RuntimeError, "r7_dataset_identity_changed"):
            _runtime(legacy)

    def test_r7_stage_a_usage_and_judge_schema_are_closed(self) -> None:
        state = {
            "executions": {
                "case:simple": {
                    "status": "completed",
                    "lane": "simple",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                    "elapsed_seconds": 1.5,
                    "estimated_cost_usd": 0.1,
                },
                "case:auto": {
                    "status": "completed",
                    "lane": "auto",
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 3,
                        "total_tokens": 23,
                    },
                    "elapsed_seconds": 2.5,
                    "estimated_cost_usd": 0.2,
                },
            }
        }
        totals = _answer_totals(state)
        self.assertEqual(totals["completed"], 2)
        self.assertEqual(totals["total_tokens"], 35)
        self.assertEqual(totals["lane_tokens"], {"simple": 12, "auto": 23})
        schema = _judge_tool().input_schema
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertEqual(
            _judge_cost(
                {
                    "prompt_tokens": 1_000_000,
                    "completion_tokens": 1_000_000,
                    "total_tokens": 2_000_000,
                }
            ),
            1.76,
        )

    def test_public_trace_keeps_supplement_tool_name(self) -> None:
        trace = _public_agent_trace(
            {
                "events": [
                    {
                        "tool": "graphiti_supplement",
                        "retrieval_lane": "graphiti_supplement",
                        "route_result_code": "admitted",
                        "rejected_claim_count": 1,
                        "rejection_reasons": ["evidence_ref"],
                        "submit_only_repair": True,
                    }
                ]
            }
        )
        self.assertEqual(trace["events"][0]["tool"], "graphiti_supplement")
        self.assertEqual(
            trace["events"][0]["retrieval_lane"],
            "graphiti_supplement",
        )
        self.assertNotIn("rejected_claim_count", trace["events"][0])
        self.assertNotIn("rejection_reasons", trace["events"][0])
        self.assertNotIn("submit_only_repair", trace["events"][0])

    def test_public_trace_drops_retired_route_reason(self) -> None:
        trace = _public_agent_trace(
            {
                "events": [
                    {
                        "tool": "search_knowledge_base",
                        "retrieval_lane": "graphiti_supplement",
                        "route_reason_code": "relational_query_without_simple_evidence",
                        "route_result_code": "rejected",
                        "new_evidence_count": 0,
                    }
                ]
            }
        )
        self.assertNotIn("route_reason_code", trace["events"][0])
        self.assertEqual(trace["events"][0]["route_result_code"], "rejected")

    def test_latest_routing_manifest_has_frozen_contract(self) -> None:
        manifest = load_manifest()
        self.assertEqual(manifest["dataset_id"], "routing-rag-v2")
        self.assertEqual(manifest["case_count"], 39)
        self.assertEqual(
            [item["id"] for item in manifest["routes"]],
            ["vector-only", "hybrid-control", "manual-graph", "auto-route"],
        )

    def test_evaluation_readiness_is_offline_and_tracks_source_bytes(self) -> None:
        readiness = validate_evaluation_readiness()
        self.assertEqual(
            {
                readiness["case_count"],
                readiness["fixture_count"],
                readiness["graph_locator_count"],
                readiness["graph_unique_relation_count"],
            },
            {39, 22, 40, 35},
        )
        for key in (
            "manifest_file_sha256",
            "cases_file_sha256",
            "fixture_file_sha256",
            "corpus_manifest_sha256",
            "graph_relations_sha256",
        ):
            self.assertRegex(readiness[key], r"^[0-9a-f]{64}$")

    def test_graph_route_metrics_prioritize_recall_accuracy_and_safe_recall_status(self) -> None:
        cases = (
            {"case_id": "graph-001", "expected_route": {"route": "graph"}},
            {"case_id": "simple-001", "expected_route": {"route": "simple"}},
        )
        observations = {
            "graph-001": {
                "graph_route_attempted": False,
                "graph_route_admitted": False,
                "graph_new_evidence_count": 0,
            },
            "simple-001": {
                "graph_route_attempted": True,
                "graph_route_admitted": True,
                "graph_new_evidence_count": 1,
            },
        }
        without_layers = aggregate_graph_routing_metrics(cases, observations)
        self.assertEqual(
            without_layers["route"]["graph_needed_route_recall"]["value"],
            0.0,
        )
        self.assertEqual(
            without_layers["route"]["graph_route_accuracy"]["value"],
            0.0,
        )
        self.assertEqual(
            without_layers["route"]["simple_false_positive_rate"]["value"],
            1.0,
        )
        self.assertEqual(
            without_layers["graph_recall"]["status"],
            "not_computed_missing_layer_evidence",
        )
        self.assertNotIn("tokens", without_layers)
        self.assertNotIn("latency", without_layers)

        with_layers = aggregate_graph_routing_metrics(
            cases,
            {
                "graph-001": {
                    "graph_route_attempted": True,
                    "graph_route_admitted": True,
                    "graph_new_evidence_count": 1,
                },
                "simple-001": {
                    "graph_route_attempted": False,
                    "graph_route_admitted": False,
                    "graph_new_evidence_count": 0,
                },
            },
            alignments={
                "graph-001": {
                    "answer_gold_chunk_ids": ["gold"],
                    "simple_chunk_ids": [],
                    "layers": {
                        "raw": ["gold"],
                        "hydrated": ["gold"],
                        "reranked": ["gold"],
                        "packed": ["gold"],
                    },
                    "redundant_hit": False,
                }
            },
        )
        self.assertEqual(
            with_layers["route"]["graph_needed_route_recall"]["value"],
            1.0,
        )
        self.assertEqual(
            with_layers["graph_recall"]["by_layer"]["packed"]["gold_recall"]["value"],
            1.0,
        )
        self.assertEqual(
            with_layers["benefit_capture"]["benefit_capture"]["value"],
            1.0,
        )

    def test_graph_route_trace_reduces_to_closed_safe_observation(self) -> None:
        result = summarize_graph_route_trace(
            {
                "events": [
                    {
                        "retrieval_lane": "simple",
                        "route_result_code": "not_requested",
                    },
                    {
                        "retrieval_lane": "graphiti_supplement",
                        "route_result_code": "admitted",
                        "new_evidence_count": 2,
                    },
                ],
                "answer": "不要进入聚合结果",
            }
        )
        self.assertEqual(
            result,
            {
                "graph_route_attempted": True,
                "graph_route_admitted": True,
                "graph_new_evidence_count": 2,
                "graph_route_result_counts": {"admitted": 1},
            },
        )
        self.assertNotIn("answer", result)

    def test_term_proxy_normalizes_spaces_unicode_and_brackets(self) -> None:
        self.assertEqual(normalize_term("5 个（工作日）"), "5个(工作日)")
        self.assertEqual(
            term_proxy("期限为 5个工作日。", ["5 个工作日"]),
            {
                "matched": 1,
                "total": 1,
                "all_matched": True,
                "empty_expected_terms": False,
            },
        )

    def test_chunk_benefit_is_answer_gold_only_and_tracks_first_loss(self) -> None:
        capability = align_chunk_layers(
            column="capability",
            answer_gold_chunk_ids=("answer",),
            simple_chunk_ids=("simple",),
            layer_chunk_ids={
                "raw": ("answer", "bridge"),
                "hydrated": ("answer",),
                "reranked": ("bridge",),
                "packed": ("bridge",),
            },
            path_context_chunk_ids=("bridge",),
        )
        replay = align_chunk_layers(
            column="agent_replay",
            answer_gold_chunk_ids=("answer",),
            simple_chunk_ids=("simple",),
            layer_chunk_ids={
                "raw": ("answer",),
                "hydrated": ("answer",),
                "reranked": ("answer",),
                "packed": ("answer",),
            },
        )
        self.assertEqual(capability.first_loss_layer, "reranked")
        self.assertTrue(capability.redundant_hit)
        self.assertFalse(capability.benefit)
        self.assertIsNone(replay.first_loss_layer)
        self.assertTrue(replay.benefit)
        record = diagnostic_record(
            case_id="graph-001",
            alignments={"capability": capability, "agent_replay": replay},
            query_source="agent_replay",
            query_count=1,
        )
        self.assertEqual(set(record["columns"]), {"capability", "agent_replay"})
        self.assertNotIn("query", record)
        self.assertEqual(
            record["columns"]["capability"]["gold_hit_by_layer"],
            {
                "raw": ["answer"],
                "hydrated": ["answer"],
                "reranked": [],
                "packed": [],
            },
        )

    def test_evaluator_k_and_layer_diagnostics_are_closed_and_content_safe(self) -> None:
        self.assertEqual(EVALUATOR_EDGE_LIMITS, (8, 16, 32, 64))
        for value in EVALUATOR_EDGE_LIMITS:
            self.assertEqual(validate_evaluator_edge_limit(value), value)
        with self.assertRaises(ValueError):
            validate_evaluator_edge_limit(10)

        alignment = align_chunk_layers(
            column="capability",
            answer_gold_chunk_ids=(str(uuid4()),),
            simple_chunk_ids=(),
            layer_chunk_ids={layer: () for layer in ("raw", "hydrated", "reranked", "packed")},
        )
        record = diagnostic_record(
            case_id="synthetic-case-001",
            alignments={"capability": alignment, "agent_replay": alignment},
            query_source="capability",
            query_count=1,
            layer_diagnostics={
                "capability": {
                    "requested_k": 16,
                    "raw_edge_uuids": [str(uuid4())],
                    "raw_episode_count": 2,
                    "unique_chunk_count": 1,
                    "duplicate_chunk_path_count": 1,
                    "gold_rerank_scores": {str(uuid4()): 0.42},
                    "top1_gold_rerank_score": 0.42,
                    "rerank_score_state": "scored",
                    "rerank_reordered_chunk_count": 1,
                    "route_reason_code": None,
                    "route_result_code": "not_requested",
                    "salvage_status": "not_attempted",
                    "final_outcome": "not_run",
                },
                "agent_replay": {
                    "requested_k": 16,
                    "raw_edge_uuids": [],
                    "raw_episode_count": 0,
                    "unique_chunk_count": 0,
                    "duplicate_chunk_path_count": 0,
                    "gold_rerank_scores": {},
                    "top1_gold_rerank_score": None,
                    "rerank_score_state": "not_applicable",
                    "rerank_reordered_chunk_count": 0,
                    "route_reason_code": "relation_chain_gap",
                    "route_result_code": "admitted",
                    "salvage_status": "not_attempted",
                    "final_outcome": "not_run",
                },
            },
        )
        serialized = json.dumps(record, ensure_ascii=False)
        self.assertNotIn("query", record)
        self.assertNotIn("Chunk正文", serialized)
        self.assertNotIn("filename.pdf", serialized)
        self.assertEqual(record["layer_diagnostics"]["capability"]["requested_k"], 16)
        self.assertEqual(
            record["layer_diagnostics"]["capability"]["rerank_score_state"],
            "scored",
        )
        with self.assertRaises(ValueError):
            diagnostic_record(
                case_id="synthetic-case-001",
                alignments={"capability": alignment, "agent_replay": alignment},
                query_source="capability",
                query_count=1,
                layer_diagnostics={"capability": {"below_threshold_count": 1}},
            )

    def test_usage_aggregation_does_not_invent_cost(self) -> None:
        result = aggregate_usage(
            [
                {
                    "total_tokens": 10,
                    "model_rounds": 2,
                    "retrieval_queries": 1,
                    "evidence_refs": 3,
                    "rejected_tools": 0,
                    "elapsed_seconds": 1.0,
                },
                {
                    "total_tokens": 20,
                    "model_rounds": 3,
                    "retrieval_queries": 2,
                    "evidence_refs": 4,
                    "rejected_tools": 1,
                    "elapsed_seconds": 2.0,
                },
            ]
        )
        self.assertEqual(result["totals"]["total_tokens"], 30)
        self.assertEqual(result["elapsed_seconds"]["p95"], 2.0)
        self.assertEqual(result["cost"]["status"], "not_computed")

    def test_answer_gold_already_in_simple_is_redundant(self) -> None:
        alignment = align_chunk_layers(
            column="agent_replay",
            answer_gold_chunk_ids=("answer",),
            simple_chunk_ids=("answer", "simple-context"),
            layer_chunk_ids={
                "raw": (),
                "hydrated": (),
                "reranked": (),
                "packed": (),
            },
        )

        self.assertTrue(alignment.redundant_hit)
        self.assertFalse(alignment.benefit)
        self.assertIsNone(alignment.first_loss_layer)

    def test_routing_judge_packet_and_closed_schema_are_cacheable(self) -> None:
        manifest = load_manifest()
        case = next(
            item for item in load_cases(Path(manifest["case_file"]))
            if item["case_id"] == "abstain-003"
        )
        packet = build_routing_judge_packet(
            case,
            {
                "outcome": "answered",
                "answer": "否，Retail 渠道占比为 40%",
                "citations": [
                    {
                        "citation_id": "cite_1",
                        "index_chunk_id": "chunk_1",
                        "document_id": "single-08",
                        "source_location": {"chart_id": "channel_mix"},
                        "modality": "image",
                    }
                ],
            },
        )
        self.assertEqual(packet["schema_version"], "routing_rag_v1_judge_v1")
        self.assertEqual(routing_judge_cache_key(packet), routing_judge_cache_key(packet))
        validate_routing_judgement(
            {
                "schema_version": "routing_rag_v1_judge_v1",
                "prompt_version": "routing_rag_v1_judge_prompt_v1",
                "answer_correctness": "correct",
                "claim_grounding": "supported",
                "citation_alignment": "supported",
                "outcome_correctness": "correct",
                "negative_stance": "denied",
                "reason_code": "answer_supported",
            }
        )

    async def test_forced_controller_overrides_only_next_call_after_simple(self) -> None:
        delegate = _RecordingModel()
        controller = ForcedGraphitiSupplementChatModelPort(
            delegate,
            controller_mode="specific_tool_choice",
        )
        tools = _tools()
        first = ChatModelRequest(
            messages=(ChatModelMessage("user", "原始问题"),),
            tools=tools,
            tool_choice="required",
        )
        await controller.complete(first)
        simple_result = ChatModelMessage(
            "tool", '{"status":"ok","groups":[]}', tool_call_id="simple-1"
        )
        second = ChatModelRequest(
            messages=(first.messages[0], simple_result),
            tools=tools,
            tool_choice="required",
        )
        await controller.complete(second)
        third = ChatModelRequest(
            messages=second.messages,
            tools=tools,
            tool_choice="required",
        )
        await controller.complete(third)
        self.assertEqual(controller.replacements, 1)
        self.assertEqual(controller.model_calls, 3)
        self.assertEqual(controller.usage, {"total_tokens": 3})
        self.assertEqual(controller.response_tool_names, [(), (), ()])
        self.assertEqual(delegate.requests[0].tool_choice, "required")
        self.assertEqual(delegate.requests[1].tool_choice, "graphiti_supplement")
        self.assertEqual(delegate.requests[2].tool_choice, "required")
        self.assertEqual(delegate.requests[1].messages, second.messages)

    async def test_provider_safe_controller_uses_single_tool_auto_and_observes_call(self) -> None:
        simple_call = ChatModelResponse(
            content="",
            model="fake-model",
            finish_reason="tool_calls",
            provider_request_id=None,
            usage={"total_tokens": 1},
            tool_calls=(
                ChatToolCall(
                    "simple-1",
                    "search_knowledge_base",
                    {"queries": ["simple"]},
                ),
            ),
        )
        supplement_call = ChatModelResponse(
            content="",
            model="fake-model",
            finish_reason="tool_calls",
            provider_request_id=None,
            usage={"total_tokens": 2},
            tool_calls=(
                ChatToolCall(
                    "graph-1",
                    "graphiti_supplement",
                    {
                        "query": "relation gap",
                        "route_reason_code": "cross_document_relation_gap",
                    },
                ),
            ),
        )
        delegate = _QueuedModel([simple_call, supplement_call])
        controller = ForcedGraphitiSupplementChatModelPort(delegate)
        tools = _tools()
        initial_tools = tuple(
            item for item in tools if item.name != "graphiti_supplement"
        )
        first = ChatModelRequest(
            messages=(ChatModelMessage("user", "原始问题"),),
            tools=initial_tools,
            tool_choice="required",
        )
        await controller.complete(first)
        second = ChatModelRequest(
            messages=(
                first.messages[0],
                ChatModelMessage(
                    "tool",
                    '{"status":"ok","groups":[]}',
                    tool_call_id="simple-1",
                ),
            ),
            tools=tools,
            tool_choice="required",
        )
        await controller.complete(second)

        self.assertEqual(controller.replacements, 1)
        self.assertEqual(controller.model_calls, 2)
        self.assertEqual(
            tuple(item.name for item in delegate.requests[0].tools),
            ("search_knowledge_base",),
        )
        self.assertEqual(delegate.requests[0].tool_choice, "auto")
        self.assertEqual(
            tuple(item.name for item in delegate.requests[1].tools),
            ("graphiti_supplement",),
        )
        self.assertEqual(delegate.requests[1].tool_choice, "auto")

    async def test_provider_safe_controller_retries_without_counting_unobserved_call(self) -> None:
        invalid = ChatModelResponse(
            content="不能确定",
            model="fake-model",
            finish_reason="stop",
            provider_request_id=None,
            usage={"total_tokens": 1},
        )
        valid = ChatModelResponse(
            content="",
            model="fake-model",
            finish_reason="tool_calls",
            provider_request_id=None,
            usage={"total_tokens": 1},
            tool_calls=(
                ChatToolCall(
                    "graph-1",
                    "graphiti_supplement",
                    {
                        "query": "relation gap",
                        "route_reason_code": "relation_chain_gap",
                    },
                ),
            ),
        )
        delegate = _QueuedModel([invalid, valid])
        controller = ForcedGraphitiSupplementChatModelPort(delegate)
        request = ChatModelRequest(
            messages=(
                ChatModelMessage("user", "原始问题"),
                ChatModelMessage(
                    "tool",
                    '{"status":"ok","groups":[]}',
                    tool_call_id="simple-1",
                ),
            ),
            tools=_tools(),
            tool_choice="required",
        )
        response = await controller.complete(request)

        self.assertEqual(response.tool_calls[0].name, "graphiti_supplement")
        self.assertEqual(controller.replacements, 1)
        self.assertEqual(controller.model_calls, 2)
        self.assertEqual(delegate.requests[1].tool_choice, "auto")
        self.assertEqual(delegate.requests[1].tools[0].name, "graphiti_supplement")
        self.assertEqual(delegate.requests[1].messages[-1].role, "user")
        self.assertNotEqual(delegate.requests[1].messages, request.messages)

    async def test_forced_controller_has_explicit_single_tool_provider_fallback(self) -> None:
        delegate = _RecordingModel()
        controller = ForcedGraphitiSupplementChatModelPort(
            delegate,
            controller_mode="single_tool_required_fallback",
        )
        original = ChatModelRequest(
            messages=(
                ChatModelMessage("user", "原始问题"),
                ChatModelMessage(
                    "tool",
                    '{"status":"ok","groups":[]}',
                    tool_call_id="simple-1",
                ),
            ),
            tools=_tools(),
            tool_choice="required",
        )
        await controller.complete(original)
        effective = delegate.requests[0]
        self.assertEqual(effective.messages, original.messages)
        self.assertEqual(effective.tool_choice, "required")
        self.assertEqual(
            tuple(item.name for item in effective.tools),
            ("graphiti_supplement",),
        )
        self.assertEqual(tuple(item.name for item in original.tools), (
            "search_knowledge_base",
            "graphiti_supplement",
        ))

    async def test_single_tool_fallback_restricts_initial_required_choice(self) -> None:
        delegate = _RecordingModel()
        controller = ForcedGraphitiSupplementChatModelPort(
            delegate,
            controller_mode="single_tool_required_fallback",
        )
        tools = tuple(item for item in _tools() if item.name != "graphiti_supplement")
        request = ChatModelRequest(
            messages=(ChatModelMessage("user", "原始问题"),),
            tools=tools,
            tool_choice="required",
        )
        await controller.complete(request)
        self.assertEqual(delegate.requests[0].tool_choice, "required")
        self.assertEqual(
            tuple(item.name for item in delegate.requests[0].tools),
            ("search_knowledge_base",),
        )

    async def test_actual_auto_observer_does_not_change_model_request(self) -> None:
        delegate = _RecordingModel()
        controller = ForcedGraphitiSupplementChatModelPort(
            delegate,
            controller_mode="actual_auto",
        )
        original = ChatModelRequest(
            messages=(
                ChatModelMessage("user", "原始问题"),
                ChatModelMessage(
                    "tool",
                    '{"status":"ok","groups":[]}',
                    tool_call_id="simple-1",
                ),
            ),
            tools=_tools(),
            tool_choice="required",
        )
        await controller.complete(original)
        self.assertEqual(delegate.requests, [original])
        self.assertEqual(controller.replacements, 0)

    def test_actual_auto_capture_artifact_allows_no_requested_supplement(self) -> None:
        runtime_ids = [str(uuid4()) for _ in range(4)]
        artifact = build_replay_capture_artifact(
            dataset_id="routing-rag-v2",
            rerank_mode="classic",
            manifest_sha256="b" * 64,
            knowledge_base_id=runtime_ids[0],
            index_revision_id=runtime_ids[1],
            graph_build_id=runtime_ids[2],
            chat_model_profile_revision_id=runtime_ids[3],
            captures=(),
            controller_mode="actual_auto",
        )
        self.assertEqual(artifact["capture_source"], "r3a_actual_auto")
        self.assertEqual(artifact["case_count"], 0)

    async def test_capture_wraps_post_validation_retrieval_boundary_once(self) -> None:
        delegate = _RecordingRetriever()
        capture = CapturingGraphitiSupplementRetriever(
            delegate,
            stop_after_capture=True,
        )
        excluded = (uuid4(), uuid4())
        capture.begin_case("route-graph-001")
        await capture.retrieve_query("context", "simple")
        with self.assertRaises(GraphitiSupplementCaptureComplete):
            await capture.retrieve_graphiti_supplement(
                "context",
                "  星澜工厂 控股方 法定代表人  ",
                excluded_index_chunk_ids=excluded,
            )
        result = capture.finish_case()
        self.assertEqual(result.case_id, "route-graph-001")
        self.assertEqual(result.query, "星澜工厂 控股方 法定代表人")
        self.assertEqual(result.excluded_index_chunk_ids, tuple(map(str, excluded)))
        self.assertEqual(delegate.simple_calls, 1)
        self.assertEqual(delegate.supplement_calls, 0)
        with self.assertRaises(RuntimeError):
            capture.finish_case()

    async def test_capture_rejects_duplicate_supplement_call(self) -> None:
        delegate = _RecordingRetriever()
        capture = CapturingGraphitiSupplementRetriever(delegate)
        capture.begin_case("route-graph-001")
        await capture.retrieve_graphiti_supplement(
            "context",
            "first relation query",
            excluded_index_chunk_ids=(),
        )
        with self.assertRaises(RuntimeError):
            await capture.retrieve_graphiti_supplement(
                "context",
                "second relation query",
                excluded_index_chunk_ids=(),
            )
        result = capture.finish_case()
        self.assertEqual(result.query, "first relation query")
        self.assertEqual(delegate.supplement_calls, 1)

    def test_replay_capture_artifact_is_versioned_immutable_and_secret_free(self) -> None:
        runtime_ids = [str(uuid4()) for _ in range(4)]
        capture = GraphitiSupplementCapture(
            case_id="route-graph-001",
            query="relation query",
            excluded_index_chunk_ids=(str(uuid4()),),
        )
        artifact = build_replay_capture_artifact(
            dataset_id="routing-rag-v2",
            rerank_mode="classic",
            manifest_sha256="a" * 64,
            knowledge_base_id=runtime_ids[0],
            index_revision_id=runtime_ids[1],
            graph_build_id=runtime_ids[2],
            chat_model_profile_revision_id=runtime_ids[3],
            captures=(capture,),
            chat_model="deepseek-v4-flash",
            chat_model_source="evaluator_override",
            chat_model_max_output_tokens=512,
            chat_model_max_retries=0,
        )
        self.assertEqual(artifact["schema_version"], "adaptive_graph_replay_capture_v2")
        self.assertEqual(artifact["controller_mode"], "single_tool_auto_fallback")
        self.assertEqual(artifact["case_count"], 1)
        self.assertNotIn("query", artifact["cases"][0])
        self.assertNotIn("query_sha256", artifact["cases"][0])
        self.assertNotIn("provider", artifact)
        self.assertNotIn("api_key", str(artifact))
        self.assertEqual(artifact["runtime"]["chat_model"], "deepseek-v4-flash")
        self.assertEqual(
            artifact["runtime"]["chat_model_source"],
            "evaluator_override",
        )
        self.assertEqual(
            artifact["runtime"]["chat_model_max_output_tokens"],
            512,
        )
        self.assertEqual(artifact["runtime"]["chat_model_max_retries"], 0)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "capture.json"
            digest = write_replay_capture_artifact(path, artifact)
            self.assertEqual(len(digest), 64)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                write_replay_capture_artifact(path, artifact)
