#!/usr/bin/env python3
"""Run the routing-rag-v4 locked evaluation for the first-class Graph tool.

The v4 evaluation executes the actual native Agent in-process against the
frozen public-fact corpus runtime: a Simple lane and an auto lane (first-class
search_graph_relations visible from round one) repeated three times per graph
case, plus an offline Graph candidate K (8/16/32) and whole-path packing target
(4/8/12/16) sweep on the same graph build.  Observations keep the stable
corpus-derived route labels (graph_needed / simple_only / negative_or_refusal),
hop stratification (1/2/3 gold paths), the first versus second Graph call
increments, and route recall/precision stability across repeats.  Historical
v3 locked artifacts are never rewritten; this tool owns its observation and
locked schema identities.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4

from apps.worker.dependencies import build_worker_dependencies
from rag_kb.answering.agent import AGENT_TRACE_ARTIFACT, NativeToolCallingAgent
from rag_kb.domain import RerankMode
from rag_kb.retrieval.profile import adaptive_graphiti_profile, exact_profile
from rag_kb.retrieval.service import _pack_graph_search_evidence  # noqa: SLF001
from rag_kb.services.chat_execution import ChatEvidenceRetriever
from tools.evaluate_adaptive_graph_route import (
    ForcedGraphSearchChatModelPort,
    summarize_graph_route_trace,
)
from tools.evaluate_open_source_rag_v3 import (
    CASES_PATH,
    MANIFEST_PATH,
    RELATIONS_PATH,
    corpus_identity,
)
from tools.evaluation_runtime import (
    DEFAULT_RUNTIME_MANIFEST,
    EvaluationRuntimeError,
    load_evaluation_runtime,
)
from tools.run_adaptive_graph_r4 import (
    R4RunnerError,
    _execution_context,
    _load_runtime_facts,
    _serving_chunk_rows,
)
from tools.run_open_source_rag_v3 import (
    _forbidden_claim_hit,
    _relation_ids_for_chunks,
    _runtime_identity,
    relation_chunk_map,
)


CONFIRM = "RUN_ROUTING_RAG_V4_EXTERNAL_CALLS"
OBSERVATION_SCHEMA = "routing_rag_v4_observations_v1"
LOCKED_SCHEMA = "routing_rag_v4_locked_evaluation_v1"
SIMPLE_TOP_K = 10
GRAPH_EDGE_LIMIT = 16
SOURCE_CHUNK_TARGET = 12
SOURCE_CHUNK_LIMIT = 16
GRAPH_CALL_TIMEOUT_SECONDS = 90
SWEEP_EDGE_LIMITS = (8, 16, 32)
SWEEP_PACK_TARGETS = (4, 8, 12, 16)
ACTUAL_AUTO_REPEATS = 3
TOTAL_TOKEN_LIMIT = 1_600_000
TOTAL_COST_LIMIT_USD = 1.0
MIMO_INPUT_USD_PER_MILLION = 0.14
MIMO_OUTPUT_USD_PER_MILLION = 0.28
ANSWER_EXECUTION_LIMIT = 100
GRAPH_CALL_LIMIT = 2
CHATRUN_DEADLINE_SECONDS = 420.0
EXPECTED_KB_NAME = "open-source-rag-v3"
ROUTE_RECALL_THRESHOLD = 0.80
ROUTE_PRECISION_THRESHOLD = 0.70


class V4RunnerError(RuntimeError):
    """Structured v4 evaluator failure."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _write_checkpoint(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    payload = _canonical_bytes(value)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload.decode("utf-8"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def _load_checkpoint(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise V4RunnerError("v4_checkpoint_permissions_invalid")
    if path.stat().st_size > 64 * 1024 * 1024:
        raise V4RunnerError("v4_checkpoint_too_large")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V4RunnerError("v4_checkpoint_invalid")
    if value.get("schema_version") != OBSERVATION_SCHEMA:
        raise V4RunnerError("v4_checkpoint_schema_mismatch")
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise V4RunnerError(f"v4_checkpoint_{key}_mismatch")
    return value


def _load_corpus() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cases = [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    relations = [
        json.loads(line)
        for line in RELATIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identity = corpus_identity()
    return {
        "manifest": manifest,
        "cases": cases,
        "relations": relations,
        "dataset_id": identity["dataset_id"],
        "manifest_sha256": identity["manifest_sha256"],
    }


def expected_route_labels(cases: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Stable per-case route labels derived only from the frozen corpus."""

    labels: dict[str, str] = {}
    for case in cases:
        case_id = str(case["case_id"])
        if not case.get("answerable", True) or case.get("negative_control_kind"):
            labels[case_id] = "negative_or_refusal"
            continue
        if case.get("semantic_intent") != "graph":
            labels[case_id] = "simple_only"
            continue
        gold = {str(item) for path in case.get("valid_paths", ()) for item in path}
        simple_gold = {
            str(item) for item in case.get("answer_gold_relation_ids", ())
        }
        labels[case_id] = (
            "graph_needed" if simple_gold and not gold <= simple_gold else "simple_only"
        )
    return labels


def gold_path_by_hop(cases: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    by_hop: dict[str, list[str]] = {"hop1": [], "hop2": [], "hop3": []}
    for case in cases:
        if case.get("semantic_intent") != "graph":
            continue
        for path in case.get("valid_paths", ()):
            hop_count = len(path)
            if hop_count in {1, 2, 3}:
                key = f"hop{hop_count}"
                if str(case["case_id"]) not in by_hop[key]:
                    by_hop[key].append(str(case["case_id"]))
    return by_hop


def _digest_path(path: Path | None) -> str:
    if path is None or not path.is_file():
        raise V4RunnerError("v4_sweep_output_missing")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "answers", "sweep", "report"))
    parser.add_argument(
        "--evaluation-runtime",
        type=Path,
        default=DEFAULT_RUNTIME_MANIFEST,
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sweep-output", type=Path)
    parser.add_argument("--locked-output", type=Path)
    parser.add_argument("--confirm")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the frozen corpus and labels without any external I/O",
    )
    parser.add_argument(
        "--repeat-override",
        type=int,
        default=ACTUAL_AUTO_REPEATS,
        help="actual-auto repeats per graph case (default 3)",
    )
    return parser


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.dry_run:
        if arguments.phase not in (None, "preflight"):
            parser.error("--dry-run does not execute phases")
        try:
            corpus = _load_corpus()
            labels = expected_route_labels(corpus["cases"])
            by_hop = gold_path_by_hop(corpus["cases"])
            print(
                json.dumps(
                    {
                        "status": "offline_dry_run_ok",
                        "dataset_id": corpus["dataset_id"],
                        "case_count": len(corpus["cases"]),
                        "manifest_sha256": corpus["manifest_sha256"],
                        "labels": labels,
                        "hop_gold": by_hop,
                    },
                    sort_keys=True,
                )
            )
            return 0
        except Exception as error:  # noqa: BLE001
            parser.exit(2, f"v4 dry run failed: {type(error).__name__}\n")
    if arguments.confirm != CONFIRM:
        parser.error(f"--confirm must equal {CONFIRM}")
    if arguments.phase == "answers" and arguments.checkpoint is None:
        parser.error("--checkpoint is required for the answers phase")
    if arguments.phase == "sweep" and arguments.sweep_output is None:
        parser.error("--sweep-output is required for the sweep phase")
    if arguments.phase == "report" and arguments.locked_output is None:
        parser.error("--locked-output is required for the report phase")
    try:
        result = asyncio.run(_run(arguments))
    except (
        EvaluationRuntimeError,
        R4RunnerError,
        V4RunnerError,
    ) as error:
        parser.exit(2, f"v4 runner failed: {type(error).__name__}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


async def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    runtime = load_evaluation_runtime(
        arguments.evaluation_runtime,
        require_adaptive_graph=True,
        allow_canonical_checkout=True,
    )
    identity = runtime.adaptive_graph
    if identity is None:
        raise V4RunnerError("v4_runtime_identity_missing")
    dependencies = build_worker_dependencies(env_file=runtime.env_file)
    try:
        await dependencies.check_readiness()
        workspace_id = dependencies.settings.identity.workspace_id
        if workspace_id != identity.workspace_id:
            raise V4RunnerError("v4_workspace_identity_changed")
        knowledge_base, bundle, model_configuration = await _load_runtime_facts(
            dependencies,
            kb_id=identity.knowledge_base_id,
            model_revision_id=identity.answer_profile_revision_id,
        )
        if (
            knowledge_base.name != EXPECTED_KB_NAME
            or knowledge_base.active_index_revision_id != identity.index_revision_id
        ):
            raise V4RunnerError("v4_knowledge_base_identity_changed")
        from rag_kb.adapters.graph_store.pg import PgGraphStore

        graph_store = PgGraphStore(dependencies.database.sessions)
        build = await graph_store.get_active_graphiti_build(
            workspace_id, identity.knowledge_base_id
        )
        if (
            build is None
            or build.build_id != identity.graph_build_id
            or build.index_revision_id != identity.index_revision_id
        ):
            raise V4RunnerError("v4_graph_build_identity_changed")
        episode_uuid = await graph_store.first_graphiti_episode_uuid(
            workspace_id, identity.knowledge_base_id, build.build_id
        )
        if not await dependencies.graphiti_runtime.probe(
            build, episode_uuid=episode_uuid, require_complete=True
        ):
            raise V4RunnerError("v4_graph_runtime_not_ready")

        corpus = _load_corpus()
        labels = expected_route_labels(corpus["cases"])
        by_hop = gold_path_by_hop(corpus["cases"])
        runtime_identity = _runtime_identity(
            workspace_id=workspace_id,
            knowledge_base=knowledge_base,
            build=build,
            index_revision_id=identity.index_revision_id,
        )
        runtime_identity.update(
            {
                "graph_edge_limit": GRAPH_EDGE_LIMIT,
                "graph_source_chunk_target": SOURCE_CHUNK_TARGET,
                "graph_source_chunk_limit": SOURCE_CHUNK_LIMIT,
                "graph_call_timeout_seconds": GRAPH_CALL_TIMEOUT_SECONDS,
                "graph_call_limit": GRAPH_CALL_LIMIT,
                "actual_auto_repeats": arguments.repeat_override,
            }
        )
        serving_rows = await _serving_chunk_rows(
            dependencies,
            workspace_id=workspace_id,
            kb_id=identity.knowledge_base_id,
            index_revision_id=identity.index_revision_id,
        )
        expected_filenames = {
            str(row["filename"]) for row in corpus["manifest"]["documents"]
        }
        actual_filenames = {str(row.get("original_filename")) for row in serving_rows}
        if actual_filenames != expected_filenames:
            raise V4RunnerError("v4_serving_document_set_changed")

        if arguments.phase == "preflight":
            return {
                "status": "preflight_ok",
                "dataset_id": corpus["dataset_id"],
                "case_count": len(corpus["cases"]),
                "labels": labels,
                "hop_gold": by_hop,
                "runtime_identity": runtime_identity,
            }

        relation_to_chunk = relation_chunk_map(
            manifest=corpus["manifest"],
            relations=corpus["relations"],
            serving_rows=serving_rows,
        )
        if arguments.phase == "sweep":
            return await _run_sweep(
                arguments, dependencies, build, identity, relation_to_chunk, corpus
            )
        if arguments.phase == "answers":
            return await _run_answers(
                arguments,
                dependencies,
                identity,
                corpus,
                labels,
                runtime_identity,
            )
        if arguments.phase == "report":
            return _run_report(arguments, corpus, labels)
        raise V4RunnerError("v4_phase_required")
    finally:
        await dependencies.close()


async def _run_sweep(
    arguments: argparse.Namespace,
    dependencies: Any,
    build: Any,
    identity: Any,
    relation_to_chunk: Mapping[str, Any],
    corpus: Mapping[str, Any],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for case in corpus["cases"]:
        if case.get("semantic_intent") != "graph":
            continue
        gold_paths = [
            [str(item) for item in path] for path in case.get("valid_paths", ())
        ]
        candidate_k: dict[str, Any] = {}
        for edge_limit in SWEEP_EDGE_LIMITS:
            try:
                candidate_set = (
                    await dependencies.retrieval_service._search_graphiti_candidates(  # noqa: SLF001
                        identity.workspace_id,
                        identity.knowledge_base_id,
                        build=build,
                        index_revision_id=identity.index_revision_id,
                        query=case["question"],
                        edge_limit=edge_limit,
                        rerank_mode=RerankMode.CLASSIC,
                    )
                )
            except Exception:  # noqa: BLE001
                candidate_k[str(edge_limit)] = {"status": "unavailable"}
                continue
            chunk_ids = {item.index_chunk_id for item in candidate_set.traversal.chunks}
            raw_ids = set(candidate_set.raw_chunk_ids)
            layers: dict[str, Any] = {
                "status": "ok",
                "raw_relation_ids": _relation_ids_for_chunks(
                    raw_ids, relation_to_chunk=relation_to_chunk
                ),
                "hydrated_relation_ids": _relation_ids_for_chunks(
                    chunk_ids, relation_to_chunk=relation_to_chunk
                ),
                "hop_counts": {
                    "hop1": sum(
                        path.hop_count == 1 for path in candidate_set.traversal.paths
                    ),
                    "hop2": sum(
                        path.hop_count == 2 for path in candidate_set.traversal.paths
                    ),
                    "hop3": sum(
                        path.hop_count == 3 for path in candidate_set.traversal.paths
                    ),
                },
            }
            for pack_target in SWEEP_PACK_TARGETS:
                packed, _ = _pack_graph_search_evidence(
                    candidate_set,
                    excluded_index_chunk_ids=frozenset(),
                    source_chunk_target=pack_target,
                    source_chunk_limit=SOURCE_CHUNK_LIMIT,
                )
                layers[f"packed_{pack_target}_relation_ids"] = _relation_ids_for_chunks(
                    {item.index_chunk_id for item in packed},
                    relation_to_chunk=relation_to_chunk,
                )
            candidate_k[str(edge_limit)] = layers
        records.append(
            {
                "case_id": str(case["case_id"]),
                "gold_paths": gold_paths,
                "candidate_k": candidate_k,
            }
        )
    payload = {
        "schema_version": OBSERVATION_SCHEMA,
        "dataset_id": corpus["dataset_id"],
        "corpus_sha256": corpus["manifest_sha256"],
        "kind": "sweep",
        "records": records,
    }
    sweep_path = arguments.sweep_output
    assert sweep_path is not None
    _write_checkpoint(sweep_path, payload)
    return {
        "status": "sweep_completed",
        "sweep_case_count": len(records),
        "sweep_sha256": _digest_path(sweep_path),
    }


async def _run_answers(
    arguments: argparse.Namespace,
    dependencies: Any,
    identity: Any,
    corpus: Mapping[str, Any],
    labels: Mapping[str, str],
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    from tools.run_open_source_rag_v3 import _evaluator_chat_model

    chat_model, _ = await _evaluator_chat_model(
        dependencies, bundle=None, model_override=None
    )
    observing_model = ForcedGraphSearchChatModelPort(
        chat_model, controller_mode="actual_auto"
    )
    agent = NativeToolCallingAgent(
        observing_model,
        ChatEvidenceRetriever(dependencies.retrieval_service),
        dependencies.visual_evidence_preparer,
        min_cosine_similarity=dependencies.settings.retrieval.min_cosine_similarity,
        min_rerank_score=dependencies.settings.retrieval.min_rerank_score,
        cross_modal_min_cosine_similarity=(
            dependencies.settings.retrieval.cross_modal_min_cosine_similarity
        ),
    )
    expected = {
        "schema_version": OBSERVATION_SCHEMA,
        "dataset_id": corpus["dataset_id"],
        "corpus_sha256": corpus["manifest_sha256"],
        "runtime": dict(runtime_identity),
    }
    checkpoint_path = arguments.checkpoint
    assert checkpoint_path is not None
    checkpoint = _load_checkpoint(checkpoint_path, expected)
    observations = checkpoint.setdefault("case_observations", [])
    total_tokens = 0
    total_cost_usd = 0.0
    plan: list[tuple[dict[str, Any], str, int]] = []
    for case in corpus["cases"]:
        label = labels[str(case["case_id"])]
        if label == "graph_needed":
            for repeat in range(1, arguments.repeat_override + 1):
                plan.append((case, "auto", repeat))
        plan.append((case, "simple", 1))
    for case, lane, repeat in plan:
        case_id = str(case["case_id"])
        if len(observations) >= ANSWER_EXECUTION_LIMIT:
            raise V4RunnerError("v4_answer_execution_budget_exhausted")
        observation = await _observe_case(
            agent,
            dependencies,
            identity=identity,
            case=case,
            lane=lane,
            repeat=repeat,
        )
        observations.append(observation)
        total_tokens += observation["usage"]["total_tokens"]
        total_cost_usd += observation["estimated_cost_usd"]
        if total_tokens > TOTAL_TOKEN_LIMIT or total_cost_usd > TOTAL_COST_LIMIT_USD:
            raise V4RunnerError("v4_usage_budget_exhausted")
        _write_checkpoint(checkpoint_path, checkpoint)
        print(
            json.dumps(
                {
                    "event": "v4_case_completed",
                    "case_id": case_id,
                    "lane": lane,
                    "repeat": repeat,
                    "completed_observation_count": len(observations),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    checkpoint["labels"] = dict(labels)
    checkpoint["completed_observation_count"] = len(observations)
    checkpoint_sha256 = _write_checkpoint(checkpoint_path, checkpoint)
    return {
        "status": "answers_completed",
        "observation_count": len(observations),
        "checkpoint_sha256": checkpoint_sha256,
    }


async def _observe_case(
    agent: NativeToolCallingAgent,
    dependencies: Any,
    *,
    identity: Any,
    case: Mapping[str, Any],
    lane: str,
    repeat: int,
) -> dict[str, Any]:
    started = datetime.now(UTC)
    retrieval_strategy = (
        exact_profile(top_k=SIMPLE_TOP_K, rerank_mode=RerankMode.CLASSIC).as_dict()
        if lane == "simple"
        else adaptive_graphiti_profile(
            top_k=SIMPLE_TOP_K, rerank_mode=RerankMode.CLASSIC
        ).as_dict()
    )
    context = _execution_context(
        settings=dependencies.settings,
        kb_id=identity.knowledge_base_id,
        index_revision_id=identity.index_revision_id,
        question=str(case["question"]),
        model_configuration=dependencies.settings.model.chat_model_configuration(),
        rerank_mode=RerankMode.CLASSIC,
    )
    context = replace(context, retrieval_strategy=retrieval_strategy)
    state = await agent.run(context)
    trace = state.artifacts.get(AGENT_TRACE_ARTIFACT)
    if trace is None or not hasattr(trace, "as_dict"):
        raise V4RunnerError("v4_agent_trace_missing")
    answered = state.answering
    if answered is None or answered.validated is None or answered.rendered is None:
        raise V4RunnerError("v4_answer_observation_missing")
    route = summarize_graph_route_trace(trace.as_dict())
    graph_events = [
        event
        for event in trace.events
        if getattr(event, "retrieval_lane", None) == "graph_relations"
    ]
    increments = {"first": 0, "second": 0}
    durations: list[int] = []
    timeout_count = 0
    for event in graph_events:
        new_count = event.new_evidence_count or 0
        if event.call_index == 1:
            increments["first"] += new_count
        elif event.call_index == 2:
            increments["second"] += new_count
        if event.duration_ms is not None:
            durations.append(event.duration_ms)
        if event.route_result_code == "timeout":
            timeout_count += 1
    chatrun_ms = int((datetime.now(UTC) - started).total_seconds() * 1000)
    if chatrun_ms >= CHATRUN_DEADLINE_SECONDS * 1000:
        raise V4RunnerError("v4_chatrun_deadline_exceeded")
    usage = trace.as_dict()["usage"]
    return {
        "case_id": str(case["case_id"]),
        "lane": lane,
        "repeat": repeat,
        "actual_outcome": answered.validated.outcome.value,
        "graph_route_attempted": bool(route["graph_route_attempted"]),
        "graph_route_admitted": bool(route["graph_route_admitted"]),
        "graph_new_evidence_count": int(route["graph_new_evidence_count"]),
        "graph_call_count": sum(1 for event in graph_events if event.status == "ok"),
        "increments": increments,
        "graph_call_duration_ms": durations,
        "timeout_count": timeout_count,
        "chatrun_duration_ms": chatrun_ms,
        "usage": usage,
        "estimated_cost_usd": (
            usage.get("input_tokens", 0) / 1_000_000 * MIMO_INPUT_USD_PER_MILLION
            + usage.get("output_tokens", 0) / 1_000_000 * MIMO_OUTPUT_USD_PER_MILLION
        ),
        "forbidden_claim_hit": _forbidden_claim_hit(
            answered.rendered.content, case.get("forbidden_claims", ())
        ),
    }


def _run_report(
    arguments: argparse.Namespace,
    corpus: Mapping[str, Any],
    labels: Mapping[str, str],
) -> dict[str, Any]:
    checkpoint_path = arguments.checkpoint
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise V4RunnerError("v4_report_requires_checkpoint")
    checkpoint = _load_checkpoint(checkpoint_path, {})
    observations = checkpoint.get("case_observations", [])
    if not isinstance(observations, list) or not observations:
        raise V4RunnerError("v4_report_requires_observations")
    auto = [row for row in observations if row.get("lane") == "auto"]
    simple = [row for row in observations if row.get("lane") == "simple"]
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in auto:
        by_case[row["case_id"]].append(row)
    tp = fp = fn = 0
    graph_needed_denominator = 0
    repeats_observed = 0
    first_ranges: list[list[int]] = []
    for case_id, repeats in sorted(by_case.items()):
        repeats_observed = max(repeats_observed, len(repeats))
        needed = labels.get(case_id) == "graph_needed"
        admitted = any(row["graph_route_admitted"] for row in repeats)
        if needed:
            graph_needed_denominator += 1
        tp += needed and admitted
        fp += not needed and admitted
        fn += needed and not admitted
        first_ranges.append([row["increments"]["first"] for row in repeats])
    recall = tp / graph_needed_denominator if graph_needed_denominator else None
    precision = tp / (tp + fp) if tp + fp else None
    first_increments = [
        row["increments"]["first"] for row in auto if row["graph_route_attempted"]
    ]
    second_increments = [
        row["increments"]["second"] for row in auto if row["graph_route_attempted"]
    ]
    durations = [ms for row in auto for ms in row["graph_call_duration_ms"]]
    p95 = (
        sorted(durations)[min(len(durations) - 1, int(len(durations) * 0.95))]
        if durations
        else None
    )
    timeout_count = sum(row["timeout_count"] for row in auto)
    chatrun_max = max((row["chatrun_duration_ms"] for row in auto), default=None)
    forbidden_hits = [row["forbidden_claim_hit"] for row in observations]
    report = {
        "schema_version": LOCKED_SCHEMA,
        "dataset_id": corpus["dataset_id"],
        "corpus_sha256": corpus["manifest_sha256"],
        "runtime": checkpoint.get("runtime"),
        "labels": dict(labels),
        "repeat_count": repeats_observed,
        "graph_needed_case_count": graph_needed_denominator,
        "simple_only_case_count": sum(
            1 for value in labels.values() if value == "simple_only"
        ),
        "negative_or_refusal_case_count": sum(
            1 for value in labels.values() if value == "negative_or_refusal"
        ),
        "hop_gold": gold_path_by_hop(corpus["cases"]),
        "route": {
            "recall": recall,
            "precision": precision,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "recall_threshold": ROUTE_RECALL_THRESHOLD,
            "precision_threshold": ROUTE_PRECISION_THRESHOLD,
            "recall_met": recall is not None and recall >= ROUTE_RECALL_THRESHOLD,
            "precision_met": (
                precision is not None and precision >= ROUTE_PRECISION_THRESHOLD
            ),
        },
        "graph_call_increments": {
            "first_call_new_evidence_mean": (
                sum(first_increments) / len(first_increments)
                if first_increments
                else None
            ),
            "second_call_new_evidence_mean": (
                sum(second_increments) / len(second_increments)
                if second_increments
                else None
            ),
            "repeat_first_increment_min": min(
                (min(r) for r in first_ranges), default=None
            ),
            "repeat_first_increment_max": max(
                (max(r) for r in first_ranges), default=None
            ),
        },
        "graph_call_timeouts": timeout_count,
        "graph_call_duration_p95_ms": p95,
        "chatrun_max_duration_ms": chatrun_max,
        "forbidden_claim_clear": not any(forbidden_hits),
        "observation_counts": {"auto": len(auto), "simple": len(simple)},
        "acceptance": {
            "route_recall": recall is not None and recall >= ROUTE_RECALL_THRESHOLD,
            "route_precision": (
                precision is not None and precision >= ROUTE_PRECISION_THRESHOLD
            ),
            "timeout_rate_is_zero": timeout_count == 0,
            "graph_call_p95_under_90s": (
                p95 is not None and p95 < GRAPH_CALL_TIMEOUT_SECONDS * 1000
            ),
            "chatrun_under_420s": (
                chatrun_max is not None
                and chatrun_max < CHATRUN_DEADLINE_SECONDS * 1000
            ),
            "forbidden_claim_clear": not any(forbidden_hits),
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    if arguments.locked_output is not None:
        _write_checkpoint(arguments.locked_output, report)
    return report


if __name__ == "__main__":
    raise SystemExit(main())
