#!/usr/bin/env python3
"""Run the explicitly authorized, read-only adaptive Graph RAG R4 diagnostic.

The runner is deliberately narrow: it targets only ``routing-rag-v1`` Graph
cases, creates no ChatRun, performs no Judge call, and never mutates a
knowledge base or Graphiti build.  It captures the validated Forced supplement
query at the Agent retrieval boundary, then replays Capability and Agent query
columns against the same Simple exclusions.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import bindparam, text

from apps.worker.dependencies import build_worker_dependencies
from rag_kb.adapters.graph_store.postgres import PgGraphStore
from rag_kb.adapters.model_api.langchain_chat import LangChainChatModelAdapter
from rag_kb.adapters.model_secrets.local import LocalModelSecretStore
from rag_kb.answering.agent import NativeToolCallingAgent
from rag_kb.domain import (
    ChatExecutionContext,
    ChatRunLease,
    GraphitiSearchQuery,
    ModelKind,
    ModelValidationStatus,
    RerankMode,
)
from rag_kb.retrieval.profile import adaptive_graphiti_profile
from rag_kb.retrieval.service import (
    GraphitiCandidateSet,
    _pack_graphiti_supplement_evidence,
)
from rag_kb.services.chat_execution import ChatEvidenceRetriever
from rag_kb.uow import (
    TransactionMode,
    UnitOfWorkPurpose,
    execute_in_transaction,
)
from tools.evaluate_adaptive_graph_route import (
    CapturingGraphitiSupplementRetriever,
    ForcedGraphitiSupplementChatModelPort,
    GraphitiSupplementCapture,
    GraphitiSupplementCaptureComplete,
    align_chunk_layers,
    build_replay_capture_artifact,
    diagnostic_record,
    load_cases,
    load_manifest,
    manifest_digest,
    write_replay_capture_artifact,
)


CONFIRM_EXTERNAL_CALLS = "RUN_ROUTING_RAG_R4_EXTERNAL_CALLS"
R4_DIAGNOSTIC_SCHEMA_VERSION = "adaptive_graph_r4_diagnostic_v1"
GRAPHITI_EDGE_LIMIT = 8
FORCED_CONTROLLER_MODE = "specific_tool_choice"
ACTUAL_AUTO_CONTROLLER_MODE = "actual_auto"
EVALUATOR_CHAT_MODEL_OVERRIDE = "deepseek-v4-flash"
EVALUATOR_CHAT_MODEL_MAX_OUTPUT_TOKENS = 512
EVALUATOR_CHAT_MODEL_MAX_RETRIES = 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-base-id", type=UUID, required=True)
    parser.add_argument("--index-revision-id", type=UUID, required=True)
    parser.add_argument("--graph-build-id", type=UUID, required=True)
    parser.add_argument("--chat-model-profile-revision-id", type=UUID, required=True)
    parser.add_argument("--capture-output", type=Path, required=True)
    parser.add_argument("--diagnostic-output", type=Path, required=True)
    parser.add_argument(
        "--replay-mode",
        choices=("forced", "actual-auto"),
        default="forced",
    )
    parser.add_argument(
        "--chat-model-override",
        choices=(EVALUATOR_CHAT_MODEL_OVERRIDE,),
        help="Evaluator-only model override using the selected profile's provider.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--confirm", required=True)
    return parser


def _write_json_artifact(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.write("\n")
    os.chmod(path, 0o600)
    return hashlib.sha256((payload + "\n").encode("utf-8")).hexdigest()


async def _load_runtime_facts(dependencies, *, kb_id: UUID, model_revision_id: UUID):
    async def load(unit_of_work):
        knowledge_base = await unit_of_work.knowledge_bases.get(kb_id)
        bundle = await unit_of_work.model_settings.get_profile_revision(
            model_revision_id
        )
        return knowledge_base, bundle

    knowledge_base, bundle = await execute_in_transaction(
        dependencies.unit_of_work,
        load,
        purpose=UnitOfWorkPurpose.READ_SNAPSHOT,
        mode=TransactionMode.REPEATABLE_READ_ONLY,
    )
    if knowledge_base is None:
        raise RuntimeError("r4_knowledge_base_not_found")
    if bundle is None:
        raise RuntimeError("r4_chat_model_revision_not_found")
    if (
        bundle.profile.kind is not ModelKind.CHAT
        or not bundle.profile.enabled
        or not bundle.provider.enabled
        or bundle.current_revision.validation_status is not ModelValidationStatus.VALID
    ):
        raise RuntimeError("r4_chat_model_revision_not_ready")
    parameters = dict(bundle.current_revision.configuration)
    model_configuration = {
        "model_profile_revision_id": str(bundle.current_revision.id),
        "resolved_model": bundle.current_revision.model,
        "max_tokens": parameters.get("max_output_tokens", 4096),
    }
    return knowledge_base, bundle, model_configuration


async def _evaluator_chat_model(dependencies, *, bundle, model_override: str | None):
    if model_override is None:
        parameters = dict(bundle.current_revision.configuration)
        return dependencies.chat_model_adapter, {
            "chat_model": bundle.current_revision.model,
            "chat_model_source": "profile_revision",
            "chat_model_max_output_tokens": parameters.get(
                "max_output_tokens",
                4096,
            ),
            "chat_model_max_retries": bundle.provider_revision.max_retries,
        }
    if model_override != EVALUATOR_CHAT_MODEL_OVERRIDE:
        raise ValueError("r4_chat_model_override_not_allowed")
    secret_store = LocalModelSecretStore(
        dependencies.settings.model_secrets.root_path
    )
    api_key = await asyncio.to_thread(
        secret_store.read,
        bundle.provider_revision.secret_reference,
    )
    adapter = LangChainChatModelAdapter(
        base_url=bundle.provider_revision.base_url,
        api_key=api_key,
        model=model_override,
        timeout_seconds=min(bundle.provider_revision.timeout_seconds, 60.0),
        max_retries=EVALUATOR_CHAT_MODEL_MAX_RETRIES,
        max_concurrency=1,
        temperature=0.0,
        top_p=0.9,
        sampling_top_k=None,
        max_tokens=EVALUATOR_CHAT_MODEL_MAX_OUTPUT_TOKENS,
        thinking_enabled=False,
        reasoning_effort="off",
    )
    return adapter, {
        "chat_model": model_override,
        "chat_model_source": "evaluator_override",
        "chat_model_max_output_tokens": EVALUATOR_CHAT_MODEL_MAX_OUTPUT_TOKENS,
        "chat_model_max_retries": EVALUATOR_CHAT_MODEL_MAX_RETRIES,
    }


async def _serving_chunk_rows(
    dependencies,
    *,
    workspace_id: UUID,
    kb_id: UUID,
    index_revision_id: UUID,
) -> tuple[Mapping[str, Any], ...]:
    statement = text(
        """
        SELECT chunk.id AS index_chunk_id,
               chunk.modality,
               chunk.content,
               chunk.source_location,
               chunk.hierarchy,
               chunk.source_metadata,
               version.original_filename
        FROM index_chunk chunk
        JOIN indexed_document_version target
          ON target.workspace_id = chunk.workspace_id
         AND target.kb_id = chunk.kb_id
         AND target.id = chunk.indexed_document_version_id
         AND target.index_revision_id = :index_revision_id
         AND target.build_status = 'ready'
         AND target.serving_status = 'serving'
        JOIN document_version version
          ON version.workspace_id = target.workspace_id
         AND version.kb_id = target.kb_id
         AND version.id = target.document_version_id
        WHERE chunk.workspace_id = :workspace_id
          AND chunk.kb_id = :kb_id
          AND chunk.excluded_at IS NULL
        ORDER BY version.original_filename, chunk.ordinal, chunk.id
        """
    )
    async with dependencies.database.sessions() as session:
        async with session.begin():
            await session.execute(text("SET TRANSACTION READ ONLY"))
            rows = (
                await session.execute(
                    statement,
                    {
                        "workspace_id": workspace_id,
                        "kb_id": kb_id,
                        "index_revision_id": index_revision_id,
                    },
                )
            ).mappings().all()
    return tuple(dict(row) for row in rows)


def _nested_strings(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _nested_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _nested_strings(item)
    elif value is not None:
        yield str(value)


def _relation_locator_chunk_ids(
    locator: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    relation_id = str(locator.get("relation_id", "")).strip()
    if not relation_id:
        raise ValueError("r4_graph_relation_locator_invalid")
    matches: list[str] = []
    relation_pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(relation_id)}(?![A-Za-z0-9])"
    )
    for row in rows:
        metadata = (
            row.get("content"),
            row.get("source_location"),
            row.get("hierarchy"),
            row.get("source_metadata"),
        )
        values = {item for value in metadata for item in _nested_strings(value)}
        if any(relation_pattern.search(value) for value in values):
            matches.append(str(row["index_chunk_id"]))
    unique = tuple(dict.fromkeys(matches))
    if len(unique) != 1:
        raise RuntimeError(
            f"r4_graph_relation_locator_not_unique:{relation_id}:{len(unique)}"
        )
    return unique


def _locator_chunk_ids(
    locators: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    chunk_ids: list[str] = []
    for locator in locators:
        if locator.get("kind") != "graph_relation":
            raise ValueError("r4_runner_accepts_graph_relation_locators_only")
        chunk_ids.extend(_relation_locator_chunk_ids(locator, rows))
    return tuple(dict.fromkeys(chunk_ids))


def _execution_context(
    *,
    settings,
    kb_id: UUID,
    index_revision_id: UUID,
    question: str,
    model_configuration: Mapping[str, Any],
) -> ChatExecutionContext:
    run_id = uuid4()
    workspace_id = settings.identity.workspace_id
    return ChatExecutionContext(
        lease=ChatRunLease(
            run_id=run_id,
            workspace_id=workspace_id,
            claimed_by="adaptive-graph-r4-evaluator",
            attempt=1,
            claimed_at=datetime.now(UTC),
        ),
        run_id=run_id,
        workspace_id=workspace_id,
        knowledge_base_id=kb_id,
        session_id=uuid4(),
        user_message_id=uuid4(),
        assistant_message_id=uuid4(),
        index_revision_id=index_revision_id,
        principal_id=settings.identity.principal_id,
        client_id=settings.identity.client_id,
        query=question,
        effective_policy={"insufficiency_policy": "partial_answer"},
        retrieval_strategy=adaptive_graphiti_profile(
            top_k=10,
            rerank_mode=RerankMode.CLASSIC,
        ).as_dict(),
        model_configuration=model_configuration,
        attempt=1,
    )


async def _capture_queries(
    dependencies,
    *,
    chat_model_adapter,
    cases: Sequence[Mapping[str, Any]],
    kb_id: UUID,
    index_revision_id: UUID,
    model_configuration: Mapping[str, Any],
    replay_mode: str,
) -> tuple[
    tuple[GraphitiSupplementCapture, ...],
    dict[str, Mapping[str, Any]],
    dict[str, tuple[str, ...]],
]:
    controller_mode = (
        ACTUAL_AUTO_CONTROLLER_MODE
        if replay_mode == "actual-auto"
        else FORCED_CONTROLLER_MODE
    )
    forced_model = ForcedGraphitiSupplementChatModelPort(
        chat_model_adapter,
        controller_mode=controller_mode,
    )
    capture_retriever = CapturingGraphitiSupplementRetriever(
        ChatEvidenceRetriever(dependencies.retrieval_service),
        stop_after_capture=True,
    )
    retrieval_settings = dependencies.settings.retrieval
    agent = NativeToolCallingAgent(
        forced_model,
        capture_retriever,
        dependencies.visual_evidence_preparer,
        min_cosine_similarity=retrieval_settings.min_cosine_similarity,
        min_rerank_score=retrieval_settings.min_rerank_score,
        cross_modal_min_cosine_similarity=(
            retrieval_settings.cross_modal_min_cosine_similarity
        ),
    )
    captures: list[GraphitiSupplementCapture] = []
    metrics: dict[str, Mapping[str, Any]] = {}
    simple_ids_by_case: dict[str, tuple[str, ...]] = {}
    for case in cases:
        case_id = str(case["case_id"])
        forced_model.reset_case()
        capture_retriever.begin_case(case_id)
        try:
            await agent.run(
                _execution_context(
                    settings=dependencies.settings,
                    kb_id=kb_id,
                    index_revision_id=index_revision_id,
                    question=str(case["question"]),
                    model_configuration=model_configuration,
                )
            )
        except GraphitiSupplementCaptureComplete:
            pass
        except BaseException:
            capture_retriever.abandon_case()
            raise
        capture, simple_ids = capture_retriever.finish_observed_case()
        if replay_mode == "forced" and capture is None:
            tool_names = ",".join(
                "+".join(names) if names else "none"
                for names in forced_model.response_tool_names
            )
            raise RuntimeError(
                "r4_supplement_capture_incomplete:"
                f"{case_id}:replacements={forced_model.replacements}:"
                f"model_calls={forced_model.model_calls}:tools={tool_names}:"
                f"finish_reasons={forced_model.response_finish_reasons}:"
                f"usage={dict(forced_model.usage)}"
            )
        expected_replacements = 1 if replay_mode == "forced" else 0
        if forced_model.replacements != expected_replacements:
            raise RuntimeError("r4_forced_replacement_cardinality")
        if capture is not None:
            if set(capture.excluded_index_chunk_ids) != set(simple_ids):
                raise RuntimeError("r4_simple_exclusion_capture_mismatch")
            captures.append(capture)
        simple_ids_by_case[case_id] = simple_ids
        metrics[case_id] = {
            "forced_replacements": forced_model.replacements,
            "model_calls": forced_model.model_calls,
            "usage": dict(forced_model.usage),
            "finish_reasons": tuple(forced_model.response_finish_reasons),
            "route_requested": capture is not None,
        }
        print(
            json.dumps(
                {
                    "event": "r4_capture_case_completed",
                    "case_id": case_id,
                    "model_calls": forced_model.model_calls,
                    "forced_replacements": forced_model.replacements,
                    "route_requested": capture is not None,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return tuple(captures), metrics, simple_ids_by_case


async def _raw_episode_mapping(
    dependencies,
    *,
    workspace_id: UUID,
    kb_id: UUID,
    build_id: UUID,
    episode_ids: tuple[str, ...],
) -> dict[str, str]:
    if not episode_ids:
        return {}
    statement = text(
        """
        SELECT episode_uuid, index_chunk_id
        FROM graphiti_episode_chunk
        WHERE workspace_id = :workspace_id
          AND kb_id = :kb_id
          AND build_id = :build_id
          AND episode_uuid IN :episode_ids
        ORDER BY created_at, id
        """
    ).bindparams(bindparam("episode_ids", expanding=True))
    async with dependencies.database.sessions() as session:
        async with session.begin():
            await session.execute(text("SET TRANSACTION READ ONLY"))
            rows = (
                await session.execute(
                    statement,
                    {
                        "workspace_id": workspace_id,
                        "kb_id": kb_id,
                        "build_id": build_id,
                        "episode_ids": episode_ids,
                    },
                )
            ).mappings().all()
    return {str(row["episode_uuid"]): str(row["index_chunk_id"]) for row in rows}


async def _column_layers(
    dependencies,
    graph_store: PgGraphStore,
    *,
    build,
    query: str,
    excluded_chunk_ids: tuple[str, ...],
) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    edges = await dependencies.graphiti_runtime.search(
        build,
        GraphitiSearchQuery(
            workspace_id=build.workspace_id,
            knowledge_base_id=build.knowledge_base_id,
            build_id=build.build_id,
            group_id=build.group_id,
            query=query,
            limit=GRAPHITI_EDGE_LIMIT,
        ),
    )
    episode_ids = tuple(
        dict.fromkeys(
            episode_id
            for edge in edges
            for episode_id in edge.episode_uuids
        )
    )
    raw_mapping = await _raw_episode_mapping(
        dependencies,
        workspace_id=build.workspace_id,
        kb_id=build.knowledge_base_id,
        build_id=build.build_id,
        episode_ids=episode_ids,
    )
    raw_chunk_ids = tuple(
        dict.fromkeys(
            raw_mapping[episode_id]
            for episode_id in episode_ids
            if episode_id in raw_mapping
        )
    )
    hydrated = await graph_store.hydrate_graphiti_edges(
        workspace_id=build.workspace_id,
        knowledge_base_id=build.knowledge_base_id,
        build_id=build.build_id,
        index_revision_id=build.index_revision_id,
        edges=edges,
    )
    if hydrated is None or hydrated.resolved_active_revision_id != build.index_revision_id:
        raise RuntimeError("r4_graph_hydration_revision_mismatch")
    edge_rank_by_path_id = {path.path_id: path.rank for path in hydrated.paths}
    reranked, scores = (
        await dependencies.retrieval_service._rerank_graphiti_candidates_with_scores(
            query,
            hydrated,
        )
    )
    candidate_set = GraphitiCandidateSet(
        build=build,
        traversal=reranked,
        edge_rank_by_path_id=edge_rank_by_path_id,
        rerank_score_by_chunk_id=scores,
    )
    packed = _pack_graphiti_supplement_evidence(
        candidate_set,
        excluded_index_chunk_ids=frozenset(UUID(item) for item in excluded_chunk_ids),
    )
    layers = {
        "raw": raw_chunk_ids,
        "hydrated": tuple(str(item.index_chunk_id) for item in hydrated.chunks),
        "reranked": tuple(str(item.index_chunk_id) for item in reranked.chunks),
        "packed": tuple(str(item.index_chunk_id) for item in packed),
    }
    packed_ids = set(layers["packed"])
    excluded_ids = set(excluded_chunk_ids)
    metrics = {
        "raw_edge_count": len(edges),
        "raw_episode_count": len(episode_ids),
        "raw_mapped_chunk_count": len(raw_chunk_ids),
        "no_mapping_episode_count": sum(
            episode_id not in raw_mapping for episode_id in episode_ids
        ),
        "hydrated_chunk_count": len(hydrated.chunks),
        "below_threshold_count": len(hydrated.chunks) - len(reranked.chunks),
        "reranked_chunk_count": len(reranked.chunks),
        "packed_chunk_count": len(packed),
        "budget_dropped_count": sum(
            str(item.index_chunk_id) not in packed_ids
            and str(item.index_chunk_id) not in excluded_ids
            for item in reranked.chunks
        ),
    }
    return layers, metrics


def _aggregate_graph_cases(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"case_count": len(records), "columns": {}}
    for column in ("capability", "agent_replay"):
        simple_missing = [
            item
            for item in records
            if not set(item["columns"][column]["answer_gold_chunk_ids"])
            <= set(item["columns"][column]["simple_chunk_ids"])
        ]
        first_loss_records = (
            [
                item
                for item in simple_missing
                if item.get("agent_replay_status") == "requested"
            ]
            if column == "agent_replay"
            else simple_missing
        )
        first_losses = Counter(
            item["columns"][column]["first_loss_layer"] or "none"
            for item in first_loss_records
        )
        result["columns"][column] = {
            "simple_missing_answer_gold": len(simple_missing),
            "answer_already_in_simple": len(records) - len(simple_missing),
            "route_requested": (
                sum(item.get("agent_replay_status") == "requested" for item in records)
                if column == "agent_replay"
                else len(records)
            ),
            "not_requested": (
                sum(
                    item.get("agent_replay_status") == "not_requested"
                    for item in records
                )
                if column == "agent_replay"
                else 0
            ),
            "distinct_benefit": sum(
                bool(item["columns"][column]["benefit"])
                for item in simple_missing
            ),
            "first_loss": dict(sorted(first_losses.items())),
            "redundant_hit": sum(
                bool(item["columns"][column]["redundant_hit"])
                for item in records
            ),
            "duplicate_count": sum(
                int(item["columns"][column]["duplicate_count"])
                for item in records
            ),
            "non_gold_admitted_count": sum(
                int(item["columns"][column]["non_gold_admitted_count"])
                for item in records
            ),
            "layer_metrics": {
                key: sum(int(item["layer_metrics"][column][key]) for item in records)
                for key in (
                    "raw_edge_count",
                    "raw_episode_count",
                    "raw_mapped_chunk_count",
                    "no_mapping_episode_count",
                    "hydrated_chunk_count",
                    "below_threshold_count",
                    "reranked_chunk_count",
                    "packed_chunk_count",
                    "budget_dropped_count",
                )
            },
        }
    return result


async def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest()
    cases = tuple(
        case
        for case in load_cases(Path(manifest["case_file"]))
        if case.get("expected_route", {}).get("route") == "graph"
    )
    if len(cases) != 20:
        raise RuntimeError("r4_graph_case_count_changed")
    dependencies = build_worker_dependencies(env_file=None)
    try:
        await dependencies.check_readiness()
        workspace_id = dependencies.settings.identity.workspace_id
        knowledge_base, bundle, model_configuration = await _load_runtime_facts(
            dependencies,
            kb_id=arguments.knowledge_base_id,
            model_revision_id=arguments.chat_model_profile_revision_id,
        )
        chat_model_adapter, chat_model_runtime = await _evaluator_chat_model(
            dependencies,
            bundle=bundle,
            model_override=arguments.chat_model_override,
        )
        model_configuration = dict(model_configuration)
        model_configuration["resolved_model"] = chat_model_runtime["chat_model"]
        model_configuration["max_tokens"] = chat_model_runtime[
            "chat_model_max_output_tokens"
        ]
        if knowledge_base.active_index_revision_id != arguments.index_revision_id:
            raise RuntimeError("r4_active_revision_changed")
        graph_store = PgGraphStore(dependencies.database.sessions)
        build = await graph_store.get_active_graphiti_build(
            workspace_id,
            arguments.knowledge_base_id,
        )
        if (
            build is None
            or build.build_id != arguments.graph_build_id
            or build.index_revision_id != arguments.index_revision_id
        ):
            raise RuntimeError("r4_active_graph_build_changed")
        episode_uuid = await graph_store.first_graphiti_episode_uuid(
            workspace_id,
            arguments.knowledge_base_id,
            build.build_id,
        )
        if not await dependencies.graphiti_runtime.probe(
            build,
            episode_uuid=episode_uuid,
            require_complete=True,
        ):
            raise RuntimeError("r4_graph_runtime_not_ready")
        serving_rows = await _serving_chunk_rows(
            dependencies,
            workspace_id=workspace_id,
            kb_id=arguments.knowledge_base_id,
            index_revision_id=arguments.index_revision_id,
        )
        locator_ids_by_case = {
            str(case["case_id"]): {
                "answer": _locator_chunk_ids(
                    case["answer_gold_source_locators"], serving_rows
                ),
                "path_context": _locator_chunk_ids(
                    case["path_context_locators"], serving_rows
                ),
            }
            for case in cases
        }
        if arguments.preflight_only:
            return {
                "status": "preflight_ok",
                "case_count": len(cases),
                "locator_count": sum(
                    len(value["answer"]) + len(value["path_context"])
                    for value in locator_ids_by_case.values()
                ),
                "knowledge_base_id": str(arguments.knowledge_base_id),
                "index_revision_id": str(arguments.index_revision_id),
                "graph_build_id": str(arguments.graph_build_id),
            }
        captures, capture_metrics, simple_ids_by_case = await _capture_queries(
            dependencies,
            chat_model_adapter=chat_model_adapter,
            cases=cases,
            kb_id=arguments.knowledge_base_id,
            index_revision_id=arguments.index_revision_id,
            model_configuration=model_configuration,
            replay_mode=arguments.replay_mode,
        )
        controller_mode = (
            ACTUAL_AUTO_CONTROLLER_MODE
            if arguments.replay_mode == "actual-auto"
            else FORCED_CONTROLLER_MODE
        )
        manifest_sha256 = manifest_digest(manifest)
        capture_artifact = build_replay_capture_artifact(
            manifest_sha256=manifest_sha256,
            knowledge_base_id=str(arguments.knowledge_base_id),
            index_revision_id=str(arguments.index_revision_id),
            graph_build_id=str(arguments.graph_build_id),
            chat_model_profile_revision_id=str(
                arguments.chat_model_profile_revision_id
            ),
            captures=captures,
            controller_mode=controller_mode,
            **chat_model_runtime,
        )
        capture_digest = write_replay_capture_artifact(
            arguments.capture_output,
            capture_artifact,
        )
        capture_by_case = {item.case_id: item for item in captures}
        records: list[dict[str, Any]] = []
        for case in cases:
            case_id = str(case["case_id"])
            capture = capture_by_case.get(case_id)
            simple_ids = simple_ids_by_case[case_id]
            answer_gold_ids = locator_ids_by_case[case_id]["answer"]
            path_context_ids = locator_ids_by_case[case_id]["path_context"]
            column_layers: dict[str, dict[str, tuple[str, ...]]] = {}
            layer_metrics: dict[str, dict[str, int]] = {}
            layers, metrics = await _column_layers(
                dependencies,
                graph_store,
                build=build,
                query=str(case["question"]),
                excluded_chunk_ids=simple_ids,
            )
            column_layers["capability"] = layers
            layer_metrics["capability"] = metrics
            if capture is not None:
                layers, metrics = await _column_layers(
                    dependencies,
                    graph_store,
                    build=build,
                    query=capture.query,
                    excluded_chunk_ids=simple_ids,
                )
                column_layers["agent_replay"] = layers
                layer_metrics["agent_replay"] = metrics
            else:
                column_layers["agent_replay"] = {
                    layer: () for layer in ("raw", "hydrated", "reranked", "packed")
                }
                layer_metrics["agent_replay"] = {
                    key: 0
                    for key in (
                        "raw_edge_count",
                        "raw_episode_count",
                        "raw_mapped_chunk_count",
                        "no_mapping_episode_count",
                        "hydrated_chunk_count",
                        "below_threshold_count",
                        "reranked_chunk_count",
                        "packed_chunk_count",
                        "budget_dropped_count",
                    )
                }
            alignments = {
                column: align_chunk_layers(
                    column=column,
                    answer_gold_chunk_ids=answer_gold_ids,
                    simple_chunk_ids=simple_ids,
                    layer_chunk_ids=column_layers[column],
                    path_context_chunk_ids=path_context_ids,
                )
                for column in ("capability", "agent_replay")
            }
            record = diagnostic_record(
                case_id=case_id,
                alignments=alignments,
                query_source="agent_replay",
                query_count=1 if capture is not None else 0,
            )
            record["agent_replay_status"] = (
                "requested" if capture is not None else "not_requested"
            )
            if capture is None:
                record["columns"]["agent_replay"]["first_loss_layer"] = None
            record["query_sha256"] = {
                "capability": hashlib.sha256(
                    str(case["question"]).encode("utf-8")
                ).hexdigest(),
                "agent_replay": (
                    hashlib.sha256(capture.query.encode("utf-8")).hexdigest()
                    if capture is not None
                    else None
                ),
            }
            record["layer_metrics"] = layer_metrics
            record["capture_metrics"] = capture_metrics[case_id]
            records.append(record)
        aggregate = _aggregate_graph_cases(records)
        decision = (
            "inconclusive_for_go"
            if aggregate["columns"]["agent_replay"]["distinct_benefit"] < 10
            else "eligible_for_r5_layer_review"
        )
        diagnostic = {
            "schema_version": R4_DIAGNOSTIC_SCHEMA_VERSION,
            "dataset_id": "routing-rag-v1",
            "scope": "expected_route_graph",
            "manifest_sha256": manifest_sha256,
            "capture_artifact_sha256": capture_digest,
            "runtime": {
                "knowledge_base_id": str(arguments.knowledge_base_id),
                "index_revision_id": str(arguments.index_revision_id),
                "graph_build_id": str(arguments.graph_build_id),
                "chat_model_profile_revision_id": str(
                    arguments.chat_model_profile_revision_id
                ),
                **chat_model_runtime,
                "graphiti_edge_limit": GRAPHITI_EDGE_LIMIT,
                "replay_mode": arguments.replay_mode,
                "forced_controller_mode": controller_mode,
            },
            "case_count": len(records),
            "records": records,
            "aggregate": aggregate,
            "decision": decision,
        }
        diagnostic_digest = _write_json_artifact(
            arguments.diagnostic_output,
            diagnostic,
        )
        return {
            "status": "completed",
            "case_count": len(records),
            "capture_artifact_sha256": capture_digest,
            "diagnostic_artifact_sha256": diagnostic_digest,
            "decision": decision,
            "agent_replay_distinct_benefit": aggregate["columns"][
                "agent_replay"
            ]["distinct_benefit"],
            "agent_replay_route_requested": aggregate["columns"][
                "agent_replay"
            ]["route_requested"],
            "capture_model_calls": sum(
                int(value["model_calls"]) for value in capture_metrics.values()
            ),
            "capture_total_tokens": sum(
                int(value["usage"].get("total_tokens", 0))
                for value in capture_metrics.values()
            ),
        }
    finally:
        await dependencies.close()


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.confirm != CONFIRM_EXTERNAL_CALLS:
        parser.error(f"--confirm must equal {CONFIRM_EXTERNAL_CALLS}")
    if arguments.capture_output.resolve() == arguments.diagnostic_output.resolve():
        parser.error("capture and diagnostic outputs must differ")
    if arguments.capture_output.exists() or arguments.diagnostic_output.exists():
        parser.error("output artifacts must not already exist")
    result = asyncio.run(_run(arguments))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
