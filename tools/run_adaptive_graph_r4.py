#!/usr/bin/env python3
"""Run the explicitly authorized, read-only adaptive Graph RAG R4 diagnostic.

The runner is deliberately narrow: it targets the default ``routing-rag-v2`` Graph
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

from sqlalchemy import text

from apps.worker.dependencies import build_worker_dependencies
from rag_kb.adapters.graph_store.postgres import PgGraphStore
from rag_kb.adapters.model_api.langchain_chat import LangChainChatModelAdapter
from rag_kb.adapters.model_secrets.local import LocalModelSecretStore
from rag_kb.answering.agent import NativeToolCallingAgent
from rag_kb.domain import (
    ChatExecutionContext,
    ChatRunLease,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatModelOperation,
    ErrorCode,
    ModelKind,
    ModelValidationStatus,
    RerankMode,
    RetrievalExecutionError,
)
from rag_kb.retrieval.profile import adaptive_graphiti_profile
from rag_kb.retrieval.service import (
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
    DEFAULT_MANIFEST,
    EVALUATOR_EDGE_LIMITS,
    ForcedGraphitiSupplementChatModelPort,
    GraphitiSupplementCapture,
    GraphitiSupplementCaptureComplete,
    aggregate_graph_routing_metrics,
    align_chunk_layers,
    build_replay_capture_artifact,
    diagnostic_record,
    evaluate_graph_extraction,
    load_cases,
    load_manifest,
    manifest_digest,
    validate_evaluation_readiness,
)
from tools.evaluation_runtime import (
    DEFAULT_RUNTIME_MANIFEST,
    EvaluationRuntimeError,
    load_evaluation_runtime,
)


CONFIRM_EXTERNAL_CALLS = "RUN_ROUTING_RAG_R4_EXTERNAL_CALLS"
R4_DIAGNOSTIC_SCHEMA_VERSION = "adaptive_graph_r4_diagnostic_v3"
R4_CHECKPOINT_SCHEMA_VERSION = "adaptive_graph_r4_checkpoint_v3"
GRAPHITI_EDGE_LIMIT = 8
FORCED_CONTROLLER_MODE = "single_tool_auto_fallback"
OVERRIDE_CONTROLLER_MODE = "single_tool_required_fallback"
ACTUAL_AUTO_CONTROLLER_MODE = "actual_auto"
EVALUATOR_CHAT_MODEL_OVERRIDE = "deepseek-v4-flash"
EVALUATOR_CHAT_MODEL_MAX_OUTPUT_TOKENS = 512
EVALUATOR_CHAT_MODEL_MAX_RETRIES = 0
_R4_CHECKPOINT_STAGES = frozenset(
    {"capture", "capability", "agent_replay", "failed"}
)
_R4_EVALUATION_PHASES = frozenset({"capture", "capability", "agent_replay"})
_R4_PIPELINE_PHASES = frozenset(item.value for item in ChatPipelinePhase)
_R4_CHECKPOINT_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "api_key",
        "body",
        "content",
        "document_name",
        "exception_message",
        "filename",
        "headers",
        "message",
        "messages",
        "provider_payload",
        "query",
        "question",
        "request",
        "response",
        "secret",
        "source_location",
        "source_text",
        "text",
        "url",
    }
)
_R4_DIAGNOSTIC_CHECKS = frozenset(
    {
        "active_lease",
        "adaptive_retrieval_snapshot",
        "adaptive_supplement_evidence",
        "agent_budget",
        "claimed_lease",
        "cross_modal_provider_required",
        "frozen_revision",
        "graph_completeness",
        "graph_edge_search",
        "graph_frozen_revision",
        "graph_local_reranker_not_configured",
        "graph_runtime_probe",
        "graph_runtime_probe_mapping",
        "graph_store",
        "model_output_limit",
        "model_snapshot",
        "retrieval_snapshot",
        "resolved_model",
        "step_contract",
        "task_deadline",
        "unified_provider_capabilities",
    }
)
_R4_RETRYABLE_ERROR_CODES = frozenset(
    {
        ErrorCode.CHAT_PROVIDER_UNAVAILABLE.value,
        ErrorCode.CHAT_PIPELINE_DEADLINE_EXCEEDED.value,
        ErrorCode.RETRIEVAL_DEADLINE_EXCEEDED.value,
        ErrorCode.GRAPH_PROVIDER_UNAVAILABLE.value,
    }
)
_R4_STABLE_ERROR_CODES = frozenset(item.value for item in ErrorCode)
_R4_RUNNER_ERROR_CODES = frozenset(
    {
        "r4_active_graph_build_changed",
        "r4_active_revision_changed",
        "r4_checkpoint_completion_order_invalid",
        "r4_checkpoint_identity_mismatch",
        "r4_checkpoint_missing_for_existing_output",
        "r4_checkpoint_permissions_invalid",
        "r4_existing_artifact_mismatch",
        "r4_existing_artifact_permissions_invalid",
        "r4_final_artifact_identity_mismatch",
        "r4_forced_replacement_cardinality",
        "r4_graph_case_count_changed",
        "r4_graph_relation_locator_not_unique",
        "r4_graph_runtime_not_ready",
        "r4_knowledge_base_not_found",
        "r4_chat_model_revision_not_found",
        "r4_chat_model_revision_not_ready",
        "r4_dataset_identity_changed",
        "r4_rerank_changed_chunk_set",
        "r4_simple_exclusion_capture_mismatch",
        "r4_supplement_capture_incomplete",
    }
)
_R4_FAILURE_CODES = _R4_RUNNER_ERROR_CODES | {
    "r4_interrupted",
    "r4_unexpected_failure",
    "r4_validation_failure",
}
_R4_MODEL_CALL_OPERATIONS = frozenset(item.value for item in ChatModelOperation)


class R4RunnerError(RuntimeError):
    """Content-safe error raised by the evaluator's own validation contract."""

    def __init__(self, code: str) -> None:
        if code not in _R4_RUNNER_ERROR_CODES:
            raise ValueError("R4 runner error code is invalid")
        super().__init__(code)
        self.code = code


def _controller_mode(*, replay_mode: str, model_override: str | None) -> str:
    """Choose the evaluator controller compatible with the selected model."""

    if replay_mode == "actual-auto":
        return ACTUAL_AUTO_CONTROLLER_MODE
    if model_override == EVALUATOR_CHAT_MODEL_OVERRIDE:
        return OVERRIDE_CONTROLLER_MODE
    return FORCED_CONTROLLER_MODE


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-runtime",
        type=Path,
        default=DEFAULT_RUNTIME_MANIFEST,
    )
    parser.add_argument("--capture-output", type=Path)
    parser.add_argument("--diagnostic-output", type=Path)
    parser.add_argument("--checkpoint-output", type=Path)
    parser.add_argument(
        "--replay-mode",
        choices=("forced", "actual-auto"),
        default="forced",
    )
    parser.add_argument(
        "--rerank-mode",
        choices=("none", "classic", "local_minilm_v1"),
        default=RerankMode.CLASSIC.value,
    )
    parser.add_argument(
        "--chat-model-override",
        choices=(EVALUATOR_CHAT_MODEL_OVERRIDE,),
        help="Evaluator-only model override using the selected profile's provider.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the frozen corpus without Docker, database, Graph, or Provider access",
    )
    parser.add_argument(
        "--edge-limit",
        type=int,
        choices=EVALUATOR_EDGE_LIMITS,
        default=GRAPHITI_EDGE_LIMIT,
        help="Evaluator-only Graphiti edge K; production retrieval remains unchanged.",
    )
    parser.add_argument("--confirm")
    return parser


def _write_json_artifact(path: Path, value: Mapping[str, Any]) -> str:
    return _write_json_atomic(path, value)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _load_jsonl_rows(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError("r4_gold_row_invalid")
        rows.append(value)
    if not rows:
        raise ValueError("r4_gold_file_empty")
    return tuple(rows)


def _assert_checkpoint_content_safe(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or key.lower() in _R4_CHECKPOINT_FORBIDDEN_KEYS:
                raise ValueError("r4_checkpoint_contains_forbidden_field")
            _assert_checkpoint_content_safe(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _assert_checkpoint_content_safe(item)
        return
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError("r4_checkpoint_contains_unsupported_value")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> str:
    """Atomically replace one owner-only checkpoint after a safety scan."""

    _assert_checkpoint_content_safe(value)
    payload = _canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            os.chmod(temporary, 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(payload).hexdigest()


def _write_or_verify_json_artifact(path: Path, value: Mapping[str, Any]) -> str:
    """Create an immutable final artifact or verify a prior crash wrote it."""

    payload = _canonical_json_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise R4RunnerError("r4_existing_artifact_mismatch")
        if path.stat().st_mode & 0o077:
            raise R4RunnerError("r4_existing_artifact_permissions_invalid")
        return digest
    return _write_json_artifact(path, value)


def _safe_usage(value: Mapping[str, Any]) -> dict[str, int]:
    return {
        key: candidate
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if isinstance(candidate := value.get(key), int)
        and not isinstance(candidate, bool)
        and candidate >= 0
    }


def _safe_diagnostic(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    safe: dict[str, Any] = {}
    check = value.get("check")
    if isinstance(check, str) and check in _R4_DIAGNOSTIC_CHECKS:
        safe["check"] = check
    http_status = value.get("http_status")
    if (
        isinstance(http_status, int)
        and not isinstance(http_status, bool)
        and 100 <= http_status <= 599
    ):
        safe["http_status"] = http_status
    return safe


def _safe_model_call_usage(error: ChatPipelineExecutionError) -> list[dict[str, Any]]:
    safe_calls: list[dict[str, Any]] = []
    for call in error.model_calls:
        operation = getattr(call.operation, "value", call.operation)
        if not isinstance(operation, str) or operation not in _R4_MODEL_CALL_OPERATIONS:
            continue
        safe_calls.append(
            {
                "operation": operation,
                "usage": _safe_usage(call.usage),
            }
        )
    return safe_calls


def _failure_record(
    error: BaseException,
    *,
    phase: str,
) -> dict[str, Any]:
    if isinstance(error, ChatPipelineExecutionError):
        error_code = error.code.value
        error_phase = error.phase.value
        diagnostic = _safe_diagnostic(error.diagnostic)
        model_call_usage = _safe_model_call_usage(error)
    elif isinstance(error, RetrievalExecutionError):
        error_code = error.code.value
        error_phase = phase
        diagnostic = _safe_diagnostic(error.diagnostic)
        model_call_usage = []
    elif isinstance(error, R4RunnerError):
        error_code = error.code
        error_phase = phase
        diagnostic = {}
        model_call_usage = []
    elif isinstance(error, ValueError):
        error_code = "r4_validation_failure"
        error_phase = phase
        diagnostic = {}
        model_call_usage = []
    elif isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)):
        error_code = "r4_interrupted"
        error_phase = phase
        diagnostic = {}
        model_call_usage = []
    else:
        error_code = "r4_unexpected_failure"
        error_phase = phase
        diagnostic = {}
        model_call_usage = []
    return {
        "error_code": error_code,
        "phase": error_phase,
        "evaluation_phase": phase,
        "retryable": error_code in _R4_RETRYABLE_ERROR_CODES,
        "diagnostic": diagnostic,
        "model_call_usage": model_call_usage,
    }


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
        raise R4RunnerError("r4_knowledge_base_not_found")
    if bundle is None:
        raise R4RunnerError("r4_chat_model_revision_not_found")
    if (
        bundle.profile.kind is not ModelKind.CHAT
        or not bundle.profile.enabled
        or not bundle.provider.enabled
        or bundle.current_revision.validation_status is not ModelValidationStatus.VALID
    ):
        raise R4RunnerError("r4_chat_model_revision_not_ready")
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
        raise R4RunnerError("r4_graph_relation_locator_not_unique")
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
    rerank_mode: RerankMode,
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
            rerank_mode=rerank_mode,
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
    rerank_mode: RerankMode,
    controller_mode: str,
) -> tuple[
    tuple[GraphitiSupplementCapture, ...],
    dict[str, Mapping[str, Any]],
    dict[str, tuple[str, ...]],
]:
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
                    rerank_mode=rerank_mode,
                )
            )
        except GraphitiSupplementCaptureComplete:
            pass
        except BaseException:
            capture_retriever.abandon_case()
            raise
        capture, simple_ids = capture_retriever.finish_observed_case()
        if replay_mode == "forced" and capture is None:
            raise R4RunnerError("r4_supplement_capture_incomplete")
        expected_replacements = 1 if replay_mode == "forced" else 0
        if forced_model.replacements != expected_replacements:
            raise R4RunnerError("r4_forced_replacement_cardinality")
        if capture is not None:
            if set(capture.excluded_index_chunk_ids) != set(simple_ids):
                raise R4RunnerError("r4_simple_exclusion_capture_mismatch")
            captures.append(capture)
        simple_ids_by_case[case_id] = simple_ids
        metrics[case_id] = {
            "forced_replacements": forced_model.replacements,
            "model_calls": forced_model.model_calls,
            "usage": _safe_usage(forced_model.usage),
            "route_requested": capture is not None,
            "route_reason_code": next(
                (
                    item
                    for item in reversed(forced_model.response_route_reason_codes)
                    if item is not None
                ),
                None,
            ),
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


async def _column_layers(
    dependencies,
    *,
    build,
    query: str,
    excluded_chunk_ids: tuple[str, ...],
    edge_limit: int,
    answer_gold_chunk_ids: Sequence[str] = (),
    rerank_mode: RerankMode,
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    candidate_set = await dependencies.retrieval_service._search_graphiti_candidates(
        build.workspace_id,
        build.knowledge_base_id,
        build=build,
        index_revision_id=build.index_revision_id,
        query=query,
        edge_limit=edge_limit,
        rerank_mode=rerank_mode,
    )
    packed_full, _ = _pack_graphiti_supplement_evidence(
        candidate_set,
        excluded_index_chunk_ids=frozenset(),
    )
    packed_incremental, new_incremental_ids = _pack_graphiti_supplement_evidence(
        candidate_set,
        excluded_index_chunk_ids=frozenset(UUID(item) for item in excluded_chunk_ids),
    )
    hydrated_ids = tuple(str(item) for item in candidate_set.hydrated_chunk_ids)
    reranked_ids = _ordered_reranked_chunk_ids(
        candidate_set.traversal,
        hydrated_ids,
    )
    if set(hydrated_ids) != set(reranked_ids):
        raise R4RunnerError("r4_rerank_changed_chunk_set")
    hydrated_positions = {item: index for index, item in enumerate(hydrated_ids)}
    rerank_reordered_chunk_count = sum(
        hydrated_positions.get(item) != index
        for index, item in enumerate(reranked_ids)
    )
    rerank_score_state = (
        "scored" if candidate_set.rerank_score_by_chunk_id else "not_applicable"
    )
    layers = {
        "raw": tuple(str(item) for item in candidate_set.raw_chunk_ids),
        "hydrated": hydrated_ids,
        "reranked": reranked_ids,
        "packed": tuple(str(item.index_chunk_id) for item in packed_full),
    }
    packed_ids = set(layers["packed"])
    excluded_ids = set(excluded_chunk_ids)
    path_chunk_ids = tuple(
        str(chunk_id)
        for path in candidate_set.traversal.paths
        for chunk_id in path.source_chunk_ids
    )
    gold_rerank_scores = {
        str(chunk_id): float(candidate_set.rerank_score_by_chunk_id[UUID(str(chunk_id))])
        for chunk_id in answer_gold_chunk_ids
        if UUID(str(chunk_id)) in candidate_set.rerank_score_by_chunk_id
    }
    metrics = {
        "requested_k": edge_limit,
        "raw_edge_uuids": list(candidate_set.raw_edge_uuids),
        "raw_edge_count": len(candidate_set.raw_edge_uuids),
        "raw_episode_count": len(candidate_set.raw_episode_ids),
        "raw_mapped_episode_count": len(candidate_set.raw_mapped_episode_ids),
        "raw_mapped_chunk_count": len(candidate_set.raw_chunk_ids),
        "no_mapping_episode_count": len(candidate_set.raw_episode_ids)
        - len(candidate_set.raw_mapped_episode_ids),
        "hydrated_chunk_count": len(candidate_set.hydrated_chunk_ids),
        "unique_chunk_count": len(set(path_chunk_ids)),
        "duplicate_chunk_path_count": len(path_chunk_ids) - len(set(path_chunk_ids)),
        "reranked_chunk_count": len(reranked_ids),
        "rerank_score_state": rerank_score_state,
        "rerank_reordered_chunk_count": rerank_reordered_chunk_count,
        "gold_rerank_scores": gold_rerank_scores,
        "top1_gold_rerank_score": max(gold_rerank_scores.values(), default=None),
        "packed_chunk_count": len(packed_full),
        "incremental_packed_chunk_ids": [
            str(item.index_chunk_id) for item in packed_incremental
        ],
        "incremental_new_chunk_ids": [str(item) for item in new_incremental_ids],
        "budget_dropped_count": sum(
            str(item.index_chunk_id) not in packed_ids
            and str(item.index_chunk_id) not in excluded_ids
            for item in candidate_set.traversal.chunks
        ),
        "route_result_code": "admitted" if packed_full else "no_new_evidence",
    }
    return layers, metrics


def _ordered_reranked_chunk_ids(
    traversal: Any,
    hydrated_ids: Sequence[str],
) -> tuple[str, ...]:
    """Reflect production path ordering while preserving the full hydrated set."""

    hydrated = tuple(str(item) for item in hydrated_ids)
    hydrated_set = set(hydrated)
    path_order = tuple(
        dict.fromkeys(
            str(chunk_id)
            for path in traversal.paths
            for chunk_id in path.source_chunk_ids
            if str(chunk_id) in hydrated_set
        )
    )
    return path_order + tuple(
        chunk_id for chunk_id in hydrated if chunk_id not in path_order
    )


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
                    "raw_mapped_episode_count",
                    "raw_mapped_chunk_count",
                    "no_mapping_episode_count",
                    "hydrated_chunk_count",
                    "reranked_chunk_count",
                    "rerank_reordered_chunk_count",
                    "packed_chunk_count",
                    "budget_dropped_count",
                )
            },
            "rerank_score_state_counts": dict(
                Counter(
                    item["layer_metrics"][column].get(
                        "rerank_score_state", "not_applicable"
                    )
                    for item in records
                )
            ),
        }
    return result


def _layer_diagnostic(
    metrics: Mapping[str, Any],
    *,
    route_reason_code: str | None,
    route_result_code: str,
) -> dict[str, Any]:
    return {
        "requested_k": int(metrics["requested_k"]),
        "raw_edge_uuids": list(metrics["raw_edge_uuids"]),
        "raw_episode_count": int(metrics["raw_episode_count"]),
        "unique_chunk_count": int(metrics["unique_chunk_count"]),
        "duplicate_chunk_path_count": int(metrics["duplicate_chunk_path_count"]),
        "gold_rerank_scores": dict(metrics["gold_rerank_scores"]),
        "top1_gold_rerank_score": metrics["top1_gold_rerank_score"],
        "rerank_score_state": metrics.get("rerank_score_state", "not_applicable"),
        "rerank_reordered_chunk_count": int(
            metrics.get("rerank_reordered_chunk_count", 0)
        ),
        "route_reason_code": route_reason_code,
        "route_result_code": route_result_code,
        "salvage_status": "not_attempted",
        "final_outcome": "not_run",
    }


def _checkpoint_identity(
    *,
    dataset_id: str,
    manifest_sha256: str,
    manifest_file_sha256: str,
    cases_sha256: str,
    fixture_sha256: str,
    graph_entities_sha256: str,
    graph_relations_sha256: str,
    arguments: argparse.Namespace,
    chat_model_runtime: Mapping[str, Any],
    controller_mode: str,
    rerank_mode: RerankMode,
    case_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": R4_CHECKPOINT_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "scope": "expected_route_graph",
        "manifest_sha256": manifest_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "cases_sha256": cases_sha256,
        "fixture_sha256": fixture_sha256,
        "graph_entities_sha256": graph_entities_sha256,
        "graph_relations_sha256": graph_relations_sha256,
        "runtime": {
            "knowledge_base_id": str(arguments.knowledge_base_id),
            "index_revision_id": str(arguments.index_revision_id),
            "graph_build_id": str(arguments.graph_build_id),
            "chat_model_profile_revision_id": str(
                arguments.chat_model_profile_revision_id
            ),
            **chat_model_runtime,
            "graphiti_edge_limit": arguments.edge_limit,
            "evaluator_edge_limit": arguments.edge_limit,
            "replay_mode": arguments.replay_mode,
            "rerank_mode": rerank_mode.value,
            "forced_controller_mode": controller_mode,
            "evaluation_owner": arguments.evaluation_owner,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "evaluator_sha256": hashlib.sha256(
                Path(__file__).with_name("evaluate_adaptive_graph_route.py").read_bytes()
            ).hexdigest(),
        },
        "case_ids": list(case_ids),
    }


def _new_checkpoint(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **identity,
        "status": "active",
        "completed_cases": [],
        "active_case": None,
        "final_artifacts": None,
    }


def _validate_redacted_capture(value: Any, *, case_id: str) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping) or set(value) != {
        "case_id",
        "excluded_index_chunk_ids",
    }:
        raise ValueError("r4_checkpoint_capture_invalid")
    if value.get("case_id") != case_id:
        raise ValueError("r4_checkpoint_capture_case_mismatch")
    excluded = value.get("excluded_index_chunk_ids")
    if not isinstance(excluded, list):
        raise ValueError("r4_checkpoint_capture_invalid")
    try:
        normalized = [str(UUID(str(item))) for item in excluded]
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("r4_checkpoint_capture_invalid") from error
    if normalized != excluded or len(normalized) != len(set(normalized)):
        raise ValueError("r4_checkpoint_capture_invalid")


def _validate_completed_case(value: Any, *, expected_case_id: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "case_id",
        "capture",
        "record",
    }:
        raise ValueError("r4_checkpoint_completed_case_invalid")
    if value.get("case_id") != expected_case_id:
        raise ValueError("r4_checkpoint_completed_case_order_invalid")
    _validate_redacted_capture(value.get("capture"), case_id=expected_case_id)
    record = value.get("record")
    if not isinstance(record, Mapping) or set(record) != {
        "case_id",
        "columns",
        "query_source",
        "query_count",
        "layer_diagnostics",
        "agent_replay_status",
        "layer_metrics",
        "capture_metrics",
    }:
        raise ValueError("r4_checkpoint_record_invalid")
    if record.get("case_id") != expected_case_id:
        raise ValueError("r4_checkpoint_record_case_mismatch")
    _assert_checkpoint_content_safe(value)


def _validate_checkpoint(
    value: Any,
    *,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        *identity.keys(),
        "status",
        "completed_cases",
        "active_case",
        "final_artifacts",
    }:
        raise ValueError("r4_checkpoint_schema_invalid")
    if any(value.get(key) != item for key, item in identity.items()):
        raise R4RunnerError("r4_checkpoint_identity_mismatch")
    if value.get("status") not in {"active", "completed"}:
        raise ValueError("r4_checkpoint_status_invalid")
    completed = value.get("completed_cases")
    case_ids = list(identity["case_ids"])
    if not isinstance(completed, list) or len(completed) > len(case_ids):
        raise ValueError("r4_checkpoint_completed_cases_invalid")
    for index, item in enumerate(completed):
        _validate_completed_case(item, expected_case_id=case_ids[index])
    active = value.get("active_case")
    if active is not None:
        if not isinstance(active, Mapping) or set(active) != {
            "case_id",
            "stage",
            "error_code",
            "phase",
            "evaluation_phase",
            "retryable",
            "diagnostic",
            "model_call_usage",
        }:
            raise ValueError("r4_checkpoint_active_case_invalid")
        if len(completed) >= len(case_ids) or active.get("case_id") != case_ids[
            len(completed)
        ]:
            raise ValueError("r4_checkpoint_active_case_order_invalid")
        if active.get("stage") not in _R4_CHECKPOINT_STAGES:
            raise ValueError("r4_checkpoint_stage_invalid")
        error_code = active.get("error_code")
        if error_code is not None and (
            not isinstance(error_code, str)
            or error_code not in _R4_FAILURE_CODES | _R4_STABLE_ERROR_CODES
        ):
            raise ValueError("r4_checkpoint_error_code_invalid")
        phase = active.get("phase")
        if phase not in _R4_EVALUATION_PHASES | _R4_PIPELINE_PHASES:
            raise ValueError("r4_checkpoint_phase_invalid")
        if active.get("evaluation_phase") not in _R4_EVALUATION_PHASES:
            raise ValueError("r4_checkpoint_evaluation_phase_invalid")
        if not isinstance(active.get("retryable"), bool):
            raise ValueError("r4_checkpoint_retryable_invalid")
        diagnostic = active.get("diagnostic")
        if not isinstance(diagnostic, Mapping) or set(diagnostic) - {
            "check",
            "http_status",
        }:
            raise ValueError("r4_checkpoint_diagnostic_invalid")
        if "check" in diagnostic and diagnostic["check"] not in _R4_DIAGNOSTIC_CHECKS:
            raise ValueError("r4_checkpoint_diagnostic_invalid")
        if "http_status" in diagnostic and (
            isinstance(diagnostic["http_status"], bool)
            or not isinstance(diagnostic["http_status"], int)
            or not 100 <= diagnostic["http_status"] <= 599
        ):
            raise ValueError("r4_checkpoint_diagnostic_invalid")
        usage = active.get("model_call_usage")
        if not isinstance(usage, list):
            raise ValueError("r4_checkpoint_usage_invalid")
        for item in usage:
            if not isinstance(item, Mapping) or set(item) != {"operation", "usage"}:
                raise ValueError("r4_checkpoint_usage_invalid")
            if not isinstance(item["operation"], str) or not isinstance(
                item["usage"], Mapping
            ):
                raise ValueError("r4_checkpoint_usage_invalid")
            if any(
                isinstance(number, bool)
                or not isinstance(number, int)
                or number < 0
                for number in item["usage"].values()
            ):
                raise ValueError("r4_checkpoint_usage_invalid")
    if value["status"] == "completed" and (
        active is not None
        or len(completed) != len(case_ids)
        or not isinstance(value.get("final_artifacts"), Mapping)
        or set(value["final_artifacts"]) != {"capture_sha256", "diagnostic_sha256"}
    ):
        raise ValueError("r4_checkpoint_completion_invalid")
    final_artifacts = value.get("final_artifacts")
    if value["status"] == "active" and final_artifacts is not None:
        raise ValueError("r4_checkpoint_completion_invalid")
    if final_artifacts is not None:
        if not isinstance(final_artifacts, Mapping) or set(final_artifacts) != {
            "capture_sha256",
            "diagnostic_sha256",
        } or any(
            not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{64}", item) is None
            for item in final_artifacts.values()
        ):
            raise ValueError("r4_checkpoint_final_artifacts_invalid")
    _assert_checkpoint_content_safe(value)
    return value


def _load_or_create_checkpoint(
    path: Path,
    *,
    identity: Mapping[str, Any],
    final_outputs: Sequence[Path],
) -> dict[str, Any]:
    if not path.exists():
        if any(item.exists() for item in final_outputs):
            raise R4RunnerError("r4_checkpoint_missing_for_existing_output")
        checkpoint = _new_checkpoint(identity)
        _write_json_atomic(path, checkpoint)
        return checkpoint
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise R4RunnerError("r4_checkpoint_permissions_invalid")
    try:
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("r4_checkpoint_too_large")
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("r4_checkpoint_unreadable") from error
    checkpoint = _validate_checkpoint(loaded, identity=identity)
    if checkpoint["status"] == "completed":
        expected_digests = checkpoint["final_artifacts"]
        for path_value, digest_key in zip(
            final_outputs,
            ("capture_sha256", "diagnostic_sha256"),
        ):
            if (
                not path_value.is_file()
                or hashlib.sha256(path_value.read_bytes()).hexdigest()
                != expected_digests[digest_key]
            ):
                raise R4RunnerError("r4_final_artifact_identity_mismatch")
    return checkpoint


def _checkpoint_stage(
    checkpoint: dict[str, Any],
    path: Path,
    *,
    case_id: str,
    stage: str,
    phase: str | None = None,
    failure: Mapping[str, Any] | None = None,
) -> str:
    if stage not in _R4_CHECKPOINT_STAGES:
        raise ValueError("r4_checkpoint_stage_invalid")
    completed = checkpoint["completed_cases"]
    if (
        len(completed) >= len(checkpoint["case_ids"])
        or case_id != checkpoint["case_ids"][len(completed)]
    ):
        raise R4RunnerError("r4_checkpoint_stage_order_invalid")
    checkpoint["status"] = "active"
    resolved_phase = phase or (stage if stage in _R4_EVALUATION_PHASES else "capture")
    if resolved_phase not in _R4_EVALUATION_PHASES:
        raise ValueError("r4_checkpoint_phase_invalid")
    details = dict(
        failure
        or {
            "error_code": None,
            "phase": resolved_phase,
            "evaluation_phase": resolved_phase,
            "retryable": False,
            "diagnostic": {},
            "model_call_usage": [],
        }
    )
    if set(details) != {
        "error_code",
        "phase",
        "evaluation_phase",
        "retryable",
        "diagnostic",
        "model_call_usage",
    }:
        raise ValueError("r4_checkpoint_failure_invalid")
    checkpoint["active_case"] = {
        "case_id": case_id,
        "stage": stage,
        **details,
    }
    return _write_json_atomic(path, checkpoint)


def _checkpoint_complete_case(
    checkpoint: dict[str, Any],
    path: Path,
    *,
    case_id: str,
    capture: GraphitiSupplementCapture | Mapping[str, Any] | None,
    record: Mapping[str, Any],
) -> str:
    completed = checkpoint["completed_cases"]
    expected_case_id = checkpoint["case_ids"][len(completed)]
    if case_id != expected_case_id:
        raise R4RunnerError("r4_checkpoint_completion_order_invalid")
    if capture is None:
        redacted_capture = None
    elif isinstance(capture, GraphitiSupplementCapture):
        redacted_capture = capture.as_dict()
    else:
        redacted_capture = dict(capture)
    value = {
        "case_id": case_id,
        "capture": redacted_capture,
        "record": dict(record),
    }
    _validate_completed_case(value, expected_case_id=case_id)
    previous_active = checkpoint["active_case"]
    completed.append(value)
    checkpoint["active_case"] = None
    try:
        return _write_json_atomic(path, checkpoint)
    except BaseException:
        completed.pop()
        checkpoint["active_case"] = previous_active
        raise


def _checkpoint_error_code(error: BaseException) -> str:
    return str(_failure_record(error, phase="capture")["error_code"])


async def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest()
    if manifest.get("dataset_id") != "routing-rag-v2":
        raise R4RunnerError("r4_dataset_identity_changed")
    cases = tuple(
        case
        for case in load_cases(Path(manifest["case_file"]))
        if case.get("expected_route", {}).get("route") == "graph"
    )
    if len(cases) != 20:
        raise R4RunnerError("r4_graph_case_count_changed")
    graph_gold_root = (
        Path(str(manifest["case_file"])).parent / "gold" / "graph-rag-v1"
    )
    graph_entities_path = graph_gold_root / "entities.jsonl"
    graph_relations_path = graph_gold_root / "relations.jsonl"
    graph_entity_rows = _load_jsonl_rows(graph_entities_path)
    graph_relation_rows = _load_jsonl_rows(graph_relations_path)
    focus_relation_ids = tuple(
        dict.fromkeys(
            str(relation_id)
            for case in cases
            for relation_id in case["source"]["gold_path"]
        )
    )
    rerank_mode = RerankMode(arguments.rerank_mode)
    dependencies = build_worker_dependencies(env_file=arguments.evaluation_env_file)
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
            raise R4RunnerError("r4_active_revision_changed")
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
            raise R4RunnerError("r4_active_graph_build_changed")
        episode_uuid = await graph_store.first_graphiti_episode_uuid(
            workspace_id,
            arguments.knowledge_base_id,
            build.build_id,
        )
        graph_runtime_ready = await dependencies.graphiti_runtime.probe(
            build,
            episode_uuid=episode_uuid,
            require_complete=True,
        )
        graph_extraction = evaluate_graph_extraction(
            await dependencies.graphiti_runtime.diagnostic_edges(build),
            entity_rows=graph_entity_rows,
            relation_rows=graph_relation_rows,
            focus_relation_ids=focus_relation_ids,
        )
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
                "status": (
                    "preflight_ok"
                    if graph_runtime_ready
                    else "preflight_graph_quality_failed"
                ),
                "case_count": len(cases),
                "locator_count": sum(
                    len(value["answer"]) + len(value["path_context"])
                    for value in locator_ids_by_case.values()
                ),
                "knowledge_base_id": str(arguments.knowledge_base_id),
                "index_revision_id": str(arguments.index_revision_id),
                "graph_build_id": str(arguments.graph_build_id),
                "graph_extraction": graph_extraction,
            }
        if not graph_runtime_ready:
            raise R4RunnerError("r4_graph_runtime_not_ready")
        controller_mode = _controller_mode(
            replay_mode=arguments.replay_mode,
            model_override=arguments.chat_model_override,
        )
        manifest_sha256 = manifest_digest(manifest)
        manifest_file_sha256 = hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest()
        cases_sha256 = hashlib.sha256(
            Path(str(manifest["case_file"])).read_bytes()
        ).hexdigest()
        fixture_sha256 = hashlib.sha256(
            Path(str(manifest["empirical_need"]["fixture_file"])).read_bytes()
        ).hexdigest()
        graph_entities_sha256 = hashlib.sha256(
            graph_entities_path.read_bytes()
        ).hexdigest()
        graph_relations_sha256 = hashlib.sha256(
            graph_relations_path.read_bytes()
        ).hexdigest()
        identity = _checkpoint_identity(
            dataset_id=str(manifest["dataset_id"]),
            manifest_sha256=manifest_sha256,
            manifest_file_sha256=manifest_file_sha256,
            cases_sha256=cases_sha256,
            fixture_sha256=fixture_sha256,
            graph_entities_sha256=graph_entities_sha256,
            graph_relations_sha256=graph_relations_sha256,
            arguments=arguments,
            chat_model_runtime=chat_model_runtime,
            controller_mode=controller_mode,
            rerank_mode=rerank_mode,
            case_ids=[str(case["case_id"]) for case in cases],
        )
        checkpoint = _load_or_create_checkpoint(
            arguments.checkpoint_output,
            identity=identity,
            final_outputs=(arguments.capture_output, arguments.diagnostic_output),
        )
        resumed_case_count = len(checkpoint["completed_cases"])
        print(
            json.dumps(
                {
                    "event": "r4_checkpoint_loaded",
                    "completed_case_count": resumed_case_count,
                    "remaining_case_count": len(cases) - resumed_case_count,
                    "status": checkpoint["status"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        for case in cases[resumed_case_count:]:
            case_id = str(case["case_id"])
            current_phase = "capture"
            try:
                _checkpoint_stage(
                    checkpoint,
                    arguments.checkpoint_output,
                    case_id=case_id,
                    stage="capture",
                    phase=current_phase,
                )
                case_captures, case_capture_metrics, case_simple_ids = (
                    await _capture_queries(
                        dependencies,
                        chat_model_adapter=chat_model_adapter,
                        cases=(case,),
                        kb_id=arguments.knowledge_base_id,
                        index_revision_id=arguments.index_revision_id,
                        model_configuration=model_configuration,
                        replay_mode=arguments.replay_mode,
                        rerank_mode=rerank_mode,
                        controller_mode=controller_mode,
                    )
                )
                capture = case_captures[0] if case_captures else None
                capture_metrics = case_capture_metrics[case_id]
                simple_ids = case_simple_ids[case_id]
                answer_gold_ids = locator_ids_by_case[case_id]["answer"]
                path_context_ids = locator_ids_by_case[case_id]["path_context"]
                column_layers: dict[str, dict[str, tuple[str, ...]]] = {}
                layer_metrics: dict[str, dict[str, Any]] = {}
                current_phase = "capability"
                _checkpoint_stage(
                    checkpoint,
                    arguments.checkpoint_output,
                    case_id=case_id,
                    stage="capability",
                    phase=current_phase,
                )
                layers, metrics = await _column_layers(
                    dependencies,
                    build=build,
                    query=str(case["question"]),
                    excluded_chunk_ids=simple_ids,
                    edge_limit=arguments.edge_limit,
                    answer_gold_chunk_ids=answer_gold_ids,
                    rerank_mode=rerank_mode,
                )
                column_layers["capability"] = layers
                layer_metrics["capability"] = metrics
                if capture is not None:
                    current_phase = "agent_replay"
                    _checkpoint_stage(
                        checkpoint,
                        arguments.checkpoint_output,
                        case_id=case_id,
                        stage="agent_replay",
                        phase=current_phase,
                    )
                    layers, metrics = await _column_layers(
                        dependencies,
                        build=build,
                        query=capture.query,
                        excluded_chunk_ids=simple_ids,
                        edge_limit=arguments.edge_limit,
                        answer_gold_chunk_ids=answer_gold_ids,
                        rerank_mode=rerank_mode,
                    )
                    column_layers["agent_replay"] = layers
                    layer_metrics["agent_replay"] = metrics
                else:
                    column_layers["agent_replay"] = {
                        layer: ()
                        for layer in ("raw", "hydrated", "reranked", "packed")
                    }
                    layer_metrics["agent_replay"] = {
                        key: 0
                        for key in (
                            "raw_edge_count",
                            "raw_episode_count",
                            "raw_mapped_episode_count",
                            "raw_mapped_chunk_count",
                            "no_mapping_episode_count",
                            "hydrated_chunk_count",
                            "reranked_chunk_count",
                            "rerank_reordered_chunk_count",
                            "packed_chunk_count",
                            "budget_dropped_count",
                        )
                    }
                    layer_metrics["agent_replay"].update(
                        {
                            "requested_k": arguments.edge_limit,
                            "raw_edge_uuids": [],
                            "gold_rerank_scores": {},
                            "top1_gold_rerank_score": None,
                            "rerank_score_state": "not_applicable",
                            "unique_chunk_count": 0,
                            "duplicate_chunk_path_count": 0,
                            "route_result_code": "not_requested",
                        }
                    )
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
                    layer_diagnostics={
                        "capability": _layer_diagnostic(
                            layer_metrics["capability"],
                            route_reason_code=None,
                            route_result_code="not_requested",
                        ),
                        "agent_replay": _layer_diagnostic(
                            layer_metrics["agent_replay"],
                            route_reason_code=(
                                capture_metrics["route_reason_code"]
                                if capture is not None
                                else None
                            ),
                            route_result_code=(
                                layer_metrics["agent_replay"]["route_result_code"]
                                if capture is not None
                                else "not_requested"
                            ),
                        ),
                    },
                )
                record["agent_replay_status"] = (
                    "requested" if capture is not None else "not_requested"
                )
                if capture is None:
                    record["columns"]["agent_replay"]["first_loss_layer"] = None
                record["layer_metrics"] = layer_metrics
                record["capture_metrics"] = capture_metrics
                checkpoint_digest = _checkpoint_complete_case(
                    checkpoint,
                    arguments.checkpoint_output,
                    case_id=case_id,
                    capture=capture,
                    record=record,
                )
                print(
                    json.dumps(
                        {
                            "event": "r4_case_checkpointed",
                            "case_id": case_id,
                            "completed_case_count": len(
                                checkpoint["completed_cases"]
                            ),
                            "checkpoint_sha256": checkpoint_digest,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except BaseException as error:
                if isinstance(error, asyncio.CancelledError):
                    raise
                failure = _failure_record(error, phase=current_phase)
                if (
                    checkpoint["completed_cases"]
                    and checkpoint["completed_cases"][-1]["case_id"] == case_id
                ):
                    raise
                _checkpoint_stage(
                    checkpoint,
                    arguments.checkpoint_output,
                    case_id=case_id,
                    stage="failed",
                    phase=current_phase,
                    failure=failure,
                )
                print(
                    json.dumps(
                        {
                            "event": "r4_case_failed",
                            "case_id": case_id,
                            "completed_case_count": len(
                                checkpoint["completed_cases"]
                            ),
                            "error_code": failure["error_code"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                raise

        completed_cases = checkpoint["completed_cases"]
        records = [dict(item["record"]) for item in completed_cases]
        captures = [
            dict(item["capture"])
            for item in completed_cases
            if item["capture"] is not None
        ]
        capture_artifact = build_replay_capture_artifact(
            dataset_id=str(manifest["dataset_id"]),
            rerank_mode=rerank_mode.value,
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
        capture_digest = _write_or_verify_json_artifact(
            arguments.capture_output,
            capture_artifact,
        )
        aggregate = _aggregate_graph_cases(records)
        route_observations = {
            str(record["case_id"]): {
                "graph_route_attempted": record.get("agent_replay_status") == "requested",
                "graph_route_admitted": record["layer_metrics"]["agent_replay"].get(
                    "route_result_code"
                ) == "admitted",
                "graph_new_evidence_count": int(
                    record["layer_metrics"]["agent_replay"].get(
                        "packed_chunk_count", 0
                    )
                ),
            }
            for record in records
        }
        graph_metrics = aggregate_graph_routing_metrics(
            cases,
            route_observations,
            alignments={
                str(record["case_id"]): record["columns"]["agent_replay"]
                for record in records
            },
        )
        aggregate["primary"] = graph_metrics
        decision = (
            "inconclusive_for_go"
            if graph_metrics["benefit_capture"].get("status") != "computed"
            or graph_metrics["benefit_capture"].get("benefit_capture", {}).get("value") is None
            or graph_metrics["benefit_capture"]["benefit_capture"]["value"] < 0.7
            else "eligible_for_r5_layer_review"
        )
        diagnostic = {
            "schema_version": R4_DIAGNOSTIC_SCHEMA_VERSION,
            "dataset_id": str(manifest["dataset_id"]),
            "scope": "expected_route_graph",
            "manifest_sha256": manifest_sha256,
            "manifest_file_sha256": manifest_file_sha256,
            "cases_sha256": cases_sha256,
            "fixture_sha256": fixture_sha256,
            "graph_entities_sha256": graph_entities_sha256,
            "graph_relations_sha256": graph_relations_sha256,
            "capture_artifact_sha256": capture_digest,
            "runtime": dict(identity["runtime"]),
            "case_count": len(records),
            "records": records,
            "aggregate": aggregate,
            "graph_extraction": graph_extraction,
            "decision": decision,
        }
        diagnostic_digest = _write_or_verify_json_artifact(
            arguments.diagnostic_output,
            diagnostic,
        )
        checkpoint["status"] = "completed"
        checkpoint["active_case"] = None
        checkpoint["final_artifacts"] = {
            "capture_sha256": capture_digest,
            "diagnostic_sha256": diagnostic_digest,
        }
        checkpoint_digest = _write_json_atomic(
            arguments.checkpoint_output,
            checkpoint,
        )
        return {
            "status": "completed",
            "case_count": len(records),
            "capture_artifact_sha256": capture_digest,
            "diagnostic_artifact_sha256": diagnostic_digest,
            "checkpoint_artifact_sha256": checkpoint_digest,
            "resumed_case_count": resumed_case_count,
            "decision": decision,
            "agent_replay_distinct_benefit": aggregate["columns"][
                "agent_replay"
            ]["distinct_benefit"],
            "agent_replay_route_requested": aggregate["columns"][
                "agent_replay"
            ]["route_requested"],
            "graph_needed_route_recall": graph_metrics["route"][
                "graph_needed_route_recall"
            ]["value"],
            "packed_required_path_recall": graph_metrics["graph_recall"][
                "by_layer"
            ].get("packed", {}).get("required_path_recall", {}).get("value"),
            "packed_answer_gold_recall": graph_metrics["graph_recall"][
                "by_layer"
            ].get("packed", {}).get("answer_gold_recall", {}).get("value"),
            "benefit_capture": graph_metrics["benefit_capture"].get(
                "benefit_capture", {}
            ).get("value"),
            "capture_model_calls": sum(
                int(record["capture_metrics"]["model_calls"]) for record in records
            ),
            "capture_total_tokens": sum(
                int(record["capture_metrics"]["usage"].get("total_tokens", 0))
                for record in records
            ),
        }
    finally:
        await dependencies.close()


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.dry_run:
        if arguments.confirm is not None or arguments.preflight_only:
            parser.error("--dry-run cannot be combined with execution options")
        manifest = load_manifest()
        readiness = validate_evaluation_readiness(DEFAULT_MANIFEST, manifest=manifest)
        print(
            json.dumps(
                {
                    "status": "offline_dry_run_ok",
                    "dataset_id": manifest["dataset_id"],
                    "case_count": manifest["case_count"],
                    "readiness": readiness,
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.confirm != CONFIRM_EXTERNAL_CALLS:
        parser.error(f"--confirm must equal {CONFIRM_EXTERNAL_CALLS}")
    if any(
        path is None
        for path in (
            arguments.capture_output,
            arguments.diagnostic_output,
            arguments.checkpoint_output,
        )
    ):
        parser.error("capture, diagnostic, and checkpoint outputs are required")
    try:
        evaluation_runtime = load_evaluation_runtime(
            arguments.evaluation_runtime,
            require_adaptive_graph=True,
        )
    except (EvaluationRuntimeError, OSError):
        parser.error("isolated evaluation runtime is unavailable")
    identity = evaluation_runtime.adaptive_graph
    assert identity is not None
    arguments.knowledge_base_id = identity.knowledge_base_id
    arguments.index_revision_id = identity.index_revision_id
    arguments.graph_build_id = identity.graph_build_id
    arguments.chat_model_profile_revision_id = identity.answer_profile_revision_id
    arguments.evaluation_env_file = evaluation_runtime.env_file
    arguments.evaluation_owner = evaluation_runtime.owner
    output_paths = {
        arguments.capture_output.resolve(),
        arguments.diagnostic_output.resolve(),
        arguments.checkpoint_output.resolve(),
    }
    if len(output_paths) != 3:
        parser.error("capture, diagnostic, and checkpoint outputs must differ")
    result = asyncio.run(_run(arguments))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
