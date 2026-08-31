#!/usr/bin/env python3
"""Run the bound, resumable large evaluation campaign.

The command is intentionally a host-native evaluator.  It consumes the
already-running isolated runtime, verifies the read-only readiness gate and a
completed provider smoke, then executes the frozen campaign in dependency
order.  The campaign state is private and is written after every case
transition; the final JSON artifact is written only after every planned case
is durably complete.

No prompt, answer, retrieved text, provider payload, or graph fact is written
to the checkpoint.  Gold identifiers and aggregate counters are retained so a
completed run can be audited without turning provider output into trusted
data.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any
from uuid import UUID

from apps.worker.dependencies import build_worker_dependencies
from rag_kb.adapters.graph_store.postgres import PgGraphStore
from rag_kb.answering.agent import AGENT_TRACE_ARTIFACT, NativeToolCallingAgent
from rag_kb.domain import (
    ChatModelExecutionError,
    ChatPipelineExecutionError,
    ModelKind,
    ModelValidationStatus,
    RerankMode,
    RetrievalExecutionError,
)
from rag_kb.retrieval.profile import adaptive_graphiti_profile, exact_profile
from rag_kb.retrieval.service import _pack_graph_search_evidence  # noqa: SLF001
from rag_kb.services.chat_execution import ChatEvidenceRetriever
from rag_kb.uow import TransactionMode, execute_in_transaction
from tools.check_large_evaluation_runtime import (
    LargeEvaluationRuntimeError,
    _load_private_plan,
    check as check_runtime,
)
from tools.evaluation_campaign_state import (
    SCHEMA_VERSION as CAMPAIGN_SCHEMA_VERSION,
    CampaignStateError,
    begin_case,
    completed_case_ids,
    complete_case,
    digest,
    load_or_create,
    phase_progress,
    schedule_case_retry,
    write_private_json,
)
from tools.evaluation_resilience import (
    HEARTBEAT_INTERVAL_SECONDS,
    PROVIDER_CASE_RETRY_POLICY,
    RESILIENCE_POLICY,
    RESILIENCE_POLICY_SHA256,
    RetryPolicy,
    is_retryable_provider_failure,
    safe_failure_summary,
    seconds_until,
    stable_error_code,
    timestamp_after,
)
from tools.evaluation_runtime import (
    DEFAULT_RUNTIME_MANIFEST,
    EvaluationRuntime,
    EvaluationRuntimeError,
    load_evaluation_runtime,
)
from tools.prepare_large_evaluation import ROOT, SCHEMA_VERSION
from tools.provision_large_evaluation_host import BINDINGS_SCHEMA, DATASET_SPECS, SPECS
from tools.run_adaptive_graph_r4 import (
    R4RunnerError,
    _execution_context,
    _load_runtime_facts,
    _serving_chunk_rows,
)
from tools.run_evaluation_provider_smoke import SCHEMA as PROVIDER_SMOKE_SCHEMA
from tools.run_evaluation_provider_smoke import EXPECTED_MODELS
from tools.run_open_source_rag_v3 import normalize_term


CONFIRM = "RUN_LARGE_EVALUATION_EXTERNAL_CALLS"
FINAL_SCHEMA = "large_evaluation_final_report_v1"
CAMPAIGN_ROOT = ROOT / ".runtime/evaluations/large-evaluation-v1"
DEFAULT_PLAN = CAMPAIGN_ROOT / "preflight.json"
DEFAULT_CAMPAIGN = CAMPAIGN_ROOT / "campaign-state.json"
DEFAULT_PROVIDER_SMOKE = CAMPAIGN_ROOT / "provider-smoke.json"
DEFAULT_LOCKED = CAMPAIGN_ROOT / "final-report.json"
DEFAULT_MARKDOWN = ROOT / "docs/test/35-0826-large-evaluation-report.md"

PUBLIC_ROOT = ROOT / "evaluation/public-rag-benchmark-suite-v1"
ROUTING_ROOT = ROOT / "evaluation/routing-rag-musique-expanded-v1"
ENTERPRISE_ROOT = ROOT / "evaluation/enterprise-profile-qualification-v1"
GRAPH_ROOT = ROOT / "evaluation/graph-rag-v1"

MIN_QUALIFIED_ROUTING_CASES = 30
ROUTING_TOP_K = 10
ROUTING_EDGE_LIMIT = 16
ROUTING_SOURCE_CHUNK_TARGET = 12
ROUTING_SOURCE_CHUNK_LIMIT = 16
AUTO_REPEATS = 3

_FORBIDDEN_PERSISTED_KEYS = frozenset(
    {
        "answer",
        "body",
        "content",
        "excerpt",
        "fact",
        "filename",
        "message",
        "messages",
        "provider_payload",
        "query",
        "question",
        "request",
        "response",
        "source_text",
        "text",
        "url",
    }
)


class LargeEvaluationError(RuntimeError):
    """Content-safe failure raised by the campaign runner."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class SuiteRuntime:
    dataset_id: str
    knowledge_base_id: UUID
    index_revision_id: UUID
    graph_build_id: UUID | None
    knowledge_base: Any
    model_configuration: Mapping[str, Any]
    build: Any | None
    serving_rows: tuple[Mapping[str, Any], ...]
    expected_filenames: frozenset[str]
    alias_map: Mapping[str, str]
    filename_by_chunk: Mapping[str, str]
    filename_by_document_id: Mapping[str, str]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise LargeEvaluationError("large_evaluation_gold_row_invalid")
        rows.append(value)
    if not rows:
        raise LargeEvaluationError("large_evaluation_gold_file_empty")
    return rows


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _norm(value: object) -> str:
    """Normalize only for an in-memory, conservative lexical score."""

    text = normalize_term(str(value or "")).casefold()
    return "".join(character for character in text if character.isalnum())


def _safe_number(value: object) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": round(numerator / denominator, 6) if denominator else None,
    }


def _f1(precision: float, recall: float) -> float:
    return round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0


def _percentile(values: Iterable[int], fraction: float) -> int | None:
    ordered = sorted(int(item) for item in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    return ordered[min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))]


def _assert_safe(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or key.lower() in _FORBIDDEN_PERSISTED_KEYS:
                raise LargeEvaluationError("large_evaluation_checkpoint_contains_content")
            _assert_safe(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_safe(item)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise LargeEvaluationError("large_evaluation_checkpoint_value_invalid")


def _write_state(path: Path, state: Mapping[str, Any]) -> str:
    _assert_safe(state)
    return write_private_json(path, dict(state))


@contextmanager
def _campaign_lock(path: Path):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        lock_path.chmod(0o600)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LargeEvaluationError("large_evaluation_already_running") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_bindings(
    runtime: EvaluationRuntime,
    *,
    bindings_path: Path | None = None,
    expected_dataset_ids: set[str] | None = None,
) -> tuple[dict[str, Any], str]:
    path = bindings_path or (runtime.runtime_root / "large-evaluation-bindings.json")
    if not path.is_absolute():
        path = runtime.runtime_root / path
    path = path.absolute()
    if path.parent != runtime.runtime_root.absolute():
        raise LargeEvaluationError("large_evaluation_bindings_path_invalid")
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise LargeEvaluationError("large_evaluation_bindings_unavailable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise LargeEvaluationError("large_evaluation_bindings_invalid") from error
    suites = value.get("suites") if isinstance(value, dict) else None
    expected = expected_dataset_ids or {spec.dataset_id for spec in SPECS.values()}
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != BINDINGS_SCHEMA
        or not isinstance(suites, dict)
        or not expected <= set(suites)
        or value.get("binding_sha256") != digest({"suites": suites})
    ):
        raise LargeEvaluationError("large_evaluation_bindings_invalid")
    return suites, str(value["binding_sha256"])


def _load_provider_smoke(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise LargeEvaluationError("large_evaluation_provider_smoke_unavailable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise LargeEvaluationError("large_evaluation_provider_smoke_invalid") from error
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != PROVIDER_SMOKE_SCHEMA
        or value.get("status") != "completed"
        or not isinstance(value.get("providers"), dict)
        or set(value["providers"]) != set(EXPECTED_MODELS)
        or any(value["providers"][name].get("status") != "completed" for name in EXPECTED_MODELS)
    ):
        raise LargeEvaluationError("large_evaluation_provider_smoke_not_completed")
    if any(value["providers"][name].get("model") != model for name, model in EXPECTED_MODELS.items()):
        raise LargeEvaluationError("large_evaluation_provider_smoke_model_mismatch")
    return value


def _load_corpus_files(*, routing_root: Path = ROUTING_ROOT) -> dict[str, Any]:
    return {
        "public": {
            "manifest": json.loads((PUBLIC_ROOT / "manifest.json").read_text(encoding="utf-8")),
            "documents": _jsonl(PUBLIC_ROOT / "documents.jsonl"),
            "cases": _jsonl(PUBLIC_ROOT / "cases.jsonl"),
        },
        "routing": {
            "manifest": json.loads((routing_root / "manifest.json").read_text(encoding="utf-8")),
            "documents": _jsonl(routing_root / "documents.jsonl"),
            "cases": _jsonl(routing_root / "cases.jsonl"),
        },
        "enterprise": {
            "manifest": json.loads((ENTERPRISE_ROOT / "manifest.json").read_text(encoding="utf-8")),
            "cases": _jsonl(ENTERPRISE_ROOT / "cases.jsonl"),
            "entities": _jsonl(ENTERPRISE_ROOT / "entities.jsonl"),
            "relations": _jsonl(ENTERPRISE_ROOT / "relations.jsonl"),
            "controls": _jsonl(ENTERPRISE_ROOT / "negative_controls.jsonl"),
        },
        "graph_rag": {
            "manifest": json.loads((GRAPH_ROOT / "manifest.json").read_text(encoding="utf-8")),
            "cases": _jsonl(GRAPH_ROOT / "cases.jsonl"),
            "entities": _jsonl(GRAPH_ROOT / "entities.jsonl"),
            "relations": _jsonl(GRAPH_ROOT / "relations.jsonl"),
        },
    }


def _documents_from_manifest(root: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = manifest.get("documents")
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    return _jsonl(root / "documents.jsonl")


async def _selected_embedding_profiles(dependencies) -> dict[str, str]:
    async def load(unit_of_work):
        selection = await unit_of_work.model_settings.get_selection()
        profiles = {
            "chat": await unit_of_work.model_settings.get_profile_revision(
                selection.chat_profile_revision_id
            ) if selection.chat_profile_revision_id is not None else None,
            "text_embedding": await unit_of_work.model_settings.get_profile_revision(
                selection.text_embedding_profile_revision_id
            ) if selection.text_embedding_profile_revision_id is not None else None,
            "multimodal_embedding": await unit_of_work.model_settings.get_profile_revision(
                selection.multimodal_embedding_profile_revision_id
            ) if selection.multimodal_embedding_profile_revision_id is not None else None,
        }
        return selection, profiles

    selection, profiles = await execute_in_transaction(
        dependencies.unit_of_work,
        load,
        mode=TransactionMode.REPEATABLE_READ_ONLY,
    )
    expected_kinds = {
        "chat": ModelKind.CHAT,
        "text_embedding": ModelKind.TEXT_EMBEDDING,
        "multimodal_embedding": ModelKind.MULTIMODAL_EMBEDDING,
    }
    expected_models = EXPECTED_MODELS
    result: dict[str, str] = {}
    for name, kind in expected_kinds.items():
        bundle = profiles.get(name)
        if (
            bundle is None
            or bundle.profile.kind is not kind
            or not bundle.profile.enabled
            or not bundle.provider.enabled
            or bundle.current_revision.validation_status is not ModelValidationStatus.VALID
            or bundle.current_revision.model != expected_models[name]
        ):
            raise LargeEvaluationError("large_evaluation_model_selection_invalid")
        result[f"{name}_profile_revision_id"] = str(bundle.current_revision.id)
    return result


async def _load_suite(
    dependencies,
    runtime: EvaluationRuntime,
    *,
    key: str,
    suite_binding: Mapping[str, Any],
    corpus: Mapping[str, Any],
    corpus_root: Path | None = None,
) -> SuiteRuntime:
    try:
        kb_id = UUID(str(suite_binding["knowledge_base_id"]))
        index_revision_id = UUID(str(suite_binding["index_revision_id"]))
        graph_build_id = (
            UUID(str(suite_binding["graph_build_id"]))
            if suite_binding.get("graph_build_id")
            else None
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LargeEvaluationError("large_evaluation_suite_identity_invalid") from error
    workspace_id = dependencies.settings.identity.workspace_id
    graph_store = PgGraphStore(dependencies.database.sessions)
    build = None
    profile_id = runtime.adaptive_graph.answer_profile_revision_id if runtime.adaptive_graph else None
    if graph_build_id is not None:
        build = await graph_store.get_active_graphiti_build(workspace_id, kb_id)
        if (
            build is None
            or build.build_id != graph_build_id
            or build.index_revision_id != index_revision_id
            or build.schema_profile_key != suite_binding.get("graph_schema_key")
            or build.schema_profile_digest != suite_binding.get("graph_schema_digest")
            or build.extractor_version != "graphiti_v4"
        ):
            raise LargeEvaluationError("large_evaluation_graph_build_identity_changed")
        profile_id = build.chat_profile_revision_id
        episode_uuid = await graph_store.first_graphiti_episode_uuid(
            workspace_id, kb_id, build.build_id
        )
        if not await dependencies.graphiti_runtime.probe(
            build, episode_uuid=episode_uuid, require_complete=True
        ):
            raise LargeEvaluationError("large_evaluation_graph_runtime_not_ready")
    if profile_id is None:
        raise LargeEvaluationError("large_evaluation_chat_profile_missing")
    knowledge_base, bundle, model_configuration = await _load_runtime_facts(
        dependencies, kb_id=kb_id, model_revision_id=profile_id
    )
    if model_configuration.get("resolved_model") != EXPECTED_MODELS["chat"]:
        raise LargeEvaluationError("large_evaluation_chat_model_mismatch")
    if knowledge_base.active_index_revision_id != index_revision_id:
        raise LargeEvaluationError("large_evaluation_index_identity_changed")
    rows = await _serving_chunk_rows(
        dependencies,
        workspace_id=workspace_id,
        kb_id=kb_id,
        index_revision_id=index_revision_id,
    )
    manifest = corpus["manifest"]
    if key == "enterprise":
        document_rows = [
            {
                "document_id": row.get("relation_id") or row.get("control_id"),
                "filename": row.get("document_filename"),
            }
            for row in (*corpus.get("relations", ()), *corpus.get("controls", ()))
        ]
    else:
        document_rows = _documents_from_manifest(
            PUBLIC_ROOT
            if key == "public"
            else corpus_root
            if key == "routing" and corpus_root is not None
            else ROUTING_ROOT
            if key == "routing"
            else GRAPH_ROOT,
            manifest,
        )
    expected_by_id = {
        str(row.get("document_id")): str(row.get("filename"))
        for row in document_rows
        if row.get("document_id") and row.get("filename")
    }
    expected_filenames = frozenset(expected_by_id.values())
    # Bindings are keyed by dataset id; use that identity rather than a caller
    # supplied alias when reading duplicate-content aliases.
    dataset_id = str(corpus["manifest"]["dataset_id"])
    provisioning_path = runtime.runtime_root / "large-evaluation-provisioning" / f"{dataset_id}.json"
    alias_map: dict[str, str] = {}
    if provisioning_path.is_file():
        checkpoint = json.loads(provisioning_path.read_text(encoding="utf-8"))
        raw_aliases = checkpoint.get("duplicate_content_aliases", {})
        if isinstance(raw_aliases, dict):
            alias_map = {str(name): str(target) for name, target in raw_aliases.items()}
        raw_parser_aliases = checkpoint.get("parser_content_aliases", {})
        if isinstance(raw_parser_aliases, dict):
            parser_aliases = {
                str(name): str(target) for name, target in raw_parser_aliases.items()
            }
            if set(alias_map).intersection(parser_aliases):
                raise LargeEvaluationError("large_evaluation_document_alias_overlap")
            alias_map.update(parser_aliases)
    actual_filenames = frozenset(str(row.get("original_filename")) for row in rows)
    # A duplicate-content alias points at an existing frozen filename, while
    # a parser-content alias may point at a new current-version filename (the
    # same bytes with a parser-compatible suffix).  Both are physical serving
    # names and must participate in the set binding.
    expected_physical = (
        expected_filenames - frozenset(alias_map)
    ) | frozenset(alias_map.values())
    if actual_filenames != expected_physical:
        raise LargeEvaluationError("large_evaluation_serving_document_set_changed")
    if any(target not in actual_filenames for target in alias_map.values()):
        raise LargeEvaluationError("large_evaluation_document_alias_invalid")
    filename_by_chunk = {
        str(row["index_chunk_id"]): next(
            (
                logical
                for logical, physical in alias_map.items()
                if physical == str(row["original_filename"])
            ),
            str(row["original_filename"]),
        )
        for row in rows
        if row.get("index_chunk_id") and row.get("original_filename")
    }
    return SuiteRuntime(
        dataset_id=dataset_id,
        knowledge_base_id=kb_id,
        index_revision_id=index_revision_id,
        graph_build_id=graph_build_id,
        knowledge_base=knowledge_base,
        model_configuration=model_configuration,
        build=build,
        serving_rows=tuple(rows),
        expected_filenames=expected_filenames,
        alias_map=alias_map,
        filename_by_chunk=filename_by_chunk,
        filename_by_document_id=expected_by_id,
    )


def _trace_usage(state: Any, trace: Any) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    answering = state.answering
    for call in answering.model_calls if answering is not None else ():
        usage = getattr(call, "usage", {})
        for key in totals:
            totals[key] += _safe_number(usage.get(key))
    if totals["total_tokens"] == 0:
        totals["total_tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]
    trace_data = trace.as_dict() if hasattr(trace, "as_dict") else {}
    trace_usage = trace_data.get("usage", {}) if isinstance(trace_data, Mapping) else {}
    totals.update(
        {
            "model_rounds": _safe_number(trace_usage.get("model_rounds")),
            "retrieval_calls": _safe_number(trace_usage.get("retrieval_calls")),
            "calculation_calls": _safe_number(trace_usage.get("calculation_calls")),
            "evidence_refs": _safe_number(trace_usage.get("evidence_refs")),
            "model_call_count": len(answering.model_calls) if answering is not None else 0,
        }
    )
    return totals


def _route_summary(trace: Any) -> dict[str, Any]:
    from tools.evaluate_adaptive_graph_route import summarize_graph_route_trace

    try:
        value = summarize_graph_route_trace(trace.as_dict())
    except (KeyError, TypeError, ValueError):
        value = {}
    return {
        "graph_route_attempted": bool(value.get("graph_route_attempted", False)),
        "graph_route_admitted": bool(value.get("graph_route_admitted", False)),
        "graph_new_evidence_count": _safe_number(value.get("graph_new_evidence_count")),
    }


async def _run_agent(
    agent: NativeToolCallingAgent,
    dependencies,
    suite: SuiteRuntime,
    *,
    question: str,
    lane: str,
    score: Callable[[str, Any, Any], Mapping[str, Any]],
) -> dict[str, Any]:
    retrieval_strategy = (
        exact_profile(top_k=10, rerank_mode=RerankMode.CLASSIC).as_dict()
        if lane == "simple"
        else adaptive_graphiti_profile(top_k=10, rerank_mode=RerankMode.CLASSIC).as_dict()
    )
    context = _execution_context(
        settings=dependencies.settings,
        kb_id=suite.knowledge_base_id,
        index_revision_id=suite.index_revision_id,
        question=question,
        model_configuration=suite.model_configuration,
        rerank_mode=RerankMode.CLASSIC,
    )
    context = replace(context, retrieval_strategy=retrieval_strategy)
    deadline_seconds = float(
        dependencies.settings.job_poller.chat_deadline_seconds
    )
    started = time.monotonic()
    state = await asyncio.wait_for(
        agent.run(context, deadline_seconds=deadline_seconds),
        timeout=deadline_seconds + 120.0,
    )
    duration_ms = int((time.monotonic() - started) * 1000)
    trace = state.artifacts.get(AGENT_TRACE_ARTIFACT)
    answering = state.answering
    if trace is None or answering is None or answering.validated is None or answering.rendered is None:
        raise LargeEvaluationError("large_evaluation_agent_result_invalid")
    route = _route_summary(trace)
    graph_events = [
        event for event in trace.events if getattr(event, "retrieval_lane", None) == "graph_relations"
    ]
    usage = _trace_usage(state, trace)
    observation: dict[str, Any] = {
        "lane": lane,
        "actual_outcome": answering.validated.outcome.value,
        "citation_count": len(answering.rendered.citations),
        "evidence_count": len(answering.evidence.items),
        "missing_aspect_count": len(answering.validated.missing_aspects),
        "graph_route_attempted": route["graph_route_attempted"],
        "graph_route_admitted": route["graph_route_admitted"],
        "graph_new_evidence_count": route["graph_new_evidence_count"],
        "graph_call_count": sum(1 for event in graph_events if getattr(event, "status", None) == "ok"),
        "graph_timeout_count": sum(
            1 for event in graph_events if getattr(event, "route_result_code", None) == "timeout"
        ),
        "model_call_usage": usage,
        "total_tokens": usage["total_tokens"],
        "budget_wrap_up": any(
            getattr(event, "budget_wrap_up", False) for event in trace.events
        ),
        "duration_ms": duration_ms,
    }
    observation.update(score(answering.rendered.content, answering.rendered.citations, answering))
    _assert_safe(observation)
    return observation


def _gold_text_values(value: object) -> list[str]:
    values: list[str] = []
    if isinstance(value, str):
        if value.strip():
            values.append(value)
    elif isinstance(value, Mapping):
        if isinstance(value.get("text"), str):
            values.append(str(value["text"]))
        else:
            for item in value.values():
                values.extend(_gold_text_values(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            values.extend(_gold_text_values(item))
    return values


def _lexical_match(content: str, case: Mapping[str, Any]) -> tuple[bool, bool]:
    candidates: list[str] = []
    direct = case.get("gold_answer")
    if isinstance(direct, str) and direct.strip():
        candidates.append(direct)
    candidates.extend(
        item for item in _gold_text_values(case.get("gold_answer_aliases")) if item.strip()
    )
    candidates.extend(
        item for item in _gold_text_values(case.get("gold_answer_facts")) if item.strip()
    )
    candidates = [item for item in candidates if len(_norm(item)) >= 3]
    if not candidates:
        return False, False
    normalized = _norm(content)
    return any(_norm(item) in normalized for item in candidates), True


def _contains_marker(content: str, markers: Sequence[str]) -> bool:
    normalized = _norm(content)
    return any(
        (marker == "?" and "?" in content)
        or (bool(_norm(marker)) and _norm(marker) in normalized)
        for marker in markers
    )


def _complete_conflict_structure(conflict: Any) -> bool:
    if conflict is None:
        return False
    supporting = tuple(getattr(conflict, "supporting_citation_ids", ()) or ())
    conflicting = tuple(getattr(conflict, "conflicting_citation_ids", ()) or ())
    conflict_type = getattr(conflict, "conflict_type", None)
    adjudication = getattr(conflict, "adjudication", None)
    if not supporting or not conflicting:
        return False
    if set(supporting) & set(conflicting):
        return False
    if conflict_type in {None, ""} or adjudication in {None, ""}:
        return False
    return True


def _has_complete_conflict_claim(answering: Any) -> bool:
    validated = getattr(answering, "validated", None)
    claims = getattr(validated, "claims", ()) or ()
    return any(
        _complete_conflict_structure(getattr(claim, "conflict", None)) for claim in claims
    )


def _forbidden_hit(content: str, claims: object) -> bool:
    if not isinstance(claims, (list, tuple)):
        return False
    normalized = _norm(content)
    return any(isinstance(item, str) and _norm(item) and _norm(item) in normalized for item in claims)


def _score_public(case: Mapping[str, Any]) -> Callable[[str, Any, Any], Mapping[str, Any]]:
    action = str(case.get("expected_action", ""))

    def score(content: str, citations: Any, answering: Any) -> Mapping[str, Any]:
        actual = answering.validated.outcome.value
        answer_like = {"answered", "partial"}
        if action in {"refuse_insufficient_evidence", "refuse_closed_world_absent", "decline_or_correct_false_premise"}:
            policy_correct = actual == "refused"
        elif action == "request_clarification":
            policy_correct = actual in answer_like and _contains_marker(
                content, ("?", "which one", "which actor", "referring", "specify", "clarify")
            )
        elif action == "surface_evidence_conflict":
            policy_correct = actual in answer_like and _has_complete_conflict_claim(
                answering
            )
        elif action == "answer_without_false_conflict":
            policy_correct = actual in answer_like and not _has_complete_conflict_claim(
                answering
            )
        else:
            policy_correct = actual in answer_like
        matched, available = _lexical_match(content, case)
        return {
            "expected_action": action,
            "stratum": str(case.get("stratum", "unknown")),
            "policy_correct": bool(policy_correct),
            "lexical_answer_match": bool(matched),
            "lexical_answer_match_available": bool(available),
            "conflict_marker_present": _contains_marker(
                content, ("conflict", "contradict", "outdated", "different", "disagree")
            ),
            "clarification_marker_present": _contains_marker(
                content, ("?", "clarify", "specify", "referring")
            ),
            "forbidden_claim_hit": _forbidden_hit(content, case.get("forbidden_claims")),
        }

    return score


def _score_routing(case: Mapping[str, Any]) -> Callable[[str, Any, Any], Mapping[str, Any]]:
    expected = str(case.get("expected_outcome", "answered"))

    def score(content: str, citations: Any, answering: Any) -> Mapping[str, Any]:
        actual = answering.validated.outcome.value
        matched, available = _lexical_match(
            content,
            {"gold_answer": case.get("expected_answer"), "gold_answer_aliases": case.get("answer_aliases")},
        )
        return {
            "expected_outcome": expected,
            "policy_correct": actual in {"answered", "partial"} if expected == "answered" else actual == expected,
            "lexical_answer_match": bool(matched),
            "lexical_answer_match_available": bool(available),
            "forbidden_claim_hit": False,
        }

    return score


def _score_enterprise(case: Mapping[str, Any], expected_filename: str) -> Callable[[str, Any, Any], Mapping[str, Any]]:
    def score(content: str, citations: Any, answering: Any) -> Mapping[str, Any]:
        matched, available = _lexical_match(content, {"gold_answer": case.get("expected_answer")})
        cited = any(
            getattr(item, "document_original_filename", None) == expected_filename
            for item in citations
        )
        return {
            "policy_correct": answering.validated.outcome.value == "answered" and matched and cited,
            "lexical_answer_match": bool(matched),
            "lexical_answer_match_available": bool(available),
            "expected_citation_hit": bool(cited),
            "forbidden_claim_hit": False,
        }

    return score


def _score_graph(case: Mapping[str, Any], expected_filename: str) -> Callable[[str, Any, Any], Mapping[str, Any]]:
    query_terms = tuple(str(item) for item in case.get("query_only_terms", ()) if isinstance(item, str))
    answer_terms = tuple(str(item) for item in case.get("answer_only_terms", ()) if isinstance(item, str))

    def score(content: str, citations: Any, answering: Any) -> Mapping[str, Any]:
        matched, available = _lexical_match(
            content,
            {"gold_answer": case.get("expected_answer"), "gold_answer_aliases": answer_terms},
        )
        citation_hit = any(
            getattr(item, "document_original_filename", None) == expected_filename
            for item in citations
        )
        citation_query_leak = any(
            any(_norm(term) in _norm(getattr(item, "quoted_text", "")) for term in query_terms)
            for item in citations
        )
        answer_query_leak = any(_norm(term) in _norm(content) for term in query_terms)
        return {
            "policy_correct": answering.validated.outcome.value in {"answered", "partial"} and matched and citation_hit,
            "lexical_answer_match": bool(matched),
            "lexical_answer_match_available": bool(available),
            "expected_citation_hit": bool(citation_hit),
            "citation_query_term_leakage": bool(citation_query_leak),
            "answer_query_term_leakage": bool(answer_query_leak),
            "forbidden_claim_hit": False,
        }

    return score


async def _run_cases(
    state_path: Path,
    state: dict[str, Any],
    *,
    phase: str,
    cases: Sequence[tuple[str, Callable[[], Any]]],
    retry_policy: RetryPolicy = PROVIDER_CASE_RETRY_POLICY,
    heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
) -> None:
    state["active_phase"] = phase
    state["planned_case_count"] = len(cases)
    _write_state(state_path, state)
    for case_id, producer in cases:
        phase_records = state.get("phases", {}).get(phase, {})
        prior = phase_records.get(case_id) if isinstance(phase_records, Mapping) else None
        if isinstance(prior, Mapping) and prior.get("status") == "retry_wait":
            await _wait_for_retry_window(
                state_path,
                state,
                phase=phase,
                case_id=case_id,
                heartbeat_interval_seconds=heartbeat_interval_seconds,
            )
        while True:
            if not begin_case(state_path, state, phase=phase, case_id=case_id):
                break
            try:
                observation = await _run_case_with_heartbeat(
                    state_path,
                    state,
                    phase=phase,
                    case_id=case_id,
                    producer=producer,
                    heartbeat_interval_seconds=heartbeat_interval_seconds,
                )
            except Exception as error:
                record = state["phases"][phase][case_id]
                attempt_count = int(record.get("attempt_count", 0))
                if (
                    not is_retryable_provider_failure(error)
                    or attempt_count >= retry_policy.max_attempts
                ):
                    raise
                delay = retry_policy.delay_after(attempt_count)
                failure = safe_failure_summary(error)
                next_retry_at = timestamp_after(delay)
                schedule_case_retry(
                    state_path,
                    state,
                    phase=phase,
                    case_id=case_id,
                    failure=failure,
                    next_retry_at=next_retry_at,
                )
                print(
                    json.dumps(
                        {
                            "event": "large_evaluation_case_retry_scheduled",
                            "phase": phase,
                            "case_id": case_id,
                            "attempt": attempt_count,
                            "failure_code": failure["code"],
                            "retry_delay_seconds": delay,
                            "next_retry_at": next_retry_at,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                await _wait_for_retry_window(
                    state_path,
                    state,
                    phase=phase,
                    case_id=case_id,
                    heartbeat_interval_seconds=heartbeat_interval_seconds,
                )
                continue
            break
        if not isinstance(phase_records, Mapping):
            phase_records = state.get("phases", {}).get(phase, {})
        current = phase_records.get(case_id) if isinstance(phase_records, Mapping) else None
        if isinstance(current, Mapping) and current.get("status") == "completed":
            continue
        if isinstance(observation, dict):
            observation.setdefault(
                "case_id",
                case_id.rsplit("::", 2)[0]
                if phase == "routing_agent_answers" and "::" in case_id
                else case_id,
            )
            observation.setdefault("observation_id", case_id)
        state.pop("active_case", None)
        state["runner_heartbeat_at"] = datetime.now(UTC).isoformat()
        complete_case(state_path, state, phase=phase, case_id=case_id, observation=observation)
        progress = phase_progress(state, phase=phase)
        print(
            json.dumps(
                {
                    "event": "large_evaluation_case_completed",
                    "phase": phase,
                    "case_id": case_id,
                    "completed": progress["completed"],
                    "planned": len(cases),
                    "metrics": {
                        key: observation[key]
                        for key in ("actual_outcome", "policy_correct", "qualified_graph_needed", "total_tokens")
                        if key in observation
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    state.setdefault("phase_summaries", {})[phase] = {
        "status": "completed",
        "planned": len(cases),
        "retry_count": sum(
            int(record.get("retry_count", 0))
            for record in state.get("phases", {}).get(phase, {}).values()
            if isinstance(record, Mapping)
        ),
        **phase_progress(state, phase=phase),
    }
    state.pop("active_case", None)
    _write_state(state_path, state)


async def _run_case_with_heartbeat(
    state_path: Path,
    state: dict[str, Any],
    *,
    phase: str,
    case_id: str,
    producer: Callable[[], Any],
    heartbeat_interval_seconds: float,
) -> Any:
    if heartbeat_interval_seconds <= 0:
        raise ValueError("evaluation heartbeat interval must be positive")
    task = asyncio.create_task(producer())
    started = time.monotonic()
    try:
        while True:
            done, _pending = await asyncio.wait(
                {task}, timeout=heartbeat_interval_seconds
            )
            if done:
                return await task
            now = datetime.now(UTC).isoformat()
            record = state["phases"][phase][case_id]
            record["heartbeat_at"] = now
            state["runner_heartbeat_at"] = now
            state["active_case"] = {
                "phase": phase,
                "case_id": case_id,
                "attempt_count": int(record.get("attempt_count", 0)),
                "elapsed_seconds": int(time.monotonic() - started),
            }
            _write_state(state_path, state)
            print(
                json.dumps(
                    {
                        "event": "large_evaluation_case_heartbeat",
                        **state["active_case"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def _wait_for_retry_window(
    state_path: Path,
    state: dict[str, Any],
    *,
    phase: str,
    case_id: str,
    heartbeat_interval_seconds: float,
) -> None:
    if heartbeat_interval_seconds <= 0:
        raise ValueError("evaluation heartbeat interval must be positive")
    while True:
        record = state["phases"][phase][case_id]
        remaining = seconds_until(record.get("next_retry_at"))
        if remaining <= 0:
            return
        await asyncio.sleep(min(heartbeat_interval_seconds, remaining))
        now = datetime.now(UTC).isoformat()
        record["heartbeat_at"] = now
        state["runner_heartbeat_at"] = now
        state["active_case"] = {
            "phase": phase,
            "case_id": case_id,
            "attempt_count": int(record.get("attempt_count", 0)),
            "retry_wait_seconds": int(seconds_until(record.get("next_retry_at"))),
        }
        _write_state(state_path, state)


async def _run_routing_qualification(
    state_path: Path,
    state: dict[str, Any],
    dependencies,
    suite: SuiteRuntime,
    cases: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> int:
    retriever = ChatEvidenceRetriever(dependencies.retrieval_service)
    filenames = {
        str(row["document_id"]): str(row["filename"])
        for row in manifest.get("documents", [])
        if isinstance(row, Mapping)
    }
    graph_cases = [case for case in cases if case.get("route_label") == "graph_needed_candidate"]
    producers: list[tuple[str, Callable[[], Any]]] = []
    for case in graph_cases:
        case_id = str(case["case_id"])

        async def produce(case=case):
            simple_context = replace(
                _execution_context(
                    settings=dependencies.settings,
                    kb_id=suite.knowledge_base_id,
                    index_revision_id=suite.index_revision_id,
                    question=str(case["question"]),
                    model_configuration=suite.model_configuration,
                    rerank_mode=RerankMode.CLASSIC,
                ),
                retrieval_strategy=exact_profile(
                    top_k=ROUTING_TOP_K, rerank_mode=RerankMode.CLASSIC
                ).as_dict(),
            )
            simple_pack = await retriever.retrieve_query(simple_context, str(case["question"]))
            simple_ids = {item.index_chunk_id for item in simple_pack.evidence}
            candidate_set = await dependencies.retrieval_service._search_graphiti_candidates(  # noqa: SLF001
                dependencies.settings.identity.workspace_id,
                suite.knowledge_base_id,
                build=suite.build,
                index_revision_id=suite.index_revision_id,
                query=str(case["question"]),
                edge_limit=ROUTING_EDGE_LIMIT,
                rerank_mode=RerankMode.CLASSIC,
            )
            packed, new_ids = _pack_graph_search_evidence(
                candidate_set,
                excluded_index_chunk_ids=frozenset(simple_ids),
                source_chunk_target=ROUTING_SOURCE_CHUNK_TARGET,
                source_chunk_limit=ROUTING_SOURCE_CHUNK_LIMIT,
            )
            simple_files = {
                suite.filename_by_chunk.get(str(item.index_chunk_id), "")
                for item in simple_pack.evidence
            }
            packed_files = {
                suite.filename_by_chunk.get(str(item.index_chunk_id), "")
                for item in packed
            }
            required = [
                {filenames[str(document_id)] for document_id in path if str(document_id) in filenames}
                for path in case.get("required_paths", ())
            ]
            required = [path for path in required if path]
            simple_complete = any(path <= simple_files for path in required)
            graph_complete = any(path <= simple_files | packed_files for path in required)
            return {
                "case_id": str(case["case_id"]),
                "hop_count": _safe_number(case.get("hop_count")),
                "simple_evidence_count": len(simple_pack.evidence),
                "graph_candidate_path_count": len(candidate_set.traversal.paths),
                "graph_hydrated_chunk_count": len(candidate_set.traversal.chunks),
                "graph_packed_chunk_count": len(packed),
                "simple_complete_path_present": bool(simple_complete),
                "graph_complete_path_present": bool(graph_complete),
                "graph_new_source_chunk_count": len(new_ids),
                "qualified_graph_needed": bool(not simple_complete and graph_complete and new_ids),
            }

        producers.append((case_id, produce))
    await _run_cases(state_path, state, phase="routing_qualification", cases=producers)
    records = [
        record["observation"]
        for record in state.get("phases", {}).get("routing_qualification", {}).values()
        if isinstance(record, Mapping) and record.get("status") == "completed" and isinstance(record.get("observation"), Mapping)
    ]
    qualified = sum(bool(record.get("qualified_graph_needed")) for record in records)
    state.setdefault("phase_summaries", {})["routing_qualification"]["qualified_count"] = qualified
    state["routing_qualification"] = {"qualified_count": qualified, "candidate_count": len(graph_cases)}
    _write_state(state_path, state)
    if qualified < MIN_QUALIFIED_ROUTING_CASES:
        raise LargeEvaluationError("routing_qualification_denominator_insufficient")
    return qualified


def _routing_simple_cases(cases: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return list(cases)


def _state_observations(state: Mapping[str, Any], phase: str) -> list[dict[str, Any]]:
    raw = state.get("phases", {}).get(phase, {})
    if not isinstance(raw, Mapping):
        return []
    return [
        dict(record["observation"])
        for record in raw.values()
        if isinstance(record, Mapping) and record.get("status") == "completed" and isinstance(record.get("observation"), Mapping)
    ]


def _public_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_action: dict[str, dict[str, Any]] = {}
    for action in sorted({str(record.get("expected_action")) for record in records}):
        subset = [record for record in records if record.get("expected_action") == action]
        outcomes = Counter(str(record.get("actual_outcome")) for record in subset)
        available = [record for record in subset if record.get("lexical_answer_match_available")]
        entry = {
            "case_count": len(subset),
            "outcomes": dict(sorted(outcomes.items())),
            "policy_correct": _rate(sum(bool(record.get("policy_correct")) for record in subset), len(subset)),
            "lexical_answer_match": _rate(sum(bool(record.get("lexical_answer_match")) for record in available), len(available)),
            "lexical_match_denominator_available": len(available),
            "forbidden_claim_hits": sum(bool(record.get("forbidden_claim_hit")) for record in subset),
        }
        # Refusal-family scoring still counts only `refused` as correct; the
        # false-premise split is tracked separately so a clarify shift stays
        # visible without changing the verdict semantics.
        if action == "decline_or_correct_false_premise":
            entry["false_premise_behavior"] = {
                "refused": int(outcomes.get("refused", 0)),
                "clarified": int(outcomes.get("clarify", 0)),
                "answered_anyway": int(
                    sum(
                        count
                        for outcome, count in outcomes.items()
                        if outcome not in {"refused", "clarify"}
                    )
                ),
            }
        by_action[action] = entry
    by_stratum: dict[str, dict[str, Any]] = {}
    for stratum in sorted({str(record.get("stratum")) for record in records}):
        subset = [record for record in records if record.get("stratum") == stratum]
        by_stratum[stratum] = {
            "case_count": len(subset),
            "policy_correct": _rate(sum(bool(record.get("policy_correct")) for record in subset), len(subset)),
            "outcomes": dict(sorted(Counter(str(record.get("actual_outcome")) for record in subset).items())),
        }
    return {
        "case_count": len(records),
        "outcomes": dict(sorted(Counter(str(record.get("actual_outcome")) for record in records).items())),
        "policy_correct": _rate(sum(bool(record.get("policy_correct")) for record in records), len(records)),
        "forbidden_claim_hit_count": sum(bool(record.get("forbidden_claim_hit")) for record in records),
        "by_expected_action": by_action,
        "by_stratum": by_stratum,
    }


def _routing_report(
    qualification: Sequence[Mapping[str, Any]],
    answers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    qualified = [record for record in qualification if record.get("qualified_graph_needed")]
    qualified_case_ids = {str(record.get("case_id")) for record in qualified}
    by_hop: dict[str, dict[str, int]] = defaultdict(lambda: {"candidate_count": 0, "qualified_count": 0})
    for record in qualification:
        key = str(record.get("hop_count"))
        by_hop[key]["candidate_count"] += 1
        by_hop[key]["qualified_count"] += bool(record.get("qualified_graph_needed"))
    auto = [record for record in answers if record.get("lane") == "auto"]
    simple = [record for record in answers if record.get("lane") == "simple"]
    auto_attempts = sum(bool(record.get("graph_route_attempted")) for record in auto)
    auto_admitted = sum(bool(record.get("graph_route_admitted")) for record in auto)
    false_attempts = sum(bool(record.get("graph_route_attempted")) for record in simple)
    # A Graph candidate that fails dynamic qualification is not evidence that Graph is
    # unnecessary; only an explicitly designed Auto control may be a negative label.
    auto_negative_controls = [
        record
        for record in auto
        if record.get("expected_graph_route") is False
        and str(record.get("case_id")) not in qualified_case_ids
    ]
    auto_true_attempts = sum(
        bool(record.get("graph_route_attempted"))
        for record in auto
        if str(record.get("case_id")) in qualified_case_ids
    )
    auto_false_attempts = sum(
        bool(record.get("graph_route_attempted")) for record in auto_negative_controls
    )
    auto_positive_controls = [
        record
        for record in auto
        if str(record.get("case_id")) in qualified_case_ids
    ]
    auto_label_conflicts = [
        record
        for record in auto_positive_controls
        if record.get("expected_graph_route") is False
    ]
    auto_unlabeled = [
        record
        for record in auto
        if str(record.get("case_id")) not in qualified_case_ids
        and record.get("expected_graph_route") is not False
    ]
    route_precision_denominator = auto_true_attempts + auto_false_attempts
    case_auto: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in auto:
        case_auto[str(record.get("case_id"))].append(record)
    case_any = sum(
        any(bool(item.get("graph_route_admitted")) for item in values)
        for case_id, values in case_auto.items()
        if case_id in qualified_case_ids
    )
    case_any_attempt = sum(
        any(bool(item.get("graph_route_attempted")) for item in values)
        for case_id, values in case_auto.items()
        if case_id in qualified_case_ids
    )
    repeat_groups = [
        values for values in case_auto.values() if len(values) == AUTO_REPEATS
    ]
    lexical_repeat_groups = [
        values
        for values in repeat_groups
        if all(bool(item.get("lexical_answer_match_available")) for item in values)
    ]
    lexical_all_correct = sum(
        all(bool(item.get("lexical_answer_match")) for item in values)
        for values in lexical_repeat_groups
    )
    lexical_all_incorrect = sum(
        not any(bool(item.get("lexical_answer_match")) for item in values)
        for values in lexical_repeat_groups
    )
    simple_by_case = {
        str(record.get("case_id")): record
        for record in simple
        if record.get("case_id") is not None
    }

    def paired_impact(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        pairs = [
            (simple_by_case[str(record.get("case_id"))], record)
            for record in records
            if str(record.get("case_id")) in simple_by_case
            and bool(record.get("lexical_answer_match_available"))
            and bool(
                simple_by_case[str(record.get("case_id"))].get(
                    "lexical_answer_match_available"
                )
            )
        ]
        rescue = sum(
            not bool(simple_record.get("lexical_answer_match"))
            and bool(auto_record.get("lexical_answer_match"))
            for simple_record, auto_record in pairs
        )
        harm = sum(
            bool(simple_record.get("lexical_answer_match"))
            and not bool(auto_record.get("lexical_answer_match"))
            for simple_record, auto_record in pairs
        )
        both_correct = sum(
            bool(simple_record.get("lexical_answer_match"))
            and bool(auto_record.get("lexical_answer_match"))
            for simple_record, auto_record in pairs
        )
        return {
            "pair_count": len(pairs),
            "rescue_count": rescue,
            "harm_count": harm,
            "both_correct_count": both_correct,
            "both_incorrect_count": len(pairs) - rescue - harm - both_correct,
            "net_rescue_count": rescue - harm,
        }

    def answer_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        lexical = [
            record
            for record in records
            if bool(record.get("lexical_answer_match_available"))
        ]
        tokens = [
            int(record["total_tokens"])
            for record in records
            if isinstance(record.get("total_tokens"), int)
        ]
        record_durations = [
            int(record["duration_ms"])
            for record in records
            if isinstance(record.get("duration_ms"), int)
        ]
        return {
            "observation_count": len(records),
            "outcomes": dict(
                sorted(
                    Counter(
                        str(record.get("actual_outcome")) for record in records
                    ).items()
                )
            ),
            "policy_correct": _rate(
                sum(bool(record.get("policy_correct")) for record in records),
                len(records),
            ),
            "lexical_answer_match": _rate(
                sum(bool(record.get("lexical_answer_match")) for record in lexical),
                len(lexical),
            ),
            "total_tokens": {
                "p50": _percentile(tokens, 0.50),
                "p95": _percentile(tokens, 0.95),
            },
            "duration_ms": {
                "p50": _percentile(record_durations, 0.50),
                "p95": _percentile(record_durations, 0.95),
            },
        }

    novel_source = [
        record
        for record in qualification
        if _safe_number(record.get("graph_new_source_chunk_count")) > 0
    ]
    novel_distinct_completion = sum(
        not bool(record.get("simple_complete_path_present"))
        and bool(record.get("graph_complete_path_present"))
        for record in novel_source
    )
    novel_redundant_completion = sum(
        bool(record.get("simple_complete_path_present")) for record in novel_source
    )
    novel_incomplete = (
        len(novel_source) - novel_distinct_completion - novel_redundant_completion
    )
    packed_counts = [
        _safe_number(record.get("graph_packed_chunk_count"))
        for record in qualification
    ]
    durations = [
        int(record.get("duration_ms", 0))
        for record in answers
        if isinstance(record.get("duration_ms"), int)
    ]
    if auto_label_conflicts:
        route_precision_status = "invalid_route_label_conflict"
    elif not auto_negative_controls:
        route_precision_status = "not_measured_no_auto_negative_controls"
    elif not route_precision_denominator:
        route_precision_status = "not_measured_no_route_attempts"
    else:
        route_precision_status = "measured"
    route_precision = (
        _rate(auto_true_attempts, route_precision_denominator)
        if route_precision_status == "measured"
        else _rate(0, 0)
    )
    return {
        "qualification": {
            "candidate_count": len(qualification),
            "qualified_count": len(qualified),
            "minimum_required": MIN_QUALIFIED_ROUTING_CASES,
            "minimum_met": len(qualified) >= MIN_QUALIFIED_ROUTING_CASES,
            "by_hop": dict(sorted(by_hop.items())),
            "simple_complete_path_count": sum(bool(item.get("simple_complete_path_present")) for item in qualification),
            "graph_complete_path_count": sum(bool(item.get("graph_complete_path_present")) for item in qualification),
            "novel_source_path_outcomes": {
                "case_count": len(novel_source),
                "distinct_required_path_completion_count": novel_distinct_completion,
                "simple_already_complete_count": novel_redundant_completion,
                "required_path_still_incomplete_count": novel_incomplete,
            },
            "bounds": {
                "candidate_path_limit": ROUTING_EDGE_LIMIT,
                "candidate_path_limit_hit_count": sum(
                    _safe_number(item.get("graph_candidate_path_count"))
                    >= ROUTING_EDGE_LIMIT
                    for item in qualification
                ),
                "source_chunk_target": ROUTING_SOURCE_CHUNK_TARGET,
                "source_chunk_target_hit_count": sum(
                    count >= ROUTING_SOURCE_CHUNK_TARGET for count in packed_counts
                ),
                "source_chunk_limit": ROUTING_SOURCE_CHUNK_LIMIT,
                "source_chunk_limit_hit_count": sum(
                    count >= ROUTING_SOURCE_CHUNK_LIMIT for count in packed_counts
                ),
                "maximum_packed_chunk_count": max(packed_counts, default=0),
            },
        },
        "answers": {
            "simple_observation_count": len(simple),
            "auto_observation_count": len(auto),
            "auto_graph_call_count": sum(
                _safe_number(item.get("graph_call_count")) for item in auto
            ),
            "expected_auto_observation_count": len(qualified) * AUTO_REPEATS,
            "auto_route_attempt_rate": _rate(auto_attempts, len(auto)),
            "auto_route_admission_rate": _rate(auto_admitted, len(auto)),
            "attempted_run_any_admission_rate": _rate(auto_admitted, auto_attempts),
            "case_level_any_admission_rate": _rate(case_any, len(qualified)),
            "qualified_case_any_attempt_rate": _rate(
                case_any_attempt,
                len(qualified),
            ),
            "observation_route_attempt_recall": _rate(
                auto_true_attempts,
                len(auto_positive_controls),
            ),
            "observation_route_attempt_recall_status": (
                "measured_against_dynamic_qualification_label"
                if auto_positive_controls
                else "not_measured_no_auto_positive_controls"
            ),
            "route_precision": route_precision,
            "route_precision_status": route_precision_status,
            "route_label_contract": (
                "positive=dynamic_qualified_graph_needed;"
                "negative=explicit_expected_graph_route_false"
            ),
            "auto_positive_control_observation_count": len(auto_positive_controls),
            "auto_negative_control_observation_count": len(auto_negative_controls),
            "auto_unlabeled_observation_count": len(auto_unlabeled),
            "auto_route_label_conflict_count": len(auto_label_conflicts),
            "simple_false_route_attempt_count": false_attempts,
            "simple_false_route_attempt_status": (
                "invariant_violation_simple_lane_graph_attempted"
                if false_attempts
                else "structural_control_graph_tool_not_exposed"
            ),
            "repeatability": {
                "case_count": len(case_auto),
                "complete_repeat_case_count": len(repeat_groups),
                "expected_repeats_per_case": AUTO_REPEATS,
                "stable_outcome": _rate(
                    sum(
                        len({str(item.get("actual_outcome")) for item in values})
                        == 1
                        for values in repeat_groups
                    ),
                    len(repeat_groups),
                ),
                "stable_route_attempt": _rate(
                    sum(
                        len(
                            {
                                bool(item.get("graph_route_attempted"))
                                for item in values
                            }
                        )
                        == 1
                        for values in repeat_groups
                    ),
                    len(repeat_groups),
                ),
                "stable_admission": _rate(
                    sum(
                        len(
                            {
                                bool(item.get("graph_route_admitted"))
                                for item in values
                            }
                        )
                        == 1
                        for values in repeat_groups
                    ),
                    len(repeat_groups),
                ),
                "lexical": {
                    "case_count": len(lexical_repeat_groups),
                    "pass_all_repeats_correct": _rate(
                        lexical_all_correct, len(lexical_repeat_groups)
                    ),
                    "all_incorrect": _rate(
                        lexical_all_incorrect, len(lexical_repeat_groups)
                    ),
                    "mixed": _rate(
                        len(lexical_repeat_groups)
                        - lexical_all_correct
                        - lexical_all_incorrect,
                        len(lexical_repeat_groups),
                    ),
                },
            },
            "paired_lexical_impact": {
                "all_auto": paired_impact(auto),
                "graph_attempted": paired_impact(
                    [item for item in auto if item.get("graph_route_attempted")]
                ),
                "graph_admitted": paired_impact(
                    [item for item in auto if item.get("graph_route_admitted")]
                ),
                "interpretation": "sampled_pair_not_causal_counterfactual",
            },
            "auto_only": answer_summary(auto),
            "auto_by_graph_attempt": {
                "attempted": answer_summary(
                    [item for item in auto if item.get("graph_route_attempted")]
                ),
                "not_attempted": answer_summary(
                    [
                        item
                        for item in auto
                        if not item.get("graph_route_attempted")
                    ]
                ),
                "interpretation": "model_self_selected_groups_not_causal",
            },
            "outcomes": dict(sorted(Counter(str(record.get("actual_outcome")) for record in answers).items())),
            "policy_correct": _rate(sum(bool(record.get("policy_correct")) for record in answers), len(answers)),
            "lexical_answer_match": _rate(
                sum(bool(record.get("lexical_answer_match")) for record in answers if record.get("lexical_answer_match_available")),
                sum(bool(record.get("lexical_answer_match_available")) for record in answers),
            ),
            "graph_timeout_count": sum(int(record.get("graph_timeout_count", 0)) for record in answers),
            "duration_ms": {"p50": _percentile(durations, 0.50), "p95": _percentile(durations, 0.95)},
        },
    }


def _enterprise_report(extraction: Sequence[Mapping[str, Any]], answers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "extraction": dict(extraction[0]) if extraction else {"status": "missing"},
        "answering": {
            "case_count": len(answers),
            "outcomes": dict(sorted(Counter(str(record.get("actual_outcome")) for record in answers).items())),
            "quality_pass": _rate(sum(bool(record.get("policy_correct")) for record in answers), len(answers)),
            "lexical_answer_match": _rate(
                sum(bool(record.get("lexical_answer_match")) for record in answers if record.get("lexical_answer_match_available")),
                sum(bool(record.get("lexical_answer_match_available")) for record in answers),
            ),
            "expected_citation_hit": _rate(sum(bool(record.get("expected_citation_hit")) for record in answers), len(answers)),
        },
    }


def _graph_report(answers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(answers),
        "outcomes": dict(sorted(Counter(str(record.get("actual_outcome")) for record in answers).items())),
        "quality_pass": _rate(sum(bool(record.get("policy_correct")) for record in answers), len(answers)),
        "lexical_answer_match": _rate(
            sum(bool(record.get("lexical_answer_match")) for record in answers if record.get("lexical_answer_match_available")),
            sum(bool(record.get("lexical_answer_match_available")) for record in answers),
        ),
        "expected_citation_hit": _rate(sum(bool(record.get("expected_citation_hit")) for record in answers), len(answers)),
        "citation_query_term_leakage_count": sum(bool(record.get("citation_query_term_leakage")) for record in answers),
        "answer_query_term_leakage_count": sum(bool(record.get("answer_query_term_leakage")) for record in answers),
        "graph_route_attempt_rate": _rate(sum(bool(record.get("graph_route_attempted")) for record in answers), len(answers)),
    }


def _usage_report(state: Mapping[str, Any]) -> dict[str, Any]:
    records: list[Mapping[str, Any]] = []
    for phase in ("public_answer_refusal", "routing_agent_answers", "enterprise_answering", "graph_answering"):
        records.extend(_state_observations(state, phase))
    totals = Counter()
    durations: list[int] = []
    for record in records:
        usage = record.get("model_call_usage", {})
        if isinstance(usage, Mapping):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "model_call_count", "model_rounds", "retrieval_calls", "evidence_refs"):
                totals[key] += _safe_number(usage.get(key))
        if isinstance(record.get("duration_ms"), int):
            durations.append(int(record["duration_ms"]))
    return {
        "agent_observation_count": len(records),
        "totals": dict(sorted(totals.items())),
        "duration_ms": {"p50": _percentile(durations, 0.50), "p95": _percentile(durations, 0.95), "max": max(durations) if durations else None},
    }


def _provision_report(
    runtime: EvaluationRuntime,
    suites: Mapping[str, Any],
    *,
    routing_dataset_id: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    report_specs = dict(SPECS)
    if routing_dataset_id != SPECS["routing"].dataset_id:
        variant_spec = next(
            (
                spec
                for spec in DATASET_SPECS.values()
                if spec.dataset_id == routing_dataset_id
            ),
            None,
        )
        if variant_spec is None:
            raise LargeEvaluationError("large_evaluation_routing_dataset_identity_invalid")
        report_specs.pop("routing")
        report_specs["routing_variant"] = variant_spec
    for name, spec in report_specs.items():
        path = runtime.runtime_root / "large-evaluation-provisioning" / f"{spec.dataset_id}.json"
        checkpoint = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        result[spec.dataset_id] = {
            "status": checkpoint.get("status"),
            "frozen_document_count": checkpoint.get("frozen_document_count") or spec.expected_document_count,
            "indexed_document_count": checkpoint.get("indexed_document_count") or checkpoint.get("uploaded_document_count") or spec.expected_document_count,
            "uploaded_document_count": checkpoint.get("uploaded_document_count"),
            "deduplicated_document_count": checkpoint.get("deduplicated_document_count", 0),
            "parser_fallback_document_count": checkpoint.get(
                "parser_fallback_document_count", 0
            ),
            "parser_content_aliases": checkpoint.get(
                "parser_content_aliases", {}
            ),
            "index_revision_id": checkpoint.get("index_revision_id"),
            "graph_build_id": checkpoint.get("graph_build_id"),
            "checkpoint_event_count": len(checkpoint.get("events", [])) if isinstance(checkpoint.get("events"), list) else 0,
        }
        if isinstance(checkpoint.get("reused_from"), Mapping):
            result[spec.dataset_id]["reused_from"] = dict(checkpoint["reused_from"])
    return result


def _human_report(report: Mapping[str, Any]) -> str:
    public = report["quality"]["public_answer_refusal"]
    routing = report["quality"]["routing"]
    enterprise = report["quality"]["enterprise"]
    graph = report["quality"]["graph_rag"]
    usage = report["usage"]
    routing_novelty = routing["qualification"]["novel_source_path_outcomes"]
    routing_bounds = routing["qualification"]["bounds"]
    routing_repeatability = routing["answers"]["repeatability"]
    routing_lexical = routing_repeatability["lexical"]
    lines = [
        "# Large RAG Evaluation v1",
        "",
        f"- 状态：`{report['status']}`",
        f"- 计划绑定：`{report['plan_binding_sha256']}`",
        f"- 运行时：`{report['runtime']['mode']}`，构建 ` {report['runtime']['build_revision']}`",
        "",
        "## 语料与索引",
        "",
        "| 语料 | 冻结文件 | 实际索引文档 | 内容别名 | 解析别名 | 状态 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for dataset, value in report["provisioning"].items():
        lines.append(
            f"| `{dataset}` | {value.get('frozen_document_count') or '—'} | {value.get('indexed_document_count') or '—'} | {value.get('deduplicated_document_count', 0)} | {value.get('parser_fallback_document_count', 0)} | `{value.get('status')}` |"
        )
    lines.extend(
        [
            "",
            "## 质量结果",
            "",
            f"公共回答/拒答：{public['case_count']} cases，策略正确率 `{public['policy_correct']['value']}`，禁用声明命中 {public['forbidden_claim_hit_count']} 次。",
            f"MuSiQue 路由：资格候选 {routing['qualification']['candidate_count']}，动态合格 {routing['qualification']['qualified_count']}，Auto 观测 {routing['answers']['auto_observation_count']}，Auto 路由尝试率 `{routing['answers']['auto_route_attempt_rate']['value']}`。",
            f"Graph 新证据分解：发现新 chunk {routing_novelty['case_count']} 例；补齐 required path {routing_novelty['distinct_required_path_completion_count']} 例，Simple 已完整 {routing_novelty['simple_already_complete_count']} 例，required path 仍不完整 {routing_novelty['required_path_still_incomplete_count']} 例。路径候选上限 {routing_bounds['candidate_path_limit']} 命中 {routing_bounds['candidate_path_limit_hit_count']} 例；source target/hard limit 命中 {routing_bounds['source_chunk_target_hit_count']}/{routing_bounds['source_chunk_limit_hit_count']} 例。",
            f"Auto 重复性：全部 {routing_repeatability['expected_repeats_per_case']} 次 lexical 正确 {routing_lexical['pass_all_repeats_correct']['numerator']}/{routing_lexical['pass_all_repeats_correct']['denominator']}；outcome 一致率 `{routing_repeatability['stable_outcome']['value']}`，Graph 尝试一致率 `{routing_repeatability['stable_route_attempt']['value']}`。发生 Graph 尝试的运行中至少一次发现新证据的比例 `{routing['answers']['attempted_run_any_admission_rate']['value']}`；route precision 状态 `{routing['answers']['route_precision_status']}`。",
            f"Enterprise 抽取：微平均 F1 `{enterprise['extraction'].get('micro_f1', {}).get('value')}`，宏平均 F1 `{enterprise['extraction'].get('macro_f1', {}).get('value')}`；答案质量通过率 `{enterprise['answering']['quality_pass']['value']}`。",
            f"Graph-RAG：{graph['case_count']} cases，答案质量通过率 `{graph['quality_pass']['value']}`，期望引用命中率 `{graph['expected_citation_hit']['value']}`。",
            "",
            "## 资源与恢复",
            "",
            f"Agent 观测 {usage['agent_observation_count']} 次；总 token `{usage['totals'].get('total_tokens', 0)}`；耗时 p50/p95 `{usage['duration_ms']['p50']}`/`{usage['duration_ms']['p95']}` ms。",
            "所有中间状态保存在 owner-only campaign checkpoint；最终报告只在全部计划案例完成后写入。公共套件的字节重复文件按内容别名记录；解析兼容别名保留同一冻结字节并单独计数。",
            "",
            "详细的逐阶段、逐动作、逐关系类型和逐 hop 指标见同目录的私有 `final-report.json` 与 `campaign-state.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--evaluation-runtime", type=Path, default=DEFAULT_RUNTIME_MANIFEST)
    parser.add_argument("--provider-smoke-checkpoint", type=Path, default=DEFAULT_PROVIDER_SMOKE)
    parser.add_argument("--campaign-checkpoint", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--locked-output", type=Path, default=DEFAULT_LOCKED)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--routing-root",
        type=Path,
        default=ROUTING_ROOT,
        help="routing corpus root; defaults to the frozen expanded v1 corpus",
    )
    parser.add_argument(
        "--bindings-path",
        type=Path,
        help="optional owner-only suite bindings file under the evaluation runtime",
    )
    parser.add_argument("--confirm")
    return parser


async def _campaign(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.confirm != CONFIRM:
        raise LargeEvaluationError("large_evaluation_confirmation_invalid")
    plan = _load_private_plan(arguments.plan)
    _load_provider_smoke(arguments.provider_smoke_checkpoint)
    provider_smoke_sha256 = _sha256_file(arguments.provider_smoke_checkpoint)
    runtime = load_evaluation_runtime(
        arguments.evaluation_runtime,
        require_adaptive_graph=True,
        allow_canonical_checkout=True,
    )
    routing_dataset_id = (
        plan.get("coverage", {}).get("routing", {}).get("dataset_id")
        or SPECS["routing"].dataset_id
    )
    if not isinstance(routing_dataset_id, str):
        raise LargeEvaluationError("large_evaluation_routing_dataset_identity_invalid")
    expected_dataset_ids = {
        spec.dataset_id for spec in SPECS.values()
    } - {SPECS["routing"].dataset_id}
    expected_dataset_ids.add(routing_dataset_id)
    await check_runtime(
        arguments.plan,
        arguments.evaluation_runtime,
        arguments.bindings_path,
    )
    suites, suites_sha256 = _load_bindings(
        runtime,
        bindings_path=arguments.bindings_path,
        expected_dataset_ids=expected_dataset_ids,
    )
    corpora = _load_corpus_files(routing_root=arguments.routing_root)
    if corpora["routing"]["manifest"].get("dataset_id") != routing_dataset_id:
        raise LargeEvaluationError("large_evaluation_routing_dataset_identity_invalid")
    dependencies = None
    try:
        dependencies = build_worker_dependencies(
            env_file=runtime.env_file,
            worker_id="large-evaluation-campaign",
        )
        await dependencies.check_readiness()
        selected_profiles = await _selected_embedding_profiles(dependencies)
        runtime_chat_id = runtime.adaptive_graph.answer_profile_revision_id if runtime.adaptive_graph else None
        if runtime_chat_id is None:
            raise LargeEvaluationError("large_evaluation_runtime_chat_profile_missing")
        binding = {
            "schema_version": SCHEMA_VERSION,
            "plan_binding_sha256": plan["plan_binding_sha256"],
            "suite_bindings_sha256": suites_sha256,
            "provider_contract": plan["plan_binding"]["provider_contract"],
            "runtime_contract": plan["plan_binding"]["runtime_contract"],
            "runtime_build_revision": runtime.build_revision,
            "chat_profile_revision_id": str(runtime_chat_id),
            "provider_smoke_sha256": provider_smoke_sha256,
            "resilience_policy_sha256": RESILIENCE_POLICY_SHA256,
            **selected_profiles,
        }
        state = load_or_create(arguments.campaign_checkpoint, binding)
        existing_policy = state.get("resilience_policy")
        if existing_policy is not None and existing_policy != RESILIENCE_POLICY:
            raise LargeEvaluationError("large_evaluation_resilience_policy_changed")
        state["resilience_policy"] = RESILIENCE_POLICY
        state["status"] = "running"
        state.pop("last_failure", None)
        _write_state(arguments.campaign_checkpoint, state)
        with _campaign_lock(arguments.campaign_checkpoint):
            graph_suite = await _load_suite(
                dependencies, runtime, key="graph_rag", suite_binding=suites["graph-rag-v1"], corpus=corpora["graph_rag"]
            )
            routing_suite = await _load_suite(
                dependencies,
                runtime,
                key="routing",
                suite_binding=suites[routing_dataset_id],
                corpus=corpora["routing"],
                corpus_root=arguments.routing_root,
            )
            enterprise_suite = await _load_suite(
                dependencies, runtime, key="enterprise", suite_binding=suites["enterprise-profile-qualification-v1"], corpus=corpora["enterprise"]
            )
            public_suite = await _load_suite(
                dependencies, runtime, key="public", suite_binding=suites["public-rag-benchmark-suite-v1"], corpus=corpora["public"]
            )
            state["suite_identity"] = {
                name: {
                    "knowledge_base_id": str(suite.knowledge_base_id),
                    "index_revision_id": str(suite.index_revision_id),
                    "graph_build_id": str(suite.graph_build_id) if suite.graph_build_id else None,
                }
                for name, suite in {
                    "graph_rag": graph_suite,
                    "routing": routing_suite,
                    "enterprise": enterprise_suite,
                    "public": public_suite,
                }.items()
            }
            _write_state(arguments.campaign_checkpoint, state)
            agent = NativeToolCallingAgent(
                dependencies.chat_model_adapter,
                ChatEvidenceRetriever(dependencies.retrieval_service),
                dependencies.visual_evidence_preparer,
                min_cosine_similarity=dependencies.settings.retrieval.min_cosine_similarity,
                min_rerank_score=dependencies.settings.retrieval.min_rerank_score,
                cross_modal_min_cosine_similarity=dependencies.settings.retrieval.cross_modal_min_cosine_similarity,
            )

            qualified_count = int(state.get("routing_qualification", {}).get("qualified_count", 0))
            graph_candidate_count = sum(
                case.get("route_label") == "graph_needed_candidate"
                for case in corpora["routing"]["cases"]
            )
            if len(completed_case_ids(state, phase="routing_qualification")) < graph_candidate_count:
                qualified_count = await _run_routing_qualification(
                    arguments.campaign_checkpoint,
                    state,
                    dependencies,
                    routing_suite,
                    corpora["routing"]["cases"],
                    corpora["routing"]["manifest"],
                )
            elif qualified_count < MIN_QUALIFIED_ROUTING_CASES:
                raise LargeEvaluationError("routing_qualification_denominator_insufficient")

            public_producers: list[tuple[str, Callable[[], Any]]] = []
            for case in corpora["public"]["cases"]:
                public_producers.append(
                    (
                        str(case["case_id"]),
                        lambda case=case: _run_agent(
                            agent, dependencies, public_suite,
                            question=str(case["question"]), lane="simple", score=_score_public(case),
                        ),
                    )
                )
            await _run_cases(arguments.campaign_checkpoint, state, phase="public_answer_refusal", cases=public_producers)

            qualified_ids = {
                str(record.get("case_id"))
                for record in _state_observations(state, "routing_qualification")
                if record.get("qualified_graph_needed")
            }
            routing_producers: list[tuple[str, Callable[[], Any]]] = []
            for case in corpora["routing"]["cases"]:
                case_id = str(case["case_id"])
                routing_producers.append(
                    (
                        f"{case_id}::simple::1",
                        lambda case=case: _run_agent(
                            agent, dependencies, routing_suite,
                            question=str(case["question"]), lane="simple", score=_score_routing(case),
                        ),
                    )
                )
                if case_id in qualified_ids:
                    for repeat in range(1, AUTO_REPEATS + 1):
                        routing_producers.append(
                            (
                                f"{case_id}::auto::{repeat}",
                                lambda case=case, repeat=repeat: _run_agent(
                                    agent, dependencies, routing_suite,
                                    question=str(case["question"]), lane="auto", score=_score_routing(case),
                                ),
                            )
                        )
            await _run_cases(arguments.campaign_checkpoint, state, phase="routing_agent_answers", cases=routing_producers)

            async def extract_enterprise():
                if enterprise_suite.build is None:
                    raise LargeEvaluationError("large_evaluation_enterprise_graph_missing")
                edges = await dependencies.graphiti_runtime.diagnostic_edges(enterprise_suite.build)
                return _enterprise_extraction_observation(edges, corpora["enterprise"]["entities"], corpora["enterprise"]["relations"], corpora["enterprise"]["controls"])

            await _run_cases(
                arguments.campaign_checkpoint,
                state,
                phase="enterprise_extraction",
                cases=[("enterprise-extraction-snapshot", extract_enterprise)],
            )

            relation_filename = {
                str(row["relation_id"]): str(row["document_filename"])
                for row in corpora["enterprise"]["relations"]
            }
            enterprise_producers: list[tuple[str, Callable[[], Any]]] = []
            for case in corpora["enterprise"]["cases"]:
                expected_filename = relation_filename.get(str(case["required_relation_ids"][0]), "")
                enterprise_producers.append(
                    (
                        str(case["case_id"]),
                        lambda case=case, expected_filename=expected_filename: _run_agent(
                            agent, dependencies, enterprise_suite,
                            question=str(case["question"]), lane="auto", score=_score_enterprise(case, expected_filename),
                        ),
                    )
                )
            await _run_cases(arguments.campaign_checkpoint, state, phase="enterprise_answering", cases=enterprise_producers)

            graph_producers: list[tuple[str, Callable[[], Any]]] = []
            graph_answer_cases = [case for case in corpora["graph_rag"]["cases"] if case.get("answerable")]
            graph_filename = {
                str(row["document_id"]): str(row["filename"])
                for row in corpora["graph_rag"]["manifest"].get("documents", [])
                if isinstance(row, Mapping)
            }
            for case in graph_answer_cases:
                expected_filename = graph_filename.get(str(case.get("answer_document_id")), "")
                graph_producers.append(
                    (
                        str(case["case_id"]),
                        lambda case=case, expected_filename=expected_filename: _run_agent(
                            agent, dependencies, graph_suite,
                            question=str(case["question"]), lane="auto", score=_score_graph(case, expected_filename),
                        ),
                    )
                )
            await _run_cases(arguments.campaign_checkpoint, state, phase="graph_answering", cases=graph_producers)

            report = _build_report(
                plan=plan,
                runtime=runtime,
                suites=suites,
                state=state,
                public_suite=public_suite,
                routing_dataset_id=routing_dataset_id,
            )
            report_digest = hashlib.sha256(_canonical(report)).hexdigest()
            report["artifact_sha256"] = report_digest
            locked_sha256 = write_private_json(arguments.locked_output, report)
            _write_markdown(arguments.markdown_output, _human_report(report))
            state["status"] = "completed"
            state["final_artifacts"] = {
                "locked_report_sha256": locked_sha256,
                "locked_report_path": str(arguments.locked_output),
                "markdown_report_path": str(arguments.markdown_output),
            }
            _write_state(arguments.campaign_checkpoint, state)
            return {
                "status": "completed",
                "locked_report_sha256": locked_sha256,
                "campaign_checkpoint": str(arguments.campaign_checkpoint),
                "markdown_report": str(arguments.markdown_output),
                "qualified_routing_case_count": qualified_count,
            }
    finally:
        if dependencies is not None:
            await dependencies.close()


def _enterprise_extraction_observation(
    edges: Sequence[Any],
    entities: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    identity_controls: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    entity_types: dict[str, str] = {}
    entity_surfaces: dict[str, str] = {}
    for row in entities:
        canonical = _norm(row.get("entity_id") or row.get("name"))
        if not canonical:
            continue
        entity_types[canonical] = str(row.get("entity_type", "unknown"))
        aliases = row.get("aliases", ())
        if not isinstance(aliases, Sequence) or isinstance(aliases, (str, bytes)):
            aliases = ()
        for value in (row.get("name"), *aliases):
            surface = _norm(value)
            if not surface:
                continue
            existing = entity_surfaces.setdefault(surface, canonical)
            if existing != canonical:
                raise LargeEvaluationError("enterprise gold entity surface is ambiguous")

    def gold_entity(row: Mapping[str, Any], prefix: str) -> str:
        explicit_id = _norm(row.get(f"{prefix}_entity_id"))
        if explicit_id:
            return explicit_id
        surface = _norm(row.get(f"{prefix}_entity"))
        return entity_surfaces.get(surface, surface)

    positive: dict[str, tuple[str, str, str]] = {}
    relation_types: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for row in relations:
        relation_id = str(row.get("relation_id", ""))
        triple = (
            gold_entity(row, "source"),
            _norm(row.get("edge_type")),
            gold_entity(row, "target"),
        )
        if relation_id and all(triple):
            positive[relation_id] = triple
            relation_types[str(row.get("edge_type"))].add(triple)
    asserted_controls = {
        (
            gold_entity(row, "source"),
            _norm(row.get("asserted_edge")),
            gold_entity(row, "target"),
        )
        for row in controls
        if (row.get("source_entity_id") or row.get("source_entity"))
        and row.get("asserted_edge")
        and (row.get("target_entity_id") or row.get("target_entity"))
    }
    known_types = {_norm(name): name for name in relation_types}
    observed: set[tuple[str, str, str]] = set()
    observed_endpoint_pairs: set[tuple[str, str]] = set()
    conditional_relation_observations: set[tuple[str, str, str]] = set()
    endpoint_pairs: Counter[str] = Counter()
    unknown_predicate_count = 0
    malformed_edge_count = 0
    endpoint_count = 0
    resolved_endpoint_count = 0
    semantic_edge_signatures: set[tuple[str, str, str, str]] = set()
    endpoint_relation_signatures: set[tuple[str, str, str]] = set()
    observed_entity_uuids: dict[str, set[str]] = defaultdict(set)
    scorable_edge_count = 0
    gold_pair_relations: dict[tuple[str, str], set[str]] = defaultdict(set)
    for source, predicate, target in (*positive.values(), *asserted_controls):
        gold_pair_relations[(source, target)].add(predicate)
    for edge in edges:
        raw_source = _norm(getattr(edge, "source_entity_name", ""))
        raw_target = _norm(getattr(edge, "target_entity_name", ""))
        predicate = _norm(getattr(edge, "relation_type", ""))
        if not predicate:
            fact = _norm(getattr(edge, "fact", ""))
            predicate = next((candidate for candidate in known_types if candidate in fact), "")
        if not raw_source or not raw_target or not predicate:
            malformed_edge_count += 1
            unknown_predicate_count += int(not predicate)
            continue
        endpoint_count += 2
        canonical_source = entity_surfaces.get(raw_source)
        canonical_target = entity_surfaces.get(raw_target)
        resolved_endpoint_count += int(canonical_source is not None)
        resolved_endpoint_count += int(canonical_target is not None)
        source = canonical_source or f"unresolved:{raw_source}"
        target = canonical_target or f"unresolved:{raw_target}"
        observed.add((source, predicate, target))
        source_type = entity_types.get(canonical_source or "", "unresolved")
        target_type = entity_types.get(canonical_target or "", "unresolved")
        endpoint_pairs[f"{source_type}->{target_type}"] += 1
        source_identity = str(getattr(edge, "source_entity_uuid", "") or source)
        target_identity = str(getattr(edge, "target_entity_uuid", "") or target)
        if canonical_source is not None:
            observed_entity_uuids[canonical_source].add(source_identity)
        if canonical_target is not None:
            observed_entity_uuids[canonical_target].add(target_identity)
        endpoint_relation_signatures.add((source_identity, predicate, target_identity))
        semantic_edge_signatures.add(
            (
                source_identity,
                predicate,
                target_identity,
                _norm(getattr(edge, "fact", "")),
            )
        )
        scorable_edge_count += 1
        if canonical_source is not None and canonical_target is not None:
            pair = (canonical_source, canonical_target)
            observed_endpoint_pairs.add(pair)
            if pair in gold_pair_relations:
                conditional_relation_observations.add(
                    (canonical_source, predicate, canonical_target)
                )
    gold = set(positive.values())
    allowed_gold = gold | asserted_controls
    precision_hits = observed & allowed_gold
    recall_hits = observed & gold
    micro_precision_value = len(precision_hits) / len(observed) if observed else 0.0
    micro_recall_value = len(recall_hits) / len(gold) if gold else 0.0
    micro = {
        "precision": _rate(len(precision_hits), len(observed)),
        "recall": _rate(len(recall_hits), len(gold)),
        "f1": {"numerator": _f1(micro_precision_value, micro_recall_value), "denominator": 1, "value": _f1(micro_precision_value, micro_recall_value)},
    }
    per_type: list[dict[str, Any]] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for edge_type in sorted(relation_types):
        normalized_type = _norm(edge_type)
        gold_type = relation_types[edge_type]
        observed_type = {triple for triple in observed if triple[1] == normalized_type}
        allowed_type = gold_type | {
            triple for triple in asserted_controls if triple[1] == normalized_type
        }
        precision_type_hits = observed_type & allowed_type
        recall_type_hits = observed_type & gold_type
        precision = (
            len(precision_type_hits) / len(observed_type) if observed_type else 0.0
        )
        recall = len(recall_type_hits) / len(gold_type) if gold_type else 0.0
        f1 = _f1(precision, recall)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        per_type.append(
            {
                "edge_type": edge_type,
                "gold_support": len(gold_type),
                "observed_support": len(observed_type),
                "true_positive_support": len(recall_type_hits),
                "precision": {"numerator": len(precision_type_hits), "denominator": len(observed_type), "value": round(precision, 6) if observed_type else 0.0},
                "recall": {"numerator": len(recall_type_hits), "denominator": len(gold_type), "value": round(recall, 6) if gold_type else 0.0},
                "f1": {"numerator": f1, "denominator": 1, "value": f1},
            }
        )
    forbidden: list[dict[str, Any]] = []
    for control in controls:
        triple = (
            gold_entity(control, "source"),
            _norm(control.get("forbidden_edge")),
            gold_entity(control, "target"),
        )
        hit_count = int(triple in observed)
        forbidden.append(
            {
                "control_id": str(control.get("control_id")),
                "forbidden_edge": str(control.get("forbidden_edge")),
                "hit_count": hit_count,
            }
        )
    expected_pairs = {
        (entity_types.get(triple[0], "unresolved"), entity_types.get(triple[2], "unresolved"))
        for triple in allowed_gold
    }
    unexpected_pairs = Counter()
    for triple in observed:
        pair = (
            entity_types.get(triple[0], "unresolved"),
            entity_types.get(triple[2], "unresolved"),
        )
        if pair not in expected_pairs:
            unexpected_pairs[f"{pair[0]}->{pair[1]}"] += 1
    relation_support = [
        {
            "relation_id": relation_id,
            "edge_type": next((str(row.get("edge_type")) for row in relations if str(row.get("relation_id")) == relation_id), ""),
            "observed": int(triple in observed),
        }
        for relation_id, triple in sorted(positive.items())
    ]
    macro = {
        "precision": {"numerator": round(statistics.fmean(precision_values), 6) if precision_values else 0.0, "denominator": 1, "value": round(statistics.fmean(precision_values), 6) if precision_values else 0.0},
        "recall": {"numerator": round(statistics.fmean(recall_values), 6) if recall_values else 0.0, "denominator": 1, "value": round(statistics.fmean(recall_values), 6) if recall_values else 0.0},
        "f1": {"numerator": round(statistics.fmean(f1_values), 6) if f1_values else 0.0, "denominator": 1, "value": round(statistics.fmean(f1_values), 6) if f1_values else 0.0},
    }
    gold_endpoint_pairs = set(gold_pair_relations)
    resolved_gold_pairs = observed_endpoint_pairs & gold_endpoint_pairs
    conditional_relation_hits = {
        triple
        for triple in conditional_relation_observations
        if triple[1] in gold_pair_relations[(triple[0], triple[2])]
    }
    unresolved_endpoint_count = endpoint_count - resolved_endpoint_count
    fragmentation_by_entity = {
        entity_id: len(uuids) - 1
        for entity_id, uuids in sorted(observed_entity_uuids.items())
        if len(uuids) > 1
    }
    identity_control_results: list[dict[str, Any]] = []
    for control in identity_controls:
        left = _norm(control.get("left_entity_id"))
        right = _norm(control.get("right_entity_id"))
        shared_uuids = observed_entity_uuids.get(left, set()) & observed_entity_uuids.get(
            right, set()
        )
        identity_control_results.append(
            {
                "control_id": str(control.get("control_id", "")),
                "left_entity_id": str(control.get("left_entity_id", "")),
                "right_entity_id": str(control.get("right_entity_id", "")),
                "shared_uuid_count": len(shared_uuids),
                "hit_count": int(bool(shared_uuids)),
            }
        )
    return {
        "case_id": "enterprise-extraction-snapshot",
        "status": "completed",
        "gold_relation_count": len(gold),
        "gold_relation_assertion_count": len(positive),
        "observed_raw_edge_count": len(edges),
        "observed_unique_edge_count": len(observed),
        "duplicate_observed_edge_count": max(
            0, scorable_edge_count - len(semantic_edge_signatures)
        ),
        "endpoint_relation_instance_excess_count": max(
            0, scorable_edge_count - len(endpoint_relation_signatures)
        ),
        "malformed_edge_count": malformed_edge_count,
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_f1": micro["f1"],
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f1": macro["f1"],
        "relation_type_count": len(relation_types),
        "per_relation_type_support": per_type,
        "per_relation_support": relation_support,
        "entity_type_confusion": {
            "endpoint_type_pair_counts": dict(sorted(endpoint_pairs.items())),
            "unexpected_endpoint_type_pair_counts": dict(sorted(unexpected_pairs.items())),
            "unresolved_endpoint_count": unresolved_endpoint_count,
        },
        "endpoint_resolution": {
            "resolved": _rate(resolved_endpoint_count, endpoint_count),
            "gold_pair_recall": _rate(len(resolved_gold_pairs), len(gold_endpoint_pairs)),
        },
        "relation_classification_given_gold_endpoints": _rate(
            len(conditional_relation_hits), len(conditional_relation_observations)
        ),
        "entity_identity": {
            "gold_entity_count": len(entity_types),
            "observed_canonical_entity_count": len(observed_entity_uuids),
            "fragmentation_excess_count": sum(fragmentation_by_entity.values()),
            "fragmentation_by_entity": fragmentation_by_entity,
            "distinct_entity_controls": identity_control_results,
            "forbidden_merge_hit_count": sum(
                item["hit_count"] for item in identity_control_results
            ),
        },
        "unknown_predicate_count": unknown_predicate_count,
        "forbidden_edge_hits": forbidden,
        "forbidden_edge_hit_count": sum(item["hit_count"] for item in forbidden),
    }


def _build_report(
    *,
    plan: Mapping[str, Any],
    runtime: EvaluationRuntime,
    suites: Mapping[str, Any],
    state: Mapping[str, Any],
    public_suite: SuiteRuntime,
    routing_dataset_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": FINAL_SCHEMA,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "plan_binding_sha256": plan["plan_binding_sha256"],
        "provider_contract": plan["plan_binding"]["provider_contract"],
        "runtime": {
            "mode": "isolated_host_native_only",
            "build_revision": runtime.build_revision,
            "runtime_root": str(runtime.runtime_root),
        },
        "provisioning": _provision_report(
            runtime,
            suites,
            routing_dataset_id=routing_dataset_id,
        ),
        "checkpoint": {
            "campaign_schema": state.get("schema_version"),
            "phase_progress": {
                phase: phase_progress(state, phase=phase)
                for phase in state.get("phases", {})
                if isinstance(state.get("phases", {}).get(phase), Mapping)
            },
            "campaign_binding_sha256": state.get("binding_sha256"),
        },
        "resilience": {
            "policy_sha256": RESILIENCE_POLICY_SHA256,
            "policy": RESILIENCE_POLICY,
            "case_retry_count": sum(
                int(record.get("retry_count", 0))
                for phase in state.get("phases", {}).values()
                if isinstance(phase, Mapping)
                for record in phase.values()
                if isinstance(record, Mapping)
            ),
            "retried_case_count": sum(
                int(int(record.get("retry_count", 0)) > 0)
                for phase in state.get("phases", {}).values()
                if isinstance(phase, Mapping)
                for record in phase.values()
                if isinstance(record, Mapping)
            ),
        },
        "quality": {
            "public_answer_refusal": _public_report(_state_observations(state, "public_answer_refusal")),
            "routing": _routing_report(
                _state_observations(state, "routing_qualification"),
                _state_observations(state, "routing_agent_answers"),
            ),
            "enterprise": _enterprise_report(
                _state_observations(state, "enterprise_extraction"),
                _state_observations(state, "enterprise_answering"),
            ),
            "graph_rag": _graph_report(_state_observations(state, "graph_answering")),
        },
        "usage": _usage_report(state),
        "notes": [
            "Metrics named lexical_answer_match are conservative in-memory string checks, not an LLM judge.",
            "Public frozen duplicate-content files are retained in the corpus binding and represented by content aliases in the serving KB.",
            "Graph-RAG quality denominator is the 24 supported answer cases; negative controls remain in the frozen corpus contract and are not silently mixed into the positive answer denominator.",
        ],
    }


def main() -> int:
    arguments = _parser().parse_args()
    try:
        result = asyncio.run(_campaign(arguments))
    except (
        EvaluationRuntimeError,
        LargeEvaluationRuntimeError,
        LargeEvaluationError,
        R4RunnerError,
        ChatModelExecutionError,
        ChatPipelineExecutionError,
        CampaignStateError,
        RetrievalExecutionError,
        OSError,
        ValueError,
    ) as error:
        failure_code = stable_error_code(error)
        try:
            if arguments.campaign_checkpoint.is_file():
                state = json.loads(
                    arguments.campaign_checkpoint.read_text(encoding="utf-8")
                )
                if isinstance(state, dict) and state.get("schema_version") == CAMPAIGN_SCHEMA_VERSION:
                    state["status"] = "blocked"
                    state["last_failure"] = {
                        "failure_code": str(failure_code),
                        "type": type(error).__name__,
                        "at": datetime.now(UTC).isoformat(),
                    }
                    _write_state(arguments.campaign_checkpoint, state)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            # Preserve the original stable failure code if the checkpoint
            # itself cannot be updated.
            pass
        print(json.dumps({"status": "blocked", "failure_code": failure_code}, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
