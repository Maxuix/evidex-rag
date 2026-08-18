#!/usr/bin/env python3
"""Offline contracts and diagnostics for the adaptive Graph RAG evaluation.

This module deliberately has no HTTP, database, embedding, Graphiti, or Judge
provider execution path.  Provider execution belongs to the separately
authorized R7 harness.  The local functions here are used by R1/R2 fake and
unit tests to freeze the evaluator contract without changing production wire
schemas or ChatRun snapshots.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
from typing import Any
from uuid import UUID

from rag_kb.domain import ChatModelMessage, ChatModelRequest
from rag_kb.ports.model_api import ChatModelAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "evaluation" / "adaptive-graph-route-v1" / "manifest.json"
ROUTE_IDS = ("vector-only", "hybrid-control", "manual-graph", "auto-route")
LAYERS = ("raw", "hydrated", "reranked", "packed")
CASE_OUTCOMES = frozenset({"answered", "refused"})
NEGATIVE_CONTROL_KINDS = frozenset(
    {"contradicted", "closed_world_absence", "open_world_unanswerable"}
)
ROUTING_JUDGE_SCHEMA_VERSION = "routing_rag_v1_judge_v1"
ROUTING_JUDGE_PROMPT_VERSION = "routing_rag_v1_judge_prompt_v1"
REPLAY_CAPTURE_SCHEMA_VERSION = "adaptive_graph_replay_capture_v1"
REPLAY_CAPTURE_SOURCE = "r2_clean_forced"
REPLAY_QUERY_MAX_CHARS = 2048
FORCED_CONTROLLER_MODES = frozenset(
    {"specific_tool_choice", "single_tool_required_fallback", "actual_auto"}
)
JUDGE_VERDICTS = frozenset({"correct", "partial", "incorrect"})
JUDGE_GROUNDING = frozenset({"supported", "partial", "unsupported"})
JUDGE_STANCES = frozenset({"affirmed", "denied", "abstained", "not_applicable"})
JUDGE_REASON_CODES = frozenset(
    {
        "answer_supported",
        "answer_partially_supported",
        "answer_incorrect",
        "refusal_correct",
        "refusal_incorrect",
        "citation_missing",
        "citation_misaligned",
    }
)
_TERM_PUNCTUATION = str.maketrans(
    {
        "（": "(",
        "）": ")",
        "［": "[",
        "］": "]",
        "｛": "{",
        "｝": "}",
        "，": ",",
        "。": ".",
        "：": ":",
        "；": ";",
        "！": "!",
        "？": "?",
        "、": ",",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
)


def normalize_term(value: str) -> str:
    """Normalize benchmark proxy terms without making them a Judge."""

    if not isinstance(value, str):
        raise TypeError("term must be a string")
    normalized = unicodedata.normalize("NFKC", value).translate(_TERM_PUNCTUATION)
    return re.sub(r"\s+", "", normalized)


def term_proxy(
    answer: str,
    expected_terms: Iterable[str],
) -> dict[str, Any]:
    normalized_answer = normalize_term(answer)
    normalized_terms = tuple(dict.fromkeys(normalize_term(item) for item in expected_terms))
    matched = tuple(item for item in normalized_terms if item and item in normalized_answer)
    return {
        "matched": len(matched),
        "total": len(normalized_terms),
        "all_matched": bool(normalized_terms) and len(matched) == len(normalized_terms),
        "empty_expected_terms": not normalized_terms,
    }


def validate_case_contract(cases: Sequence[Mapping[str, Any]]) -> None:
    """Validate the content contract independently of generated files."""

    seen: set[str] = set()
    negative_kinds: set[str] = set()
    for raw_case in cases:
        case_id = str(raw_case.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError(f"duplicate or missing case_id: {case_id}")
        seen.add(case_id)
        outcome = raw_case.get("expected_outcome")
        if outcome not in CASE_OUTCOMES:
            raise ValueError(f"{case_id}: invalid expected_outcome")
        answerable = raw_case.get("answerable")
        if not isinstance(answerable, bool):
            raise ValueError(f"{case_id}: answerable must be bool")
        if answerable != (outcome == "answered"):
            raise ValueError(f"{case_id}: answerable/outcome mismatch")
        aspects = raw_case.get("expected_answer_aspects")
        if not isinstance(aspects, list):
            raise ValueError(f"{case_id}: expected_answer_aspects must be a list")
        if outcome == "answered" and not aspects:
            raise ValueError(f"{case_id}: answered case has no aspects")
        for aspect in aspects:
            if not isinstance(aspect, Mapping) or not str(aspect.get("aspect_id", "")):
                raise ValueError(f"{case_id}: invalid answer aspect")
            variants = aspect.get("answer_variants")
            if not isinstance(variants, list) or not variants or any(
                not isinstance(item, str) or not item.strip() for item in variants
            ):
                raise ValueError(f"{case_id}: answer aspect variants are invalid")
        locators = raw_case.get("answer_gold_source_locators")
        if not isinstance(locators, list):
            raise ValueError(f"{case_id}: answer_gold_source_locators must be a list")
        if outcome == "answered" and not locators:
            raise ValueError(f"{case_id}: answered case has no answer gold locator")
        if not isinstance(raw_case.get("path_context_locators"), list):
            raise ValueError(f"{case_id}: path_context_locators must be a list")
        if not isinstance(raw_case.get("forbidden_claims"), list):
            raise ValueError(f"{case_id}: forbidden_claims must be a list")
        kind = raw_case.get("negative_control_kind")
        if raw_case.get("category") == "negative_control":
            if kind not in NEGATIVE_CONTROL_KINDS:
                raise ValueError(f"{case_id}: invalid negative control kind")
            negative_kinds.add(str(kind))
            if kind == "contradicted" and outcome != "answered":
                raise ValueError(f"{case_id}: contradicted control must be answered")
            if kind == "open_world_unanswerable" and outcome != "refused":
                raise ValueError(f"{case_id}: open-world control must be refused")
        elif kind is not None:
            raise ValueError(f"{case_id}: non-negative case has negative kind")
        if raw_case.get("expected_route", {}).get("route") == "graph":
            source = raw_case.get("source")
            if not isinstance(source, Mapping):
                raise ValueError(f"{case_id}: graph case source is invalid")
            answer_relation_ids = {
                str(item) for item in source.get("answer_relation_ids", ())
            }
            locator_ids = {
                str(item.get("relation_id"))
                for item in locators
                if isinstance(item, Mapping) and item.get("kind") == "graph_relation"
            }
            if locator_ids != answer_relation_ids:
                raise ValueError(f"{case_id}: answer locator set differs from answer relations")
            context_ids = {
                str(item.get("relation_id"))
                for item in raw_case["path_context_locators"]
                if isinstance(item, Mapping) and item.get("kind") == "graph_relation"
            }
            if locator_ids & context_ids:
                raise ValueError(f"{case_id}: answer/path locators overlap")
    if negative_kinds != set(NEGATIVE_CONTROL_KINDS):
        raise ValueError("negative controls must cover all three semantic kinds")


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_case_contract(cases)
    return cases


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "adaptive_graph_route_manifest_v2":
        raise ValueError("adaptive route manifest schema mismatch")
    if tuple(item.get("id") for item in manifest.get("routes", ())) != ROUTE_IDS:
        raise ValueError("adaptive route manifest lanes are not frozen")
    case_file = (path.parent / str(manifest.get("case_file", ""))).resolve()
    cases = load_cases(case_file)
    if manifest.get("case_count") != len(cases):
        raise ValueError("adaptive route manifest case count mismatch")
    manifest["case_file"] = str(case_file)
    manifest["case_ids"] = [str(item["case_id"]) for item in cases]
    return manifest


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    import hashlib

    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_routing_judge_packet(
    case: Mapping[str, Any],
    run: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a bounded, routing-specific offline Judge packet."""

    case_id = str(case.get("case_id", ""))
    if not case_id or case.get("expected_outcome") not in CASE_OUTCOMES:
        raise ValueError("Judge case contract is invalid")
    aspects = case.get("expected_answer_aspects")
    if not isinstance(aspects, list):
        raise ValueError("Judge case aspects are invalid")
    citations = run.get("citations", ())
    if not isinstance(citations, Sequence) or isinstance(citations, (str, bytes)):
        citations = ()
    safe_citations: list[dict[str, Any]] = []
    for citation in citations:
        if not isinstance(citation, Mapping):
            raise ValueError("Judge citation is invalid")
        safe_citations.append(
            {
                "citation_id": str(citation.get("citation_id", "")),
                "index_chunk_id": str(citation.get("index_chunk_id", "")),
                "document_id": str(citation.get("document_id", "")),
                "source_location": citation.get("source_location", {}),
                "modality": citation.get("modality"),
            }
        )
    return {
        "schema_version": ROUTING_JUDGE_SCHEMA_VERSION,
        "prompt_version": ROUTING_JUDGE_PROMPT_VERSION,
        "case_id": case_id,
        "question": str(case["question"]),
        "reference": {
            "expected_outcome": case["expected_outcome"],
            "negative_control_kind": case.get("negative_control_kind"),
            "expected_answer_aspects": [
                {
                    "aspect_id": str(aspect["aspect_id"]),
                    "answer_variants": [str(item) for item in aspect["answer_variants"]],
                }
                for aspect in aspects
            ],
            "answer_gold_source_locators": case["answer_gold_source_locators"],
            "forbidden_claims": [str(item) for item in case["forbidden_claims"]],
        },
        "agent_result": {
            "outcome": str(run.get("outcome", "")),
            "answer": str(run.get("answer", "")),
            "citations": safe_citations,
        },
    }


def routing_judge_cache_key(
    packet: Mapping[str, Any],
    *,
    profile_revision: str = "offline-routing-judge-v1",
) -> str:
    import hashlib

    value = {
        "schema_version": ROUTING_JUDGE_SCHEMA_VERSION,
        "prompt_version": ROUTING_JUDGE_PROMPT_VERSION,
        "profile_revision": profile_revision,
        "packet": packet,
    }
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_routing_judgement(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "prompt_version",
        "answer_correctness",
        "claim_grounding",
        "citation_alignment",
        "outcome_correctness",
        "negative_stance",
        "reason_code",
    }
    if set(value) != expected:
        raise ValueError("routing Judge result fields are invalid")
    if value["schema_version"] != ROUTING_JUDGE_SCHEMA_VERSION:
        raise ValueError("routing Judge schema version is invalid")
    if value["prompt_version"] != ROUTING_JUDGE_PROMPT_VERSION:
        raise ValueError("routing Judge prompt version is invalid")
    if value["answer_correctness"] not in JUDGE_VERDICTS:
        raise ValueError("routing Judge answer correctness is invalid")
    if value["claim_grounding"] not in JUDGE_GROUNDING:
        raise ValueError("routing Judge grounding is invalid")
    if value["citation_alignment"] not in JUDGE_GROUNDING:
        raise ValueError("routing Judge Citation alignment is invalid")
    if value["outcome_correctness"] not in {"correct", "incorrect"}:
        raise ValueError("routing Judge outcome correctness is invalid")
    if value["negative_stance"] not in JUDGE_STANCES:
        raise ValueError("routing Judge negative stance is invalid")
    if value["reason_code"] not in JUDGE_REASON_CODES:
        raise ValueError("routing Judge reason code is invalid")


def _successful_simple_result(messages: Sequence[ChatModelMessage]) -> bool:
    for message in reversed(messages):
        if message.role != "tool":
            continue
        try:
            payload = json.loads(message.content)
        except (TypeError, ValueError):
            return False
        return isinstance(payload, Mapping) and payload.get("status") == "ok"
    return False


@dataclass
class ForcedGraphitiSupplementChatModelPort:
    """Evaluator-only decorator that overrides exactly one next tool choice."""

    delegate: ChatModelAdapter
    controller_mode: str = "specific_tool_choice"
    replacements: int = 0
    model_calls: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    response_tool_names: list[tuple[str, ...]] = field(default_factory=list)
    response_finish_reasons: list[str | None] = field(default_factory=list)
    _used: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.controller_mode not in FORCED_CONTROLLER_MODES:
            raise ValueError("Forced controller mode is invalid")

    def reset_case(self) -> None:
        self.replacements = 0
        self.model_calls = 0
        self.usage.clear()
        self.response_tool_names.clear()
        self.response_finish_reasons.clear()
        self._used = False

    async def complete(self, request: ChatModelRequest):
        effective = request
        tool_names = {item.name for item in request.tools}
        if (
            self.controller_mode != "actual_auto"
            and not self._used
            and "graphiti_supplement" in tool_names
            and _successful_simple_result(request.messages)
        ):
            if self.controller_mode == "single_tool_required_fallback":
                supplement_tools = tuple(
                    item
                    for item in request.tools
                    if item.name == "graphiti_supplement"
                )
                if len(supplement_tools) != 1:
                    raise RuntimeError("Forced supplement tool cardinality is invalid")
                effective = replace(
                    request,
                    tools=supplement_tools,
                    tool_choice="required",
                )
            else:
                effective = replace(request, tool_choice="graphiti_supplement")
            self._used = True
            self.replacements += 1
        response = await self.delegate.complete(effective)
        self.model_calls += 1
        self.response_tool_names.append(
            tuple(tool_call.name for tool_call in response.tool_calls)
        )
        self.response_finish_reasons.append(response.finish_reason)
        for key, value in response.usage.items():
            self.usage[key] = self.usage.get(key, 0) + value
        return response


class GraphitiSupplementCaptureComplete(RuntimeError):
    """Content-safe evaluator stop after one validated supplement invocation."""


@dataclass(frozen=True, slots=True)
class GraphitiSupplementCapture:
    """One post-validation supplement call captured outside production trace."""

    case_id: str
    query: str = field(repr=False)
    excluded_index_chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        normalized_case_id = self.case_id.strip()
        normalized_query = self.query.strip()
        if (
            not normalized_case_id
            or len(normalized_case_id) > 128
            or not normalized_query
            or len(normalized_query) > REPLAY_QUERY_MAX_CHARS
        ):
            raise ValueError("supplement capture is invalid")
        try:
            normalized_ids = tuple(
                str(UUID(str(item))) for item in self.excluded_index_chunk_ids
            )
        except (TypeError, ValueError, AttributeError) as error:
            raise ValueError("supplement capture exclusions are invalid") from error
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("supplement capture exclusions must be unique")
        object.__setattr__(self, "case_id", normalized_case_id)
        object.__setattr__(self, "query", normalized_query)
        object.__setattr__(self, "excluded_index_chunk_ids", normalized_ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "query_sha256": hashlib.sha256(self.query.encode("utf-8")).hexdigest(),
            "excluded_index_chunk_ids": list(self.excluded_index_chunk_ids),
        }


@dataclass
class CapturingGraphitiSupplementRetriever:
    """Evaluator-only retriever wrapper capturing the accepted Agent query."""

    delegate: Any
    stop_after_capture: bool = False
    _active_case_id: str | None = field(default=None, init=False, repr=False)
    _active_capture: GraphitiSupplementCapture | None = field(
        default=None, init=False, repr=False
    )
    _simple_index_chunk_ids: list[str] = field(
        default_factory=list, init=False, repr=False
    )

    def begin_case(self, case_id: str) -> None:
        if self._active_case_id is not None:
            raise RuntimeError("supplement capture case is already active")
        normalized = case_id.strip()
        if not normalized or len(normalized) > 128:
            raise ValueError("supplement capture case_id is invalid")
        self._active_case_id = normalized
        self._active_capture = None
        self._simple_index_chunk_ids.clear()

    def finish_case(self) -> GraphitiSupplementCapture:
        capture, _ = self.finish_observed_case()
        if capture is None:
            raise RuntimeError("supplement capture case is incomplete")
        return capture

    def finish_observed_case(
        self,
    ) -> tuple[GraphitiSupplementCapture | None, tuple[str, ...]]:
        if self._active_case_id is None:
            raise RuntimeError("supplement capture case is not active")
        capture = self._active_capture
        simple_ids = tuple(self._simple_index_chunk_ids)
        self._active_case_id = None
        self._active_capture = None
        self._simple_index_chunk_ids.clear()
        return capture, simple_ids

    def abandon_case(self) -> None:
        self._active_case_id = None
        self._active_capture = None
        self._simple_index_chunk_ids.clear()

    async def retrieve_query(self, *args, **kwargs):
        result = await self.delegate.retrieve_query(*args, **kwargs)
        for item in getattr(result, "evidence", ()):
            chunk_id = str(item.index_chunk_id)
            if chunk_id not in self._simple_index_chunk_ids:
                self._simple_index_chunk_ids.append(chunk_id)
        return result

    async def retrieve_graphiti_supplement(
        self,
        context,
        query: str,
        *,
        excluded_index_chunk_ids,
    ):
        if self._active_case_id is None:
            raise RuntimeError("supplement capture has no active case")
        if self._active_capture is not None:
            raise RuntimeError("supplement capture received more than one call")
        capture = GraphitiSupplementCapture(
            case_id=self._active_case_id,
            query=query,
            excluded_index_chunk_ids=tuple(
                str(item) for item in excluded_index_chunk_ids
            ),
        )
        self._active_capture = capture
        if self.stop_after_capture:
            raise GraphitiSupplementCaptureComplete("graphiti_supplement_captured")
        return await self.delegate.retrieve_graphiti_supplement(
            context,
            capture.query,
            excluded_index_chunk_ids=tuple(
                UUID(item) for item in capture.excluded_index_chunk_ids
            ),
        )


def build_replay_capture_artifact(
    *,
    manifest_sha256: str,
    knowledge_base_id: str,
    index_revision_id: str,
    graph_build_id: str,
    chat_model_profile_revision_id: str,
    captures: Sequence[GraphitiSupplementCapture],
    controller_mode: str = "specific_tool_choice",
    chat_model: str | None = None,
    chat_model_source: str | None = None,
    chat_model_max_output_tokens: int | None = None,
    chat_model_max_retries: int | None = None,
) -> dict[str, Any]:
    """Build a bounded synthetic-corpus replay artifact with no provider secrets."""

    identifiers = {
        "knowledge_base_id": knowledge_base_id,
        "index_revision_id": index_revision_id,
        "graph_build_id": graph_build_id,
        "chat_model_profile_revision_id": chat_model_profile_revision_id,
    }
    try:
        normalized_identifiers = {
            key: str(UUID(str(value))) for key, value in identifiers.items()
        }
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("replay capture runtime identity is invalid") from error
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256):
        raise ValueError("replay capture manifest digest is invalid")
    if controller_mode not in FORCED_CONTROLLER_MODES:
        raise ValueError("replay capture controller mode is invalid")
    case_ids = [item.case_id for item in captures]
    if (
        len(case_ids) != len(set(case_ids))
        or not captures
        and controller_mode != "actual_auto"
    ):
        raise ValueError("replay captures must contain unique cases")
    capture_source = (
        "r3a_actual_auto" if controller_mode == "actual_auto" else REPLAY_CAPTURE_SOURCE
    )
    if (chat_model is None) != (chat_model_source is None):
        raise ValueError("replay capture chat model identity is incomplete")
    if chat_model is not None:
        if not chat_model.strip() or chat_model_source not in {
            "profile_revision",
            "evaluator_override",
        }:
            raise ValueError("replay capture chat model identity is invalid")
        if chat_model_max_output_tokens is None or chat_model_max_output_tokens < 1:
            raise ValueError("replay capture chat model token limit is invalid")
        if chat_model_max_retries is None or chat_model_max_retries < 0:
            raise ValueError("replay capture chat model retry limit is invalid")
        normalized_identifiers.update(
            {
                "chat_model": chat_model.strip(),
                "chat_model_source": chat_model_source,
                "chat_model_max_output_tokens": chat_model_max_output_tokens,
                "chat_model_max_retries": chat_model_max_retries,
            }
        )
    return {
        "schema_version": REPLAY_CAPTURE_SCHEMA_VERSION,
        "dataset_id": "routing-rag-v1",
        "capture_source": capture_source,
        "controller_mode": controller_mode,
        "manifest_sha256": manifest_sha256,
        "runtime": normalized_identifiers,
        "case_count": len(captures),
        "cases": [item.as_dict() for item in captures],
    }


def write_replay_capture_artifact(path: Path, artifact: Mapping[str, Any]) -> str:
    """Write one immutable, owner-readable evaluator artifact and return its digest."""

    if artifact.get("schema_version") != REPLAY_CAPTURE_SCHEMA_VERSION:
        raise ValueError("replay capture artifact schema is invalid")
    payload = json.dumps(
        artifact,
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


@dataclass(frozen=True)
class ChunkAlignment:
    column: str
    answer_gold_chunk_ids: tuple[str, ...]
    simple_chunk_ids: tuple[str, ...]
    layer_chunk_ids: dict[str, tuple[str, ...]]
    first_loss_layer: str | None
    new_answer_gold_chunk_ids: tuple[str, ...]
    redundant_hit: bool
    benefit: bool
    duplicate_count: int
    non_gold_admitted_count: int


def align_chunk_layers(
    *,
    column: str,
    answer_gold_chunk_ids: Iterable[str],
    simple_chunk_ids: Iterable[str],
    layer_chunk_ids: Mapping[str, Iterable[str]],
    path_context_chunk_ids: Iterable[str] = (),
) -> ChunkAlignment:
    if column not in {"capability", "agent_replay"}:
        raise ValueError("diagnostic column is invalid")
    gold = tuple(dict.fromkeys(str(item) for item in answer_gold_chunk_ids))
    simple = tuple(dict.fromkeys(str(item) for item in simple_chunk_ids))
    layers: dict[str, tuple[str, ...]] = {}
    for layer in LAYERS:
        if layer not in layer_chunk_ids:
            raise ValueError(f"missing diagnostic layer: {layer}")
        layers[layer] = tuple(str(item) for item in layer_chunk_ids[layer])
    first_loss: str | None = None
    for gold_id in gold:
        if gold_id in simple:
            continue
        for layer in LAYERS:
            if gold_id not in layers[layer]:
                first_loss = layer
                break
        if first_loss is not None:
            break
    packed = layers["packed"]
    new_gold = tuple(item for item in gold if item not in simple and item in packed)
    graph_new = tuple(item for item in packed if item not in simple)
    context_ids = set(str(item) for item in path_context_chunk_ids)
    answer_already_in_simple = bool(gold) and set(gold) <= set(simple)
    redundant = (
        answer_already_in_simple
        or bool(graph_new) and not new_gold
        or bool(graph_new) and set(graph_new) <= context_ids
    )
    return ChunkAlignment(
        column=column,
        answer_gold_chunk_ids=gold,
        simple_chunk_ids=simple,
        layer_chunk_ids=layers,
        first_loss_layer=first_loss,
        new_answer_gold_chunk_ids=new_gold,
        redundant_hit=redundant,
        benefit=bool(new_gold),
        duplicate_count=len(packed) - len(set(packed)),
        non_gold_admitted_count=sum(item not in set(gold) for item in packed),
    )


def diagnostic_record(
    *,
    case_id: str,
    alignments: Mapping[str, ChunkAlignment],
    query_source: str,
    query_count: int,
) -> dict[str, Any]:
    if set(alignments) != {"capability", "agent_replay"}:
        raise ValueError("diagnostic record requires both columns")
    if query_source not in {"capability", "agent_replay"} or query_count < 0:
        raise ValueError("diagnostic query metadata is invalid")
    return {
        "case_id": case_id,
        "columns": {
            name: {
                "first_loss_layer": value.first_loss_layer,
                "answer_gold_chunk_ids": list(value.answer_gold_chunk_ids),
                "simple_chunk_ids": list(value.simple_chunk_ids),
                "layers": {layer: list(ids) for layer, ids in value.layer_chunk_ids.items()},
                "new_answer_gold_chunk_ids": list(value.new_answer_gold_chunk_ids),
                "redundant_hit": value.redundant_hit,
                "benefit": value.benefit,
                "duplicate_count": value.duplicate_count,
                "non_gold_admitted_count": value.non_gold_admitted_count,
            }
            for name, value in alignments.items()
        },
        "query_source": query_source,
        "query_count": query_count,
    }


def aggregate_usage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate safe numeric usage only; unknown pricing remains uncomputed."""

    fields = (
        "total_tokens",
        "model_rounds",
        "retrieval_queries",
        "evidence_refs",
        "rejected_tools",
    )
    totals = {field_name: 0 for field_name in fields}
    elapsed: list[float] = []
    for record in records:
        for field_name in fields:
            value = record.get(field_name, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"usage field is invalid: {field_name}")
            totals[field_name] += value
        value = record.get("elapsed_seconds", 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError("elapsed_seconds is invalid")
        elapsed.append(float(value))
    ordered = sorted(elapsed)
    return {
        "case_count": len(records),
        "totals": totals,
        "elapsed_seconds": {
            "p50": _percentile(ordered, 0.50),
            "p95": _percentile(ordered, 0.95),
        },
        "cost": {"status": "not_computed", "reason": "price_table_not_frozen"},
    }


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 6)
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * fraction))))
    return round(values[index], 6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    try:
        manifest = load_manifest(arguments.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if not arguments.dry_run:
        parser.error(
            "provider/database execution is disabled; use --dry-run or the separately authorized R7 harness"
        )
    print(
        json.dumps(
            {
                "status": "dry_run_ok",
                "dataset_id": manifest["dataset_id"],
                "case_count": manifest["case_count"],
                "route_ids": list(ROUTE_IDS),
                "manifest_digest": manifest_digest(manifest),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
