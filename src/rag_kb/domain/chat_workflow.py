"""Bounded workflow, routing, and research facts for one ChatRun."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


CHAT_WORKFLOW_VERSION = "chat_workflow_v1"
RESEARCH_RESULT_VERSION = "research_result_v1"
SEARCH_TRACE_VERSION = "search_trace_v1"


class ChatWorkflowMode(StrEnum):
    SIMPLE = "simple"
    AGENT = "agent"
    AUTO = "auto"


class ChatResolvedMode(StrEnum):
    PENDING = "pending"
    SIMPLE = "simple"
    AGENT = "agent"


class ChatRouteStatus(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"
    RESOLVED = "resolved"
    FALLBACK = "fallback"


class ChatRouteReason(StrEnum):
    SINGLE_LOOKUP = "single_lookup"
    DIRECT_SUMMARY = "direct_summary"
    MULTI_VIEW_REQUIRED = "multi_view_required"
    MULTI_HOP_REQUIRED = "multi_hop_required"
    EVIDENCE_UNCERTAIN = "evidence_uncertain"
    ROUTER_INVALID = "router_invalid"
    ROUTER_UNAVAILABLE = "router_unavailable"


class ResearchStatus(StrEnum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    NO_EVIDENCE = "no_evidence"
    CONFLICT = "conflict"
    PREMISE_UNSUPPORTED = "premise_unsupported"


class ResearchAspectStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    MISSING = "missing"
    CONFLICT = "conflict"


class ResearchTerminationReason(StrEnum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    NO_EVIDENCE = "no_evidence"
    NO_PROGRESS = "no_progress"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CONFLICT_UNRESOLVED = "conflict_unresolved"
    PREMISE_UNSUPPORTED = "premise_unsupported"


@dataclass(frozen=True, slots=True)
class ChatWorkflowBudget:
    decision_rounds: int = 4
    retrieval_calls: int = 6
    parallel_queries: int = 3
    verifier_continuations: int = 1
    no_progress_rounds: int = 1

    def __post_init__(self) -> None:
        if (
            not 1 <= self.decision_rounds <= 8
            or not 1 <= self.retrieval_calls <= 12
            or not 1 <= self.parallel_queries <= 3
            or not 0 <= self.verifier_continuations <= 1
            or self.no_progress_rounds != 1
        ):
            raise ValueError("chat workflow budget is invalid")

    def as_dict(self) -> dict[str, int]:
        return {
            "decision_rounds": self.decision_rounds,
            "retrieval_calls": self.retrieval_calls,
            "parallel_queries": self.parallel_queries,
            "verifier_continuations": self.verifier_continuations,
            "no_progress_rounds": self.no_progress_rounds,
        }


@dataclass(frozen=True, slots=True)
class ChatWorkflowConfiguration:
    requested_mode: ChatWorkflowMode
    budget: ChatWorkflowBudget = ChatWorkflowBudget()
    version: str = CHAT_WORKFLOW_VERSION

    def __post_init__(self) -> None:
        if self.version != CHAT_WORKFLOW_VERSION:
            raise ValueError("chat workflow configuration version is unsupported")

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "requested_mode": self.requested_mode.value,
            "budget": self.budget.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class ResearchAspect:
    aspect: str
    status: ResearchAspectStatus
    evidence_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _bounded_unique_strings((self.aspect,), maximum=1, field="research aspect")
        _bounded_unique_strings(
            self.evidence_keys, maximum=100, field="research aspect evidence"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "aspect": self.aspect,
            "status": self.status.value,
            "evidence_keys": list(self.evidence_keys),
        }


@dataclass(frozen=True, slots=True)
class ResearchResult:
    status: ResearchStatus
    selected_evidence_keys: tuple[str, ...]
    aspects: tuple[ResearchAspect, ...]
    covered_aspects: tuple[str, ...]
    missing_aspects: tuple[str, ...]
    conflicts: tuple[str, ...]
    termination_reason: ResearchTerminationReason
    version: str = RESEARCH_RESULT_VERSION
    scope_status: str = "all"
    resolved_document_count: int = 0
    complete_scan_document_count: int = 0
    scope_rejection_count: int = 0
    scope_downgrade_reason: str | None = None

    def __post_init__(self) -> None:
        if self.version != RESEARCH_RESULT_VERSION:
            raise ValueError("research result version is unsupported")
        _bounded_unique_strings(
            self.selected_evidence_keys,
            maximum=100,
            field="selected research evidence",
        )
        for field, values in (
            ("covered research aspects", self.covered_aspects),
            ("missing research aspects", self.missing_aspects),
            ("research conflicts", self.conflicts),
        ):
            _bounded_unique_strings(values, maximum=100, field=field)
        if len(self.aspects) > 100:
            raise ValueError("research result has too many aspects")
        allowed = set(self.selected_evidence_keys)
        if any(not set(item.evidence_keys) <= allowed for item in self.aspects):
            raise ValueError("research aspects reference unselected evidence")
        if self.status is ResearchStatus.SUFFICIENT and (
            not self.selected_evidence_keys
            or self.missing_aspects
            or self.conflicts
        ):
            raise ValueError("sufficient research result is inconsistent")
        if self.status is ResearchStatus.NO_EVIDENCE and self.selected_evidence_keys:
            raise ValueError("no-evidence research result selected evidence")
        if self.status is ResearchStatus.PARTIAL and (
            not self.selected_evidence_keys or not self.missing_aspects
        ):
            raise ValueError("partial research result is inconsistent")
        if self.status is ResearchStatus.CONFLICT and (
            not self.selected_evidence_keys or not self.conflicts
        ):
            raise ValueError("conflict research result is inconsistent")
        if (
            self.status is ResearchStatus.PREMISE_UNSUPPORTED
            and not self.selected_evidence_keys
        ):
            raise ValueError("unsupported premise requires contradictory evidence")
        _validate_scope_facts(
            self.scope_status,
            self.resolved_document_count,
            self.complete_scan_document_count,
            self.scope_rejection_count,
            self.scope_downgrade_reason,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status.value,
            "selected_evidence_keys": list(self.selected_evidence_keys),
            "aspects": [item.as_dict() for item in self.aspects],
            "covered_aspects": list(self.covered_aspects),
            "missing_aspects": list(self.missing_aspects),
            "conflicts": list(self.conflicts),
            "termination_reason": self.termination_reason.value,
            "scope_status": self.scope_status,
            "resolved_document_count": self.resolved_document_count,
            "complete_scan_document_count": self.complete_scan_document_count,
            "scope_rejection_count": self.scope_rejection_count,
            "scope_downgrade_reason": self.scope_downgrade_reason,
        }


@dataclass(frozen=True, slots=True)
class SearchTraceStep:
    observation_id: str
    objective: str
    queries: tuple[str, ...]
    based_on_observation_ids: tuple[str, ...]
    result: str
    new_evidence_count: int

    def __post_init__(self) -> None:
        _bounded_unique_strings(
            (self.observation_id,), maximum=1, field="observation identifier"
        )
        _bounded_unique_strings((self.objective,), maximum=1, field="search objective")
        _bounded_unique_strings(self.queries, maximum=3, field="search queries")
        _bounded_unique_strings(
            self.based_on_observation_ids,
            maximum=24,
            field="observation references",
        )
        if self.result not in {"evidence_found", "no_evidence", "verification_gap"}:
            raise ValueError("search trace result is unsupported")
        if not 0 <= self.new_evidence_count <= 100:
            raise ValueError("search trace evidence count is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "objective": self.objective,
            "queries": list(self.queries),
            "based_on_observation_ids": list(self.based_on_observation_ids),
            "result": self.result,
            "new_evidence_count": self.new_evidence_count,
        }


@dataclass(frozen=True, slots=True)
class SearchTrace:
    steps: tuple[SearchTraceStep, ...]
    decision_rounds: int
    retrieval_calls: int
    verifier_calls: int
    evidence_count: int
    adjacency_loaded_count: int = 0
    adjacency_selected_count: int = 0
    version: str = SEARCH_TRACE_VERSION
    scope_status: str = "all"
    resolved_document_count: int = 0
    complete_scan_document_count: int = 0
    scope_rejection_count: int = 0
    scope_downgrade_reason: str | None = None

    def __post_init__(self) -> None:
        if self.version != SEARCH_TRACE_VERSION:
            raise ValueError("search trace version is unsupported")
        if len(self.steps) > 12:
            raise ValueError("search trace has too many steps")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                self.decision_rounds,
                self.retrieval_calls,
                self.verifier_calls,
                self.evidence_count,
                self.adjacency_loaded_count,
                self.adjacency_selected_count,
            )
        ):
            raise ValueError("search trace counters are invalid")
        if self.adjacency_selected_count > self.adjacency_loaded_count:
            raise ValueError("selected adjacency count exceeds loaded count")
        _validate_scope_facts(
            self.scope_status,
            self.resolved_document_count,
            self.complete_scan_document_count,
            self.scope_rejection_count,
            self.scope_downgrade_reason,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "steps": [item.as_dict() for item in self.steps],
            "decision_rounds": self.decision_rounds,
            "retrieval_calls": self.retrieval_calls,
            "verifier_calls": self.verifier_calls,
            "evidence_count": self.evidence_count,
            "adjacency_loaded_count": self.adjacency_loaded_count,
            "adjacency_selected_count": self.adjacency_selected_count,
            "scope_status": self.scope_status,
            "resolved_document_count": self.resolved_document_count,
            "complete_scan_document_count": self.complete_scan_document_count,
            "scope_rejection_count": self.scope_rejection_count,
            "scope_downgrade_reason": self.scope_downgrade_reason,
        }


@dataclass(frozen=True, slots=True)
class ChatWorkflowState:
    resolved_mode: ChatResolvedMode
    route_status: ChatRouteStatus
    route_reason_codes: tuple[ChatRouteReason, ...] = ()
    research_result: ResearchResult | None = None
    search_trace: SearchTrace | None = None
    version: str = CHAT_WORKFLOW_VERSION

    def __post_init__(self) -> None:
        if self.version != CHAT_WORKFLOW_VERSION:
            raise ValueError("chat workflow state version is unsupported")
        if len(self.route_reason_codes) != len(set(self.route_reason_codes)):
            raise ValueError("workflow route reasons must be unique")
        if len(self.route_reason_codes) > 8:
            raise ValueError("workflow route reasons exceed the bound")
        if self.route_status is ChatRouteStatus.PENDING:
            if (
                self.resolved_mode is not ChatResolvedMode.PENDING
                or self.route_reason_codes
            ):
                raise ValueError("pending workflow route is inconsistent")
        elif self.resolved_mode is ChatResolvedMode.PENDING:
            raise ValueError("resolved workflow route cannot remain pending")
        if (
            self.route_status is ChatRouteStatus.NOT_APPLICABLE
            and self.route_reason_codes
        ):
            raise ValueError("explicit workflow cannot contain route reasons")
        if (
            self.route_status is ChatRouteStatus.RESOLVED
            and not self.route_reason_codes
        ):
            raise ValueError("resolved Auto workflow requires a route reason")
        if self.route_status is ChatRouteStatus.FALLBACK and (
            self.resolved_mode is not ChatResolvedMode.SIMPLE
            or len(self.route_reason_codes) != 1
            or self.route_reason_codes[0]
            not in {
                ChatRouteReason.ROUTER_INVALID,
                ChatRouteReason.ROUTER_UNAVAILABLE,
            }
        ):
            raise ValueError("Auto workflow fallback is inconsistent")
        if (self.research_result is None) != (self.search_trace is None):
            raise ValueError("research result and search trace must be stored together")
        if (
            self.research_result is not None
            and self.resolved_mode is not ChatResolvedMode.AGENT
        ):
            raise ValueError("only Agent workflow state may contain research results")

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "resolved_mode": self.resolved_mode.value,
            "route_status": self.route_status.value,
            "route_reason_codes": [item.value for item in self.route_reason_codes],
            "research_result": (
                self.research_result.as_dict()
                if self.research_result is not None
                else None
            ),
            "search_trace": (
                self.search_trace.as_dict() if self.search_trace is not None else None
            ),
        }


def initial_chat_workflow(
    mode: ChatWorkflowMode,
) -> tuple[ChatWorkflowConfiguration, ChatWorkflowState]:
    configuration = ChatWorkflowConfiguration(requested_mode=mode)
    if mode is ChatWorkflowMode.AUTO:
        state = ChatWorkflowState(
            resolved_mode=ChatResolvedMode.PENDING,
            route_status=ChatRouteStatus.PENDING,
        )
    else:
        state = ChatWorkflowState(
            resolved_mode=ChatResolvedMode(mode.value),
            route_status=ChatRouteStatus.NOT_APPLICABLE,
        )
    return configuration, state


def hydrate_chat_workflow_configuration(
    value: Mapping[str, Any],
) -> ChatWorkflowConfiguration:
    if set(value) != {"version", "requested_mode", "budget"}:
        raise ValueError("chat workflow configuration fields are invalid")
    budget = value["budget"]
    if not isinstance(budget, Mapping) or set(budget) != {
        "decision_rounds",
        "retrieval_calls",
        "parallel_queries",
        "verifier_continuations",
        "no_progress_rounds",
    }:
        raise ValueError("chat workflow budget fields are invalid")
    return ChatWorkflowConfiguration(
        version=str(value["version"]),
        requested_mode=ChatWorkflowMode(value["requested_mode"]),
        budget=ChatWorkflowBudget(
            decision_rounds=_strict_int(budget["decision_rounds"]),
            retrieval_calls=_strict_int(budget["retrieval_calls"]),
            parallel_queries=_strict_int(budget["parallel_queries"]),
            verifier_continuations=_strict_int(budget["verifier_continuations"]),
            no_progress_rounds=_strict_int(budget["no_progress_rounds"]),
        ),
    )


def hydrate_chat_workflow_state(
    value: Mapping[str, Any],
) -> ChatWorkflowState:
    if set(value) != {
        "version",
        "resolved_mode",
        "route_status",
        "route_reason_codes",
        "research_result",
        "search_trace",
    }:
        raise ValueError("chat workflow state fields are invalid")
    raw_reasons = value["route_reason_codes"]
    if not isinstance(raw_reasons, (list, tuple)):
        raise ValueError("chat route reasons must be a collection")
    raw_result = value["research_result"]
    raw_trace = value["search_trace"]
    return ChatWorkflowState(
        version=str(value["version"]),
        resolved_mode=ChatResolvedMode(value["resolved_mode"]),
        route_status=ChatRouteStatus(value["route_status"]),
        route_reason_codes=tuple(ChatRouteReason(item) for item in raw_reasons),
        research_result=(
            _hydrate_research_result(raw_result) if raw_result is not None else None
        ),
        search_trace=_hydrate_search_trace(raw_trace) if raw_trace is not None else None,
    )


def _hydrate_research_result(value: Any) -> ResearchResult:
    legacy_fields = {
        "version",
        "status",
        "selected_evidence_keys",
        "aspects",
        "covered_aspects",
        "missing_aspects",
        "conflicts",
        "termination_reason",
    }
    scope_fields = legacy_fields | {
        "scope_status",
        "resolved_document_count",
        "complete_scan_document_count",
        "scope_rejection_count",
        "scope_downgrade_reason",
    }
    if not isinstance(value, Mapping) or set(value) not in (
        legacy_fields,
        scope_fields,
    ):
        raise ValueError("research result fields are invalid")
    aspects = value["aspects"]
    if not isinstance(aspects, (list, tuple)):
        raise ValueError("research aspects must be a collection")
    hydrated_aspects: list[ResearchAspect] = []
    for item in aspects:
        if not isinstance(item, Mapping) or set(item) != {
            "aspect",
            "status",
            "evidence_keys",
        }:
            raise ValueError("research aspect fields are invalid")
        hydrated_aspects.append(
            ResearchAspect(
                aspect=str(item["aspect"]),
                status=ResearchAspectStatus(item["status"]),
                evidence_keys=_string_tuple(item["evidence_keys"]),
            )
        )
    return ResearchResult(
        version=str(value["version"]),
        status=ResearchStatus(value["status"]),
        selected_evidence_keys=_string_tuple(value["selected_evidence_keys"]),
        aspects=tuple(hydrated_aspects),
        covered_aspects=_string_tuple(value["covered_aspects"]),
        missing_aspects=_string_tuple(value["missing_aspects"]),
        conflicts=_string_tuple(value["conflicts"]),
        termination_reason=ResearchTerminationReason(value["termination_reason"]),
        scope_status=str(value.get("scope_status", "all")),
        resolved_document_count=_strict_int(value.get("resolved_document_count", 0)),
        complete_scan_document_count=_strict_int(
            value.get("complete_scan_document_count", 0)
        ),
        scope_rejection_count=_strict_int(value.get("scope_rejection_count", 0)),
        scope_downgrade_reason=(
            str(value["scope_downgrade_reason"])
            if value.get("scope_downgrade_reason") is not None
            else None
        ),
    )


def _hydrate_search_trace(value: Any) -> SearchTrace:
    legacy_fields = {
        "version",
        "steps",
        "decision_rounds",
        "retrieval_calls",
        "verifier_calls",
        "evidence_count",
    }
    current_fields = legacy_fields | {
        "adjacency_loaded_count",
        "adjacency_selected_count",
    }
    scope_fields = current_fields | {
        "scope_status",
        "resolved_document_count",
        "complete_scan_document_count",
        "scope_rejection_count",
        "scope_downgrade_reason",
    }
    scope_legacy_fields = legacy_fields | {
        "scope_status",
        "resolved_document_count",
        "complete_scan_document_count",
        "scope_rejection_count",
        "scope_downgrade_reason",
    }
    if not isinstance(value, Mapping):
        raise ValueError("search trace fields are invalid")
    fields = frozenset(value)
    if fields not in {
        frozenset(legacy_fields),
        frozenset(current_fields),
        frozenset(scope_legacy_fields),
        frozenset(scope_fields),
    }:
        raise ValueError("search trace fields are invalid")
    raw_steps = value["steps"]
    if not isinstance(raw_steps, (list, tuple)):
        raise ValueError("search trace steps must be a collection")
    steps: list[SearchTraceStep] = []
    for item in raw_steps:
        if not isinstance(item, Mapping) or set(item) != {
            "observation_id",
            "objective",
            "queries",
            "based_on_observation_ids",
            "result",
            "new_evidence_count",
        }:
            raise ValueError("search trace step fields are invalid")
        steps.append(
            SearchTraceStep(
                observation_id=str(item["observation_id"]),
                objective=str(item["objective"]),
                queries=_string_tuple(item["queries"]),
                based_on_observation_ids=_string_tuple(
                    item["based_on_observation_ids"]
                ),
                result=str(item["result"]),
                new_evidence_count=_strict_int(item["new_evidence_count"]),
            )
        )
    return SearchTrace(
        version=str(value["version"]),
        steps=tuple(steps),
        decision_rounds=_strict_int(value["decision_rounds"]),
        retrieval_calls=_strict_int(value["retrieval_calls"]),
        verifier_calls=_strict_int(value["verifier_calls"]),
        evidence_count=_strict_int(value["evidence_count"]),
        adjacency_loaded_count=_strict_int(
            value.get("adjacency_loaded_count", 0)
        ),
        adjacency_selected_count=_strict_int(
            value.get("adjacency_selected_count", 0)
        ),
        scope_status=str(value.get("scope_status", "all")),
        resolved_document_count=_strict_int(value.get("resolved_document_count", 0)),
        complete_scan_document_count=_strict_int(
            value.get("complete_scan_document_count", 0)
        ),
        scope_rejection_count=_strict_int(value.get("scope_rejection_count", 0)),
        scope_downgrade_reason=(
            str(value["scope_downgrade_reason"])
            if value.get("scope_downgrade_reason") is not None
            else None
        ),
    )


def _strict_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("workflow counter must be an integer")
    return value


def _validate_scope_facts(
    status: str,
    resolved_count: int,
    complete_scan_count: int,
    rejection_count: int,
    downgrade_reason: str | None,
) -> None:
    if status not in {"all", "resolved", "ambiguous", "unresolved"}:
        raise ValueError("document scope status is invalid")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (resolved_count, complete_scan_count, rejection_count)
    ):
        raise ValueError("document scope counters are invalid")
    if complete_scan_count > resolved_count:
        raise ValueError("complete scan count exceeds resolved documents")
    if resolved_count > 4 or rejection_count > 4:
        raise ValueError("document scope counters exceed the bound")
    if downgrade_reason is not None and (
        not isinstance(downgrade_reason, str) or len(downgrade_reason) > 256
    ):
        raise ValueError("document scope downgrade reason is invalid")


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError("workflow string collection is invalid")
    return tuple(value)


def _bounded_unique_strings(
    values: tuple[str, ...], *, maximum: int, field: str
) -> None:
    if (
        len(values) > maximum
        or len(values) != len(set(values))
        or any(not value.strip() or len(value) > 1024 for value in values)
    ):
        raise ValueError(f"{field} is invalid")
