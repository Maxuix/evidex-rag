"""Loop guard and diagnostics for the single native tool-calling Chat agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CHAT_AGENT_VERSION = "native_tool_calling_agent_v3"
CHAT_AGENT_TRACE_ARTIFACT = "chat_agent_trace"
CHAT_RETRIEVAL_LANES = frozenset({"simple", "graph_relations"})
CHAT_GRAPH_SEARCH_REASONS = frozenset(
    {
        "direct_relation",
        "relation_chain",
        "entity_alias",
        "cross_document_relation",
    }
)
CHAT_AGENT_REJECTION_REASONS = frozenset(
    {
        "claim_shape",
        "claim_text",
        "evidence_ref",
        "calculation_ref",
        "visual_ref",
    }
)
CHAT_GRAPH_SEARCH_RESULTS = frozenset(
    {
        "not_requested",
        "admitted",
        "no_evidence",
        "not_ready",
        "timeout",
        "unavailable",
        "rejected",
    }
)
CHAT_AGENT_INVOCATION_SOURCES = frozenset({"agent", "legacy_guard"})
CHAT_AGENT_STOP_REASONS = frozenset(
    {
        "submitted",
        "token_budget",
        "retrieval_query_budget",
        "evidence_budget",
        "no_new_evidence",
        "model_round_limit",
        "submit_protocol_invalid",
        "deadline_exceeded",
    }
)
CHAT_AGENT_DEFAULT_MODEL_ROUNDS = 8
CHAT_AGENT_MAX_MODEL_ROUNDS = 12
CHAT_AGENT_DEFAULT_GRAPH_CALLS = 2
CHAT_AGENT_MAX_GRAPH_CALLS = 2
CHAT_AGENT_DEFAULT_TOTAL_TOKENS = 150_000
CHAT_AGENT_MIN_TOTAL_TOKENS = 1_000
CHAT_AGENT_MAX_TOTAL_TOKENS = 10_000_000
CHAT_AGENT_DEFAULT_EVIDENCE_ITEMS = 64
CHAT_AGENT_MAX_EVIDENCE_ITEMS = 512
CHAT_AGENT_DEFAULT_RETRIEVAL_CALLS = 16
CHAT_AGENT_MAX_RETRIEVAL_CALLS = 64
CHAT_AGENT_TRACE_REF_LIMIT = 100
CHAT_AGENT_TRACE_EVENT_LIMIT = 32
CHAT_AGENT_CLAIM_LIMIT = 100
CHAT_AGENT_UNANSWERED_LIMIT = 100
CHAT_AGENT_EVIDENCE_REF_LIMIT = 4
# Per-run ceiling matched by ChatAgentBudget.max_graph_calls.
CHAT_GRAPH_CALL_LIMIT = CHAT_AGENT_MAX_GRAPH_CALLS
# Per-call hard ceiling for new source chunks returned by one Graph search.
CHAT_GRAPH_NEW_CHUNK_LIMIT = 16


@dataclass(frozen=True, slots=True)
class ChatAgentBudget:
    max_model_rounds: int = CHAT_AGENT_DEFAULT_MODEL_ROUNDS
    max_graph_calls: int = CHAT_AGENT_DEFAULT_GRAPH_CALLS
    max_total_tokens: int = CHAT_AGENT_DEFAULT_TOTAL_TOKENS
    max_evidence_items: int = CHAT_AGENT_DEFAULT_EVIDENCE_ITEMS
    max_retrieval_calls: int = CHAT_AGENT_DEFAULT_RETRIEVAL_CALLS

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_model_rounds, bool)
            or not isinstance(self.max_model_rounds, int)
            or not 1 <= self.max_model_rounds <= CHAT_AGENT_MAX_MODEL_ROUNDS
        ):
            raise ValueError("chat agent budget is invalid")
        if (
            isinstance(self.max_graph_calls, bool)
            or not isinstance(self.max_graph_calls, int)
            or not 1 <= self.max_graph_calls <= CHAT_AGENT_MAX_GRAPH_CALLS
        ):
            raise ValueError("chat agent graph budget is invalid")
        if (
            isinstance(self.max_total_tokens, bool)
            or not isinstance(self.max_total_tokens, int)
            or not CHAT_AGENT_MIN_TOTAL_TOKENS
            <= self.max_total_tokens
            <= CHAT_AGENT_MAX_TOTAL_TOKENS
        ):
            raise ValueError("chat agent token budget is invalid")
        if (
            isinstance(self.max_evidence_items, bool)
            or not isinstance(self.max_evidence_items, int)
            or not 1 <= self.max_evidence_items <= CHAT_AGENT_MAX_EVIDENCE_ITEMS
        ):
            raise ValueError("chat agent evidence budget is invalid")
        if (
            isinstance(self.max_retrieval_calls, bool)
            or not isinstance(self.max_retrieval_calls, int)
            or not 1 <= self.max_retrieval_calls <= CHAT_AGENT_MAX_RETRIEVAL_CALLS
        ):
            raise ValueError("chat agent retrieval budget is invalid")
    def as_dict(self) -> dict[str, int | float]:
        return {
            "max_model_rounds": self.max_model_rounds,
            "max_graph_calls": self.max_graph_calls,
            "max_total_tokens": self.max_total_tokens,
            "max_evidence_items": self.max_evidence_items,
            "max_retrieval_calls": self.max_retrieval_calls,
        }


@dataclass(frozen=True, slots=True)
class ChatAgentTraceEvent:
    tool: str
    status: str
    tool_call_id: str
    refs: tuple[str, ...] = ()
    count: int = 0
    retrieval_lane: str | None = None
    route_reason_code: str | None = None
    route_result_code: str | None = None
    new_evidence_count: int | None = None
    rejected_claim_count: int = 0
    rejection_reasons: tuple[str, ...] = ()
    budget_wrap_up: bool = False
    call_index: int | None = None
    invocation_source: str | None = None
    duration_ms: int | None = None
    candidate_count: int | None = None
    path_count: int | None = None
    hydrated_chunk_count: int | None = None
    returned_chunk_count: int | None = None
    hop1_count: int | None = None
    hop2_count: int | None = None
    hop3_count: int | None = None

    def __post_init__(self) -> None:
        if (
            self.tool
            not in {
                "search_knowledge_base",
                "search_graph_relations",
                "calculate",
                "submit_answer",
                "protocol",
            }
            or self.status not in {"ok", "rejected", "salvaged", "refused"}
            or not self.tool_call_id.strip()
            or len(self.tool_call_id) > 128
            or len(self.refs) > CHAT_AGENT_TRACE_REF_LIMIT
            or len(self.refs) != len(set(self.refs))
            or any(not value.strip() or len(value) > 128 for value in self.refs)
            or self.count < 0
            or self.rejected_claim_count < 0
            or len(self.rejection_reasons) > len(CHAT_AGENT_REJECTION_REASONS)
            or len(self.rejection_reasons) != len(set(self.rejection_reasons))
            or any(item not in CHAT_AGENT_REJECTION_REASONS for item in self.rejection_reasons)
            or self.retrieval_lane not in CHAT_RETRIEVAL_LANES | {None}
            or self.route_reason_code not in CHAT_GRAPH_SEARCH_REASONS | {None}
            or self.route_result_code not in CHAT_GRAPH_SEARCH_RESULTS | {None}
            or (
                self.new_evidence_count is not None
                and not 0 <= self.new_evidence_count <= CHAT_GRAPH_NEW_CHUNK_LIMIT
            )
            or (
                self.call_index is not None
                and (
                    isinstance(self.call_index, bool)
                    or not isinstance(self.call_index, int)
                    or not 1 <= self.call_index <= CHAT_GRAPH_CALL_LIMIT
                )
            )
            or self.invocation_source not in CHAT_AGENT_INVOCATION_SOURCES | {None}
            or (
                self.duration_ms is not None
                and (
                    isinstance(self.duration_ms, bool)
                    or not isinstance(self.duration_ms, int)
                    or self.duration_ms < 0
                )
            )
        ):
            raise ValueError("chat agent trace event is invalid")
        for counter in (
            self.candidate_count,
            self.path_count,
            self.hydrated_chunk_count,
            self.returned_chunk_count,
            self.hop1_count,
            self.hop2_count,
            self.hop3_count,
        ):
            if counter is not None and (
                isinstance(counter, bool)
                or not isinstance(counter, int)
                or counter < 0
            ):
                raise ValueError("chat agent trace counters are invalid")
        if self.retrieval_lane is None and any(
            value is not None
            for value in (
                self.route_reason_code,
                self.route_result_code,
                self.new_evidence_count,
                self.call_index,
                self.invocation_source,
                self.duration_ms,
                self.candidate_count,
                self.path_count,
                self.hydrated_chunk_count,
                self.returned_chunk_count,
                self.hop1_count,
                self.hop2_count,
                self.hop3_count,
            )
        ):
            raise ValueError("trace route fields require a retrieval lane")
        if self.retrieval_lane == "simple":
            if self.route_reason_code is not None or self.route_result_code != "not_requested":
                raise ValueError("simple trace route fields are invalid")
            if self.new_evidence_count is not None:
                raise ValueError("simple trace cannot report graph evidence")
            if any(
                value is not None
                for value in (
                    self.call_index,
                    self.invocation_source,
                    self.duration_ms,
                    self.candidate_count,
                    self.path_count,
                    self.hydrated_chunk_count,
                    self.returned_chunk_count,
                    self.hop1_count,
                    self.hop2_count,
                    self.hop3_count,
                )
            ):
                raise ValueError("simple trace cannot report graph counters")
        if self.retrieval_lane == "graph_relations":
            if (
                self.route_reason_code is None
                or self.route_result_code is None
                or self.new_evidence_count is None
                or self.call_index is None
                or self.invocation_source is None
            ):
                raise ValueError("Graph trace route fields are incomplete")
            if self.route_result_code == "admitted":
                if self.count < 1 or self.new_evidence_count < 1:
                    raise ValueError("admitted Graph trace must carry new evidence")
            elif self.new_evidence_count != 0:
                raise ValueError("non-admitted Graph trace cannot add evidence")
            if self.invocation_source == "legacy_guard":
                if self.duration_ms is not None:
                    raise ValueError("legacy guard events cannot fake a duration")
            elif self.duration_ms is None:
                raise ValueError("agent-invoked Graph trace requires a duration")
            if (
                self.returned_chunk_count is not None
                and self.returned_chunk_count != self.count
            ):
                raise ValueError("Graph returned chunk count must match event refs")
            hop_counts = (
                self.hop1_count,
                self.hop2_count,
                self.hop3_count,
            )
            if any(value is not None for value in hop_counts) and not all(
                value is not None for value in hop_counts
            ):
                raise ValueError("Graph hop counts must be reported together")
            if all(value is not None for value in hop_counts):
                total = sum(hop_counts)  # type: ignore[arg-type]
                returned = self.returned_chunk_count
                if returned is None:
                    returned = self.count
                if total != returned:
                    raise ValueError("Graph hop counts must match returned evidence")

    def as_dict(self) -> dict[str, Any]:
        value = {
            "tool": self.tool,
            "status": self.status,
            "tool_call_id": self.tool_call_id,
            "refs": list(self.refs),
            "count": self.count,
        }
        if self.retrieval_lane is not None:
            value.update(
                {
                    "retrieval_lane": self.retrieval_lane,
                    "route_reason_code": self.route_reason_code,
                    "route_result_code": self.route_result_code,
                    "new_evidence_count": self.new_evidence_count,
                }
            )
        if self.retrieval_lane == "graph_relations":
            value.update(
                {
                    "call_index": self.call_index,
                    "invocation_source": self.invocation_source,
                    "duration_ms": self.duration_ms,
                    "candidate_count": self.candidate_count,
                    "path_count": self.path_count,
                    "hydrated_chunk_count": self.hydrated_chunk_count,
                    "returned_chunk_count": self.returned_chunk_count,
                    "hop1_count": self.hop1_count,
                    "hop2_count": self.hop2_count,
                    "hop3_count": self.hop3_count,
                }
            )
        if (
            self.rejected_claim_count
            or self.rejection_reasons
            or self.budget_wrap_up
        ):
            value.update(
                {
                    "rejected_claim_count": self.rejected_claim_count,
                    "rejection_reasons": list(self.rejection_reasons),
                    "budget_wrap_up": self.budget_wrap_up,
                }
            )
        return value


@dataclass(frozen=True, slots=True)
class ChatAgentTrace:
    events: tuple[ChatAgentTraceEvent, ...]
    budget: ChatAgentBudget
    model_rounds: int
    retrieval_calls: int
    calculation_calls: int
    evidence_ref_count: int
    outcome: str
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieval_tool_calls: int = 0
    simple_tool_calls: int = 0
    graph_tool_calls: int = 0
    consecutive_no_new_evidence: int = 0
    stop_reason: str = "submitted"
    forced_finalize: bool = False
    elapsed_ms: int | None = None
    deadline_ms: int | None = None
    deadline_remaining_ms: int | None = None
    deadline_exceeded: bool = False
    version: str = CHAT_AGENT_VERSION

    def __post_init__(self) -> None:
        if (
            self.version != CHAT_AGENT_VERSION
            or len(self.events) > CHAT_AGENT_TRACE_EVENT_LIMIT
            or not 0 <= self.model_rounds <= self.budget.max_model_rounds + 1
            or self.retrieval_calls < 0
            or self.calculation_calls < 0
            or self.evidence_ref_count < 0
            or self.total_tokens < 0
            or self.prompt_tokens < 0
            or self.completion_tokens < 0
            or self.retrieval_tool_calls < 0
            or self.simple_tool_calls < 0
            or self.graph_tool_calls < 0
            or self.retrieval_tool_calls
            != self.simple_tool_calls + self.graph_tool_calls
            or self.consecutive_no_new_evidence < 0
            or self.stop_reason not in CHAT_AGENT_STOP_REASONS
            or not isinstance(self.forced_finalize, bool)
            or not isinstance(self.deadline_exceeded, bool)
            or self.outcome not in {"answered", "partial", "refused", "clarify"}
        ):
            raise ValueError("chat agent trace is invalid")
        for value in (self.elapsed_ms, self.deadline_ms, self.deadline_remaining_ms):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError("chat agent trace timing is invalid")
        if self.deadline_ms is None and (
            self.deadline_remaining_ms is not None
            or self.deadline_exceeded
        ):
            raise ValueError("chat agent trace deadline diagnostics are invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "events": [item.as_dict() for item in self.events],
            "budget": self.budget.as_dict(),
            "usage": {
                "model_rounds": self.model_rounds,
                # Kept as a compatibility alias for historical traces and API clients.
                "retrieval_calls": self.retrieval_calls,
                "retrieval_queries": self.retrieval_calls,
                "retrieval_tool_calls": self.retrieval_tool_calls,
                "simple_tool_calls": self.simple_tool_calls,
                "graph_tool_calls": self.graph_tool_calls,
                "calculation_calls": self.calculation_calls,
                "evidence_refs": self.evidence_ref_count,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "diagnostics": {
                "stop_reason": self.stop_reason,
                "forced_finalize": self.forced_finalize,
                "consecutive_no_new_evidence": self.consecutive_no_new_evidence,
                "elapsed_ms": self.elapsed_ms,
                "deadline_ms": self.deadline_ms,
                "deadline_remaining_ms": self.deadline_remaining_ms,
                "deadline_exceeded": self.deadline_exceeded,
            },
            "outcome": self.outcome,
        }
