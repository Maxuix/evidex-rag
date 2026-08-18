"""Loop guard and diagnostics for the single native tool-calling Chat agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CHAT_AGENT_VERSION = "native_tool_calling_agent_v2"
CHAT_RETRIEVAL_LANES = frozenset({"simple", "graphiti_supplement"})
CHAT_GRAPHITI_ROUTE_REASONS = frozenset(
    {
        "cross_document_relation_gap",
        "entity_alias_gap",
        "relational_query_without_simple_evidence",
    }
)
CHAT_GRAPHITI_ROUTE_RESULTS = frozenset(
    {
        "not_requested",
        "admitted",
        "no_new_evidence",
        "not_configured",
        "not_ready",
        "runtime_unavailable",
        "rejected",
    }
)


@dataclass(frozen=True, slots=True)
class ChatAgentBudget:
    max_model_rounds: int = 8

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_model_rounds, bool)
            or not isinstance(self.max_model_rounds, int)
            or not 1 <= self.max_model_rounds <= 12
        ):
            raise ValueError("chat agent budget is invalid")

    def as_dict(self) -> dict[str, int]:
        return {"max_model_rounds": self.max_model_rounds}


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

    def __post_init__(self) -> None:
        if (
            self.tool
            not in {
                "search_knowledge_base",
                "graphiti_supplement",
                "calculate",
                "submit_answer",
                "protocol",
            }
            or self.status not in {"ok", "rejected", "salvaged", "refused"}
            or not self.tool_call_id.strip()
            or len(self.tool_call_id) > 128
            or len(self.refs) > 100
            or len(self.refs) != len(set(self.refs))
            or any(not value.strip() or len(value) > 128 for value in self.refs)
            or self.count < 0
            or self.retrieval_lane not in CHAT_RETRIEVAL_LANES | {None}
            or self.route_reason_code not in CHAT_GRAPHITI_ROUTE_REASONS | {None}
            or self.route_result_code not in CHAT_GRAPHITI_ROUTE_RESULTS | {None}
            or (
                self.new_evidence_count is not None
                and not 0 <= self.new_evidence_count <= 4
            )
        ):
            raise ValueError("chat agent trace event is invalid")
        if self.retrieval_lane is None and any(
            value is not None
            for value in (
                self.route_reason_code,
                self.route_result_code,
                self.new_evidence_count,
            )
        ):
            raise ValueError("trace route fields require a retrieval lane")
        if self.retrieval_lane == "simple":
            if self.route_reason_code is not None or self.route_result_code != "not_requested":
                raise ValueError("simple trace route fields are invalid")
            if self.new_evidence_count is not None:
                raise ValueError("simple trace cannot report supplement evidence")
        if self.retrieval_lane == "graphiti_supplement":
            if (
                self.route_reason_code is None
                or self.route_result_code is None
                or self.new_evidence_count is None
            ):
                raise ValueError("Graphiti trace route fields are incomplete")
            if self.route_result_code == "admitted" and self.new_evidence_count < 1:
                raise ValueError("admitted Graphiti trace must add evidence")
            if self.route_result_code != "admitted" and self.new_evidence_count != 0:
                raise ValueError("non-admitted Graphiti trace cannot add evidence")

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
    version: str = CHAT_AGENT_VERSION

    def __post_init__(self) -> None:
        if (
            self.version != CHAT_AGENT_VERSION
            or len(self.events) > 32
            or not 0 <= self.model_rounds <= self.budget.max_model_rounds + 1
            or self.retrieval_calls < 0
            or self.calculation_calls < 0
            or self.evidence_ref_count < 0
            or self.outcome not in {"answered", "partial", "refused"}
        ):
            raise ValueError("chat agent trace is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "events": [item.as_dict() for item in self.events],
            "budget": self.budget.as_dict(),
            "usage": {
                "model_rounds": self.model_rounds,
                "retrieval_calls": self.retrieval_calls,
                "calculation_calls": self.calculation_calls,
                "evidence_refs": self.evidence_ref_count,
            },
            "outcome": self.outcome,
        }
