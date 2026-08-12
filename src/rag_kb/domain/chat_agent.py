"""Bounded facts for the single native tool-calling Chat agent."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


CHAT_AGENT_VERSION = "native_tool_calling_agent_v1"


@dataclass(frozen=True, slots=True)
class ChatAgentBudget:
    model_rounds: int = 8
    retrieval_calls: int = 6
    calculation_calls: int = 4
    evidence_refs: int = 20

    def __post_init__(self) -> None:
        if (
            not 2 <= self.model_rounds <= 12
            or not 1 <= self.retrieval_calls <= 12
            or not 0 <= self.calculation_calls <= 4
            or not 1 <= self.evidence_refs <= 100
        ):
            raise ValueError("chat agent budget is invalid")

    def as_dict(self) -> dict[str, int]:
        return {
            "model_rounds": self.model_rounds,
            "retrieval_calls": self.retrieval_calls,
            "calculation_calls": self.calculation_calls,
            "evidence_refs": self.evidence_refs,
        }


@dataclass(frozen=True, slots=True)
class ChatAgentTraceEvent:
    tool: str
    status: str
    tool_call_id: str
    refs: tuple[str, ...] = ()
    count: int = 0

    def __post_init__(self) -> None:
        if (
            self.tool not in {"search_knowledge_base", "calculate", "submit_answer", "protocol"}
            or self.status not in {"ok", "rejected", "salvaged", "refused"}
            or not self.tool_call_id.strip()
            or len(self.tool_call_id) > 128
            or len(self.refs) > 100
            or len(self.refs) != len(set(self.refs))
            or any(not value.strip() or len(value) > 128 for value in self.refs)
            or self.count < 0
        ):
            raise ValueError("chat agent trace event is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "status": self.status,
            "tool_call_id": self.tool_call_id,
            "refs": list(self.refs),
            "count": self.count,
        }


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
            or not 0 <= self.model_rounds <= self.budget.model_rounds
            or not 0 <= self.retrieval_calls <= self.budget.retrieval_calls
            or not 0 <= self.calculation_calls <= self.budget.calculation_calls
            or not 0 <= self.evidence_ref_count <= self.budget.evidence_refs
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


def frozen_agent_trace(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Make a shallow immutable trace payload for domain transport."""

    return MappingProxyType(dict(value))
