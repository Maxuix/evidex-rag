"""Compile the fixed, branching, checkpoint-free chat StateGraph."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph

from rag_kb.workflows.state_mapping import ChatGraphState


CHAT_GRAPH_NODES = (
    "load_context",
    "contextualize_query",
    "route_workflow",
    "retrieve_evidence",
    "research_evidence",
    "assess_evidence",
    "prepare_visual_evidence",
    "generate_or_refuse",
    "validate_structure",
    "persist_result",
)

ChatGraphNode = Callable[
    [ChatGraphState], Awaitable[dict[str, Any]]
]


def compile_chat_graph(
    nodes: Mapping[str, ChatGraphNode],
) -> Any:
    if tuple(nodes) != CHAT_GRAPH_NODES:
        raise ValueError("chat graph nodes must match the fixed execution order")
    builder = StateGraph(ChatGraphState)
    for name, node in nodes.items():
        builder.add_node(name, node)
    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "contextualize_query")
    builder.add_edge("contextualize_query", "route_workflow")
    builder.add_conditional_edges(
        "route_workflow",
        _retrieval_branch,
        {
            "simple": "retrieve_evidence",
            "agent": "research_evidence",
        },
    )
    builder.add_edge("retrieve_evidence", "assess_evidence")
    builder.add_edge("research_evidence", "assess_evidence")
    builder.add_edge("assess_evidence", "prepare_visual_evidence")
    builder.add_edge("prepare_visual_evidence", "generate_or_refuse")
    builder.add_edge("generate_or_refuse", "validate_structure")
    builder.add_edge("validate_structure", "persist_result")
    builder.add_edge("persist_result", END)
    return builder.compile(checkpointer=None)


def _retrieval_branch(state: ChatGraphState) -> str:
    value = state["workflow_state"].resolved_mode.value
    if value not in {"simple", "agent"}:
        raise ValueError("chat workflow must resolve before retrieval")
    return value
