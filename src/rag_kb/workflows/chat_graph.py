"""Compile the fixed, linear, checkpoint-free chat StateGraph."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph

from rag_kb.workflows.state_mapping import ChatGraphState


CHAT_GRAPH_NODES = (
    "load_context",
    "contextualize_query",
    "retrieve_evidence",
    "assess_evidence",
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
    builder.add_edge("contextualize_query", "retrieve_evidence")
    builder.add_edge("retrieve_evidence", "assess_evidence")
    builder.add_edge("assess_evidence", "generate_or_refuse")
    builder.add_edge("generate_or_refuse", "validate_structure")
    builder.add_edge("validate_structure", "persist_result")
    builder.add_edge("persist_result", END)
    return builder.compile(checkpointer=None)
