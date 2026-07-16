"""Compile the fixed, linear, checkpoint-free chat StateGraph."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph

from rag_kb.workflows.state_mapping import ChatGraphState


CHAT_GRAPH_NODES = (
    "load_context",
    "retrieve_evidence",
    "assess_evidence",
    "generate_or_refuse",
    "validate_structure",
    "persist_result",
)

ChatGraphNode = Callable[
    [ChatGraphState], Awaitable[dict[str, Any]]
]


def compile_chat_graph(nodes: Mapping[str, ChatGraphNode]) -> Any:
    if tuple(nodes) != CHAT_GRAPH_NODES:
        raise ValueError("chat graph nodes must match the fixed execution order")
    builder = StateGraph(ChatGraphState)
    for name, node in nodes.items():
        builder.add_node(name, node)
    builder.add_edge(START, CHAT_GRAPH_NODES[0])
    for source, target in zip(CHAT_GRAPH_NODES, CHAT_GRAPH_NODES[1:]):
        builder.add_edge(source, target)
    builder.add_edge(CHAT_GRAPH_NODES[-1], END)
    return builder.compile(checkpointer=None)
