"""Built-in, immutable Graphiti schema profiles."""

from rag_kb.graph.schema_profiles.registry import (
    ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY,
    GENERIC_GRAPH_SCHEMA_PROFILE_KEY,
    SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY,
    GraphSchemaProfileError,
    GraphSchemaProfileMismatch,
    CompiledGraphSchema,
    get_graph_schema_registry,
)

__all__ = [
    "CompiledGraphSchema",
    "ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY",
    "GENERIC_GRAPH_SCHEMA_PROFILE_KEY",
    "GraphSchemaProfileError",
    "GraphSchemaProfileMismatch",
    "SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY",
    "get_graph_schema_registry",
]
