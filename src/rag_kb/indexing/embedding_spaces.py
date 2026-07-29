"""Framework-free embedding-space compatibility policy."""

from __future__ import annotations

from dataclasses import fields

from rag_kb.domain import (
    EmbeddingSpaceDefinition,
    ErrorCode,
    IndexingExecutionError,
    IndexingPhase,
)


def require_compatible_embedding_spaces(
    configured: EmbeddingSpaceDefinition,
    persisted: EmbeddingSpaceDefinition,
    provider: EmbeddingSpaceDefinition,
) -> None:
    mismatches = tuple(
        field.name
        for field in fields(EmbeddingSpaceDefinition)
        if getattr(persisted, field.name) != getattr(configured, field.name)
        or getattr(provider, field.name) != getattr(configured, field.name)
    )
    if mismatches:
        raise IndexingExecutionError(
            ErrorCode.EMBEDDING_SPACE_MISMATCH,
            phase=IndexingPhase.EMBEDDING,
            diagnostic={"fields": mismatches},
        )
