"""Resolve the one migration-created P1A vector space without runtime DDL."""

from __future__ import annotations

from dataclasses import fields

from rag_kb.domain import (
    EmbeddingSpaceDefinition,
    ErrorCode,
    IndexingExecutionError,
    IndexingPhase,
)


class FixedPgVectorSpace:
    physical_table = "vector_record_1024"
    dimension = 1024
    metric = "cosine"
    vector_data_type = "float32"

    def __init__(self, configured: EmbeddingSpaceDefinition) -> None:
        if (
            configured.dimension != self.dimension
            or configured.distance_metric != self.metric
            or configured.vector_data_type != self.vector_data_type
        ):
            raise ValueError("configured embedding space is not the fixed P1A space")
        self._configured = configured

    @property
    def configured_space(self) -> EmbeddingSpaceDefinition:
        return self._configured

    def require_compatible(
        self,
        persisted: EmbeddingSpaceDefinition,
        provider: EmbeddingSpaceDefinition,
    ) -> None:
        mismatches = tuple(
            field.name
            for field in fields(EmbeddingSpaceDefinition)
            if getattr(persisted, field.name) != getattr(self._configured, field.name)
            or getattr(provider, field.name) != getattr(self._configured, field.name)
        )
        if mismatches:
            raise IndexingExecutionError(
                ErrorCode.EMBEDDING_SPACE_MISMATCH,
                phase=IndexingPhase.EMBEDDING,
                diagnostic={"fields": mismatches},
            )
