"""Resolve migration-created vector spaces through a compile-time allowlist."""

from __future__ import annotations

from dataclasses import fields

from rag_kb.domain import (
    EmbeddingSpaceDefinition,
    ErrorCode,
    IndexingExecutionError,
    IndexingPhase,
)


class FixedPgVectorSpace:
    metric = "cosine"
    vector_data_type = "float32"
    _TABLE_BY_DIMENSION = {
        768: "vector_record_768",
        1024: "vector_record_1024",
    }

    def __init__(self, configured: EmbeddingSpaceDefinition) -> None:
        if (
            configured.dimension not in self._TABLE_BY_DIMENSION
            or configured.distance_metric != self.metric
            or configured.vector_data_type != self.vector_data_type
        ):
            raise ValueError("configured embedding space is not an allowed fixed space")
        self._configured = configured

    @property
    def dimension(self) -> int:
        return self._configured.dimension

    @property
    def physical_table(self) -> str:
        return self._TABLE_BY_DIMENSION[self.dimension]

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
