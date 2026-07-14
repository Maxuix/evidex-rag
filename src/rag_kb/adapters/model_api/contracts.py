"""Application-facing embedding provider contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_kb.domain import EmbeddingBatch, EmbeddingSpaceDefinition


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def embedding_space(self) -> EmbeddingSpaceDefinition: ...

    @property
    def max_batch_size(self) -> int: ...

    async def embed(self, texts: tuple[str, ...]) -> EmbeddingBatch: ...
