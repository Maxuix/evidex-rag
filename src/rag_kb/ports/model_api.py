"""Application-facing model provider contracts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_kb.domain import (
    ChatModelRequest,
    ChatModelResponse,
    EmbeddingBatch,
    EmbeddingSpaceDefinition,
    ImageEmbeddingInput,
)


@runtime_checkable
class ChatModelAdapter(Protocol):
    async def complete(self, request: ChatModelRequest) -> ChatModelResponse: ...


@runtime_checkable
class EmbeddingModelAdapter(Protocol):
    @property
    def embedding_space(self) -> EmbeddingSpaceDefinition: ...

    @property
    def max_batch_size(self) -> int: ...

    async def embed_documents(self, texts: tuple[str, ...]) -> EmbeddingBatch: ...

    async def embed_query(self, text: str) -> tuple[float, ...]: ...


@runtime_checkable
class MultimodalEmbeddingAdapter(Protocol):
    @property
    def embedding_space(self) -> EmbeddingSpaceDefinition: ...

    @property
    def max_batch_size(self) -> int: ...

    async def embed_texts(self, texts: tuple[str, ...]) -> EmbeddingBatch: ...

    async def embed_images(
        self,
        images: tuple[ImageEmbeddingInput, ...],
    ) -> EmbeddingBatch: ...
