"""Application-facing model provider contracts."""

from __future__ import annotations

from typing import Protocol

from rag_kb.domain import (
    ChatModelRequest,
    ChatModelResponse,
    EmbeddingBatch,
    EmbeddingSpaceDefinition,
    ImageEmbeddingInput,
    ModelRerankScore,
    RerankDocument,
    RerankMode,
)


class ChatModelAdapter(Protocol):
    async def complete(self, request: ChatModelRequest) -> ChatModelResponse: ...


class EmbeddingModelAdapter(Protocol):
    @property
    def embedding_space(self) -> EmbeddingSpaceDefinition: ...

    @property
    def max_batch_size(self) -> int: ...

    async def embed_documents(self, texts: tuple[str, ...]) -> EmbeddingBatch: ...

    async def embed_query(self, text: str) -> tuple[float, ...]: ...


class MultimodalEmbeddingAdapter(EmbeddingModelAdapter, Protocol):
    @property
    def embedding_space(self) -> EmbeddingSpaceDefinition: ...

    @property
    def max_batch_size(self) -> int: ...

    async def embed_images(
        self,
        images: tuple[ImageEmbeddingInput, ...],
    ) -> EmbeddingBatch: ...


class RerankerAdapterError(RuntimeError):
    """Content-safe local reranker adapter failure."""


class TextRerankerAdapter(Protocol):
    @property
    def profile(self) -> RerankMode: ...

    @property
    def max_documents(self) -> int: ...

    async def score(
        self,
        query: str,
        documents: tuple[RerankDocument, ...],
    ) -> tuple[ModelRerankScore, ...]: ...
