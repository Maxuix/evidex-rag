"""Fail-closed model adapters used before the user configures a model."""

from __future__ import annotations

from rag_kb.domain import (
    ChatModelExecutionError,
    ChatModelRequest,
    ChatModelResponse,
    EmbeddingBatch,
    EmbeddingSpaceDefinition,
    ErrorCode,
    IndexingExecutionError,
    IndexingPhase,
)


class UnconfiguredChatModelAdapter:
    async def complete(self, request: ChatModelRequest) -> ChatModelResponse:
        raise ChatModelExecutionError(
            ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
            diagnostic={"check": "chat_model_not_configured"},
        )


class UnconfiguredEmbeddingModelAdapter:
    def __init__(self, embedding_space: EmbeddingSpaceDefinition) -> None:
        self._embedding_space = embedding_space

    @property
    def embedding_space(self) -> EmbeddingSpaceDefinition:
        return self._embedding_space

    @property
    def max_batch_size(self) -> int:
        return 1

    async def embed_documents(self, texts: tuple[str, ...]) -> EmbeddingBatch:
        raise self._error()

    async def embed_query(self, text: str) -> tuple[float, ...]:
        raise self._error()

    @staticmethod
    def _error() -> IndexingExecutionError:
        return IndexingExecutionError(
            ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE,
            phase=IndexingPhase.EMBEDDING,
            diagnostic={"check": "embedding_model_not_configured"},
        )
