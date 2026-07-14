"""Embedding model adapter contracts and implementations."""

from rag_kb.adapters.model_api.contracts import EmbeddingProvider
from rag_kb.adapters.model_api.openai_compatible import OpenAICompatibleEmbeddingProvider

__all__ = ["EmbeddingProvider", "OpenAICompatibleEmbeddingProvider"]
