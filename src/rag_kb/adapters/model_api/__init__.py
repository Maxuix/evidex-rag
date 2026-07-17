"""Embedding model adapter contracts and implementations."""

from rag_kb.adapters.model_api.contracts import ChatModelAdapter, EmbeddingProvider
from rag_kb.adapters.model_api.langchain_chat import LangChainChatModelAdapter
from rag_kb.adapters.model_api.openai_compatible import OpenAICompatibleEmbeddingProvider

__all__ = [
    "ChatModelAdapter",
    "EmbeddingProvider",
    "LangChainChatModelAdapter",
    "OpenAICompatibleEmbeddingProvider",
]
