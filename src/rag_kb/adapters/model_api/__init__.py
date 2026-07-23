"""Model adapter contracts and implementations."""

from rag_kb.adapters.model_api.contracts import (
    ChatModelAdapter,
    EmbeddingModelAdapter,
    MultimodalEmbeddingAdapter,
)
from rag_kb.adapters.model_api.langchain_chat import LangChainChatModelAdapter
from rag_kb.adapters.model_api.langchain_embeddings import (
    LangChainEmbeddingModelAdapter,
)
from rag_kb.adapters.model_api.multimodal_embeddings import TongyiVisionEmbeddingAdapter

__all__ = [
    "ChatModelAdapter",
    "EmbeddingModelAdapter",
    "LangChainChatModelAdapter",
    "LangChainEmbeddingModelAdapter",
    "MultimodalEmbeddingAdapter",
    "TongyiVisionEmbeddingAdapter",
]
