"""Model adapter contracts and implementations."""

from rag_kb.adapters.model_api.contracts import (
    ChatModelAdapter,
    EmbeddingModelAdapter,
    ImageDescriptionAdapter,
    MultimodalEmbeddingAdapter,
)
from rag_kb.adapters.model_api.langchain_chat import LangChainChatModelAdapter
from rag_kb.adapters.model_api.langchain_embeddings import (
    LangChainEmbeddingModelAdapter,
)
from rag_kb.adapters.model_api.multimodal_embeddings import (
    QwenMultimodalEmbeddingAdapter,
)

__all__ = [
    "ChatModelAdapter",
    "EmbeddingModelAdapter",
    "ImageDescriptionAdapter",
    "LangChainChatModelAdapter",
    "LangChainEmbeddingModelAdapter",
    "MultimodalEmbeddingAdapter",
    "QwenMultimodalEmbeddingAdapter",
]
