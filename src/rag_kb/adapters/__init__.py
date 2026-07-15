"""External protocol adapter implementations."""
from rag_kb.adapters.file_store import LocalFileStore, SourceFileStore
from rag_kb.adapters.parser import (
    DocumentProcessor,
    IsolatedPlainTextProcessor,
    PlainTextTestParser,
)
from rag_kb.adapters.model_api import (
    ChatModelAdapter,
    EmbeddingProvider,
    OpenAICompatibleChatModelAdapter,
    OpenAICompatibleEmbeddingProvider,
)
from rag_kb.adapters.vector_store import FixedPgVectorSpace, PgVectorStore, VectorStore

__all__ = [
    "DocumentProcessor",
    "ChatModelAdapter",
    "EmbeddingProvider",
    "FixedPgVectorSpace",
    "IsolatedPlainTextProcessor",
    "LocalFileStore",
    "OpenAICompatibleEmbeddingProvider",
    "OpenAICompatibleChatModelAdapter",
    "PlainTextTestParser",
    "PgVectorStore",
    "SourceFileStore",
    "VectorStore",
]
