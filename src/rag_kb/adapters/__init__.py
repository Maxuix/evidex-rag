"""External protocol adapter implementations."""
from rag_kb.adapters.file_store import LocalFileStore, SourceFileStore
from rag_kb.adapters.parser import (
    DocumentProcessor,
    IsolatedPlainTextProcessor,
    PlainTextTestParser,
)
from rag_kb.adapters.model_api import (
    EmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from rag_kb.adapters.vector_store import FixedPgVectorSpace, VectorStore

__all__ = [
    "DocumentProcessor",
    "EmbeddingProvider",
    "FixedPgVectorSpace",
    "IsolatedPlainTextProcessor",
    "LocalFileStore",
    "OpenAICompatibleEmbeddingProvider",
    "PlainTextTestParser",
    "SourceFileStore",
    "VectorStore",
]
