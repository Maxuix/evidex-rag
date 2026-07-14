"""External protocol adapter implementations."""
from rag_kb.adapters.file_store import LocalFileStore, SourceFileStore
from rag_kb.adapters.parser import (
    DocumentProcessor,
    IsolatedPlainTextProcessor,
    PlainTextTestParser,
)

__all__ = [
    "DocumentProcessor",
    "IsolatedPlainTextProcessor",
    "LocalFileStore",
    "PlainTextTestParser",
    "SourceFileStore",
]
