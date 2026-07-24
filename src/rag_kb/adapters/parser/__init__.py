"""Document-parser adapter boundary."""

from rag_kb.adapters.parser.contracts import DocumentParser, DocumentProcessor
from rag_kb.adapters.parser.local import UnstructuredProcessor

__all__ = [
    "DocumentParser",
    "DocumentProcessor",
    "UnstructuredProcessor",
]
