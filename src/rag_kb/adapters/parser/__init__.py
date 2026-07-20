"""Document-parser adapter boundary."""

from rag_kb.adapters.parser.contracts import DocumentProcessor
from rag_kb.adapters.parser.local import UnstructuredProcessor

__all__ = ["DocumentProcessor", "UnstructuredProcessor"]
