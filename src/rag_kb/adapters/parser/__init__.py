"""Isolated document-parser adapter boundary."""

from rag_kb.adapters.parser.contracts import DocumentProcessor
from rag_kb.adapters.parser.isolated import IsolatedUnstructuredProcessor

__all__ = ["DocumentProcessor", "IsolatedUnstructuredProcessor"]
