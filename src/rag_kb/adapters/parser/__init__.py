"""Isolated document-parser adapter boundary."""

from rag_kb.adapters.parser.contracts import DocumentProcessor
from rag_kb.adapters.parser.isolated import IsolatedPlainTextProcessor
from rag_kb.adapters.parser.plain_text import PlainTextTestParser

__all__ = ["DocumentProcessor", "IsolatedPlainTextProcessor", "PlainTextTestParser"]
