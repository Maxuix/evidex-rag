"""Parser adapters: the native Docling runtime and its bounded probes."""

from rag_kb.adapters.parser.contracts import DocumentParser
from rag_kb.adapters.parser.docling import DoclingParser
from rag_kb.adapters.parser.ooxml_metadata import worksheet_labels
from rag_kb.adapters.parser.scanned_pages import scanned_surfaces

__all__ = [
    "DoclingParser",
    "DocumentParser",
    "scanned_surfaces",
    "worksheet_labels",
]
