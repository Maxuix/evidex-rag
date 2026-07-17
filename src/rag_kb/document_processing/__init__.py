"""Server-owned document-processing profiles."""

from rag_kb.document_processing.profiles import (
    UNSTRUCTURED_CHUNKING_CONFIG,
    UNSTRUCTURED_PARSER_CONFIG,
    index_profile,
)

__all__ = [
    "UNSTRUCTURED_CHUNKING_CONFIG",
    "UNSTRUCTURED_PARSER_CONFIG",
    "index_profile",
]
