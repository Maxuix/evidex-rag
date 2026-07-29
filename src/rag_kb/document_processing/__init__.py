"""Server-owned document-processing profiles and pure assembly."""

from rag_kb.document_processing.profiles import (
    DOCLING_ENRICHMENT_CONFIG,
    DOCLING_MULTIMODAL_PARSER_CONFIG,
    DOCLING_REPRESENTATION_CONFIG,
    DOCLING_TEXT_PARSER_CONFIG,
    SEMANTIC_CHUNKING_CONFIG,
    STRUCTURAL_CHUNKING_CONFIG_V3,
    index_profile,
    profile_fingerprint,
    profile_for_preset,
    public_parsing_descriptor,
    parsing_preset,
    public_descriptor,
    resolve,
)
from rag_kb.document_processing.tokenization import (
    count_chunk_tokens,
    split_by_tokens,
)
from rag_kb.document_processing.composite_text import with_composite_embedding_text
from rag_kb.document_processing.docling.figures import normalize_figure_labels

__all__ = [
    "DOCLING_ENRICHMENT_CONFIG",
    "DOCLING_MULTIMODAL_PARSER_CONFIG",
    "DOCLING_REPRESENTATION_CONFIG",
    "DOCLING_TEXT_PARSER_CONFIG",
    "SEMANTIC_CHUNKING_CONFIG",
    "STRUCTURAL_CHUNKING_CONFIG_V3",
    "count_chunk_tokens",
    "with_composite_embedding_text",
    "index_profile",
    "normalize_figure_labels",
    "profile_fingerprint",
    "profile_for_preset",
    "public_parsing_descriptor",
    "parsing_preset",
    "public_descriptor",
    "resolve",
    "split_by_tokens",
]
