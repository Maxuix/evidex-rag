"""Server-owned document-processing profiles."""

from rag_kb.document_processing.profiles import (
    LEGACY_SEMANTIC_PROFILE,
    LEGACY_MULTIMODAL_PARSER_CONFIG_V1,
    MULTIMODAL_ENRICHMENT_CONFIG,
    MULTIMODAL_PARSER_CONFIG,
    MULTIMODAL_REPRESENTATION_CONFIG,
    SEMANTIC_CHUNKING_CONFIG,
    STRUCTURAL_CHUNKING_CONFIG_V3,
    UNSTRUCTURED_CHUNKING_CONFIG,
    UNSTRUCTURED_PARSER_CONFIG,
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
from rag_kb.document_processing.multimodal_assembly import (
    assemble_composite_evidence,
    assemble_multimodal_units,
    asset_manifest_hash,
    element_sequence_hash,
    semantic_text_elements,
    unit_plan_hash,
    normalize_figure_labels,
    relate_composite_units,
)
from rag_kb.document_processing.multimodal_boundaries import (
    VisualDisposition,
    classify_visual,
)

__all__ = [
    "LEGACY_SEMANTIC_PROFILE",
    "LEGACY_MULTIMODAL_PARSER_CONFIG_V1",
    "MULTIMODAL_ENRICHMENT_CONFIG",
    "MULTIMODAL_PARSER_CONFIG",
    "MULTIMODAL_REPRESENTATION_CONFIG",
    "SEMANTIC_CHUNKING_CONFIG",
    "STRUCTURAL_CHUNKING_CONFIG_V3",
    "UNSTRUCTURED_CHUNKING_CONFIG",
    "UNSTRUCTURED_PARSER_CONFIG",
    "count_chunk_tokens",
    "with_composite_embedding_text",
    "assemble_multimodal_units",
    "assemble_composite_evidence",
    "asset_manifest_hash",
    "classify_visual",
    "element_sequence_hash",
    "index_profile",
    "profile_fingerprint",
    "profile_for_preset",
    "public_parsing_descriptor",
    "parsing_preset",
    "public_descriptor",
    "resolve",
    "split_by_tokens",
    "semantic_text_elements",
    "unit_plan_hash",
    "normalize_figure_labels",
    "relate_composite_units",
    "VisualDisposition",
]
