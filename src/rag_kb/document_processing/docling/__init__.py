"""Pure consumers that read a native ``DoclingDocument`` directly.

Structural chunking, semantic analysis and visual asset extraction all traverse
the same converted document. Nothing here re-parses a source, copies Docling's
document model, or keeps a Docling item alive past assembly.
"""

from rag_kb.document_processing.docling.assets import (
    extract_docling_assets,
    relate_assets_to_chunks,
)
from rag_kb.document_processing.docling.evidence import (
    asset_manifest_hash,
    composite_evidence,
    docling_item_sequence_hash,
    text_only_document,
)
from rag_kb.document_processing.docling.provenance import (
    chunk_assembly_key,
    project_source_location,
)
from rag_kb.document_processing.docling.semantic import (
    assemble_semantic_chunks,
    docling_semantic_units,
    docling_unit_sequence_hash,
)
from rag_kb.document_processing.docling.structural import assemble_structural
from rag_kb.document_processing.docling.traversal import (
    ItemKind,
    classify_item,
    iterate_body_items,
    item_text,
    section_paths,
)

__all__ = [
    "ItemKind",
    "assemble_semantic_chunks",
    "assemble_structural",
    "chunk_assembly_key",
    "asset_manifest_hash",
    "classify_item",
    "composite_evidence",
    "docling_item_sequence_hash",
    "docling_semantic_units",
    "docling_unit_sequence_hash",
    "extract_docling_assets",
    "text_only_document",
    "iterate_body_items",
    "item_text",
    "project_source_location",
    "relate_assets_to_chunks",
    "section_paths",
]
