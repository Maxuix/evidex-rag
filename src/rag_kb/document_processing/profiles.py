"""Immutable parser/chunking preset registry for index revisions."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from rag_kb.domain import (
    ChunkingPreset,
    ChunkingStrategyKind,
    IndexProfileDefinition,
    ParsingPreset,
)


UNSTRUCTURED_PARSER_CONFIG = {
    "profile": "unstructured_local_v1",
    "integration": "langchain-unstructured",
    "integration_version": "1.0.1",
    "engine": "unstructured",
    "engine_version": "0.24.1",
    "partition_via_api": False,
    "strategy": "fast",
    "include_page_breaks": True,
    "supported_extensions": [".txt", ".md", ".pdf", ".docx"],
}

LEGACY_MULTIMODAL_PARSER_CONFIG_V1 = {
    "profile": "unstructured_multimodal_local_v1",
    "integration": "unstructured",
    "engine_version": "0.24.1",
    "partition_via_api": False,
    "pdf_strategy": "hi_res",
    "infer_table_structure": True,
    "extract_image_block_types": ["Image", "Table"],
    "include_page_breaks": True,
    "docx_picture_partitioner": "bounded_ooxml_relationship_v1",
    "supported_extensions": [".txt", ".md", ".pdf", ".docx"],
}

MULTIMODAL_PARSER_CONFIG = {
    **LEGACY_MULTIMODAL_PARSER_CONFIG_V1,
    "profile": "unstructured_multimodal_local_v2",
    "docx_picture_partitioner": "bounded_ooxml_relationship_v2",
    "composite_assembly": "deterministic_chunk_asset_relations_v2",
}

MULTIMODAL_ENRICHMENT_CONFIG = {
    "profile": "composite_visual_enrichment_v2",
    "ocr": "local_unstructured_ocr_v1",
    "author_caption": "bounded_author_caption_v2",
    "figure_reference": "deterministic_figure_reference_v2",
    "relation_builder": "deterministic_chunk_asset_relations_v2",
    "visual_filter": "bounded_visual_disposition_v2",
    "table_normalization": "bounded_table_text_html_v1",
}

MULTIMODAL_REPRESENTATION_CONFIG = {
    "profile": "composite_multimodal_representations_v2",
    "embedding": "tongyi_vision_flash_20260306_independent_768_v1",
    "embedding_text": {
        "profile": "composite_embedding_text_v2",
        "sections": ["body", "figure_label", "author_caption", "ocr", "table"],
        "separator": "\\n",
        "unicode": "NFC",
        "max_tokens": 1200,
        "max_attachment_tokens": 256,
    },
    "text": {"required": ["text"]},
    "image": {"required": ["native_image"], "optional": []},
    "table": {"required": ["table_text"], "optional": ["table_image"]},
    "relations": {
        "profile": "deterministic_chunk_asset_relations_v2",
        "max_per_chunk": 32,
        "max_total": 50000,
    },
}

_DOCLING_PARSER_BASE = {
    "engine": "docling",
    "engine_version": "2.114.0",
    "core_version": "2.87.1",
    "document_schema": "DoclingDocument",
    "document_version": "1.10.0",
    "allowed_formats": [
        ".txt", ".md", ".html", ".csv", ".pdf", ".docx", ".pptx", ".xlsx",
    ],
    "supported_extensions": [
        ".txt", ".md", ".html", ".csv", ".pdf", ".docx", ".pptx", ".xlsx",
    ],
    "sheet_name_supplement": "bounded_ooxml_workbook_v1",
    "pdf_pipeline": "standard",
    "pdf_backend": "default_2_114_0",
    "ocr_engine": "rapidocr",
    "ocr_models": ["chinese"],
    "ocr_force_full_page": False,
    "do_table_structure": True,
    "enable_remote_services": False,
    "allow_external_plugins": False,
    "picture_classification": False,
    "picture_description": False,
    "chart_extraction": False,
    "code_formula_enrichment": False,
    "accelerator": "cpu_single_thread",
    "document_timeout_seconds": 600,
    "max_num_pages": 500,
    "max_file_size": 10_485_760,
    "max_docling_items": 20_000,
    "provenance_projection": "docling_prov_v1",
    "unknown_item_policy": "text_or_skip_v1",
    "model_artifact_manifest": "docling-artifacts-v1",
}

DOCLING_TEXT_PARSER_CONFIG = {
    **_DOCLING_PARSER_BASE,
    "profile": "docling_text_local_v1",
    "generate_page_images": False,
    "generate_picture_images": False,
    "asset_mapping": "none",
}

DOCLING_MULTIMODAL_PARSER_CONFIG = {
    **_DOCLING_PARSER_BASE,
    "profile": "docling_multimodal_local_v1",
    "generate_page_images": True,
    "generate_picture_images": True,
    "asset_mapping": "docling_picture_table_page_v1",
    "page_image_policy": "scanned_surface_v1",
}

DOCLING_ENRICHMENT_CONFIG = {
    "profile": "composite_visual_enrichment_v3",
    "ocr": "docling_rapidocr_v1",
    "author_caption": "docling_caption_ref_v1",
    "figure_reference": "deterministic_figure_reference_v2",
    "relation_builder": "docling_chunk_asset_relations_v1",
    "visual_filter": "docling_decorative_repeat_v1",
    "table_normalization": "docling_markdown_bounded_html_v1",
}

DOCLING_REPRESENTATION_CONFIG = {
    "profile": "composite_multimodal_representations_v3",
    "embedding": "tongyi_vision_flash_20260306_independent_768_v1",
    "embedding_text": {
        "profile": "composite_embedding_text_v2",
        "sections": ["body", "figure_label", "author_caption", "ocr", "table"],
        "separator": "\\n",
        "unicode": "NFC",
        "max_tokens": 1200,
        "max_attachment_tokens": 256,
    },
    "text": {"required": ["text"]},
    "image": {"required": ["native_image"], "optional": []},
    "table": {"required": ["table_text"], "optional": ["table_image"]},
    "relations": {
        "profile": "docling_chunk_asset_relations_v1",
        "max_per_chunk": 32,
        "max_total": 50000,
    },
}

UNSTRUCTURED_CHUNKING_CONFIG = {
    "profile": "unstructured_by_title_token_v2",
    "strategy": "by_title",
    "max_tokens": 800,
    "new_after_n_tokens": 600,
    "tokenizer": "cl100k_base",
    "tokenizer_library": "tiktoken",
    "tokenizer_version": "0.13.0",
    "overlap": 100,
    "overlap_unit": "tokens",
    "overlap_all": False,
    "combine_text_under_n_chars": 300,
    "combine_text_under_n_chars_unit": "characters",
    "multipage_sections": False,
    "include_orig_elements": True,
    "metadata_policy": "bounded_v2",
}

# The native Docling structural profile. Token thresholds and tokenizer stay
# identical to the retired v2 profile so parity against the Unstructured chain
# stays measurable; boundary sources, unknown-item policy and the provenance
# projection are the new frozen facts.
STRUCTURAL_CHUNKING_CONFIG_V3 = {
    "profile": "structural_by_title_token_v3",
    "strategy": "docling_structural",
    "max_tokens": 800,
    "new_after_n_tokens": 600,
    "tokenizer": "cl100k_base",
    "tokenizer_library": "tiktoken",
    "tokenizer_version": "0.13.0",
    "overlap": 100,
    "overlap_unit": "tokens",
    "overlap_all": False,
    "boundary_sources": [
        "title",
        "section_header",
        "table",
        "surface",
        "max_tokens",
    ],
    "multipage_sections": False,
    "caption_policy": "relation_only_v1",
    "picture_policy": "relation_only_v1",
    "table_serialization": "docling_markdown_bounded_html_v1",
    "unknown_item_policy": "text_or_skip_v1",
    "provenance_projection": "docling_prov_v1",
    "metadata_policy": "bounded_v3",
}

SEMANTIC_CHUNKING_CONFIG = {
    "profile": "semantic_breakpoint_v1",
    "preset": "semantic_balanced_v1",
    "strategy": "semantic_breakpoint",
    "analysis_embedding": "index_embedding_space",
    "tokenizer": "cl100k_base",
    "tokenizer_library": "tiktoken",
    "tokenizer_version": "0.13.0",
    "analysis_unit_target_tokens": 80,
    "analysis_unit_max_tokens": 160,
    "max_analysis_units": 5000,
    "max_analysis_tokens": 500000,
    "min_chunk_tokens": 220,
    "target_chunk_tokens": 550,
    "max_chunk_tokens": 800,
    "distance_metric": "cosine",
    "distance_smoothing": "weighted_3_v1",
    "distance_quantization": 1000000,
    "semantic_threshold": "median_plus_mad_v1",
    "semantic_mad_multiplier_micros": 1000000,
    "selector": "constrained_dp_v1",
    "size_penalty_weight_micros": 150000,
    "preserve_page_boundaries": True,
    "preserve_table_boundaries": True,
    "attach_title_to_following": True,
    "normal_overlap_tokens": 0,
    "oversized_element_overlap_tokens": 50,
    "metadata_policy": "bounded_v3",
}

# Read-only compatibility for local revisions created by the retired prototype.
# Index execution deliberately does not resolve this profile.
LEGACY_SEMANTIC_PROFILE = "unstructured_title_semantic_qwen_v1"


def profile_for_preset(
    preset: ChunkingPreset | str,
    parsing_preset: ParsingPreset | str = ParsingPreset.TEXT_LOCAL_V1,
) -> IndexProfileDefinition:
    """Return fresh JSON-compatible documents for persistence."""

    resolved = ChunkingPreset(preset)
    chunking = (
        STRUCTURAL_CHUNKING_CONFIG_V3
        if resolved is ChunkingPreset.STRUCTURAL_BALANCED_V2
        else SEMANTIC_CHUNKING_CONFIG
    )
    parsing = ParsingPreset(parsing_preset)
    multimodal = parsing is ParsingPreset.MULTIMODAL_LOCAL_V1
    return IndexProfileDefinition(
        parser_config=deepcopy(
            DOCLING_MULTIMODAL_PARSER_CONFIG
            if multimodal
            else DOCLING_TEXT_PARSER_CONFIG
        ),
        chunking_config=deepcopy(chunking),
        enrichment_config=deepcopy(DOCLING_ENRICHMENT_CONFIG) if multimodal else {},
        representation_config=(
            deepcopy(DOCLING_REPRESENTATION_CONFIG) if multimodal else {}
        ),
    )


def index_profile() -> IndexProfileDefinition:
    """Compatibility alias for the default structural preset."""

    return profile_for_preset(ChunkingPreset.STRUCTURAL_BALANCED_V2)


def resolve(
    parser_config: dict,
    chunking_config: dict,
) -> ChunkingStrategyKind:
    """Fail closed unless the complete persisted profile exactly matches a preset.

    Only native Docling profiles are executable. Revisions produced by the
    retired Unstructured chain stay readable through the descriptor helpers, but
    re-running or rebuilding one is a stable incompatibility rather than a
    crash inside a parser that no longer matches the recorded profile.
    """

    if parser_config not in (
        DOCLING_TEXT_PARSER_CONFIG,
        DOCLING_MULTIMODAL_PARSER_CONFIG,
    ):
        raise ValueError("unknown parser profile")
    if chunking_config == STRUCTURAL_CHUNKING_CONFIG_V3:
        return ChunkingStrategyKind.STRUCTURAL
    if chunking_config == SEMANTIC_CHUNKING_CONFIG:
        return ChunkingStrategyKind.SEMANTIC
    raise ValueError("unknown chunking profile")


def parsing_preset(parser_config: dict) -> ParsingPreset:
    if parser_config in (DOCLING_TEXT_PARSER_CONFIG, UNSTRUCTURED_PARSER_CONFIG):
        return ParsingPreset.TEXT_LOCAL_V1
    if parser_config in (
        DOCLING_MULTIMODAL_PARSER_CONFIG,
        MULTIMODAL_PARSER_CONFIG,
        LEGACY_MULTIMODAL_PARSER_CONFIG_V1,
    ):
        return ParsingPreset.MULTIMODAL_LOCAL_V1
    raise ValueError("unknown parser profile")


def public_parsing_descriptor(parser_config: dict) -> dict[str, str]:
    preset = parsing_preset(parser_config)
    return {"preset": preset.value, "profile": parser_config["profile"]}


def public_descriptor(chunking_config: dict) -> dict[str, str]:
    for config in (STRUCTURAL_CHUNKING_CONFIG_V3, UNSTRUCTURED_CHUNKING_CONFIG):
        if chunking_config == config:
            return {
                "preset": ChunkingPreset.STRUCTURAL_BALANCED_V2.value,
                "profile": config["profile"],
            }
    if chunking_config == SEMANTIC_CHUNKING_CONFIG:
        return {
            "preset": ChunkingPreset.SEMANTIC_BALANCED_V1.value,
            "profile": SEMANTIC_CHUNKING_CONFIG["profile"],
        }
    if chunking_config.get("profile") == LEGACY_SEMANTIC_PROFILE:
        return {
            "preset": "legacy_incompatible",
            "profile": LEGACY_SEMANTIC_PROFILE,
        }
    raise ValueError("unknown chunking profile")


def profile_fingerprint(
    parser_config: dict,
    chunking_config: dict,
    enrichment_config: dict | None = None,
    representation_config: dict | None = None,
) -> str:
    payload = {
        "parser_config": parser_config,
        "chunking_config": chunking_config,
    }
    if enrichment_config:
        payload["enrichment_config"] = enrichment_config
    if representation_config:
        payload["representation_config"] = representation_config
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
