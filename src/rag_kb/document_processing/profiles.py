"""Immutable parser/chunking preset registry for index revisions."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from rag_kb.domain import (
    ChunkingPreset,
    ChunkingStrategyKind,
    IndexProfileDefinition,
    ParserProfile,
    ParsingPreset,
)
from rag_kb.tokenizer import (
    CL100K_BASE_ENCODING_NAME,
    CL100K_BASE_TOKENIZER_VERSION,
    TOKENIZER_LIBRARY,
)


#: The one tokenizer every executable chunking profile is frozen to.
CHUNK_TOKENIZER = {
    "tokenizer": CL100K_BASE_ENCODING_NAME,
    "tokenizer_library": TOKENIZER_LIBRARY,
    "tokenizer_version": CL100K_BASE_TOKENIZER_VERSION,
}

_DOCLING_PARSER_BASE_V1 = {
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

DOCLING_TEXT_PARSER_CONFIG_V1 = {
    **_DOCLING_PARSER_BASE_V1,
    "profile": "docling_text_local_v1",
    "generate_page_images": False,
    "generate_picture_images": False,
    "asset_mapping": "none",
}

DOCLING_MULTIMODAL_PARSER_CONFIG_V2 = {
    **_DOCLING_PARSER_BASE_V1,
    "profile": "docling_multimodal_local_v2",
    "generate_page_images": True,
    "generate_picture_images": True,
    "asset_mapping": "docling_picture_table_page_v1",
    "page_image_policy": "scanned_surface_v1",
    "markdown_media": {
        "bundle_format": "markdown_bundle_v1",
        "normalizer": "markdown_media_snapshot_v1",
        "admission_remote_snapshot": True,
        "docling_fetch_images": True,
        "docling_local_fetch": True,
        "docling_remote_fetch": False,
        "unresolved_media_policy": "reject",
        "supported_media_types": ["image/png", "image/jpeg", "image/webp"],
        "max_references": 64,
        "max_image_bytes": 8_388_608,
        "max_total_image_bytes": 9_437_184,
        "max_bundle_bytes": 10_485_760,
        "max_redirects": 3,
        "connect_timeout_seconds": 5,
        "read_timeout_seconds": 10,
        "remote_address_policy": "all_dns_answers_public_v1",
    },
}

_DOCLING_PARSER_BASE_V2 = {
    **_DOCLING_PARSER_BASE_V1,
    "pdf_pipeline": "progress_standard_segmented_v1",
    "accelerator": "cpu_single_thread_m4_benchmarked_v1",
    "ocr_batch_size": 1,
    "layout_batch_size": 1,
    "table_batch_size": 1,
    "table_structure_mode": "accurate",
    "pdf_segment_pages": 20,
    "pdf_segment_timeout_seconds": 180,
    "pdf_total_timeout_seconds": 1800,
    "progress_protocol": "pdf_parsing_progress_v1",
    "checkpoint_protocol": "docling_page_range_json_v1",
}

DOCLING_TEXT_PARSER_CONFIG_V2 = {
    **_DOCLING_PARSER_BASE_V2,
    "profile": "docling_text_local_v2",
    "generate_page_images": False,
    "generate_picture_images": False,
    "asset_mapping": "none",
}

DOCLING_MULTIMODAL_PARSER_CONFIG_V3 = {
    **_DOCLING_PARSER_BASE_V2,
    "profile": "docling_multimodal_local_v3",
    "generate_page_images": True,
    "generate_picture_images": True,
    "asset_mapping": "docling_picture_table_page_v1",
    "page_image_policy": "scanned_surface_v1",
    "markdown_media": deepcopy(
        DOCLING_MULTIMODAL_PARSER_CONFIG_V2["markdown_media"]
    ),
}

_DOCLING_PARSER_BASE_V3 = {
    key: value
    for key, value in _DOCLING_PARSER_BASE_V2.items()
    if key not in {
        "document_timeout_seconds",
        "pdf_segment_timeout_seconds",
        "pdf_total_timeout_seconds",
    }
}
_DOCLING_PARSER_BASE_V3["time_limit_policy"] = (
    "unbounded_with_long_running_notice_v1"
)

DOCLING_TEXT_PARSER_CONFIG_V3 = {
    **_DOCLING_PARSER_BASE_V3,
    "profile": "docling_text_local_v3",
    "generate_page_images": False,
    "generate_picture_images": False,
    "asset_mapping": "none",
}

DOCLING_MULTIMODAL_PARSER_CONFIG_V4 = {
    **_DOCLING_PARSER_BASE_V3,
    "profile": "docling_multimodal_local_v4",
    "generate_page_images": True,
    "generate_picture_images": True,
    "asset_mapping": "docling_picture_table_page_v1",
    "page_image_policy": "scanned_surface_v1",
    "markdown_media": deepcopy(
        DOCLING_MULTIMODAL_PARSER_CONFIG_V3["markdown_media"]
    ),
}

DOCLING_TEXT_PARSER_CONFIG = {
    **DOCLING_TEXT_PARSER_CONFIG_V3,
    "profile": "docling_text_local_v4",
    "resource_policy": "bounded_xlsx_pdf_checkpoint_v2",
    "pdf_probe": "isolated_pdf_page_facts_v2",
}

DOCLING_MULTIMODAL_PARSER_CONFIG = {
    **DOCLING_MULTIMODAL_PARSER_CONFIG_V4,
    "profile": "docling_multimodal_local_v5",
    "resource_policy": "bounded_xlsx_pdf_checkpoint_v2",
    "pdf_probe": "isolated_pdf_page_facts_v2",
    "page_image_policy": "image_coverage_surface_v2",
    "asset_mapping": "docling_picture_table_crop_page_v2",
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

# The native Docling structural profile.
STRUCTURAL_CHUNKING_CONFIG_V4 = {
    "profile": "structural_by_title_token_v4",
    "strategy": "docling_structural",
    "consumer_projection": "docling_inline_group_atoms_v1",
    "max_tokens": 800,
    "new_after_n_tokens": 600,
    **CHUNK_TOKENIZER,
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

SEMANTIC_CHUNKING_CONFIG_V3 = {
    "profile": "semantic_breakpoint_v3",
    "preset": "semantic_balanced_v1",
    "strategy": "semantic_breakpoint",
    "consumer_projection": "docling_effective_role_container_v1",
    "analysis_embedding": "index_embedding_space",
    "required_embedding_roles": ["semantic_analysis"],
    **CHUNK_TOKENIZER,
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
    "selector": "constrained_dp_section_merge_v2",
    "section_boundary_policy": "merge_under_min_adjacent_v1",
    "size_penalty_weight_micros": 150000,
    "preserve_page_boundaries": True,
    "preserve_table_boundaries": True,
    "attach_title_to_following": True,
    "normal_overlap_tokens": 0,
    "oversized_element_overlap_tokens": 50,
    "metadata_policy": "bounded_v3",
}

SEMANTIC_CHUNKING_CONFIG_V4 = {
    **SEMANTIC_CHUNKING_CONFIG_V3,
    "profile": "semantic_breakpoint_v4",
    "consumer_projection": "docling_effective_role_internal_record_v2",
    "internal_record_boundary_policy": "blank_line_heading_record_v1",
}

# Legacy dictionaries above stay byte-for-byte executable for existing revisions.
STRUCTURAL_CHUNKING_CONFIG = {
    **STRUCTURAL_CHUNKING_CONFIG_V4,
    "profile": "structural_by_title_token_v5",
    "heading_policy": "attach_bounded_fragments_v1",
    "caption_policy": "text_only_source_caption_v1",
    "table_serialization": "docling_rows_repeated_headers_v2",
}

SEMANTIC_CHUNKING_CONFIG = {
    **SEMANTIC_CHUNKING_CONFIG_V4,
    "profile": "semantic_breakpoint_v5",
    "embedding_table_serialization": "compact_markdown_cells_v1",
    "source_preservation": "source_spans_v1",
    "caption_policy": "text_only_source_caption_v1",
    "table_serialization": "docling_rows_repeated_headers_v2",
    "analysis_scope": "oversized_hard_regions_v1",
    "oversized_element_overlap_tokens": 0,
}

def profile_for_preset(
    preset: ChunkingPreset | str,
    parsing_preset: ParsingPreset | str = ParsingPreset.TEXT_LOCAL_V1,
) -> IndexProfileDefinition:
    """Return fresh JSON-compatible documents for persistence."""

    resolved = ChunkingPreset(preset)
    chunking = (
        STRUCTURAL_CHUNKING_CONFIG
        if resolved is ChunkingPreset.STRUCTURAL_BALANCED_V2
        else SEMANTIC_CHUNKING_CONFIG
    )
    parsing = ParsingPreset(parsing_preset)
    multimodal = parsing is ParsingPreset.MULTIMODAL_LOCAL_V2
    parser_config = (
        DOCLING_MULTIMODAL_PARSER_CONFIG
        if multimodal
        else DOCLING_TEXT_PARSER_CONFIG
    )
    return IndexProfileDefinition(
        parser_config=deepcopy(parser_config),
        chunking_config=deepcopy(chunking),
        enrichment_config=deepcopy(DOCLING_ENRICHMENT_CONFIG) if multimodal else {},
        representation_config=(
            deepcopy(DOCLING_REPRESENTATION_CONFIG) if multimodal else {}
        ),
    )


def index_profile() -> IndexProfileDefinition:
    """Return the default current structural profile."""

    return profile_for_preset(ChunkingPreset.STRUCTURAL_BALANCED_V2)


def resolve(
    parser_config: dict,
    chunking_config: dict,
) -> ChunkingStrategyKind:
    """Fail closed unless the complete persisted profile exactly matches a preset.

    Current profiles and their exact legacy predecessors remain executable.
    """

    if not any(parser_config == known for known, _ in _EXECUTABLE_PARSER_CONFIGS):
        raise ValueError("unknown parser profile")
    if chunking_config in (STRUCTURAL_CHUNKING_CONFIG, STRUCTURAL_CHUNKING_CONFIG_V4):
        return ChunkingStrategyKind.STRUCTURAL
    if chunking_config in (SEMANTIC_CHUNKING_CONFIG, SEMANTIC_CHUNKING_CONFIG_V4, SEMANTIC_CHUNKING_CONFIG_V3):
        return ChunkingStrategyKind.SEMANTIC
    raise ValueError("unknown chunking profile")


def parsing_preset(parser_config: dict) -> ParsingPreset:
    return parser_profile(parser_config).preset


def parser_profile(parser_config: dict) -> ParserProfile:
    for known, profile in _EXECUTABLE_PARSER_CONFIGS:
        if parser_config == known:
            return profile
    raise ValueError("unknown parser profile")


def public_parsing_descriptor(parser_config: dict) -> dict[str, str]:
    preset = parsing_preset(parser_config)
    return {"preset": preset.value, "profile": parser_config["profile"]}


def public_descriptor(chunking_config: dict) -> dict[str, str]:
    if chunking_config in (STRUCTURAL_CHUNKING_CONFIG, STRUCTURAL_CHUNKING_CONFIG_V4):
        return {
            "preset": ChunkingPreset.STRUCTURAL_BALANCED_V2.value,
            "profile": str(chunking_config["profile"]),
        }
    if chunking_config in (SEMANTIC_CHUNKING_CONFIG, SEMANTIC_CHUNKING_CONFIG_V4, SEMANTIC_CHUNKING_CONFIG_V3):
        return {
            "preset": ChunkingPreset.SEMANTIC_BALANCED_V1.value,
            "profile": str(chunking_config["profile"]),
        }
    raise ValueError("unknown chunking profile")


_EXECUTABLE_PARSER_CONFIGS = (
    (DOCLING_TEXT_PARSER_CONFIG_V1, ParserProfile.DOCLING_TEXT_LOCAL_V1),
    (
        DOCLING_MULTIMODAL_PARSER_CONFIG_V2,
        ParserProfile.DOCLING_MULTIMODAL_LOCAL_V2,
    ),
    (DOCLING_TEXT_PARSER_CONFIG_V2, ParserProfile.DOCLING_TEXT_LOCAL_V2),
    (
        DOCLING_MULTIMODAL_PARSER_CONFIG_V3,
        ParserProfile.DOCLING_MULTIMODAL_LOCAL_V3,
    ),
    (DOCLING_TEXT_PARSER_CONFIG_V3, ParserProfile.DOCLING_TEXT_LOCAL_V3),
    (DOCLING_TEXT_PARSER_CONFIG, ParserProfile.DOCLING_TEXT_LOCAL_V4),
    (
        DOCLING_MULTIMODAL_PARSER_CONFIG_V4,
        ParserProfile.DOCLING_MULTIMODAL_LOCAL_V4,
    ),
    (DOCLING_MULTIMODAL_PARSER_CONFIG, ParserProfile.DOCLING_MULTIMODAL_LOCAL_V5),
)


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
