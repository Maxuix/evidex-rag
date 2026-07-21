"""Immutable parser/chunking preset registry for index revisions."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from rag_kb.domain import (
    ChunkingPreset,
    ChunkingStrategyKind,
    IndexProfileDefinition,
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
) -> IndexProfileDefinition:
    """Return fresh JSON-compatible documents for persistence."""

    resolved = ChunkingPreset(preset)
    chunking = (
        UNSTRUCTURED_CHUNKING_CONFIG
        if resolved is ChunkingPreset.STRUCTURAL_BALANCED_V2
        else SEMANTIC_CHUNKING_CONFIG
    )
    return IndexProfileDefinition(
        parser_config=deepcopy(UNSTRUCTURED_PARSER_CONFIG),
        chunking_config=deepcopy(chunking),
    )


def index_profile() -> IndexProfileDefinition:
    """Compatibility alias for the default structural preset."""

    return profile_for_preset(ChunkingPreset.STRUCTURAL_BALANCED_V2)


def resolve(
    parser_config: dict,
    chunking_config: dict,
) -> ChunkingStrategyKind:
    """Fail closed unless the complete persisted profile exactly matches a preset."""

    if parser_config != UNSTRUCTURED_PARSER_CONFIG:
        raise ValueError("unknown parser profile")
    if chunking_config == UNSTRUCTURED_CHUNKING_CONFIG:
        return ChunkingStrategyKind.STRUCTURAL
    if chunking_config == SEMANTIC_CHUNKING_CONFIG:
        return ChunkingStrategyKind.SEMANTIC
    raise ValueError("unknown chunking profile")


def public_descriptor(chunking_config: dict) -> dict[str, str]:
    if chunking_config == UNSTRUCTURED_CHUNKING_CONFIG:
        return {
            "preset": ChunkingPreset.STRUCTURAL_BALANCED_V2.value,
            "profile": UNSTRUCTURED_CHUNKING_CONFIG["profile"],
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
) -> str:
    canonical = json.dumps(
        {"parser_config": parser_config, "chunking_config": chunking_config},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
