"""Immutable parser and chunking identity for new index revisions."""

from __future__ import annotations

from copy import deepcopy

from rag_kb.domain import IndexProfileDefinition


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


def index_profile() -> IndexProfileDefinition:
    """Return fresh JSON-compatible documents for persistence."""

    return IndexProfileDefinition(
        parser_config=deepcopy(UNSTRUCTURED_PARSER_CONFIG),
        chunking_config=deepcopy(UNSTRUCTURED_CHUNKING_CONFIG),
    )
