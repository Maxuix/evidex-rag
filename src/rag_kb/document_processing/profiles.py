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
    "profile": "unstructured_by_title_v1",
    "strategy": "by_title",
    "max_characters": 2_000,
    "new_after_n_chars": 1_800,
    "overlap": 200,
    "overlap_all": False,
    "combine_text_under_n_chars": 500,
    "multipage_sections": False,
    "include_orig_elements": True,
    "metadata_policy": "bounded_v1",
}


def index_profile() -> IndexProfileDefinition:
    """Return fresh JSON-compatible documents for persistence."""

    return IndexProfileDefinition(
        parser_config=deepcopy(UNSTRUCTURED_PARSER_CONFIG),
        chunking_config=deepcopy(UNSTRUCTURED_CHUNKING_CONFIG),
    )
