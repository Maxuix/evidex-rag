"""Token counting shared by document chunking validation and persistence."""

from __future__ import annotations

from functools import lru_cache

import tiktoken

from rag_kb.document_processing.profiles import UNSTRUCTURED_CHUNKING_CONFIG


@lru_cache(maxsize=None)
def _encoding(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def count_chunk_tokens(text: str) -> int:
    """Count tokens using the tokenizer frozen in the chunking profile."""

    tokenizer = UNSTRUCTURED_CHUNKING_CONFIG["tokenizer"]
    if not isinstance(tokenizer, str):
        raise TypeError("chunking tokenizer must be a string")
    return len(_encoding(tokenizer).encode(text))
