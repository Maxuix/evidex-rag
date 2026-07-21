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


def split_by_tokens(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> tuple[str, ...]:
    """Split canonical text on frozen tokenizer boundaries."""

    if max_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("invalid token split bounds")
    tokenizer = UNSTRUCTURED_CHUNKING_CONFIG["tokenizer"]
    if not isinstance(tokenizer, str):
        raise TypeError("chunking tokenizer must be a string")
    encoding = _encoding(tokenizer)
    tokens = encoding.encode(text)
    if not tokens:
        return ()
    step = max_tokens - overlap_tokens
    return tuple(
        part
        for start in range(0, len(tokens), step)
        if (part := encoding.decode(tokens[start : start + max_tokens]).strip())
    )
