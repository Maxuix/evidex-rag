"""Token counting shared by document chunking validation and persistence."""

from __future__ import annotations

import tiktoken

from rag_kb.document_processing.profiles import CHUNK_TOKENIZER
from rag_kb.tokenizer import (
    CL100K_BASE_ENCODING_NAME,
    get_cl100k_base_encoding,
)


def _encoding(name: str) -> tiktoken.Encoding:
    if name != CL100K_BASE_ENCODING_NAME:
        raise ValueError("unsupported chunking tokenizer")
    return get_cl100k_base_encoding()


def count_chunk_tokens(text: str) -> int:
    """Count tokens using the tokenizer frozen in the chunking profile."""

    tokenizer = CHUNK_TOKENIZER["tokenizer"]
    if not isinstance(tokenizer, str):
        raise TypeError("chunking tokenizer must be a string")
    return len(_encoding(tokenizer).encode(text))


def split_by_tokens(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> tuple[str, ...]:
    """Split canonical text into bounded, Unicode-safe token windows."""

    if max_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("invalid token split bounds")
    tokenizer = CHUNK_TOKENIZER["tokenizer"]
    if not isinstance(tokenizer, str):
        raise TypeError("chunking tokenizer must be a string")
    encoding = _encoding(tokenizer)
    tokens = encoding.encode(text)
    if not tokens:
        return ()

    decode_token = getattr(encoding, "decode_single_token_bytes", None)
    if not callable(decode_token):
        # Test doubles and non-tiktoken compatible implementations cannot
        # expose byte boundaries. Their decoded token windows retain the
        # historical behavior.
        step = max_tokens - overlap_tokens
        return tuple(
            part
            for start in range(0, len(tokens), step)
            if (part := encoding.decode(tokens[start : start + max_tokens]).strip())
        )

    token_bytes = tuple(decode_token(token) for token in tokens)
    raw = b"".join(token_bytes)
    if raw != text.encode("utf-8"):
        raise ValueError("token bytes do not reproduce the source text")
    offsets = [0]
    for value in token_bytes:
        offsets.append(offsets[-1] + len(value))

    def is_character_boundary(token_index: int) -> bool:
        offset = offsets[token_index]
        return offset == len(raw) or raw[offset] & 0xC0 != 0x80

    parts: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        while end > start and not is_character_boundary(end):
            end -= 1
        if end == start:
            raise ValueError("max_tokens cannot preserve a Unicode character")

        part = raw[offsets[start] : offsets[end]].decode("utf-8").strip()
        while part and len(encoding.encode(part)) > max_tokens:
            end -= 1
            while end > start and not is_character_boundary(end):
                end -= 1
            if end == start:
                raise ValueError("max_tokens cannot preserve a Unicode character")
            part = raw[offsets[start] : offsets[end]].decode("utf-8").strip()
        if part:
            parts.append(part)
        if end == len(tokens):
            break

        start = max(start + 1, end - overlap_tokens)
        while start < end and not is_character_boundary(start):
            start += 1

    return tuple(parts)
