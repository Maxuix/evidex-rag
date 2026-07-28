"""Versioned, deterministic lexical analysis for PostgreSQL FTS."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID


LEXICAL_ANALYZER_VERSION = "lexical_simple_cjk_bigram_v1"
LEXICAL_QUERY_VERSION = "lexical_or_query_v1"
MAX_DOCUMENT_LEXEMES = 4096
MAX_QUERY_LEXEMES = 64

_CJK_RUN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]+")
_COMPOUND = re.compile(
    r"(?<![\w])[\w]+(?:[-./:]+[\w]+)+(?![\w])",
    re.UNICODE,
)
_WORD = re.compile(r"[\w]+", re.UNICODE)
_SAFE_LEXEME = re.compile(r"^\w+$", re.UNICODE)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "by",
        "for",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "与",
        "和",
        "在",
        "是",
        "的",
        "了",
        "或",
        "及",
    }
)


@dataclass(frozen=True, slots=True)
class LexicalDocument:
    lexemes: tuple[str, ...]
    lexical_text: str
    lexical_text_hash: str


def analyze_document(value: str) -> LexicalDocument | None:
    lexemes = analyze_lexemes(value, limit=MAX_DOCUMENT_LEXEMES)
    if not lexemes:
        return None
    lexical_text = " ".join(lexemes)
    return LexicalDocument(
        lexemes=lexemes,
        lexical_text=lexical_text,
        lexical_text_hash=hashlib.sha256(lexical_text.encode("utf-8")).hexdigest(),
    )


def analyze_query(value: str) -> tuple[str, ...]:
    return analyze_lexemes(value, limit=MAX_QUERY_LEXEMES)


def analyze_lexemes(value: str, *, limit: int) -> tuple[str, ...]:
    if limit < 1:
        raise ValueError("lexeme limit must be positive")
    normalized = unicodedata.normalize("NFC", value).casefold()
    candidates: list[tuple[int, int, tuple[str, ...]]] = []
    compound_spans: list[tuple[int, int]] = []
    for match in _COMPOUND.finditer(normalized):
        parts = tuple(
            part for part in re.split(r"[-./:]+", match.group()) if part
        )
        compound = "_".join(parts)
        values = ((compound,) if compound else ()) + parts
        candidates.append((match.start(), 0, values))
        compound_spans.append(match.span())
    for match in _CJK_RUN.finditer(normalized):
        run = match.group()
        values = (run,) if len(run) == 1 else tuple(
            run[index : index + 2] for index in range(len(run) - 1)
        )
        candidates.append((match.start(), 1, values))
    for match in _WORD.finditer(normalized):
        if _inside_any(match.span(), compound_spans) or _CJK_RUN.fullmatch(
            match.group()
        ):
            continue
        candidates.append((match.start(), 2, (match.group(),)))

    observed: set[str] = set()
    result: list[str] = []
    for _position, _priority, values in sorted(candidates):
        for candidate in values:
            if (
                candidate in observed
                or candidate in _STOPWORDS
                or not _SAFE_LEXEME.fullmatch(candidate)
            ):
                continue
            observed.add(candidate)
            result.append(candidate)
            if len(result) >= limit:
                return tuple(result)
    return tuple(result)


def build_or_tsquery(lexemes: Iterable[str]) -> str | None:
    values = tuple(dict.fromkeys(lexemes))
    if not values:
        return None
    if len(values) > MAX_QUERY_LEXEMES or any(
        _SAFE_LEXEME.fullmatch(value) is None for value in values
    ):
        raise ValueError("tsquery lexemes must be analyzer-produced safe values")
    return " | ".join(f"'{value}'" for value in values)


def lexical_manifest_hash(
    analyzer_version: str,
    rows: Iterable[tuple[UUID, str]],
) -> str:
    digest = hashlib.sha256()
    digest.update(analyzer_version.encode("ascii"))
    digest.update(b"\n")
    for chunk_id, lexical_text_hash in sorted(rows, key=lambda item: item[0].int):
        digest.update(str(chunk_id).encode("ascii"))
        digest.update(b":")
        digest.update(lexical_text_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _inside_any(
    span: tuple[int, int],
    containers: Iterable[tuple[int, int]],
) -> bool:
    return any(left <= span[0] and span[1] <= right for left, right in containers)
