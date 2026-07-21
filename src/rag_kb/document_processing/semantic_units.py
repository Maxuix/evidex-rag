"""Deterministic conversion from Unstructured elements to semantic units."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import replace

from rag_kb.document_processing.profiles import SEMANTIC_CHUNKING_CONFIG
from rag_kb.document_processing.tokenization import (
    count_chunk_tokens,
    split_by_tokens,
)
from rag_kb.domain import (
    ErrorCode,
    ParsedDocument,
    ParserExecutionError,
    SemanticUnit,
)


_SENTENCE_BREAK = re.compile(r"(?<=[。！？；.!?;])(?:[ \t]+|\n*)|\n+")


def semantic_units(document: ParsedDocument) -> tuple[SemanticUnit, ...]:
    """Create bounded units without provider or persistence I/O."""

    fragments: list[SemanticUnit] = []
    pending_titles: list[str] = []
    previous_page: int | None = None
    previous_table = False

    for element in document.elements:
        page = _page(element.source_location)
        if element.is_title:
            pending_titles.append(element.text)
            continue

        boundary: str | None = None
        if previous_page is not None and page is not None and page != previous_page:
            boundary = "page"
        elif element.is_table or previous_table:
            boundary = "table"
        elif pending_titles:
            boundary = "section"

        text = element.text
        hierarchy = dict(element.hierarchy)
        if pending_titles:
            text = "\n".join((*pending_titles, text))
            hierarchy = _with_titles(hierarchy, pending_titles)
            pending_titles.clear()

        pieces = (
            _table_pieces(text)
            if element.is_table
            else _text_pieces(text)
        )
        for position, piece in enumerate(pieces):
            fragments.append(
                SemanticUnit(
                    ordinal=len(fragments),
                    text=piece,
                    token_count=count_chunk_tokens(piece),
                    source_location=dict(element.source_location),
                    hierarchy=dict(hierarchy),
                    element_ordinals=(element.ordinal,),
                    hard_boundary_before=boundary if position == 0 else None,
                )
            )
        previous_page = page if page is not None else previous_page
        previous_table = element.is_table

    if not fragments:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            phase="semantic_analysis",
            diagnostic={"check": "non_empty_semantic_units"},
        )

    merged = _merge_short_fragments(fragments)
    limited = tuple(replace(unit, ordinal=index) for index, unit in enumerate(merged))
    _require_limits(limited)
    return limited


def unit_sequence_hash(units: tuple[SemanticUnit, ...]) -> str:
    projection = [
        {
            "ordinal": unit.ordinal,
            "text_sha256": hashlib.sha256(unit.text.encode("utf-8")).hexdigest(),
            "token_count": unit.token_count,
            "source_location": unit.source_location,
            "hierarchy": unit.hierarchy,
            "element_ordinals": list(unit.element_ordinals),
            "hard_boundary_before": unit.hard_boundary_before,
        }
        for unit in units
    ]
    canonical = json.dumps(
        projection,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _text_pieces(text: str) -> tuple[str, ...]:
    canonical = _canonical_text(text)
    sentences = tuple(
        part
        for value in _SENTENCE_BREAK.split(canonical)
        if (part := _canonical_text(value))
    )
    if not sentences:
        return ()
    pieces: list[str] = []
    for sentence in sentences:
        if count_chunk_tokens(sentence) <= _max_unit_tokens():
            pieces.append(sentence)
        else:
            pieces.extend(
                split_by_tokens(
                    sentence,
                    max_tokens=_max_unit_tokens(),
                    overlap_tokens=int(
                        SEMANTIC_CHUNKING_CONFIG[
                            "oversized_element_overlap_tokens"
                        ]
                    ),
                )
            )
    return tuple(pieces)


def _table_pieces(text: str) -> tuple[str, ...]:
    canonical = _canonical_text(text)
    if count_chunk_tokens(canonical) <= _max_unit_tokens():
        return (canonical,)
    rows = tuple(row.strip() for row in canonical.splitlines() if row.strip())
    if len(rows) < 2:
        return split_by_tokens(
            canonical,
            max_tokens=_max_unit_tokens(),
            overlap_tokens=int(
                SEMANTIC_CHUNKING_CONFIG["oversized_element_overlap_tokens"]
            ),
        )
    pieces: list[str] = []
    current: list[str] = []
    for row in rows:
        candidate = "\n".join((*current, row))
        if current and count_chunk_tokens(candidate) > _max_unit_tokens():
            pieces.append("\n".join(current))
            current = []
        if count_chunk_tokens(row) > _max_unit_tokens():
            pieces.extend(
                split_by_tokens(
                    row,
                    max_tokens=_max_unit_tokens(),
                    overlap_tokens=int(
                        SEMANTIC_CHUNKING_CONFIG[
                            "oversized_element_overlap_tokens"
                        ]
                    ),
                )
            )
        else:
            current.append(row)
    if current:
        pieces.append("\n".join(current))
    return tuple(pieces)


def _merge_short_fragments(fragments: list[SemanticUnit]) -> list[SemanticUnit]:
    target = int(SEMANTIC_CHUNKING_CONFIG["analysis_unit_target_tokens"])
    merged: list[SemanticUnit] = []
    for fragment in fragments:
        if not merged or fragment.hard_boundary_before is not None:
            merged.append(fragment)
            continue
        prior = merged[-1]
        candidate = _canonical_text(f"{prior.text} {fragment.text}")
        candidate_tokens = count_chunk_tokens(candidate)
        if candidate_tokens > target:
            merged.append(fragment)
            continue
        merged[-1] = SemanticUnit(
            ordinal=prior.ordinal,
            text=candidate,
            token_count=candidate_tokens,
            source_location=_merge_locations(
                prior.source_location, fragment.source_location
            ),
            hierarchy=_common_hierarchy(prior.hierarchy, fragment.hierarchy),
            element_ordinals=tuple(
                dict.fromkeys((*prior.element_ordinals, *fragment.element_ordinals))
            ),
            hard_boundary_before=prior.hard_boundary_before,
        )
    return merged


def _require_limits(units: tuple[SemanticUnit, ...]) -> None:
    max_units = int(SEMANTIC_CHUNKING_CONFIG["max_analysis_units"])
    max_tokens = int(SEMANTIC_CHUNKING_CONFIG["max_analysis_tokens"])
    if len(units) > max_units:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            phase="semantic_analysis",
            diagnostic={"limit_name": "max_analysis_units", "limit": max_units},
        )
    total = sum(unit.token_count for unit in units)
    if total > max_tokens:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            phase="semantic_analysis",
            diagnostic={"limit_name": "max_analysis_tokens", "limit": max_tokens},
        )
    if any(unit.token_count > _max_unit_tokens() for unit in units):
        raise ParserExecutionError(
            ErrorCode.SEMANTIC_CHUNKING_FAILED,
            phase="semantic_analysis",
            diagnostic={"check": "analysis_unit_token_limit"},
        )


def _max_unit_tokens() -> int:
    return int(SEMANTIC_CHUNKING_CONFIG["analysis_unit_max_tokens"])


def _canonical_text(value: str) -> str:
    return unicodedata.normalize(
        "NFC", value.replace("\r\n", "\n").replace("\r", "\n")
    ).strip()


def _page(location: dict) -> int | None:
    page = location.get("page_start")
    return page if isinstance(page, int) and not isinstance(page, bool) else None


def _with_titles(hierarchy: dict, titles: list[str]) -> dict:
    existing = hierarchy.get("titles")
    resolved = list(existing) if isinstance(existing, list) else []
    resolved.extend(
        {"depth": min(index, 32), "text": title[:256]}
        for index, title in enumerate(titles)
    )
    return {**hierarchy, "titles": resolved[-64:]}


def _merge_locations(left: dict, right: dict) -> dict:
    pages = [
        value
        for location in (left, right)
        for key in ("page_start", "page_end")
        if isinstance((value := location.get(key)), int)
        and not isinstance(value, bool)
    ]
    if not pages:
        return {}
    return {"page_start": min(pages), "page_end": max(pages)}


def _common_hierarchy(left: dict, right: dict) -> dict:
    left_titles = left.get("titles")
    right_titles = right.get("titles")
    if not isinstance(left_titles, list) or not isinstance(right_titles, list):
        return {}
    common: list = []
    for first, second in zip(left_titles, right_titles):
        if first != second:
            break
        common.append(first)
    return {"titles": common} if common else {}
