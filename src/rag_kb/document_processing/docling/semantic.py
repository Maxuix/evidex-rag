"""Semantic analysis units and chunk assembly read from a ``DoclingDocument``.

Semantic units stay a project algorithm input, not a second document model:
they carry text, boundaries and Docling references only, so a chunk plan or an
asynchronous embedding round never extends the converted document's lifetime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Any

from docling_core.types.doc import DoclingDocument
from rag_kb.document_processing.docling.provenance import (
    item_surfaces,
    project_source_location,
    surface_kind,
)
from rag_kb.document_processing.docling.traversal import (
    ChunkingItem,
    ItemKind,
    canonical_text,
    common_hierarchy,
    iterate_chunking_items,
    section_paths,
)
from rag_kb.document_processing.profiles import SEMANTIC_CHUNKING_CONFIG, SEMANTIC_CHUNKING_CONFIG_V4
from rag_kb.document_processing.semantic_text import joined_units
from rag_kb.document_processing.tokenization import count_chunk_tokens, split_by_tokens
from rag_kb.domain import (
    ChunkAssemblyDraft,
    SemanticUnit,
    ErrorCode,
    IndexChunkPlan,
    ParserExecutionError,
    ParserLimits,
)


_SENTENCE_BREAK = re.compile(r"(?<=[。！？；.!?;])(?:[ \t]+|\n*)|\n+")
_RECORD_BREAK = re.compile(r"\n[ \t]*\n+")

_BOUNDARY_SECTION = "section"
_BOUNDARY_SURFACE = "page"
_BOUNDARY_TABLE = "table"
_BOUNDARY_BLOCK = "block"
_BOUNDARY_RECORD = "record"

_BLOCK_KINDS = frozenset({ItemKind.CODE, ItemKind.FORMULA})


@dataclass(frozen=True, slots=True)
class _Fragment:
    text: str
    token_count: int
    refs: tuple[str, ...]
    fragment_index: int
    fragment_count: int
    charspan: tuple[int, int] | None
    boundary: str | None


@dataclass(frozen=True, slots=True)
class _Piece:
    text: str
    boundary: str | None = None


def docling_semantic_units(
    document: DoclingDocument,
    limits: ParserLimits | None = None,
    *,
    surface_labels: Mapping[int, str] | None = None,
    chunking_config: Mapping[str, Any] | None = None,
    include_captions: bool = False,
) -> tuple[SemanticUnit, ...]:
    """Build bounded analysis units without provider or persistence I/O."""

    resolved = limits or ParserLimits()
    if (chunking_config or SEMANTIC_CHUNKING_CONFIG).get("source_preservation") == "source_spans_v1":
        return _source_units(document, resolved, surface_labels=surface_labels, include_captions=include_captions)
    kind = surface_kind(document)
    fragments: list[_Fragment] = []
    pending_titles: list[tuple[tuple[str, ...], str]] = []
    pending_context: list[str] = []
    previous_surface: int | None = None
    previous_container: str | None = None
    previous_kind: ItemKind | None = None
    has_previous_content = False
    recovered_record_has_body = False
    preserve_internal_records = (
        (chunking_config or SEMANTIC_CHUNKING_CONFIG).get(
            "internal_record_boundary_policy"
        )
        == "blank_line_heading_record_v1"
    )

    for item in iterate_chunking_items(document):
        item_kind = _effective_kind(item)
        if item_kind in {ItemKind.PICTURE, ItemKind.CAPTION}:
            # Visual and caption items never spend an analysis unit's token
            # budget, but their references stay attached to the neighbouring
            # unit so relation building can still see them.
            pending_context.extend(item.refs)
            continue
        if item_kind in {ItemKind.TITLE, ItemKind.SECTION_HEADER}:
            if item.text:
                pending_titles.append((item.refs, item.text))
            continue
        text = item.text
        if not text:
            continue

        surface = _surface_ordinal(item, kind)
        container = item.semantic_container
        boundary = _boundary(
            item_kind=item_kind,
            previous_kind=previous_kind,
            surface=surface,
            previous_surface=previous_surface,
            container=container,
            previous_container=previous_container,
            has_pending_titles=bool(pending_titles),
            has_previous_content=has_previous_content,
        )
        projection_record = preserve_internal_records and _record_projection(item)
        record_heading = preserve_internal_records and (
            projection_record or _looks_like_record_heading(item_kind, text)
        )
        if boundary is None and record_heading and recovered_record_has_body:
            boundary = _BOUNDARY_RECORD
            recovered_record_has_body = False

        refs = (
            *pending_context,
            *(
                reference
                for references, _title in pending_titles
                for reference in references
            ),
            *item.refs,
        )
        pending_context.clear()
        if pending_titles:
            text = "\n".join((*(title for _refs, title in pending_titles), text))
            pending_titles.clear()

        pieces = (
            tuple(_Piece(piece) for piece in _table_pieces(text))
            if item_kind is ItemKind.TABLE
            else _text_pieces(
                text,
                preserve_internal_records=preserve_internal_records,
            )
        )
        cursor = 0
        for position, piece in enumerate(pieces):
            start = text.find(piece.text, cursor)
            charspan = (
                (start, start + len(piece.text))
                if start >= 0 and len(pieces) > 1
                else None
            )
            if start >= 0:
                cursor = start + len(piece.text)
            fragments.append(
                _Fragment(
                    text=piece.text,
                    token_count=count_chunk_tokens(piece.text),
                    refs=refs,
                    fragment_index=position,
                    fragment_count=len(pieces),
                    charspan=charspan,
                    boundary=(boundary if position == 0 else piece.boundary),
                )
            )
        if surface is not None:
            previous_surface = surface
        previous_container = container
        previous_kind = item_kind
        has_previous_content = True
        if not record_heading or projection_record:
            recovered_record_has_body = True

    if not fragments:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            phase="semantic_analysis",
            diagnostic={"check": "non_empty_semantic_units"},
        )
    if pending_context:
        last = fragments[-1]
        fragments[-1] = replace(
            last,
            refs=tuple(dict.fromkeys((*last.refs, *pending_context))),
        )
    merged = _merge_short_fragments(fragments)
    units = tuple(
        SemanticUnit(
            ordinal=ordinal,
            text=fragment.text,
            token_count=fragment.token_count,
            item_refs=fragment.refs,
            source_location=_unit_location(
                document, fragment, resolved, surface_labels
            ),
            hard_boundary_before=fragment.boundary,
        )
        for ordinal, fragment in enumerate(merged)
    )
    _require_limits(units)
    return units


def assemble_semantic_chunks(
    document: DoclingDocument,
    units: tuple[SemanticUnit, ...],
    plan: IndexChunkPlan,
    limits: ParserLimits | None = None,
    *,
    surface_labels: Mapping[int, str] | None = None,
) -> tuple[ChunkAssemblyDraft, ...]:
    """Cut the analysis units at the immutable plan boundaries."""

    resolved = limits or ParserLimits()
    paths = section_paths(document)
    cuts = (*[item.after_unit_ordinal + 1 for item in plan.boundaries], len(units))
    maximum = _config_int("max_chunk_tokens")
    drafts: list[ChunkAssemblyDraft] = []
    start = 0
    for end in cuts:
        selected = units[start:end]
        if not selected:
            raise _failed("empty_plan_range")
        text = joined_units(selected)
        if not selected[0].source_preserving:
            text = text.strip()
        token_count = count_chunk_tokens(text)
        if not text.strip() or token_count > maximum:
            raise _failed("assembled_chunk_token_limit")
        refs = tuple(
            dict.fromkeys(
                reference for unit in selected for reference in unit.item_refs
            )
        )
        drafts.append(
            ChunkAssemblyDraft(
                text=text,
                token_count=token_count,
                item_refs=refs,
                source_location=project_source_location(
                    document, refs, resolved, surface_labels=surface_labels
                ),
                hierarchy=common_hierarchy(
                    tuple(paths.get(reference, ()) for reference in refs)
                ),
            )
        )
        start = end
    if start != len(units) or len(drafts) != plan.chunk_count:
        raise _failed("plan_coverage")
    return tuple(drafts)


def docling_unit_sequence_hash(units: tuple[SemanticUnit, ...]) -> str:
    """Hash the complete unit projection that a chunk plan is bound to."""

    projection = [
        {
            "ordinal": unit.ordinal,
            "text_sha256": hashlib.sha256(unit.text.encode("utf-8")).hexdigest(),
            "token_count": unit.token_count,
            "item_refs": list(unit.item_refs),
            "source_location": unit.source_location,
            "hard_boundary_before": unit.hard_boundary_before,
            **({"separator_before": unit.separator_before} if unit.separator_before != "\n\n" else {}),
            **({"content_kind": unit.content_kind} if unit.content_kind != "text" else {}),
            **({"source_preserving": True} if unit.source_preserving else {}),
        }
        for unit in units
    ]
    return hashlib.sha256(
        json.dumps(
            projection,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _boundary(
    *,
    item_kind: ItemKind,
    previous_kind: ItemKind | None,
    surface: int | None,
    previous_surface: int | None,
    container: str | None,
    previous_container: str | None,
    has_pending_titles: bool,
    has_previous_content: bool,
) -> str | None:
    if (
        previous_surface is not None
        and surface is not None
        and surface != previous_surface
    ):
        return _BOUNDARY_SURFACE
    if item_kind is ItemKind.TABLE or previous_kind is ItemKind.TABLE:
        return _BOUNDARY_TABLE
    if item_kind in _BLOCK_KINDS or previous_kind in _BLOCK_KINDS:
        return _BOUNDARY_BLOCK
    if has_pending_titles:
        return _BOUNDARY_SECTION
    if has_previous_content and container != previous_container:
        return _BOUNDARY_SECTION
    return None


def _unit_location(
    document: DoclingDocument,
    fragment: _Fragment,
    limits: ParserLimits,
    surface_labels: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    location = project_source_location(
        document, fragment.refs, limits, surface_labels=surface_labels
    )
    if fragment.fragment_count > 1:
        detail: dict[str, Any] = {
            "index": fragment.fragment_index,
            "count": fragment.fragment_count,
        }
        if fragment.charspan is not None:
            detail["charspan"] = list(fragment.charspan)
        location = {**location, "fragment": detail}
    return location


def _text_pieces(
    text: str,
    *,
    preserve_internal_records: bool,
) -> tuple[_Piece, ...]:
    blocks = _record_blocks(text) if preserve_internal_records else ((text, None),)
    pieces: list[_Piece] = []
    for block, block_boundary in blocks:
        sentences = tuple(
            part
            for value in _SENTENCE_BREAK.split(canonical_text(block))
            if (part := canonical_text(value))
        )
        for sentence_index, sentence in enumerate(sentences):
            split_sentences = (
                (sentence,)
                if count_chunk_tokens(sentence) <= _max_unit_tokens()
                else _split(sentence)
            )
            for split_index, split_sentence in enumerate(split_sentences):
                pieces.append(
                    _Piece(
                        split_sentence,
                        block_boundary
                        if sentence_index == 0 and split_index == 0
                        else None,
                    )
                )
    return tuple(pieces)


def _record_blocks(text: str) -> tuple[tuple[str, str | None], ...]:
    raw_blocks = tuple(
        block
        for value in _RECORD_BREAK.split(canonical_text(text))
        if (block := canonical_text(value))
    )
    if len(raw_blocks) < 2:
        return ((canonical_text(text), None),)
    blocks = raw_blocks
    if _standalone_heading(blocks[0]) and _heading_bearing_record(blocks[1]):
        blocks = (canonical_text(f"{blocks[0]}\n{blocks[1]}"), *blocks[2:])
    return tuple(
        (
            block,
            _BOUNDARY_RECORD
            if index > 0 and _heading_bearing_record(block)
            else None,
        )
        for index, block in enumerate(blocks)
    )


def _standalone_heading(value: str) -> bool:
    return "\n" not in value and 0 < count_chunk_tokens(value) <= 24


def _heading_bearing_record(value: str) -> bool:
    first_line, separator, remainder = value.partition("\n")
    return bool(
        separator
        and canonical_text(remainder)
        and 0 < count_chunk_tokens(first_line) <= 24
        and len(first_line) <= 160
    )


def _looks_like_record_heading(item_kind: ItemKind, value: str) -> bool:
    canonical = canonical_text(value)
    return bool(
        item_kind is ItemKind.TEXT
        and canonical
        and "\n" not in canonical
        and len(canonical) <= 160
        and not re.search(r"[。！？；.!?;:]$", canonical)
    )


def _record_projection(item: ChunkingItem) -> bool:
    if len(item.items) < 2:
        return False
    first_value = getattr(item.items[0], "text", "")
    first = canonical_text(first_value if isinstance(first_value, str) else "")
    remainder = tuple(
        canonical_text(value if isinstance(value, str) else "")
        for child in item.items[1:]
        for value in (getattr(child, "text", ""),)
    )
    return bool(
        first
        and len(first) <= 160
        and not re.search(r"[。！？；.!?;:]$", first)
        and any(remainder)
    )


def _table_pieces(text: str) -> tuple[str, ...]:
    canonical = canonical_text(text)
    if count_chunk_tokens(canonical) <= _max_unit_tokens():
        return (canonical,)
    rows = tuple(row.strip() for row in canonical.splitlines() if row.strip())
    if len(rows) < 2:
        return _split(canonical)
    pieces: list[str] = []
    current: list[str] = []
    for row in rows:
        candidate = "\n".join((*current, row))
        if current and count_chunk_tokens(candidate) > _max_unit_tokens():
            pieces.append("\n".join(current))
            current = []
        if count_chunk_tokens(row) > _max_unit_tokens():
            pieces.extend(_split(row))
        else:
            current.append(row)
    if current:
        pieces.append("\n".join(current))
    return tuple(pieces)


def _split(text: str) -> tuple[str, ...]:
    return split_by_tokens(
        text,
        max_tokens=_max_unit_tokens(),
        overlap_tokens=_config_int("oversized_element_overlap_tokens"),
    )


def _merge_short_fragments(fragments: list[_Fragment]) -> list[_Fragment]:
    target = _config_int("analysis_unit_target_tokens")
    merged: list[_Fragment] = []
    for fragment in fragments:
        if not merged or fragment.boundary is not None:
            merged.append(fragment)
            continue
        prior = merged[-1]
        candidate = canonical_text(f"{prior.text} {fragment.text}")
        candidate_tokens = count_chunk_tokens(candidate)
        if candidate_tokens > target:
            merged.append(fragment)
            continue
        merged[-1] = _Fragment(
            text=candidate,
            token_count=candidate_tokens,
            refs=tuple(dict.fromkeys((*prior.refs, *fragment.refs))),
            fragment_index=0,
            # A merged unit no longer represents one item's split position, so
            # it stops advertising fragment identity.
            fragment_count=1,
            charspan=None,
            boundary=prior.boundary,
        )
    return merged


def _require_limits(units: tuple[SemanticUnit, ...]) -> None:
    max_units = _config_int("max_analysis_units")
    max_tokens = _config_int("max_analysis_tokens")
    if len(units) > max_units:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            phase="semantic_analysis",
            diagnostic={"limit_name": "max_analysis_units", "limit": max_units},
        )
    if sum(unit.token_count for unit in units) > max_tokens:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            phase="semantic_analysis",
            diagnostic={"limit_name": "max_analysis_tokens", "limit": max_tokens},
        )
    if any(unit.token_count > (800 if unit.content_kind == "table" else _max_unit_tokens()) for unit in units):
        raise ParserExecutionError(
            ErrorCode.SEMANTIC_CHUNKING_FAILED,
            phase="semantic_analysis",
            diagnostic={"check": "analysis_unit_token_limit"},
        )


def _surface_ordinal(item: ChunkingItem, kind: str) -> int | None:
    surfaces = tuple(
        surface
        for source in item.items
        for surface in item_surfaces(source, kind=kind)
    )
    return min(surface.ordinal for surface in surfaces) if surfaces else None


def _effective_kind(item: ChunkingItem) -> ItemKind:
    if item.inline and item.kind in _BLOCK_KINDS:
        return ItemKind.TEXT
    return item.kind


def _max_unit_tokens() -> int:
    return _config_int("analysis_unit_max_tokens")


def _config_int(name: str) -> int:
    value = SEMANTIC_CHUNKING_CONFIG_V4[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _failed(check: str) -> ParserExecutionError:
    return ParserExecutionError(
        ErrorCode.SEMANTIC_CHUNKING_FAILED,
        phase="semantic_analysis",
        diagnostic={"check": check},
    )


# Split only at explicit sentence/line boundaries. Source separators are retained;
# punctuation such as decimals and URLs never gets reconstructed with spaces.
_SOURCE_BREAK = re.compile(r"(?<=[。！？；])|(?<=[.!?;])(?=\s)|\n+")


def _source_pieces(text: str, *, code: bool = False) -> tuple[str, ...]:
    """Return disjoint substrings whose concatenation is exactly the input."""
    # Code is split at lines, never at punctuation inside expressions/strings.
    breaks = re.finditer(r"\n+", text) if code else _SOURCE_BREAK.finditer(text)
    spans: list[str] = []
    cursor = 0
    for match in breaks:
        if match.end() > cursor:
            spans.append(text[cursor:match.end()])
            cursor = match.end()
    if cursor < len(text):
        spans.append(text[cursor:])
    pieces: list[str] = []
    for span in spans:
        if count_chunk_tokens(span) <= _max_unit_tokens():
            pieces.append(span)
            continue
        # The shared Unicode-safe splitter trims windows; recover each exact gap
        # from the source, then keep it with the following source substring.
        offset = 0
        for part in split_by_tokens(span, max_tokens=_max_unit_tokens() - 8, overlap_tokens=0):
            start = span.find(part, offset)
            if start < 0:
                raise _failed("source_piece_span")
            end = start + len(part)
            raw = span[offset:end]
            if count_chunk_tokens(raw) > _max_unit_tokens():
                # Unusually long whitespace runs are source too. Split by characters
                # with token validation rather than silently dropping indentation.
                pieces.extend(_bounded_source(raw))
            else:
                pieces.append(raw)
            offset = end
        if offset < len(span):
            pieces.extend(_bounded_source(span[offset:]))
    if "".join(pieces) != text:
        raise _failed("source_piece_coverage")
    return tuple(pieces)


def _bounded_source(text: str) -> tuple[str, ...]:
    parts: list[str] = []
    while text:
        lo, hi = 1, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count_chunk_tokens(text[:mid]) <= _max_unit_tokens():
                lo = mid
            else:
                hi = mid - 1
        parts.append(text[:lo])
        text = text[lo:]
    return tuple(parts)


def _source_units(
    document: DoclingDocument, limits: ParserLimits, *,
    surface_labels: Mapping[int, str] | None, include_captions: bool,
) -> tuple[SemanticUnit, ...]:
    """Current unit projection; legacy projection above remains executable."""
    from rag_kb.document_processing.docling.table_chunks import table_chunks, table_header_rows
    from docling_core.types.doc.document import TableItem

    units: list[SemanticUnit] = []
    titles: list[tuple[tuple[str, ...], str]] = []
    context: list[str] = []
    previous_surface = None
    previous_container = None
    previous_kind = None
    record_has_body = False
    kind = surface_kind(document)

    def append(text: str, refs: tuple[str, ...], boundary: str | None,
               separator: str, content_kind: str, span: tuple[int, int] | None = None) -> None:
        location = project_source_location(document, refs, limits, surface_labels=surface_labels)
        if span is not None:
            location = {**location, "source_span": list(span)}
        unit = SemanticUnit(len(units), text, count_chunk_tokens(text), refs,
                            location, boundary, separator, content_kind, source_preserving=True)
        if units and boundary is None and content_kind != "table":
            prior = units[-1]
            combined = prior.text + separator + text
            if prior.content_kind == content_kind and count_chunk_tokens(combined) <= _config_int("analysis_unit_target_tokens"):
                merged_refs = tuple(dict.fromkeys((*prior.item_refs, *refs)))
                units[-1] = replace(prior, text=combined, token_count=count_chunk_tokens(combined),
                                    item_refs=merged_refs,
                                    source_location=project_source_location(document, merged_refs, limits, surface_labels=surface_labels))
                return
        units.append(unit)

    for item in iterate_chunking_items(document):
        item_kind = _effective_kind(item)
        if item_kind is ItemKind.PICTURE or (item_kind is ItemKind.CAPTION and not include_captions):
            context.extend(item.refs)
            continue
        if item_kind in {ItemKind.TITLE, ItemKind.SECTION_HEADER}:
            if item.text:
                titles.append((item.refs, item.text))
            continue
        if not item.text:
            continue
        surface = _surface_ordinal(item, kind)
        boundary = _boundary(item_kind=item_kind, previous_kind=previous_kind,
                             surface=surface, previous_surface=previous_surface,
                             container=item.semantic_container, previous_container=previous_container,
                             has_pending_titles=bool(titles), has_previous_content=bool(units))
        record = _record_projection(item) or _looks_like_record_heading(item_kind, item.text)
        if boundary is None and record and record_has_body:
            boundary = _BOUNDARY_RECORD
            record_has_body = False
        refs = tuple(dict.fromkeys((*context, *(ref for rr, _ in titles for ref in rr), *item.refs)))
        prefix = "\n".join(title for _, title in titles)
        context.clear()
        titles.clear()
        if item_kind is ItemKind.TABLE:
            table = item.items[0]
            if not isinstance(table, TableItem):
                raise _failed("docling_table_item")
            for index, part in enumerate(table_chunks(item.text, prefix=prefix, header_rows=table_header_rows(table))):
                append(part, refs, boundary if index == 0 else _BOUNDARY_TABLE, "\n\n", "table")
        else:
            text = f"{prefix}\n{item.text}" if prefix else item.text
            code = item_kind in _BLOCK_KINDS
            blocks = ((text, None),) if code else _source_record_blocks(text)
            cursor = 0
            first = True
            for block, record_boundary in blocks:
                start = text.find(block, cursor)
                if start < 0:
                    raise _failed("source_record_span")
                gap = text[cursor:start]
                pieces = _source_pieces(block, code=code)
                piece_start = start
                for i, piece in enumerate(pieces):
                    append(piece, refs, boundary if first else (record_boundary if i == 0 else None),
                           "\n\n" if first else (gap if i == 0 else ""),
                           "block" if code else "text", (piece_start, piece_start + len(piece)))
                    first = False
                    piece_start += len(piece)
                cursor = start + len(block)
        if surface is not None:
            previous_surface = surface
        previous_container = item.semantic_container
        previous_kind = item_kind
        if not record or _record_projection(item):
            record_has_body = True

    if titles:
        refs = tuple(dict.fromkeys((*context, *(ref for rr, _ in titles for ref in rr))))
        for index, part in enumerate(_source_pieces("\n".join(title for _, title in titles))):
            append(part, refs, _BOUNDARY_SECTION if index == 0 else None,
                   "\n\n" if index == 0 else "", "text")
        context.clear()
    if not units:
        raise ParserExecutionError(ErrorCode.PARSER_OUTPUT_INVALID, phase="semantic_analysis",
                                   diagnostic={"check": "non_empty_semantic_units"})
    if context:
        last = units[-1]
        refs = tuple(dict.fromkeys((*last.item_refs, *context)))
        units[-1] = replace(last, item_refs=refs,
                            source_location=project_source_location(document, refs, limits, surface_labels=surface_labels))
    result = tuple(units)
    _require_limits(result)
    return result


def _source_record_blocks(text: str) -> tuple[tuple[str, str | None], ...]:
    starts = [0, *(match.end() for match in _RECORD_BREAK.finditer(text))]
    spans = [(start, starts[index + 1] if index + 1 < len(starts) else len(text))
             for index, start in enumerate(starts)]
    blocks = []
    for index, (start, end) in enumerate(spans):
        block = text[start:end]
        boundary = _BOUNDARY_RECORD if index and _heading_bearing_record(canonical_text(block)) else None
        if index == 1 and _standalone_heading(canonical_text(text[:start])):
            boundary = None
        blocks.append((block, boundary))
    return tuple(blocks)
