"""Structural chunk assembly read directly from a ``DoclingDocument``.

Boundaries come from Docling's own headings, tables and surfaces; the frozen
current frozen profiles add token budgets and source context. No parser
or loader produces chunks any more.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import TableItem

from rag_kb.document_processing.docling.provenance import (
    item_surfaces,
    project_source_location,
    surface_kind,
)
from rag_kb.document_processing.docling.traversal import (
    ChunkingItem,
    ItemKind,
    common_hierarchy,
    iterate_chunking_items,
    section_paths,
    table_html,
)
from rag_kb.document_processing.profiles import STRUCTURAL_CHUNKING_CONFIG, STRUCTURAL_CHUNKING_CONFIG_V4
from rag_kb.document_processing.docling.table_chunks import table_chunks, table_header_rows
from rag_kb.document_processing.tokenization import count_chunk_tokens, split_by_tokens
from rag_kb.domain import (
    ChunkAssemblyDraft,
    ErrorCode,
    ParserExecutionError,
    ParserLimits,
)


@dataclass(frozen=True, slots=True)
class _Entry:
    """One item's contribution to the region being packed.

    A picture or an author caption contributes an empty text: it never spends
    the chunk's token budget, but its reference stays in reading order so the
    relation builder can see that the visual sits inside this chunk's span.
    """

    refs: tuple[str, ...]
    text: str
    heading: bool = False


def assemble_structural(
    document: DoclingDocument,
    limits: ParserLimits | None = None,
    *,
    surface_labels: Mapping[int, str] | None = None,
    chunking_config: Mapping[str, Any] | None = None,
    include_captions: bool = False,
) -> tuple[ChunkAssemblyDraft, ...]:
    """Assemble structural chunks without materializing a second document model."""

    current_profile = (chunking_config or STRUCTURAL_CHUNKING_CONFIG) == STRUCTURAL_CHUNKING_CONFIG
    resolved = limits or ParserLimits()
    paths = section_paths(document)
    kind = surface_kind(document)
    drafts: list[ChunkAssemblyDraft] = []
    region: list[_Entry] = []
    headings: list[_Entry] = []
    surface: int | None = None
    visuals = 0

    def flush() -> None:
        """Emit the open region, or hold a heading-only region for what follows."""

        nonlocal region, headings
        if region and not any(entry.text and not entry.heading for entry in region):
            # Headings introduce content, so they wait for it instead of
            # becoming chunks that answer nothing.
            headings.extend(region)
            region = []
            return
        parts, trailing = _region_parts([*headings, *region], attach_headings=current_profile)
        headings = []
        region = []
        for text, refs in parts:
            drafts.append(_draft(document, paths, text, refs, resolved, surface_labels))
            _require_chunk_limit(drafts, resolved)
        if trailing and drafts:
            # A region that ends with a visual keeps that reference on the
            # chunk it follows rather than dropping it.
            last = drafts[-1]
            drafts[-1] = _draft(
                document,
                paths,
                last.text,
                tuple(dict.fromkeys((*last.item_refs, *trailing))),
                resolved,
                surface_labels,
            )

    for item in iterate_chunking_items(document):
        item_kind = item.kind
        ordinal = _surface_ordinal(item, kind)
        if ordinal is not None and surface is not None and ordinal != surface:
            # The v4 profile keeps sections inside one surface so a citation
            # never claims a page the chunk only partially covers.
            flush()
        if ordinal is not None:
            surface = ordinal
        if item_kind in {ItemKind.TITLE, ItemKind.SECTION_HEADER}:
            flush()
            if item.text:
                region.append(_Entry(item.refs, item.text, heading=True))
            continue
        if item_kind is ItemKind.TABLE:
            flush()
            visuals += 1
            headings = _append_table(
                document, paths, drafts, item, headings, resolved, surface_labels, current_profile
            )
            _require_chunk_limit(drafts, resolved)
            continue
        if item_kind is ItemKind.PICTURE:
            visuals += 1
        if item_kind is ItemKind.PICTURE or (item_kind is ItemKind.CAPTION and not (current_profile and include_captions)):
            # A picture never becomes an empty chunk and an author caption never
            # becomes a context-free one; both stay reachable through the
            # reference they leave in the surrounding chunk.
            region.append(_Entry(item.refs, ""))
            continue
        if not item.text:
            continue
        region.append(_Entry(item.refs, item.text))
    flush()
    if headings:
        # Trailing headings introduced nothing; they are still document content.
        for text, refs in _region_parts(headings)[0]:
            drafts.append(_draft(document, paths, text, refs, resolved, surface_labels))
            _require_chunk_limit(drafts, resolved)

    if not drafts and not visuals:
        # An image-only document is legitimate under the multimodal preset, so
        # the contract is that a document yields something referencable — not
        # that it always yields text.
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "non_empty_structural_chunks"},
        )
    return tuple(drafts)


def _append_table(
    document: DoclingDocument,
    paths: dict[str, tuple[dict[str, Any], ...]],
    drafts: list[ChunkAssemblyDraft],
    item: ChunkingItem,
    headings: list[_Entry],
    limits: ParserLimits,
    surface_labels: Mapping[int, str] | None,
    current_profile: bool = False,
) -> list[_Entry]:
    """Emit a table as its own chunks; return the headings still unattached."""

    if len(item.items) != 1 or not isinstance(item.items[0], TableItem):
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "docling_table_item"},
        )
    table = item.items[0]
    text = item.text
    # Structural HTML is not persisted, but an oversized table is still a
    # bounded-resource failure rather than a silently text-only table.
    table_html(table, document, limits)
    if not text:
        return headings
    references = item.refs
    prefix = [entry.text for entry in headings if entry.text]
    if current_profile:
        parts = table_chunks(text, prefix="\n\n".join(prefix), header_rows=table_header_rows(table))
        refs = tuple(dict.fromkeys((*[ref for entry in headings for ref in entry.refs], *references)))
        for part in parts:
            drafts.append(_draft(document, paths, part, refs, limits, surface_labels))
        return []
    parts = _table_parts(text)
    attached = bool(prefix) and count_chunk_tokens(
        "\n\n".join((*prefix, parts[0]))
    ) <= _config_int("max_tokens")
    for index, part in enumerate(parts):
        if index == 0 and attached:
            drafts.append(
                _draft(
                    document,
                    paths,
                    "\n\n".join((*prefix, part)),
                    tuple(
                        dict.fromkeys(
                            (
                                *(
                                    reference
                                    for entry in headings
                                    for reference in entry.refs
                                ),
                                *references,
                            )
                        )
                    ),
                    limits,
                    surface_labels,
                )
            )
            continue
        drafts.append(
            _draft(document, paths, part, references, limits, surface_labels)
        )
    return [] if attached else headings


def _draft(
    document: DoclingDocument,
    paths: dict[str, tuple[dict[str, Any], ...]],
    text: str,
    refs: tuple[str, ...],
    limits: ParserLimits,
    surface_labels: Mapping[int, str] | None = None,
) -> ChunkAssemblyDraft:
    return ChunkAssemblyDraft(
        text=text,
        token_count=count_chunk_tokens(text),
        item_refs=refs,
        source_location=project_source_location(
            document, refs, limits, surface_labels=surface_labels
        ),
        hierarchy=common_hierarchy(tuple(paths.get(ref, ()) for ref in refs)),
    )


def _region_parts(
    region: list[_Entry], *, attach_headings: bool = False,
) -> tuple[tuple[tuple[str, tuple[str, ...]], ...], tuple[str, ...]]:
    """Apply the frozen token budget to one region of consecutive items.

    Returns the emitted parts plus the references of any trailing zero-text
    items, which belong to the chunk that precedes them.
    """

    maximum = _config_int("max_tokens")
    soft = _config_int("new_after_n_tokens")
    overlap = _config_int("overlap")
    parts: list[tuple[str, tuple[str, ...]]] = []
    current: list[_Entry] = []
    current_tokens = 0
    carried: tuple[str, ...] = ()

    def flush() -> None:
        nonlocal current, current_tokens, carried
        texts = [entry.text for entry in current if entry.text]
        refs = tuple(
            dict.fromkeys(
                (*carried, *(reference for entry in current for reference in entry.refs))
            )
        )
        current = []
        current_tokens = 0
        if not texts:
            carried = refs
            return
        carried = ()
        parts.append(("\n\n".join(texts), refs))

    for entry in region:
        if not entry.text:
            current.append(entry)
            continue
        tokens = count_chunk_tokens(entry.text)
        if attach_headings and current and all(not item.text or item.heading for item in current):
            prefix = "\n\n".join(item.text for item in current if item.text)
            if count_chunk_tokens(f"{prefix}\n\n{entry.text}") > maximum:
                budget = maximum - count_chunk_tokens(prefix) - 2
                if budget > 0:
                    pieces = split_by_tokens(entry.text, max_tokens=budget, overlap_tokens=overlap if budget > overlap else 0)
                    while pieces and any(count_chunk_tokens(f"{prefix}\n\n{piece}") > maximum for piece in pieces):
                        budget -= 1
                        pieces = split_by_tokens(entry.text, max_tokens=budget, overlap_tokens=overlap if budget > overlap else 0)
                    refs = tuple(dict.fromkeys((*carried, *(ref for item in current for ref in item.refs), *entry.refs)))
                    parts.extend((f"{prefix}\n\n{piece}", refs) for piece in pieces)
                    current = []
                    current_tokens = 0
                    carried = ()
                    continue
        if tokens > maximum:
            flush()
            pieces = split_by_tokens(
                entry.text,
                max_tokens=maximum,
                overlap_tokens=overlap,
            )
            for index, piece in enumerate(pieces):
                refs = (
                    tuple(dict.fromkeys((*carried, *entry.refs)))
                    if index == 0
                    else entry.refs
                )
                parts.append((piece, refs))
            if pieces:
                carried = ()
            continue
        proposed = "\n\n".join(
            (*(item.text for item in current if item.text), entry.text)
        )
        if current_tokens and (
            current_tokens >= soft or count_chunk_tokens(proposed) > maximum
        ):
            flush()
        current.append(entry)
        current_tokens = count_chunk_tokens(
            "\n\n".join(item.text for item in current if item.text)
        )
    flush()
    return tuple(parts), carried


def _table_parts(text: str) -> tuple[str, ...]:
    """Split an oversized table on row boundaries, repeating its header."""

    maximum = _config_int("max_tokens")
    if count_chunk_tokens(text) <= maximum:
        return (text,)
    lines = [line for line in text.splitlines() if line.strip()]
    header = lines[:2] if len(lines) > 2 and set(lines[1].strip()) <= set("|-: ") else []
    rows = lines[len(header) :]
    if not rows:
        return tuple(
            split_by_tokens(text, max_tokens=maximum, overlap_tokens=0)
        )
    parts: list[str] = []
    current: list[str] = []
    for row in rows:
        proposed = "\n".join((*header, *current, row))
        if current and count_chunk_tokens(proposed) > maximum:
            parts.append("\n".join((*header, *current)))
            current = []
        if count_chunk_tokens("\n".join((*header, row))) > maximum:
            parts.extend(
                split_by_tokens(row, max_tokens=maximum, overlap_tokens=0)
            )
            continue
        current.append(row)
    if current:
        parts.append("\n".join((*header, *current)))
    return tuple(part for part in parts if part.strip())


def _surface_ordinal(item: ChunkingItem, kind: str) -> int | None:
    surfaces = tuple(
        surface
        for source in item.items
        for surface in item_surfaces(source, kind=kind)
    )
    return min(surface.ordinal for surface in surfaces) if surfaces else None


def _require_chunk_limit(
    drafts: list[ChunkAssemblyDraft], limits: ParserLimits
) -> None:
    if len(drafts) > limits.max_chunks:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic={"limit_name": "max_chunks", "limit": limits.max_chunks},
        )


def _config_int(name: str) -> int:
    value = STRUCTURAL_CHUNKING_CONFIG_V4[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value
