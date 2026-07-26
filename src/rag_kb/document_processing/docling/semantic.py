"""Semantic analysis units and chunk assembly read from a ``DoclingDocument``.

Semantic units stay a project algorithm input, not a second document model:
they carry text, boundaries and Docling references only, so a chunk plan or an
asynchronous embedding round never extends the converted document's lifetime.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Any

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocItem

from rag_kb.document_processing.docling.provenance import (
    item_surfaces,
    project_source_location,
    surface_kind,
)
from rag_kb.document_processing.docling.traversal import (
    ItemKind,
    canonical_text,
    classify_item,
    common_hierarchy,
    item_ref,
    item_text,
    iterate_body_items,
    parent_ref,
    section_paths,
)
from rag_kb.document_processing.profiles import SEMANTIC_CHUNKING_CONFIG
from rag_kb.document_processing.tokenization import count_chunk_tokens, split_by_tokens
from rag_kb.domain import (
    ChunkAssemblyDraft,
    DoclingSemanticUnit,
    ErrorCode,
    IndexChunkPlan,
    ParserExecutionError,
    ParserLimits,
)


_SENTENCE_BREAK = re.compile(r"(?<=[。！？；.!?;])(?:[ \t]+|\n*)|\n+")

_BOUNDARY_SECTION = "section"
_BOUNDARY_SURFACE = "page"
_BOUNDARY_TABLE = "table"
_BOUNDARY_BLOCK = "block"

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


def docling_semantic_units(
    document: DoclingDocument,
    limits: ParserLimits | None = None,
) -> tuple[DoclingSemanticUnit, ...]:
    """Build bounded analysis units without provider or persistence I/O."""

    resolved = limits or ParserLimits()
    kind = surface_kind(document)
    fragments: list[_Fragment] = []
    pending_titles: list[tuple[str, str]] = []
    pending_context: list[str] = []
    previous_surface: int | None = None
    previous_parent: str | None = None
    previous_kind: ItemKind | None = None

    for item, _level in iterate_body_items(document):
        item_kind = classify_item(item)
        if item_kind in {ItemKind.PICTURE, ItemKind.CAPTION}:
            # Visual and caption items never spend an analysis unit's token
            # budget, but their references stay attached to the neighbouring
            # unit so relation building can still see them.
            pending_context.append(item_ref(item))
            continue
        if item_kind in {ItemKind.TITLE, ItemKind.SECTION_HEADER}:
            heading = item_text(item, document)
            if heading:
                pending_titles.append((item_ref(item), heading))
            continue
        text = item_text(item, document)
        if not text:
            continue

        surface = _surface_ordinal(item, kind)
        parent = parent_ref(item)
        boundary = _boundary(
            item_kind=item_kind,
            previous_kind=previous_kind,
            surface=surface,
            previous_surface=previous_surface,
            parent=parent,
            previous_parent=previous_parent,
            has_pending_titles=bool(pending_titles),
        )

        refs = (
            *pending_context,
            *(reference for reference, _title in pending_titles),
            item_ref(item),
        )
        pending_context.clear()
        if pending_titles:
            text = "\n".join((*(title for _ref, title in pending_titles), text))
            pending_titles.clear()

        pieces = (
            _table_pieces(text)
            if item_kind is ItemKind.TABLE
            else _text_pieces(text)
        )
        cursor = 0
        for position, piece in enumerate(pieces):
            start = text.find(piece, cursor)
            charspan = (
                (start, start + len(piece)) if start >= 0 and len(pieces) > 1 else None
            )
            if start >= 0:
                cursor = start + len(piece)
            fragments.append(
                _Fragment(
                    text=piece,
                    token_count=count_chunk_tokens(piece),
                    refs=refs,
                    fragment_index=position,
                    fragment_count=len(pieces),
                    charspan=charspan,
                    boundary=boundary if position == 0 else None,
                )
            )
        if surface is not None:
            previous_surface = surface
        previous_parent = parent
        previous_kind = item_kind

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
        DoclingSemanticUnit(
            ordinal=ordinal,
            text=fragment.text,
            token_count=fragment.token_count,
            item_refs=fragment.refs,
            source_location=_unit_location(document, fragment, resolved),
            hard_boundary_before=fragment.boundary,
        )
        for ordinal, fragment in enumerate(merged)
    )
    _require_limits(units)
    return units


def assemble_semantic_chunks(
    document: DoclingDocument,
    units: tuple[DoclingSemanticUnit, ...],
    plan: IndexChunkPlan,
    limits: ParserLimits | None = None,
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
        text = "\n\n".join(unit.text for unit in selected).strip()
        token_count = count_chunk_tokens(text)
        if not text or token_count > maximum:
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
                source_location=project_source_location(document, refs, resolved),
                hierarchy=common_hierarchy(
                    tuple(paths.get(reference, ()) for reference in refs)
                ),
            )
        )
        start = end
    if start != len(units) or len(drafts) != plan.chunk_count:
        raise _failed("plan_coverage")
    return tuple(drafts)


def docling_unit_sequence_hash(units: tuple[DoclingSemanticUnit, ...]) -> str:
    """Hash the complete unit projection that a chunk plan is bound to."""

    projection = [
        {
            "ordinal": unit.ordinal,
            "text_sha256": hashlib.sha256(unit.text.encode("utf-8")).hexdigest(),
            "token_count": unit.token_count,
            "item_refs": list(unit.item_refs),
            "source_location": unit.source_location,
            "hard_boundary_before": unit.hard_boundary_before,
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
    parent: str | None,
    previous_parent: str | None,
    has_pending_titles: bool,
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
    if previous_parent is not None and parent is not None and parent != previous_parent:
        return _BOUNDARY_SECTION
    return None


def _unit_location(
    document: DoclingDocument,
    fragment: _Fragment,
    limits: ParserLimits,
) -> dict[str, Any]:
    location = project_source_location(document, fragment.refs, limits)
    if fragment.fragment_count > 1:
        detail: dict[str, Any] = {
            "index": fragment.fragment_index,
            "count": fragment.fragment_count,
        }
        if fragment.charspan is not None:
            detail["charspan"] = list(fragment.charspan)
        location = {**location, "fragment": detail}
    return location


def _text_pieces(text: str) -> tuple[str, ...]:
    canonical = canonical_text(text)
    sentences = tuple(
        part
        for value in _SENTENCE_BREAK.split(canonical)
        if (part := canonical_text(value))
    )
    if not sentences:
        return ()
    pieces: list[str] = []
    for sentence in sentences:
        if count_chunk_tokens(sentence) <= _max_unit_tokens():
            pieces.append(sentence)
        else:
            pieces.extend(_split(sentence))
    return tuple(pieces)


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


def _require_limits(units: tuple[DoclingSemanticUnit, ...]) -> None:
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
    if any(unit.token_count > _max_unit_tokens() for unit in units):
        raise ParserExecutionError(
            ErrorCode.SEMANTIC_CHUNKING_FAILED,
            phase="semantic_analysis",
            diagnostic={"check": "analysis_unit_token_limit"},
        )


def _surface_ordinal(item: DocItem, kind: str) -> int | None:
    surfaces = item_surfaces(item, kind=kind)
    return min(surface.ordinal for surface in surfaces) if surfaces else None


def _max_unit_tokens() -> int:
    return _config_int("analysis_unit_max_tokens")


def _config_int(name: str) -> int:
    value = SEMANTIC_CHUNKING_CONFIG[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _failed(check: str) -> ParserExecutionError:
    return ParserExecutionError(
        ErrorCode.SEMANTIC_CHUNKING_FAILED,
        phase="semantic_analysis",
        diagnostic={"check": check},
    )
