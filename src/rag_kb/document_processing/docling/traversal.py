"""The single Docling item traversal shared by every downstream consumer."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
import unicodedata

from docling_core.types.doc import DoclingDocument, DocItemLabel
from docling_core.types.doc.document import (
    DocItem,
    GroupItem,
    GroupLabel,
    InlineGroup,
    PictureItem,
    TableItem,
)

from rag_kb.domain import ErrorCode, ParserExecutionError, ParserLimits


_MAX_TITLE_CHARS = 256
_MAX_TITLE_DEPTH = 32
_MAX_TITLES = 64


class ItemKind(StrEnum):
    """The traversal's view of one Docling item, not a second document model."""

    TITLE = "title"
    SECTION_HEADER = "section_header"
    TEXT = "text"
    LIST_ITEM = "list_item"
    CODE = "code"
    FORMULA = "formula"
    TABLE = "table"
    PICTURE = "picture"
    CAPTION = "caption"
    FURNITURE = "furniture"
    UNKNOWN = "unknown"


_KIND_BY_LABEL = {
    DocItemLabel.TITLE: ItemKind.TITLE,
    DocItemLabel.SECTION_HEADER: ItemKind.SECTION_HEADER,
    DocItemLabel.TEXT: ItemKind.TEXT,
    DocItemLabel.PARAGRAPH: ItemKind.TEXT,
    DocItemLabel.REFERENCE: ItemKind.TEXT,
    DocItemLabel.FOOTNOTE: ItemKind.TEXT,
    DocItemLabel.HANDWRITTEN_TEXT: ItemKind.TEXT,
    DocItemLabel.LIST_ITEM: ItemKind.LIST_ITEM,
    DocItemLabel.CODE: ItemKind.CODE,
    DocItemLabel.FORMULA: ItemKind.FORMULA,
    DocItemLabel.TABLE: ItemKind.TABLE,
    DocItemLabel.DOCUMENT_INDEX: ItemKind.TABLE,
    DocItemLabel.PICTURE: ItemKind.PICTURE,
    DocItemLabel.CHART: ItemKind.PICTURE,
    DocItemLabel.CAPTION: ItemKind.CAPTION,
    DocItemLabel.PAGE_HEADER: ItemKind.FURNITURE,
    DocItemLabel.PAGE_FOOTER: ItemKind.FURNITURE,
}

#: Kinds whose text belongs in an assembled chunk. Captions stay out because
#: they are the author's description of a visual and are carried by relations;
#: pictures carry no text of their own.
TEXT_KINDS = frozenset(
    {
        ItemKind.TITLE,
        ItemKind.SECTION_HEADER,
        ItemKind.TEXT,
        ItemKind.LIST_ITEM,
        ItemKind.CODE,
        ItemKind.FORMULA,
        ItemKind.UNKNOWN,
    }
)

_WRAPPER_KINDS = frozenset(
    {ItemKind.TITLE, ItemKind.SECTION_HEADER, ItemKind.LIST_ITEM}
)
_TRANSPARENT_GROUP_LABELS = frozenset(
    {GroupLabel.INLINE, GroupLabel.LIST, GroupLabel.ORDERED_LIST}
)
_INLINE_TEXT_KINDS = frozenset(
    {ItemKind.TEXT, ItemKind.CODE, ItemKind.FORMULA}
)


@dataclass(frozen=True, slots=True)
class ChunkingItem:
    """One transient logical atom consumed only while chunking.

    ``refs`` and ``items`` always contain ordered DocItem leaves. Group refs and
    empty wrapper refs never cross the persistence boundary.
    """

    kind: ItemKind
    text: str
    refs: tuple[str, ...]
    items: tuple[DocItem, ...]
    header_depth: int | None
    semantic_container: str | None
    inline: bool = False


def classify_item(item: DocItem) -> ItemKind:
    """Map a Docling label onto the frozen traversal policy."""

    if isinstance(item, TableItem):
        return ItemKind.TABLE
    if isinstance(item, PictureItem):
        return ItemKind.PICTURE
    return _KIND_BY_LABEL.get(getattr(item, "label", None), ItemKind.UNKNOWN)


def iterate_body_items(document: DoclingDocument) -> Iterator[tuple[DocItem, int]]:
    """Yield body items in reading order, without groups and without furniture.

    ``iterate_items()`` already restricts itself to the body content layer and
    skips group nodes, so this stays the only place that decides what the
    project considers document content.
    """

    for item, level in document.iterate_items():
        if not isinstance(item, DocItem):
            continue
        if classify_item(item) is ItemKind.FURNITURE:
            continue
        yield item, level


def iterate_chunking_items(document: DoclingDocument) -> Iterator[ChunkingItem]:
    """Yield the internal-only logical projection used by both chunkers.

    Raw traversal remains unchanged for asset and relation consumers. This
    projection folds only the observed empty wrapper + InlineGroup shape and
    direct-body InlineGroups; every other group shape retains raw item order.
    """

    raw = tuple(item for item, _level in iterate_body_items(document))
    raw_refs = tuple(item_ref(item) for item in raw)
    raw_positions = {reference: index for index, reference in enumerate(raw_refs)}
    if len(raw_positions) != len(raw_refs):
        raise _invalid_projection("duplicate_docitem_ref")

    emissions: dict[str, ChunkingItem] = {}
    consumed: set[str] = set()
    claimed_groups: set[str] = set()

    for wrapper in raw:
        wrapper_kind = classify_item(wrapper)
        if (
            wrapper_kind not in _WRAPPER_KINDS
            or item_text(wrapper, document)
            or len(getattr(wrapper, "children", ()) or ()) != 1
        ):
            continue
        group = _resolve_child(document, wrapper.children[0])
        if not isinstance(group, InlineGroup):
            continue
        if parent_ref(group) != item_ref(wrapper):
            raise _invalid_projection("inline_wrapper_parent")
        projection = _inline_projection(
            document,
            group,
            kind=wrapper_kind,
            header_depth=(
                _header_depth(wrapper)
                if wrapper_kind in {ItemKind.TITLE, ItemKind.SECTION_HEADER}
                else None
            ),
            semantic_container=_semantic_container(document, wrapper),
        )
        _claim_projection(
            projection,
            emission_ref=item_ref(wrapper),
            raw_positions=raw_positions,
            emissions=emissions,
            consumed=consumed,
            claimed_groups=claimed_groups,
            group_ref=item_ref(group),
        )

    for group in getattr(document, "groups", ()) or ():
        if (
            not isinstance(group, InlineGroup)
            or parent_ref(group) != "#/body"
            or not (getattr(group, "children", ()) or ())
        ):
            continue
        projection = _inline_projection(
            document,
            group,
            kind=ItemKind.TEXT,
            header_depth=None,
            semantic_container=None,
        )
        _claim_projection(
            projection,
            emission_ref=projection.refs[0],
            raw_positions=raw_positions,
            emissions=emissions,
            consumed=consumed,
            claimed_groups=claimed_groups,
            group_ref=item_ref(group),
        )

    for item in raw:
        reference = item_ref(item)
        projection = emissions.get(reference)
        if projection is not None:
            yield projection
            continue
        if reference in consumed:
            continue
        yield ChunkingItem(
            kind=classify_item(item),
            text=item_text(item, document),
            refs=(reference,),
            items=(item,),
            header_depth=(
                _header_depth(item)
                if classify_item(item) in {ItemKind.TITLE, ItemKind.SECTION_HEADER}
                else None
            ),
            semantic_container=_semantic_container(document, item),
            inline=_has_inline_ancestor(document, item),
        )


def item_text(item: DocItem, document: DoclingDocument) -> str:
    """Return the canonical chunk text an item contributes, possibly empty."""

    if isinstance(item, TableItem):
        return table_text(item, document)
    if isinstance(item, PictureItem):
        return ""
    value = getattr(item, "text", None)
    return canonical_text(value) if isinstance(value, str) else ""


def table_text(table: TableItem, document: DoclingDocument) -> str:
    """Serialize a table through Docling instead of re-deriving cell layout."""

    try:
        return canonical_text(table.export_to_markdown(document))
    except ParserExecutionError:
        raise
    except Exception as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "docling_table_serialization"},
        ) from error


def table_html(
    table: TableItem,
    document: DoclingDocument,
    limits: ParserLimits,
) -> str | None:
    """Return bounded structural HTML, or ``None`` when the table has no cells."""

    try:
        value = table.export_to_html(document, add_caption=False)
    except ParserExecutionError:
        raise
    except Exception as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "docling_table_serialization"},
        ) from error
    value = value.strip()
    if not value:
        return None
    if len(value.encode("utf-8")) > limits.max_table_html_bytes:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic={
                "limit_name": "max_table_html_bytes",
                "limit": limits.max_table_html_bytes,
            },
        )
    return value


def section_paths(
    document: DoclingDocument,
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Build every item's heading path from traversal order and header levels.

    Docling already carries the heading structure, so the path is derived here
    once and shared by both chunking strategies instead of being recomputed as
    a per-element hierarchy.
    """

    paths: dict[str, tuple[dict[str, Any], ...]] = {}
    stack: list[tuple[int, dict[str, Any]]] = []
    for item in iterate_chunking_items(document):
        kind = item.kind
        if kind in {ItemKind.TITLE, ItemKind.SECTION_HEADER}:
            depth = 0 if kind is ItemKind.TITLE else (item.header_depth or 1)
            while stack and stack[-1][0] >= depth:
                stack.pop()
            if item.text:
                stack.append(
                    (depth, {"depth": depth, "text": item.text[:_MAX_TITLE_CHARS]})
                )
        path = tuple(entry for _depth, entry in stack[-_MAX_TITLES:])
        for reference in item.refs:
            paths[reference] = path
    return paths


def bounded_hierarchy(
    titles: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Return the persisted hierarchy shape, or an empty mapping."""

    return {"titles": [dict(entry) for entry in titles]} if titles else {}


def common_hierarchy(
    paths: tuple[tuple[dict[str, Any], ...], ...],
) -> dict[str, Any]:
    """Keep only the heading prefix every item in one chunk shares."""

    if not paths:
        return {}
    common = list(paths[0])
    for path in paths[1:]:
        length = 0
        for left, right in zip(common, path):
            if left != right:
                break
            length += 1
        common = common[:length]
    return bounded_hierarchy(tuple(common))


def _header_depth(item: DocItem) -> int:
    level = getattr(item, "level", None)
    if isinstance(level, bool) or not isinstance(level, int) or level < 1:
        return 1
    return min(level, _MAX_TITLE_DEPTH)


def item_ref(item: DocItem | GroupItem) -> str:
    """Return the Docling self reference used as the only item identity."""

    reference = getattr(item, "self_ref", "")
    if not isinstance(reference, str) or not reference:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "docling_item_self_ref"},
        )
    return reference


def parent_ref(item: DocItem | GroupItem) -> str | None:
    parent = getattr(item, "parent", None)
    reference = getattr(parent, "cref", None)
    return reference if isinstance(reference, str) and reference else None


def caption_refs(item: DocItem) -> tuple[str, ...]:
    captions = getattr(item, "captions", ())
    return tuple(
        reference.cref
        for reference in captions or ()
        if isinstance(getattr(reference, "cref", None), str) and reference.cref
    )


def canonical_text(value: str) -> str:
    return unicodedata.normalize(
        "NFC", value.replace("\r\n", "\n").replace("\r", "\n")
    ).strip()


def _inline_projection(
    document: DoclingDocument,
    group: InlineGroup,
    *,
    kind: ItemKind,
    header_depth: int | None,
    semantic_container: str | None,
) -> ChunkingItem:
    leaves = _inline_leaves(document, group, seen_groups=set(), seen_items=set())
    if not leaves:
        raise _invalid_projection("inline_group_empty")
    texts = tuple(item_text(item, document) for item in leaves)
    text = canonical_text(" ".join(value for value in texts if value))
    if not text:
        raise _invalid_projection("inline_group_text")
    surfaces = {
        page
        for item in leaves
        for entry in getattr(item, "prov", ()) or ()
        if isinstance((page := getattr(entry, "page_no", None)), int)
        and not isinstance(page, bool)
    }
    if len(surfaces) > 1:
        raise _invalid_projection("inline_group_cross_surface")
    return ChunkingItem(
        kind=kind,
        text=text,
        refs=tuple(item_ref(item) for item in leaves),
        items=leaves,
        header_depth=header_depth,
        semantic_container=semantic_container,
        inline=True,
    )


def _inline_leaves(
    document: DoclingDocument,
    group: InlineGroup,
    *,
    seen_groups: set[str],
    seen_items: set[str],
) -> tuple[DocItem, ...]:
    group_reference = item_ref(group)
    if group_reference in seen_groups:
        raise _invalid_projection("inline_group_cycle")
    seen_groups.add(group_reference)
    leaves: list[DocItem] = []
    for child_ref in getattr(group, "children", ()) or ():
        child = _resolve_child(document, child_ref)
        if parent_ref(child) != group_reference:
            raise _invalid_projection("inline_child_parent")
        if isinstance(child, InlineGroup):
            leaves.extend(
                _inline_leaves(
                    document,
                    child,
                    seen_groups=seen_groups,
                    seen_items=seen_items,
                )
            )
            continue
        if isinstance(child, GroupItem) or not isinstance(child, DocItem):
            raise _invalid_projection("inline_child_type")
        if classify_item(child) not in _INLINE_TEXT_KINDS:
            raise _invalid_projection("inline_child_role")
        reference = item_ref(child)
        if reference in seen_items or (getattr(child, "children", ()) or ()):
            raise _invalid_projection("inline_child_leaf")
        seen_items.add(reference)
        leaves.append(child)
    seen_groups.remove(group_reference)
    return tuple(leaves)


def _resolve_child(document: DoclingDocument, reference: Any) -> Any:
    cref = getattr(reference, "cref", None)
    if not isinstance(cref, str) or not cref:
        raise _invalid_projection("inline_child_ref")
    try:
        return reference.resolve(document)
    except ParserExecutionError:
        raise
    except Exception as error:
        raise _invalid_projection("inline_child_resolve") from error


def _claim_projection(
    projection: ChunkingItem,
    *,
    emission_ref: str,
    raw_positions: dict[str, int],
    emissions: dict[str, ChunkingItem],
    consumed: set[str],
    claimed_groups: set[str],
    group_ref: str,
) -> None:
    if group_ref in claimed_groups or emission_ref in emissions:
        raise _invalid_projection("inline_group_duplicate")
    if emission_ref not in raw_positions or any(
        reference not in raw_positions for reference in projection.refs
    ):
        raise _invalid_projection("inline_group_reading_order")
    positions = tuple(raw_positions[reference] for reference in projection.refs)
    if positions != tuple(sorted(positions)) or len(set(positions)) != len(positions):
        raise _invalid_projection("inline_group_reading_order")
    overlap = consumed.intersection(projection.refs)
    if overlap:
        raise _invalid_projection("inline_group_duplicate")
    claimed_groups.add(group_ref)
    emissions[emission_ref] = projection
    consumed.update(projection.refs)
    if emission_ref not in projection.refs:
        consumed.add(emission_ref)


def _semantic_container(document: DoclingDocument, item: DocItem) -> str | None:
    current: DocItem | GroupItem = item
    seen: set[str] = set()
    while (reference := parent_ref(current)) is not None:
        if reference in {"#/body", "#/furniture"}:
            return None
        if reference in seen:
            raise _invalid_projection("semantic_container_cycle")
        seen.add(reference)
        parent = _resolve_reference(document, reference)
        if isinstance(parent, GroupItem):
            if getattr(parent, "label", None) in _TRANSPARENT_GROUP_LABELS:
                current = parent
                continue
            return item_ref(parent)
        if isinstance(parent, DocItem):
            parent_kind = classify_item(parent)
            if parent_kind in _WRAPPER_KINDS and not item_text(parent, document):
                current = parent
                continue
            return item_ref(parent)
        raise _invalid_projection("semantic_container_type")
    return None


def _has_inline_ancestor(document: DoclingDocument, item: DocItem) -> bool:
    current: DocItem | GroupItem = item
    seen: set[str] = set()
    while (reference := parent_ref(current)) is not None:
        if reference in {"#/body", "#/furniture"}:
            return False
        if reference in seen:
            raise _invalid_projection("semantic_container_cycle")
        seen.add(reference)
        parent = _resolve_reference(document, reference)
        if isinstance(parent, InlineGroup):
            return True
        if not isinstance(parent, (DocItem, GroupItem)):
            return False
        current = parent
    return False


def _resolve_reference(document: DoclingDocument, reference: str) -> Any:
    try:
        from docling_core.types.doc.common.reference import RefItem

        return RefItem(cref=reference).resolve(document)
    except ParserExecutionError:
        raise
    except Exception as error:
        raise _invalid_projection("docling_parent_resolve") from error


def _invalid_projection(check: str) -> ParserExecutionError:
    return ParserExecutionError(
        ErrorCode.PARSER_OUTPUT_INVALID,
        diagnostic={"check": check},
    )
