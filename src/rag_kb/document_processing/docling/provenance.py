"""Bounded projection from Docling item provenance to persisted locations.

Docling ``item.prov`` stays the source of truth. A persisted chunk can cover
several items, so this module derives one bounded aggregate view for JSONB,
citations and the frontend without inventing coordinates Docling never
reported.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.common.reference import RefItem
from docling_core.types.doc.document import DocItem

from rag_kb.domain import ErrorCode, ParserExecutionError, ParserLimits


PROVENANCE_VERSION = "v1"

SURFACE_PAGE = "page"
SURFACE_SLIDE = "slide"
SURFACE_SHEET = "sheet"
SURFACE_LOGICAL = "logical"

_OOXML_PREFIX = "application/vnd.openxmlformats-officedocument"
_SURFACE_BY_MIMETYPE = {
    "application/pdf": SURFACE_PAGE,
    "image/png": SURFACE_PAGE,
    "image/jpeg": SURFACE_PAGE,
    "image/tiff": SURFACE_PAGE,
    # Docling reports the template mimetype for decks built from one, so both
    # presentation packages have to map onto the slide surface.
    f"{_OOXML_PREFIX}.presentationml.presentation": SURFACE_SLIDE,
    f"{_OOXML_PREFIX}.presentationml.template": SURFACE_SLIDE,
    f"{_OOXML_PREFIX}.spreadsheetml.sheet": SURFACE_SHEET,
    f"{_OOXML_PREFIX}.spreadsheetml.template": SURFACE_SHEET,
}

#: Bound on the sheet or slide names one chunk may name before the range alone
#: has to speak for it.
_MAX_SURFACE_LABELS = 32

@dataclass(frozen=True, slots=True)
class ItemSurface:
    """One Docling provenance entry reduced to the facts the project persists."""

    kind: str
    ordinal: int
    bbox: dict[str, float] | None = None
    coord_origin: str | None = None


def surface_kind(document: DoclingDocument) -> str:
    """Return the paginated surface a format exposes, or ``logical``."""

    origin = getattr(document, "origin", None)
    mimetype = getattr(origin, "mimetype", None)
    if not isinstance(mimetype, str):
        return SURFACE_LOGICAL
    return _SURFACE_BY_MIMETYPE.get(mimetype.strip().lower(), SURFACE_LOGICAL)


def item_surfaces(
    item: DocItem,
    *,
    kind: str,
) -> tuple[ItemSurface, ...]:
    """Read ``item.prov`` for paginated formats; never fabricate a surface."""

    if kind == SURFACE_LOGICAL:
        return ()
    surfaces: list[ItemSurface] = []
    for entry in getattr(item, "prov", ()) or ():
        ordinal = getattr(entry, "page_no", None)
        if not isinstance(ordinal, int) or isinstance(ordinal, bool):
            continue
        bbox = getattr(entry, "bbox", None)
        surfaces.append(
            ItemSurface(
                kind=kind,
                ordinal=ordinal,
                bbox=_bbox_json(bbox),
                coord_origin=_coord_origin(bbox),
            )
        )
    return tuple(surfaces)


def document_surfaces(
    document: DoclingDocument,
    item_refs: Sequence[str],
) -> tuple[tuple[ItemSurface, ...], ...]:
    """Resolve references back to their Docling provenance, in chunk order."""

    kind = surface_kind(document)
    return tuple(
        item_surfaces(_resolve(document, reference), kind=kind)
        for reference in item_refs
    )


def project_source_location(
    document: DoclingDocument,
    item_refs: Sequence[str],
    limits: ParserLimits,
    *,
    surface_labels: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Project one chunk's items onto the frozen bounded location view."""

    return aggregate_provenance(
        document_surfaces(document, item_refs),
        item_refs,
        max_metadata_bytes=limits.max_metadata_bytes,
        surface_labels=surface_labels,
    )


def aggregate_provenance(
    surfaces_by_item: Sequence[Sequence[ItemSurface]],
    item_refs: Sequence[str],
    *,
    max_metadata_bytes: int,
    surface_labels: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Apply the frozen aggregation rules to already-resolved provenance."""

    if len(surfaces_by_item) != len(item_refs):
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "provenance_item_alignment"},
        )
    flattened = [surface for surfaces in surfaces_by_item for surface in surfaces]
    kinds = {surface.kind for surface in flattened}
    if len(kinds) > 1:
        # A chunk that spans two surface kinds cannot be cited; the assembler
        # must have placed a hard boundary first.
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "chunk_surface_kind"},
        )
    projection: dict[str, Any] = {
        "provenance_version": PROVENANCE_VERSION,
        "surface_type": next(iter(kinds), SURFACE_LOGICAL),
        "item_ref_count": len(item_refs),
    }
    ordinals = [surface.ordinal for surface in flattened]
    if ordinals:
        projection["surface_start"] = min(ordinals)
        projection["surface_end"] = max(ordinals)
        projection.update(_labels(sorted(set(ordinals)), surface_labels))
    if len(item_refs) == 1 and len(flattened) == 1:
        # Only a single item on a single surface has one true rectangle; several
        # items never get a merged box that covers content between them.
        surface = flattened[0]
        if surface.bbox is not None and surface.coord_origin is not None:
            projection["bbox"] = surface.bbox
            projection["coord_origin"] = surface.coord_origin
    return _with_bounded_refs(projection, item_refs, max_metadata_bytes)


def surface_location(
    kind: str,
    ordinal: int,
    surface_labels: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Describe a whole surface, used by assets that belong to no single item."""

    return {
        "provenance_version": PROVENANCE_VERSION,
        "surface_type": kind,
        "surface_start": ordinal,
        "surface_end": ordinal,
        **_labels([ordinal], surface_labels),
        "item_refs": [],
        "item_ref_count": 0,
    }


def _labels(
    ordinals: Sequence[int],
    surface_labels: Mapping[int, str] | None,
) -> dict[str, Any]:
    """Name the covered surfaces when the format supplies names."""

    if not surface_labels:
        return {}
    named = [
        surface_labels[ordinal] for ordinal in ordinals if ordinal in surface_labels
    ]
    if not named:
        return {}
    if len(named) == 1 and len(ordinals) == 1:
        return {"surface_label": named[0]}
    return {"surface_labels": named[:_MAX_SURFACE_LABELS]}


def chunk_assembly_key(
    *,
    profile: str,
    source_checksum_sha256: str,
    assembly_ordinal: int,
    item_refs: Sequence[str],
    text: str,
) -> str:
    """Derive a stable chunk identity from its ordered current assembly facts."""

    if assembly_ordinal < 0:
        raise ValueError("assembly ordinal must be non-negative")
    payload = {
        "profile": profile,
        "source_checksum": source_checksum_sha256,
        "assembly_ordinal": assembly_ordinal,
        "item_refs": list(item_refs),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def surface_ordinals(source_location: dict[str, Any]) -> frozenset[int]:
    """Read surface ordinals from the current provenance projection."""

    values = {
        ordinal
        for key in ("surface_start", "surface_end")
        if isinstance((ordinal := source_location.get(key)), int)
        and not isinstance(ordinal, bool)
    }
    return frozenset(values)


def _with_bounded_refs(
    projection: dict[str, Any],
    item_refs: Sequence[str],
    max_metadata_bytes: int,
) -> dict[str, Any]:
    complete = {**projection, "item_refs": list(item_refs)}
    if len(_canonical_json(complete)) <= max_metadata_bytes:
        return complete
    # Truncating the sequence would claim complete provenance the chunk no
    # longer carries, so the omission is explicit and the full sequence stays
    # verifiable through its hash.
    return {
        **projection,
        "item_refs_omitted": True,
        "item_refs_sha256": hashlib.sha256(
            _canonical_json(list(item_refs))
        ).hexdigest(),
    }


def _resolve(document: DoclingDocument, reference: str) -> DocItem:
    try:
        item = RefItem(cref=reference).resolve(document)
    except ParserExecutionError:
        raise
    except Exception as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "docling_item_ref_resolution"},
        ) from error
    if not isinstance(item, DocItem):
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "docling_item_ref_resolution"},
        )
    return item


def _bbox_json(bbox: Any) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for name in ("l", "t", "r", "b"):
        value = getattr(bbox, name, None)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        values[name] = float(value)
    return values


def _coord_origin(bbox: Any) -> str | None:
    origin = getattr(bbox, "coord_origin", None)
    value = getattr(origin, "value", origin)
    return value if isinstance(value, str) and value else None


def _canonical_json(value: Iterable[Any] | dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
