"""Visual assets and chunk relations derived from Docling's own facts.

Pictures, table images and page images all come from the single converted
document. Relations prefer Docling's explicit references and only fall back to
surface heuristics when no explicit reference exists, recording which source
was actually used.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from typing import Any

from PIL import Image
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocItem, PictureItem, TableItem

from rag_kb.document_processing.docling.provenance import (
    SURFACE_LOGICAL,
    item_surfaces,
    project_source_location,
    surface_kind,
    surface_location,
    surface_ordinals,
)
from rag_kb.document_processing.docling.traversal import (
    caption_refs,
    canonical_text,
    item_ref,
    iterate_body_items,
    parent_ref,
)
from rag_kb.document_processing.multimodal_assembly import normalize_figure_labels
from rag_kb.document_processing.tokenization import count_chunk_tokens
from rag_kb.domain import (
    ChunkAssemblyDraft,
    ChunkAssetRelationProvenance,
    ChunkAssetRelationType,
    DoclingAssetRelationDraft,
    ErrorCode,
    ParsedAssetDraft,
    ParserExecutionError,
    ParserLimits,
)


ASSET_KIND_PICTURE = "docling_picture"
ASSET_KIND_TABLE_IMAGE = "table_image"
ASSET_KIND_PAGE_IMAGE = "page_image"

_PNG_MEDIA_TYPE = "image/png"
_ENCODABLE_MODES = frozenset({"L", "LA", "RGB", "RGBA"})


@dataclass(frozen=True, slots=True)
class _Candidate:
    relation_type: ChunkAssetRelationType
    provenance: ChunkAssetRelationProvenance
    confidence_micros: int
    figure_label: str | None = None


def extract_docling_assets(
    document: DoclingDocument,
    limits: ParserLimits | None = None,
    *,
    page_image_surfaces: frozenset[int] | None = None,
    surface_labels: Mapping[int, str] | None = None,
) -> tuple[ParsedAssetDraft, ...]:
    """Materialize every visual the converted document already carries.

    ``page_image_surfaces`` names the surfaces whose rendered image is itself
    evidence — a scanned page. ``DoclingDocument`` records no OCR provenance, so
    that judgement needs the source bytes and belongs to the caller; omitting
    the argument falls back to the surfaces on which Docling found nothing at
    all.
    """

    resolved = limits or ParserLimits()
    assets: list[ParsedAssetDraft] = []
    total_bytes = 0

    for picture in document.pictures:
        image = _image_of(picture, document)
        if image is None:
            continue
        reference = item_ref(picture)
        asset = _bounded_asset(
            image,
            kind=ASSET_KIND_PICTURE,
            source_location=project_source_location(
                document, (reference,), resolved, surface_labels=surface_labels
            ),
            metadata={
                "item_ref": reference,
                "source": "docling_picture",
                **_caption_metadata(picture, document, resolved),
            },
            limits=resolved,
        )
        assets.append(asset)
        total_bytes = _require_asset_limits(assets, total_bytes, asset, resolved)

    for table in document.tables:
        if table.image is None:
            continue
        image = _image_of(table, document)
        if image is None:
            continue
        reference = item_ref(table)
        asset = _bounded_asset(
            image,
            kind=ASSET_KIND_TABLE_IMAGE,
            source_location=project_source_location(
                document, (reference,), resolved, surface_labels=surface_labels
            ),
            metadata={
                "item_ref": reference,
                "source": "docling_table",
                **_caption_metadata(table, document, resolved),
            },
            limits=resolved,
        )
        assets.append(asset)
        total_bytes = _require_asset_limits(assets, total_bytes, asset, resolved)

    kind = surface_kind(document)
    for ordinal in _page_image_surfaces(document, kind, page_image_surfaces):
        page = document.pages[ordinal]
        image = page.image.pil_image if page.image is not None else None
        if image is None:
            continue
        asset = _bounded_asset(
            image,
            kind=ASSET_KIND_PAGE_IMAGE,
            source_location=surface_location(kind, ordinal, surface_labels),
            metadata={"source": "docling_page", "surface_ordinal": ordinal},
            limits=resolved,
        )
        assets.append(asset)
        total_bytes = _require_asset_limits(assets, total_bytes, asset, resolved)

    return tuple(assets)


def relate_assets_to_chunks(
    document: DoclingDocument,
    chunks: tuple[ChunkAssemblyDraft, ...],
    assets: tuple[ParsedAssetDraft, ...],
    limits: ParserLimits | None = None,
) -> tuple[DoclingAssetRelationDraft, ...]:
    """Emit the strongest relation Docling supports for each chunk and asset."""

    resolved = limits or ParserLimits()
    if not chunks or not assets:
        return ()
    items = {item_ref(item): item for item, _level in iterate_body_items(document)}
    order = {reference: position for position, reference in enumerate(items)}
    relations: list[DoclingAssetRelationDraft] = []
    per_chunk: Counter[int] = Counter()

    for index, chunk in enumerate(chunks):
        refs = frozenset(chunk.item_refs)
        surfaces = surface_ordinals(chunk.source_location)
        positions = tuple(
            order[reference] for reference in chunk.item_refs if reference in order
        )
        labels = frozenset(normalize_figure_labels(chunk.text))
        for asset in assets:
            candidate = _candidate(
                items=items,
                asset=asset,
                refs=refs,
                surfaces=surfaces,
                positions=positions,
                labels=labels,
                order=order,
            )
            if candidate is None:
                continue
            per_chunk[index] += 1
            if per_chunk[index] > resolved.max_relations_per_chunk:
                raise ParserExecutionError(
                    ErrorCode.PARSER_RESOURCE_LIMIT,
                    diagnostic={
                        "limit_name": "max_relations_per_chunk",
                        "limit": resolved.max_relations_per_chunk,
                    },
                )
            relations.append(
                DoclingAssetRelationDraft(
                    chunk_index=index,
                    asset_key=asset.asset_key,
                    relation_type=candidate.relation_type,
                    confidence_micros=candidate.confidence_micros,
                    ordinal=len(relations),
                    provenance=candidate.provenance,
                    figure_label=candidate.figure_label,
                )
            )
            if len(relations) > resolved.max_relations:
                raise ParserExecutionError(
                    ErrorCode.PARSER_RESOURCE_LIMIT,
                    diagnostic={
                        "limit_name": "max_relations",
                        "limit": resolved.max_relations,
                    },
                )
    return tuple(relations)


def _candidate(
    *,
    items: dict[str, DocItem],
    asset: ParsedAssetDraft,
    refs: frozenset[str],
    surfaces: frozenset[int],
    positions: tuple[int, ...],
    labels: frozenset[str],
    order: dict[str, int],
) -> _Candidate | None:
    reference = asset.processing_metadata.get("item_ref")
    asset_surfaces = surface_ordinals(dict(asset.source_location))

    if asset.kind == ASSET_KIND_PAGE_IMAGE:
        if surfaces and asset_surfaces and surfaces.intersection(asset_surfaces):
            return _Candidate(
                ChunkAssetRelationType.OCR_OF,
                ChunkAssetRelationProvenance.DOCLING_PAGE_OCR_V1,
                1_000_000,
            )
        return None

    if not isinstance(reference, str):
        return None
    asset_labels = _asset_labels(asset)
    own_label = next(iter(sorted(asset_labels)), None)
    if asset.kind == ASSET_KIND_TABLE_IMAGE and reference in refs:
        return _Candidate(
            ChunkAssetRelationType.TABLE_OF,
            ChunkAssetRelationProvenance.DOCLING_TABLE_IDENTITY_V1,
            1_000_000,
            own_label,
        )

    item = items.get(reference)
    if item is None:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "docling_asset_item_ref"},
        )
    if refs.intersection(caption_refs(item)):
        return _Candidate(
            ChunkAssetRelationType.CAPTION_OF,
            ChunkAssetRelationProvenance.DOCLING_CAPTION_REF_V1,
            1_000_000,
            own_label,
        )
    if reference in refs:
        return _Candidate(
            ChunkAssetRelationType.INLINE_FIGURE,
            ChunkAssetRelationProvenance.DOCLING_ITEM_REF_V1,
            1_000_000,
            own_label,
        )
    parent = parent_ref(item)
    if parent is not None and parent in refs:
        # Weaker than direct containment: Docling's DOCX backend parents every
        # item in a section to its heading, so this only says "same section".
        return _Candidate(
            ChunkAssetRelationType.INLINE_FIGURE,
            ChunkAssetRelationProvenance.DOCLING_PARENT_REF_V1,
            1_000_000,
            own_label,
        )

    shared = sorted(labels.intersection(asset_labels))
    if shared:
        return _Candidate(
            ChunkAssetRelationType.EXPLICIT_FIGURE_REFERENCE,
            ChunkAssetRelationProvenance.AUTHOR_REFERENCE_V2,
            1_000_000,
            shared[0],
        )

    if not (surfaces and asset_surfaces and surfaces.intersection(asset_surfaces)):
        return None
    position = order.get(reference)
    if position is not None and positions:
        if position in {min(positions) - 1, max(positions) + 1}:
            return _Candidate(
                ChunkAssetRelationType.SPATIAL_NEIGHBOR,
                ChunkAssetRelationProvenance.DOCLING_SURFACE_NEIGHBOR_V1,
                500_000,
            )
    return _Candidate(
        ChunkAssetRelationType.SAME_PAGE,
        ChunkAssetRelationProvenance.DOCLING_SURFACE_CO_LOCATION_V1,
        250_000,
    )


def _asset_labels(asset: ParsedAssetDraft) -> frozenset[str]:
    caption = asset.processing_metadata.get("caption")
    return frozenset(normalize_figure_labels(caption)) if isinstance(caption, str) else frozenset()


def _page_image_surfaces(
    document: DoclingDocument,
    kind: str,
    requested: frozenset[int] | None,
) -> tuple[int, ...]:
    """Pick the surfaces whose rendered image is the asset, availability first."""

    if kind == SURFACE_LOGICAL:
        return ()
    available = tuple(
        ordinal
        for ordinal in sorted(document.pages)
        if document.pages[ordinal].image is not None
    )
    if requested is not None:
        return tuple(ordinal for ordinal in available if ordinal in requested)
    # Without a caller judgement, only a surface Docling read nothing from is
    # taken as evidence; rendering every text page would flood the index.
    occupied = {
        surface.ordinal
        for item, _level in iterate_body_items(document)
        for surface in item_surfaces(item, kind=kind)
    }
    return tuple(ordinal for ordinal in available if ordinal not in occupied)


def _caption_metadata(
    item: DocItem,
    document: DoclingDocument,
    limits: ParserLimits,
) -> dict[str, Any]:
    references = caption_refs(item)
    if not references:
        return {}
    caption = canonical_text(item.caption_text(document))
    if not caption:
        return {"caption_refs": list(references)}
    if count_chunk_tokens(caption) > limits.max_caption_tokens:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic={
                "limit_name": "max_caption_tokens",
                "limit": limits.max_caption_tokens,
            },
        )
    return {"caption": caption, "caption_refs": list(references)}


def _image_of(item: DocItem, document: DoclingDocument) -> Image.Image | None:
    try:
        return item.get_image(document)
    except ParserExecutionError:
        raise
    except Exception as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "docling_asset_image"},
        ) from error


def _bounded_asset(
    image: Image.Image,
    *,
    kind: str,
    source_location: dict[str, Any],
    metadata: dict[str, Any],
    limits: ParserLimits,
) -> ParsedAssetDraft:
    width, height = image.size
    if (
        width < 1
        or height < 1
        or width > limits.max_image_width
        or height > limits.max_image_height
        or width * height > limits.max_image_pixels
    ):
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic={
                "limit_name": "max_image_pixels",
                "limit": limits.max_image_pixels,
            },
        )
    content = _encode_png(image)
    digest = hashlib.sha256(content).hexdigest()
    return ParsedAssetDraft(
        asset_key=hashlib.sha256(
            f"{kind}\x1f{_canonical(source_location)}\x1f{digest}".encode("utf-8")
        ).hexdigest(),
        kind=kind,
        media_type=_PNG_MEDIA_TYPE,
        content=content,
        content_sha256=digest,
        width=width,
        height=height,
        source_location=dict(source_location),
        processing_metadata={"format": "png", **metadata},
    )


def _encode_png(image: Image.Image) -> bytes:
    """Re-encode deterministically so a repeated conversion hashes identically."""

    prepared = image if image.mode in _ENCODABLE_MODES else image.convert(
        "RGBA" if "A" in image.getbands() else "RGB"
    )
    try:
        # Rebuilding from raw pixels drops any decoder metadata the source
        # carried, which would otherwise leak into the encoded bytes.
        clean = Image.frombytes(prepared.mode, prepared.size, prepared.tobytes())
        buffer = BytesIO()
        clean.save(buffer, format="PNG", optimize=False, compress_level=6)
    except ParserExecutionError:
        raise
    except Exception as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "asset_encoding"},
        ) from error
    return buffer.getvalue()


def _require_asset_limits(
    assets: list[ParsedAssetDraft],
    total_bytes: int,
    asset: ParsedAssetDraft,
    limits: ParserLimits,
) -> int:
    if len(assets) > limits.max_assets:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic={"limit_name": "max_assets", "limit": limits.max_assets},
        )
    total = total_bytes + len(asset.content)
    if total > limits.max_total_asset_bytes:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic={
                "limit_name": "max_total_asset_bytes",
                "limit": limits.max_total_asset_bytes,
            },
        )
    return total


def _canonical(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
