"""Pure partition-to-Evidence-Unit assembly for multimodal revisions."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace

from rag_kb.document_processing.multimodal_boundaries import (
    VisualDisposition,
    classify_visual,
)
from rag_kb.document_processing.profiles import UNSTRUCTURED_CHUNKING_CONFIG
from rag_kb.document_processing.tokenization import count_chunk_tokens, split_by_tokens
from rag_kb.domain import (
    ContentModality,
    EvidenceUnitDraft,
    ParsedAssetDraft,
    ParsedDocument,
    ParsedElement,
    ParserLimits,
    ParserExecutionError,
    ErrorCode,
)


def assemble_multimodal_units(
    parsed: ParsedDocument, limits: ParserLimits | None = None
) -> tuple[EvidenceUnitDraft, ...]:
    """Build bounded text regions plus independent image/table evidence units."""

    resolved_limits = limits or ParserLimits()
    assets = {asset.asset_key: asset for asset in parsed.assets}
    repeats = Counter(asset.content_sha256 for asset in parsed.assets)
    units: list[EvidenceUnitDraft] = []
    text_region: list[ParsedElement] = []
    anchor_key: str | None = None
    text_insert_at: int | None = None

    def flush_text() -> None:
        nonlocal text_region, anchor_key, text_insert_at
        if not text_region:
            return
        parts = _structural_region_parts(text_region)
        if not parts:
            text_region = []
            return
        created: list[EvidenceUnitDraft] = []
        for part_index, part in enumerate(parts):
            keys = ":".join(item.element_key or str(item.ordinal) for item in text_region)
            key = _key("text", keys, str(part_index), hashlib.sha256(part.encode()).hexdigest())
            created.append(
                EvidenceUnitDraft(
                    unit_key=key,
                    ordinal=len(units) + len(created),
                    modality=ContentModality.TEXT,
                    content=part,
                    token_count=count_chunk_tokens(part),
                    asset_key=None,
                    evidence_group_key=None,
                    related_unit_keys=((anchor_key,) if anchor_key else ()),
                    source_location=_span_location(text_region),
                    hierarchy=dict(text_region[0].hierarchy),
                    processing_metadata={"assembly": "by_title_token_region_v1"},
                    required_representations=("text",),
                )
            )
        if text_insert_at is None:
            units.extend(created)
        else:
            units[text_insert_at:text_insert_at] = created
        text_region = []
        anchor_key = None
        text_insert_at = None

    for element in parsed.elements:
        if element.category == "PageBreak":
            flush_text()
            continue
        if element.category in {"Image", "Table", "TableChunk"} or element.asset_key:
            asset = assets.get(element.asset_key or "")
            if element.category in {"Table", "TableChunk"}:
                flush_text()
                _append_table_units(units, element)
                continue
            if asset is None:
                raise ParserExecutionError(
                    ErrorCode.PARSER_OUTPUT_INVALID,
                    diagnostic={"check": "element_asset_reference"},
                )
            disposition = classify_visual(
                element, asset, repeated_hash_count=repeats[asset.content_sha256]
            )
            if disposition is VisualDisposition.DECORATIVE:
                continue
            if disposition is VisualDisposition.HARD_BOUNDARY:
                flush_text()
            elif text_region and text_insert_at is None:
                # An anchored figure does not split its surrounding prose, but
                # the merged text unit still sorts by its first source element.
                text_insert_at = len(units)
            image_key = _key("image", element.element_key, asset.asset_key)
            image_group = _key("asset-group", asset.asset_key)
            content = element.text.strip()
            if (
                len(content) > resolved_limits.max_ocr_characters
                or (content and count_chunk_tokens(content) > resolved_limits.max_ocr_tokens)
            ):
                raise ParserExecutionError(
                    ErrorCode.PARSER_RESOURCE_LIMIT,
                    diagnostic={
                        "limit_name": "max_ocr_tokens",
                        "limit": resolved_limits.max_ocr_tokens,
                    },
                )
            units.append(
                EvidenceUnitDraft(
                    unit_key=image_key,
                    ordinal=len(units),
                    modality=ContentModality.IMAGE,
                    content=content,
                    token_count=count_chunk_tokens(content) if content else 0,
                    asset_key=asset.asset_key,
                    evidence_group_key=image_group,
                    related_unit_keys=(),
                    source_location=dict(element.source_location),
                    hierarchy=dict(element.hierarchy),
                    processing_metadata={"visual_disposition": disposition.value},
                    required_representations=("native_image",),
                )
            )
            anchor_key = image_key
            continue
        if element.category == "FigureCaption" and anchor_key:
            # Author captions are their own text representation; never consume the
            # adjacent body chunk's token budget.
            if element.token_count > resolved_limits.max_caption_tokens:
                raise ParserExecutionError(
                    ErrorCode.PARSER_RESOURCE_LIMIT,
                    diagnostic={
                        "limit_name": "max_caption_tokens",
                        "limit": resolved_limits.max_caption_tokens,
                    },
                )
            for index in range(len(units) - 1, -1, -1):
                if units[index].unit_key == anchor_key:
                    units[index] = replace(
                        units[index],
                        content=element.text,
                        token_count=element.token_count,
                        processing_metadata={
                            **units[index].processing_metadata,
                            "author_caption": True,
                        },
                    )
                    break
            continue
        if element.is_title and text_region:
            flush_text()
        if element.text:
            text_region.append(element)
    flush_text()
    if len(units) > resolved_limits.max_units:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic={
                "limit_name": "max_units",
                "limit": resolved_limits.max_units,
            },
        )
    return tuple(
        EvidenceUnitDraft(
            unit_key=unit.unit_key,
            ordinal=ordinal,
            modality=unit.modality,
            content=unit.content,
            token_count=unit.token_count,
            asset_key=unit.asset_key,
            evidence_group_key=unit.evidence_group_key,
            related_unit_keys=unit.related_unit_keys,
            source_location=unit.source_location,
            hierarchy=unit.hierarchy,
            processing_metadata=unit.processing_metadata,
            required_representations=unit.required_representations,
        )
        for ordinal, unit in enumerate(units)
    )


def semantic_text_elements(parsed: ParsedDocument) -> ParsedDocument:
    """Expose only author body text to semantic distance planning."""

    elements = tuple(
        item
        for item in parsed.elements
        if item.text
        and item.category not in {"Image", "Table", "TableChunk", "FigureCaption"}
        and item.asset_key is None
    )
    return ParsedDocument(elements, sum(len(item.text) for item in elements))


def element_sequence_hash(parsed: ParsedDocument) -> str:
    return _canonical_hash(
        [
            {
                "element_key": item.element_key,
                "ordinal": item.ordinal,
                "category": item.category,
                "asset_key": item.asset_key,
            }
            for item in parsed.elements
        ]
    )


def asset_manifest_hash(assets: tuple[ParsedAssetDraft, ...]) -> str:
    return _canonical_hash(
        [
            {
                "asset_key": item.asset_key,
                "kind": item.kind,
                "media_type": item.media_type,
                "checksum": item.content_sha256,
                "width": item.width,
                "height": item.height,
            }
            for item in assets
        ]
    )


def unit_plan_hash(units: tuple[EvidenceUnitDraft, ...]) -> str:
    return _canonical_hash(
        [
            {
                "unit_key": item.unit_key,
                "ordinal": item.ordinal,
                "modality": item.modality.value,
                "asset_key": item.asset_key,
                "evidence_group_key": item.evidence_group_key,
                "related_unit_keys": item.related_unit_keys,
                "required_representations": item.required_representations,
            }
            for item in units
        ]
    )


def _append_table_units(units: list[EvidenceUnitDraft], element: ParsedElement) -> None:
    lines = [line for line in element.text.splitlines() if line.strip()]
    header = lines[0] if lines else ""
    groups: list[list[str]] = [[]]
    for line in lines[1:]:
        proposed = "\n".join(([header] if header else []) + groups[-1] + [line])
        if groups[-1] and count_chunk_tokens(proposed) > UNSTRUCTURED_CHUNKING_CONFIG["max_tokens"]:
            groups.append([])
        groups[-1].append(line)
    if not groups[0] and header:
        groups = [[]]
    group_key = _key("table-group", element.element_key)
    for child, rows in enumerate(groups):
        content = "\n".join(([header] if header else []) + rows).strip()
        units.append(
            EvidenceUnitDraft(
                unit_key=_key("table", element.element_key, str(child)),
                ordinal=len(units),
                modality=ContentModality.TABLE,
                content=content,
                token_count=count_chunk_tokens(content) if content else 0,
                asset_key=element.asset_key,
                evidence_group_key=group_key,
                related_unit_keys=(),
                source_location=dict(element.source_location),
                hierarchy=dict(element.hierarchy),
                processing_metadata={"table_child": child, "table_html_present": bool(element.table_html)},
                required_representations=("table_text",),
            )
        )


def _structural_region_parts(elements: list[ParsedElement]) -> tuple[str, ...]:
    """Apply the frozen by-title token thresholds to one text-only region."""

    maximum = UNSTRUCTURED_CHUNKING_CONFIG["max_tokens"]
    soft = UNSTRUCTURED_CHUNKING_CONFIG["new_after_n_tokens"]
    overlap = UNSTRUCTURED_CHUNKING_CONFIG["overlap"]
    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            parts.append("\n\n".join(current))
        current = []
        current_tokens = 0

    for element in elements:
        text = element.text.strip()
        if not text:
            continue
        tokens = count_chunk_tokens(text)
        if tokens > maximum:
            flush()
            parts.extend(
                split_by_tokens(
                    text,
                    max_tokens=maximum,
                    overlap_tokens=overlap,
                )
            )
            continue
        proposed = "\n\n".join((*current, text))
        proposed_tokens = count_chunk_tokens(proposed)
        if current and (current_tokens >= soft or proposed_tokens > maximum):
            flush()
        current.append(text)
        current_tokens = count_chunk_tokens("\n\n".join(current))
    flush()
    return tuple(parts)


def _span_location(elements: list[ParsedElement]) -> dict:
    pages = sorted(
        {
            page
            for item in elements
            if isinstance((page := item.source_location.get("page_number")), int)
        }
    )
    return {"page_numbers": pages} if pages else dict(elements[0].source_location)


def _key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
