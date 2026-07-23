"""Pure partition-to-Evidence-Unit assembly for multimodal revisions."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace

from rag_kb.document_processing.multimodal_boundaries import (
    VisualDisposition,
    classify_visual,
)
from rag_kb.document_processing.profiles import UNSTRUCTURED_CHUNKING_CONFIG
from rag_kb.document_processing.tokenization import count_chunk_tokens, split_by_tokens
from rag_kb.domain import (
    ContentModality,
    ChunkAssetRelationDraft,
    ChunkAssetRelationProvenance,
    ChunkAssetRelationType,
    CompositeEvidenceDraft,
    EvidenceUnitDraft,
    ParsedAssetDraft,
    ParsedDocument,
    ParsedElement,
    ParserLimits,
    ParserExecutionError,
    ErrorCode,
)


_FIGURE_REFERENCE = re.compile(
    r"(?:(?:fig(?:ure)?)[.\s]*|图\s*)([0-9]+(?:[A-Za-z]|[.\-][0-9A-Za-z]+)?)",
    re.IGNORECASE,
)


def assemble_multimodal_units(
    parsed: ParsedDocument, limits: ParserLimits | None = None
) -> tuple[EvidenceUnitDraft, ...]:
    """Preserve the v1 Evidence Unit assembly for persisted v1 revisions."""

    return _assemble_multimodal_units(parsed, limits or ParserLimits())


def assemble_composite_evidence(
    parsed: ParsedDocument, limits: ParserLimits | None = None
) -> CompositeEvidenceDraft:
    """Build deterministic units plus bounded normalized Chunk/Asset relations."""

    resolved_limits = limits or ParserLimits()
    units = _assemble_multimodal_units(parsed, resolved_limits)
    return relate_composite_units(units, resolved_limits)


def relate_composite_units(
    units: tuple[EvidenceUnitDraft, ...], limits: ParserLimits | None = None
) -> CompositeEvidenceDraft:
    """Normalize groups and derive relations after structural or semantic boundaries."""

    resolved_limits = limits or ParserLimits()
    enriched = _with_stable_evidence_groups(units)
    relations = _assemble_relations(enriched, resolved_limits)
    return CompositeEvidenceDraft(units=enriched, relations=relations)


def _assemble_multimodal_units(
    parsed: ParsedDocument, limits: ParserLimits
) -> tuple[EvidenceUnitDraft, ...]:
    """Build bounded text regions plus independent image/table evidence units."""

    resolved_limits = limits
    assets = {asset.asset_key: asset for asset in parsed.assets}
    repeats = Counter(asset.content_sha256 for asset in parsed.assets)
    units: list[EvidenceUnitDraft] = []
    text_region: list[ParsedElement] = []
    anchor_key: str | None = None
    anchor_group_key: str | None = None
    anchor_asset_key: str | None = None
    anchor_text_kind = "text"
    text_insert_at: int | None = None

    def reset_text_context() -> None:
        nonlocal text_region, anchor_key, anchor_group_key, anchor_asset_key
        nonlocal anchor_text_kind, text_insert_at
        text_region = []
        anchor_key = None
        anchor_group_key = None
        anchor_asset_key = None
        anchor_text_kind = "text"
        text_insert_at = None

    def flush_text() -> None:
        nonlocal text_region, anchor_key, anchor_group_key, anchor_asset_key
        nonlocal anchor_text_kind, text_insert_at
        if not text_region:
            reset_text_context()
            return
        parts = _structural_region_parts(text_region)
        if not parts:
            reset_text_context()
            return
        created: list[EvidenceUnitDraft] = []
        for part_index, part in enumerate(parts):
            keys = ":".join(item.element_key or str(item.ordinal) for item in text_region)
            key = _key(
                anchor_text_kind,
                keys,
                str(part_index),
                hashlib.sha256(part.encode()).hexdigest(),
            )
            created.append(
                EvidenceUnitDraft(
                    unit_key=key,
                    ordinal=len(units) + len(created),
                    modality=ContentModality.TEXT,
                    content=part,
                    token_count=count_chunk_tokens(part),
                    asset_key=(
                        anchor_asset_key if anchor_text_kind == "ocr_text" else None
                    ),
                    evidence_group_key=(
                        anchor_group_key if anchor_text_kind == "ocr_text" else None
                    ),
                    related_unit_keys=((anchor_key,) if anchor_key else ()),
                    source_location=_span_location(text_region),
                    hierarchy=dict(text_region[0].hierarchy),
                    processing_metadata={
                        "assembly": "by_title_token_region_v1",
                        "representation_kind": anchor_text_kind,
                    },
                    required_representations=(anchor_text_kind,),
                )
            )
        if text_insert_at is None:
            units.extend(created)
        else:
            units[text_insert_at:text_insert_at] = created
        reset_text_context()

    for element in parsed.elements:
        if element.category == "PageBreak":
            flush_text()
            continue
        if element.category in {"Image", "PageImage", "Table", "TableChunk"} or (
            element.asset_key and element.category != "OCRText"
        ):
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
            elif anchor_text_kind == "ocr_text":
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
            if asset.kind == "page_image":
                anchor_group_key = image_group
                anchor_asset_key = asset.asset_key
                anchor_text_kind = "ocr_text"
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
        and item.category
        not in {
            "Image",
            "PageImage",
            "OCRText",
            "Table",
            "TableChunk",
            "FigureCaption",
        }
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


def normalize_figure_labels(text: str) -> tuple[str, ...]:
    """Return stable Figure identities without treating arbitrary numbers as labels."""

    normalized = unicodedata.normalize("NFC", text)
    labels = {
        f"figure:{match.group(1).casefold().replace(' ', '')}"
        for match in _FIGURE_REFERENCE.finditer(normalized)
    }
    return tuple(sorted(labels))


def _with_stable_evidence_groups(
    units: tuple[EvidenceUnitDraft, ...],
) -> tuple[EvidenceUnitDraft, ...]:
    enriched: list[EvidenceUnitDraft] = []
    for unit in units:
        labels = normalize_figure_labels(unit.content)
        metadata = dict(unit.processing_metadata)
        if labels:
            metadata["figure_labels"] = list(labels)
        group = unit.evidence_group_key
        if group is None:
            group = _key("composite-chunk-group-v2", unit.unit_key)
        elif unit.modality is ContentModality.IMAGE and labels and unit.asset_key:
            group = _key("figure-group-v2", labels[0], unit.asset_key)
        enriched.append(
            replace(
                unit,
                evidence_group_key=group,
                processing_metadata=metadata,
            )
        )
    return tuple(enriched)


@dataclass(frozen=True, slots=True)
class _RelationCandidate:
    chunk: EvidenceUnitDraft
    visual: EvidenceUnitDraft
    relation_type: ChunkAssetRelationType
    confidence_micros: int
    provenance: ChunkAssetRelationProvenance
    figure_label: str | None = None


def _assemble_relations(
    units: tuple[EvidenceUnitDraft, ...], limits: ParserLimits
) -> tuple[ChunkAssetRelationDraft, ...]:
    by_key = {unit.unit_key: unit for unit in units}
    visuals = tuple(
        unit
        for unit in units
        if unit.asset_key is not None
        and unit.modality in {ContentModality.IMAGE, ContentModality.TABLE}
    )
    labels_to_visuals: dict[str, list[EvidenceUnitDraft]] = {}
    for visual in visuals:
        for label in normalize_figure_labels(visual.content):
            labels_to_visuals.setdefault(label, []).append(visual)

    candidates: list[_RelationCandidate] = []
    for chunk in units:
        if chunk.modality is ContentModality.IMAGE:
            continue
        explicit_labels = _explicit_figure_reference_labels(chunk.content)
        for visual_key in chunk.related_unit_keys:
            visual = by_key.get(visual_key)
            if visual is None or visual.asset_key is None:
                raise ParserExecutionError(
                    ErrorCode.PARSER_OUTPUT_INVALID,
                    diagnostic={"check": "normalized_relation_target"},
                )
            is_ocr = (
                chunk.processing_metadata.get("representation_kind") == "ocr_text"
            )
            author_caption = bool(
                visual.processing_metadata.get("author_caption")
            )
            relation_type = (
                ChunkAssetRelationType.OCR_OF
                if is_ocr
                else ChunkAssetRelationType.CAPTION_OF
                if author_caption
                else ChunkAssetRelationType.INLINE_FIGURE
            )
            provenance = (
                ChunkAssetRelationProvenance.OCR_ASSET_IDENTITY_V2
                if is_ocr
                else ChunkAssetRelationProvenance.AUTHOR_CAPTION_V2
                if author_caption
                else ChunkAssetRelationProvenance.OOXML_RELATIONSHIP_V2
            )
            labels = normalize_figure_labels(visual.content)
            candidates.append(
                _RelationCandidate(
                    chunk,
                    visual,
                    relation_type,
                    1_000_000,
                    provenance,
                    labels[0] if labels else None,
                )
            )
            if explicit_labels and len(chunk.related_unit_keys) == 1:
                candidates.append(
                    _RelationCandidate(
                        chunk,
                        visual,
                        ChunkAssetRelationType.EXPLICIT_FIGURE_REFERENCE,
                        1_000_000,
                        ChunkAssetRelationProvenance.AUTHOR_REFERENCE_V2,
                        explicit_labels[0],
                    )
                )

        for label in normalize_figure_labels(chunk.content):
            for visual in labels_to_visuals.get(label, ()):
                if visual.unit_key == chunk.unit_key:
                    continue
                candidates.append(
                    _RelationCandidate(
                        chunk,
                        visual,
                        ChunkAssetRelationType.EXPLICIT_FIGURE_REFERENCE,
                        1_000_000,
                        ChunkAssetRelationProvenance.AUTHOR_REFERENCE_V2,
                        label,
                    )
                )

        if chunk.modality is ContentModality.TABLE and chunk.asset_key:
            candidates.append(
                _RelationCandidate(
                    chunk,
                    chunk,
                    ChunkAssetRelationType.TABLE_OF,
                    1_000_000,
                    ChunkAssetRelationProvenance.TABLE_IDENTITY_V2,
                )
            )

    strong_pairs = {
        (item.chunk.unit_key, item.visual.unit_key) for item in candidates
    }
    for chunk in units:
        if chunk.modality is not ContentModality.TEXT:
            continue
        chunk_pages = _unit_pages(chunk)
        if not chunk_pages:
            continue
        for visual in visuals:
            if (
                (chunk.unit_key, visual.unit_key) not in strong_pairs
                and chunk_pages.intersection(_unit_pages(visual))
            ):
                candidates.append(
                    _RelationCandidate(
                        chunk,
                        visual,
                        ChunkAssetRelationType.SAME_PAGE,
                        250_000,
                        ChunkAssetRelationProvenance.PAGE_IDENTITY_V2,
                    )
                )

    relation_priority = {
        relation: index
        for index, relation in enumerate(
            (
                ChunkAssetRelationType.EXPLICIT_FIGURE_REFERENCE,
                ChunkAssetRelationType.CAPTION_OF,
                ChunkAssetRelationType.INLINE_FIGURE,
                ChunkAssetRelationType.OCR_OF,
                ChunkAssetRelationType.TABLE_OF,
                ChunkAssetRelationType.SPATIAL_NEIGHBOR,
                ChunkAssetRelationType.SAME_PAGE,
            )
        )
    }
    candidates.sort(
        key=lambda item: (
            item.chunk.ordinal,
            relation_priority[item.relation_type],
            item.visual.ordinal,
            item.visual.asset_key or "",
        )
    )
    relations: list[ChunkAssetRelationDraft] = []
    seen: set[tuple[str, str, ChunkAssetRelationType]] = set()
    per_chunk: Counter[str] = Counter()
    for candidate in candidates:
        assert candidate.visual.asset_key is not None
        identity = (
            candidate.chunk.unit_key,
            candidate.visual.asset_key,
            candidate.relation_type,
        )
        if identity in seen:
            continue
        if per_chunk[candidate.chunk.unit_key] >= limits.max_relations_per_chunk:
            raise ParserExecutionError(
                ErrorCode.PARSER_RESOURCE_LIMIT,
                diagnostic={
                    "limit_name": "max_relations_per_chunk",
                    "limit": limits.max_relations_per_chunk,
                },
            )
        seen.add(identity)
        per_chunk[candidate.chunk.unit_key] += 1
        relations.append(
            ChunkAssetRelationDraft(
                chunk_unit_key=candidate.chunk.unit_key,
                visual_unit_key=candidate.visual.unit_key,
                asset_key=candidate.visual.asset_key,
                relation_type=candidate.relation_type,
                confidence_micros=candidate.confidence_micros,
                figure_label=candidate.figure_label,
                ordinal=len(relations),
                provenance=candidate.provenance,
                evidence_group_key=candidate.visual.evidence_group_key
                or _key("asset-group", candidate.visual.asset_key),
            )
        )
    if len(relations) > limits.max_relations:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic={
                "limit_name": "max_relations",
                "limit": limits.max_relations,
            },
        )
    return tuple(relations)


def _explicit_figure_reference_labels(value: str) -> tuple[str, ...]:
    labels: list[str] = []
    patterns = (
        r"(?i)(?:as\s+shown\s+in|shown\s+in|see|refer\s+to)\s+"
        r"(?:fig(?:ure)?\.?)\s*([0-9]+[A-Za-z]?)",
        r"(?:如图|见图|参见图)\s*([0-9]+[A-Za-z]?)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, value):
            label = f"figure:{match.group(1).lower()}"
            if label not in labels:
                labels.append(label)
    return tuple(sorted(labels))


def _unit_pages(unit: EvidenceUnitDraft) -> set[int]:
    pages = unit.source_location.get("page_numbers")
    if isinstance(pages, list):
        return {item for item in pages if isinstance(item, int) and not isinstance(item, bool)}
    page = unit.source_location.get("page_number")
    return {page} if isinstance(page, int) and not isinstance(page, bool) else set()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
