"""Project Docling assembly output onto the persisted composite evidence model.

Chunk assembly, asset extraction and relation building all speak Docling
references. Persistence, representation planning and embedding speak evidence
units keyed by stable identities. This module is the single translation between
them, so no Docling type reaches a repository.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from docling_core.types.doc import DoclingDocument

from rag_kb.document_processing.docling.assets import (
    ASSET_KIND_PAGE_IMAGE,
    ASSET_KIND_TABLE_IMAGE,
)
from rag_kb.document_processing.docling.provenance import (
    chunk_assembly_key,
    item_surfaces,
    surface_kind,
    surface_ordinals,
)
from rag_kb.document_processing.docling.traversal import (
    classify_item,
    item_ref,
    iterate_body_items,
)
from rag_kb.document_processing.multimodal_assembly import normalize_figure_labels
from rag_kb.domain import (
    ChunkAssemblyDraft,
    ChunkAssetRelationDraft,
    CompositeEvidenceDraft,
    ContentModality,
    DoclingAssetRelationDraft,
    ErrorCode,
    EvidenceUnitDraft,
    IndexChunkDraft,
    ParsedAssetDraft,
    ParserExecutionError,
    ParserLimits,
    ProcessedDocument,
)


_TABLE_REF_PREFIX = "#/tables/"


def text_only_document(
    chunks: tuple[ChunkAssemblyDraft, ...],
    *,
    profile: str,
) -> ProcessedDocument:
    """Project text-only assembly onto the existing chunk persistence shape."""

    if not chunks:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "non_empty_chunks"},
        )
    drafts = tuple(
        IndexChunkDraft(
            ordinal=ordinal,
            text=chunk.text,
            token_count=chunk.token_count,
            source_location=dict(chunk.source_location),
            hierarchy=dict(chunk.hierarchy),
            processing_metadata={
                "profile": profile,
                "item_ref_count": len(chunk.item_refs),
            },
            content_sha256=hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
        )
        for ordinal, chunk in enumerate(chunks)
    )
    return ProcessedDocument(
        chunks=drafts,
        extracted_character_count=sum(len(chunk.text) for chunk in chunks),
    )


def composite_evidence(
    document: DoclingDocument,
    chunks: tuple[ChunkAssemblyDraft, ...],
    assets: tuple[ParsedAssetDraft, ...],
    relations: tuple[DoclingAssetRelationDraft, ...],
    *,
    profile: str,
    source_checksum_sha256: str,
    limits: ParserLimits | None = None,
) -> CompositeEvidenceDraft:
    """Build ordered evidence units plus relations keyed by stable identities."""

    resolved = limits or ParserLimits()
    order: dict[str, int] = {}
    surface_starts: dict[int, int] = {}
    kind = surface_kind(document)
    for position, (item, _level) in enumerate(iterate_body_items(document)):
        order[item_ref(item)] = position
        for surface in item_surfaces(item, kind=kind):
            surface_starts.setdefault(surface.ordinal, position)
    table_images = {
        asset.processing_metadata.get("item_ref"): asset
        for asset in assets
        if asset.kind == ASSET_KIND_TABLE_IMAGE
    }

    chunk_keys: list[str] = []
    table_unit_keys: dict[str, str] = {}
    placed: list[tuple[int, int, EvidenceUnitDraft]] = []
    for index, chunk in enumerate(chunks):
        key = chunk_assembly_key(
            profile=profile,
            source_checksum_sha256=source_checksum_sha256,
            item_refs=chunk.item_refs,
            text=chunk.text,
        )
        chunk_keys.append(key)
        table_ref = next(
            (
                reference
                for reference in chunk.item_refs
                if reference.startswith(_TABLE_REF_PREFIX)
            ),
            None,
        )
        table_image = table_images.get(table_ref) if table_ref else None
        modality = (
            ContentModality.TABLE if table_ref is not None else ContentModality.TEXT
        )
        if table_image is not None:
            table_unit_keys[table_image.asset_key] = key
        placed.append(
            (
                _position(order, chunk.item_refs),
                0,
                EvidenceUnitDraft(
                    unit_key=key,
                    ordinal=index,
                    modality=modality,
                    content=chunk.text,
                    token_count=chunk.token_count,
                    asset_key=table_image.asset_key if table_image else None,
                    evidence_group_key=(
                        _key("docling-table-group-v1", table_ref)
                        if table_ref is not None
                        else _key("docling-chunk-group-v1", key)
                    ),
                    related_unit_keys=(),
                    source_location=dict(chunk.source_location),
                    hierarchy=dict(chunk.hierarchy),
                    processing_metadata=_chunk_metadata(chunk, profile, table_ref),
                    required_representations=(
                        ("table_text",)
                        if modality is ContentModality.TABLE
                        else ("text",)
                    ),
                ),
            )
        )

    asset_units: dict[str, str] = {}
    for asset in assets:
        if asset.kind == ASSET_KIND_TABLE_IMAGE:
            # A table image is an optional representation of its table chunk, not
            # an independent visual unit.
            continue
        key = _key("docling-visual-v1", asset.asset_key)
        asset_units[asset.asset_key] = key
        caption = asset.processing_metadata.get("caption")
        content = caption if isinstance(caption, str) else ""
        placed.append(
            (
                _asset_position(order, surface_starts, asset),
                1,
                EvidenceUnitDraft(
                    unit_key=key,
                    ordinal=0,
                    modality=ContentModality.IMAGE,
                    content=content,
                    token_count=0,
                    asset_key=asset.asset_key,
                    evidence_group_key=_evidence_group(asset, content),
                    related_unit_keys=(),
                    source_location=dict(asset.source_location),
                    hierarchy={},
                    processing_metadata={
                        "asset_kind": asset.kind,
                        "author_caption": bool(content),
                        **(
                            {"figure_labels": list(normalize_figure_labels(content))}
                            if content and normalize_figure_labels(content)
                            else {}
                        ),
                    },
                    required_representations=("native_image",),
                ),
            )
        )

    ordered = tuple(
        _reordinal(unit, ordinal)
        for ordinal, (_position_value, _kind, unit) in enumerate(
            sorted(placed, key=lambda item: (item[0], item[1], item[2].unit_key))
        )
    )
    if len(ordered) > resolved.max_units:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic={"limit_name": "max_units", "limit": resolved.max_units},
        )
    by_key = {unit.unit_key: unit for unit in ordered}
    return CompositeEvidenceDraft(
        units=ordered,
        relations=_relations(
            relations,
            chunk_keys=chunk_keys,
            visual_keys={**table_unit_keys, **asset_units},
            by_key=by_key,
            limits=resolved,
        ),
    )


def docling_item_sequence_hash(document: DoclingDocument) -> str:
    """Hash the traversal a manifest was built from, without item contents."""

    projection = [
        {
            "item_ref": item_ref(item),
            "position": position,
            "kind": classify_item(item).value,
        }
        for position, (item, _level) in enumerate(iterate_body_items(document))
    ]
    return _canonical_hash(projection)


def _relations(
    relations: tuple[DoclingAssetRelationDraft, ...],
    *,
    chunk_keys: list[str],
    visual_keys: dict[str, str],
    by_key: dict[str, EvidenceUnitDraft],
    limits: ParserLimits,
) -> tuple[ChunkAssetRelationDraft, ...]:
    drafts: list[ChunkAssetRelationDraft] = []
    per_chunk: dict[str, int] = {}
    for relation in relations:
        if not 0 <= relation.chunk_index < len(chunk_keys):
            raise ParserExecutionError(
                ErrorCode.PARSER_OUTPUT_INVALID,
                diagnostic={"check": "relation_chunk_index"},
            )
        chunk_key = chunk_keys[relation.chunk_index]
        # A table image is carried by its own table chunk, so its relations
        # point back at that unit rather than at a separate visual unit.
        visual = by_key.get(visual_keys.get(relation.asset_key, ""))
        visual_key = visual.unit_key if visual is not None else ""
        if visual is None or visual.asset_key != relation.asset_key:
            raise ParserExecutionError(
                ErrorCode.PARSER_OUTPUT_INVALID,
                diagnostic={"check": "relation_visual_unit"},
            )
        observed = per_chunk.get(chunk_key, 0) + 1
        if observed > limits.max_relations_per_chunk:
            raise ParserExecutionError(
                ErrorCode.PARSER_RESOURCE_LIMIT,
                diagnostic={
                    "limit_name": "max_relations_per_chunk",
                    "limit": limits.max_relations_per_chunk,
                },
            )
        per_chunk[chunk_key] = observed
        drafts.append(
            ChunkAssetRelationDraft(
                chunk_unit_key=chunk_key,
                visual_unit_key=visual_key,
                asset_key=relation.asset_key,
                relation_type=relation.relation_type,
                confidence_micros=relation.confidence_micros,
                figure_label=relation.figure_label,
                ordinal=len(drafts),
                provenance=relation.provenance,
                evidence_group_key=visual.evidence_group_key
                or _key("docling-visual-group-v1", relation.asset_key),
            )
        )
    if len(drafts) > limits.max_relations:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic={"limit_name": "max_relations", "limit": limits.max_relations},
        )
    return tuple(drafts)


def _chunk_metadata(
    chunk: ChunkAssemblyDraft,
    profile: str,
    table_ref: str | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "profile": profile,
        "item_ref_count": len(chunk.item_refs),
    }
    if table_ref is not None:
        metadata["table_item_ref"] = table_ref
    labels = normalize_figure_labels(chunk.text)
    if labels:
        metadata["figure_labels"] = list(labels)
    return metadata


def _evidence_group(asset: ParsedAssetDraft, caption: str) -> str:
    labels = normalize_figure_labels(caption) if caption else ()
    if labels:
        return _key("docling-figure-group-v1", labels[0], asset.asset_key)
    return _key("docling-visual-group-v1", asset.asset_key)


def _position(order: dict[str, int], refs: tuple[str, ...]) -> int:
    positions = [order[reference] for reference in refs if reference in order]
    return min(positions) if positions else len(order)


def _asset_position(
    order: dict[str, int],
    surface_starts: dict[int, int],
    asset: ParsedAssetDraft,
) -> int:
    reference = asset.processing_metadata.get("item_ref")
    if isinstance(reference, str) and reference in order:
        return order[reference]
    if asset.kind == ASSET_KIND_PAGE_IMAGE:
        # A page image belongs to its whole surface, so it sorts with that
        # surface's first item rather than at the end of the document.
        surfaces = sorted(surface_ordinals(dict(asset.source_location)))
        for ordinal in surfaces:
            if ordinal in surface_starts:
                return surface_starts[ordinal]
    return len(order)


def _reordinal(unit: EvidenceUnitDraft, ordinal: int) -> EvidenceUnitDraft:
    return EvidenceUnitDraft(
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
        embedding_text=unit.embedding_text,
        embedding_text_hash=unit.embedding_text_hash,
    )


def _key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
