"""Deterministic bounded text representation for Composite Evidence v2."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import replace

from rag_kb.document_processing.profiles import MULTIMODAL_REPRESENTATION_CONFIG
from rag_kb.document_processing.tokenization import count_chunk_tokens, split_by_tokens
from rag_kb.domain import (
    ChunkAssetRelationDraft,
    ChunkAssetRelationType,
    ContentModality,
    EvidenceUnitDraft,
    ErrorCode,
    ParserExecutionError,
)


def with_composite_embedding_text(
    units: tuple[EvidenceUnitDraft, ...],
    relations: tuple[ChunkAssetRelationDraft, ...],
) -> tuple[EvidenceUnitDraft, ...]:
    """Attach stable embedding text without changing display content or boundaries."""

    config = MULTIMODAL_REPRESENTATION_CONFIG["embedding_text"]
    maximum = int(config["max_tokens"])
    attachment_maximum = int(config["max_attachment_tokens"])
    by_key = {unit.unit_key: unit for unit in units}
    by_chunk: dict[str, list[ChunkAssetRelationDraft]] = {}
    for relation in relations:
        by_chunk.setdefault(relation.chunk_unit_key, []).append(relation)

    enriched: list[EvidenceUnitDraft] = []
    for unit in units:
        if unit.modality is ContentModality.IMAGE:
            enriched.append(unit)
            continue
        body = _canonical(unit.content)
        if not body:
            raise ParserExecutionError(
                ErrorCode.PARSER_OUTPUT_INVALID,
                diagnostic={"check": "composite_embedding_body"},
            )
        if unit.modality is ContentModality.TABLE:
            body_label = "table"
        elif unit.processing_metadata.get("representation_kind") == "ocr_text":
            body_label = "ocr"
        else:
            body_label = "body"
        sections = [f"[{body_label}]\n{body}"]

        related = sorted(
            by_chunk.get(unit.unit_key, ()),
            key=lambda item: (
                item.ordinal,
                item.relation_type.value,
                item.visual_unit_key,
            ),
        )
        labels = tuple(
            dict.fromkeys(
                item.figure_label for item in related if item.figure_label is not None
            )
        )
        if labels:
            _append_bounded_section(
                sections,
                "figure_label",
                "\n".join(labels),
                maximum=maximum,
                section_maximum=attachment_maximum,
            )

        captions: list[str] = []
        for relation in related:
            visual = by_key.get(relation.visual_unit_key)
            if visual is None:
                raise ParserExecutionError(
                    ErrorCode.PARSER_OUTPUT_INVALID,
                    diagnostic={"check": "composite_embedding_relation_target"},
                )
            if (
                relation.relation_type
                in {
                    ChunkAssetRelationType.CAPTION_OF,
                    ChunkAssetRelationType.EXPLICIT_FIGURE_REFERENCE,
                }
                and visual.processing_metadata.get("author_caption")
                and visual.content
            ):
                caption = _canonical(visual.content)
                if caption and caption not in captions:
                    captions.append(caption)
        if captions:
            _append_bounded_section(
                sections,
                "author_caption",
                "\n".join(captions),
                maximum=maximum,
                section_maximum=attachment_maximum,
            )

        embedding_text = "\n".join(sections)
        if count_chunk_tokens(embedding_text) > maximum:
            raise ParserExecutionError(
                ErrorCode.PARSER_RESOURCE_LIMIT,
                diagnostic={
                    "limit_name": "composite_embedding_text_tokens",
                    "limit": maximum,
                },
            )
        enriched.append(
            replace(
                unit,
                embedding_text=embedding_text,
                embedding_text_hash=hashlib.sha256(
                    embedding_text.encode("utf-8")
                ).hexdigest(),
            )
        )
    return tuple(enriched)


def _append_bounded_section(
    sections: list[str],
    label: str,
    value: str,
    *,
    maximum: int,
    section_maximum: int,
) -> None:
    canonical = _canonical(value)
    if not canonical:
        return
    if count_chunk_tokens(canonical) > section_maximum:
        canonical = split_by_tokens(
            canonical, max_tokens=section_maximum, overlap_tokens=0
        )[0]
    prefix = f"[{label}]\n"
    current = "\n".join(sections)
    remaining = maximum - count_chunk_tokens(f"{current}\n{prefix}")
    if remaining <= 0:
        return
    if count_chunk_tokens(canonical) > remaining:
        canonical = split_by_tokens(
            canonical, max_tokens=remaining, overlap_tokens=0
        )[0]
    if canonical:
        sections.append(f"{prefix}{canonical}")


def _canonical(value: str) -> str:
    return unicodedata.normalize(
        "NFC", value.replace("\r\n", "\n").replace("\r", "\n")
    ).strip()
