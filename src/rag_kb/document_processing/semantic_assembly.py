"""Assemble final index chunks from immutable semantic plans."""

from __future__ import annotations

import hashlib

from rag_kb.document_processing.profiles import SEMANTIC_CHUNKING_CONFIG
from rag_kb.document_processing.tokenization import count_chunk_tokens
from rag_kb.domain import (
    ErrorCode,
    IndexChunkDraft,
    IndexChunkPlan,
    IndexingExecutionError,
    IndexingPhase,
    ProcessedDocument,
    SemanticUnit,
)


def assemble_semantic_document(
    units: tuple[SemanticUnit, ...],
    plan: IndexChunkPlan,
) -> ProcessedDocument:
    cuts = (*[item.after_unit_ordinal + 1 for item in plan.boundaries], len(units))
    start = 0
    drafts: list[IndexChunkDraft] = []
    for ordinal, end in enumerate(cuts):
        selected = units[start:end]
        if not selected:
            raise _failed("empty_plan_range")
        text = "\n\n".join(unit.text for unit in selected).strip()
        token_count = count_chunk_tokens(text)
        if not text or token_count > int(SEMANTIC_CHUNKING_CONFIG["max_chunk_tokens"]):
            raise _failed("assembled_chunk_token_limit")
        reason = (
            plan.boundaries[ordinal].reason.value
            if ordinal < len(plan.boundaries)
            else None
        )
        drafts.append(
            IndexChunkDraft(
                ordinal=ordinal,
                text=text,
                token_count=token_count,
                source_location=_source_location(selected),
                hierarchy=_hierarchy(selected),
                processing_metadata={
                    "profile": SEMANTIC_CHUNKING_CONFIG["profile"],
                    "unit_start": selected[0].ordinal,
                    "unit_end": selected[-1].ordinal,
                    "boundary_reason": reason,
                    "plan_hash": plan.plan_hash,
                },
                content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        )
        start = end
    if start != len(units) or len(drafts) != plan.chunk_count:
        raise _failed("plan_coverage")
    return ProcessedDocument(
        chunks=tuple(drafts),
        extracted_character_count=sum(len(unit.text) for unit in units),
    )


def _source_location(units: tuple[SemanticUnit, ...]) -> dict:
    pages = [
        value
        for unit in units
        for key in ("page_start", "page_end")
        if isinstance((value := unit.source_location.get(key)), int)
        and not isinstance(value, bool)
    ]
    if not pages:
        return {}
    result = {"page_start": min(pages), "page_end": max(pages)}
    if result["page_start"] == result["page_end"]:
        coordinates = [unit.source_location.get("coordinates") for unit in units]
        if coordinates and all(value == coordinates[0] for value in coordinates):
            if coordinates[0] is not None:
                result["coordinates"] = coordinates[0]
    return result


def _hierarchy(units: tuple[SemanticUnit, ...]) -> dict:
    title_lists = [
        value
        for unit in units
        if isinstance((value := unit.hierarchy.get("titles")), list)
    ]
    if not title_lists:
        return {}
    common = list(title_lists[0])
    for titles in title_lists[1:]:
        length = 0
        for left, right in zip(common, titles):
            if left != right:
                break
            length += 1
        common = common[:length]
    return {"titles": common} if common else {}


def _failed(check: str) -> IndexingExecutionError:
    return IndexingExecutionError(
        ErrorCode.SEMANTIC_CHUNKING_FAILED,
        phase=IndexingPhase.SEMANTIC_ANALYSIS,
        diagnostic={"check": check},
    )
