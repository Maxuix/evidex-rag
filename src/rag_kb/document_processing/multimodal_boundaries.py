"""Deterministic visual element classification and hard-boundary policy."""

from __future__ import annotations

from enum import StrEnum

from rag_kb.domain import ParsedAssetDraft, ParsedElement


class VisualDisposition(StrEnum):
    DECORATIVE = "decorative"
    ANCHORED = "anchored"
    HARD_BOUNDARY = "hard_boundary"


def classify_visual(
    element: ParsedElement,
    asset: ParsedAssetDraft,
    *,
    repeated_hash_count: int = 1,
) -> VisualDisposition:
    width = asset.width or 0
    height = asset.height or 0
    location = element.source_location
    page_width = location.get("page_width")
    page_height = location.get("page_height")
    area_ratio = 0.0
    if (
        isinstance(page_width, (int, float))
        and isinstance(page_height, (int, float))
        and page_width > 0
        and page_height > 0
    ):
        area_ratio = (width * height) / float(page_width * page_height)
    if width < 64 or height < 64 or (repeated_hash_count >= 3 and area_ratio < 0.08):
        return VisualDisposition.DECORATIVE
    if element.category in {"Table", "TableChunk"} or asset.kind in {
        "page_image",
        "table_image",
    }:
        return VisualDisposition.HARD_BOUNDARY
    if area_ratio >= 0.5:
        return VisualDisposition.HARD_BOUNDARY
    return VisualDisposition.ANCHORED
