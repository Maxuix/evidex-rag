"""Bounded source relationships used to complete table retrieval candidates."""

from __future__ import annotations

from rag_kb.domain import VectorSearchHit
from rag_kb.retrieval.reranker import RerankedHit, score_hits


SOURCE_CONTEXT_ANCHOR_LIMIT = 100


def rank_with_source_context(
    query: str,
    core: tuple[VectorSearchHit, ...],
    supplements: tuple[VectorSearchHit, ...],
    *,
    top_k: int,
    vector_weight: float = 0.65,
    lexical_weight: float = 0.35,
) -> tuple[RerankedHit, ...]:
    """Protect the leading core half and score table supplements against fixed statistics."""

    def ordered(
        hits: tuple[VectorSearchHit, ...],
        reference: tuple[VectorSearchHit, ...] | None = None,
    ) -> list[RerankedHit]:
        return sorted(
            score_hits(
                query,
                hits,
                vector_weight=vector_weight,
                lexical_weight=lexical_weight,
                reference_hits=reference,
            ),
            key=lambda s: (-s.score, s.hit.cosine_distance, s.hit.index_chunk_id.int),
        )

    if top_k < 1:
        raise ValueError("source context top_k must be positive")
    if not core:
        if supplements:
            raise ValueError("source context requires a core candidate pool")
        return ()
    if len(supplements) > 2 * SOURCE_CONTEXT_ANCHOR_LIMIT:
        raise ValueError("source context supplement budget exceeded")
    ids = [h.index_chunk_id for h in (*core, *supplements)]
    if len(ids) != len(set(ids)):
        raise ValueError("source context candidates must be unique")
    baseline = ordered(core)
    if not supplements:
        return tuple(baseline[:top_k])
    protected = baseline[: (top_k + 1) // 2]
    protected_ids = {s.hit.index_chunk_id for s in protected}
    remaining = [
        s for s in ordered((*core, *supplements), core)
        if s.hit.index_chunk_id not in protected_ids
    ]
    return tuple((protected + remaining)[:top_k])


def table_neighbor_compatible(anchor: VectorSearchHit, neighbor: VectorSearchHit) -> bool:
    """Admit a physical table neighbor without crossing a known section boundary."""
    if anchor.indexed_document_version_id != neighbor.indexed_document_version_id:
        return False
    if abs(anchor.ordinal - neighbor.ordinal) != 1:
        return False
    if anchor.modality not in {"text", "table"} or neighbor.modality != "table":
        return False
    left = tuple(
        t["text"] for t in anchor.hierarchy.get("titles", ())
        if isinstance(t, dict) and t.get("text")
    )
    right = tuple(
        t["text"] for t in neighbor.hierarchy.get("titles", ())
        if isinstance(t, dict) and t.get("text")
    )
    if left and right and left != right:
        return False
    a, b = anchor.source_location, neighbor.source_location
    if a.get("surface_type") != b.get("surface_type"):
        return False
    if a.get("surface_type") == "page":
        start, end = a.get("surface_start"), b.get("surface_start")
        if type(start) is not int or type(end) is not int or abs(start - end) > 1:
            return False
    return True
