"""Bounded prompt evidence projection and deterministic answer rendering."""

from __future__ import annotations

import re

from rag_kb.domain import (
    AnswerControlReason,
    AnswerOutcome,
    EvidenceEnvelope,
    EvidencePack,
    PromptEvidence,
    RenderedAnswer,
    RenderedCitation,
    ValidatedAnswer,
)


_OMITTED = "\n[… omitted by bounded evidence projection …]\n"


def build_evidence_envelope(pack: EvidencePack) -> EvidenceEnvelope:
    return EvidenceEnvelope(
        knowledge_base_id=pack.knowledge_base_id,
        index_revision_id=pack.index_revision_id,
        items=tuple(
            PromptEvidence(
                citation_id=f"cite_{item.rank}",
                rank=item.rank,
                index_chunk_id=item.index_chunk_id,
                document_id=item.document_id,
                document_version_id=item.document_version_id,
                excerpt=item.text or f"[{item.modality} visual evidence]",
                source_location=item.source_location,
                score=item.score,
                modality=item.modality,
                asset_snapshot=(
                    {
                        "id": str(item.asset.id),
                        "media_type": item.asset.media_type,
                        "checksum_sha256": item.asset.checksum_sha256,
                        "content_url": item.asset.content_url,
                        "width": item.asset.width,
                        "height": item.asset.height,
                    }
                    if item.asset is not None
                    else None
                ),
                matched_representations=item.matched_representations,
                document_display_name=item.document_display_name or "document",
                document_original_filename=item.document_original_filename or "document",
            )
            for item in pack.evidence
        ),
    )


def project_evidence_text(
    text: str,
    focus: tuple[str, ...],
    *,
    max_chars: int = 2400,
) -> str:
    """Keep deterministic query-related windows within one hard character bound."""

    if max_chars < 1:
        raise ValueError("projection max_chars must be positive")
    if len(text) <= max_chars:
        return text
    terms = tuple(
        dict.fromkeys(
            token.casefold()
            for value in focus
            for token in re.findall(r"[\w\u3400-\u9fff]{2,}", value)
        )
    )[:32]
    lowered = text.casefold()
    positions = sorted(
        {
            position
            for term in terms
            if (position := lowered.find(term)) >= 0
        }
    )
    if not positions:
        return text[:max_chars]
    marker_budget = len(_OMITTED)
    window = max(80, (max_chars - marker_budget * 2) // min(3, len(positions)))
    spans: list[tuple[int, int]] = []
    for position in positions:
        start = max(0, position - window // 2)
        end = min(len(text), start + window)
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))
        if len(spans) == 3:
            break
    pieces: list[str] = []
    for index, (start, end) in enumerate(spans):
        if index:
            pieces.append(_OMITTED)
        pieces.append(text[start:end])
    return "".join(pieces)[:max_chars]


def render_validated_answer(
    answer: ValidatedAnswer,
    evidence: EvidenceEnvelope,
    *,
    current_query: str,
) -> RenderedAnswer:
    if answer.outcome is AnswerOutcome.REFUSED:
        content = (
            "当前知识库没有足够证据回答这个问题。"
            if _contains_cjk(current_query)
            else "The available evidence is insufficient to answer reliably."
        )
        return RenderedAnswer(
            outcome=answer.outcome,
            content=content,
            citations=(),
            control_reason=answer.control_reason or AnswerControlReason.INSUFFICIENT_EVIDENCE,
        )

    lookup = {item.citation_id: item for item in evidence.items}
    ordinals: dict[str, int] = {}
    citations: list[RenderedCitation] = []
    paragraphs: list[str] = []
    for claim in answer.claims:
        markers: list[str] = []
        for citation_id in claim.citation_ids:
            item = lookup[citation_id]
            if citation_id not in ordinals:
                ordinal = len(citations)
                ordinals[citation_id] = ordinal
                citations.append(
                    RenderedCitation(
                        ordinal=ordinal,
                        citation_id=citation_id,
                        index_chunk_id=item.index_chunk_id,
                        document_id=item.document_id,
                        document_version_id=item.document_version_id,
                        document_display_name=item.document_display_name,
                        document_original_filename=item.document_original_filename,
                        quoted_text=item.excerpt,
                        source_location=item.source_location,
                        score=item.score,
                        modality=item.modality,
                        asset_snapshot=item.asset_snapshot,
                        matched_representations=item.matched_representations,
                    )
                )
            markers.append(f"[{ordinals[citation_id] + 1}]")
        paragraphs.append(f"{claim.text} {''.join(markers)}")
    if answer.outcome is AnswerOutcome.PARTIAL:
        topics = tuple(value.rstrip("。.!！?？;；") for value in answer.missing_aspects)
        paragraphs.append(
            f"另外，关于{'、'.join(f'“{item}”' for item in topics)}，我目前无法给出可靠回答。"
            if _contains_cjk(current_query)
            else "I can’t reliably answer these parts yet: " + "; ".join(topics) + "."
        )
    return RenderedAnswer(
        outcome=answer.outcome,
        content="\n\n".join(paragraphs),
        citations=tuple(citations),
    )


def _contains_cjk(value: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in value)
