"""Prompt evidence construction and deterministic answer rendering."""

from __future__ import annotations

from collections.abc import Mapping
import re

from rag_kb.domain import (
    AnswerClaim,
    AnswerControlReason,
    AnswerDraftSource,
    AnswerOutcome,
    EvidenceEnvelope,
    EvidencePack,
    PromptEvidence,
    RenderedAnswer,
    RenderedCitation,
    ValidatedAnswer,
)


_INLINE_EVIDENCE_GROUP = re.compile(
    r"[\[\uFF3B\u3010\(\uFF08]\s*"
    r"(?P<refs>ev_\d+(?:\s*[,，;；、]\s*ev_\d+)*)\s*"
    r"[\]\uFF3D\u3011\)\uFF09]",
    re.IGNORECASE,
)
_INLINE_EVIDENCE_REF = re.compile(r"ev_\d+", re.IGNORECASE)


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
                asset_snapshot=None,
                matched_representations=item.matched_representations,
                document_display_name=item.document_display_name or "document",
                document_original_filename=item.document_original_filename or "document",
                graph_path_id=item.graph_path_id,
                graph_anchor_index_chunk_id=item.graph_anchor_index_chunk_id,
                graph_hop_count=item.graph_hop_count,
                graph_path_rank=item.graph_path_rank,
            )
            for item in pack.evidence
        ),
    )


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

    if answer.outcome is AnswerOutcome.CLARIFY:
        questions = tuple(value.rstrip() for value in answer.missing_aspects)
        content = (
            "在回答之前，我需要先和你确认：" + "；".join(questions)
            if _contains_cjk(current_query)
            else "Before I can answer, I need to clarify: " + "; ".join(questions)
        )
        return RenderedAnswer(
            outcome=answer.outcome,
            content=content,
            citations=(),
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
                    RenderedCitation(ordinal=ordinal, evidence=item)
                )
            markers.append(f"[{ordinals[citation_id] + 1}]")
        paragraphs.append(f"{claim.text} {''.join(markers)}")
    if answer.missing_aspects:
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


def render_text_final_answer(
    content: str,
    prompt_by_ref: Mapping[str, PromptEvidence],
    *,
    loaded_visual_refs: set[str],
    current_query: str,
) -> tuple[ValidatedAnswer, RenderedAnswer, tuple[str, ...], tuple[str, ...]]:
    """Resolve provider-written inline ``ev_N`` groups into display citations.

    The returned ref tuples contain, respectively, retained refs in first-use
    order and all syntactically observed refs. Unknown refs and visual-only
    refs whose asset was not sent to the model disappear without blocking the
    answer.
    """

    ordinals: dict[str, int] = {}
    citations: list[RenderedCitation] = []
    observed_refs: list[str] = []

    def replace_group(match: re.Match[str]) -> str:
        markers: list[str] = []
        for raw_ref in _INLINE_EVIDENCE_REF.findall(match.group("refs")):
            ref = raw_ref.lower()
            observed_refs.append(ref)
            prompt = prompt_by_ref.get(ref)
            if prompt is None or (
                _requires_loaded_visual(prompt) and ref not in loaded_visual_refs
            ):
                continue
            if ref not in ordinals:
                ordinal = len(citations)
                ordinals[ref] = ordinal
                citations.append(RenderedCitation(ordinal=ordinal, evidence=prompt))
            markers.append(f"[{ordinals[ref] + 1}]")
        return "".join(markers)

    rendered_content = _INLINE_EVIDENCE_GROUP.sub(replace_group, content).strip()
    if not rendered_content:
        rendered_content = (
            "当前知识库没有足够证据回答这个问题。"
            if _contains_cjk(current_query)
            else "The available evidence is insufficient to answer reliably."
        )
    retained_refs = tuple(ordinals)
    citation_ids = tuple(
        prompt_by_ref[ref].citation_id for ref in retained_refs
    )
    outcome = AnswerOutcome.ANSWERED if citations else AnswerOutcome.REFUSED
    validated = ValidatedAnswer(
        outcome=outcome,
        claims=(AnswerClaim(rendered_content, citation_ids),),
        missing_aspects=(),
        source=AnswerDraftSource.PROVIDER,
        control_reason=(
            AnswerControlReason.INSUFFICIENT_EVIDENCE
            if outcome is AnswerOutcome.REFUSED
            else None
        ),
    )
    rendered = RenderedAnswer(
        outcome=outcome,
        content=rendered_content,
        citations=tuple(citations),
        control_reason=validated.control_reason,
    )
    return validated, rendered, retained_refs, tuple(observed_refs)


def _requires_loaded_visual(prompt: PromptEvidence) -> bool:
    return not any(
        item in {"text", "caption_text", "ocr_text", "table_text"}
        for item in prompt.matched_representations
    )


def _contains_cjk(value: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in value)
