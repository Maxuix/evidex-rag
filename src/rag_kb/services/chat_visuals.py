"""Prepare authorized, bounded visual evidence for the final chat model."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol
from uuid import UUID

from rag_kb.auth import AuthContext
from rag_kb.domain import (
    ChatAnsweringState,
    ChatModelVisualContent,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatPipelineState,
    ErrorCode,
    EvidenceAssessment,
    EvidenceCoverage,
    EvidenceEnvelope,
    FileStoreError,
    IndexAssetContent,
    PromptEvidence,
    ResourceNotFoundError,
    VisualEvidenceReason,
)
from rag_kb.services.visual_admission import VisualEvidenceAdmissionPolicy


class IndexAssetReader(Protocol):
    async def read(
        self, context: AuthContext, asset_id: UUID
    ) -> IndexAssetContent: ...


class VisualEvidencePreparationStep:
    """Attach only admitted and integrity-checked images to model requests."""

    def __init__(
        self,
        asset_reader: IndexAssetReader | None,
        *,
        max_images: int = 2,
        max_image_bytes: int = 5 * 1024 * 1024,
        max_total_bytes: int = 12 * 1024 * 1024,
        max_pixels: int = 16_000_000,
        admission_policy: VisualEvidenceAdmissionPolicy | None = None,
    ) -> None:
        if (
            max_images < 1
            or max_images > VisualEvidenceAdmissionPolicy.HARD_MAX_IMAGES
            or max_image_bytes < 1
            or max_total_bytes < max_image_bytes
            or max_pixels < 1
        ):
            raise ValueError("visual evidence limits are invalid")
        self._asset_reader = asset_reader
        self._max_images = max_images
        self._max_image_bytes = max_image_bytes
        self._max_total_bytes = max_total_bytes
        self._max_pixels = max_pixels
        self._admission_policy = admission_policy or VisualEvidenceAdmissionPolicy()

    async def run(self, state: ChatPipelineState) -> ChatPipelineState:
        context = state.context
        pack = state.evidence_pack
        answering = state.answering
        if (
            context is None
            or pack is None
            or answering is None
            or answering.draft is not None
            or answering.visual_content
        ):
            raise _context_error("visual_preparation_state")

        usable = list(answering.assessment.usable_citation_ids)
        initially_usable = frozenset(usable)
        usable_set = set(usable)
        attached_native_citation_ids: set[str] = set()
        visual_content: list[ChatModelVisualContent] = []
        evidence_items = list(answering.evidence.items)
        total_bytes = 0
        auth = AuthContext(
            principal_id=context.principal_id,
            client_id=context.client_id,
            workspace_id=context.workspace_id,
        )

        candidates = self._admission_policy.rank_candidates(
            pack, answering.assessment.usable_citation_ids
        )
        decisions = list(
            self._admission_policy.decide(
                pack,
                answering.assessment.usable_citation_ids,
                max_images=self._max_images,
            )
        )
        decision_indexes = {
            (item.visual_unit_id, item.asset_id): index
            for index, item in enumerate(decisions)
        }
        for candidate in candidates:
            if len(visual_content) >= self._max_images:
                break
            evidence = candidate.parent_evidence
            citation_id = candidate.parent_citation_id
            asset = candidate.asset
            text_fallback = _has_textual_representation(evidence)
            if (
                self._asset_reader is None
            ):
                _record_decision(
                    decisions,
                    decision_indexes,
                    candidate,
                    VisualEvidenceReason.REJECTED_UNAUTHORIZED,
                )
                if not text_fallback:
                    usable_set.discard(citation_id)
                continue
            if (
                asset.media_type not in {"image/jpeg", "image/png", "image/webp"}
            ):
                _record_decision(
                    decisions,
                    decision_indexes,
                    candidate,
                    VisualEvidenceReason.REJECTED_UNSUPPORTED_MEDIA,
                )
                if not text_fallback:
                    usable_set.discard(citation_id)
                continue
            if (
                asset.width is None
                or asset.height is None
                or asset.width * asset.height > self._max_pixels
            ):
                _record_decision(
                    decisions,
                    decision_indexes,
                    candidate,
                    VisualEvidenceReason.REJECTED_VISUAL_BUDGET,
                )
                if not text_fallback:
                    usable_set.discard(citation_id)
                continue

            try:
                loaded = await self._asset_reader.read(auth, asset.id)
            except (FileStoreError, ResourceNotFoundError):
                _record_decision(
                    decisions,
                    decision_indexes,
                    candidate,
                    VisualEvidenceReason.REJECTED_ASSET_INTEGRITY,
                )
                if not text_fallback:
                    usable_set.discard(citation_id)
                continue
            snapshot = loaded.snapshot
            if (
                snapshot.id != asset.id
                or snapshot.workspace_id != context.workspace_id
                or snapshot.kb_id != context.knowledge_base_id
                or snapshot.document_id != evidence.document_id
                or snapshot.document_version_id != evidence.document_version_id
                or snapshot.indexed_document_version_id
                != evidence.indexed_document_version_id
                or snapshot.media_type != asset.media_type
                or snapshot.checksum_sha256 != asset.checksum_sha256
            ):
                _record_decision(
                    decisions,
                    decision_indexes,
                    candidate,
                    VisualEvidenceReason.REJECTED_ASSET_INTEGRITY,
                )
                if not text_fallback:
                    usable_set.discard(citation_id)
                continue

            content_size = len(loaded.content)
            if (
                content_size > self._max_image_bytes
                or total_bytes + content_size > self._max_total_bytes
            ):
                _record_decision(
                    decisions,
                    decision_indexes,
                    candidate,
                    VisualEvidenceReason.REJECTED_VISUAL_BUDGET,
                )
                if not text_fallback:
                    usable_set.discard(citation_id)
                continue
            try:
                visual_citation_id = citation_id
                prompt_visual = None
                if candidate.relation_type is not None:
                    visual_rank = len(evidence_items) + 1
                    visual_citation_id = f"cite_{visual_rank}"
                    prompt_visual = PromptEvidence(
                        citation_id=visual_citation_id,
                        rank=visual_rank,
                        index_chunk_id=candidate.visual_unit_id,
                        document_id=evidence.document_id,
                        document_version_id=evidence.document_version_id,
                        excerpt=(
                            f"[{candidate.figure_label} visual evidence]"
                            if candidate.figure_label
                            else f"[{candidate.modality} visual evidence]"
                        ),
                        source_location=candidate.source_location,
                        score=evidence.score,
                        modality=candidate.modality,
                        asset_snapshot={
                            "id": str(asset.id),
                            "media_type": asset.media_type,
                            "checksum_sha256": asset.checksum_sha256,
                            "content_url": asset.content_url,
                            "width": asset.width,
                            "height": asset.height,
                            "visual_unit_id": str(candidate.visual_unit_id),
                            "parent_citation_id": citation_id,
                            "relation_type": candidate.relation_type.value,
                            "selection_reason": candidate.reason_code.value,
                        },
                        matched_representations=(
                            "table_image"
                            if candidate.modality == "table"
                            else "native_image",
                        ),
                        document_display_name=evidence.document_display_name,
                        document_original_filename=(
                            evidence.document_original_filename
                        ),
                    )
                visual = ChatModelVisualContent(
                    citation_ids=(visual_citation_id,),
                    asset_id=asset.id,
                    media_type=asset.media_type,
                    checksum_sha256=asset.checksum_sha256,
                    content=loaded.content,
                    width=asset.width,
                    height=asset.height,
                )
            except ValueError:
                _record_decision(
                    decisions,
                    decision_indexes,
                    candidate,
                    VisualEvidenceReason.REJECTED_ASSET_INTEGRITY,
                )
                if not text_fallback:
                    usable_set.discard(citation_id)
                continue
            visual_content.append(visual)
            if candidate.relation_type is None:
                attached_native_citation_ids.add(citation_id)
            if prompt_visual is not None:
                evidence_items.append(prompt_visual)
            _record_decision(
                decisions,
                decision_indexes,
                candidate,
                candidate.reason_code,
            )
            if visual_citation_id not in usable_set:
                if visual_citation_id not in usable:
                    usable.append(visual_citation_id)
                usable_set.add(visual_citation_id)
            total_bytes += content_size

        for evidence in pack.evidence:
            citation_id = f"cite_{evidence.rank}"
            if (
                citation_id in initially_usable
                and evidence.modality in {"image", "table"}
                and not _has_textual_representation(evidence)
                and citation_id not in attached_native_citation_ids
            ):
                usable_set.discard(citation_id)

        retained = tuple(value for value in usable if value in usable_set)
        assessment = answering.assessment
        if not retained and assessment.coverage is not EvidenceCoverage.NONE:
            assessment = EvidenceAssessment(
                coverage=EvidenceCoverage.NONE,
                usable_citation_ids=(),
                supported_aspects=(),
                missing_aspects=(),
            )
        elif retained != assessment.usable_citation_ids:
            assessment = replace(assessment, usable_citation_ids=retained)

        return ChatPipelineState(
            context=context,
            evidence_pack=pack,
            answering=ChatAnsweringState(
                evidence=EvidenceEnvelope(
                    knowledge_base_id=answering.evidence.knowledge_base_id,
                    index_revision_id=answering.evidence.index_revision_id,
                    items=tuple(evidence_items),
                ),
                assessment=assessment,
                model_calls=answering.model_calls,
                visual_content=tuple(visual_content),
                visual_decisions=tuple(decisions),
                visual_total_bytes=total_bytes,
            ),
            query_context=state.query_context,
            artifacts=state.artifacts,
        )


def _has_textual_representation(evidence) -> bool:
    return bool(evidence.text.strip()) and any(
        representation in {"text", "caption_text", "ocr_text", "table_text"}
        for representation in evidence.matched_representations
    )


def _record_decision(
    decisions,
    indexes,
    candidate,
    reason: VisualEvidenceReason,
) -> None:
    key = (candidate.visual_unit_id, candidate.asset.id)
    value = candidate.decision(reason)
    index = indexes.get(key)
    if index is None:
        indexes[key] = len(decisions)
        decisions.append(value)
    else:
        decisions[index] = value


def _context_error(check: str) -> ChatPipelineExecutionError:
    return ChatPipelineExecutionError(
        ErrorCode.CHAT_CONTEXT_INVALID,
        phase=ChatPipelinePhase.PREPARE_VISUAL_EVIDENCE,
        diagnostic={"check": check},
    )
