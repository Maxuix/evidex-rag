"""Prepare authorized, bounded visual evidence for the final chat model."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from rag_kb.auth import AuthContext
from rag_kb.domain import (
    ChatAnsweringState,
    ChatModelVisualContent,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatPipelineState,
    ErrorCode,
    EvidenceEnvelope,
    FileStoreError,
    PromptEvidence,
    ResourceNotFoundError,
    VisualEvidenceReason,
)
from rag_kb.services.visual_admission import VisualEvidenceAdmissionPolicy

if TYPE_CHECKING:
    from rag_kb.services.assets import IndexAssetService


DEFAULT_CHAT_MAX_VISUAL_IMAGES = 2
DEFAULT_CHAT_MAX_VISUAL_IMAGE_BYTES = 5 * 1024 * 1024
DEFAULT_CHAT_MAX_VISUAL_TOTAL_BYTES = 12 * 1024 * 1024
DEFAULT_CHAT_MAX_VISUAL_PIXELS = 16_000_000
DEFAULT_CHAT_VISUAL_MEDIA_PROFILE = "jpeg_png_webp_v1"


class VisualEvidencePreparationStep:
    """Attach only admitted and integrity-checked images to model requests."""

    def __init__(
        self,
        asset_reader: IndexAssetService | None,
        *,
        max_images: int = DEFAULT_CHAT_MAX_VISUAL_IMAGES,
        max_image_bytes: int = DEFAULT_CHAT_MAX_VISUAL_IMAGE_BYTES,
        max_total_bytes: int = DEFAULT_CHAT_MAX_VISUAL_TOTAL_BYTES,
        max_pixels: int = DEFAULT_CHAT_MAX_VISUAL_PIXELS,
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

    async def run(
        self,
        state: ChatPipelineState,
        *,
        previous_visuals: tuple[ChatModelVisualContent, ...] = (),
    ) -> ChatPipelineState:
        context = state.context
        pack = state.evidence_pack
        answering = state.answering
        if (
            context is None
            or pack is None
            or answering is None
            or answering.validated is not None
            or answering.visual_content
        ):
            raise _context_error("visual_preparation_state")

        usable = list(answering.usable_citation_ids)
        initially_usable = frozenset(usable)
        usable_set = set(usable)
        attached_native_citation_ids: set[str] = set()
        visual_content: list[ChatModelVisualContent] = []
        evidence_items = list(answering.evidence.items)
        total_bytes = 0
        previous_asset_ids = {item.asset_id for item in previous_visuals}
        previous_total_bytes = sum(len(item.content) for item in previous_visuals)
        vision_enabled, max_images, max_image_bytes, max_total_bytes, max_pixels = (
            self._frozen_limits(context.model_configuration)
        )
        remaining_images = max(0, max_images - len(previous_visuals))
        remaining_total_bytes = max(0, max_total_bytes - previous_total_bytes)
        auth = AuthContext(
            principal_id=context.principal_id,
            client_id=context.client_id,
            workspace_id=context.workspace_id,
        )

        candidates = self._admission_policy.rank_candidates(
            pack, answering.usable_citation_ids
        )
        decisions = list(
            self._admission_policy.decide(
                pack,
                answering.usable_citation_ids,
                max_images=max_images,
            )
        )
        decision_indexes = {
            (item.visual_unit_id, item.asset_id): index
            for index, item in enumerate(decisions)
        }
        for candidate in candidates:
            if candidate.asset.id in previous_asset_ids:
                continue
            if not vision_enabled or len(visual_content) >= remaining_images:
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
                or asset.width * asset.height > max_pixels
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
                content_size > max_image_bytes
                or total_bytes + content_size > remaining_total_bytes
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
                asset_snapshot = {
                    "id": str(asset.id),
                    "media_type": asset.media_type,
                    "checksum_sha256": asset.checksum_sha256,
                    "content_url": asset.content_url,
                    "width": asset.width,
                    "height": asset.height,
                    "visual_unit_id": str(candidate.visual_unit_id),
                    "parent_citation_id": citation_id,
                    "selection_reason": candidate.reason_code.value,
                }
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
                            **asset_snapshot,
                            "relation_type": candidate.relation_type.value,
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
                evidence_items = [
                    replace(item, asset_snapshot=asset_snapshot)
                    if item.citation_id == citation_id
                    else item
                    for item in evidence_items
                ]
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

        attached_or_previous = previous_asset_ids | {
            item.asset_id for item in visual_content
        }
        for candidate in candidates:
            if candidate.asset.id not in attached_or_previous:
                _record_decision(
                    decisions,
                    decision_indexes,
                    candidate,
                    VisualEvidenceReason.REJECTED_VISUAL_BUDGET,
                )

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
        return ChatPipelineState(
            context=context,
            evidence_pack=pack,
            answering=ChatAnsweringState(
                evidence=EvidenceEnvelope(
                    knowledge_base_id=answering.evidence.knowledge_base_id,
                    index_revision_id=answering.evidence.index_revision_id,
                    items=tuple(evidence_items),
                ),
                usable_citation_ids=retained,
                model_calls=answering.model_calls,
                visual_content=tuple(visual_content),
                visual_decisions=tuple(decisions),
                visual_total_bytes=total_bytes,
            ),
            query_context=state.query_context,
            artifacts=state.artifacts,
        )

    def _frozen_limits(
        self,
        configuration,
    ) -> tuple[bool, int, int, int, int]:
        vision_enabled = configuration.get("vision_enabled", True)
        if not isinstance(vision_enabled, bool):
            raise _context_error("visual_configuration")

        def bounded(name: str, hard_limit: int) -> int:
            value = configuration.get(name, hard_limit)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise _context_error("visual_configuration")
            return min(value, hard_limit)

        max_images = bounded("max_visual_images", self._max_images)
        max_image_bytes = bounded(
            "max_visual_image_bytes",
            self._max_image_bytes,
        )
        max_total_bytes = bounded(
            "max_visual_total_bytes",
            self._max_total_bytes,
        )
        max_pixels = bounded("max_visual_pixels", self._max_pixels)
        if max_total_bytes < max_image_bytes:
            raise _context_error("visual_configuration")
        return (
            vision_enabled,
            max_images,
            max_image_bytes,
            max_total_bytes,
            max_pixels,
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
