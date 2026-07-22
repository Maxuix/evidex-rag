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
    FileStoreError,
    IndexAssetContent,
    ResourceNotFoundError,
)


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
        max_images: int = 4,
        max_image_bytes: int = 5 * 1024 * 1024,
        max_total_bytes: int = 12 * 1024 * 1024,
        max_pixels: int = 16_000_000,
    ) -> None:
        if (
            max_images < 1
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
        usable_set = set(usable)
        visual_content: list[ChatModelVisualContent] = []
        visual_by_asset: dict[UUID, int] = {}
        total_bytes = 0
        auth = AuthContext(
            principal_id=context.principal_id,
            client_id=context.client_id,
            workspace_id=context.workspace_id,
        )

        for evidence, prompt_item in zip(
            pack.evidence, answering.evidence.items, strict=True
        ):
            citation_id = prompt_item.citation_id
            if citation_id not in usable_set:
                continue
            if evidence.modality not in {"image", "table"}:
                continue

            text_fallback = _has_textual_representation(evidence)
            if evidence.asset is None:
                if not text_fallback:
                    usable_set.remove(citation_id)
                continue
            existing = visual_by_asset.get(evidence.asset.id)
            if existing is not None:
                previous = visual_content[existing]
                visual_content[existing] = replace(
                    previous,
                    citation_ids=previous.citation_ids + (citation_id,),
                )
                continue

            if (
                self._asset_reader is None
                or len(visual_content) >= self._max_images
                or evidence.asset.media_type
                not in {"image/jpeg", "image/png", "image/webp"}
                or evidence.asset.width is None
                or evidence.asset.height is None
                or evidence.asset.width * evidence.asset.height > self._max_pixels
            ):
                if not text_fallback:
                    usable_set.remove(citation_id)
                continue

            try:
                loaded = await self._asset_reader.read(auth, evidence.asset.id)
            except (FileStoreError, ResourceNotFoundError):
                if not text_fallback:
                    usable_set.remove(citation_id)
                continue
            snapshot = loaded.snapshot
            if (
                snapshot.id != evidence.asset.id
                or snapshot.workspace_id != context.workspace_id
                or snapshot.kb_id != context.knowledge_base_id
                or snapshot.document_id != evidence.document_id
                or snapshot.document_version_id != evidence.document_version_id
                or snapshot.indexed_document_version_id
                != evidence.indexed_document_version_id
                or snapshot.media_type != evidence.asset.media_type
                or snapshot.checksum_sha256 != evidence.asset.checksum_sha256
            ):
                if not text_fallback:
                    usable_set.remove(citation_id)
                continue

            content_size = len(loaded.content)
            if (
                content_size > self._max_image_bytes
                or total_bytes + content_size > self._max_total_bytes
            ):
                if not text_fallback:
                    usable_set.remove(citation_id)
                continue
            try:
                visual = ChatModelVisualContent(
                    citation_ids=(citation_id,),
                    asset_id=evidence.asset.id,
                    media_type=evidence.asset.media_type,
                    checksum_sha256=evidence.asset.checksum_sha256,
                    content=loaded.content,
                    width=evidence.asset.width,
                    height=evidence.asset.height,
                )
            except ValueError:
                if not text_fallback:
                    usable_set.remove(citation_id)
                continue
            visual_by_asset[evidence.asset.id] = len(visual_content)
            visual_content.append(visual)
            total_bytes += content_size

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
                evidence=answering.evidence,
                assessment=assessment,
                model_calls=answering.model_calls,
                visual_content=tuple(visual_content),
            ),
            query_context=state.query_context,
            artifacts=state.artifacts,
        )


def _has_textual_representation(evidence) -> bool:
    return bool(evidence.text.strip()) and any(
        representation in {"text", "caption_text", "ocr_text", "table_text"}
        for representation in evidence.matched_representations
    )


def _context_error(check: str) -> ChatPipelineExecutionError:
    return ChatPipelineExecutionError(
        ErrorCode.CHAT_CONTEXT_INVALID,
        phase=ChatPipelinePhase.PREPARE_VISUAL_EVIDENCE,
        diagnostic={"check": check},
    )
