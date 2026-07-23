from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4
from uuid import UUID

from rag_kb.answering import (
    AnswerGenerationStep,
    AnswerStructureValidationStep,
    CosineEvidenceAssessmentStep,
)
from rag_kb.domain import (
    ChatExecutionContext,
    ChatModelRequest,
    ChatModelResponse,
    ChatPipelineState,
    ChatRunLease,
    Evidence,
    EvidenceAsset,
    EvidenceCoverage,
    EvidencePack,
    EvidenceScoreKind,
    IndexAssetContent,
    IndexAssetSnapshot,
    RetrievalStrategy,
    RelatedVisualEvidence,
    VisualEvidenceReason,
)
from rag_kb.services import VisualEvidenceAdmissionPolicy, VisualEvidencePreparationStep
from rag_kb.domain import ResourceNotFoundError


class _AssetReader:
    def __init__(self, content: IndexAssetContent) -> None:
        self.content = content
        self.calls = []

    async def read(self, context, asset_id):
        self.calls.append((context, asset_id))
        return self.content


class _MultiAssetReader:
    def __init__(self, contents) -> None:
        self.contents = contents
        self.calls = []

    async def read(self, context, asset_id):
        self.calls.append((context, asset_id))
        value = self.contents[asset_id]
        if isinstance(value, Exception):
            raise value
        return value


class _Model:
    def __init__(self) -> None:
        self.requests: list[ChatModelRequest] = []

    async def complete(self, request: ChatModelRequest) -> ChatModelResponse:
        self.requests.append(request)
        return ChatModelResponse(
            content=(
                '{"outcome":"answered","claims":['
                '{"text":"The diagram shows a workflow.",'
                '"citation_ids":["cite_1"]}],"missing_aspects":[]}'
            ),
            model="vision-model",
            finish_reason="stop",
            provider_request_id="request-1",
            usage={},
        )


class _SequenceModel:
    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)
        self.requests: list[ChatModelRequest] = []

    async def complete(self, request: ChatModelRequest) -> ChatModelResponse:
        self.requests.append(request)
        return ChatModelResponse(
            content=self.contents.pop(0),
            model="vision-model",
            finish_reason="stop",
            provider_request_id=f"request-{len(self.requests)}",
            usage={},
        )


def _context() -> ChatExecutionContext:
    run_id = uuid4()
    workspace_id = uuid4()
    return ChatExecutionContext(
        lease=ChatRunLease(
            run_id=run_id,
            workspace_id=workspace_id,
            claimed_by="worker",
            attempt=1,
            claimed_at=datetime.now(UTC),
        ),
        run_id=run_id,
        workspace_id=workspace_id,
        knowledge_base_id=uuid4(),
        session_id=uuid4(),
        user_message_id=uuid4(),
        assistant_message_id=uuid4(),
        index_revision_id=uuid4(),
        principal_id="principal",
        client_id="client",
        query="What does the diagram show?",
        effective_policy={
            "grounding_policy": "evidence_only",
            "answer_style": "concise",
            "insufficiency_policy": "refuse",
            "citation_required": True,
            "citation_granularity": "claim_level",
            "answer_task": "answer",
            "policy_version": "p1",
        },
        retrieval_strategy={
            "strategy": "exact_vector",
            "top_k": 3,
            "rerank": False,
        },
        model_configuration={"resolved_model": "vision-model"},
        attempt=1,
    )


def _visual_pack(
    context: ChatExecutionContext,
    content: bytes,
    *,
    text: str = "",
    representations: tuple[str, ...] = ("native_image",),
) -> tuple[EvidencePack, IndexAssetContent]:
    asset_id = uuid4()
    indexed_version_id = uuid4()
    document_id = uuid4()
    document_version_id = uuid4()
    checksum = hashlib.sha256(content).hexdigest()
    asset = EvidenceAsset(
        id=asset_id,
        media_type="image/png",
        checksum_sha256=checksum,
        content_url=f"/api/v1/index-assets/{asset_id}/content",
        width=320,
        height=200,
    )
    pack = EvidencePack(
        knowledge_base_id=context.knowledge_base_id,
        index_revision_id=context.index_revision_id,
        strategy=RetrievalStrategy.EXACT_VECTOR,
        evidence=(
            Evidence(
                rank=1,
                index_chunk_id=uuid4(),
                indexed_document_version_id=indexed_version_id,
                document_id=document_id,
                document_version_id=document_version_id,
                index_revision_id=context.index_revision_id,
                ordinal=0,
                text=text,
                source_location={"page_number": 1},
                hierarchy={},
                source_metadata={},
                score=0.016393,
                score_kind=EvidenceScoreKind.RECIPROCAL_RANK_FUSION,
                vector_similarity=0.82,
                modality="image",
                asset=asset,
                matched_representations=representations,
                cross_modal_rank=1,
            ),
        ),
    )
    loaded = IndexAssetContent(
        snapshot=IndexAssetSnapshot(
            id=asset_id,
            workspace_id=context.workspace_id,
            kb_id=context.knowledge_base_id,
            document_id=document_id,
            document_version_id=document_version_id,
            indexed_document_version_id=indexed_version_id,
            storage_uri=(
                f"local-index-asset://{context.workspace_id}/"
                f"{indexed_version_id}/{checksum}"
            ),
            media_type="image/png",
            checksum_sha256=checksum,
            size_bytes=len(content),
        ),
        content=content,
    )
    return pack, loaded


def _related_visual_pack(
    context: ChatExecutionContext,
    *,
    count: int,
    same_group: bool = False,
) -> tuple[EvidencePack, dict[UUID, IndexAssetContent]]:
    indexed_version_id = uuid4()
    document_id = uuid4()
    document_version_id = uuid4()
    related = []
    loaded = {}
    for ordinal in range(count):
        asset_id = UUID(int=100 + ordinal)
        visual_unit_id = UUID(int=200 + ordinal)
        content = f"visual-{ordinal}".encode()
        checksum = hashlib.sha256(content).hexdigest()
        asset = EvidenceAsset(
            id=asset_id,
            media_type="image/png",
            checksum_sha256=checksum,
            content_url=f"/api/v1/index-assets/{asset_id}/content",
            width=320,
            height=200,
        )
        related.append(
            RelatedVisualEvidence(
                visual_unit_id=visual_unit_id,
                asset=asset,
                relation_type=(
                    "explicit_figure_reference" if ordinal == 0 else "caption_of"
                ),
                relation_confidence_micros=900_000,
                relation_provenance="author_reference_v2",
                evidence_group_key="figure:shared" if same_group else f"figure:{ordinal}",
                figure_label=f"Figure {ordinal + 1}",
                text_space_rank=1,
            )
        )
        loaded[asset_id] = IndexAssetContent(
            snapshot=IndexAssetSnapshot(
                id=asset_id,
                workspace_id=context.workspace_id,
                kb_id=context.knowledge_base_id,
                document_id=document_id,
                document_version_id=document_version_id,
                indexed_document_version_id=indexed_version_id,
                storage_uri=(
                    f"local-index-asset://{context.workspace_id}/"
                    f"{indexed_version_id}/{checksum}"
                ),
                media_type="image/png",
                checksum_sha256=checksum,
                size_bytes=len(content),
            ),
            content=content,
        )
    pack = EvidencePack(
        knowledge_base_id=context.knowledge_base_id,
        index_revision_id=context.index_revision_id,
        strategy=RetrievalStrategy.EXACT_VECTOR,
        evidence=(
            Evidence(
                rank=1,
                index_chunk_id=uuid4(),
                indexed_document_version_id=indexed_version_id,
                document_id=document_id,
                document_version_id=document_version_id,
                index_revision_id=context.index_revision_id,
                ordinal=0,
                text="The architecture refers to the figures.",
                source_location={"page_number": 1},
                hierarchy={},
                source_metadata={},
                score=0.016393,
                score_kind=EvidenceScoreKind.RECIPROCAL_RANK_FUSION,
                vector_similarity=0.82,
                matched_representations=("text",),
                text_space_rank=1,
                related_visuals=tuple(related),
            ),
        ),
    )
    return pack, loaded


class MultimodalAnsweringTests(unittest.IsolatedAsyncioTestCase):
    def test_visual_admission_decisions_are_deterministic_and_bounded(self) -> None:
        context = _context()
        pack, _ = _related_visual_pack(context, count=3)
        policy = VisualEvidenceAdmissionPolicy()

        first = policy.decide(pack, ("cite_1",), max_images=2)
        second = policy.decide(pack, ("cite_1",), max_images=2)

        self.assertEqual(first, second)
        self.assertEqual(
            [item.reason_code for item in first],
            [
                VisualEvidenceReason.SELECTED_EXPLICIT_REFERENCE,
                VisualEvidenceReason.SELECTED_STRONG_RELATION,
                VisualEvidenceReason.REJECTED_VISUAL_BUDGET,
            ],
        )
        rejected = policy.decide(pack, (), max_images=2)
        self.assertTrue(
            all(
                item.reason_code
                is VisualEvidenceReason.REJECTED_PARENT_NOT_ADMITTED
                for item in rejected
            )
        )

    async def test_strong_related_visuals_are_ranked_deduplicated_and_read_lazily(
        self,
    ) -> None:
        context = _context()
        pack, loaded = _related_visual_pack(context, count=3)
        state = await CosineEvidenceAssessmentStep(0.35, 0.45, 0.25).run(
            ChatPipelineState(context=context, evidence_pack=pack)
        )
        reader = _MultiAssetReader(loaded)

        result = await VisualEvidencePreparationStep(reader, max_images=2).run(state)

        assert result.answering is not None
        self.assertEqual(len(result.answering.visual_content), 2)
        self.assertEqual(len(reader.calls), 2)
        self.assertEqual(
            result.answering.assessment.usable_citation_ids, ("cite_1",)
        )

    async def test_failed_selected_asset_promotes_next_candidate_without_losing_text(
        self,
    ) -> None:
        context = _context()
        pack, loaded = _related_visual_pack(context, count=3)
        ordered = VisualEvidenceAdmissionPolicy().rank_candidates(pack, ("cite_1",))
        loaded[ordered[0].asset.id] = ResourceNotFoundError("missing")
        state = await CosineEvidenceAssessmentStep(0.35, 0.45, 0.25).run(
            ChatPipelineState(context=context, evidence_pack=pack)
        )
        reader = _MultiAssetReader(loaded)

        result = await VisualEvidencePreparationStep(reader, max_images=2).run(state)

        assert result.answering is not None
        self.assertEqual(len(reader.calls), 3)
        self.assertEqual(len(result.answering.visual_content), 2)
        self.assertEqual(
            result.answering.assessment.usable_citation_ids, ("cite_1",)
        )

    async def test_same_group_visuals_are_descriptor_deduplicated_before_read(self) -> None:
        context = _context()
        pack, loaded = _related_visual_pack(context, count=2, same_group=True)
        state = await CosineEvidenceAssessmentStep(0.35, 0.45, 0.25).run(
            ChatPipelineState(context=context, evidence_pack=pack)
        )
        reader = _MultiAssetReader(loaded)

        result = await VisualEvidencePreparationStep(reader).run(state)

        assert result.answering is not None
        self.assertEqual(len(result.answering.visual_content), 1)
        self.assertEqual(len(reader.calls), 1)

    async def test_native_image_is_admitted_loaded_and_attached_to_generation(self) -> None:
        context = _context()
        pack, loaded = _visual_pack(context, b"validated-image-bytes")
        state = await CosineEvidenceAssessmentStep(0.35, 0.45, 0.25).run(
            ChatPipelineState(context=context, evidence_pack=pack)
        )
        reader = _AssetReader(loaded)
        state = await VisualEvidencePreparationStep(reader).run(state)
        model = _Model()

        result = await AnswerGenerationStep(model).run(state)

        assert result.answering is not None
        self.assertEqual(
            result.answering.assessment.usable_citation_ids, ("cite_1",)
        )
        self.assertEqual(len(result.answering.visual_content), 1)
        request = model.requests[0]
        self.assertEqual(request.messages[1].visual_content[0].citation_ids, ("cite_1",))
        payload = json.loads(request.messages[1].content)
        self.assertTrue(payload["evidence"][0]["visual_input_attached"])
        self.assertEqual(payload["evidence"][0]["untrusted_excerpt"], "[image visual evidence]")
        validated = await AnswerStructureValidationStep(model).run(result)
        assert validated.answering is not None
        assert validated.answering.rendered is not None
        citation = validated.answering.rendered.citations[0]
        self.assertEqual(citation.modality, "image")
        self.assertIsNotNone(citation.asset_snapshot)

    async def test_structure_repair_reuses_identical_visual_content(self) -> None:
        context = _context()
        pack, loaded = _visual_pack(context, b"validated-image-bytes")
        state = await CosineEvidenceAssessmentStep(0.35, 0.45, 0.25).run(
            ChatPipelineState(context=context, evidence_pack=pack)
        )
        state = await VisualEvidencePreparationStep(_AssetReader(loaded)).run(
            state
        )
        model = _SequenceModel(
            "not-json",
            (
                '{"outcome":"answered","claims":['
                '{"text":"The diagram shows a workflow.",'
                '"citation_ids":["cite_1"]}],"missing_aspects":[]}'
            ),
        )

        generated = await AnswerGenerationStep(model).run(state)
        repaired = await AnswerStructureValidationStep(model).run(generated)

        self.assertEqual(len(model.requests), 2)
        self.assertIs(
            model.requests[0].messages[1].visual_content[0],
            model.requests[1].messages[1].visual_content[0],
        )
        assert repaired.answering is not None
        assert repaired.answering.validation is not None
        self.assertTrue(repaired.answering.validation.repair_succeeded)

    async def test_over_budget_native_only_image_is_removed_and_refused(self) -> None:
        context = _context()
        pack, loaded = _visual_pack(context, b"too-large")
        state = await CosineEvidenceAssessmentStep(0.35, 0.45, 0.25).run(
            ChatPipelineState(context=context, evidence_pack=pack)
        )
        state = await VisualEvidencePreparationStep(
            _AssetReader(loaded), max_image_bytes=4, max_total_bytes=4
        ).run(state)
        model = _Model()

        result = await AnswerGenerationStep(model).run(state)

        assert result.answering is not None
        self.assertEqual(result.answering.assessment.coverage, EvidenceCoverage.NONE)
        self.assertEqual(model.requests, [])

    async def test_text_representation_survives_when_image_exceeds_budget(self) -> None:
        context = _context()
        pack, loaded = _visual_pack(
            context,
            b"too-large",
            text="Author caption",
            representations=("caption_text", "native_image"),
        )
        state = await CosineEvidenceAssessmentStep(0.35, 0.45, 0.25).run(
            ChatPipelineState(context=context, evidence_pack=pack)
        )
        state = await VisualEvidencePreparationStep(
            _AssetReader(loaded), max_image_bytes=4, max_total_bytes=4
        ).run(state)

        assert state.answering is not None
        self.assertEqual(state.answering.assessment.usable_citation_ids, ("cite_1",))
        self.assertEqual(state.answering.visual_content, ())

    async def test_asset_snapshot_mismatch_removes_native_only_citation(self) -> None:
        context = _context()
        pack, loaded = _visual_pack(context, b"validated-image-bytes")
        mismatched = IndexAssetContent(
            snapshot=replace(loaded.snapshot, document_id=uuid4()),
            content=loaded.content,
        )
        state = await CosineEvidenceAssessmentStep(0.35, 0.45, 0.25).run(
            ChatPipelineState(context=context, evidence_pack=pack)
        )

        result = await VisualEvidencePreparationStep(_AssetReader(mismatched)).run(
            state
        )

        assert result.answering is not None
        self.assertEqual(result.answering.assessment.coverage, EvidenceCoverage.NONE)
        self.assertEqual(result.answering.visual_content, ())


if __name__ == "__main__":
    unittest.main()
