from __future__ import annotations

import hashlib
import math
import unittest
from uuid import uuid4

from rag_kb.domain import (
    ChunkAssetRelationDraft,
    ChunkAssetRelationProvenance,
    ChunkAssetRelationType,
    CompositeChunkDraft,
    VisualEvidenceDecision,
    VisualEvidenceReason,
    quantize_score_micros,
)


class CompositeEvidenceDomainTests(unittest.TestCase):
    def test_composite_chunk_freezes_metadata_and_requires_embedding_hash(self) -> None:
        content = "Body text"
        embedding_text = "[body]\nBody text\n[figure_label]\nFigure 7"
        metadata = {"assembly": {"version": "v2"}}

        chunk = CompositeChunkDraft(
            unit_key="chunk-1",
            ordinal=0,
            content=content,
            embedding_text=embedding_text,
            embedding_text_hash=hashlib.sha256(embedding_text.encode()).hexdigest(),
            token_count=3,
            source_location={"page_number": 1},
            hierarchy={},
            processing_metadata=metadata,
            evidence_group_key="group-1",
        )

        metadata["assembly"]["version"] = "mutated"
        self.assertEqual(chunk.processing_metadata["assembly"]["version"], "v2")
        with self.assertRaises(ValueError):
            CompositeChunkDraft(
                unit_key="chunk-1",
                ordinal=0,
                content=content,
                embedding_text=embedding_text,
                embedding_text_hash="not-a-hash",
                token_count=3,
                source_location={},
                hierarchy={},
                processing_metadata={},
                evidence_group_key="group-1",
            )

    def test_relation_strength_is_closed_and_confidence_is_integer_micros(self) -> None:
        self.assertTrue(ChunkAssetRelationType.CAPTION_OF.is_strong)
        self.assertFalse(ChunkAssetRelationType.SAME_PAGE.is_strong)
        relation = ChunkAssetRelationDraft(
            chunk_unit_key="chunk-1",
            visual_unit_key="visual-1",
            asset_key="asset-1",
            relation_type=ChunkAssetRelationType.EXPLICIT_FIGURE_REFERENCE,
            confidence_micros=1_000_000,
            ordinal=0,
            provenance=ChunkAssetRelationProvenance.AUTHOR_REFERENCE_V2,
            evidence_group_key="figure-7",
            figure_label=" Figure 7 ",
        )
        self.assertEqual(relation.figure_label, "Figure 7")
        with self.assertRaises(ValueError):
            ChunkAssetRelationDraft(
                chunk_unit_key="chunk-1",
                visual_unit_key="visual-1",
                asset_key="asset-1",
                relation_type=ChunkAssetRelationType.SAME_PAGE,
                confidence_micros=1_000_001,
                ordinal=0,
                provenance=ChunkAssetRelationProvenance.PAGE_IDENTITY_V2,
                evidence_group_key="page-1",
            )

    def test_visual_decision_selection_is_derived_from_closed_reason_code(self) -> None:
        selected = VisualEvidenceDecision(
            visual_unit_id=uuid4(),
            asset_id=uuid4(),
            reason_code=VisualEvidenceReason.SELECTED_STRONG_RELATION,
            parent_text_citation_ids=("cite_1",),
            relation_type=ChunkAssetRelationType.CAPTION_OF,
            text_rank=1,
            priority_micros=2_000_000,
        )
        rejected = VisualEvidenceDecision(
            visual_unit_id=uuid4(),
            asset_id=uuid4(),
            reason_code=VisualEvidenceReason.REJECTED_WEAK_RELATION,
        )
        self.assertTrue(selected.selected)
        self.assertFalse(rejected.selected)
        with self.assertRaises(ValueError):
            VisualEvidenceDecision(
                visual_unit_id=uuid4(),
                asset_id=uuid4(),
                reason_code=VisualEvidenceReason.SELECTED_IMAGE_ONLY,
                cross_modal_rank=0,
            )

    def test_score_quantization_is_repeatable_and_rejects_non_finite_values(self) -> None:
        self.assertEqual(quantize_score_micros(1 / 61), 16_393)
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                quantize_score_micros(value)


if __name__ == "__main__":
    unittest.main()
