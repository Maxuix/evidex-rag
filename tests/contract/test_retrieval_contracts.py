from __future__ import annotations

import unittest
from uuid import UUID

from pydantic import ValidationError

from rag_kb.domain import (
    Evidence,
    EvidenceAsset,
    EvidencePack,
    RetrievalDebug,
    RetrievalQueryPlan,
    RetrievalStrategy,
    RelatedVisualEvidence,
)
from rag_kb.schemas import EvidencePackResponse, RetrievalQueryRequest


WORKSPACE = UUID("01900000-0000-7000-8000-000000000901")
KB_ID = UUID("01900000-0000-7000-8000-000000000902")
REVISION_ID = UUID("01900000-0000-7000-8000-000000000903")
CHUNK_ID = UUID("01900000-0000-7000-8000-000000000904")
VISUAL_ID = UUID("01900000-0000-7000-8000-000000000908")
ASSET_ID = UUID("01900000-0000-7000-8000-000000000909")


class RetrievalTransportContractTests(unittest.TestCase):
    def test_request_accepts_known_unsupported_capabilities_for_explicit_service_error(self) -> None:
        request = RetrievalQueryRequest.model_validate(
            {
                "knowledge_base_id": str(KB_ID),
                "query": "  混合检索 ABC-42  ",
                "strategy": "hybrid",
                "rerank": True,
                "include_debug": True,
                "top_k": 5,
            }
        )
        self.assertEqual(request.query, "混合检索 ABC-42")
        self.assertIs(request.strategy, RetrievalStrategy.HYBRID)

    def test_request_rejects_client_owned_scope_and_serving_filters(self) -> None:
        forbidden_fields = (
            "workspace_id",
            "index_revision_id",
            "revision_selector",
            "build_status",
            "serving_status",
            "current_document_version_only",
            "distance_metric",
            "candidate_count",
            "ef_search",
            "iterative_scan",
            "filters",
            "debug",
        )
        for field_name in forbidden_fields:
            with self.subTest(field_name=field_name), self.assertRaises(ValidationError):
                RetrievalQueryRequest.model_validate(
                    {
                        "knowledge_base_id": str(KB_ID),
                        "query": "query",
                        field_name: "client-value",
                    }
                )

    def test_evidence_pack_serializes_stable_identity_and_safe_debug_plan(self) -> None:
        plan = RetrievalQueryPlan(
            workspace_id=WORKSPACE,
            knowledge_base_id=KB_ID,
            strategy=RetrievalStrategy.EXACT_VECTOR,
            top_k=5,
        )
        evidence = Evidence(
            rank=1,
            index_chunk_id=CHUNK_ID,
            indexed_document_version_id=UUID(
                "01900000-0000-7000-8000-000000000905"
            ),
            document_id=UUID("01900000-0000-7000-8000-000000000906"),
            document_version_id=UUID("01900000-0000-7000-8000-000000000907"),
            index_revision_id=REVISION_ID,
            ordinal=3,
            text="quoted evidence",
            source_location={"line_start": 4, "line_end": 7},
            hierarchy={"section": "S1"},
            source_metadata={"filename": "guide.md"},
            score=0.75,
            related_visuals=(
                RelatedVisualEvidence(
                    visual_unit_id=VISUAL_ID,
                    asset=EvidenceAsset(
                        id=ASSET_ID,
                        media_type="image/png",
                        checksum_sha256="a" * 64,
                        content_url=f"/api/v1/index-assets/{ASSET_ID}/content",
                        width=320,
                        height=200,
                    ),
                    relation_type="explicit_figure_reference",
                    relation_confidence_micros=950_000,
                    relation_provenance="author_reference_v2",
                    evidence_group_key="figure:7",
                    figure_label="Figure 7",
                    parent_chunk_id=CHUNK_ID,
                    source_location={"page": 2},
                    text_space_rank=1,
                ),
            ),
        )
        response = EvidencePackResponse.from_domain(
            EvidencePack(
                knowledge_base_id=KB_ID,
                index_revision_id=REVISION_ID,
                strategy=RetrievalStrategy.EXACT_VECTOR,
                evidence=(evidence,),
                debug=RetrievalDebug(
                    plan,
                    REVISION_ID,
                    1,
                    text_candidate_count=3,
                    cross_modal_candidate_count=2,
                    hydrated_relation_count=1,
                    evidence_group_count=1,
                ),
            )
        )
        body = response.model_dump(mode="json")

        self.assertEqual(body["evidence"][0]["index_chunk_id"], str(CHUNK_ID))
        self.assertEqual(
            body["evidence"][0]["document_version_id"],
            "01900000-0000-7000-8000-000000000907",
        )
        self.assertEqual(body["debug"]["query_plan"]["revision_selector"], "active")
        self.assertTrue(body["debug"]["query_plan"]["current_document_version_only"])
        self.assertEqual(body["debug"]["query_plan"]["build_status"], "ready")
        self.assertEqual(body["debug"]["query_plan"]["serving_status"], "serving")
        self.assertNotIn("query", body["debug"]["query_plan"])
        self.assertEqual(body["debug"]["hydrated_relation_count"], 1)
        related = body["evidence"][0]["related_visuals"][0]
        self.assertEqual(related["visual_unit_id"], str(VISUAL_ID))
        self.assertEqual(related["asset"]["id"], str(ASSET_ID))
        self.assertNotIn("storage_uri", related["asset"])


if __name__ == "__main__":
    unittest.main()
