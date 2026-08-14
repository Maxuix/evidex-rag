from __future__ import annotations

import unittest
from dataclasses import replace
from uuid import UUID

from pydantic import ValidationError

from rag_kb.domain import (
    AdjacentChunkAnchor,
    AdjacentChunkQuery,
    Evidence,
    EvidenceAsset,
    EvidencePack,
    EvidenceScoreKind,
    GraphDebug,
    GraphEvidenceBundle,
    GraphPathCandidate,
    GraphPathHop,
    RerankMode,
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
    def test_adjacency_contract_is_fixed_to_two_anchors_and_explicit_score(self) -> None:
        anchor = AdjacentChunkAnchor(
            index_chunk_id=CHUNK_ID,
            indexed_document_version_id=UUID(
                "01900000-0000-7000-8000-000000000905"
            ),
            ordinal=3,
        )
        query = AdjacentChunkQuery(
            workspace_id=WORKSPACE,
            knowledge_base_id=KB_ID,
            index_revision_id=REVISION_ID,
            anchors=(anchor,),
        )
        self.assertEqual(query.anchors, (anchor,))
        with self.assertRaises(ValueError):
            replace(query, anchors=(anchor, anchor))

        neighbor = Evidence(
            rank=1,
            index_chunk_id=UUID("01900000-0000-7000-8000-000000000910"),
            indexed_document_version_id=anchor.indexed_document_version_id,
            document_id=UUID("01900000-0000-7000-8000-000000000906"),
            document_version_id=UUID(
                "01900000-0000-7000-8000-000000000907"
            ),
            index_revision_id=REVISION_ID,
            ordinal=4,
            text="continued evidence",
            source_location={"line_start": 8},
            hierarchy={},
            source_metadata={},
            score=0.0,
            score_kind=EvidenceScoreKind.ADJACENCY,
            adjacency_anchor_index_chunk_id=CHUNK_ID,
            adjacency_offset=1,
        )
        self.assertIs(neighbor.score_kind, EvidenceScoreKind.ADJACENCY)
        with self.assertRaises(ValueError):
            replace(neighbor, vector_similarity=0.9)
        with self.assertRaises(ValueError):
            replace(neighbor, lexical_score=0.5)
        with self.assertRaises(ValueError):
            replace(neighbor, modality="image")

    def test_request_accepts_hybrid_strategy(self) -> None:
        request = RetrievalQueryRequest.model_validate(
            {
                "knowledge_base_id": str(KB_ID),
                "query": "  混合检索 ABC-42  ",
                "strategy": "hybrid",
                "rerank_mode": "classic",
                "include_debug": True,
                "top_k": 5,
            }
        )
        self.assertEqual(request.query, "混合检索 ABC-42")
        self.assertIs(request.strategy, RetrievalStrategy.HYBRID)
        self.assertIs(request.rerank_mode, RerankMode.CLASSIC)

    def test_request_rejects_unimplemented_strategies(self) -> None:
        for strategy in ("ann_vector", "lexical"):
            with self.subTest(strategy=strategy), self.assertRaises(
                ValidationError
            ):
                RetrievalQueryRequest.model_validate(
                    {
                        "knowledge_base_id": str(KB_ID),
                        "query": "query",
                        "strategy": strategy,
                    }
                )

    def test_request_rejects_invalid_rerank_combinations(self) -> None:
        for changes in (
            {
                "strategy": "hybrid",
                "rerank_mode": "none",
            },
            {
                "top_k": 21,
                "rerank_mode": "local_minilm_v1",
            },
        ):
            with self.subTest(changes=changes), self.assertRaises(
                ValidationError
            ):
                RetrievalQueryRequest.model_validate(
                    {
                        "knowledge_base_id": str(KB_ID),
                        "query": "query",
                        **changes,
                    }
                )

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
            lexical_rank=2,
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
                    lexical_rank=2,
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
                    lexical_candidate_count=4,
                    cross_modal_candidate_count=2,
                    lexical_analyzer_version="lexical_simple_cjk_bigram_v1",
                    lexical_manifest_target_count=1,
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
        self.assertEqual(
            set(body["debug"]["query_plan"]),
            {
                "workspace_id",
                "knowledge_base_id",
                "strategy",
                "top_k",
                "distance_metric",
                "candidate_count",
                "rerank_mode",
            },
        )
        self.assertNotIn("query", body["debug"]["query_plan"])
        self.assertEqual(body["debug"]["hydrated_relation_count"], 1)
        self.assertEqual(body["debug"]["lexical_candidate_count"], 4)
        self.assertEqual(
            body["debug"]["lexical_analyzer_version"],
            "lexical_simple_cjk_bigram_v1",
        )
        self.assertEqual(body["evidence"][0]["lexical_rank"], 2)
        related = body["evidence"][0]["related_visuals"][0]
        self.assertEqual(related["visual_unit_id"], str(VISUAL_ID))
        self.assertEqual(related["asset"]["id"], str(ASSET_ID))
        self.assertEqual(related["lexical_rank"], 2)
        self.assertNotIn("storage_uri", related["asset"])

    def test_graph_debug_and_path_evidence_serialize_safe_wire_fields(self) -> None:
        hop = GraphPathHop(
            subject_entity_key="a" * 64,
            object_entity_key="b" * 64,
            predicate="released",
            normalized_predicate="released",
            relation_id=UUID("01900000-0000-7000-8000-000000000911"),
            source_chunk_id=CHUNK_ID,
            source_index_revision_id=REVISION_ID,
            source_location={"line_start": 3},
            support_count=2,
        )
        path = GraphPathCandidate(
            path_id="graph-path-1",
            entry_entity_key="a" * 64,
            hops=(hop,),
            anchor_chunk_id=CHUNK_ID,
            rank=1,
            seed_entry=True,
        )
        graph_debug = GraphDebug(
            dense_seed_count=3,
            lexical_seed_count=2,
            fused_seed_count=2,
            query_entity_count=1,
            one_hop_path_count=1,
            bundle_count=1,
            paths=(path,),
            bundles=(GraphEvidenceBundle(path, (CHUNK_ID,)),),
        )
        plan = RetrievalQueryPlan(
            workspace_id=WORKSPACE,
            knowledge_base_id=KB_ID,
            strategy=RetrievalStrategy.HYBRID,
            top_k=4,
            candidate_count=4,
            rerank_mode=RerankMode.CLASSIC,
        )
        evidence = Evidence(
            rank=1,
            index_chunk_id=CHUNK_ID,
            indexed_document_version_id=UUID(
                "01900000-0000-7000-8000-000000000905"
            ),
            document_id=UUID("01900000-0000-7000-8000-000000000906"),
            document_version_id=UUID(
                "01900000-0000-7000-8000-000000000907"
            ),
            index_revision_id=REVISION_ID,
            ordinal=3,
            text="graph evidence",
            source_location={"line_start": 3},
            hierarchy={},
            source_metadata={},
            score=1.0,
            score_kind=EvidenceScoreKind.GRAPH_PATH,
            graph_path_id=path.path_id,
            graph_anchor_index_chunk_id=CHUNK_ID,
            graph_hop_count=1,
            graph_path_rank=1,
        )
        body = EvidencePackResponse.from_domain(
            EvidencePack(
                knowledge_base_id=KB_ID,
                index_revision_id=REVISION_ID,
                strategy=RetrievalStrategy.HYBRID,
                evidence=(evidence,),
                debug=RetrievalDebug(
                    plan,
                    REVISION_ID,
                    1,
                    graph=graph_debug,
                ),
            )
        ).model_dump(mode="json")

        self.assertEqual(body["evidence"][0]["score_kind"], "graph_path")
        self.assertEqual(body["evidence"][0]["graph_path_id"], path.path_id)
        self.assertEqual(body["debug"]["graph"]["bundle_count"], 1)
        self.assertEqual(body["debug"]["graph"]["paths"][0]["support_counts"], [2])
        self.assertNotIn("provider_payload", body["debug"]["graph"])


if __name__ == "__main__":
    unittest.main()
