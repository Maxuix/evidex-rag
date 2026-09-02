from __future__ import annotations

import unittest
from dataclasses import replace
from uuid import UUID, uuid4

from rag_kb.domain import (
    ChatAgentTraceEvent,
    Evidence,
    EvidenceScoreKind,
    LexicalManifestStatus,
    SERVING_DOCUMENT_LIST_LIMIT,
    SERVING_DOCUMENT_OUTLINE_LIMIT,
    ServingDocumentEntry,
    ServingDocumentList,
)
from rag_kb.retrieval.eligibility import EvidenceEligibilityPolicy


REVISION_ID = UUID("01900000-0000-7000-8000-000000000903")
CHUNK_ID = UUID("01900000-0000-7000-8000-000000000904")
TARGET_ID = UUID("01900000-0000-7000-8000-000000000905")
DOCUMENT_ID = UUID("01900000-0000-7000-8000-000000000906")
VERSION_ID = UUID("01900000-0000-7000-8000-000000000907")


def _evidence(**overrides: object) -> Evidence:
    values: dict[str, object] = {
        "rank": 1,
        "index_chunk_id": CHUNK_ID,
        "indexed_document_version_id": TARGET_ID,
        "document_id": DOCUMENT_ID,
        "document_version_id": VERSION_ID,
        "index_revision_id": REVISION_ID,
        "ordinal": 0,
        "text": "lexical hit",
        "source_location": {},
        "hierarchy": {},
        "source_metadata": {},
        "score": 1.0,
        "score_kind": EvidenceScoreKind.COSINE_SIMILARITY,
        "vector_similarity": 0.9,
    }
    values.update(overrides)
    return Evidence(**values)  # type: ignore[arg-type]


def _lexical_evidence(*, rank: int = 2) -> Evidence:
    return _evidence(
        score=1.0 / rank,
        score_kind=EvidenceScoreKind.LEXICAL,
        vector_similarity=None,
        lexical_rank=rank,
    )


def _policy() -> EvidenceEligibilityPolicy:
    return EvidenceEligibilityPolicy(
        min_cosine_similarity=0.35,
        min_rerank_score=0.45,
        cross_modal_min_cosine_similarity=0.25,
    )


class LexicalEvidenceTests(unittest.TestCase):
    def test_lexical_evidence_requires_rank_and_reciprocal_score(self) -> None:
        item = _lexical_evidence(rank=3)
        self.assertIs(item.score_kind, EvidenceScoreKind.LEXICAL)
        self.assertEqual(item.lexical_rank, 3)
        self.assertEqual(item.score, 1.0 / 3)

    def test_lexical_evidence_rejects_missing_rank(self) -> None:
        with self.assertRaises(ValueError):
            _evidence(
                score=1.0,
                score_kind=EvidenceScoreKind.LEXICAL,
                vector_similarity=None,
            )

    def test_lexical_evidence_rejects_score_not_matching_rank(self) -> None:
        with self.assertRaises(ValueError):
            _evidence(
                score=0.5,
                score_kind=EvidenceScoreKind.LEXICAL,
                vector_similarity=None,
                lexical_rank=3,
            )

    def test_lexical_evidence_rejects_vector_similarity(self) -> None:
        with self.assertRaises(ValueError):
            _evidence(
                score=1.0,
                score_kind=EvidenceScoreKind.LEXICAL,
                vector_similarity=0.9,
                lexical_rank=1,
            )


class ServingDocumentDomainTests(unittest.TestCase):
    def test_serving_entry_strips_and_bounds_outline(self) -> None:
        entry = ServingDocumentEntry(
            document_id=DOCUMENT_ID,
            document_version_id=VERSION_ID,
            indexed_document_version_id=TARGET_ID,
            display_name="  Report  ",
            original_filename=" report.pdf ",
            version_number=1,
            chunk_count=0,
            outline=("  Intro  ", "Body"),
        )
        self.assertEqual(entry.display_name, "Report")
        self.assertEqual(entry.original_filename, "report.pdf")
        self.assertEqual(entry.outline, ("Intro", "Body"))

    def test_serving_entry_rejects_duplicate_or_oversized_outline(self) -> None:
        with self.assertRaises(ValueError):
            ServingDocumentEntry(
                document_id=DOCUMENT_ID,
                document_version_id=VERSION_ID,
                indexed_document_version_id=TARGET_ID,
                display_name="Report",
                original_filename="report.pdf",
                version_number=1,
                chunk_count=1,
                outline=("Intro", "Intro"),
            )
        with self.assertRaises(ValueError):
            ServingDocumentEntry(
                document_id=DOCUMENT_ID,
                document_version_id=VERSION_ID,
                indexed_document_version_id=TARGET_ID,
                display_name="Report",
                original_filename="report.pdf",
                version_number=1,
                chunk_count=1,
                outline=tuple(f"title-{index}" for index in range(SERVING_DOCUMENT_OUTLINE_LIMIT + 1)),
            )
        with self.assertRaises(ValueError):
            ServingDocumentEntry(
                document_id=DOCUMENT_ID,
                document_version_id=VERSION_ID,
                indexed_document_version_id=TARGET_ID,
                display_name="Report",
                original_filename="report.pdf",
                version_number=1,
                chunk_count=1,
                outline=("x" * 257,),
            )

    def test_serving_list_rejects_overflow_and_duplicate_documents(self) -> None:
        entries = tuple(
            ServingDocumentEntry(
                document_id=uuid4(),
                document_version_id=uuid4(),
                indexed_document_version_id=uuid4(),
                display_name=f"Doc {index}",
                original_filename=f"doc-{index}.pdf",
                version_number=1,
                chunk_count=1,
            )
            for index in range(SERVING_DOCUMENT_LIST_LIMIT)
        )
        listed = ServingDocumentList(
            resolved_active_revision_id=REVISION_ID,
            entries=entries,
        )
        self.assertEqual(len(listed.entries), SERVING_DOCUMENT_LIST_LIMIT)
        extra = ServingDocumentEntry(
            document_id=uuid4(),
            document_version_id=uuid4(),
            indexed_document_version_id=uuid4(),
            display_name="Overflow",
            original_filename="overflow.pdf",
            version_number=1,
            chunk_count=1,
        )
        with self.assertRaises(ValueError):
            ServingDocumentList(
                resolved_active_revision_id=REVISION_ID,
                entries=entries + (extra,),
            )
        with self.assertRaises(ValueError):
            ServingDocumentList(
                resolved_active_revision_id=REVISION_ID,
                entries=(entries[0], replace(entries[0], document_version_id=uuid4())),
            )

    def test_lexical_manifest_status_complete_requires_full_coverage(self) -> None:
        empty = LexicalManifestStatus(
            resolved_active_revision_id=REVISION_ID,
            serving_target_count=0,
            manifested_target_count=0,
        )
        self.assertFalse(empty.complete)
        complete = LexicalManifestStatus(
            resolved_active_revision_id=REVISION_ID,
            serving_target_count=2,
            manifested_target_count=2,
        )
        self.assertTrue(complete.complete)
        partial = LexicalManifestStatus(
            resolved_active_revision_id=REVISION_ID,
            serving_target_count=2,
            manifested_target_count=1,
        )
        self.assertFalse(partial.complete)
        with self.assertRaises(ValueError):
            LexicalManifestStatus(
                resolved_active_revision_id=REVISION_ID,
                serving_target_count=1,
                manifested_target_count=2,
            )


class EligibilityUsableTests(unittest.TestCase):
    def test_lexical_rank_admits_and_adjacency_still_rejects(self) -> None:
        policy = _policy()
        self.assertTrue(policy.usable(_lexical_evidence()))
        neighbor = _evidence(
            score=0.0,
            score_kind=EvidenceScoreKind.ADJACENCY,
            vector_similarity=None,
            adjacency_anchor_index_chunk_id=CHUNK_ID,
            adjacency_offset=1,
            text="neighbor",
        )
        self.assertFalse(policy.usable(neighbor))

    def test_graph_path_and_rrf_do_not_regress(self) -> None:
        policy = _policy()
        graph = _evidence(
            score=1.0,
            score_kind=EvidenceScoreKind.GRAPH_PATH,
            vector_similarity=None,
            graph_path_id="path",
            graph_anchor_index_chunk_id=CHUNK_ID,
            graph_hop_count=1,
            graph_path_rank=1,
        )
        self.assertTrue(policy.usable(graph))
        rrf_lexical = _evidence(
            score=0.1,
            score_kind=EvidenceScoreKind.RECIPROCAL_RANK_FUSION,
            vector_similarity=None,
            lexical_rank=2,
        )
        self.assertTrue(policy.usable(rrf_lexical))
        rrf_text = _evidence(
            score=0.1,
            score_kind=EvidenceScoreKind.RECIPROCAL_RANK_FUSION,
            vector_similarity=0.4,
            text_space_rank=1,
        )
        self.assertTrue(policy.usable(rrf_text))
        rrf_below = _evidence(
            score=0.1,
            score_kind=EvidenceScoreKind.RECIPROCAL_RANK_FUSION,
            vector_similarity=0.2,
            text_space_rank=1,
        )
        self.assertFalse(policy.usable(rrf_below))


class ChatAgentLaneRuleTests(unittest.TestCase):
    def test_new_lanes_forbid_graph_route_fields(self) -> None:
        for lane in ("semantic", "keyword", "chunk_context", "document_list"):
            with self.subTest(lane=lane):
                event = ChatAgentTraceEvent(
                    tool="semantic_search",
                    status="ok",
                    tool_call_id=f"{lane}-1",
                    retrieval_lane=lane,
                    route_result_code="not_requested",
                )
                self.assertEqual(event.retrieval_lane, lane)
                with self.assertRaises(ValueError):
                    ChatAgentTraceEvent(
                        tool="semantic_search",
                        status="ok",
                        tool_call_id=f"{lane}-bad",
                        retrieval_lane=lane,
                        route_result_code="admitted",
                    )
                with self.assertRaises(ValueError):
                    ChatAgentTraceEvent(
                        tool="semantic_search",
                        status="ok",
                        tool_call_id=f"{lane}-count",
                        retrieval_lane=lane,
                        route_result_code="not_requested",
                        call_index=1,
                    )

    def test_graph_lane_rules_do_not_regress(self) -> None:
        event = ChatAgentTraceEvent(
            tool="search_graph_relations",
            status="ok",
            tool_call_id="graph-1",
            refs=("ev_1",),
            count=1,
            retrieval_lane="graph_relations",
            route_reason_code="direct_relation",
            route_result_code="admitted",
            new_evidence_count=1,
            call_index=1,
            invocation_source="agent",
            duration_ms=12,
        )
        self.assertEqual(event.new_evidence_count, 1)
        with self.assertRaises(ValueError):
            ChatAgentTraceEvent(
                tool="search_graph_relations",
                status="ok",
                tool_call_id="graph-incomplete",
                retrieval_lane="graph_relations",
                route_reason_code="direct_relation",
                route_result_code="admitted",
                new_evidence_count=1,
            )
