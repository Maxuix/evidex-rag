from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

from rag_kb.domain import (
    AnswerClaim,
    AnswerConflict,
    AnswerConflictAdjudication,
    AnswerConflictType,
    AnswerDraftSource,
    AnswerOutcome,
    EvidenceEnvelope,
    PromptEvidence,
    RenderedCitation,
    ValidatedAnswer,
)
from rag_kb.answering.evidence import render_validated_answer
from rag_kb.repositories.sqlalchemy_chat import _citations_equal, _serialized_success
from tools.run_large_evaluation import _score_enterprise, _score_graph


def _conflict(
    *,
    supporting: tuple[str, ...] = ("cite_1",),
    conflicting: tuple[str, ...] = ("cite_2",),
    conflict_type: AnswerConflictType = AnswerConflictType.VERSION,
    adjudication: AnswerConflictAdjudication = AnswerConflictAdjudication.RESOLVABLE,
) -> AnswerConflict:
    return AnswerConflict(
        supporting_citation_ids=supporting,
        conflicting_citation_ids=conflicting,
        conflict_type=conflict_type,
        adjudication=adjudication,
    )


class AnswerConflictDomainTests(unittest.TestCase):
    def test_conflict_requires_nonempty_disjoint_sides(self) -> None:
        conflict = _conflict()
        self.assertEqual(conflict.supporting_citation_ids, ("cite_1",))
        self.assertEqual(conflict.conflicting_citation_ids, ("cite_2",))
        self.assertEqual(conflict.conflict_type, AnswerConflictType.VERSION)
        self.assertEqual(conflict.adjudication, AnswerConflictAdjudication.RESOLVABLE)
        with self.assertRaises(ValueError):
            _conflict(supporting=())
        with self.assertRaises(ValueError):
            _conflict(conflicting=())
        with self.assertRaises(ValueError):
            _conflict(supporting=("cite_1",), conflicting=("cite_1",))
        with self.assertRaises(ValueError):
            _conflict(supporting=("cite_1", "cite_1"))

    def test_claim_without_conflict_keeps_the_existing_shape(self) -> None:
        claim = AnswerClaim(text="Revenue was 10.", citation_ids=("cite_1",))
        self.assertIsNone(claim.conflict)

    def test_conflict_citations_must_be_a_subset_of_claim_citations(self) -> None:
        conflict = _conflict()
        claim = AnswerClaim(
            text="Version 2 supersedes version 1.",
            citation_ids=("cite_1", "cite_2", "cite_3"),
            conflict=conflict,
        )
        self.assertIs(claim.conflict, conflict)
        with self.assertRaises(ValueError):
            AnswerClaim(
                text="Version 2 supersedes version 1.",
                citation_ids=("cite_1",),
                conflict=conflict,
            )


class RenderedEvidenceTests(unittest.TestCase):
    def test_rendering_reuses_admitted_metadata_in_first_citation_order(self) -> None:
        first = PromptEvidence(
            citation_id="cite_1", rank=1, index_chunk_id=uuid4(),
            document_id=uuid4(), document_version_id=uuid4(),
            document_display_name="source", document_original_filename="source.txt",
            excerpt="Revenue was 10.", source_location={"page": 1},
            score=0.9, modality="table", asset_snapshot={"id": "asset-1"},
            matched_representations=("text", "table"),
        )
        second = replace(first, citation_id="cite_2", rank=2, index_chunk_id=uuid4())
        envelope = EvidenceEnvelope(uuid4(), uuid4(), (first, second))
        answer = ValidatedAnswer(
            outcome=AnswerOutcome.ANSWERED,
            claims=(
                AnswerClaim("Revenue was 10.", ("cite_2", "cite_1")),
                AnswerClaim("The earlier source agrees.", ("cite_2",)),
            ),
            missing_aspects=(), source=AnswerDraftSource.PROVIDER,
        )
        rendered = render_validated_answer(answer, envelope, current_query="Revenue?")
        self.assertEqual([item.ordinal for item in rendered.citations], [0, 1])
        self.assertIs(rendered.citations[0].evidence, second)
        self.assertIs(rendered.citations[1].evidence, first)
        self.assertEqual(rendered.content, "Revenue was 10. [1][2]\n\nThe earlier source agrees. [1]")
        with self.assertRaises(TypeError):
            rendered.citations[0].evidence.source_location["page"] = 2
        with self.assertRaises(TypeError):
            rendered.citations[0].evidence.asset_snapshot["id"] = "changed"
        with self.assertRaises(ValueError):
            replace(rendered, citations=(RenderedCitation(1, first),))
        with self.assertRaises(ValueError):
            replace(rendered, citations=(RenderedCitation(0, first), RenderedCitation(1, first)))

        rows = [
            SimpleNamespace(
                ordinal=ordinal, index_chunk_id=item.index_chunk_id,
                document_id_snapshot=item.document_id,
                document_version_id_snapshot=item.document_version_id,
                document_display_name_snapshot=item.document_display_name,
                document_original_filename_snapshot=item.document_original_filename,
                quoted_text=item.excerpt, source_location=dict(item.source_location),
                score=item.score, modality=item.modality,
                asset_snapshot=dict(item.asset_snapshot),
                matched_representations=list(item.matched_representations),
            )
            for ordinal, item in enumerate((second, first))
        ]
        command = SimpleNamespace(
            rendered=rendered, retrieval_diagnostics={}, visual_decisions=(),
            visual_image_count=0, visual_total_bytes=0,
        )
        self.assertEqual(_serialized_success(command)["citation_ids"], ["cite_2", "cite_1"])
        self.assertTrue(_citations_equal(rows, command))
        rows[0].quoted_text = "changed"
        self.assertFalse(_citations_equal(rows, command))
        self.assertFalse(_citations_equal(rows[:1], command))

        answering = SimpleNamespace(validated=answer)
        case = {"expected_answer": "Revenue was 10", "query_only_terms": ["Revenue"]}
        enterprise = _score_enterprise(case, "source.txt")(rendered.content, rendered.citations, answering)
        graph = _score_graph(case, "source.txt")(rendered.content, rendered.citations, answering)
        self.assertTrue(enterprise["expected_citation_hit"])
        self.assertTrue(enterprise["policy_correct"])
        self.assertTrue(graph["expected_citation_hit"])
        self.assertTrue(graph["citation_query_term_leakage"])
        absent = _score_graph(case, "other.txt")(rendered.content, rendered.citations, answering)
        self.assertFalse(absent["expected_citation_hit"])


if __name__ == "__main__":
    unittest.main()
