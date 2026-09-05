from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

from rag_kb.domain import (
    AnswerClaim,
    AnswerDraftSource,
    AnswerOutcome,
    EvidenceEnvelope,
    PromptEvidence,
    ValidatedAnswer,
)
from rag_kb.answering.evidence import render_text_final_answer, render_validated_answer
from rag_kb.repositories.sqlalchemy_chat import _citations_equal, _serialized_success


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
            replace(rendered, content=" ")
        with self.assertRaises(ValueError):
            replace(rendered, content="x" * (2 * 1024 * 1024 + 1))

        rows = [
            SimpleNamespace(
                ordinal=ordinal, index_chunk_id=item.index_chunk_id,
                knowledge_base_id_snapshot=item.knowledge_base_id,
                knowledge_base_name_snapshot=item.knowledge_base_name,
                index_revision_id_snapshot=item.index_revision_id,
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

class TextFinalBoundaryTests(unittest.TestCase):
    def test_inline_refs_are_resolved_with_tolerant_brackets_and_first_use_order(self) -> None:
        first = SimpleNamespace(citation_id="cite_1", matched_representations=("text",))
        second = SimpleNamespace(citation_id="cite_2", matched_representations=("ocr_text",))
        validated, rendered, retained, observed = render_text_final_answer(
            "Second 【EV_2】, first （ev_1，ev_404）, second again [ev_2].",
            {"ev_1": first, "ev_2": second},
            loaded_visual_refs=set(),
            current_query="Explain.",
        )
        self.assertEqual(validated.outcome, AnswerOutcome.ANSWERED)
        self.assertEqual(rendered.content, "Second [1], first [2], second again [1].")
        self.assertEqual(retained, ("ev_2", "ev_1"))
        self.assertEqual(observed, ("ev_2", "ev_1", "ev_404", "ev_2"))
        self.assertEqual(validated.claims[0].citation_ids, ("cite_2", "cite_1"))
        self.assertEqual([item.ordinal for item in rendered.citations], [0, 1])

    def test_unknown_refs_are_silently_removed_and_zero_resolved_refs_refuse(self) -> None:
        validated, rendered, retained, observed = render_text_final_answer(
            "The premise is not established [ev_999].",
            {},
            loaded_visual_refs=set(),
            current_query="Did it happen?",
        )
        self.assertEqual(validated.outcome, AnswerOutcome.REFUSED)
        self.assertEqual(rendered.outcome, AnswerOutcome.REFUSED)
        self.assertEqual(rendered.content, "The premise is not established .")
        self.assertEqual(rendered.citations, ())
        self.assertEqual(retained, ())
        self.assertEqual(observed, ("ev_999",))
        self.assertEqual(validated.missing_aspects, ())

    def test_visual_only_ref_requires_the_image_to_have_been_loaded(self) -> None:
        visual = SimpleNamespace(citation_id="cite_1", matched_representations=("image",))
        args = {
            "prompt_by_ref": {"ev_1": visual},
            "current_query": "What is shown?",
        }
        refused = render_text_final_answer(
            "A chart [ev_1]", loaded_visual_refs=set(), **args
        )
        self.assertEqual(refused[0].outcome, AnswerOutcome.REFUSED)
        answered = render_text_final_answer(
            "A chart [ev_1]", loaded_visual_refs={"ev_1"}, **args
        )
        self.assertEqual(answered[0].outcome, AnswerOutcome.ANSWERED)
        self.assertEqual(answered[1].content, "A chart [1]")

    def test_plain_text_without_a_resolved_ref_is_a_normal_refusal(self) -> None:
        validated, rendered, retained, observed = render_text_final_answer(
            "The available knowledge base does not establish the premise.",
            {},
            loaded_visual_refs=set(),
            current_query="Question",
        )
        self.assertEqual(validated.outcome, AnswerOutcome.REFUSED)
        self.assertEqual(rendered.content, "The available knowledge base does not establish the premise.")
        self.assertEqual(retained, ())
        self.assertEqual(observed, ())


if __name__ == "__main__":
    unittest.main()
