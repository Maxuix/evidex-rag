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
from rag_kb.answering.evidence import render_validated_answer
from rag_kb.answering.agent import _validate_submission
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

class SubmissionBoundaryTests(unittest.TestCase):
    def test_invalid_provider_claims_are_rejected_without_internal_dto_validation(self) -> None:
        prompt = SimpleNamespace(citation_id="cite_1", matched_representations=("text",))
        good = {"text": " Supported fact ", "evidence_refs": ["ev_1"]}
        for patch, reason in (
            ({"text": ""}, "claim_text"),
            ({"text": "x" * 4001}, "claim_text"),
            ({"text": 17}, "claim_text"),
            ({"evidence_refs": []}, "evidence_ref"),
            ({"evidence_refs": ["unknown"]}, "evidence_ref"),
            ({"evidence_refs": ["ev_1", "ev_1"]}, "evidence_ref"),
            ({"calculation_refs": ["unknown"]}, "calculation_ref"),
            ({"legacy_field": True}, "claim_shape"),
        ):
            with self.subTest(patch=patch):
                result = _validate_submission(
                    {"outcome": "answered", "claims": [good, {**good, **patch}], "unanswered": []},
                    prompt_by_ref={"ev_1": prompt}, loaded_visual_refs=set(), calculations={},
                )
                self.assertEqual(result.validated.outcome, AnswerOutcome.PARTIAL)
                self.assertEqual(result.validated.claims, (AnswerClaim("Supported fact", ("cite_1",)),))
                self.assertEqual(result.rejected_claim_count, 1)
                self.assertIn(reason, result.rejection_reasons)

    def test_submission_limits_and_visual_admission_remain_boundary_checks(self) -> None:
        claim = {"text": "A fact", "evidence_refs": ["ev_1"]}
        args = {
            "prompt_by_ref": {"ev_1": SimpleNamespace(citation_id="cite_1", matched_representations=("image",))},
            "loaded_visual_refs": set(), "calculations": {},
        }
        payload = {"outcome": "answered", "claims": [claim], "unanswered": []}
        result = _validate_submission(payload, **args)
        self.assertEqual(result.validated.outcome, AnswerOutcome.REFUSED)
        self.assertEqual(result.rejection_reasons, ("visual_ref",))
        args["loaded_visual_refs"] = {"ev_1"}
        self.assertEqual(_validate_submission(payload, **args).validated.outcome, AnswerOutcome.ANSWERED)
        for invalid in (
            {**payload, "claims": [claim] * 101},
            {**payload, "unanswered": ["x" * 1001]},
            {**payload, "outcome": "invented"},
        ):
            self.assertIsNone(_validate_submission(invalid, **args))


if __name__ == "__main__":
    unittest.main()
