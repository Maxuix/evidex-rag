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
    def test_malformed_claims_invalidate_the_whole_submission(self) -> None:
        prompt = SimpleNamespace(citation_id="cite_1", matched_representations=("text",))
        good = {"text": " Supported fact ", "evidence_refs": ["ev_1"]}
        for patch in (
            {"text": ""},
            {"text": "x" * 4001},
            {"text": 17},
            {"evidence_refs": "ev_1"},
            {"legacy_field": True},
        ):
            with self.subTest(patch=patch):
                result = _validate_submission(
                    {"outcome": "answered", "claims": [good, {**good, **patch}], "unanswered": []},
                    prompt_by_ref={"ev_1": prompt}, loaded_visual_refs=set(),
                )
                self.assertIsNone(result)

    def test_unresolvable_refs_are_dropped_without_blocking_the_answer(self) -> None:
        prompt = SimpleNamespace(citation_id="cite_1", matched_representations=("text",))
        result = _validate_submission(
            {
                "outcome": "answered",
                "claims": [
                    {"text": "Cited fact", "evidence_refs": ["ev_1", "ev_1", "unknown"]},
                    {"text": "Uncited remark", "evidence_refs": []},
                ],
                "unanswered": [],
            },
            prompt_by_ref={"ev_1": prompt}, loaded_visual_refs=set(),
        )
        self.assertIsNotNone(result)
        # The model's outcome stands; no rewrite, no claim deletion.
        self.assertEqual(result.validated.outcome, AnswerOutcome.ANSWERED)
        self.assertEqual(
            result.validated.claims,
            (
                AnswerClaim("Cited fact", ("cite_1",)),
                AnswerClaim("Uncited remark", ()),
            ),
        )
        self.assertEqual(result.retained_refs, ("ev_1",))

    def test_refused_and_partial_shapes(self) -> None:
        prompt = SimpleNamespace(citation_id="cite_1", matched_representations=("text",))
        args = {"prompt_by_ref": {"ev_1": prompt}, "loaded_visual_refs": set()}
        refused = _validate_submission(
            {"outcome": "refused", "claims": [], "unanswered": []}, **args
        )
        self.assertEqual(refused.validated.outcome, AnswerOutcome.REFUSED)
        # refused must not carry claims.
        self.assertIsNone(
            _validate_submission(
                {
                    "outcome": "refused",
                    "claims": [{"text": "x", "evidence_refs": []}],
                    "unanswered": [],
                },
                **args,
            )
        )
        # answered/partial require at least one claim.
        self.assertIsNone(
            _validate_submission({"outcome": "answered", "claims": [], "unanswered": []}, **args)
        )
        partial = _validate_submission(
            {
                "outcome": "partial",
                "claims": [{"text": "Half an answer", "evidence_refs": ["ev_1"]}],
                "unanswered": ["the rest"],
            },
            **args,
        )
        self.assertEqual(partial.validated.outcome, AnswerOutcome.PARTIAL)
        self.assertEqual(partial.validated.missing_aspects, ("the rest",))

    def test_submission_limits_and_visual_admission_remain_boundary_checks(self) -> None:
        claim = {"text": "A fact", "evidence_refs": ["ev_1"]}
        args = {
            "prompt_by_ref": {"ev_1": SimpleNamespace(citation_id="cite_1", matched_representations=("image",))},
            "loaded_visual_refs": set(),
        }
        payload = {"outcome": "answered", "claims": [claim], "unanswered": []}
        # An unloaded visual ref drops out of citations but never blocks the claim.
        result = _validate_submission(payload, **args)
        self.assertEqual(result.validated.outcome, AnswerOutcome.ANSWERED)
        self.assertEqual(result.validated.claims, (AnswerClaim("A fact", ()),))
        args["loaded_visual_refs"] = {"ev_1"}
        with_visual = _validate_submission(payload, **args)
        self.assertEqual(with_visual.validated.claims, (AnswerClaim("A fact", ("cite_1",)),))
        for invalid in (
            {**payload, "claims": [claim] * 101},
            {**payload, "unanswered": ["x" * 1001]},
            {**payload, "outcome": "invented"},
            {**payload, "outcome": "clarify"},
        ):
            self.assertIsNone(_validate_submission(invalid, **args))


if __name__ == "__main__":
    unittest.main()
