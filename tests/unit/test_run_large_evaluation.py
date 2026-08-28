from __future__ import annotations

from types import SimpleNamespace
import unittest

from rag_kb.domain import (
    AnswerClaim,
    AnswerConflict,
    AnswerConflictAdjudication,
    AnswerConflictType,
    AnswerControlReason,
    AnswerDraftSource,
    AnswerOutcome,
    ValidatedAnswer,
)
from tools.run_large_evaluation import _score_public


def _answering(*, outcome: str, conflict: bool) -> SimpleNamespace:
    claims: tuple[AnswerClaim, ...] = ()
    missing: tuple[str, ...] = ()
    if outcome in {"answered", "partial"}:
        if conflict:
            claims = (
                AnswerClaim(
                    text="A later version reports 12; an older memo reports 10.",
                    citation_ids=("cite_1", "cite_2"),
                    conflict=AnswerConflict(
                        supporting_citation_ids=("cite_1",),
                        conflicting_citation_ids=("cite_2",),
                        conflict_type=AnswerConflictType.VERSION,
                        adjudication=AnswerConflictAdjudication.RESOLVABLE,
                    ),
                ),
            )
        else:
            claims = (AnswerClaim(text="Revenue was 10.", citation_ids=("cite_1",)),)
        if outcome == "partial":
            missing = ("cause",)
    validated = ValidatedAnswer(
        outcome=AnswerOutcome(outcome),
        claims=claims,
        missing_aspects=missing,
        source=AnswerDraftSource.PROVIDER,
        control_reason=(
            AnswerControlReason.INSUFFICIENT_EVIDENCE if outcome == "refused" else None
        ),
    )
    return SimpleNamespace(validated=validated)


class PublicScorerConflictTests(unittest.TestCase):
    def test_surface_evidence_conflict_requires_a_complete_conflict_claim(self) -> None:
        score = _score_public(
            {
                "expected_action": "surface_evidence_conflict",
                "stratum": "conflicting_info",
            }
        )
        keyword_only = score(
            "The sources conflict and disagree on the amount.",
            (),
            _answering(outcome="answered", conflict=False),
        )
        self.assertFalse(keyword_only["policy_correct"])
        self.assertTrue(keyword_only["conflict_marker_present"])

        structured = score(
            "The sources conflict and disagree on the amount.",
            (),
            _answering(outcome="answered", conflict=True),
        )
        self.assertTrue(structured["policy_correct"])
        self.assertTrue(structured["conflict_marker_present"])

        refused = score(
            "The sources conflict.",
            (),
            _answering(outcome="refused", conflict=False),
        )
        self.assertFalse(refused["policy_correct"])

    def test_answer_without_false_conflict_rejects_conflict_claims(self) -> None:
        score = _score_public(
            {"expected_action": "answer_without_false_conflict"}
        )
        clean = score(
            "Revenue was 10.",
            (),
            _answering(outcome="answered", conflict=False),
        )
        self.assertTrue(clean["policy_correct"])
        self.assertFalse(clean["conflict_marker_present"])

        false_conflict = score(
            "Revenue was 10.",
            (),
            _answering(outcome="answered", conflict=True),
        )
        self.assertFalse(false_conflict["policy_correct"])

        refused = score(
            "Revenue was 10.",
            (),
            _answering(outcome="refused", conflict=False),
        )
        self.assertFalse(refused["policy_correct"])


if __name__ == "__main__":
    unittest.main()
