from __future__ import annotations

import unittest

from rag_kb.domain import (
    AnswerClaim,
    AnswerConflict,
    AnswerConflictAdjudication,
    AnswerConflictType,
)


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


if __name__ == "__main__":
    unittest.main()
