from __future__ import annotations

import unittest
from uuid import UUID

from rag_kb.domain import Evidence, EvidenceScoreKind
from rag_kb.retrieval.calculator import (
    DecimalCalculationRejectReason,
    DecimalCalculationRejected,
    evaluate_decimal_expression,
)


def _evidence(key: str, text: str) -> tuple[str, Evidence]:
    value = Evidence(
        rank=1,
        index_chunk_id=UUID(int=1),
        indexed_document_version_id=UUID(int=2),
        document_id=UUID(int=3),
        document_version_id=UUID(int=4),
        index_revision_id=UUID(int=5),
        ordinal=0,
        text=text,
        source_location={},
        hierarchy={},
        source_metadata={},
        score=1.0,
        score_kind=EvidenceScoreKind.COSINE_SIMILARITY,
    )
    return key, value


class DecimalCalculatorTests(unittest.TestCase):
    def test_preserves_decimal_amounts_and_accepts_currency_and_grouping_source(self) -> None:
        evidence = dict(
            (
                _evidence(
                    "chunk:1",
                    "Revenue $229,104,123.45; R&D costs (262,106,374.56).",
                ),
            )
        )

        result = evaluate_decimal_expression(
            "229104123.45 + 262106374.56",
            source_evidence_keys=("chunk:1",),
            evidence=evidence,
        )

        self.assertEqual(result.result, "491210498.01")
        self.assertEqual(result.source_evidence_keys, ("chunk:1",))

    def test_supports_parentheses_negative_and_percentage_change(self) -> None:
        evidence = dict(
            (
                _evidence("chunk:1", "Old amount 1,500; new amount -1,200."),
            )
        )

        result = evaluate_decimal_expression(
            "(1200 - 1500) / 1500 * 100",
            source_evidence_keys=("chunk:1",),
            evidence=evidence,
        )

        self.assertEqual(result.result, "-20")

    def test_all_non_ratio_operands_must_match_declared_evidence(self) -> None:
        evidence = dict((_evidence("chunk:1", "Revenue 1,000."),))

        with self.assertRaises(DecimalCalculationRejected) as raised:
            evaluate_decimal_expression(
                "1000 + 7",
                source_evidence_keys=("chunk:1",),
                evidence=evidence,
            )

        self.assertEqual(
            raised.exception.reason,
            DecimalCalculationRejectReason.OPERAND_SOURCE_MISMATCH,
        )

    def test_rejects_unsupported_ast_and_division_by_zero(self) -> None:
        evidence = dict((_evidence("chunk:1", "Value 10; divisor 0."),))

        cases = (
            ("__import__('os').system('id')", DecimalCalculationRejectReason.UNSUPPORTED_AST),
            ("10 ** 2", DecimalCalculationRejectReason.UNSUPPORTED_AST),
            ("10 / 0", DecimalCalculationRejectReason.DIVISION_BY_ZERO),
            ("10e2", DecimalCalculationRejectReason.SYNTAX),
        )
        for expression, reason in cases:
            with self.subTest(expression=expression):
                with self.assertRaises(DecimalCalculationRejected) as raised:
                    evaluate_decimal_expression(
                        expression,
                        source_evidence_keys=("chunk:1",),
                        evidence=evidence,
                    )
                self.assertEqual(raised.exception.reason, reason)

    def test_rejects_long_expression_and_unknown_source_key(self) -> None:
        evidence = dict((_evidence("chunk:1", "Value 10."),))

        with self.assertRaises(DecimalCalculationRejected) as raised:
            evaluate_decimal_expression(
                "1" * 513,
                source_evidence_keys=("chunk:1",),
                evidence=evidence,
            )
        self.assertEqual(
            raised.exception.reason,
            DecimalCalculationRejectReason.EXPRESSION_TOO_LONG,
        )

        with self.assertRaises(DecimalCalculationRejected) as raised:
            evaluate_decimal_expression(
                "10",
                source_evidence_keys=("chunk:missing",),
                evidence=evidence,
            )
        self.assertEqual(
            raised.exception.reason,
            DecimalCalculationRejectReason.SOURCE_KEY_INVALID,
        )

if __name__ == "__main__":
    unittest.main()
