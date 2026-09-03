from __future__ import annotations

import unittest

from rag_kb.retrieval.calculator import (
    DecimalCalculationRejectReason,
    DecimalCalculationRejected,
    evaluate_decimal_expression,
)


class DecimalCalculatorTests(unittest.TestCase):
    def test_preserves_decimal_amounts(self) -> None:
        result = evaluate_decimal_expression("229104123.45 + 262106374.56")

        self.assertEqual(result.result, "491210498.01")
        self.assertEqual(result.expression, "229104123.45 + 262106374.56")

    def test_supports_parentheses_negative_and_percentage_change(self) -> None:
        result = evaluate_decimal_expression("(1200 - 1500) / 1500 * 100")

        self.assertEqual(result.result, "-20")

    def test_pure_function_evaluates_without_evidence_binding(self) -> None:
        result = evaluate_decimal_expression("1000 + 7")

        self.assertEqual(result.result, "1007")

    def test_rejects_unsupported_ast_and_division_by_zero(self) -> None:
        cases = (
            ("__import__('os').system('id')", DecimalCalculationRejectReason.UNSUPPORTED_AST),
            ("10 ** 2", DecimalCalculationRejectReason.UNSUPPORTED_AST),
            ("10 / 0", DecimalCalculationRejectReason.DIVISION_BY_ZERO),
            ("10e2", DecimalCalculationRejectReason.SYNTAX),
        )
        for expression, reason in cases:
            with self.subTest(expression=expression):
                with self.assertRaises(DecimalCalculationRejected) as raised:
                    evaluate_decimal_expression(expression)
                self.assertEqual(raised.exception.reason, reason)

    def test_rejects_long_or_empty_expression(self) -> None:
        with self.assertRaises(DecimalCalculationRejected) as raised:
            evaluate_decimal_expression("1" * 513)
        self.assertEqual(
            raised.exception.reason,
            DecimalCalculationRejectReason.EXPRESSION_TOO_LONG,
        )

        with self.assertRaises(DecimalCalculationRejected) as raised:
            evaluate_decimal_expression("   ")
        self.assertEqual(
            raised.exception.reason,
            DecimalCalculationRejectReason.EXPRESSION_EMPTY,
        )


if __name__ == "__main__":
    unittest.main()
