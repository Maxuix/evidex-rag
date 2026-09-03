"""One bounded pure Decimal calculation tool for the Chat agent."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from decimal import Decimal, DecimalException, DivisionByZero, localcontext
from enum import StrEnum
import re
from typing import Final


MAX_EXPRESSION_LENGTH: Final = 512
MAX_AST_NODES: Final = 64
MAX_LITERAL_LENGTH: Final = 64
MAX_RESULT_LENGTH: Final = 256
_DECIMAL_LITERAL = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)")


class DecimalCalculationRejectReason(StrEnum):
    EXPRESSION_EMPTY = "expression_empty"
    EXPRESSION_TOO_LONG = "expression_too_long"
    SYNTAX = "syntax"
    UNSUPPORTED_AST = "unsupported_ast"
    TOO_COMPLEX = "too_complex"
    DIVISION_BY_ZERO = "division_by_zero"
    NUMERIC_LIMIT = "numeric_limit"


class DecimalCalculationRejected(ValueError):
    """A calculation was refused for a stable, content-free reason."""

    def __init__(self, reason: DecimalCalculationRejectReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


@dataclass(frozen=True, slots=True)
class DecimalCalculationFact:
    """A validated pure-arithmetic result."""

    expression: str
    result: str

    def __post_init__(self) -> None:
        if not 1 <= len(self.expression) <= MAX_EXPRESSION_LENGTH:
            raise ValueError("calculation expression is out of bounds")
        if not self.result or len(self.result) > MAX_RESULT_LENGTH:
            raise ValueError("calculation result is out of bounds")

    def as_dict(self) -> dict[str, object]:
        return {
            "expression": self.expression,
            "result": self.result,
        }


def evaluate_decimal_expression(expression: str) -> DecimalCalculationFact:
    """Evaluate a bounded ``+ - * /`` expression without ``eval``."""

    if not isinstance(expression, str) or not expression.strip():
        raise DecimalCalculationRejected(
            DecimalCalculationRejectReason.EXPRESSION_EMPTY
        )
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise DecimalCalculationRejected(
            DecimalCalculationRejectReason.EXPRESSION_TOO_LONG
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, TypeError):
        raise DecimalCalculationRejected(DecimalCalculationRejectReason.SYNTAX)

    nodes = tuple(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise DecimalCalculationRejected(DecimalCalculationRejectReason.TOO_COMPLEX)
    _validate_tree(expression, tree)

    try:
        with localcontext() as context:
            context.prec = 80
            context.Emax = 10000
            context.Emin = -10000
            value = _evaluate(tree.body, expression)
    except DivisionByZero:
        raise DecimalCalculationRejected(
            DecimalCalculationRejectReason.DIVISION_BY_ZERO
        )
    except (DecimalException, ArithmeticError, ValueError, OverflowError):
        raise DecimalCalculationRejected(
            DecimalCalculationRejectReason.NUMERIC_LIMIT
        )
    if not value.is_finite():
        raise DecimalCalculationRejected(DecimalCalculationRejectReason.NUMERIC_LIMIT)
    result = format(value.normalize(), "f")
    if len(result) > MAX_RESULT_LENGTH:
        raise DecimalCalculationRejected(DecimalCalculationRejectReason.NUMERIC_LIMIT)
    return DecimalCalculationFact(
        expression=expression.strip(),
        result=result,
    )


def _validate_tree(expression: str, tree: ast.Expression) -> tuple[Decimal, ...]:
    literals: list[Decimal] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Expression | ast.BinOp | ast.UnaryOp):
            continue
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                node.value, (int, float)
            ):
                raise DecimalCalculationRejected(
                    DecimalCalculationRejectReason.UNSUPPORTED_AST
                )
            segment = ast.get_source_segment(expression, node)
            if (
                segment is None
                or len(segment) > MAX_LITERAL_LENGTH
                or _DECIMAL_LITERAL.fullmatch(segment.replace("_", "")) is None
            ):
                raise DecimalCalculationRejected(DecimalCalculationRejectReason.SYNTAX)
            try:
                literals.append(Decimal(segment.replace("_", "")))
            except (DecimalException, ValueError):
                raise DecimalCalculationRejected(
                    DecimalCalculationRejectReason.SYNTAX
                )
            continue
        if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.UAdd, ast.USub)):
            continue
        raise DecimalCalculationRejected(DecimalCalculationRejectReason.UNSUPPORTED_AST)
    return tuple(literals)


def _evaluate(node: ast.AST, expression: str) -> Decimal:
    if isinstance(node, ast.Constant):
        segment = ast.get_source_segment(expression, node)
        if segment is None:
            raise ValueError("missing numeric source segment")
        return Decimal(segment.replace("_", ""))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand, expression)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left, expression)
        right = _evaluate(node.right, expression)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError("unsupported calculation node")
