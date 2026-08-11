"""One bounded, source-grounded Decimal calculation for the fixed Agent."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, DecimalException, DivisionByZero, localcontext
from enum import StrEnum
import re
from typing import Final

from rag_kb.domain.retrieval import Evidence


MAX_EXPRESSION_LENGTH: Final = 512
MAX_SOURCE_EVIDENCE_KEYS: Final = 4
MAX_AST_NODES: Final = 64
MAX_LITERAL_LENGTH: Final = 64
MAX_RESULT_LENGTH: Final = 256
CALCULATION_FACTS_ARTIFACT: Final = "chat_calculation_facts"
_DECIMAL_LITERAL = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)")
_SOURCE_NUMBER = re.compile(
    r"(?<![\w.])"
    r"[-+]?\s*"
    r"(?:[$€£¥]\s*)?"
    r"(?:\(\s*)?"
    r"[-+]?\d[\d,_]*(?:\.\d+)?"
    r"(?:\s*\))?"
    r"(?![\w])"
)


class DecimalCalculationRejectReason(StrEnum):
    EXPRESSION_EMPTY = "expression_empty"
    EXPRESSION_TOO_LONG = "expression_too_long"
    SYNTAX = "syntax"
    UNSUPPORTED_AST = "unsupported_ast"
    TOO_COMPLEX = "too_complex"
    SOURCE_KEY_INVALID = "source_key_invalid"
    OPERAND_SOURCE_MISMATCH = "operand_source_mismatch"
    DIVISION_BY_ZERO = "division_by_zero"
    NUMERIC_LIMIT = "numeric_limit"


class DecimalCalculationRejected(ValueError):
    """A calculation was refused for a stable, content-free reason."""

    def __init__(self, reason: DecimalCalculationRejectReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


@dataclass(frozen=True, slots=True)
class DecimalCalculationFact:
    """A validated result; source text is deliberately not copied into it."""

    expression: str
    result: str
    source_evidence_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.expression) <= MAX_EXPRESSION_LENGTH:
            raise ValueError("calculation expression is out of bounds")
        if not self.result or len(self.result) > MAX_RESULT_LENGTH:
            raise ValueError("calculation result is out of bounds")
        if not 1 <= len(self.source_evidence_keys) <= MAX_SOURCE_EVIDENCE_KEYS:
            raise ValueError("calculation source keys are out of bounds")
        if len(set(self.source_evidence_keys)) != len(self.source_evidence_keys):
            raise ValueError("calculation source keys must be unique")

    def as_dict(self) -> dict[str, object]:
        return {
            "expression": self.expression,
            "result": self.result,
            "source_evidence_keys": list(self.source_evidence_keys),
        }


def evaluate_decimal_expression(
    expression: str,
    *,
    source_evidence_keys: Sequence[str],
    evidence: Mapping[str, Evidence],
) -> DecimalCalculationFact:
    """Evaluate a source-grounded ``+ - * /`` expression without ``eval``."""

    if not isinstance(expression, str) or not expression.strip():
        raise DecimalCalculationRejected(
            DecimalCalculationRejectReason.EXPRESSION_EMPTY
        )
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise DecimalCalculationRejected(
            DecimalCalculationRejectReason.EXPRESSION_TOO_LONG
        )
    keys = tuple(source_evidence_keys)
    if (
        not 1 <= len(keys) <= MAX_SOURCE_EVIDENCE_KEYS
        or len(set(keys)) != len(keys)
        or any(not isinstance(key, str) or not key.strip() for key in keys)
        or any(key not in evidence for key in keys)
    ):
        raise DecimalCalculationRejected(
            DecimalCalculationRejectReason.SOURCE_KEY_INVALID
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, TypeError):
        raise DecimalCalculationRejected(DecimalCalculationRejectReason.SYNTAX)

    nodes = tuple(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise DecimalCalculationRejected(DecimalCalculationRejectReason.TOO_COMPLEX)
    literals = _validate_tree(expression, tree)
    source_numbers = tuple(
        number
        for key in keys
        for number in _source_numbers(evidence[key].text)
    )
    for literal in literals:
        if literal != Decimal("100") and not any(
            literal == source_number for source_number in source_numbers
        ):
            raise DecimalCalculationRejected(
                DecimalCalculationRejectReason.OPERAND_SOURCE_MISMATCH
            )

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
        source_evidence_keys=keys,
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


def _source_numbers(text: str) -> tuple[Decimal, ...]:
    values: list[Decimal] = []
    for match in _SOURCE_NUMBER.finditer(text):
        token = match.group(0)
        normalized = token.strip().replace(" ", "").replace(",", "").replace("_", "")
        if normalized.startswith("(") and normalized.endswith(")"):
            normalized = normalized[1:-1]
        sign = ""
        if normalized[:1] in {"+", "-"}:
            sign, normalized = normalized[0], normalized[1:]
        normalized = sign + normalized.lstrip("$€£¥")
        try:
            value = Decimal(normalized)
        except (DecimalException, ValueError):
            continue
        values.append(abs(value))
    return tuple(values)


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
