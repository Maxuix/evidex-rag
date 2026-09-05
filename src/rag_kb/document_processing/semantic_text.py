"""The exact text that a semantic span contributes to a final chunk."""
from rag_kb.domain import SemanticUnit


def joined_units(units: tuple[SemanticUnit, ...]) -> str:
    return "".join((unit.separator_before if index else "") + unit.text
                   for index, unit in enumerate(units))
