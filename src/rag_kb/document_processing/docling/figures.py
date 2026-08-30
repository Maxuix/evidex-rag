"""Deterministic figure identities shared by relation building.

Author text names figures explicitly. These helpers recover those names
without treating arbitrary numbers as labels, so a relation can record an
author reference rather than a guess.
"""

from __future__ import annotations

import re
import unicodedata


_FIGURE_REFERENCE = re.compile(
    r"(?:(?:fig(?:ure)?)[.\s]*|图\s*)([0-9]+(?:[A-Za-z]|[.\-][0-9A-Za-z]+)?)",
    re.IGNORECASE,
)


def normalize_figure_labels(text: str) -> tuple[str, ...]:
    """Return stable Figure identities without treating arbitrary numbers as labels."""

    normalized = unicodedata.normalize("NFC", text)
    labels = {
        f"figure:{match.group(1).casefold().replace(' ', '')}"
        for match in _FIGURE_REFERENCE.finditer(normalized)
    }
    return tuple(sorted(labels))
