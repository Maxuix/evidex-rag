"""Deterministic, fail-closed document-name resolution for frozen chat runs."""

from __future__ import annotations

import re
import unicodedata
from uuid import UUID

from rag_kb.domain import RetrievalScopeCandidate, RetrievalScopeResolution


MAX_SCOPE_DOCUMENTS = 4
_FILENAME_TOKEN = re.compile(
    r"(?<![\w])([\w][\w.\-]{2,}\.(?:pdf|txt|html?|docx?))(?![\w])",
    flags=re.IGNORECASE,
)


def normalize_scope_name(value: str) -> str:
    """Normalize user-visible names without changing the stored source name."""

    normalized = unicodedata.normalize("NFC", value).replace("\\", "/")
    return normalized.rsplit("/", 1)[-1].strip().casefold()


def _stem(value: str) -> str:
    normalized = normalize_scope_name(value)
    return normalized.rsplit(".", 1)[0] if "." in normalized else normalized


def resolve_document_scope(
    query: str,
    candidates: tuple[RetrievalScopeCandidate, ...],
) -> RetrievalScopeResolution:
    """Resolve explicit document mentions; no match means the whole KB."""

    normalized_text = unicodedata.normalize("NFC", query)
    normalized_query = normalized_text.casefold()
    aliases: dict[str, list[RetrievalScopeCandidate]] = {}
    for candidate in candidates:
        values = {
            normalize_scope_name(candidate.original_filename),
            normalize_scope_name(candidate.display_name),
            _stem(candidate.original_filename),
        }
        for alias in values:
            if len(alias) >= 4:
                aliases.setdefault(alias, []).append(candidate)

    matches: list[tuple[int, str, tuple[RetrievalScopeCandidate, ...]]] = []
    for alias, values in aliases.items():
        position = normalized_query.find(alias)
        if position >= 0:
            matches.append((position, alias, tuple(values)))

    unknown_names = [
        unicodedata.normalize("NFC", item.group(1)).strip()
        for item in _FILENAME_TOKEN.finditer(normalized_text)
        if normalize_scope_name(item.group(1)) not in aliases
    ]
    if not matches and not unknown_names:
        return RetrievalScopeResolution(status="all")

    matches.sort(key=lambda item: (item[0], -len(item[1]), item[1]))
    required_names: list[str] = []
    resolved_ids: set[UUID] = set()
    unresolved: list[str] = list(dict.fromkeys(unknown_names))
    ambiguous: list[str] = []
    for _, alias, values in matches:
        display = values[0].original_filename or values[0].display_name
        if len(values) > 1:
            ambiguous.append(display)
            continue
        candidate = values[0]
        required_names.append(candidate.original_filename or candidate.display_name)
        if not candidate.eligible:
            unresolved.append(candidate.original_filename or candidate.display_name)
            continue
        resolved_ids.add(candidate.document_id)

    if len(resolved_ids) > MAX_SCOPE_DOCUMENTS:
        unresolved.append("scope_limit")
        resolved_ids.clear()

    resolved = tuple(
        candidate
        for candidate in candidates
        if candidate.document_id in resolved_ids
    )
    if ambiguous:
        status = "ambiguous"
        resolved = ()
    elif unresolved:
        status = "unresolved"
        resolved = ()
    else:
        status = "resolved"
    return RetrievalScopeResolution(
        status=status,
        required_names=tuple(dict.fromkeys(required_names)),
        resolved=resolved,
        unresolved_names=tuple(dict.fromkeys(unresolved)),
        ambiguous_names=tuple(dict.fromkeys(ambiguous)),
    )
