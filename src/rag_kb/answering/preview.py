"""Safe projection of partial answer JSON into non-authoritative preview text."""

from __future__ import annotations

from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any
from uuid import UUID

from langchain_core.utils.json import parse_partial_json

from rag_kb.domain.chat_preview import ChatPreviewResetReason
from rag_kb.ports.chat_preview import ChatPreviewSink


@dataclass(frozen=True, slots=True)
class PreviewProjection:
    delta: str | None = None
    invalidated: bool = False

    def __post_init__(self) -> None:
        if self.delta == "":
            raise ValueError("preview projection delta must not be empty")
        if self.delta is not None and self.invalidated:
            raise ValueError("preview projection cannot emit and invalidate together")


class PartialAnswerPreviewProjector:
    """Expose only a monotonic suffix of parsed ``claims[*].text`` values."""

    def __init__(self, *, max_visible_bytes: int) -> None:
        if max_visible_bytes < 1:
            raise ValueError("preview visible-byte limit must be positive")
        self._max_visible_bytes = max_visible_bytes
        self._published = ""
        self._disabled = False

    @property
    def published(self) -> str:
        return self._published

    @property
    def disabled(self) -> bool:
        return self._disabled

    def feed(self, accumulated_json: str) -> PreviewProjection:
        if self._disabled or not accumulated_json:
            return PreviewProjection()
        try:
            parsed = parse_partial_json(accumulated_json)
        except (JSONDecodeError, TypeError, ValueError):
            return self._invalidate()
        if not isinstance(parsed, dict):
            return self._invalidate()

        outcome = parsed.get("outcome")
        if outcome is None or (
            isinstance(outcome, str)
            and any(
                candidate.startswith(outcome)
                for candidate in (
                    "answered",
                    "partial",
                    "acknowledged",
                    "refused",
                )
            )
            and outcome not in {"answered", "partial", "acknowledged", "refused"}
        ):
            return PreviewProjection()
        if outcome in {"acknowledged", "refused"}:
            return self._invalidate() if self._published else PreviewProjection()
        if outcome not in {"answered", "partial"}:
            return self._invalidate()

        claims = parsed.get("claims")
        if claims is None:
            return PreviewProjection()
        if not isinstance(claims, list):
            return self._invalidate()
        text = self._claim_text(claims)
        if text is None or not text.startswith(self._published):
            return self._invalidate()
        try:
            visible_bytes = len(text.encode("utf-8"))
        except UnicodeEncodeError:
            return self._invalidate()
        if visible_bytes > self._max_visible_bytes:
            return self._invalidate()
        delta = text[len(self._published) :]
        if not delta:
            return PreviewProjection()
        self._published = text
        return PreviewProjection(delta=delta)

    def _claim_text(self, claims: list[Any]) -> str | None:
        values: list[str] = []
        for claim in claims:
            if not isinstance(claim, dict):
                return None
            value = claim.get("text")
            if value is None:
                continue
            if not isinstance(value, str):
                return None
            if value:
                values.append(value)
        return "\n\n".join(values)

    def _invalidate(self) -> PreviewProjection:
        if self._disabled:
            return PreviewProjection()
        self._disabled = True
        return PreviewProjection(invalidated=bool(self._published))


class NoOpChatPreviewSink:
    @property
    def enabled(self) -> bool:
        return False

    async def emit_delta(
        self,
        *,
        run_id: UUID,
        attempt: int,
        delta: str,
    ) -> None:
        del run_id, attempt, delta

    async def emit_reset(
        self,
        *,
        run_id: UUID,
        attempt: int,
        reason: ChatPreviewResetReason,
    ) -> None:
        del run_id, attempt, reason


async def emit_preview_delta_safely(
    sink: ChatPreviewSink,
    *,
    run_id: UUID,
    attempt: int,
    delta: str,
) -> bool:
    try:
        await sink.emit_delta(run_id=run_id, attempt=attempt, delta=delta)
    except Exception:
        return False
    return True


async def emit_preview_reset_safely(
    sink: ChatPreviewSink,
    *,
    run_id: UUID,
    attempt: int,
    reason: ChatPreviewResetReason,
) -> bool:
    try:
        await sink.emit_reset(run_id=run_id, attempt=attempt, reason=reason)
    except Exception:
        return False
    return True
