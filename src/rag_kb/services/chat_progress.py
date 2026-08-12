"""Best-effort, content-safe progress reporting for one Chat execution."""

from __future__ import annotations

from uuid import UUID

from rag_kb.domain import (
    MAX_PROGRESS_LIST_ITEMS,
    MAX_PROGRESS_TEXT_LENGTH,
    ChatProgressActivity,
    ChatProgressFacts,
    ChatProgressStage,
    ChatProgressStatus,
    ChatProgressUpdate,
)
from rag_kb.ports.chat_preview import ChatPreviewSink


class ChatProgressReporter:
    """Build full snapshots and isolate the pipeline from delivery failures."""

    def __init__(
        self,
        run_id: UUID,
        attempt: int,
        sink: ChatPreviewSink | None,
    ) -> None:
        self._run_id = run_id
        self._attempt = attempt
        self._sink = sink
        self._completed: list[ChatProgressStage] = []
        self._active_stage = ChatProgressStage.UNDERSTAND_QUERY

    async def show(
        self,
        stage: ChatProgressStage,
        activity: ChatProgressActivity,
        *,
        facts: ChatProgressFacts | None = None,
        completed: tuple[ChatProgressStage, ...] = (),
    ) -> None:
        for item in completed:
            if item not in self._completed:
                self._completed.append(item)
        self._active_stage = stage
        visible_facts = facts or ChatProgressFacts()
        await self._emit(
            ChatProgressUpdate(
                active_stage=stage,
                activity=activity,
                completed_stages=tuple(self._completed),
                facts=visible_facts,
            )
        )

    async def finish(self, activity: ChatProgressActivity) -> None:
        if self._active_stage not in self._completed:
            self._completed.append(self._active_stage)
        await self._emit(
            ChatProgressUpdate(
                active_stage=self._active_stage,
                activity=activity,
                completed_stages=tuple(self._completed),
                status=ChatProgressStatus.COMPLETED,
                facts=ChatProgressFacts(),
            )
        )

    async def _emit(self, update: ChatProgressUpdate) -> None:
        if self._sink is None or not self._sink.enabled:
            return
        try:
            await self._sink.emit_progress(
                run_id=self._run_id,
                attempt=self._attempt,
                update=update,
            )
        except Exception:
            # Live progress is explicitly non-authoritative.
            return


def bounded_progress_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    if not normalized:
        return None
    if len(normalized) <= MAX_PROGRESS_TEXT_LENGTH:
        return normalized
    return normalized[: MAX_PROGRESS_TEXT_LENGTH - 1].rstrip() + "…"


def bounded_progress_values(
    values: tuple[str, ...] | list[str],
    *,
    maximum: int = MAX_PROGRESS_LIST_ITEMS,
) -> tuple[str, ...]:
    output: list[str] = []
    for value in values:
        bounded = bounded_progress_text(value)
        if bounded and bounded not in output:
            output.append(bounded)
        if len(output) >= maximum:
            break
    return tuple(output)
