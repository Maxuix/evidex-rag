"""Best-effort PostgreSQL transport for non-authoritative Chat previews."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
import json
import logging
from typing import Any
from uuid import UUID

import asyncpg

from rag_kb.domain.chat_activity import (
    CHAT_ACTIVITY_VERSION, ActivityStep, ChatActivityEvent, activity_json,
)
from rag_kb.domain.chat_preview import (
    CHAT_PROGRESS_VERSION,
    ChatProgressActivity,
    ChatProgressFacts,
    ChatProgressSnapshot,
    ChatProgressStage,
    ChatProgressStatus,
    ChatProgressUpdate,
    ChatPreviewEvent,
)
from rag_kb.observability import get_logger, log_exception


CHAT_PREVIEW_CHANNEL = "rag_kb_chat_preview_v1"
MAX_NOTIFY_PAYLOAD_BYTES = 4_000
_MAX_ATTEMPT_STATES = 1_024
_RECONNECT_DELAYS_SECONDS = (0.25, 1.0, 2.0, 5.0)
_LOGGER = get_logger("rag_kb.chat_preview.pg_notify")
_Connect = Callable[..., Awaitable[Any]]
_Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class _ProgressCommand:
    run_id: UUID
    attempt: int
    update: ChatProgressUpdate


class PgNotifyPreviewSink:
    """Publish bounded, versioned Agent-progress NOTIFY payloads."""

    def __init__(
        self,
        sqlalchemy_dsn: str,
        *,
        command_queue_size: int = 256,
        connect: _Connect = asyncpg.connect,
    ) -> None:
        if command_queue_size < 1:
            raise ValueError("preview sink limits must be positive")
        self._dsn = _asyncpg_dsn(sqlalchemy_dsn)
        self._connect = connect
        self._commands: asyncio.Queue[_ProgressCommand | ChatActivityEvent] = asyncio.Queue(
            maxsize=command_queue_size
        )
        self._worker_task: asyncio.Task[None] | None = None
        self._connection: Any | None = None
        self._connection_lock = asyncio.Lock()
        self._closed = False
        self._reported_unavailable = False

    @property
    def enabled(self) -> bool:
        return not self._closed

    async def start(self) -> bool:
        if self._closed:
            return False
        self._ensure_worker()
        return await self._ensure_connection() is not None

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        task = self._worker_task
        self._worker_task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._discard_connection()

    def emit_activity(self, event: ChatActivityEvent) -> None:
        if self._closed:
            return
        self._ensure_worker()
        try:
            self._commands.put_nowait(event)
        except asyncio.QueueFull:
            # Producer-owned sequences reveal gaps; terminal snapshots reconcile.
            pass

    async def emit_progress(
        self,
        *,
        run_id: UUID,
        attempt: int,
        update: ChatProgressUpdate,
    ) -> None:
        if self._closed or attempt < 1:
            return
        self._ensure_worker()
        try:
            self._commands.put_nowait(_ProgressCommand(run_id, attempt, update))
        except asyncio.QueueFull:
            # A later full snapshot recovers the visible state.
            return

    def _ensure_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(
                self._run(),
                name="chat-preview-notify-sink",
            )

    async def _run(self) -> None:
        progress_sequences: OrderedDict[tuple[UUID, int], int] = OrderedDict()
        while not self._closed:
            command = await self._commands.get()
            if isinstance(command, ChatActivityEvent):
                try:
                    payload = serialize_preview_event(command)
                except (UnicodeError, ValueError):
                    continue
                await self._notify(payload)
            else:
                await self._publish_progress(progress_sequences, command)

    async def _publish_progress(
        self,
        sequences: OrderedDict[tuple[UUID, int], int],
        command: _ProgressCommand,
    ) -> None:
        key = (command.run_id, command.attempt)
        seq = sequences.pop(key, 1)
        sequences[key] = seq + 1
        if len(sequences) > _MAX_ATTEMPT_STATES:
            sequences.popitem(last=False)
        try:
            payload = serialize_preview_event(
                ChatProgressSnapshot(
                    run_id=command.run_id,
                    attempt=command.attempt,
                    seq=seq,
                    update=command.update,
                )
            )
        except (UnicodeError, ValueError):
            return
        await self._notify(payload)

    async def _notify(self, payload: str) -> None:
        connection = await self._ensure_connection()
        if connection is None:
            return
        try:
            await connection.execute(
                "SELECT pg_notify($1, $2)",
                CHAT_PREVIEW_CHANNEL,
                payload,
            )
        except Exception as error:
            await self._discard_connection(connection)
            self._report_unavailable(error)

    async def _ensure_connection(self) -> Any | None:
        if self._closed:
            return None
        async with self._connection_lock:
            if _connection_open(self._connection):
                return self._connection
            try:
                self._connection = await self._connect(
                    self._dsn,
                    timeout=5,
                    command_timeout=5,
                    server_settings={
                        "application_name": "rag-kb-worker-chat-preview"
                    },
                )
            except Exception as error:
                self._connection = None
                self._report_unavailable(error)
                return None
            self._reported_unavailable = False
            return self._connection

    async def _discard_connection(self, expected: Any | None = None) -> None:
        async with self._connection_lock:
            connection = self._connection
            if expected is not None and connection is not expected:
                return
            self._connection = None
            if connection is not None and _connection_open(connection):
                try:
                    await connection.close(timeout=2)
                except Exception:
                    pass

    def _report_unavailable(self, error: Exception) -> None:
        if self._reported_unavailable:
            return
        self._reported_unavailable = True
        log_exception(
            _LOGGER,
            "chat_preview_unavailable",
            error,
            level=logging.WARNING,
            component="notify_sink",
        )


class PgNotifyPreviewSubscription:
    """One bounded in-memory run subscription owned by an API request."""

    def __init__(
        self,
        broker: PgNotifyPreviewBroker,
        *,
        run_id: UUID,
        token: int,
        queue: asyncio.Queue[ChatPreviewEvent],
    ) -> None:
        self._broker = broker
        self.run_id = run_id
        self._token = token
        self._queue = queue
        self._closed = False

    async def next_event(self) -> ChatPreviewEvent:
        return await self._queue.get()

    def discard_pending(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._broker.unsubscribe(self.run_id, self._token)


class PgNotifyPreviewBroker:
    """Maintain one LISTEN connection and fan out events without blocking."""

    def __init__(
        self,
        sqlalchemy_dsn: str,
        *,
        subscriber_queue_size: int,
        connect: _Connect = asyncpg.connect,
        sleep: _Sleep = asyncio.sleep,
    ) -> None:
        if subscriber_queue_size < 1:
            raise ValueError("preview subscriber queue size must be positive")
        self._dsn = _asyncpg_dsn(sqlalchemy_dsn)
        self._subscriber_queue_size = subscriber_queue_size
        self._connect = connect
        self._sleep = sleep
        self._connection: Any | None = None
        self._connection_lock = asyncio.Lock()
        self._subscribers: dict[
            UUID, dict[int, asyncio.Queue[ChatPreviewEvent]]
        ] = {}
        self._next_token = 1
        self._reconnect_task: asyncio.Task[None] | None = None
        self._closed = False
        self._reported_unavailable = False

    async def start(self) -> bool:
        if self._closed:
            return False
        connected = await self._connect_listener()
        if not connected:
            self._schedule_reconnect()
        return connected

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._subscribers.clear()
        task = self._reconnect_task
        self._reconnect_task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        async with self._connection_lock:
            connection = self._connection
            self._connection = None
            if connection is not None and _connection_open(connection):
                try:
                    await connection.remove_listener(
                        CHAT_PREVIEW_CHANNEL,
                        self._on_notification,
                    )
                except Exception:
                    pass
                try:
                    await connection.close(timeout=2)
                except Exception:
                    pass

    async def subscribe(self, run_id: UUID) -> PgNotifyPreviewSubscription:
        token = self._next_token
        self._next_token += 1
        queue: asyncio.Queue[ChatPreviewEvent] = asyncio.Queue(
            maxsize=self._subscriber_queue_size
        )
        self._subscribers.setdefault(run_id, {})[token] = queue
        return PgNotifyPreviewSubscription(
            self,
            run_id=run_id,
            token=token,
            queue=queue,
        )

    def unsubscribe(self, run_id: UUID, token: int) -> None:
        subscribers = self._subscribers.get(run_id)
        if subscribers is None:
            return
        subscribers.pop(token, None)
        if not subscribers:
            self._subscribers.pop(run_id, None)

    def _on_notification(
        self,
        connection: Any,
        process_id: int,
        channel: str,
        payload: str,
    ) -> None:
        del connection, process_id
        if self._closed or channel != CHAT_PREVIEW_CHANNEL:
            return
        try:
            event = parse_preview_payload(payload)
        except (UnicodeError, ValueError):
            return
        for queue in tuple(self._subscribers.get(event.run_id, {}).values()):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def _connect_listener(self) -> bool:
        if self._closed:
            return False
        async with self._connection_lock:
            if _connection_open(self._connection):
                return True
            connection = None
            try:
                connection = await self._connect(
                    self._dsn,
                    timeout=5,
                    command_timeout=5,
                    server_settings={
                        "application_name": "rag-kb-api-chat-preview"
                    },
                )
                await connection.add_listener(
                    CHAT_PREVIEW_CHANNEL,
                    self._on_notification,
                )
                connection.add_termination_listener(self._on_termination)
            except Exception as error:
                if connection is not None and _connection_open(connection):
                    try:
                        await connection.close(timeout=2)
                    except Exception:
                        pass
                self._connection = None
                self._report_unavailable(error)
                return False
            self._connection = connection
            self._reported_unavailable = False
            return True

    def _on_termination(self, connection: Any) -> None:
        if connection is self._connection:
            self._connection = None
        self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        if self._closed:
            return
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(
                self._reconnect(),
                name="chat-preview-listen-reconnect",
            )

    async def _reconnect(self) -> None:
        attempt = 0
        while not self._closed:
            delay = _RECONNECT_DELAYS_SECONDS[
                min(attempt, len(_RECONNECT_DELAYS_SECONDS) - 1)
            ]
            await self._sleep(delay)
            if self._closed:
                return
            if await self._connect_listener():
                return
            attempt += 1

    def _report_unavailable(self, error: Exception) -> None:
        if self._reported_unavailable:
            return
        self._reported_unavailable = True
        log_exception(
            _LOGGER,
            "chat_preview_unavailable",
            error,
            level=logging.WARNING,
            component="listen_broker",
        )


def serialize_preview_event(event: ChatPreviewEvent) -> str:
    if isinstance(event, ChatActivityEvent):
        return _serialize_activity(event)
    update = event.update
    facts = update.facts
    payload: dict[str, object] = {
        "version": CHAT_PROGRESS_VERSION,
        "event": "agent.progress",
        "run_id": str(event.run_id),
        "attempt": event.attempt,
        "seq": event.seq,
        "active_stage": update.active_stage.value,
        "activity": update.activity.value,
        "completed_stages": [item.value for item in update.completed_stages],
        "status": update.status.value,
        "facts": {
            "objective": facts.objective,
            "queries": list(facts.queries),
            "evidence_count": facts.evidence_count,
            "new_evidence_count": facts.new_evidence_count,
            "retrieval_calls": facts.retrieval_calls,
            "covered_aspects": list(facts.covered_aspects),
            "missing_aspects": list(facts.missing_aspects),
            "conflict_count": facts.conflict_count,
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(encoded.encode("utf-8")) > MAX_NOTIFY_PAYLOAD_BYTES:
        raise ValueError("chat preview NOTIFY payload exceeds its byte limit")
    return encoded


def parse_preview_payload(payload: str) -> ChatPreviewEvent:
    if len(payload.encode("utf-8")) > MAX_NOTIFY_PAYLOAD_BYTES:
        raise ValueError("chat preview NOTIFY payload exceeds its byte limit")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("invalid chat preview JSON") from error
    if isinstance(value, dict) and value.get("version") == CHAT_ACTIVITY_VERSION:
        if set(value) != {"version", "event", "run_id", "attempt", "seq", "step", "elapsed_ms"} or value.get("event") != "agent.activity":
            raise ValueError("invalid activity envelope")
        try:
            return ChatActivityEvent(
                UUID(value["run_id"]), value["attempt"], value["seq"],
                ActivityStep.from_dict(value["step"]), value["elapsed_ms"],
            )
        except (TypeError, AttributeError) as error:
            raise ValueError("invalid activity values") from error
    if not isinstance(value, dict) or value.get("version") != CHAT_PROGRESS_VERSION:
        raise ValueError("invalid chat preview version")
    if value.get("event") != "agent.progress":
        raise ValueError("invalid chat preview event type")
    return _parse_progress_payload(value)


def _parse_progress_payload(value: dict[str, Any]) -> ChatProgressSnapshot:
    expected_keys = {
        "version",
        "event",
        "run_id",
        "attempt",
        "seq",
        "active_stage",
        "activity",
        "completed_stages",
        "status",
        "facts",
    }
    if value.get("version") != CHAT_PROGRESS_VERSION or set(value) != expected_keys:
        raise ValueError("invalid chat progress fields")
    attempt = value.get("attempt")
    seq = value.get("seq")
    if type(attempt) is not int or type(seq) is not int:
        raise ValueError("invalid chat progress sequence")
    try:
        run_id = UUID(value["run_id"])
        active_stage = ChatProgressStage(value.get("active_stage"))
        activity = ChatProgressActivity(value.get("activity"))
        status = ChatProgressStatus(value.get("status"))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("invalid chat progress enum or run ID") from error
    try:
        completed = _enum_tuple(
            value.get("completed_stages"), ChatProgressStage
        )
    except (TypeError, ValueError) as error:
        raise ValueError("invalid chat progress workflow values") from error
    facts_value = value.get("facts")
    if not isinstance(facts_value, dict) or set(facts_value) != {
        "objective",
        "queries",
        "evidence_count",
        "new_evidence_count",
        "retrieval_calls",
        "covered_aspects",
        "missing_aspects",
        "conflict_count",
    }:
        raise ValueError("invalid chat progress facts")
    objective = facts_value.get("objective")
    if objective is not None and not isinstance(objective, str):
        raise ValueError("invalid chat progress objective")
    for field in (
        "evidence_count",
        "new_evidence_count",
        "retrieval_calls",
        "conflict_count",
    ):
        item = facts_value.get(field)
        if item is not None and type(item) is not int:
            raise ValueError("invalid chat progress counter")
    try:
        facts = ChatProgressFacts(
            objective=objective,
            queries=_string_tuple(facts_value.get("queries")),
            evidence_count=facts_value.get("evidence_count"),
            new_evidence_count=facts_value.get("new_evidence_count"),
            retrieval_calls=facts_value.get("retrieval_calls"),
            covered_aspects=_string_tuple(
                facts_value.get("covered_aspects")
            ),
            missing_aspects=_string_tuple(
                facts_value.get("missing_aspects")
            ),
            conflict_count=facts_value.get("conflict_count"),
        )
        update = ChatProgressUpdate(
            active_stage=active_stage,
            activity=activity,
            completed_stages=completed,
            status=status,
            facts=facts,
        )
        return ChatProgressSnapshot(run_id, attempt, seq, update)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid chat progress payload") from error


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("invalid chat progress string list")
    return tuple(value)


def _enum_tuple(value: Any, enum: Any) -> tuple[Any, ...]:
    return tuple(enum(item) for item in _string_tuple(value))


def _connection_open(connection: Any | None) -> bool:
    if connection is None:
        return False
    is_closed = getattr(connection, "is_closed", None)
    return not callable(is_closed) or not is_closed()


def _asyncpg_dsn(sqlalchemy_dsn: str) -> str:
    prefix = "postgresql+asyncpg://"
    if not sqlalchemy_dsn.startswith(prefix):
        raise ValueError("chat preview DSN must use postgresql+asyncpg")
    return "postgresql://" + sqlalchemy_dsn[len(prefix) :]


def _serialize_activity(event: ChatActivityEvent) -> str:
    def encode(step: ActivityStep) -> str:
        return activity_json({
            "version": CHAT_ACTIVITY_VERSION, "event": "agent.activity",
            "run_id": str(event.run_id), "attempt": event.attempt,
            "seq": event.seq, "step": step.as_dict(), "elapsed_ms": event.elapsed_ms,
        })

    step = event.step
    encoded = encode(step)
    if len(encoded.encode("utf-8")) <= MAX_NOTIFY_PAYLOAD_BYTES:
        return encoded
    step = replace(
        step, details_truncated=True,
        queries=tuple(query[:120] for query in step.queries),
        sources=tuple(replace(source, title=source.title[:80], location=source.location[:80] if source.location else None) for source in step.sources[:2]),
        expression=step.expression[:160] if step.expression else None,
        result_value=step.result_value[:160] if step.result_value else None,
    )
    encoded = encode(step)
    if len(encoded.encode("utf-8")) > MAX_NOTIFY_PAYLOAD_BYTES:
        step = replace(step, sources=(), queries=tuple(query[:40] for query in step.queries), refs=())
        encoded = encode(step)
    if len(encoded.encode("utf-8")) > MAX_NOTIFY_PAYLOAD_BYTES:
        raise ValueError("activity preview exceeds byte limit")
    return encoded
