"""Best-effort PostgreSQL transport for non-authoritative Chat previews."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import json
import logging
from typing import Any
from uuid import UUID

import asyncpg

from rag_kb.domain.chat_preview import (
    CHAT_PROGRESS_VERSION,
    CHAT_PREVIEW_VERSION,
    ChatProgressActivity,
    ChatProgressDecision,
    ChatProgressFacts,
    ChatProgressSnapshot,
    ChatProgressStage,
    ChatProgressStatus,
    ChatProgressUpdate,
    ChatPreviewDelta,
    ChatPreviewEvent,
    ChatPreviewReset,
    ChatPreviewResetReason,
)
from rag_kb.domain.chat_workflow import (
    ChatResolvedMode,
    ChatRouteReason,
    ChatRouteStatus,
    ChatWorkflowMode,
    ResearchStatus,
)
from rag_kb.observability import get_logger, log_exception


CHAT_PREVIEW_CHANNEL = "rag_kb_chat_preview_v1"
MAX_NOTIFY_PAYLOAD_BYTES = 4_000
_MAX_ATTEMPT_STATES = 1_024
_MAX_STOPPED_ATTEMPTS = 1_024
_RECONNECT_DELAYS_SECONDS = (0.25, 1.0, 2.0, 5.0)
_LOGGER = get_logger("rag_kb.chat_preview.pg_notify")
_Connect = Callable[..., Awaitable[Any]]
_Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class _DeltaCommand:
    run_id: UUID
    attempt: int
    delta: str


@dataclass(frozen=True, slots=True)
class _ResetCommand:
    run_id: UUID
    attempt: int
    reason: ChatPreviewResetReason


@dataclass(frozen=True, slots=True)
class _ProgressCommand:
    run_id: UUID
    attempt: int
    update: ChatProgressUpdate


@dataclass(slots=True)
class _AttemptState:
    next_seq: int = 1
    total_bytes: int = 0
    buffer: str = ""
    flush_at: float | None = None


class PgNotifyPreviewSink:
    """Batch preview text and publish bounded, versioned NOTIFY payloads."""

    def __init__(
        self,
        sqlalchemy_dsn: str,
        *,
        flush_interval_ms: int,
        max_total_bytes: int,
        command_queue_size: int = 256,
        connect: _Connect = asyncpg.connect,
    ) -> None:
        if flush_interval_ms < 1:
            raise ValueError("preview flush interval must be positive")
        if max_total_bytes < 1 or command_queue_size < 1:
            raise ValueError("preview sink limits must be positive")
        self._dsn = _asyncpg_dsn(sqlalchemy_dsn)
        self._flush_interval_seconds = flush_interval_ms / 1_000
        self._max_total_bytes = max_total_bytes
        self._connect = connect
        self._commands: asyncio.Queue[
            _DeltaCommand | _ResetCommand | _ProgressCommand
        ] = (
            asyncio.Queue(maxsize=command_queue_size)
        )
        self._forced_resets: dict[
            tuple[UUID, int], ChatPreviewResetReason
        ] = {}
        self._stopped_keys: OrderedDict[tuple[UUID, int], None] = OrderedDict()
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

    async def emit_delta(
        self,
        *,
        run_id: UUID,
        attempt: int,
        delta: str,
    ) -> None:
        if self._closed or not delta:
            return
        key = (run_id, attempt)
        if key in self._stopped_keys or key in self._forced_resets:
            return
        self._ensure_worker()
        try:
            self._commands.put_nowait(_DeltaCommand(run_id, attempt, delta))
        except asyncio.QueueFull:
            self._forced_resets[key] = ChatPreviewResetReason.PREVIEW_INVALID
            self._mark_stopped(key)

    async def emit_reset(
        self,
        *,
        run_id: UUID,
        attempt: int,
        reason: ChatPreviewResetReason,
    ) -> None:
        if self._closed:
            return
        key = (run_id, attempt)
        self._mark_stopped(key)
        self._ensure_worker()
        try:
            self._commands.put_nowait(_ResetCommand(run_id, attempt, reason))
        except asyncio.QueueFull:
            self._forced_resets[key] = reason

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
        states: dict[tuple[UUID, int], _AttemptState] = {}
        progress_sequences: OrderedDict[tuple[UUID, int], int] = OrderedDict()
        loop = asyncio.get_running_loop()
        while not self._closed:
            await self._publish_forced_resets(states)
            timeout = _next_flush_timeout(states, loop.time())
            try:
                if timeout is None:
                    command = await self._commands.get()
                else:
                    command = await asyncio.wait_for(
                        self._commands.get(),
                        timeout=timeout,
                    )
            except TimeoutError:
                command = None

            if isinstance(command, _DeltaCommand):
                await self._accept_delta(states, command, loop.time())
            elif isinstance(command, _ResetCommand):
                await self._publish_reset(states, command)
            elif isinstance(command, _ProgressCommand):
                await self._publish_progress(progress_sequences, command)
            await self._flush_due(states, loop.time())

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

    async def _accept_delta(
        self,
        states: dict[tuple[UUID, int], _AttemptState],
        command: _DeltaCommand,
        now: float,
    ) -> None:
        key = (command.run_id, command.attempt)
        if key in self._stopped_keys:
            return
        state = states.get(key)
        if state is None:
            if len(states) >= _MAX_ATTEMPT_STATES:
                states.pop(next(iter(states)))
            state = _AttemptState()
            states[key] = state
        available = self._max_total_bytes - state.total_bytes
        accepted = _utf8_prefix(command.delta, available)
        if not accepted:
            self._mark_stopped(key)
            return
        accepted_bytes = len(accepted.encode("utf-8"))
        state.buffer += accepted
        state.total_bytes += accepted_bytes
        if state.flush_at is None:
            state.flush_at = now + self._flush_interval_seconds
        if accepted != command.delta or state.total_bytes >= self._max_total_bytes:
            self._mark_stopped(key)
        if not _delta_payload_fits(
            command.run_id,
            command.attempt,
            state.next_seq,
            state.buffer,
        ):
            await self._flush(key, state)

    async def _flush_due(
        self,
        states: dict[tuple[UUID, int], _AttemptState],
        now: float,
    ) -> None:
        for key, state in tuple(states.items()):
            if state.flush_at is not None and state.flush_at <= now:
                await self._flush(key, state)

    async def _flush(
        self,
        key: tuple[UUID, int],
        state: _AttemptState,
    ) -> None:
        if not state.buffer:
            state.flush_at = None
            return
        run_id, attempt = key
        payloads = serialize_preview_delta_payloads(
            run_id=run_id,
            attempt=attempt,
            starting_seq=state.next_seq,
            delta=state.buffer,
        )
        state.buffer = ""
        state.flush_at = None
        state.next_seq += len(payloads)
        for payload in payloads:
            await self._notify(payload)

    async def _publish_reset(
        self,
        states: dict[tuple[UUID, int], _AttemptState],
        command: _ResetCommand,
    ) -> None:
        key = (command.run_id, command.attempt)
        state = states.pop(key, _AttemptState())
        payload = serialize_preview_event(
            ChatPreviewReset(
                run_id=command.run_id,
                attempt=command.attempt,
                seq=state.next_seq,
                reason=command.reason,
            )
        )
        await self._notify(payload)

    async def _publish_forced_resets(
        self,
        states: dict[tuple[UUID, int], _AttemptState],
    ) -> None:
        pending = tuple(self._forced_resets.items())
        self._forced_resets.clear()
        for (run_id, attempt), reason in pending:
            await self._publish_reset(
                states,
                _ResetCommand(run_id, attempt, reason),
            )

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

    def _mark_stopped(self, key: tuple[UUID, int]) -> None:
        self._stopped_keys.pop(key, None)
        self._stopped_keys[key] = None
        if len(self._stopped_keys) > _MAX_STOPPED_ATTEMPTS:
            self._stopped_keys.popitem(last=False)


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
    if isinstance(event, ChatPreviewDelta):
        payload: dict[str, object] = {
            "version": CHAT_PREVIEW_VERSION,
            "event": "answer.preview.delta",
            "run_id": str(event.run_id),
            "attempt": event.attempt,
            "seq": event.seq,
            "delta": event.delta,
        }
    elif isinstance(event, ChatPreviewReset):
        payload = {
            "version": CHAT_PREVIEW_VERSION,
            "event": "answer.preview.reset",
            "run_id": str(event.run_id),
            "attempt": event.attempt,
            "seq": event.seq,
            "reason": event.reason.value,
        }
    else:
        update = event.update
        facts = update.facts
        payload = {
            "version": CHAT_PROGRESS_VERSION,
            "event": "workflow.progress",
            "run_id": str(event.run_id),
            "attempt": event.attempt,
            "seq": event.seq,
            "active_stage": update.active_stage.value,
            "activity": update.activity.value,
            "completed_stages": [item.value for item in update.completed_stages],
            "status": update.status.value,
            "requested_mode": (
                update.requested_mode.value
                if update.requested_mode is not None
                else None
            ),
            "resolved_mode": update.resolved_mode.value,
            "facts": {
                "objective": facts.objective,
                "queries": list(facts.queries),
                "evidence_count": facts.evidence_count,
                "new_evidence_count": facts.new_evidence_count,
                "retrieval_calls": facts.retrieval_calls,
                "route_status": (
                    facts.route_status.value
                    if facts.route_status is not None
                    else None
                ),
                "route_reason_codes": [
                    item.value for item in facts.route_reason_codes
                ],
                "research_status": (
                    facts.research_status.value
                    if facts.research_status is not None
                    else None
                ),
                "covered_aspects": list(facts.covered_aspects),
                "missing_aspects": list(facts.missing_aspects),
                "conflict_count": facts.conflict_count,
                "decision": (
                    facts.decision.value if facts.decision is not None else None
                ),
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


def serialize_preview_delta_payloads(
    *,
    run_id: UUID,
    attempt: int,
    starting_seq: int,
    delta: str,
) -> tuple[str, ...]:
    if not delta or attempt < 1 or starting_seq < 1:
        raise ValueError("invalid preview delta serialization input")
    payloads: list[str] = []
    remaining = delta
    seq = starting_seq
    while remaining:
        low = 1
        high = len(remaining)
        accepted = 0
        while low <= high:
            middle = (low + high) // 2
            event = ChatPreviewDelta(
                run_id=run_id,
                attempt=attempt,
                seq=seq,
                delta=remaining[:middle],
            )
            try:
                serialize_preview_event(event)
            except (UnicodeError, ValueError):
                high = middle - 1
            else:
                accepted = middle
                low = middle + 1
        if accepted == 0:
            raise ValueError("preview delta cannot fit in one NOTIFY payload")
        payloads.append(
            serialize_preview_event(
                ChatPreviewDelta(
                    run_id=run_id,
                    attempt=attempt,
                    seq=seq,
                    delta=remaining[:accepted],
                )
            )
        )
        remaining = remaining[accepted:]
        seq += 1
    return tuple(payloads)


def parse_preview_payload(payload: str) -> ChatPreviewEvent:
    if len(payload.encode("utf-8")) > MAX_NOTIFY_PAYLOAD_BYTES:
        raise ValueError("chat preview NOTIFY payload exceeds its byte limit")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("invalid chat preview JSON") from error
    if not isinstance(value, dict) or value.get("version") not in {
        CHAT_PREVIEW_VERSION,
        CHAT_PROGRESS_VERSION,
    }:
        raise ValueError("invalid chat preview version")
    event_type = value.get("event")
    if event_type == "workflow.progress":
        return _parse_progress_payload(value)
    expected_keys = {
        "version",
        "event",
        "run_id",
        "attempt",
        "seq",
        "delta" if event_type == "answer.preview.delta" else "reason",
    }
    if set(value) != expected_keys:
        raise ValueError("invalid chat preview fields")
    attempt = value.get("attempt")
    seq = value.get("seq")
    if type(attempt) is not int or type(seq) is not int:
        raise ValueError("invalid chat preview sequence")
    try:
        run_id = UUID(value["run_id"])
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("invalid chat preview run ID") from error
    if event_type == "answer.preview.delta":
        delta = value.get("delta")
        if not isinstance(delta, str):
            raise ValueError("invalid chat preview delta")
        return ChatPreviewDelta(run_id, attempt, seq, delta)
    if event_type == "answer.preview.reset":
        try:
            reason = ChatPreviewResetReason(value.get("reason"))
        except (TypeError, ValueError) as error:
            raise ValueError("invalid chat preview reset reason") from error
        return ChatPreviewReset(run_id, attempt, seq, reason)
    raise ValueError("invalid chat preview event type")


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
        "requested_mode",
        "resolved_mode",
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
        resolved_mode = ChatResolvedMode(value.get("resolved_mode"))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("invalid chat progress enum or run ID") from error
    requested_value = value.get("requested_mode")
    try:
        requested_mode = (
            ChatWorkflowMode(requested_value)
            if requested_value is not None
            else None
        )
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
        "route_status",
        "route_reason_codes",
        "research_status",
        "covered_aspects",
        "missing_aspects",
        "conflict_count",
        "decision",
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
        route_status_value = facts_value.get("route_status")
        research_status_value = facts_value.get("research_status")
        decision_value = facts_value.get("decision")
        facts = ChatProgressFacts(
            objective=objective,
            queries=_string_tuple(facts_value.get("queries")),
            evidence_count=facts_value.get("evidence_count"),
            new_evidence_count=facts_value.get("new_evidence_count"),
            retrieval_calls=facts_value.get("retrieval_calls"),
            route_status=(
                ChatRouteStatus(route_status_value)
                if route_status_value is not None
                else None
            ),
            route_reason_codes=_enum_tuple(
                facts_value.get("route_reason_codes"), ChatRouteReason
            ),
            research_status=(
                ResearchStatus(research_status_value)
                if research_status_value is not None
                else None
            ),
            covered_aspects=_string_tuple(
                facts_value.get("covered_aspects")
            ),
            missing_aspects=_string_tuple(
                facts_value.get("missing_aspects")
            ),
            conflict_count=facts_value.get("conflict_count"),
            decision=(
                ChatProgressDecision(decision_value)
                if decision_value is not None
                else None
            ),
        )
        update = ChatProgressUpdate(
            active_stage=active_stage,
            activity=activity,
            completed_stages=completed,
            status=status,
            requested_mode=requested_mode,
            resolved_mode=resolved_mode,
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


def _delta_payload_fits(
    run_id: UUID,
    attempt: int,
    seq: int,
    delta: str,
) -> bool:
    try:
        serialize_preview_event(ChatPreviewDelta(run_id, attempt, seq, delta))
    except (UnicodeError, ValueError):
        return False
    return True


def _utf8_prefix(value: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _next_flush_timeout(
    states: dict[tuple[UUID, int], _AttemptState],
    now: float,
) -> float | None:
    deadlines = tuple(
        state.flush_at
        for state in states.values()
        if state.flush_at is not None
    )
    if not deadlines:
        return None
    return max(0.0, min(deadlines) - now)


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
