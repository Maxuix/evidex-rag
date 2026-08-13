from __future__ import annotations

import asyncio
import json
import unittest
from uuid import uuid4

from rag_kb.adapters.chat_preview.pg_notify import (
    CHAT_PREVIEW_CHANNEL,
    MAX_NOTIFY_PAYLOAD_BYTES,
    PgNotifyPreviewBroker,
    PgNotifyPreviewSink,
    parse_preview_payload,
    serialize_preview_event,
)
from rag_kb.domain import (
    ChatProgressActivity,
    ChatProgressFacts,
    ChatProgressSnapshot,
    ChatProgressStage,
    ChatProgressUpdate,
)


_DSN = "postgresql+asyncpg://rag_kb_runtime:secret@localhost/rag_kb"


class _Connection:
    def __init__(self) -> None:
        self.executions: list[tuple[str, str, str]] = []
        self.listeners: dict[str, object] = {}
        self.closed = False

    def is_closed(self) -> bool:
        return self.closed

    async def execute(self, query: str, channel: str, payload: str) -> None:
        self.executions.append((query, channel, payload))

    async def add_listener(self, channel: str, callback) -> None:
        self.listeners[channel] = callback

    async def remove_listener(self, channel: str, callback) -> None:
        if self.listeners.get(channel) == callback:
            self.listeners.pop(channel)

    def add_termination_listener(self, callback) -> None:
        del callback

    async def close(self, *, timeout: float) -> None:
        del timeout
        self.closed = True

    def notify(self, payload: str) -> None:
        callback = self.listeners[CHAT_PREVIEW_CHANNEL]
        callback(self, 1, CHAT_PREVIEW_CHANNEL, payload)


class _Connector:
    def __init__(
        self,
        *connections: _Connection,
        error: Exception | None = None,
    ) -> None:
        self.connections = list(connections)
        self.error = error
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def __call__(self, dsn: str, **kwargs: object) -> _Connection:
        self.calls.append((dsn, kwargs))
        if self.error is not None:
            raise self.error
        if not self.connections:
            raise AssertionError("unexpected connection")
        return self.connections.pop(0)


class _RecoveringConnector:
    def __init__(self, connection: _Connection, *, failures: int) -> None:
        self.connection = connection
        self.failures = failures
        self.calls = 0

    async def __call__(self, dsn: str, **kwargs: object) -> _Connection:
        del dsn, kwargs
        self.calls += 1
        if self.calls <= self.failures:
            raise OSError("database unavailable")
        return self.connection


async def _yield_without_delay(delay: float) -> None:
    del delay
    await asyncio.sleep(0)


def _progress(
    run_id,
    *,
    attempt: int = 1,
    seq: int = 1,
    update: ChatProgressUpdate | None = None,
) -> ChatProgressSnapshot:
    return ChatProgressSnapshot(
        run_id,
        attempt,
        seq,
        update
        or ChatProgressUpdate(
            ChatProgressStage.UNDERSTAND_QUERY,
            ChatProgressActivity.LOAD_CONTEXT,
        ),
    )


class ChatPreviewPayloadTests(unittest.TestCase):
    def test_unknown_extra_or_invalid_sequence_fields_fail_closed(self) -> None:
        payload = json.loads(
            serialize_preview_event(_progress(uuid4()))
        )
        cases = (
            {**payload, "extra": "forbidden"},
            {**payload, "attempt": True},
            {**payload, "version": "future"},
            {**payload, "event": "unknown"},
        )
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                parse_preview_payload(
                    json.dumps(case, separators=(",", ":"))
                )

    def test_progress_snapshot_round_trips_allowlisted_bounded_facts(self) -> None:
        run_id = uuid4()
        event = ChatProgressSnapshot(
            run_id,
            2,
            4,
            ChatProgressUpdate(
                active_stage=ChatProgressStage.RETRIEVE_EVIDENCE,
                activity=ChatProgressActivity.SEARCH_KNOWLEDGE_BASE,
                completed_stages=(
                    ChatProgressStage.UNDERSTAND_QUERY,
                ),
                facts=ChatProgressFacts(
                    objective="查找负责人与期限",
                    queries=("负责人", "截止日期"),
                    evidence_count=3,
                ),
            ),
        )

        payload = serialize_preview_event(event)

        self.assertLessEqual(len(payload.encode("utf-8")), MAX_NOTIFY_PAYLOAD_BYTES)
        self.assertEqual(parse_preview_payload(payload), event)
        self.assertNotIn("evidence_text", json.loads(payload)["facts"])

    def test_progress_payload_rejects_unknown_fact_and_overlong_text(self) -> None:
        event = ChatProgressSnapshot(
            uuid4(),
            1,
            1,
            ChatProgressUpdate(
                ChatProgressStage.RETRIEVE_EVIDENCE,
                ChatProgressActivity.TOOL_DECISION,
            ),
        )
        payload = json.loads(serialize_preview_event(event))
        payload["facts"]["evidence_excerpt"] = "must not cross the boundary"
        with self.assertRaises(ValueError):
            parse_preview_payload(json.dumps(payload))
        with self.assertRaises(ValueError):
            ChatProgressFacts(objective="x" * 161)


class ChatPreviewTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_sink_publishes_progress_and_uses_one_dedicated_connection(
        self,
    ) -> None:
        connection = _Connection()
        connector = _Connector(connection)
        sink = PgNotifyPreviewSink(_DSN, connect=connector)
        run_id = uuid4()
        update = ChatProgressUpdate(
            ChatProgressStage.UNDERSTAND_QUERY,
            ChatProgressActivity.LOAD_CONTEXT,
        )
        try:
            self.assertTrue(await sink.start())
            await sink.emit_progress(run_id=run_id, attempt=1, update=update)
            await asyncio.sleep(0.03)
        finally:
            await sink.close()

        self.assertEqual(len(connector.calls), 1)
        self.assertTrue(connector.calls[0][0].startswith("postgresql://"))
        self.assertEqual(len(connection.executions), 1)
        query, channel, payload = connection.executions[0]
        self.assertEqual(query, "SELECT pg_notify($1, $2)")
        self.assertEqual(channel, CHAT_PREVIEW_CHANNEL)
        self.assertEqual(
            parse_preview_payload(payload),
            _progress(run_id, update=update),
        )
        self.assertTrue(connection.closed)

    async def test_sink_publishes_full_progress_snapshots_with_own_sequence(self) -> None:
        connection = _Connection()
        sink = PgNotifyPreviewSink(
            _DSN,
            connect=_Connector(connection),
        )
        run_id = uuid4()
        first = ChatProgressUpdate(
            ChatProgressStage.UNDERSTAND_QUERY,
            ChatProgressActivity.LOAD_CONTEXT,
        )
        second = ChatProgressUpdate(
            ChatProgressStage.RETRIEVE_EVIDENCE,
            ChatProgressActivity.TOOL_DECISION,
            completed_stages=(ChatProgressStage.UNDERSTAND_QUERY,),
        )
        try:
            await sink.start()
            await sink.emit_progress(run_id=run_id, attempt=1, update=first)
            await sink.emit_progress(run_id=run_id, attempt=1, update=second)
            await asyncio.sleep(0.02)
        finally:
            await sink.close()

        events = [parse_preview_payload(item[2]) for item in connection.executions]
        self.assertEqual([item.seq for item in events], [1, 2])
        self.assertEqual(events[1].update, second)

    async def test_sink_connection_failure_is_best_effort(self) -> None:
        sink = PgNotifyPreviewSink(
            _DSN,
            connect=_Connector(error=OSError("database unavailable")),
        )
        try:
            self.assertFalse(await sink.start())
            await sink.emit_progress(
                run_id=uuid4(),
                attempt=1,
                update=ChatProgressUpdate(
                    ChatProgressStage.UNDERSTAND_QUERY,
                    ChatProgressActivity.LOAD_CONTEXT,
                ),
            )
            await asyncio.sleep(0.02)
        finally:
            await sink.close()

    async def test_broker_fans_out_by_run_and_drops_queue_overflow(self) -> None:
        connection = _Connection()
        broker = PgNotifyPreviewBroker(
            _DSN,
            subscriber_queue_size=1,
            connect=_Connector(connection),
        )
        run_id = uuid4()
        other_run_id = uuid4()
        first = await broker.subscribe(run_id)
        second = await broker.subscribe(run_id)
        other = await broker.subscribe(other_run_id)
        try:
            self.assertTrue(await broker.start())
            connection.notify(
                serialize_preview_event(_progress(run_id, seq=1))
            )
            connection.notify(
                serialize_preview_event(_progress(run_id, seq=2))
            )

            self.assertEqual(
                await asyncio.wait_for(first.next_event(), timeout=0.1),
                _progress(run_id, seq=1),
            )
            self.assertEqual(
                await asyncio.wait_for(second.next_event(), timeout=0.1),
                _progress(run_id, seq=1),
            )
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(other.next_event(), timeout=0.01)
        finally:
            await first.close()
            await second.close()
            await other.close()
            await broker.close()

        self.assertTrue(connection.closed)

    async def test_broker_keeps_reconnecting_after_initial_backoff_window(
        self,
    ) -> None:
        connection = _Connection()
        connector = _RecoveringConnector(connection, failures=5)
        broker = PgNotifyPreviewBroker(
            _DSN,
            subscriber_queue_size=2,
            connect=connector,
            sleep=_yield_without_delay,
        )
        try:
            self.assertFalse(await broker.start())
            for _ in range(20):
                if CHAT_PREVIEW_CHANNEL in connection.listeners:
                    break
                await asyncio.sleep(0)

            self.assertIn(CHAT_PREVIEW_CHANNEL, connection.listeners)
            self.assertEqual(connector.calls, 6)
        finally:
            await broker.close()

    async def test_broker_drops_malformed_payload_and_unregisters(self) -> None:
        connection = _Connection()
        broker = PgNotifyPreviewBroker(
            _DSN,
            subscriber_queue_size=2,
            connect=_Connector(connection),
        )
        run_id = uuid4()
        subscription = await broker.subscribe(run_id)
        try:
            await broker.start()
            connection.notify('{"version":"chat_preview_v1","delta":"secret"}')
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(
                    subscription.next_event(),
                    timeout=0.01,
                )
            await subscription.close()
            connection.notify(
                serialize_preview_event(_progress(run_id))
            )
        finally:
            await broker.close()


if __name__ == "__main__":
    unittest.main()
