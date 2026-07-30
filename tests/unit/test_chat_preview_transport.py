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
    serialize_preview_delta_payloads,
    serialize_preview_event,
)
from rag_kb.domain import (
    ChatPreviewDelta,
    ChatPreviewReset,
    ChatPreviewResetReason,
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


class ChatPreviewPayloadTests(unittest.TestCase):
    def test_delta_and_reset_round_trip_through_strict_payload(self) -> None:
        run_id = uuid4()
        events = (
            ChatPreviewDelta(run_id, 2, 3, "中文 delta"),
            ChatPreviewReset(
                run_id,
                2,
                4,
                ChatPreviewResetReason.VALIDATION_REPAIR,
            ),
        )

        for event in events:
            with self.subTest(event=event):
                payload = serialize_preview_event(event)
                self.assertLessEqual(
                    len(payload.encode("utf-8")),
                    MAX_NOTIFY_PAYLOAD_BYTES,
                )
                self.assertEqual(parse_preview_payload(payload), event)

    def test_large_unicode_delta_is_split_without_cutting_code_points(self) -> None:
        run_id = uuid4()
        delta = "回答🙂" * 2_000

        payloads = serialize_preview_delta_payloads(
            run_id=run_id,
            attempt=1,
            starting_seq=1,
            delta=delta,
        )
        events = tuple(parse_preview_payload(payload) for payload in payloads)

        self.assertGreater(len(events), 1)
        self.assertEqual(
            "".join(event.delta for event in events),  # type: ignore[attr-defined]
            delta,
        )
        self.assertEqual(
            [event.seq for event in events],
            list(range(1, len(events) + 1)),
        )
        self.assertTrue(
            all(
                len(payload.encode("utf-8")) <= MAX_NOTIFY_PAYLOAD_BYTES
                for payload in payloads
            )
        )

    def test_unknown_extra_or_invalid_sequence_fields_fail_closed(self) -> None:
        payload = json.loads(
            serialize_preview_event(
                ChatPreviewDelta(uuid4(), 1, 1, "safe")
            )
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


class ChatPreviewTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_sink_batches_delta_and_uses_one_dedicated_connection(self) -> None:
        connection = _Connection()
        connector = _Connector(connection)
        sink = PgNotifyPreviewSink(
            _DSN,
            flush_interval_ms=10,
            max_total_bytes=1024,
            connect=connector,
        )
        run_id = uuid4()
        try:
            self.assertTrue(await sink.start())
            await sink.emit_delta(run_id=run_id, attempt=1, delta="你")
            await sink.emit_delta(run_id=run_id, attempt=1, delta="好")
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
            ChatPreviewDelta(run_id, 1, 1, "你好"),
        )
        self.assertTrue(connection.closed)

    async def test_sink_truncates_at_total_limit_without_splitting_unicode(
        self,
    ) -> None:
        connection = _Connection()
        sink = PgNotifyPreviewSink(
            _DSN,
            flush_interval_ms=10,
            max_total_bytes=4,
            connect=_Connector(connection),
        )
        run_id = uuid4()
        try:
            await sink.start()
            await sink.emit_delta(run_id=run_id, attempt=1, delta="中文")
            await sink.emit_delta(run_id=run_id, attempt=1, delta="ignored")
            await asyncio.sleep(0.03)
        finally:
            await sink.close()

        event = parse_preview_payload(connection.executions[0][2])
        self.assertEqual(event, ChatPreviewDelta(run_id, 1, 1, "中"))

    async def test_sink_connection_failure_is_best_effort(self) -> None:
        sink = PgNotifyPreviewSink(
            _DSN,
            flush_interval_ms=5,
            max_total_bytes=1024,
            connect=_Connector(error=OSError("database unavailable")),
        )
        try:
            self.assertFalse(await sink.start())
            await sink.emit_delta(
                run_id=uuid4(),
                attempt=1,
                delta="still does not fail generation",
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
                serialize_preview_event(
                    ChatPreviewDelta(run_id, 1, 1, "first")
                )
            )
            connection.notify(
                serialize_preview_event(
                    ChatPreviewDelta(run_id, 1, 2, "dropped")
                )
            )

            self.assertEqual(
                await asyncio.wait_for(first.next_event(), timeout=0.1),
                ChatPreviewDelta(run_id, 1, 1, "first"),
            )
            self.assertEqual(
                await asyncio.wait_for(second.next_event(), timeout=0.1),
                ChatPreviewDelta(run_id, 1, 1, "first"),
            )
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(other.next_event(), timeout=0.01)
        finally:
            await first.close()
            await second.close()
            await other.close()
            await broker.close()

        self.assertTrue(connection.closed)

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
                serialize_preview_event(
                    ChatPreviewDelta(run_id, 1, 1, "ignored")
                )
            )
        finally:
            await broker.close()


if __name__ == "__main__":
    unittest.main()
