from __future__ import annotations

import asyncio
import os
import unittest
from uuid import uuid4

from rag_kb.adapters.chat_preview.pg_notify import (
    PgNotifyPreviewBroker,
    PgNotifyPreviewSink,
)
from rag_kb.domain import (
    ChatPreviewDelta,
    ChatPreviewReset,
    ChatPreviewResetReason,
)


RUNTIME_SQLALCHEMY_DSN = os.environ.get("RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN")


@unittest.skipUnless(
    RUNTIME_SQLALCHEMY_DSN,
    "database integration DSN is not configured",
)
class ChatPreviewPostgresTests(unittest.IsolatedAsyncioTestCase):
    async def test_notify_sink_reaches_listener_with_monotonic_events(self) -> None:
        assert RUNTIME_SQLALCHEMY_DSN is not None
        broker = PgNotifyPreviewBroker(
            RUNTIME_SQLALCHEMY_DSN,
            subscriber_queue_size=8,
        )
        sink = PgNotifyPreviewSink(
            RUNTIME_SQLALCHEMY_DSN,
            flush_interval_ms=10,
            max_total_bytes=4_096,
        )
        run_id = uuid4()
        subscription = await broker.subscribe(run_id)
        try:
            self.assertTrue(await broker.start())
            self.assertTrue(await sink.start())

            await sink.emit_delta(run_id=run_id, attempt=2, delta="你")
            await sink.emit_delta(run_id=run_id, attempt=2, delta="好")
            first = await asyncio.wait_for(
                subscription.next_event(),
                timeout=2,
            )

            await sink.emit_reset(
                run_id=run_id,
                attempt=2,
                reason=ChatPreviewResetReason.VALIDATION_REPAIR,
            )
            second = await asyncio.wait_for(
                subscription.next_event(),
                timeout=2,
            )
        finally:
            await subscription.close()
            await sink.close()
            await broker.close()

        self.assertEqual(first, ChatPreviewDelta(run_id, 2, 1, "你好"))
        self.assertEqual(
            second,
            ChatPreviewReset(
                run_id,
                2,
                2,
                ChatPreviewResetReason.VALIDATION_REPAIR,
            ),
        )


if __name__ == "__main__":
    unittest.main()
