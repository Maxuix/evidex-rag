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
    ChatProgressActivity,
    ChatProgressSnapshot,
    ChatProgressStage,
    ChatProgressUpdate,
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
        sink = PgNotifyPreviewSink(RUNTIME_SQLALCHEMY_DSN)
        run_id = uuid4()
        subscription = await broker.subscribe(run_id)
        first_update = ChatProgressUpdate(
            ChatProgressStage.UNDERSTAND_QUERY,
            ChatProgressActivity.LOAD_CONTEXT,
        )
        second_update = ChatProgressUpdate(
            ChatProgressStage.RETRIEVE_EVIDENCE,
            ChatProgressActivity.TOOL_DECISION,
            completed_stages=(ChatProgressStage.UNDERSTAND_QUERY,),
        )
        try:
            self.assertTrue(await broker.start())
            self.assertTrue(await sink.start())

            await sink.emit_progress(
                run_id=run_id, attempt=2, update=first_update
            )
            first = await asyncio.wait_for(
                subscription.next_event(),
                timeout=2,
            )

            await sink.emit_progress(
                run_id=run_id, attempt=2, update=second_update
            )
            second = await asyncio.wait_for(
                subscription.next_event(),
                timeout=2,
            )
        finally:
            await subscription.close()
            await sink.close()
            await broker.close()

        self.assertEqual(
            first,
            ChatProgressSnapshot(run_id, 2, 1, first_update),
        )
        self.assertEqual(
            second,
            ChatProgressSnapshot(run_id, 2, 2, second_update),
        )


if __name__ == "__main__":
    unittest.main()
