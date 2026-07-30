from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime
from uuid import UUID

from rag_kb.auth import AuthContext
from rag_kb.domain import ChatPreviewDelta, ChatRun
from rag_kb.memory import empty_conversation_context, serialize_conversation_context
from rag_kb.services.chat_delivery import (
    ChatEventWatcher,
    ChatSseConnectionLimiter,
    ChatTerminalWatcher,
)


RUN_ID = UUID("01900000-0000-7000-8000-000000000701")
WORKSPACE = UUID("01900000-0000-7000-8000-000000000702")
CONTEXT = AuthContext("principal", "client", WORKSPACE)


class ChatTerminalWatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_immediate_terminal_result_never_sleeps_or_reloads(self) -> None:
        chat = _Chat()
        slept: list[float] = []
        watcher = ChatTerminalWatcher(
            chat,  # type: ignore[arg-type]
            poll_interval_seconds=1,
            jitter_ratio=0.2,
            max_duration_seconds=10,
            sleep=lambda delay: _record_sleep(slept, delay),
        )

        results = [
            item
            async for item in watcher.watch(
                CONTEXT,
                RUN_ID,
                initial=_run("completed"),
                disconnected=_connected,
            )
        ]

        self.assertEqual([item.status for item in results], ["completed"])
        self.assertEqual(chat.calls, 0)
        self.assertEqual(slept, [])

    async def test_polling_is_jittered_and_yields_only_committed_terminal(self) -> None:
        clock = _Clock()
        chat = _Chat(_run("running"), _run("completed"))
        jitters = iter((0.2, -0.2))
        watcher = ChatTerminalWatcher(
            chat,  # type: ignore[arg-type]
            poll_interval_seconds=1,
            jitter_ratio=0.2,
            max_duration_seconds=10,
            sleep=clock.sleep,
            monotonic=clock.now,
            uniform=lambda low, high: next(jitters),
        )

        results = [
            item
            async for item in watcher.watch(
                CONTEXT,
                RUN_ID,
                initial=_run("queued"),
                disconnected=_connected,
            )
        ]

        self.assertEqual([item.status for item in results], ["completed"])
        self.assertEqual(chat.calls, 2)
        self.assertEqual(clock.sleeps, [1.2, 0.8])

    async def test_disconnect_and_timeout_end_without_changing_the_run(self) -> None:
        chat = _Chat(_run("running"))
        watcher = ChatTerminalWatcher(
            chat,  # type: ignore[arg-type]
            poll_interval_seconds=1,
            jitter_ratio=0,
            max_duration_seconds=10,
        )
        disconnected = [
            item
            async for item in watcher.watch(
                CONTEXT,
                RUN_ID,
                initial=_run("queued"),
                disconnected=_disconnected,
            )
        ]
        self.assertEqual(disconnected, [])
        self.assertEqual(chat.calls, 0)

        clock = _Clock()
        timed_chat = _Chat(_run("running"))
        timed = ChatTerminalWatcher(
            timed_chat,  # type: ignore[arg-type]
            poll_interval_seconds=1,
            jitter_ratio=0,
            max_duration_seconds=2,
            sleep=clock.sleep,
            monotonic=clock.now,
        )
        results = [
            item
            async for item in timed.watch(
                CONTEXT,
                RUN_ID,
                initial=_run("queued"),
                disconnected=_connected,
            )
        ]
        self.assertEqual(results, [])
        self.assertEqual(timed_chat.calls, 1)
        self.assertEqual(clock.sleeps, [1, 1])


class ChatEventWatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_preview_arrives_before_terminal_and_terminal_is_last(self) -> None:
        terminal = _ControlledTerminalWatcher(_run("completed"))
        preview = _PreviewSubscription()
        watcher = ChatEventWatcher(terminal)  # type: ignore[arg-type]
        collected = asyncio.create_task(
            _collect_events(
                watcher,
                initial=_run("running"),
                preview=preview,
            )
        )
        event = ChatPreviewDelta(RUN_ID, 1, 1, "draft")

        await preview.queue.put(event)
        await preview.delivered.wait()
        await asyncio.sleep(0)
        terminal.ready.set()
        results = await collected

        self.assertEqual(results, [event, terminal.result])
        self.assertTrue(preview.discarded)

    async def test_immediate_terminal_discards_pending_preview(self) -> None:
        preview = _PreviewSubscription()
        await preview.queue.put(ChatPreviewDelta(RUN_ID, 1, 1, "stale"))
        watcher = ChatEventWatcher(
            ChatTerminalWatcher(
                _Chat(),  # type: ignore[arg-type]
                poll_interval_seconds=1,
                jitter_ratio=0,
                max_duration_seconds=1,
            )
        )

        results = await _collect_events(
            watcher,
            initial=_run("completed"),
            preview=preview,
        )

        self.assertEqual(results, [_run("completed")])
        self.assertTrue(preview.discarded)
        self.assertTrue(preview.queue.empty())

    async def test_preview_failure_degrades_to_terminal_only(self) -> None:
        terminal = _ControlledTerminalWatcher(_run("failed"))
        preview = _FailedPreviewSubscription()
        watcher = ChatEventWatcher(terminal)  # type: ignore[arg-type]
        collected = asyncio.create_task(
            _collect_events(
                watcher,
                initial=_run("running"),
                preview=preview,
            )
        )

        await asyncio.sleep(0)
        terminal.ready.set()
        results = await collected

        self.assertEqual(results, [terminal.result])


class ChatSseConnectionLimiterTests(unittest.IsolatedAsyncioTestCase):
    async def test_limit_is_atomic_per_principal_and_run_and_releases(self) -> None:
        limiter = ChatSseConnectionLimiter(2)

        self.assertTrue(await limiter.acquire("principal", RUN_ID))
        self.assertTrue(await limiter.acquire("principal", RUN_ID))
        self.assertFalse(await limiter.acquire("principal", RUN_ID))
        self.assertTrue(await limiter.acquire("other", RUN_ID))
        self.assertEqual(await limiter.active("principal", RUN_ID), 2)

        await limiter.release("principal", RUN_ID)
        self.assertTrue(await limiter.acquire("principal", RUN_ID))
        await limiter.release("principal", RUN_ID)
        await limiter.release("principal", RUN_ID)
        await limiter.release("other", RUN_ID)
        self.assertEqual(await limiter.active("principal", RUN_ID), 0)


class _Chat:
    def __init__(self, *results: ChatRun) -> None:
        self.results = list(results)
        self.calls = 0

    async def get_run(self, context, run_id):
        self.calls += 1
        self.assert_scope = (context, run_id)
        if not self.results:
            raise AssertionError("unexpected status reload")
        return self.results.pop(0)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.value

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.value += delay


class _ControlledTerminalWatcher:
    def __init__(self, result: ChatRun) -> None:
        self.result = result
        self.ready = asyncio.Event()

    async def watch(self, *args, **kwargs):
        del args, kwargs
        await self.ready.wait()
        yield self.result


class _PreviewSubscription:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[ChatPreviewDelta] = asyncio.Queue()
        self.discarded = False
        self.delivered = asyncio.Event()

    async def next_event(self) -> ChatPreviewDelta:
        event = await self.queue.get()
        self.delivered.set()
        return event

    def discard_pending(self) -> None:
        self.discarded = True
        while not self.queue.empty():
            self.queue.get_nowait()

    async def close(self) -> None:
        pass


class _FailedPreviewSubscription(_PreviewSubscription):
    async def next_event(self) -> ChatPreviewDelta:
        raise RuntimeError("preview unavailable")


async def _collect_events(
    watcher: ChatEventWatcher,
    *,
    initial: ChatRun,
    preview,
) -> list[object]:
    return [
        event
        async for event in watcher.watch(
            CONTEXT,
            RUN_ID,
            initial=initial,
            disconnected=_connected,
            preview=preview,
        )
    ]


async def _record_sleep(values: list[float], delay: float) -> None:
    values.append(delay)


async def _connected() -> bool:
    return False


async def _disconnected() -> bool:
    return True


def _run(status: str) -> ChatRun:
    now = datetime(2026, 7, 15, tzinfo=UTC)
    assistant_status = "completed" if status == "completed" else "generating"
    return ChatRun(
        id=RUN_ID,
        workspace_id=WORKSPACE,
        kb_id=UUID("01900000-0000-7000-8000-000000000703"),
        session_id=UUID("01900000-0000-7000-8000-000000000704"),
        user_message_id=UUID("01900000-0000-7000-8000-000000000705"),
        assistant_message_id=UUID("01900000-0000-7000-8000-000000000706"),
        index_revision_id=UUID("01900000-0000-7000-8000-000000000707"),
        status=status,
        principal_id="principal",
        client_id="client",
        endpoint="POST /api/v1/chat/runs",
        idempotency_key=UUID("01900000-0000-7000-8000-000000000708"),
        request_hash="sha256:" + "1" * 64,
        requested_policy={},
        effective_policy={},
        retrieval_strategy={},
        model_configuration={},
        assistant_status=assistant_status,
        assistant_content="answer" if status == "completed" else "",
        citations=(),
        attempt=1,
        error_code=None,
        error_detail=None,
        error_retryable=None,
        usage=None,
        timing=None,
        created_at=now,
        updated_at=now,
        completed_at=now if status == "completed" else None,
        conversation_context=serialize_conversation_context(
            empty_conversation_context()
        ),
    )
