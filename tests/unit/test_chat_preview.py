from __future__ import annotations

import unittest
from uuid import uuid4

from rag_kb.answering.preview import (
    NoOpChatPreviewSink,
    PartialAnswerPreviewProjector,
    emit_preview_delta_safely,
    emit_preview_reset_safely,
)
from rag_kb.domain.chat_preview import ChatPreviewResetReason


class ChatPreviewProjectionTests(unittest.TestCase):
    def test_projects_monotonic_answer_claim_suffixes(self) -> None:
        projector = PartialAnswerPreviewProjector(max_visible_bytes=1024)

        first = projector.feed(
            '{"outcome":"answered","claims":[{"text":"你好'
        )
        second = projector.feed(
            '{"outcome":"answered","claims":[{"text":"你好，世界",'
        )
        third = projector.feed(
            '{"outcome":"answered","claims":[{"text":"你好，世界",'
            '"citation_ids":["cite_1"]},{"text":"第二段"}]'
        )

        self.assertEqual(first.delta, "你好")
        self.assertEqual(second.delta, "，世界")
        self.assertEqual(third.delta, "\n\n第二段")
        self.assertEqual(projector.published, "你好，世界\n\n第二段")

    def test_decodes_json_escapes_before_projection(self) -> None:
        projector = PartialAnswerPreviewProjector(max_visible_bytes=1024)
        result = projector.feed(
            '{"outcome":"partial","claims":[{"text":"第一行\\n'
            '第二行：\\\"规则\\\"与\\\\路径 😀"}]'
        )

        self.assertEqual(result.delta, '第一行\n第二行："规则"与\\路径 😀')

    def test_waits_for_partial_outcome_prefix(self) -> None:
        projector = PartialAnswerPreviewProjector(max_visible_bytes=1024)

        pending = projector.feed('{"outcome":"ans')
        ready = projector.feed(
            '{"outcome":"answered","claims":[{"text":"可见"}]'
        )

        self.assertIsNone(pending.delta)
        self.assertFalse(pending.invalidated)
        self.assertEqual(ready.delta, "可见")

    def test_refusal_and_acknowledgement_never_preview(self) -> None:
        for outcome in ("refused", "acknowledged"):
            with self.subTest(outcome=outcome):
                projector = PartialAnswerPreviewProjector(max_visible_bytes=1024)
                result = projector.feed(
                    f'{{"outcome":"{outcome}","claims":[]}}'
                )
                self.assertIsNone(result.delta)
                self.assertFalse(result.invalidated)

    def test_non_monotonic_or_invalid_shape_invalidates_published_preview(self) -> None:
        for invalid in (
            '{"outcome":"answered","claims":[{"text":"另一个"}]}',
            '{"outcome":"answered","claims":"not-a-list"}',
            '{"outcome":"answered","claims":[{"text":7}]}',
        ):
            with self.subTest(invalid=invalid):
                projector = PartialAnswerPreviewProjector(max_visible_bytes=1024)
                projector.feed(
                    '{"outcome":"answered","claims":[{"text":"原文"}]'
                )
                result = projector.feed(invalid)
                self.assertTrue(result.invalidated)
                self.assertTrue(projector.disabled)

    def test_limit_invalidates_only_after_content_was_visible(self) -> None:
        projector = PartialAnswerPreviewProjector(max_visible_bytes=3)
        first = projector.feed(
            '{"outcome":"answered","claims":[{"text":"abc"}]'
        )
        invalid = projector.feed(
            '{"outcome":"answered","claims":[{"text":"abcd"}]'
        )

        self.assertEqual(first.delta, "abc")
        self.assertTrue(invalid.invalidated)

    def test_invalid_unicode_surrogate_disables_preview_without_escaping(self) -> None:
        projector = PartialAnswerPreviewProjector(max_visible_bytes=1024)
        projector.feed(
            '{"outcome":"answered","claims":[{"text":"visible"}]'
        )

        invalid = projector.feed(
            '{"outcome":"answered","claims":[{"text":"visible\\ud800"}]'
        )

        self.assertTrue(invalid.invalidated)
        self.assertTrue(projector.disabled)


class _FailingSink:
    enabled = True

    async def emit_delta(self, **values: object) -> None:
        del values
        raise RuntimeError("content must not escape through the error")

    async def emit_reset(self, **values: object) -> None:
        del values
        raise RuntimeError("content must not escape through the error")


class ChatPreviewSinkTests(unittest.IsolatedAsyncioTestCase):
    async def test_noop_sink_is_disabled(self) -> None:
        sink = NoOpChatPreviewSink()
        self.assertFalse(sink.enabled)
        await sink.emit_delta(run_id=uuid4(), attempt=1, delta="x")
        await sink.emit_reset(
            run_id=uuid4(),
            attempt=1,
            reason=ChatPreviewResetReason.PREVIEW_INVALID,
        )

    async def test_safe_emitters_swallow_sink_failures(self) -> None:
        sink = _FailingSink()
        run_id = uuid4()

        self.assertFalse(
            await emit_preview_delta_safely(
                sink,
                run_id=run_id,
                attempt=1,
                delta="secret",
            )
        )
        self.assertFalse(
            await emit_preview_reset_safely(
                sink,
                run_id=run_id,
                attempt=1,
                reason=ChatPreviewResetReason.GENERATION_FAILED,
            )
        )
