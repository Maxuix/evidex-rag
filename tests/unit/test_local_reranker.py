from __future__ import annotations

import unittest
from types import SimpleNamespace
from uuid import UUID

import numpy as np

from rag_kb.adapters.local_reranker import (
    LOCAL_RERANKER_MAX_SEQUENCE_LENGTH,
    LocalMiniLmReranker,
    _sliding_windows,
    _table_windows,
    build_local_rerank_windows,
)
from rag_kb.domain import RerankDocument
from rag_kb.ports.model_api import RerankerAdapterError


CHUNK_1 = UUID("01900000-0000-7000-8000-000000000101")
CHUNK_2 = UUID("01900000-0000-7000-8000-000000000102")


class _CharacterTokenizer:
    model_max_length = 512
    cls_token_id = 0
    sep_token_id = 2
    pad_token_id = 1

    def __init__(self) -> None:
        self.backend_tokenizer = self

    def encode(self, text: str, *, add_special_tokens: bool) -> SimpleNamespace:
        assert not add_special_tokens
        return SimpleNamespace(ids=[10 + ord(value) for value in text])

    @staticmethod
    def num_special_tokens_to_add(*, pair: bool) -> int:
        return 4 if pair else 2


class _FakeSession:
    def __init__(self, logits: tuple[float, ...]) -> None:
        self._logits = list(logits)

    def run(self, _outputs, inputs):
        count = inputs["input_ids"].shape[0]
        values = self._logits[:count]
        del self._logits[:count]
        return [np.asarray(values, dtype=np.float32).reshape(-1, 1)]


class _ExplodingSession:
    def run(self, _outputs, _inputs):
        raise _ModelRuntimeFailure


class _ModelRuntimeFailure(Exception):
    pass


class LocalRerankerWindowTests(unittest.TestCase):
    def test_document_context_does_not_displace_hierarchy_or_drop_body(self):
        tokenizer = _CharacterTokenizer()
        document = RerankDocument(CHUNK_1, "正文" * 600, {"titles": [{"text": "损益表"}]},
                                  document_context="Boeing 2022 annual report" * 10)
        windows = build_local_rerank_windows(tokenizer, "毛利率", (document,))
        decoded = ["".join(chr(token - 10) for token in window.input_ids if token >= 10) for window in windows]
        self.assertTrue(all("Boeing 2022" in value and "损益表" in value for value in decoded))
        self.assertTrue(all(len(window.input_ids) <= 512 for window in windows))
        self.assertEqual(document.text, "正文" * 600)

    def test_caption_and_multirow_years_repeat_with_table_rows(self):
        tokenizer = _CharacterTokenizer()
        header = "Annual sales (USD millions)\n| Item | A | B |\n| --- | --- | --- |\n| | Years Ended | |\n| | 2022 | 2021 |\n"
        rows = [f"| product{index} | 66608 | 62286 |" for index in range(30)]
        windows = _table_windows(tokenizer, header + "\n".join(rows), 300, 32)
        decoded = ["".join(chr(token - 10) for token in value) for value in windows]
        self.assertGreater(len(windows), 1)
        self.assertTrue(all(value.startswith(header) for value in decoded))
        # Every financial row must still occur intact in at least one window.
        self.assertTrue(all(any(row in value for value in decoded) for row in rows))

    def test_oversized_header_is_preserved_instead_of_silently_truncated(self):
        tokenizer = _CharacterTokenizer()
        text = "| " + "TITLE" * 50 + " | value |\n| --- | --- |\n| last row | 987 |"
        windows = _table_windows(tokenizer, text, 100, 32)
        self.assertGreater(len(windows), 2)
        expected = tuple(tokenizer.encode(text, add_special_tokens=False).ids)
        self.assertTrue(all(len(value) <= 100 for value in windows))
        self.assertTrue(set(expected).issubset(set().union(*(set(value) for value in windows))))
        self.assertTrue(any("last row | 987" in "".join(chr(token - 10) for token in value) for value in windows))

    def test_long_document_pair_windows_never_exceed_model_limit(self) -> None:
        tokenizer = _CharacterTokenizer()
        windows = build_local_rerank_windows(
            tokenizer,
            "问" * 160,
            (
                RerankDocument(
                    CHUNK_1,
                    ("第一段" * 180) + "\n\n" + ("第二段" * 180),
                    {"titles": [{"text": "章节" * 30}]},
                ),
            ),
        )

        self.assertGreater(len(windows), 1)
        self.assertEqual(
            [item.window_index for item in windows],
            list(range(len(windows))),
        )
        self.assertTrue(
            all(
                len(item.input_ids) <= LOCAL_RERANKER_MAX_SEQUENCE_LENGTH
                and len(item.input_ids) == len(item.attention_mask)
                for item in windows
            )
        )
        # XLM-R pair format is <s> query </s></s> passage </s>; query is capped.
        self.assertEqual(windows[0].input_ids[0], tokenizer.cls_token_id)
        self.assertEqual(windows[0].input_ids[97:99], (2, 2))

    def test_sliding_windows_cover_every_token_with_overlap(self) -> None:
        token_ids = tuple(range(1_000))
        windows = _sliding_windows(
            token_ids,
            boundaries=(300, 600, 900, 1_000),
            window_size=400,
            overlap=64,
        )

        self.assertEqual(set().union(*(set(item) for item in windows)), set(token_ids))
        self.assertTrue(all(len(item) <= 400 for item in windows))
        for left, right in zip(windows, windows[1:]):
            self.assertGreaterEqual(len(set(left) & set(right)), 64)

    def test_table_windows_repeat_markdown_header(self) -> None:
        tokenizer = _CharacterTokenizer()
        text = "| 项目 | 值 |\n| --- | --- |\n" + "\n".join(
            f"| 第{index}行 | {'内容' * 12} |" for index in range(30)
        )
        windows = _table_windows(tokenizer, text, window_size=160, overlap=32)
        header = tuple(
            tokenizer.encode(
                "| 项目 | 值 |\n| --- | --- |\n",
                add_special_tokens=False,
            ).ids
        )

        self.assertGreater(len(windows), 1)
        self.assertTrue(all(item[: len(header)] == header for item in windows))
        self.assertTrue(all(len(item) <= 160 for item in windows))


class LocalRerankerAggregationTests(unittest.IsolatedAsyncioTestCase):
    async def test_adapter_uses_max_window_logit_per_original_chunk(self) -> None:
        tokenizer = _CharacterTokenizer()
        documents = (
            RerankDocument(CHUNK_1, "长文" * 260, {}),
            RerankDocument(CHUNK_2, "短文", {}),
        )
        windows = build_local_rerank_windows(tokenizer, "问题", documents)
        first_count = sum(
            item.index_chunk_id == CHUNK_1 for item in windows
        )
        logits = tuple(
            [-3.0] * (first_count - 1) + [2.0] + [-1.0]
        )
        reranker = LocalMiniLmReranker()
        reranker._runtime = (tokenizer, _FakeSession(logits))  # noqa: SLF001

        scores = await reranker.score("问题", documents)

        self.assertEqual(scores[0].window_count, first_count)
        self.assertEqual(scores[0].raw_logit, 2.0)
        self.assertEqual(scores[0].winning_window_index, first_count - 1)
        self.assertEqual(scores[1].window_count, 1)
        self.assertGreater(scores[0].score, scores[1].score)

    async def test_unexpected_runtime_failure_is_wrapped_safely(self) -> None:
        reranker = LocalMiniLmReranker()
        reranker._runtime = (  # noqa: SLF001
            _CharacterTokenizer(),
            _ExplodingSession(),
        )

        with self.assertRaisesRegex(
            RerankerAdapterError,
            "local_reranker_inference",
        ):
            await reranker.score(
                "问题",
                (RerankDocument(CHUNK_1, "正文", {}),),
            )


if __name__ == "__main__":
    unittest.main()
