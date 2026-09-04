from __future__ import annotations

import json
import unittest
from uuid import uuid4

from rag_kb.domain import (
    AUTO_QA_QUESTIONS_PER_CHUNK,
    ChatModelResponse,
    ChatToolCall,
    ContentModality,
    EmbeddingBatch,
    ErrorCode,
    IndexChunkWrite,
    IndexingExecutionError,
    IndexingPhase,
)
from rag_kb.indexing.auto_qa import (
    embed_auto_qa_questions,
    eligible_auto_qa_chunks,
    generate_auto_qa_questions,
    is_auto_qa_eligible,
    normalize_question,
    title_path,
    validate_auto_qa_items,
)
from rag_kb.indexing.pipeline import _lexical_rows
from rag_kb.document_processing.lexical import analyze_document


def _chunk(*, content: str, embedding_text: str | None = None, ordinal: int = 0):
    return IndexChunkWrite(
        id=uuid4(),
        ordinal=ordinal,
        content=content,
        content_hash="0" * 64,
        token_count=8,
        source_location={},
        hierarchy={"titles": [{"depth": 1, "text": "安装"}]},
        source_metadata={},
        unit_key=f"unit-{ordinal}",
        modality=ContentModality.TEXT,
        embedding_text=embedding_text,
        embedding_text_hash="1" * 64 if embedding_text else None,
    )


class AutoQAGenerationTests(unittest.TestCase):
    def test_disabled_eligibility_skips_empty_image_text(self) -> None:
        text = _chunk(content="如何安装客户端", embedding_text="如何安装客户端")
        image = _chunk(content="", embedding_text=None, ordinal=1)
        self.assertTrue(is_auto_qa_eligible(embedding_text=text.embedding_text, content=text.content))
        self.assertFalse(is_auto_qa_eligible(embedding_text=image.embedding_text, content=image.content))
        self.assertEqual(eligible_auto_qa_chunks((text, image)), (text,))

    def test_normalize_strips_list_markers_and_noise(self) -> None:
        self.assertEqual(normalize_question("  1. 如何安装？  "), "如何安装？")
        self.assertEqual(normalize_question("• 什么是配额。"), "什么是配额")

    def test_validate_requires_exact_five_unique_questions_per_ref(self) -> None:
        payload = {
            "items": [
                {
                    "ref": "c01",
                    "questions": [
                        "如何安装？",
                        "安装需要什么？",
                        "安装失败怎么办？",
                        "支持哪些系统？",
                        "如何升级？",
                    ],
                }
            ]
        }
        result = validate_auto_qa_items(payload, ("c01",))
        self.assertEqual(len(result["c01"]), AUTO_QA_QUESTIONS_PER_CHUNK)

        with self.assertRaises(IndexingExecutionError) as missing:
            validate_auto_qa_items({"items": []}, ("c01",))
        self.assertEqual(missing.exception.code, ErrorCode.AUTO_QA_RESPONSE_INVALID)
        self.assertEqual(missing.exception.phase, IndexingPhase.AUTO_QA_GENERATION)

        with self.assertRaises(IndexingExecutionError) as unknown:
            validate_auto_qa_items(payload, ("c02",))
        self.assertEqual(unknown.exception.diagnostic["check"], "unknown_ref")

        with self.assertRaises(IndexingExecutionError):
            validate_auto_qa_items(
                {
                    "items": [
                        {
                            "ref": "c01",
                            "questions": ["如何安装？"] * 5,
                        }
                    ]
                },
                ("c01",),
            )

        with self.assertRaises(IndexingExecutionError):
            validate_auto_qa_items(
                {
                    "items": [
                        {
                            "ref": "c01",
                            "questions": [
                                "如何安装？",
                                "安装需要什么？",
                                "安装失败怎么办？",
                                "支持哪些系统？",
                                "",
                            ],
                        }
                    ]
                },
                ("c01",),
            )

        with self.assertRaises(IndexingExecutionError):
            validate_auto_qa_items({"items": payload["items"], "extra": True}, ("c01",))

    def test_title_path_is_bounded(self) -> None:
        self.assertEqual(title_path({"titles": [{"text": "安装"}]}), ("安装",))

    def test_disabled_lexical_hash_matches_baseline(self) -> None:
        chunk = _chunk(content="安装客户端", embedding_text="安装客户端")
        baseline = analyze_document("安装客户端")
        assert baseline is not None
        rows = _lexical_rows((chunk,), frozenset({chunk.id}), {})
        self.assertEqual(rows[0].lexical_text_hash, baseline.lexical_text_hash)

    def test_enhanced_lexical_includes_question_terms_without_duplicating(self) -> None:
        chunk = _chunk(content="安装客户端", embedding_text="安装客户端")
        questions = (
            "如何申请配额",
            "配额上限是多少",
            "谁负责审批",
            "申请失败怎么办",
            "如何查询额度",
        )
        rows = _lexical_rows((chunk,), frozenset({chunk.id}), {chunk.id: questions})
        self.assertIn("配额", rows[0].lexical_text)
        self.assertEqual(rows[0].lexical_text.split().count("配额"), 1)


class AutoQAChatContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_forced_tool_output_is_validated_and_normalized(self) -> None:
        chunk = _chunk(content="安装客户端", embedding_text="安装客户端")
        chat = _ChatModel()
        generated, usage, calls = await generate_auto_qa_questions(
            chat,
            (chunk,),
            model_profile_revision_id=uuid4(),
        )
        self.assertEqual(calls, 1)
        self.assertEqual(usage["prompt_tokens"], 10)
        self.assertEqual(len(generated[chunk.id]), 5)
        self.assertEqual(generated[chunk.id][0], "如何安装？")
        self.assertEqual(chat.requests[0].thinking_enabled, False)
        self.assertEqual(chat.requests[0].tool_choice, "submit_auto_qa_questions")

    async def test_unknown_ref_exhausts_bounded_schema_repair(self) -> None:
        chunk = _chunk(content="安装客户端", embedding_text="安装客户端")
        chat = _ChatModel(ref="c99")
        with self.assertRaises(IndexingExecutionError) as raised:
            await generate_auto_qa_questions(
                chat,
                (chunk,),
                model_profile_revision_id=uuid4(),
            )
        self.assertEqual(raised.exception.code, ErrorCode.AUTO_QA_RESPONSE_INVALID)
        self.assertEqual(len(chat.requests), 3)

    async def test_question_count_gets_bounded_schema_repair_retry(self) -> None:
        chunk = _chunk(content="安装客户端", embedding_text="安装客户端")
        chat = _RepairingChatModel()
        generated, usage, calls = await generate_auto_qa_questions(
            chat,
            (chunk,),
            model_profile_revision_id=uuid4(),
        )
        self.assertEqual(len(generated[chunk.id]), 5)
        self.assertEqual(calls, 2)
        self.assertEqual(usage["prompt_tokens"], 10)
        self.assertIn("schema-repair retry", chat.requests[1].messages[0].content)

    async def test_question_length_gets_bounded_schema_repair_retry(self) -> None:
        chunk = _chunk(content="安装客户端", embedding_text="安装客户端")
        chat = _RepairingLongQuestionChatModel()
        generated, _, calls = await generate_auto_qa_questions(
            chat,
            (chunk,),
            model_profile_revision_id=uuid4(),
        )
        self.assertEqual(len(generated[chunk.id]), 5)
        self.assertEqual(calls, 2)

    async def test_exhausted_multi_chunk_batch_splits_before_failing(self) -> None:
        chunks = (
            _chunk(content="第一段", ordinal=1),
            _chunk(content="第二段", ordinal=2),
        )
        chat = _SplittingChatModel()
        generated, _, calls = await generate_auto_qa_questions(
            chat,
            chunks,
            model_profile_revision_id=uuid4(),
        )
        self.assertEqual(set(generated), {chunk.id for chunk in chunks})
        self.assertEqual(calls, 5)
        self.assertEqual([len(batch) for batch in chat.chunk_batches], [2, 2, 2, 1, 1])

    async def test_question_embeddings_respect_provider_batch_limit(self) -> None:
        embeddings = _EmbeddingModel(max_batch_size=2)
        vectors = await embed_auto_qa_questions(
            embeddings,
            ("问题一", "问题二", "问题三", "问题四", "问题五"),
        )
        self.assertEqual(
            embeddings.batches,
            [("问题一", "问题二"), ("问题三", "问题四"), ("问题五",)],
        )
        self.assertEqual(len(vectors), 5)

    async def test_question_embedding_rejects_short_provider_response(self) -> None:
        embeddings = _EmbeddingModel(max_batch_size=2, short_response=True)
        with self.assertRaises(IndexingExecutionError) as raised:
            await embed_auto_qa_questions(embeddings, ("问题一", "问题二"))
        self.assertEqual(raised.exception.code, ErrorCode.EMBEDDING_RESPONSE_INVALID)
        self.assertEqual(
            raised.exception.diagnostic["check"],
            "auto_qa_embedding_batch_count",
        )


class _ChatModel:
    def __init__(self, ref: str = "c01") -> None:
        self.ref = ref
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return ChatModelResponse(
            content="",
            model="mimo-v2.5",
            finish_reason="tool_calls",
            provider_request_id="req-1",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            tool_calls=(
                ChatToolCall(
                    id="call-1",
                    name="submit_auto_qa_questions",
                    arguments={
                        "items": [
                            {
                                "ref": self.ref,
                                "questions": [
                                    "1. 如何安装？",
                                    "安装需要什么",
                                    "安装失败怎么办",
                                    "支持哪些系统",
                                    "如何升级",
                                ],
                            }
                        ]
                    },
                ),
            ),
        )


class _RepairingChatModel(_ChatModel):
    async def complete(self, request):
        if not self.requests:
            self.requests.append(request)
            return ChatModelResponse(
                content="",
                model="mimo-v2.5",
                finish_reason="tool_calls",
                provider_request_id="req-invalid",
                usage={"prompt_tokens": 10, "completion_tokens": 10},
                tool_calls=(
                    ChatToolCall(
                        id="call-invalid",
                        name="submit_auto_qa_questions",
                        arguments={
                            "items": [
                                {
                                    "ref": "c01",
                                    "questions": ["一", "二", "三", "四"],
                                }
                            ]
                        },
                    ),
                ),
            )
        return await super().complete(request)


class _RepairingLongQuestionChatModel(_ChatModel):
    async def complete(self, request):
        if not self.requests:
            self.requests.append(request)
            return ChatModelResponse(
                content="",
                model="mimo-v2.5",
                finish_reason="tool_calls",
                provider_request_id="req-invalid-long",
                usage={"prompt_tokens": 10, "completion_tokens": 10},
                tool_calls=(
                    ChatToolCall(
                        id="call-invalid-long",
                        name="submit_auto_qa_questions",
                        arguments={
                            "items": [
                                {
                                    "ref": "c01",
                                    "questions": [
                                        "问" * 501,
                                        "安装需要什么",
                                        "安装失败怎么办",
                                        "支持哪些系统",
                                        "如何升级",
                                    ],
                                }
                            ]
                        },
                    ),
                ),
            )
        return await super().complete(request)


class _SplittingChatModel:
    def __init__(self) -> None:
        self.chunk_batches: list[list[dict]] = []

    async def complete(self, request):
        chunks = json.loads(request.messages[1].content)["chunks"]
        self.chunk_batches.append(chunks)
        items = []
        for chunk in chunks:
            questions = ["问题一", "问题二", "问题三", "问题四", "问题五"]
            if len(chunks) > 1:
                questions.pop()
            items.append({"ref": chunk["ref"], "questions": questions})
        return ChatModelResponse(
            content="",
            model="mimo-v2.5",
            finish_reason="tool_calls",
            provider_request_id=f"req-split-{len(self.chunk_batches)}",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            tool_calls=(
                ChatToolCall(
                    id=f"call-split-{len(self.chunk_batches)}",
                    name="submit_auto_qa_questions",
                    arguments={"items": items},
                ),
            ),
        )


class _EmbeddingModel:
    def __init__(self, *, max_batch_size: int, short_response: bool = False) -> None:
        self.max_batch_size = max_batch_size
        self.short_response = short_response
        self.batches: list[tuple[str, ...]] = []

    async def embed_documents(self, texts: tuple[str, ...]) -> EmbeddingBatch:
        self.batches.append(texts)
        count = len(texts) - 1 if self.short_response else len(texts)
        return EmbeddingBatch(vectors=tuple((float(index),) for index in range(count)))


if __name__ == "__main__":
    unittest.main()
