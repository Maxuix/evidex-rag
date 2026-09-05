from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import threading
import unittest
from dataclasses import replace
from unittest.mock import patch
from uuid import UUID, uuid4

from PIL import Image
from docling_core.types.doc import DocItemLabel, DoclingDocument
from docling_core.types.doc.common.origin import DocumentOrigin
from docling_core.types.doc.common.reference import ImageRef
from docling_core.types.doc.items.table.table_data import TableCell, TableData

import rag_kb.indexing.pipeline as pipeline_module
from rag_kb.document_processing.profiles import (
    DOCLING_ENRICHMENT_CONFIG,
    DOCLING_REPRESENTATION_CONFIG,
    STRUCTURAL_CHUNKING_CONFIG_V4,
    index_profile,
    public_parsing_descriptor,
    profile_for_preset,
)
from rag_kb.document_processing.tokenization import count_chunk_tokens
from rag_kb.domain import (
    ChunkingPreset,
    ContentModality,
    EmbeddingBatch,
    EmbeddingSpaceDefinition,
    EmbeddingSpaceRole,
    ErrorCode,
    IndexingCommand,
    IndexingExecutionError,
    IndexingPhase,
    IndexingTarget,
    PromotionReason,
    PromotionResult,
    PromotionStatus,
    ParserExecutionError,
    ParserProgress,
    ParsingPreset,
    SourceFileIdentity,
    stable_chunk_id,
    stable_vector_id,
    validate_embedding_vector,
)
from rag_kb.indexing.embedding_spaces import require_compatible_embedding_spaces
from rag_kb.indexing.pipeline import IndexingPipeline
from rag_kb.ports.parsing import DocumentParseContinuation, DocumentParseResult


WORKSPACE = UUID("01900000-0000-7000-8000-000000000401")
CONTENT = b"first\n\nsecond"


class _WordEncoding:
    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


class IndexingDomainTests(unittest.TestCase):
    def test_index_profile_uses_one_fixed_token_chunking_identity(self) -> None:
        profile = index_profile()

        self.assertEqual(
            profile.chunking_config["profile"],
            "structural_by_title_token_v5",
        )
        self.assertEqual(
            (
                profile.chunking_config["max_tokens"],
                profile.chunking_config["new_after_n_tokens"],
                profile.chunking_config["tokenizer"],
            ),
            (800, 600, "cl100k_base"),
        )
        self.assertEqual(
            profile.parser_config["profile"], "docling_text_local_v3"
        )
        self.assertNotIn("max_characters", STRUCTURAL_CHUNKING_CONFIG_V4)
        self.assertNotIn("new_after_n_chars", STRUCTURAL_CHUNKING_CONFIG_V4)

    def test_stable_chunk_and_vector_business_keys(self) -> None:
        target = uuid4()
        space = uuid4()
        first = stable_chunk_id(
            target, profile_fingerprint="f" * 64, unit_key="unit-a"
        )
        self.assertEqual(
            first,
            stable_chunk_id(
                target, profile_fingerprint="f" * 64, unit_key="unit-a"
            ),
        )
        self.assertNotEqual(
            first,
            stable_chunk_id(
                target, profile_fingerprint="f" * 64, unit_key="unit-b"
            ),
        )
        self.assertEqual(
            stable_vector_id(space, first), stable_vector_id(space, first)
        )
        with self.assertRaises(ValueError):
            stable_chunk_id(
                target, profile_fingerprint="f" * 64, unit_key=""
            )

    def test_current_multimodal_profile_removes_generated_caption(self) -> None:
        markdown_profile = profile_for_preset(
            ChunkingPreset.STRUCTURAL_BALANCED_V2,
            "multimodal_local_v2",
        )

        self.assertNotIn("caption", DOCLING_ENRICHMENT_CONFIG)
        self.assertEqual(
            DOCLING_REPRESENTATION_CONFIG["image"]["optional"], []
        )
        self.assertNotIn(
            "caption_text", str(DOCLING_REPRESENTATION_CONFIG)
        )
        self.assertEqual(
            public_parsing_descriptor(markdown_profile.parser_config),
            {
                "preset": "multimodal_local_v2",
                "profile": "docling_multimodal_local_v4",
            },
        )
        self.assertTrue(
            markdown_profile.parser_config["markdown_media"][
                "admission_remote_snapshot"
            ]
        )
        self.assertFalse(
            markdown_profile.parser_config["markdown_media"][
                "docling_remote_fetch"
            ]
        )
        with self.assertRaises(ValueError):
            profile_for_preset(
                ChunkingPreset.STRUCTURAL_BALANCED_V2,
                "multimodal_local_v1",
            )

    def test_embedding_space_and_output_validation_fail_closed(self) -> None:
        expected = _embedding()
        require_compatible_embedding_spaces(expected, expected, expected)
        with self.assertRaises(IndexingExecutionError) as mismatch:
            require_compatible_embedding_spaces(
                expected,
                expected,
                replace(expected, configuration_fingerprint="sha256:changed"),
            )
        self.assertEqual(mismatch.exception.code, ErrorCode.EMBEDDING_SPACE_MISMATCH)

        validate_embedding_vector(_vector(), expected)
        for invalid in ((_vector()[:-1]), tuple(0.0 for _ in range(1024))):
            with self.assertRaises(IndexingExecutionError) as response:
                validate_embedding_vector(invalid, expected)
            self.assertEqual(response.exception.code, ErrorCode.EMBEDDING_RESPONSE_INVALID)

class IndexingPipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        tokenizer = patch(
            "rag_kb.document_processing.tokenization._encoding",
            return_value=_WordEncoding(),
        )
        tokenizer.start()
        self.addCleanup(tokenizer.stop)

    async def test_rejected_questions_are_not_embedded_and_zero_counts_persist(self):
        target = replace(_target(), auto_qa_enabled=True,
                         auto_qa_model_profile_revision_id=uuid4())
        repository = _Repository(target)
        factory = _Factory(repository)
        provider = _Provider(factory)
        pipeline = _pipeline(factory, provider)
        pipeline._chat_model = object()

        async def generate(chat, chunks, **kwargs):
            self.assertFalse(factory.active)
            return {chunk.id: ("UNSUPPORTED_QUESTION",) for chunk in chunks}, {}, 1

        async def verify(chat, chunks, generated, **kwargs):
            self.assertFalse(factory.active)
            return {chunk.id: () for chunk in chunks}, {}, 1

        with patch.object(pipeline_module, "generate_auto_qa_questions", generate), patch.object(
            pipeline_module, "verify_auto_qa_questions", verify,
        ):
            await pipeline.execute(IndexingCommand(target.job_id, target.indexed_document_version_id))
        self.assertTrue(repository.chunks)
        self.assertEqual(repository.processed_question_counts,
                         {chunk.id: 0 for chunk in repository.chunks.values()})
        self.assertEqual(repository.questions, {})
        self.assertFalse(any("UNSUPPORTED_QUESTION" in text for batch in provider.inputs for text in batch))

    async def test_pdf_segment_continuation_yields_without_consuming_embeddings(self) -> None:
        repository = _Repository(_target())
        factory = _Factory(repository)
        provider = _Provider(factory)
        parser = _ContinuationParser(factory)
        pipeline = _pipeline(factory, provider, parser)
        command = IndexingCommand(
            repository.target.job_id,
            repository.target.indexed_document_version_id,
        )

        result = await pipeline.execute(command)

        self.assertEqual(result.status, "queued")
        self.assertEqual(repository.status, "queued")
        self.assertEqual(provider.calls, 0)
        self.assertEqual(parser.calls, 1)

    async def test_rejected_continuation_is_cancelled_and_checkpoint_is_discarded(
        self,
    ) -> None:
        repository = _Repository(_target())
        repository.yield_allowed = False
        factory = _Factory(repository)
        provider = _Provider(factory)
        parser = _ContinuationParser(factory)
        pipeline = _pipeline(factory, provider, parser)
        command = IndexingCommand(
            repository.target.job_id,
            repository.target.indexed_document_version_id,
        )

        result = await pipeline.execute(command)

        self.assertEqual(result.status, "cancelled")
        self.assertIsNone(repository.failure)
        self.assertEqual(
            parser.discarded,
            [str(repository.target.indexed_document_version_id)],
        )
        self.assertEqual(provider.calls, 0)

    async def test_markdown_v2_reuses_multimodal_assets_and_both_spaces(self) -> None:
        repository = _Repository(_target(multimodal=True, markdown_v2=True))
        factory = _Factory(repository)
        text_provider = _Provider(factory)
        text_provider.max_batch_size = 10
        visual_provider = _MultimodalProvider(factory)
        asset_store = _AssetStore(factory)
        parser = _MultimodalParser(factory)
        global _CURRENT_FACTORY
        _CURRENT_FACTORY = factory
        pipeline = IndexingPipeline(
            factory,
            _FileStore(factory),
            parser,
            text_provider,
            _embedding(),
            asset_store=asset_store,
            multimodal_embedding_provider=visual_provider,
        )
        command = IndexingCommand(
            repository.target.job_id,
            repository.target.indexed_document_version_id,
        )

        result = await pipeline.execute(command)

        self.assertEqual((result.status, result.chunk_count), ("ready", 2))
        self.assertEqual(len(repository.assets), 1)
        self.assertIsNotNone(repository.manifest)
        self.assertEqual(repository.manifest.unit_count, 2)
        self.assertEqual(repository.manifest.representation_count, 2)
        self.assertEqual(repository.manifest.relation_count, 1)
        self.assertEqual(len(asset_store.writes), 1)
        self.assertEqual(parser.presets, [ParsingPreset.MULTIMODAL_LOCAL_V2])
        self.assertEqual(text_provider.calls, 1)
        self.assertEqual(text_provider.inputs, [("[body]\nbody evidence",)])
        self.assertEqual(visual_provider.image_calls, 1)
        self.assertNotIn(
            "caption_text",
            {
                item["representation_kind"]
                for item in repository.manifest.representation_matrix
            },
        )
        text_chunk = next(
            chunk
            for chunk in repository.chunks.values()
            if chunk.modality is ContentModality.TEXT
        )
        self.assertEqual(text_chunk.content, "body evidence")
        self.assertEqual(text_chunk.embedding_text, "[body]\nbody evidence")
        self.assertEqual(len(repository.relations), 1)
        self.assertEqual(
            {chunk.modality for chunk in repository.chunks.values()},
            {ContentModality.TEXT, ContentModality.IMAGE},
        )

    async def test_unified_multimodal_uses_one_provider_and_one_space(self) -> None:
        repository = _Repository(_target(multimodal=True, unified=True))
        factory = _Factory(repository)
        unified_provider = _MultimodalProvider(factory)
        unified_provider.embedding_space = repository.target.embedding_space
        asset_store = _AssetStore(factory)
        global _CURRENT_FACTORY
        _CURRENT_FACTORY = factory
        pipeline = IndexingPipeline(
            factory,
            _FileStore(factory),
            _MultimodalParser(factory),
            _Provider(factory),
            _embedding(),
            asset_store=asset_store,
            multimodal_embedding_provider=unified_provider,
        )
        command = IndexingCommand(
            repository.target.job_id,
            repository.target.indexed_document_version_id,
        )

        result = await pipeline.execute(command)

        self.assertEqual(result.status, "ready")
        self.assertEqual(unified_provider.text_calls, 1)
        self.assertEqual(unified_provider.image_calls, 1)
        self.assertEqual(
            {item.embedding_space_id for item in repository.vectors.values()},
            {repository.target.embedding_space_id},
        )
        self.assertEqual(
            {item.representation_kind for item in repository.vectors.values()},
            {"text", "native_image"},
        )

    async def test_dynamic_multimodal_revision_uses_local_assets_without_legacy_adapter(
        self,
    ) -> None:
        cross_role = EmbeddingSpaceRole.CROSS_MODAL_RETRIEVAL.value
        target = _target(multimodal=True)
        dynamic_space = replace(
            target.embedding_spaces[cross_role],
            model_profile_revision_id=uuid4(),
        )
        target = replace(
            target,
            embedding_spaces={
                **target.embedding_spaces,
                cross_role: dynamic_space,
            },
        )
        repository = _Repository(target)
        factory = _Factory(repository)
        visual_provider = _MultimodalProvider(factory)
        visual_provider.embedding_space = dynamic_space
        resolver_calls: list[EmbeddingSpaceDefinition] = []

        async def resolve(space: EmbeddingSpaceDefinition):
            resolver_calls.append(space)
            return visual_provider

        asset_store = _AssetStore(factory)
        global _CURRENT_FACTORY
        _CURRENT_FACTORY = factory
        pipeline = IndexingPipeline(
            factory,
            _FileStore(factory),
            _MultimodalParser(factory),
            _Provider(factory),
            _embedding(),
            asset_store=asset_store,
            multimodal_embedding_provider=None,
            multimodal_embedding_model_resolver=resolve,
        )

        result = await pipeline.execute(
            IndexingCommand(target.job_id, target.indexed_document_version_id)
        )

        self.assertEqual(result.status, "ready")
        self.assertEqual(resolver_calls, [dynamic_space])
        self.assertEqual(len(asset_store.writes), 1)
        self.assertEqual(visual_provider.image_calls, 1)

    async def test_external_operations_hold_no_transaction_and_replay_is_idempotent(self) -> None:
        repository = _Repository(_target())
        factory = _Factory(repository)
        provider = _Provider(factory)
        pipeline = _pipeline(factory, provider)
        command = IndexingCommand(repository.target.job_id, repository.target.indexed_document_version_id)

        result = await pipeline.execute(command)
        replay = await pipeline.execute(command)

        self.assertEqual((result.status, result.chunk_count), ("ready", 2))
        self.assertEqual((replay.replayed, replay.chunk_count), (True, 2))
        self.assertEqual(len(repository.chunks), 2)
        self.assertEqual(len(repository.vectors), 2)
        self.assertEqual(
            [repository.chunks[ordinal].token_count for ordinal in range(2)],
            [
                count_chunk_tokens("Alpha\n\nfirst"),
                count_chunk_tokens("Beta\n\nsecond"),
            ],
        )
        self.assertEqual(provider.calls, 2)
        self.assertEqual(repository.status, "completed")
        self.assertEqual(
            (result.serving_status, replay.serving_status),
            ("serving", "serving"),
        )
        self.assertEqual(repository.serving, "serving")

    async def test_partial_embedding_failure_retries_from_an_empty_candidate(self) -> None:
        repository = _Repository(_target())
        factory = _Factory(repository)
        provider = _Provider(factory, fail_call=2)
        pipeline = _pipeline(factory, provider)
        command = IndexingCommand(repository.target.job_id, repository.target.indexed_document_version_id)

        with self.assertRaises(IndexingExecutionError) as failed:
            await pipeline.execute(command)
        self.assertEqual(failed.exception.code, ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE)
        self.assertEqual(repository.status, "failed")
        self.assertEqual(repository.serving, "candidate")
        self.assertEqual(len(repository.vectors), 1)

        provider.fail_call = None
        provider.calls = 0
        provider.inputs = []
        result = await pipeline.execute(command)
        self.assertEqual((result.status, result.chunk_count), ("ready", 2))
        self.assertEqual(result.serving_status, "serving")
        self.assertEqual(len(repository.chunks), 2)
        self.assertEqual(len(repository.vectors), 2)
        self.assertEqual(provider.calls, 2)
        self.assertEqual(
            provider.inputs,
            [("[body]\nAlpha\n\nfirst",), ("[body]\nBeta\n\nsecond",)],
        )

    async def test_multimodal_retry_rebuilds_text_and_image_representations(
        self,
    ) -> None:
        repository = _Repository(_target(multimodal=True))
        factory = _Factory(repository)
        text_provider = _Provider(factory)
        text_provider.max_batch_size = 10
        visual_provider = _MultimodalProvider(factory, fail_call=2)
        visual_provider.max_batch_size = 1
        asset_store = _AssetStore(factory)
        global _CURRENT_FACTORY
        _CURRENT_FACTORY = factory
        pipeline = IndexingPipeline(
            factory,
            _FileStore(factory),
            _MultimodalParser(factory, document=_table_and_picture_document()),
            text_provider,
            _embedding(),
            asset_store=asset_store,
            multimodal_embedding_provider=visual_provider,
        )
        command = IndexingCommand(
            repository.target.job_id,
            repository.target.indexed_document_version_id,
        )

        with self.assertRaises(IndexingExecutionError) as failed:
            await pipeline.execute(command)
        self.assertEqual(
            failed.exception.code,
            ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE,
        )
        self.assertEqual(text_provider.calls, 1)
        self.assertEqual(len(repository.vectors), 2)

        text_provider.calls = 0
        text_provider.inputs = []
        visual_provider.fail_call = None
        visual_provider.image_calls = 0
        result = await pipeline.execute(command)

        self.assertEqual(result.status, "ready")
        self.assertEqual(text_provider.calls, 1)
        self.assertEqual(visual_provider.image_calls, 2)
        self.assertEqual(len(repository.vectors), 3)
        self.assertEqual(repository.asset_discard_calls, 2)
        self.assertEqual(
            asset_store.discards,
            [
                (WORKSPACE, repository.target.indexed_document_version_id),
                (WORKSPACE, repository.target.indexed_document_version_id),
            ],
        )
        self.assertEqual(
            asset_store.assets_visible_during_discard,
            [False, True],
        )

    async def test_retry_discards_the_previous_attempt_manifest(
        self,
    ) -> None:
        repository = _Repository(_target())
        factory = _Factory(repository)
        provider = _Provider(factory, fail_call=2)
        parser = _Parser(factory)
        pipeline = _pipeline(factory, provider, parser)
        command = IndexingCommand(
            repository.target.job_id,
            repository.target.indexed_document_version_id,
        )
        with self.assertRaises(IndexingExecutionError):
            await pipeline.execute(command)

        changed = _document("guide")
        changed.add_heading(text="Alpha", level=1)
        changed.add_text(label=DocItemLabel.TEXT, text="changed")
        changed.add_heading(text="Beta", level=1)
        changed.add_text(label=DocItemLabel.TEXT, text="second")
        parser.document = changed
        provider.fail_call = None
        provider.calls = 0

        result = await pipeline.execute(command)

        self.assertEqual(result.status, "ready")
        self.assertEqual(provider.calls, 2)
        self.assertTrue(
            any(
                "changed" in chunk.content
                for chunk in repository.chunks.values()
            )
        )

    async def test_retry_rebuilds_all_embedding_batches(self) -> None:
        repository = _Repository(_target())
        factory = _Factory(repository)
        provider = _Provider(factory, fail_call=3, max_batch_size=2)
        parser = _Parser(factory, document=_many_text_document(5))
        pipeline = _pipeline(factory, provider, parser)
        command = IndexingCommand(
            repository.target.job_id,
            repository.target.indexed_document_version_id,
        )
        with self.assertRaises(IndexingExecutionError):
            await pipeline.execute(command)
        self.assertEqual(len(repository.vectors), 4)

        provider.fail_call = None
        provider.calls = 0
        result = await pipeline.execute(command)

        self.assertEqual(result.status, "ready")
        self.assertEqual(provider.calls, 3)
        self.assertEqual(len(repository.vectors), 5)

    async def test_completed_replay_compensates_interrupted_promotion(self) -> None:
        repository = _Repository(_target())
        repository.status = "completed"
        repository.chunks = {0: object(), 1: object()}
        factory = _Factory(repository)
        provider = _Provider(factory)
        parser = _Parser(factory)
        pipeline = _pipeline(factory, provider, parser)
        command = IndexingCommand(
            repository.target.job_id,
            repository.target.indexed_document_version_id,
        )

        result = await pipeline.execute(command)

        self.assertTrue(result.replayed)
        self.assertEqual(result.serving_status, "serving")
        self.assertEqual(repository.serving, "serving")
        self.assertEqual(provider.calls, 0)
        self.assertEqual(parser.calls, 0)

    async def test_completed_replay_rejects_old_structural_profile(self) -> None:
        target = _target()
        chunking_config = deepcopy(target.chunking_config)
        chunking_config["profile"] = "structural_by_title_token_v3"
        chunking_config.pop("consumer_projection")

        await self._assert_invalid_completed_replay(
            replace(target, chunking_config=chunking_config),
            expected_check="parser_chunking_profile",
        )

    async def test_completed_replay_rejects_old_semantic_profile(self) -> None:
        target = _target(ChunkingPreset.SEMANTIC_BALANCED_V1)
        chunking_config = deepcopy(target.chunking_config)
        chunking_config["profile"] = "semantic_breakpoint_v2"
        chunking_config.pop("consumer_projection")
        chunking_config.pop("required_embedding_roles")

        await self._assert_invalid_completed_replay(
            replace(target, chunking_config=chunking_config),
            expected_check="parser_chunking_profile",
        )

    async def test_completed_replay_rejects_missing_or_non_required_semantic_role(
        self,
    ) -> None:
        target = _target(ChunkingPreset.SEMANTIC_BALANCED_V1)
        role = EmbeddingSpaceRole.SEMANTIC_ANALYSIS.value

        await self._assert_invalid_completed_replay(
            replace(
                target,
                embedding_space_ids={
                    key: value
                    for key, value in target.embedding_space_ids.items()
                    if key != role
                },
                embedding_spaces={
                    key: value
                    for key, value in target.embedding_spaces.items()
                    if key != role
                },
            ),
            expected_check="semantic_analysis_space_role",
        )

    async def test_completed_replay_rejects_wrong_semantic_role_space(self) -> None:
        target = _target(ChunkingPreset.SEMANTIC_BALANCED_V1)
        role = EmbeddingSpaceRole.SEMANTIC_ANALYSIS.value

        await self._assert_invalid_completed_replay(
            replace(
                target,
                embedding_space_ids={**target.embedding_space_ids, role: uuid4()},
            ),
            expected_check="semantic_analysis_space_role",
        )

    async def _assert_invalid_completed_replay(
        self,
        target: IndexingTarget,
        *,
        expected_check: str,
    ) -> None:
        repository = _Repository(target)
        repository.status = "completed"
        repository.chunks = {0: object()}
        factory = _Factory(repository)
        provider = _Provider(factory)
        parser = _Parser(factory)
        pipeline = _pipeline(factory, provider, parser)
        command = IndexingCommand(target.job_id, target.indexed_document_version_id)

        with self.assertRaises(IndexingExecutionError) as failure:
            await pipeline.execute(command)

        self.assertEqual(
            failure.exception.code, ErrorCode.INDEX_REVISION_INCOMPATIBLE
        )
        self.assertEqual(failure.exception.diagnostic, {"check": expected_check})
        self.assertEqual(repository.serving, "candidate")
        self.assertEqual(provider.calls, 0)
        self.assertEqual(parser.calls, 0)

    async def test_parser_failure_persists_parser_phase_and_stable_code(self) -> None:
        repository = _Repository(_target())
        factory = _Factory(repository)
        pipeline = IndexingPipeline(
            factory,
            _FileStore(factory),
            _FailingParser(factory),
            _Provider(factory),
            _embedding(),
        )
        global _CURRENT_FACTORY
        _CURRENT_FACTORY = factory
        command = IndexingCommand(
            repository.target.job_id,
            repository.target.indexed_document_version_id,
        )

        with self.assertRaises(IndexingExecutionError) as failure:
            await pipeline.execute(command)
        self.assertEqual(failure.exception.code, ErrorCode.PARSER_CRASHED)
        self.assertEqual(repository.failure[0:2], ("parsing", "PARSER_CRASHED"))
        self.assertEqual(repository.serving, "candidate")

    async def test_semantic_retry_rebuilds_plan_and_analysis(self) -> None:
        repository = _Repository(_target(ChunkingPreset.SEMANTIC_BALANCED_V1))
        factory = _Factory(repository)
        provider = _Provider(factory, fail_call=8)
        pipeline = _pipeline(factory, provider, _SemanticParser(factory))
        command = IndexingCommand(
            repository.target.job_id,
            repository.target.indexed_document_version_id,
        )

        with self.assertRaises(IndexingExecutionError) as failed:
            await pipeline.execute(command)
        self.assertEqual(failed.exception.code, ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE)
        self.assertIsNotNone(repository.plan)
        self.assertEqual(provider.calls, 8)

        provider.fail_call = None
        provider.calls = 0
        result = await pipeline.execute(command)

        self.assertEqual(result.status, "ready")
        self.assertGreater(provider.calls, result.chunk_count)
        self.assertEqual(
            [repository.chunks[index].ordinal for index in repository.chunks],
            list(range(result.chunk_count)),
        )

    async def test_final_semantic_chunks_use_configured_batch_size(self) -> None:
        repository = _Repository(_target(ChunkingPreset.SEMANTIC_BALANCED_V1))
        factory = _Factory(repository)
        provider = _Provider(factory, max_batch_size=10)
        pipeline = _pipeline(factory, provider, _SemanticParser(factory))
        command = IndexingCommand(
            repository.target.job_id,
            repository.target.indexed_document_version_id,
        )

        result = await pipeline.execute(command)

        self.assertGreater(result.chunk_count, 1)
        self.assertEqual(len(provider.inputs[-1]), result.chunk_count)
        self.assertLessEqual(len(provider.inputs[-1]), provider.max_batch_size)

    async def test_semantic_strategy_requires_analysis_role_on_primary_text_space(
        self,
    ) -> None:
        target = _target(ChunkingPreset.SEMANTIC_BALANCED_V1)
        analysis_role = EmbeddingSpaceRole.SEMANTIC_ANALYSIS.value
        repository = _Repository(
            replace(
                target,
                embedding_space_ids={
                    role: space_id
                    for role, space_id in target.embedding_space_ids.items()
                    if role != analysis_role
                },
                embedding_spaces={
                    role: space
                    for role, space in target.embedding_spaces.items()
                    if role != analysis_role
                },
            )
        )
        factory = _Factory(repository)
        provider = _Provider(factory)
        pipeline = _pipeline(factory, provider, _SemanticParser(factory))
        command = IndexingCommand(
            repository.target.job_id,
            repository.target.indexed_document_version_id,
        )

        with self.assertRaises(IndexingExecutionError) as failure:
            await pipeline.execute(command)

        self.assertEqual(
            failure.exception.code, ErrorCode.INDEX_REVISION_INCOMPATIBLE
        )
        self.assertEqual(
            failure.exception.diagnostic,
            {"check": "semantic_analysis_space_role"},
        )
        self.assertEqual(provider.calls, 0)

    async def test_semantic_analysis_role_rejects_a_different_space(self) -> None:
        target = _target(ChunkingPreset.SEMANTIC_BALANCED_V1)
        analysis_role = EmbeddingSpaceRole.SEMANTIC_ANALYSIS.value
        repository = _Repository(
            replace(
                target,
                embedding_space_ids={
                    **target.embedding_space_ids,
                    analysis_role: uuid4(),
                },
            )
        )
        factory = _Factory(repository)
        provider = _Provider(factory)
        pipeline = _pipeline(factory, provider, _SemanticParser(factory))
        command = IndexingCommand(
            repository.target.job_id,
            repository.target.indexed_document_version_id,
        )

        with self.assertRaises(IndexingExecutionError) as failure:
            await pipeline.execute(command)

        self.assertEqual(
            failure.exception.code, ErrorCode.INDEX_REVISION_INCOMPATIBLE
        )
        self.assertEqual(provider.calls, 0)

    async def test_semantic_planning_does_not_block_event_loop(self) -> None:
        repository = _Repository(_target(ChunkingPreset.SEMANTIC_BALANCED_V1))
        factory = _Factory(repository)
        provider = _Provider(factory)
        pipeline = _pipeline(factory, provider, _SemanticParser(factory))
        command = IndexingCommand(
            repository.target.job_id,
            repository.target.indexed_document_version_id,
        )
        loop_thread = threading.get_ident()
        planner_entered = threading.Event()
        planner_release = threading.Event()
        planner_thread: list[int] = []
        original = pipeline_module.build_chunk_plan

        def blocked_planner(*args, **kwargs):
            planner_thread.append(threading.get_ident())
            planner_entered.set()
            if not planner_release.wait(timeout=2):
                raise AssertionError("event loop did not release semantic planner")
            return original(*args, **kwargs)

        async def release_after_planner_starts() -> None:
            while not planner_entered.is_set():
                await asyncio.sleep(0)
            planner_release.set()

        release_task = asyncio.create_task(release_after_planner_starts())
        try:
            with patch.object(
                pipeline_module,
                "build_chunk_plan",
                side_effect=blocked_planner,
            ):
                result = await pipeline.execute(command)
            await release_task
        finally:
            planner_release.set()
            if not release_task.done():
                release_task.cancel()
                await asyncio.gather(release_task, return_exceptions=True)

        self.assertEqual(result.status, "ready")
        self.assertEqual(len(planner_thread), 1)
        self.assertNotEqual(planner_thread[0], loop_thread)

class _Factory:
    def __init__(self, repository) -> None:
        self.repository = repository
        self.active = False

    def __call__(self, *, mode=None):
        del mode
        return _UnitOfWork(self)


class _UnitOfWork:
    def __init__(self, factory) -> None:
        self.factory = factory
        self.workspace_id = WORKSPACE
        self.indexing = factory.repository

    async def __aenter__(self):
        if self.factory.active:
            raise AssertionError("nested transaction")
        self.factory.active = True
        return self

    async def commit(self):
        return None

    async def __aexit__(self, *args):
        self.factory.active = False


class _Repository:
    def __init__(self, target) -> None:
        self.target = target
        self.status = "queued"
        self.serving = "candidate"
        self.chunks = {}
        self.vectors = {}
        self.failure = None
        self.plan = None
        self.manifest = None
        self.assets = ()
        self.relations = ()
        self.lexical_rows = {}
        self.lexical_manifest = None
        self.questions = {}
        self.asset_discard_calls = 0
        self.yield_allowed = True

    async def prepare(self, command):
        self._active()
        if command.job_id != self.target.job_id:
            return None
        if self.status == "completed":
            return replace(self.target, already_complete=True)
        self.chunks.clear()
        self.vectors.clear()
        self.plan = None
        self.manifest = None
        self.relations = ()
        self.lexical_rows.clear()
        self.lexical_manifest = None
        self.questions.clear()
        self.status = "running"
        self.failure = None
        return self.target

    async def discard_partial_assets(self, command):
        del command
        self._active()
        self.asset_discard_calls += 1
        self.assets = ()
        return True

    async def promote(self, command):
        self._active()
        if (
            command.job_id != self.target.job_id
            or command.indexed_document_version_id
            != self.target.indexed_document_version_id
        ):
            return None
        if self.serving == "serving":
            reason = PromotionReason.ALREADY_SERVING
        elif self.serving == "retired":
            return PromotionResult(
                command.job_id,
                command.indexed_document_version_id,
                PromotionStatus.RETIRED,
                PromotionReason.ALREADY_RETIRED,
            )
        elif self.status != "completed":
            return PromotionResult(
                command.job_id,
                command.indexed_document_version_id,
                PromotionStatus.NOT_READY,
                PromotionReason.JOB_INCOMPLETE,
            )
        else:
            self.serving = "serving"
            reason = PromotionReason.PROMOTED
        return PromotionResult(
            command.job_id,
            command.indexed_document_version_id,
            PromotionStatus.SERVING,
            reason,
        )

    async def set_phase(self, command, phase):
        del command, phase
        self._active()
        return self.status == "running"

    async def set_progress(self, command, progress):
        del command, progress
        self._active()
        return self.status == "running"

    async def yield_continuation(self, command, progress):
        del command, progress
        self._active()
        if self.yield_allowed:
            self.status = "queued"
        return self.yield_allowed

    async def save_chunk_plan(self, command, proposed):
        del command
        self._active()
        self.plan = proposed
        return True

    async def upsert_assets(self, command, assets):
        del command
        self._active()
        self.assets = assets
        return True

    async def upsert_relations(self, command, relations):
        del command
        self._active()
        self.relations = relations
        return True

    async def save_artifact_manifest(self, command, proposed):
        del command
        self._active()
        self.manifest = proposed
        return True

    async def upsert_batch(self, command, chunks, vectors):
        del command
        self._active()
        for chunk in chunks:
            existing = self.chunks.get(chunk.ordinal)
            if existing is not None and existing.content_hash != chunk.content_hash:
                raise IndexingExecutionError(
                    ErrorCode.INDEX_PERSISTENCE_FAILED,
                    phase=IndexingPhase.PERSISTING,
                    diagnostic={"check": "stable_chunk_key"},
                )
            self.chunks[chunk.ordinal] = chunk
        for vector in vectors:
            self.vectors[
                (
                    vector.id,
                    vector.index_chunk_id,
                    vector.embedding_space_id,
                    vector.representation_kind,
                )
            ] = vector
        return True

    async def set_auto_qa_progress(self, command, progress):
        del command, progress
        self._active()
        return self.status == "running"

    async def upsert_questions(self, command, rows, *, processed_chunk_counts=None):
        self.processed_question_counts = processed_chunk_counts
        del command
        self._active()
        for row in rows:
            self.questions[(row.index_chunk_id, row.ordinal)] = row
        return True

    async def upsert_lexical_rows(self, command, rows):
        del command
        self._active()
        for row in rows:
            existing = self.lexical_rows.get(row.index_chunk_id)
            if existing is not None and existing != row:
                raise AssertionError("lexical row mismatch")
            self.lexical_rows[row.index_chunk_id] = row
        return True

    async def complete_lexical_manifest(self, command, proposed):
        del command
        self._active()
        if self.lexical_manifest is None:
            self.lexical_manifest = proposed
        if self.lexical_manifest != proposed:
            raise AssertionError("lexical manifest mismatch")
        return True

    async def complete(self, command, *, expected_chunks):
        del command
        self._active()
        if len(self.chunks) != expected_chunks:
            raise AssertionError("incomplete")
        if self.lexical_manifest is None:
            raise AssertionError("lexical manifest missing")
        if self.manifest is None:
            raise AssertionError("artifact manifest missing")
        identities = {
            (
                str(vector.index_chunk_id),
                str(vector.embedding_space_id),
                vector.representation_kind,
            )
            for vector in self.vectors.values()
        }
        if any(
            item["required"]
            and (
                item["unit_id"],
                item["space_id"],
                item["representation_kind"],
            )
            not in identities
            for item in self.manifest.representation_matrix
        ):
            raise AssertionError("incomplete")
        self.status = "completed"
        return True

    async def count_chunks(self, command):
        del command
        self._active()
        return len(self.chunks)

    async def fail(self, command, *, phase, error_code, error_detail):
        del command
        self._active()
        self.status = "failed"
        self.failure = (phase.value, error_code, error_detail)
        return True

    def _active(self):
        if not _CURRENT_FACTORY.active:
            raise AssertionError("repository call requires transaction")


class _FileStore:
    def __init__(self, factory) -> None:
        self.factory = factory

    def parse_uri(self, storage_uri):
        self._outside()
        del storage_uri
        return SourceFileIdentity(WORKSPACE, "a" * 64)

    async def read_final(self, identity):
        self._outside()
        del identity
        return CONTENT

    def _outside(self):
        if self.factory.active:
            raise AssertionError("file I/O ran inside transaction")


class _Parser:
    """A text-only parser double returning a synthetic converted document."""

    def __init__(self, factory, *, document=None) -> None:
        self.factory = factory
        self.calls = 0
        self.document = document

    async def parse(self, source, *, profile, checkpoint_key, on_progress=None):
        if self.factory.active:
            raise AssertionError("parser ran inside transaction")
        del source, profile, checkpoint_key, on_progress
        self.calls += 1
        return DocumentParseResult(self.document or _text_document())


class _MultimodalParser:
    def __init__(self, factory, *, document=None) -> None:
        self.factory = factory
        self.calls = 0
        self.presets = []
        self.document = document

    async def parse(self, source, *, profile, checkpoint_key, on_progress=None):
        if self.factory.active:
            raise AssertionError("parser ran inside transaction")
        del source
        self.calls += 1
        del checkpoint_key, on_progress
        self.presets.append(profile.preset)
        return DocumentParseResult(self.document or _visual_document())


class _SemanticParser:
    def __init__(self, factory) -> None:
        self.factory = factory
        self.calls = 0

    async def parse(self, source, *, profile, checkpoint_key, on_progress=None):
        if self.factory.active:
            raise AssertionError("parser ran inside transaction")
        del source, profile, checkpoint_key, on_progress
        self.calls += 1
        return DocumentParseResult(_analysis_document())


class _ContinuationParser:
    def __init__(self, factory) -> None:
        self.factory = factory
        self.calls = 0
        self.discarded = []

    async def parse(self, source, *, profile, checkpoint_key, on_progress=None):
        del source, profile, checkpoint_key
        if self.factory.active:
            raise AssertionError("parser ran inside transaction")
        self.calls += 1
        progress = ParserProgress(
            stage="segment_checkpointed",
            total_pages=40,
            completed_pages=20,
            segment_number=2,
            segment_count=2,
            page_from=21,
            page_to=40,
        )
        if on_progress is not None:
            await on_progress(progress)
        return DocumentParseContinuation(progress)

    def discard_checkpoint(self, checkpoint_key):
        self.discarded.append(checkpoint_key)


def _document(name: str) -> DoclingDocument:
    document = DoclingDocument(name=name)
    document.origin = DocumentOrigin(
        mimetype="text/plain", binary_hash=17, filename=f"{name}.txt"
    )
    return document


def _text_document() -> DoclingDocument:
    document = _document("guide")
    for heading, body in (("Alpha", "first"), ("Beta", "second")):
        document.add_heading(text=heading, level=1)
        document.add_text(label=DocItemLabel.TEXT, text=body)
    return document


def _many_text_document(count: int) -> DoclingDocument:
    document = _document("many")
    for ordinal in range(count):
        document.add_heading(text=f"Section {ordinal}", level=1)
        document.add_text(label=DocItemLabel.TEXT, text=f"evidence {ordinal}")
    return document


def _analysis_document() -> DoclingDocument:
    document = _document("analysis")
    for ordinal in range(7):
        document.add_text(
            label=DocItemLabel.TEXT,
            text=("word " * 120 + f"topic{ordinal}").strip(),
        )
    return document


def _visual_document() -> DoclingDocument:
    document = _document("evidence")
    document.add_text(label=DocItemLabel.TEXT, text="body evidence")
    document.add_picture(
        image=ImageRef.from_pil(Image.new("RGB", (120, 80), (10, 20, 30)), dpi=72)
    )
    return document


def _table_and_picture_document() -> DoclingDocument:
    document = _document("table-evidence")
    cells = [
        TableCell(
            text=value,
            start_row_offset_idx=row,
            end_row_offset_idx=row + 1,
            start_col_offset_idx=column,
            end_col_offset_idx=column + 1,
            column_header=row == 0,
        )
        for row, values in enumerate((("Service", "Owner"), ("API", "Platform")))
        for column, value in enumerate(values)
    ]
    table = document.add_table(
        data=TableData(num_rows=2, num_cols=2, table_cells=cells)
    )
    table.image = ImageRef.from_pil(
        Image.new("RGB", (160, 100), (60, 70, 80)), dpi=72
    )
    document.add_picture(
        image=ImageRef.from_pil(
            Image.new("RGB", (120, 80), (10, 20, 30)), dpi=72
        )
    )
    return document


class _Provider:
    def __init__(self, factory, *, fail_call=None, max_batch_size=1) -> None:
        self.factory = factory
        self.embedding_space = _embedding()
        self.max_batch_size = max_batch_size
        self.calls = 0
        self.inputs = []
        self.fail_call = fail_call

    async def embed_documents(self, texts):
        if self.factory.active:
            raise AssertionError("provider ran inside transaction")
        self.calls += 1
        self.inputs.append(texts)
        if self.calls == self.fail_call:
            raise IndexingExecutionError(
                ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE,
                phase=IndexingPhase.EMBEDDING,
                diagnostic={"retry_exhausted": True},
            )
        return EmbeddingBatch(tuple(_vector() for _ in texts))


class _MultimodalProvider:
    def __init__(self, factory, *, fail_call=None) -> None:
        self.factory = factory
        self.embedding_space = _multimodal_embedding()
        self.max_batch_size = 20
        self.image_calls = 0
        self.text_calls = 0
        self.text_inputs = []
        self.fail_call = fail_call

    async def embed_documents(self, texts):
        if self.factory.active:
            raise AssertionError("provider ran inside transaction")
        self.text_calls += 1
        self.text_inputs.append(texts)
        return EmbeddingBatch(tuple(_multimodal_vector() for _ in texts))

    async def embed_images(self, images):
        if self.factory.active:
            raise AssertionError("provider ran inside transaction")
        self.image_calls += 1
        if self.image_calls == self.fail_call:
            raise IndexingExecutionError(
                ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE,
                phase=IndexingPhase.MULTIMODAL_EMBEDDING,
                diagnostic={"retry_exhausted": True},
            )
        return EmbeddingBatch(tuple(_multimodal_vector() for _ in images))


class _AssetStore:
    def __init__(self, factory) -> None:
        self.factory = factory
        self.writes = []
        self.discards = []
        self.assets_visible_during_discard = []

    async def put(self, identity, content, checksum):
        if self.factory.active:
            raise AssertionError("asset I/O ran inside transaction")
        self.writes.append((identity, content, checksum))

    async def discard_target(self, workspace_id, indexed_document_version_id):
        if self.factory.active:
            raise AssertionError("asset I/O ran inside transaction")
        self.assets_visible_during_discard.append(
            bool(self.factory.repository.assets)
        )
        self.discards.append((workspace_id, indexed_document_version_id))


class _FailingParser:
    def __init__(self, factory) -> None:
        self.factory = factory

    async def parse(self, source, *, profile, checkpoint_key, on_progress=None):
        del source, profile, checkpoint_key, on_progress
        if self.factory.active:
            raise AssertionError("parser ran inside transaction")
        raise ParserExecutionError(
            ErrorCode.PARSER_CRASHED,
            diagnostic={"check": "docling_conversion"},
        )


def _pipeline(factory, provider, parser=None):
    global _CURRENT_FACTORY
    _CURRENT_FACTORY = factory
    return IndexingPipeline(
        factory,
        _FileStore(factory),
        parser or _Parser(factory),
        provider,
        _embedding(),
    )


def _target(
    preset: ChunkingPreset = ChunkingPreset.STRUCTURAL_BALANCED_V2,
    *,
    multimodal: bool = False,
    markdown_v2: bool = False,
    unified: bool = False,
):
    version = uuid4()
    target = uuid4()
    del markdown_v2
    parsing = "multimodal_local_v2" if multimodal else "text_local_v1"
    profile = profile_for_preset(preset, parsing)
    text_space_id = uuid4()
    cross_space_id = text_space_id if unified else uuid4()
    text_space = (
        replace(_multimodal_embedding(), model_profile_revision_id=uuid4())
        if unified
        else _embedding()
    )
    space_ids = {
        EmbeddingSpaceRole.TEXT_RETRIEVAL.value: text_space_id,
    }
    spaces = {
        EmbeddingSpaceRole.TEXT_RETRIEVAL.value: text_space,
    }
    if preset is ChunkingPreset.SEMANTIC_BALANCED_V1:
        space_ids[EmbeddingSpaceRole.SEMANTIC_ANALYSIS.value] = text_space_id
        spaces[EmbeddingSpaceRole.SEMANTIC_ANALYSIS.value] = text_space
    if multimodal:
        space_ids[EmbeddingSpaceRole.CROSS_MODAL_RETRIEVAL.value] = cross_space_id
        spaces[EmbeddingSpaceRole.CROSS_MODAL_RETRIEVAL.value] = (
            text_space if unified else _multimodal_embedding()
        )
    return IndexingTarget(
        job_id=uuid4(),
        indexed_document_version_id=target,
        workspace_id=WORKSPACE,
        kb_id=uuid4(),
        document_id=uuid4(),
        document_version_id=version,
        index_revision_id=uuid4(),
        embedding_space_id=text_space_id,
        source_change_seq=1,
        storage_uri=f"local-source://{WORKSPACE}/{'a' * 64}",
        checksum_sha256=hashlib.sha256(CONTENT).hexdigest(),
        size_bytes=len(CONTENT),
        original_filename="guide.txt",
        media_type="text/plain",
        parser_config=profile.parser_config,
        chunking_config=profile.chunking_config,
        embedding_space=text_space,
        enrichment_config=profile.enrichment_config,
        representation_config=profile.representation_config,
        embedding_space_ids=space_ids,
        embedding_spaces=spaces,
    )


def _embedding():
    return EmbeddingSpaceDefinition(
        provider_identity="alibaba-cloud-model-studio-qwen",
        endpoint_identity="alibaba-model-studio-beijing-embedding",
        requested_model="qwen3.7-text-embedding",
        resolved_model="qwen3.7-text-embedding",
        model_version="qwen3.7-text-embedding",
        deployment_revision=None,
        dimension=1024,
        distance_metric="cosine",
        vector_data_type="float32",
        normalization="l2",
        configuration_fingerprint="sha256:c135",
        tokenizer_fingerprint=None,
        compatibility_fingerprint="sha256:7bd7",
    )


def _vector():
    return (1.0,) + (0.0,) * 1023


def _multimodal_embedding():
    return replace(
        _embedding(),
        endpoint_identity="alibaba-model-studio-beijing-multimodal-embedding",
        requested_model="tongyi-embedding-vision-flash-2026-03-06",
        resolved_model="tongyi-embedding-vision-flash-2026-03-06",
        model_version="tongyi-embedding-vision-flash-2026-03-06",
        dimension=768,
        configuration_fingerprint="sha256:mm-config",
        compatibility_fingerprint="sha256:mm-compat",
    )


def _multimodal_vector():
    return (1.0,) + (0.0,) * 767


_CURRENT_FACTORY = None


if __name__ == "__main__":
    unittest.main()
