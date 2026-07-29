from __future__ import annotations

import asyncio
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
from rag_kb.adapters import FixedPgVectorSpace
from rag_kb.document_processing import (
    DOCLING_ENRICHMENT_CONFIG,
    DOCLING_REPRESENTATION_CONFIG,
    STRUCTURAL_CHUNKING_CONFIG_V3,
    count_chunk_tokens,
    index_profile,
    public_parsing_descriptor,
    profile_for_preset,
)
from rag_kb.domain import (
    ChunkingPreset,
    ContentModality,
    EmbeddingBatch,
    EmbeddingSpaceDefinition,
    ErrorCode,
    IndexingCommand,
    IndexingExecutionError,
    IndexingPhase,
    IndexingTarget,
    PromotionReason,
    PromotionResult,
    PromotionStatus,
    ParserExecutionError,
    ParsingPreset,
    PersistedVectorRepresentation,
    SourceFileIdentity,
    stable_chunk_id,
    stable_vector_id,
    validate_embedding_vector,
)
from rag_kb.indexing import IndexingPipeline


WORKSPACE = UUID("01900000-0000-7000-8000-000000000401")
CONTENT = b"first\n\nsecond"


class IndexingDomainTests(unittest.TestCase):
    def test_index_profile_uses_one_fixed_token_chunking_identity(self) -> None:
        profile = index_profile()

        self.assertEqual(
            profile.chunking_config["profile"],
            "structural_by_title_token_v3",
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
            profile.parser_config["profile"], "docling_text_local_v1"
        )
        self.assertNotIn("max_characters", STRUCTURAL_CHUNKING_CONFIG_V3)
        self.assertNotIn("new_after_n_chars", STRUCTURAL_CHUNKING_CONFIG_V3)

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
                "profile": "docling_multimodal_local_v2",
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

    def test_fixed_space_and_output_validation_fail_closed(self) -> None:
        expected = _embedding()
        adapter = FixedPgVectorSpace(expected)
        cross_modal = FixedPgVectorSpace(_multimodal_embedding())
        self.assertEqual(adapter.physical_table, "vector_record_1024")
        self.assertEqual(cross_modal.physical_table, "vector_record_768")
        adapter.require_compatible(expected, expected)
        with self.assertRaises(IndexingExecutionError) as mismatch:
            adapter.require_compatible(
                expected, replace(expected, configuration_fingerprint="sha256:changed")
            )
        self.assertEqual(mismatch.exception.code, ErrorCode.EMBEDDING_SPACE_MISMATCH)

        validate_embedding_vector(_vector(), expected)
        for invalid in ((_vector()[:-1]), tuple(0.0 for _ in range(1024))):
            with self.assertRaises(IndexingExecutionError) as response:
                validate_embedding_vector(invalid, expected)
            self.assertEqual(response.exception.code, ErrorCode.EMBEDDING_RESPONSE_INVALID)

class IndexingPipelineTests(unittest.IsolatedAsyncioTestCase):
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
            FixedPgVectorSpace(_embedding()),
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

    async def test_partial_embedding_failure_stays_non_serving_and_replay_converges(self) -> None:
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
        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.inputs, [("[body]\nBeta\n\nsecond",)])

    async def test_multimodal_retry_reuses_text_and_only_repurchases_missing_native_image(
        self,
    ) -> None:
        repository = _Repository(_target(multimodal=True))
        factory = _Factory(repository)
        text_provider = _Provider(factory)
        text_provider.max_batch_size = 10
        visual_provider = _MultimodalProvider(factory, fail_call=2)
        visual_provider.max_batch_size = 1
        global _CURRENT_FACTORY
        _CURRENT_FACTORY = factory
        pipeline = IndexingPipeline(
            factory,
            _FileStore(factory),
            _MultimodalParser(factory, document=_table_and_picture_document()),
            text_provider,
            FixedPgVectorSpace(_embedding()),
            asset_store=_AssetStore(factory),
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
        self.assertEqual(text_provider.calls, 0)
        self.assertEqual(visual_provider.image_calls, 1)
        self.assertEqual(len(repository.vectors), 3)

    async def test_persisted_representation_rejects_wrong_target_space_and_kind(
        self,
    ) -> None:
        repository = _Repository(_target())
        first_chunk_id = stable_chunk_id(
            repository.target.indexed_document_version_id,
            profile_fingerprint="f" * 64,
            unit_key="not-a-persisted-unit",
        )
        wrong_space = uuid4()
        repository.persisted_override = (
            PersistedVectorRepresentation(
                id=stable_vector_id(
                    repository.target.embedding_space_id, first_chunk_id
                ),
                index_chunk_id=first_chunk_id,
                embedding_space_id=wrong_space,
                representation_kind="text",
            ),
            PersistedVectorRepresentation(
                id=stable_vector_id(
                    repository.target.embedding_space_id, first_chunk_id
                ),
                index_chunk_id=uuid4(),
                embedding_space_id=repository.target.embedding_space_id,
                representation_kind="text",
            ),
            PersistedVectorRepresentation(
                id=stable_vector_id(
                    repository.target.embedding_space_id, first_chunk_id
                ),
                index_chunk_id=first_chunk_id,
                embedding_space_id=repository.target.embedding_space_id,
                representation_kind="native_image",
            ),
        )
        factory = _Factory(repository)
        provider = _Provider(factory)
        pipeline = _pipeline(factory, provider)

        result = await pipeline.execute(
            IndexingCommand(
                repository.target.job_id,
                repository.target.indexed_document_version_id,
            )
        )

        self.assertEqual(result.status, "ready")
        self.assertEqual(provider.calls, 2)

    async def test_reused_representation_still_enforces_stable_chunk_cas(
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

        with self.assertRaises(IndexingExecutionError) as failure:
            await pipeline.execute(command)

        self.assertEqual(
            failure.exception.code,
            ErrorCode.INDEX_CHUNK_PLAN_MISMATCH,
        )
        self.assertEqual(provider.calls, 0)

    async def test_reused_chunk_cas_is_split_into_safe_batches(self) -> None:
        repository = _Repository(_target())
        factory = _Factory(repository)
        provider = _Provider(factory, fail_call=3)
        provider.max_batch_size = 2
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
        repository.chunk_only_batch_sizes = []
        result = await pipeline.execute(command)

        self.assertEqual(result.status, "ready")
        self.assertEqual(repository.chunk_only_batch_sizes, [2, 2])
        self.assertEqual(provider.calls, 1)

    async def test_completed_replay_compensates_interrupted_promotion(self) -> None:
        repository = _Repository(_target())
        repository.status = "completed"
        repository.chunks = {0: object(), 1: object()}
        factory = _Factory(repository)
        provider = _Provider(factory)
        pipeline = _pipeline(factory, provider)
        command = IndexingCommand(
            repository.target.job_id,
            repository.target.indexed_document_version_id,
        )

        result = await pipeline.execute(command)

        self.assertTrue(result.replayed)
        self.assertEqual(result.serving_status, "serving")
        self.assertEqual(repository.serving, "serving")
        self.assertEqual(provider.calls, 0)

    async def test_parser_failure_persists_parser_phase_and_stable_code(self) -> None:
        repository = _Repository(_target())
        factory = _Factory(repository)
        pipeline = IndexingPipeline(
            factory,
            _FileStore(factory),
            _FailingParser(factory),
            _Provider(factory),
            FixedPgVectorSpace(_embedding()),
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

    async def test_semantic_retry_reuses_plan_without_analysis_embedding(self) -> None:
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
        self.assertEqual(provider.calls, result.chunk_count)
        self.assertEqual(
            [repository.chunks[index].ordinal for index in repository.chunks],
            list(range(result.chunk_count)),
        )

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

    async def test_scanned_surface_probe_does_not_block_event_loop(self) -> None:
        repository = _Repository(_target(multimodal=True))
        factory = _Factory(repository)
        text_provider = _Provider(factory)
        visual_provider = _MultimodalProvider(factory)
        global _CURRENT_FACTORY
        _CURRENT_FACTORY = factory
        pipeline = IndexingPipeline(
            factory,
            _FileStore(factory),
            _MultimodalParser(factory),
            text_provider,
            FixedPgVectorSpace(_embedding()),
            asset_store=_AssetStore(factory),
            multimodal_embedding_provider=visual_provider,
        )
        command = IndexingCommand(
            repository.target.job_id,
            repository.target.indexed_document_version_id,
        )
        loop_thread = threading.get_ident()
        probe_entered = threading.Event()
        probe_release = threading.Event()
        probe_thread: list[int] = []

        def blocked_probe(source):
            del source
            probe_thread.append(threading.get_ident())
            probe_entered.set()
            if not probe_release.wait(timeout=2):
                raise AssertionError("event loop did not release scanned-page probe")
            return frozenset()

        async def release_after_probe_starts() -> None:
            while not probe_entered.is_set():
                await asyncio.sleep(0)
            probe_release.set()

        release_task = asyncio.create_task(release_after_probe_starts())
        try:
            with patch.object(
                pipeline_module,
                "scanned_surfaces",
                side_effect=blocked_probe,
            ):
                result = await pipeline.execute(command)
            await release_task
        finally:
            probe_release.set()
            if not release_task.done():
                release_task.cancel()
                await asyncio.gather(release_task, return_exceptions=True)

        self.assertEqual(result.status, "ready")
        self.assertEqual(len(probe_thread), 1)
        self.assertNotEqual(probe_thread[0], loop_thread)


class _Factory:
    def __init__(self, repository) -> None:
        self.repository = repository
        self.active = False

    def __call__(self, *, purpose, mode):
        del purpose, mode
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
        self.persisted_override = None
        self.chunk_only_batch_sizes = []

    async def prepare(self, command):
        self._active()
        if command.job_id != self.target.job_id:
            return None
        if self.status == "completed":
            return replace(self.target, already_complete=True)
        self.status = "running"
        self.failure = None
        return self.target

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

    async def get_chunk_plan(self, command):
        del command
        self._active()
        return self.plan

    async def create_or_get_chunk_plan(self, command, proposed):
        del command
        self._active()
        if self.plan is None:
            self.plan = proposed
        if self.plan != proposed:
            raise AssertionError("chunk plan mismatch")
        return self.plan

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

    async def create_or_get_artifact_manifest(self, command, proposed):
        del command
        self._active()
        if self.manifest is None:
            self.manifest = proposed
        return self.manifest

    async def list_persisted_representations(self, command, *, limit):
        del command
        self._active()
        records = (
            self.persisted_override
            if self.persisted_override is not None
            else tuple(self.vectors.values())
        )
        if len(records) > limit:
            raise AssertionError("persisted representation limit exceeded")
        return records

    async def upsert_batch(self, command, chunks, vectors):
        del command
        self._active()
        if chunks and not vectors:
            self.chunk_only_batch_sizes.append(len(chunks))
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

    async def parse(self, source, *, preset):
        if self.factory.active:
            raise AssertionError("parser ran inside transaction")
        del source, preset
        self.calls += 1
        return self.document or _text_document()


class _MultimodalParser:
    def __init__(self, factory, *, document=None) -> None:
        self.factory = factory
        self.calls = 0
        self.presets = []
        self.document = document

    async def parse(self, source, *, preset):
        if self.factory.active:
            raise AssertionError("parser ran inside transaction")
        del source
        self.calls += 1
        self.presets.append(preset)
        return self.document or _visual_document()


class _SemanticParser:
    def __init__(self, factory) -> None:
        self.factory = factory
        self.calls = 0

    async def parse(self, source, *, preset):
        if self.factory.active:
            raise AssertionError("parser ran inside transaction")
        del source, preset
        self.calls += 1
        return _analysis_document()


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
    def __init__(self, factory, *, fail_call=None) -> None:
        self.factory = factory
        self.embedding_space = _embedding()
        self.max_batch_size = 1
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
        self.fail_call = fail_call

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

    async def put(self, identity, content, checksum):
        if self.factory.active:
            raise AssertionError("asset I/O ran inside transaction")
        self.writes.append((identity, content, checksum))


class _FailingParser:
    def __init__(self, factory) -> None:
        self.factory = factory

    async def parse(self, source, *, preset):
        del source, preset
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
        FixedPgVectorSpace(_embedding()),
    )


def _target(
    preset: ChunkingPreset = ChunkingPreset.STRUCTURAL_BALANCED_V2,
    *,
    multimodal: bool = False,
    markdown_v2: bool = False,
):
    version = uuid4()
    target = uuid4()
    del markdown_v2
    parsing = "multimodal_local_v2" if multimodal else "text_local_v1"
    profile = profile_for_preset(preset, parsing)
    cross_space_id = uuid4()
    return IndexingTarget(
        job_id=uuid4(),
        indexed_document_version_id=target,
        workspace_id=WORKSPACE,
        kb_id=uuid4(),
        document_id=uuid4(),
        document_version_id=version,
        index_revision_id=uuid4(),
        embedding_space_id=uuid4(),
        source_change_seq=1,
        storage_uri=f"local-source://{WORKSPACE}/{'a' * 64}",
        checksum_sha256=hashlib.sha256(CONTENT).hexdigest(),
        size_bytes=len(CONTENT),
        original_filename="guide.txt",
        media_type="text/plain",
        parser_config=profile.parser_config,
        chunking_config=profile.chunking_config,
        embedding_space=_embedding(),
        enrichment_config=profile.enrichment_config,
        representation_config=profile.representation_config,
        embedding_space_ids=(
            {"cross_modal_retrieval": cross_space_id} if multimodal else {}
        ),
        embedding_spaces=(
            {"cross_modal_retrieval": _multimodal_embedding()}
            if multimodal
            else {}
        ),
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
