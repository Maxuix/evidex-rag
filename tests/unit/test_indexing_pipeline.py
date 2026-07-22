from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace
from uuid import UUID, uuid4

from rag_kb.adapters import FixedPgVectorSpace
from rag_kb.document_processing import (
    UNSTRUCTURED_CHUNKING_CONFIG,
    index_profile,
    profile_for_preset,
)
from rag_kb.domain import (
    ChunkingPreset,
    EmbeddingBatch,
    EmbeddingSpaceDefinition,
    ErrorCode,
    IndexChunkDraft,
    IndexingCommand,
    IndexingExecutionError,
    IndexingPhase,
    IndexingTarget,
    PromotionReason,
    PromotionResult,
    PromotionStatus,
    ProcessedDocument,
    ParsedDocument,
    ParsedElement,
    ParsedAssetDraft,
    ContentModality,
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
            "unstructured_by_title_token_v2",
        )
        self.assertEqual(
            (
                profile.chunking_config["max_tokens"],
                profile.chunking_config["new_after_n_tokens"],
                profile.chunking_config["tokenizer"],
            ),
            (800, 600, "cl100k_base"),
        )
        self.assertNotIn("max_characters", UNSTRUCTURED_CHUNKING_CONFIG)
        self.assertNotIn("new_after_n_chars", UNSTRUCTURED_CHUNKING_CONFIG)

    def test_stable_chunk_and_vector_business_keys(self) -> None:
        target = uuid4()
        space = uuid4()
        first = stable_chunk_id(target, 3)
        self.assertEqual(first, stable_chunk_id(target, 3))
        self.assertNotEqual(first, stable_chunk_id(target, 4))
        self.assertEqual(
            stable_vector_id(space, first), stable_vector_id(space, first)
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
    async def test_multimodal_execution_persists_manifest_assets_and_both_spaces(self) -> None:
        repository = _Repository(_target(multimodal=True))
        factory = _Factory(repository)
        text_provider = _Provider(factory)
        text_provider.max_batch_size = 10
        visual_provider = _MultimodalProvider(factory)
        asset_store = _AssetStore(factory)
        global _CURRENT_FACTORY
        _CURRENT_FACTORY = factory
        pipeline = IndexingPipeline(
            factory,
            _FileStore(factory),
            _MultimodalProcessor(factory),
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
        self.assertEqual(repository.manifest.representation_count, 3)
        self.assertEqual(len(asset_store.writes), 1)
        self.assertEqual(text_provider.calls, 1)
        self.assertEqual(visual_provider.image_calls, 1)
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
            [11, 12],
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
        result = await pipeline.execute(command)
        self.assertEqual((result.status, result.chunk_count), ("ready", 2))
        self.assertEqual(result.serving_status, "serving")
        self.assertEqual(len(repository.chunks), 2)
        self.assertEqual(len(repository.vectors), 2)

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
            _FailingProcessor(factory),
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
        pipeline = _pipeline(factory, provider)
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

    async def create_or_get_artifact_manifest(self, command, proposed):
        del command
        self._active()
        if self.manifest is None:
            self.manifest = proposed
        return self.manifest

    async def upsert_batch(self, command, chunks, vectors):
        del command
        self._active()
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.chunks[chunk.ordinal] = chunk
            self.vectors[vector.index_chunk_id] = vector
        return True

    async def complete(self, command, *, expected_chunks):
        del command
        self._active()
        if len(self.chunks) != expected_chunks or len(self.vectors) != expected_chunks:
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


class _Processor:
    def __init__(self, factory) -> None:
        self.factory = factory

    async def process(self, source):
        if self.factory.active:
            raise AssertionError("parser ran inside transaction")
        del source
        drafts = tuple(
            IndexChunkDraft(
                ordinal=ordinal,
                text=text,
                token_count=ordinal + 11,
                source_location={
                    "page_start": ordinal + 1,
                    "page_end": ordinal + 1,
                },
                hierarchy={"titles": []},
                processing_metadata={"integration": "test"},
                content_sha256=hashlib.sha256(text.encode()).hexdigest(),
            )
            for ordinal, text in enumerate(("first", "second"))
        )
        return ProcessedDocument(
            chunks=drafts,
            extracted_character_count=sum(len(draft.text) for draft in drafts),
        )

    async def partition(self, source):
        if self.factory.active:
            raise AssertionError("parser ran inside transaction")
        del source
        elements = tuple(
            ParsedElement(
                ordinal=ordinal,
                text=("word " * 120 + f"topic{ordinal}").strip(),
                token_count=0,
                category="NarrativeText",
                source_location={},
                hierarchy={},
                is_title=False,
                is_table=False,
            )
            for ordinal in range(7)
        )
        return ParsedDocument(
            elements=elements,
            extracted_character_count=sum(len(item.text) for item in elements),
        )


class _MultimodalProcessor:
    def __init__(self, factory) -> None:
        self.factory = factory

    async def partition_multimodal(self, source):
        if self.factory.active:
            raise AssertionError("parser ran inside transaction")
        del source
        content = b"fixture-image"
        checksum = hashlib.sha256(content).hexdigest()
        asset = ParsedAssetDraft(
            asset_key="b" * 64,
            kind="docx_picture",
            media_type="image/png",
            content=content,
            content_sha256=checksum,
            width=120,
            height=80,
            source_location={"block_ordinal": 1},
            processing_metadata={},
        )
        return ParsedDocument(
            elements=(
                ParsedElement(
                    ordinal=0,
                    text="body evidence",
                    token_count=2,
                    category="NarrativeText",
                    source_location={"block_ordinal": 0},
                    hierarchy={},
                    element_key="text-element",
                ),
                ParsedElement(
                    ordinal=1,
                    text="author diagram caption",
                    token_count=3,
                    category="Image",
                    source_location={"block_ordinal": 1},
                    hierarchy={},
                    element_key="image-element",
                    asset_key=asset.asset_key,
                ),
            ),
            extracted_character_count=36,
            assets=(asset,),
        )


class _Provider:
    def __init__(self, factory, *, fail_call=None) -> None:
        self.factory = factory
        self.embedding_space = _embedding()
        self.max_batch_size = 1
        self.calls = 0
        self.fail_call = fail_call

    async def embed_documents(self, texts):
        if self.factory.active:
            raise AssertionError("provider ran inside transaction")
        self.calls += 1
        if self.calls == self.fail_call:
            raise IndexingExecutionError(
                ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE,
                phase=IndexingPhase.EMBEDDING,
                diagnostic={"retry_exhausted": True},
            )
        return EmbeddingBatch(tuple(_vector() for _ in texts))


class _MultimodalProvider:
    def __init__(self, factory) -> None:
        self.factory = factory
        self.embedding_space = _multimodal_embedding()
        self.max_batch_size = 20
        self.image_calls = 0

    async def embed_images(self, images):
        if self.factory.active:
            raise AssertionError("provider ran inside transaction")
        self.image_calls += 1
        return EmbeddingBatch(tuple(_multimodal_vector() for _ in images))


class _AssetStore:
    def __init__(self, factory) -> None:
        self.factory = factory
        self.writes = []

    async def put(self, identity, content, checksum):
        if self.factory.active:
            raise AssertionError("asset I/O ran inside transaction")
        self.writes.append((identity, content, checksum))


class _FailingProcessor:
    def __init__(self, factory) -> None:
        self.factory = factory

    async def process(self, source):
        del source
        if self.factory.active:
            raise AssertionError("parser ran inside transaction")
        from rag_kb.domain import ParserExecutionError

        raise ParserExecutionError(
            ErrorCode.PARSER_CRASHED,
            diagnostic={"check": "unstructured_loader"},
        )


def _pipeline(factory, provider):
    global _CURRENT_FACTORY
    _CURRENT_FACTORY = factory
    return IndexingPipeline(
        factory,
        _FileStore(factory),
        _Processor(factory),
        provider,
        FixedPgVectorSpace(_embedding()),
    )


def _target(
    preset: ChunkingPreset = ChunkingPreset.STRUCTURAL_BALANCED_V2,
    *,
    multimodal: bool = False,
):
    version = uuid4()
    target = uuid4()
    profile = profile_for_preset(
        preset, "multimodal_local_v1" if multimodal else "text_local_v1"
    )
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
