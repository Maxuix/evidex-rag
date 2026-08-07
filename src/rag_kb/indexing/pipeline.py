"""One retry-safe indexing execution with no external I/O inside DB transactions."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from collections.abc import Awaitable, Callable, Mapping
from typing import TypeVar
from uuid import UUID

from docling_core.types.doc import DoclingDocument

from rag_kb.domain import (
    ChunkAssemblyDraft,
    ChunkingStrategyKind,
    EmbeddingSpaceDefinition,
    EmbeddingSpaceRole,
    ErrorCode,
    ContentModality,
    CompositeEvidenceDraft,
    FileStoreError,
    IndexChunkWrite,
    IndexChunkLexicalWrite,
    IndexLexicalManifest,
    IndexChunkAssetRelationWrite,
    IndexArtifactManifest,
    IndexAssetIdentity,
    IndexAssetWrite,
    IndexingCancelled,
    IndexingCommand,
    IndexingExecutionError,
    IndexingPhase,
    IndexingResult,
    IndexingTarget,
    InvalidStorageIdentityError,
    ParsedAssetDraft,
    ParserExecutionError,
    ParserLimits,
    ParserSource,
    ParsingPreset,
    PromotionCommand,
    SemanticUnit,
    SourceFileMissingError,
    VectorRecordWrite,
    ImageEmbeddingInput,
    stable_asset_id,
    stable_chunk_id,
    stable_relation_id,
    stable_vector_id,
    validate_embedding_vector,
)
from rag_kb.document_processing.composite_text import with_composite_embedding_text
from rag_kb.document_processing.profiles import (
    DOCLING_ENRICHMENT_CONFIG,
    DOCLING_MULTIMODAL_PARSER_CONFIG,
    DOCLING_REPRESENTATION_CONFIG,
    SEMANTIC_CHUNKING_CONFIG,
    profile_fingerprint,
    parsing_preset,
    resolve,
)
from rag_kb.document_processing.tokenization import count_chunk_tokens
from rag_kb.document_processing.docling import (
    asset_manifest_hash,
    assemble_semantic_chunks,
    assemble_structural,
    composite_evidence,
    docling_item_sequence_hash,
    docling_semantic_units,
    docling_unit_sequence_hash,
    extract_docling_assets,
    relate_assets_to_chunks,
)
from rag_kb.document_processing.semantic_boundaries import (
    build_chunk_plan,
    validate_plan,
)
from rag_kb.document_processing.lexical import (
    LEXICAL_ANALYZER_VERSION,
    analyze_document,
    lexical_manifest_hash,
)
from rag_kb.indexing.promotion import CandidatePromotionService
from rag_kb.indexing.embedding_spaces import require_compatible_embedding_spaces
from rag_kb.ports.files import IndexAssetStore, SourceFileStore
from rag_kb.ports.model_api import EmbeddingModelAdapter, MultimodalEmbeddingAdapter
from rag_kb.ports.parsing import DocumentParseResult, DocumentParser
from rag_kb.uow import UnitOfWork, UnitOfWorkFactory, UnitOfWorkPurpose, execute_in_transaction


ResultT = TypeVar("ResultT")
EmbeddingModelResolver = Callable[
    [EmbeddingSpaceDefinition], Awaitable[EmbeddingModelAdapter]
]
MultimodalEmbeddingModelResolver = Callable[
    [EmbeddingSpaceDefinition], Awaitable[MultimodalEmbeddingAdapter]
]
_LEXICAL_CAS_BATCH_SIZE = 250


class IndexingPipeline:
    def __init__(
        self,
        unit_of_work: UnitOfWorkFactory,
        file_store: SourceFileStore,
        document_parser: DocumentParser,
        embedding_provider: EmbeddingModelAdapter,
        embedding_space: EmbeddingSpaceDefinition,
        *,
        asset_store: IndexAssetStore | None = None,
        multimodal_embedding_provider: MultimodalEmbeddingAdapter | None = None,
        parser_limits: ParserLimits | None = None,
        embedding_model_resolver: EmbeddingModelResolver | None = None,
        multimodal_embedding_model_resolver: MultimodalEmbeddingModelResolver | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._file_store = file_store
        self._document_parser = document_parser
        self._embedding_provider = embedding_provider
        self._embedding_space = embedding_space
        self._asset_store = asset_store
        self._multimodal_embedding_provider = multimodal_embedding_provider
        self._parser_limits = parser_limits or ParserLimits()
        self._embedding_model_resolver = embedding_model_resolver
        self._multimodal_embedding_model_resolver = multimodal_embedding_model_resolver
        self._promotion = CandidatePromotionService(unit_of_work)

    async def execute(self, command: IndexingCommand) -> IndexingResult:
        phase = IndexingPhase.SOURCE_READ
        try:
            target = await self._prepare(command)
            if target is None:
                raise IndexingExecutionError(
                    ErrorCode.INDEX_TARGET_INVALID,
                    phase=phase,
                    diagnostic={"check": "job_target_mapping"},
                )
            strategy = self._require_revision_profile(target)
            embedding_provider = (
                await self._embedding_model_resolver(target.embedding_space)
                if self._embedding_model_resolver is not None
                and target.embedding_space.model_profile_revision_id is not None
                else self._embedding_provider
            )
            if strategy is ChunkingStrategyKind.SEMANTIC:
                self._require_semantic_space_role(target)
            if target.already_complete:
                promotion = await self._promotion.promote(
                    _promotion_command(command)
                )
                return IndexingResult(
                    command.job_id,
                    command.indexed_document_version_id,
                    "ready",
                    await self._stored_chunk_count(command),
                    replayed=True,
                    serving_status=promotion.status.value,
                )
            await self._discard_partial_assets(command, target)
            require_compatible_embedding_spaces(
                (
                    target.embedding_space
                    if target.embedding_space.model_profile_revision_id is not None
                    else self._embedding_space
                ),
                target.embedding_space,
                embedding_provider.embedding_space,
            )
            content = await self._read_source(target)

            source = ParserSource(
                original_filename=target.original_filename,
                media_type=target.media_type,
                content=content,
            )
            resolved_parsing = parsing_preset(target.parser_config)
            multimodal = resolved_parsing is ParsingPreset.MULTIMODAL_LOCAL_V2
            cross_space = None
            cross_provider = None
            if multimodal:
                cross_space, cross_provider = await self._require_multimodal_runtime(
                    target
                )
            parsed = await self._parse(
                command,
                source,
                preset=resolved_parsing,
            )
            document = parsed.document
            labels = dict(parsed.surface_labels)
            chunks = await self._chunks(
                command, target, document, strategy, labels, embedding_provider
            )
            result = await self._execute_current(
                command,
                target,
                document,
                chunks,
                cross_space,
                embedding_provider,
                cross_provider,
                labels,
                page_image_surfaces=parsed.page_image_surfaces,
                multimodal=multimodal,
            )
            promotion = await self._promotion.promote(_promotion_command(command))
            return IndexingResult(
                command.job_id,
                command.indexed_document_version_id,
                "ready",
                result,
                serving_status=promotion.status.value,
            )
        except IndexingCancelled:
            return IndexingResult(
                command.job_id,
                command.indexed_document_version_id,
                "cancelled",
                0,
                serving_status="retired",
            )
        except IndexingExecutionError as error:
            await self._record_failure(command, error)
            raise
        except Exception as error:
            failure = IndexingExecutionError(
                ErrorCode.INDEX_PERSISTENCE_FAILED,
                phase=phase,
                diagnostic={"operation": phase.value},
            )
            await self._record_failure(command, failure)
            raise failure from error

    async def _prepare(self, command: IndexingCommand) -> IndexingTarget | None:
        return await self._transaction(lambda uow: uow.indexing.prepare(command))

    async def _discard_partial_assets(
        self,
        command: IndexingCommand,
        target: IndexingTarget,
    ) -> None:
        if self._asset_store is None:
            return
        try:
            await self._asset_store.discard_target(
                target.workspace_id,
                target.indexed_document_version_id,
            )
        except (FileStoreError, OSError) as error:
            raise IndexingExecutionError(
                ErrorCode.INDEX_PERSISTENCE_FAILED,
                phase=IndexingPhase.SOURCE_READ,
                diagnostic={"operation": "discard_partial_assets"},
            ) from error
        changed = await self._transaction(
            lambda uow: uow.indexing.discard_partial_assets(command)
        )
        if not changed:
            raise IndexingCancelled

    async def _set_phase(self, command: IndexingCommand, phase: IndexingPhase) -> None:
        changed = await self._transaction(
            lambda uow: uow.indexing.set_phase(command, phase)
        )
        if not changed:
            raise IndexingCancelled

    async def _upsert(
        self,
        command: IndexingCommand,
        chunks: tuple[IndexChunkWrite, ...],
        vectors: tuple[VectorRecordWrite, ...],
    ) -> None:
        try:
            changed = await self._transaction(
                lambda uow: uow.indexing.upsert_batch(command, chunks, vectors)
            )
        except IndexingExecutionError:
            raise
        except Exception as error:
            raise IndexingExecutionError(
                ErrorCode.INDEX_PERSISTENCE_FAILED,
                phase=IndexingPhase.PERSISTING,
                diagnostic={"operation": "upsert_batch"},
            ) from error
        if not changed:
            raise IndexingCancelled

    async def _complete(
        self,
        command: IndexingCommand,
        *,
        expected_chunks: int,
    ) -> None:
        changed = await self._transaction(
            lambda uow: uow.indexing.complete(
                command, expected_chunks=expected_chunks
            )
        )
        if not changed:
            raise IndexingCancelled

    async def _record_failure(
        self,
        command: IndexingCommand,
        error: IndexingExecutionError,
    ) -> None:
        async def persist(uow: UnitOfWork) -> bool:
            return await uow.indexing.fail(
                command,
                phase=error.phase,
                error_code=error.code.value,
                error_detail=_safe_diagnostic(error.diagnostic),
            )

        await self._transaction(persist)

    async def _stored_chunk_count(self, command: IndexingCommand) -> int:
        return await self._transaction(lambda uow: uow.indexing.count_chunks(command))

    async def _transaction(
        self,
        operation: Callable[[UnitOfWork], Awaitable[ResultT]],
    ) -> ResultT:
        return await execute_in_transaction(
            self._unit_of_work,
            operation,
            purpose=UnitOfWorkPurpose.INDEXING,
        )

    async def _read_source(self, target: IndexingTarget) -> bytes:
        try:
            identity = self._file_store.parse_uri(target.storage_uri)
            if identity.workspace_id != target.workspace_id:
                raise InvalidStorageIdentityError("source workspace differs")
            content = await self._file_store.read_final(identity)
        except SourceFileMissingError as error:
            raise IndexingExecutionError(
                ErrorCode.SOURCE_FILE_MISSING,
                phase=IndexingPhase.SOURCE_READ,
                diagnostic={"check": "final_file"},
            ) from error
        except (FileStoreError, OSError) as error:
            raise IndexingExecutionError(
                ErrorCode.SOURCE_FILE_INTEGRITY,
                phase=IndexingPhase.SOURCE_READ,
                diagnostic={"check": "storage_identity"},
            ) from error
        if (
            len(content) != target.size_bytes
            or hashlib.sha256(content).hexdigest() != target.checksum_sha256
        ):
            raise IndexingExecutionError(
                ErrorCode.SOURCE_FILE_INTEGRITY,
                phase=IndexingPhase.SOURCE_READ,
                diagnostic={"check": "checksum_and_size"},
            )
        return content

    @staticmethod
    def _require_revision_profile(
        target: IndexingTarget,
    ) -> ChunkingStrategyKind:
        try:
            strategy = resolve(target.parser_config, target.chunking_config)
            if target.parser_config == DOCLING_MULTIMODAL_PARSER_CONFIG and (
                target.enrichment_config != DOCLING_ENRICHMENT_CONFIG
                or target.representation_config != DOCLING_REPRESENTATION_CONFIG
            ):
                raise ValueError("unknown multimodal enrichment profile")
            return strategy
        except ValueError as error:
            raise IndexingExecutionError(
                ErrorCode.INDEX_REVISION_INCOMPATIBLE,
                phase=IndexingPhase.SOURCE_READ,
                diagnostic={"check": "parser_chunking_profile"},
            ) from error

    async def _parse(
        self,
        command: IndexingCommand,
        source: ParserSource,
        *,
        preset: ParsingPreset,
    ) -> DocumentParseResult:
        """Convert the source exactly once for the whole job."""

        await self._set_phase(command, IndexingPhase.PARSING)
        try:
            return await self._document_parser.parse(source, preset=preset)
        except ParserExecutionError as error:
            raise IndexingExecutionError(
                error.code,
                phase=IndexingPhase.PARSING,
                diagnostic=error.diagnostic,
            ) from error
        except IndexingExecutionError:
            raise
        except Exception as error:
            raise IndexingExecutionError(
                ErrorCode.PARSER_CRASHED,
                phase=IndexingPhase.PARSING,
                diagnostic={"check": "parser_contract"},
            ) from error

    async def _chunks(
        self,
        command: IndexingCommand,
        target: IndexingTarget,
        document: DoclingDocument,
        strategy: ChunkingStrategyKind,
        surface_labels: Mapping[int, str],
        embedding_provider: EmbeddingModelAdapter,
    ) -> tuple[ChunkAssemblyDraft, ...]:
        """Apply the revision's chunking strategy to the converted document."""

        if strategy is ChunkingStrategyKind.STRUCTURAL:
            try:
                return assemble_structural(
                    document, self._parser_limits, surface_labels=surface_labels
                )
            except ParserExecutionError as error:
                raise IndexingExecutionError(
                    error.code,
                    phase=IndexingPhase.PARSING,
                    diagnostic=error.diagnostic,
                ) from error
        return await self._semantic_chunks(
            command, target, document, surface_labels, embedding_provider
        )

    async def _semantic_chunks(
        self,
        command: IndexingCommand,
        target: IndexingTarget,
        document: DoclingDocument,
        surface_labels: Mapping[int, str],
        embedding_provider: EmbeddingModelAdapter,
    ) -> tuple[ChunkAssemblyDraft, ...]:
        await self._set_phase(command, IndexingPhase.SEMANTIC_ANALYSIS)
        try:
            units = await asyncio.to_thread(
                docling_semantic_units,
                document,
                self._parser_limits,
                surface_labels=surface_labels,
            )
        except ParserExecutionError as error:
            raise IndexingExecutionError(
                error.code,
                phase=IndexingPhase.SEMANTIC_ANALYSIS,
                diagnostic=error.diagnostic,
            ) from error
        sequence_hash = await asyncio.to_thread(docling_unit_sequence_hash, units)
        fingerprint = self._fingerprint(target)
        plan_facts = {
            "indexed_document_version_id": target.indexed_document_version_id,
            "source_checksum_sha256": target.checksum_sha256,
            "profile_fingerprint": fingerprint,
        }
        requires_analysis = await asyncio.to_thread(
            _requires_semantic_analysis,
            units,
        )
        vectors = (
            await self._embed_analysis_units(target, units, embedding_provider)
            if requires_analysis
            else None
        )
        proposed = await asyncio.to_thread(
            build_chunk_plan,
            **plan_facts,
            units=units,
            vectors=vectors,
            sequence_hash=sequence_hash,
        )
        changed = await self._transaction(
            lambda uow: uow.indexing.save_chunk_plan(command, proposed)
        )
        if not changed:
            raise IndexingCancelled
        await asyncio.to_thread(
            validate_plan,
            proposed,
            **plan_facts,
            units=units,
            sequence_hash=sequence_hash,
        )
        return await asyncio.to_thread(
            self._assemble_semantic,
            document,
            units,
            proposed,
            surface_labels,
        )

    def _assemble_semantic(
        self,
        document: DoclingDocument,
        units,
        plan,
        surface_labels: Mapping[int, str],
    ) -> tuple[ChunkAssemblyDraft, ...]:
        try:
            return assemble_semantic_chunks(
                document,
                units,
                plan,
                self._parser_limits,
                surface_labels=surface_labels,
            )
        except ParserExecutionError as error:
            raise IndexingExecutionError(
                error.code,
                phase=IndexingPhase.SEMANTIC_ANALYSIS,
                diagnostic=error.diagnostic,
            ) from error

    @staticmethod
    def _fingerprint(target: IndexingTarget) -> str:
        return profile_fingerprint(
            target.parser_config,
            target.chunking_config,
            target.enrichment_config,
            target.representation_config,
        )

    async def _require_multimodal_runtime(self, target: IndexingTarget):
        """Reject an unusable multimodal revision before converting its source."""

        if self._asset_store is None:
            raise IndexingExecutionError(
                ErrorCode.INDEX_REVISION_INCOMPATIBLE,
                phase=IndexingPhase.SOURCE_READ,
                diagnostic={"check": "multimodal_runtime_dependencies"},
            )
        cross_space = target.embedding_spaces.get("cross_modal_retrieval")
        if cross_space is None or "cross_modal_retrieval" not in target.embedding_space_ids:
            raise IndexingExecutionError(
                ErrorCode.INDEX_REVISION_INCOMPATIBLE,
                phase=IndexingPhase.SOURCE_READ,
                diagnostic={"check": "cross_modal_space_role"},
            )
        provider = (
            await self._multimodal_embedding_model_resolver(cross_space)
            if self._multimodal_embedding_model_resolver is not None
            and cross_space.model_profile_revision_id is not None
            else self._multimodal_embedding_provider
        )
        if provider is None:
            raise IndexingExecutionError(
                ErrorCode.INDEX_REVISION_INCOMPATIBLE,
                phase=IndexingPhase.SOURCE_READ,
                diagnostic={"check": "multimodal_runtime_dependencies"},
            )
        require_compatible_embedding_spaces(
            cross_space,
            cross_space,
            provider.embedding_space,
        )
        return cross_space, provider

    @staticmethod
    def _require_semantic_space_role(target: IndexingTarget) -> None:
        """Require semantic analysis to use the revision's primary text space."""

        text_role = EmbeddingSpaceRole.TEXT_RETRIEVAL.value
        analysis_role = EmbeddingSpaceRole.SEMANTIC_ANALYSIS.value
        if (
            target.embedding_space_ids.get(text_role) != target.embedding_space_id
            or target.embedding_space_ids.get(analysis_role)
            != target.embedding_space_id
            or target.embedding_spaces.get(text_role) != target.embedding_space
            or target.embedding_spaces.get(analysis_role) != target.embedding_space
        ):
            raise IndexingExecutionError(
                ErrorCode.INDEX_REVISION_INCOMPATIBLE,
                phase=IndexingPhase.SOURCE_READ,
                diagnostic={"check": "semantic_analysis_space_role"},
            )

    async def _execute_current(
        self,
        command: IndexingCommand,
        target: IndexingTarget,
        document: DoclingDocument,
        assembled: tuple[ChunkAssemblyDraft, ...],
        cross_space,
        embedding_provider: EmbeddingModelAdapter,
        cross_provider: MultimodalEmbeddingAdapter | None,
        surface_labels: Mapping[int, str],
        *,
        page_image_surfaces: frozenset[int],
        multimodal: bool,
    ) -> int:
        cross_space_id = (
            target.embedding_space_ids["cross_modal_retrieval"]
            if multimodal
            else None
        )
        await self._set_phase(command, IndexingPhase.ASSET_EXTRACTION)
        if multimodal:
            assembly, extracted = await asyncio.to_thread(
                self._composite_evidence,
                target,
                document,
                assembled,
                surface_labels,
                page_image_surfaces,
            )
        else:
            try:
                draft = await asyncio.to_thread(
                    composite_evidence,
                    document,
                    assembled,
                    (),
                    (),
                    profile=_profile(target),
                    source_checksum_sha256=target.checksum_sha256,
                    limits=self._parser_limits,
                )
                assembly = replace(
                    draft,
                    units=with_composite_embedding_text(
                        draft.units, draft.relations
                    ),
                )
                extracted = ()
            except ParserExecutionError as error:
                raise IndexingExecutionError(
                    error.code,
                    phase=IndexingPhase.ENRICHMENT,
                    diagnostic=error.diagnostic,
                ) from error
        units = assembly.units
        relations = assembly.relations
        if not units:
            raise IndexingExecutionError(
                ErrorCode.PARSER_OUTPUT_INVALID,
                phase=IndexingPhase.PARSING,
                diagnostic={"check": "non_empty_units"},
            )
        referenced_asset_keys = {
            unit.asset_key for unit in units if unit.asset_key is not None
        } | {relation.asset_key for relation in relations}
        assets = tuple(
            asset
            for asset in extracted
            if asset.asset_key in referenced_asset_keys
        )
        asset_writes: list[IndexAssetWrite] = []
        if assets:
            assert self._asset_store is not None
        for asset in assets:
            identity = IndexAssetIdentity(
                target.workspace_id,
                target.indexed_document_version_id,
                asset.asset_key,
            )
            await self._asset_store.put(
                identity, asset.content, asset.content_sha256
            )
            asset_writes.append(
                IndexAssetWrite(
                    id=stable_asset_id(
                        target.indexed_document_version_id, asset.asset_key
                    ),
                    asset_key=asset.asset_key,
                    kind=asset.kind,
                    storage_uri=identity.storage_uri,
                    media_type=asset.media_type,
                    checksum_sha256=asset.content_sha256,
                    width=asset.width,
                    height=asset.height,
                    source_location=asset.source_location,
                    processing_metadata=asset.processing_metadata,
                )
            )
        changed = await self._transaction(
            lambda uow: uow.indexing.upsert_assets(command, tuple(asset_writes))
        )
        if not changed:
            raise IndexingCancelled

        fingerprint = self._fingerprint(target)
        asset_ids = {item.asset_key: item.id for item in asset_writes}
        chunks = tuple(
            self._unit_write(target, unit, fingerprint, asset_ids) for unit in units
        )
        chunk_ids = {unit.unit_key: chunk.id for unit, chunk in zip(units, chunks, strict=True)}
        relation_writes = tuple(
            IndexChunkAssetRelationWrite(
                id=stable_relation_id(
                    target.indexed_document_version_id,
                    chunk_ids[relation.chunk_unit_key],
                    asset_ids[relation.asset_key],
                    relation.relation_type.value,
                ),
                chunk_id=chunk_ids[relation.chunk_unit_key],
                visual_unit_id=chunk_ids[relation.visual_unit_key],
                asset_id=asset_ids[relation.asset_key],
                relation_type=relation.relation_type.value,
                confidence_micros=relation.confidence_micros,
                figure_label=relation.figure_label,
                ordinal=relation.ordinal,
                provenance=relation.provenance.value,
                evidence_group_key=relation.evidence_group_key,
            )
            for relation in relations
        )
        planned = self._representation_plan(
            chunks, units, target.embedding_space_id, cross_space_id, assets
        )
        if len(planned) > self._parser_limits.max_representations:
            raise IndexingExecutionError(
                ErrorCode.PARSER_RESOURCE_LIMIT,
                phase=IndexingPhase.ENRICHMENT,
                diagnostic={
                    "limit_name": "max_representations",
                    "limit": self._parser_limits.max_representations,
                },
            )
        proposed = self._manifest(
            target,
            document,
            assets,
            units,
            chunks,
            planned,
            relations,
            relation_writes,
            fingerprint,
        )
        changed = await self._transaction(
            lambda uow: uow.indexing.save_artifact_manifest(command, proposed)
        )
        if not changed:
            raise IndexingCancelled

        await self._embed_representations(
            command,
            target,
            assets,
            units,
            chunks,
            planned,
            cross_space,
            embedding_provider,
            cross_provider,
        )
        changed = await self._transaction(
            lambda uow: uow.indexing.upsert_relations(command, relation_writes)
        )
        if not changed:
            raise IndexingCancelled
        await self._set_phase(command, IndexingPhase.VALIDATING)
        await self._persist_lexical(
            command,
            target,
            chunks,
            allowed_chunk_ids=frozenset(
                UUID(item["unit_id"])
                for item in planned
                if item["space_role"] == "text_retrieval"
            ),
        )
        await self._complete(command, expected_chunks=len(chunks))
        return len(chunks)

    async def _persist_lexical(
        self,
        command: IndexingCommand,
        target: IndexingTarget,
        chunks: tuple[IndexChunkWrite, ...],
        *,
        allowed_chunk_ids: frozenset[UUID],
    ) -> None:
        rows = await asyncio.to_thread(
            _lexical_rows,
            chunks,
            allowed_chunk_ids,
        )
        for offset in range(0, len(rows), _LEXICAL_CAS_BATCH_SIZE):
            batch = rows[offset : offset + _LEXICAL_CAS_BATCH_SIZE]
            changed = await self._transaction(
                lambda uow, batch=batch: uow.indexing.upsert_lexical_rows(
                    command, batch
                )
            )
            if not changed:
                raise IndexingCancelled
        manifest = IndexLexicalManifest(
            indexed_document_version_id=target.indexed_document_version_id,
            analyzer_version=LEXICAL_ANALYZER_VERSION,
            lexical_chunk_count=len(rows),
            lexical_manifest_hash=lexical_manifest_hash(
                LEXICAL_ANALYZER_VERSION,
                (
                    (item.index_chunk_id, item.lexical_text_hash)
                    for item in rows
                ),
            ),
        )
        changed = await self._transaction(
            lambda uow: uow.indexing.complete_lexical_manifest(
                command, manifest
            )
        )
        if not changed:
            raise IndexingCancelled

    def _composite_evidence(
        self,
        target: IndexingTarget,
        document: DoclingDocument,
        assembled: tuple[ChunkAssemblyDraft, ...],
        surface_labels: Mapping[int, str],
        page_image_surfaces: frozenset[int],
    ) -> tuple[CompositeEvidenceDraft, tuple[ParsedAssetDraft, ...]]:
        """Derive assets, relations and evidence units from the one conversion."""

        try:
            assets = extract_docling_assets(
                document,
                self._parser_limits,
                page_image_surfaces=page_image_surfaces,
                surface_labels=surface_labels,
            )
            relations = relate_assets_to_chunks(
                document, assembled, assets, self._parser_limits
            )
            draft = composite_evidence(
                document,
                assembled,
                assets,
                relations,
                profile=_profile(target),
                source_checksum_sha256=target.checksum_sha256,
                limits=self._parser_limits,
            )
        except ParserExecutionError as error:
            raise IndexingExecutionError(
                error.code,
                phase=IndexingPhase.ENRICHMENT,
                diagnostic=error.diagnostic,
            ) from error
        return (
            replace(
                draft,
                units=with_composite_embedding_text(draft.units, draft.relations),
            ),
            assets,
        )

    @staticmethod
    def _unit_write(target, unit, fingerprint, asset_ids) -> IndexChunkWrite:
        unit_id = stable_chunk_id(
            target.indexed_document_version_id,
            profile_fingerprint=fingerprint,
            unit_key=unit.unit_key,
        )
        return IndexChunkWrite(
            id=unit_id,
            ordinal=unit.ordinal,
            unit_key=unit.unit_key,
            modality=unit.modality,
            index_asset_id=asset_ids.get(unit.asset_key),
            evidence_group_key=unit.evidence_group_key,
            relations={"related_unit_keys": list(unit.related_unit_keys)},
            content=unit.content,
            content_hash=hashlib.sha256(unit.content.encode()).hexdigest(),
            token_count=unit.token_count,
            source_location=unit.source_location,
            hierarchy=unit.hierarchy,
            source_metadata={
                "document_id": str(target.document_id),
                "document_version_id": str(target.document_version_id),
                "original_filename": target.original_filename,
                "media_type": target.media_type,
                "checksum_sha256": target.checksum_sha256,
                "processing": unit.processing_metadata,
            },
            embedding_text=unit.embedding_text,
            embedding_text_hash=unit.embedding_text_hash,
        )

    @staticmethod
    def _representation_plan(
        chunks, units, text_space_id, cross_space_id, extracted
    ) -> tuple[dict, ...]:
        assets = {item.asset_key: item for item in extracted}
        plan: list[dict] = []
        for chunk, unit in zip(chunks, units, strict=True):
            if unit.modality is ContentModality.IMAGE:
                plan.append(
                    {
                        "unit_id": str(chunk.id),
                        "unit_key": unit.unit_key,
                        "space_role": "cross_modal_retrieval",
                        "space_id": str(cross_space_id),
                        "representation_kind": "native_image",
                        "required": True,
                        "asset_key": unit.asset_key,
                    }
                )
            else:
                kind = (
                    "table_text"
                    if unit.modality is ContentModality.TABLE
                    else unit.required_representations[0]
                )
                plan.append(
                    {
                        "unit_id": str(chunk.id),
                        "unit_key": unit.unit_key,
                        "space_role": "text_retrieval",
                        "space_id": str(text_space_id),
                        "representation_kind": kind,
                        "required": True,
                    }
                )
                if unit.modality is ContentModality.TABLE and unit.asset_key in assets:
                    plan.append(
                        {
                            "unit_id": str(chunk.id),
                            "unit_key": unit.unit_key,
                            "space_role": "cross_modal_retrieval",
                            "space_id": str(cross_space_id),
                            "representation_kind": "table_image",
                            "required": False,
                            "asset_key": unit.asset_key,
                        }
                    )
        return tuple(plan)

    @staticmethod
    def _manifest(
        target,
        document,
        extracted,
        units,
        chunks,
        planned,
        relations,
        relation_writes,
        fingerprint,
    ):
        unit_plan = tuple(
            {
                "unit_id": str(chunk.id),
                "unit_key": unit.unit_key,
                "ordinal": unit.ordinal,
                "modality": unit.modality.value,
                "asset_key": unit.asset_key,
                "evidence_group_key": unit.evidence_group_key,
                "related_unit_keys": list(unit.related_unit_keys),
                "embedding_text_hash": unit.embedding_text_hash,
            }
            for chunk, unit in zip(chunks, units, strict=True)
        )
        relation_plan = tuple(
            {
                "relation_id": str(write.id),
                "chunk_id": str(write.chunk_id),
                "visual_unit_id": str(write.visual_unit_id),
                "asset_id": str(write.asset_id),
                "relation_type": relation.relation_type.value,
                "confidence_micros": relation.confidence_micros,
                "figure_label": relation.figure_label,
                "ordinal": relation.ordinal,
                "provenance": relation.provenance.value,
                "evidence_group_key": relation.evidence_group_key,
            }
            for relation, write in zip(relations, relation_writes, strict=True)
        )
        relation_manifest_hash = hashlib.sha256(
            json.dumps(
                relation_plan,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        payload = {
            "source_checksum_sha256": target.checksum_sha256,
            "profile_fingerprint": fingerprint,
            "element_sequence_hash": docling_item_sequence_hash(document),
            "asset_manifest_hash": asset_manifest_hash(extracted),
            "unit_plan": unit_plan,
            "representation_matrix": planned,
            "relation_plan": relation_plan,
            "relation_manifest_hash": relation_manifest_hash,
        }
        manifest_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        return IndexArtifactManifest(
            indexed_document_version_id=target.indexed_document_version_id,
            unit_count=len(units),
            asset_count=len(extracted),
            representation_count=len(planned),
            relation_count=len(relation_plan),
            manifest_hash=manifest_hash,
            **payload,
        )

    async def _embed_representations(
        self,
        command,
        target,
        extracted,
        units,
        chunks,
        planned,
        cross_space,
        embedding_provider: EmbeddingModelAdapter,
        cross_provider: MultimodalEmbeddingAdapter | None,
    ) -> None:
        assets = {item.asset_key: item for item in extracted}
        units_by_id = {str(chunk.id): unit for chunk, unit in zip(chunks, units, strict=True)}
        chunks_by_id = {str(chunk.id): chunk for chunk in chunks}
        text_items = tuple(
            item
            for item in planned
            if item["space_role"] == "text_retrieval"
        )
        text_batch_size = embedding_provider.max_batch_size
        for offset in range(0, len(text_items), text_batch_size):
            batch = text_items[offset : offset + text_batch_size]
            usable = tuple(
                item
                for item in batch
                if units_by_id[item["unit_id"]].embedding_text
            )
            if any(
                item["required"]
                and not units_by_id[item["unit_id"]].embedding_text
                for item in batch
            ):
                raise IndexingExecutionError(
                    ErrorCode.INDEX_INCOMPLETE,
                    phase=IndexingPhase.EMBEDDING,
                    diagnostic={"check": "required_text_representation"},
                )
            if not usable:
                continue
            await self._set_phase(command, IndexingPhase.EMBEDDING)
            try:
                embedded = await embedding_provider.embed_documents(
                    tuple(
                        units_by_id[item["unit_id"]].embedding_text or ""
                        for item in usable
                    )
                )
            except IndexingExecutionError:
                if any(item["required"] for item in usable):
                    raise
                continue
            if len(embedded.vectors) != len(usable):
                raise IndexingExecutionError(
                    ErrorCode.EMBEDDING_RESPONSE_INVALID,
                    phase=IndexingPhase.EMBEDDING,
                    diagnostic={"check": "multimodal_text_batch_count"},
                )
            batch_chunks: list[IndexChunkWrite] = []
            writes: list[VectorRecordWrite] = []
            for item, vector in zip(usable, embedded.vectors, strict=True):
                validate_embedding_vector(vector, target.embedding_space)
                chunk = chunks_by_id[item["unit_id"]]
                batch_chunks.append(chunk)
                writes.append(
                    VectorRecordWrite(
                        id=stable_vector_id(
                            target.embedding_space_id,
                            chunk.id,
                            item["representation_kind"],
                        ),
                        index_chunk_id=chunk.id,
                        embedding_space_id=target.embedding_space_id,
                        embedding=vector,
                        representation_kind=item["representation_kind"],
                    )
                )
            await self._upsert(command, tuple(batch_chunks), tuple(writes))

        image_items = tuple(
            item
            for item in planned
            if item["space_role"] == "cross_modal_retrieval"
        )
        if not image_items:
            return
        provider = cross_provider
        assert provider is not None
        for offset in range(0, len(image_items), provider.max_batch_size):
            batch = image_items[offset : offset + provider.max_batch_size]
            usable = tuple(
                (item, assets[item["asset_key"]])
                for item in batch
                if item.get("asset_key") in assets
            )
            if any(
                item["required"] and item.get("asset_key") not in assets
                for item in batch
            ):
                raise IndexingExecutionError(
                    ErrorCode.INDEX_INCOMPLETE,
                    phase=IndexingPhase.MULTIMODAL_EMBEDDING,
                    diagnostic={"check": "required_image_representation"},
                )
            if not usable:
                continue
            await self._set_phase(command, IndexingPhase.MULTIMODAL_EMBEDDING)
            try:
                embedded = await provider.embed_images(
                    tuple(
                        ImageEmbeddingInput(
                            asset.content, asset.media_type, asset.content_sha256
                        )
                        for _, asset in usable
                    )
                )
            except IndexingExecutionError:
                if any(item["required"] for item, _ in usable):
                    raise
                continue
            if len(embedded.vectors) != len(usable):
                raise IndexingExecutionError(
                    ErrorCode.EMBEDDING_RESPONSE_INVALID,
                    phase=IndexingPhase.MULTIMODAL_EMBEDDING,
                    diagnostic={"check": "multimodal_image_batch_count"},
                )
            batch_chunks = []
            writes = []
            cross_space_id = target.embedding_space_ids["cross_modal_retrieval"]
            for (item, _), vector in zip(usable, embedded.vectors, strict=True):
                validate_embedding_vector(vector, cross_space)
                chunk = chunks_by_id[item["unit_id"]]
                batch_chunks.append(chunk)
                writes.append(
                    VectorRecordWrite(
                        id=stable_vector_id(
                            cross_space_id, chunk.id, item["representation_kind"]
                        ),
                        index_chunk_id=chunk.id,
                        embedding_space_id=cross_space_id,
                        embedding=vector,
                        representation_kind=item["representation_kind"],
                    )
                )
            await self._upsert(command, tuple(batch_chunks), tuple(writes))

    async def _embed_analysis_units(
        self,
        target: IndexingTarget,
        units,
        embedding_provider: EmbeddingModelAdapter,
    ) -> tuple[tuple[float, ...], ...]:
        vectors: list[tuple[float, ...]] = []
        batch_size = embedding_provider.max_batch_size
        for offset in range(0, len(units), batch_size):
            batch = units[offset : offset + batch_size]
            try:
                embedded = await embedding_provider.embed_documents(
                    tuple(unit.text for unit in batch)
                )
            except IndexingExecutionError as error:
                raise IndexingExecutionError(
                    error.code,
                    phase=IndexingPhase.SEMANTIC_ANALYSIS,
                    diagnostic=error.diagnostic,
                ) from error
            except Exception as error:
                raise IndexingExecutionError(
                    ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE,
                    phase=IndexingPhase.SEMANTIC_ANALYSIS,
                    diagnostic={"check": "analysis_provider_contract"},
                ) from error
            if len(embedded.vectors) != len(batch):
                raise IndexingExecutionError(
                    ErrorCode.EMBEDDING_RESPONSE_INVALID,
                    phase=IndexingPhase.SEMANTIC_ANALYSIS,
                    diagnostic={
                        "check": "analysis_batch_count",
                        "expected": len(batch),
                        "observed": len(embedded.vectors),
                    },
                )
            for vector in embedded.vectors:
                try:
                    validate_embedding_vector(vector, target.embedding_space)
                except IndexingExecutionError as error:
                    raise IndexingExecutionError(
                        error.code,
                        phase=IndexingPhase.SEMANTIC_ANALYSIS,
                        diagnostic=error.diagnostic,
                    ) from error
                vectors.append(vector)
        return tuple(vectors)

def _requires_semantic_analysis(units: tuple[SemanticUnit, ...]) -> bool:
    return (
        count_chunk_tokens("\n\n".join(unit.text for unit in units))
        > int(SEMANTIC_CHUNKING_CONFIG["max_chunk_tokens"])
        or any(unit.hard_boundary_before for unit in units[1:])
    )


def _lexical_rows(
    chunks: tuple[IndexChunkWrite, ...],
    allowed_chunk_ids: frozenset[UUID],
) -> tuple[IndexChunkLexicalWrite, ...]:
    rows: list[IndexChunkLexicalWrite] = []
    for chunk in chunks:
        if chunk.id not in allowed_chunk_ids:
            continue
        analyzed = analyze_document(chunk.embedding_text or chunk.content)
        if analyzed is None:
            continue
        rows.append(
            IndexChunkLexicalWrite(
                index_chunk_id=chunk.id,
                analyzer_version=LEXICAL_ANALYZER_VERSION,
                lexical_text=analyzed.lexical_text,
                lexical_text_hash=analyzed.lexical_text_hash,
            )
        )
    return tuple(rows)


def _safe_diagnostic(value: dict) -> dict:
    allowed = {
        "check",
        "expected",
        "observed",
        "expected_chunks",
        "observed_chunks",
        "observed_vectors",
        "fields",
        "http_status",
        "retryable",
        "retry_exhausted",
        "operation",
        "limit_name",
        "limit",
        "exit_kind",
        "unit_count",
        "chunk_count",
        "analysis_batch_count",
    }
    return {key: item for key, item in value.items() if key in allowed}


def _profile(target: IndexingTarget) -> str:
    """Identify chunks by the complete profile that produced them."""

    return f"{target.parser_config['profile']}:{target.chunking_config['profile']}"


def _promotion_command(command: IndexingCommand) -> PromotionCommand:
    return PromotionCommand(
        job_id=command.job_id,
        indexed_document_version_id=command.indexed_document_version_id,
    )
