"""One retry-safe indexing execution with no external I/O inside DB transactions."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from collections.abc import Awaitable, Callable, Mapping
from typing import TypeVar

from rag_kb.adapters import (
    DocumentParser,
    EmbeddingModelAdapter,
    FixedPgVectorSpace,
    SourceFileStore,
)
from rag_kb.adapters.file_store import IndexAssetStore
from rag_kb.adapters.model_api import MultimodalEmbeddingAdapter
from rag_kb.adapters.parser.ooxml_metadata import worksheet_labels
from rag_kb.adapters.parser.scanned_pages import scanned_surfaces
from docling_core.types.doc import DoclingDocument

from rag_kb.domain import (
    ChunkAssemblyDraft,
    ChunkingStrategyKind,
    ErrorCode,
    ContentModality,
    CompositeEvidenceDraft,
    FileStoreError,
    IndexChunkWrite,
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
from rag_kb.document_processing import (
    DOCLING_ENRICHMENT_CONFIG,
    DOCLING_MULTIMODAL_PARSER_CONFIG,
    DOCLING_MULTIMODAL_PARSER_CONFIG_V2,
    DOCLING_REPRESENTATION_CONFIG,
    SEMANTIC_CHUNKING_CONFIG,
    count_chunk_tokens,
    profile_fingerprint,
    parsing_preset,
    resolve,
    with_composite_embedding_text,
)
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
    text_only_document,
)
from rag_kb.document_processing.semantic_boundaries import (
    build_chunk_plan,
    validate_plan,
)
from rag_kb.indexing.promotion import CandidatePromotionService
from rag_kb.uow import UnitOfWork, UnitOfWorkFactory, UnitOfWorkPurpose, execute_in_transaction


ResultT = TypeVar("ResultT")


class IndexingPipeline:
    def __init__(
        self,
        unit_of_work: UnitOfWorkFactory,
        file_store: SourceFileStore,
        document_parser: DocumentParser,
        embedding_provider: EmbeddingModelAdapter,
        vector_space: FixedPgVectorSpace,
        *,
        asset_store: IndexAssetStore | None = None,
        multimodal_embedding_provider: MultimodalEmbeddingAdapter | None = None,
        parser_limits: ParserLimits | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._file_store = file_store
        self._document_parser = document_parser
        self._embedding_provider = embedding_provider
        self._vector_space = vector_space
        self._asset_store = asset_store
        self._multimodal_embedding_provider = multimodal_embedding_provider
        self._parser_limits = parser_limits or ParserLimits()
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
            strategy = self._require_revision_profile(target)
            self._vector_space.require_compatible(
                target.embedding_space,
                self._embedding_provider.embedding_space,
            )
            content = await self._read_source(target)

            source = ParserSource(
                original_filename=target.original_filename,
                media_type=target.media_type,
                content=content,
            )
            resolved_parsing = parsing_preset(target.parser_config)
            multimodal = resolved_parsing in {
                ParsingPreset.MULTIMODAL_LOCAL_V1,
                ParsingPreset.MULTIMODAL_LOCAL_V2,
            }
            cross_space = (
                self._require_multimodal_runtime(target) if multimodal else None
            )
            document = await self._parse(
                command,
                source,
                preset=resolved_parsing,
            )
            labels = self._surface_labels(source)
            chunks = await self._chunks(
                command, target, document, strategy, labels
            )
            if multimodal:
                result = await self._execute_multimodal(
                    command, target, source, document, chunks, cross_space, labels
                )
                promotion = await self._promotion.promote(_promotion_command(command))
                return IndexingResult(
                    command.job_id,
                    command.indexed_document_version_id,
                    "ready",
                    result,
                    serving_status=promotion.status.value,
                )
            try:
                processed = text_only_document(chunks, profile=_profile(target))
            except ParserExecutionError as error:
                raise IndexingExecutionError(
                    error.code,
                    phase=IndexingPhase.PARSING,
                    diagnostic=error.diagnostic,
                ) from error

            phase = IndexingPhase.EMBEDDING
            await self._set_phase(command, phase)
            batch_size = self._embedding_provider.max_batch_size
            for offset in range(0, len(processed.chunks), batch_size):
                drafts = processed.chunks[offset : offset + batch_size]
                try:
                    embedded = await self._embedding_provider.embed_documents(
                        tuple(draft.text for draft in drafts)
                    )
                except IndexingExecutionError:
                    raise
                except Exception as error:
                    raise IndexingExecutionError(
                        ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE,
                        phase=phase,
                        diagnostic={"check": "provider_contract"},
                    ) from error
                if len(embedded.vectors) != len(drafts):
                    raise IndexingExecutionError(
                        ErrorCode.EMBEDDING_RESPONSE_INVALID,
                        phase=phase,
                        diagnostic={
                            "check": "batch_count",
                            "expected": len(drafts),
                            "observed": len(embedded.vectors),
                        },
                    )
                for vector in embedded.vectors:
                    validate_embedding_vector(vector, target.embedding_space)
                writes, vectors = self._writes(target, drafts, embedded.vectors)
                phase = IndexingPhase.PERSISTING
                await self._upsert(command, writes, vectors)
                phase = IndexingPhase.EMBEDDING

            phase = IndexingPhase.VALIDATING
            await self._set_phase(command, phase)
            await self._complete(command, expected_chunks=len(processed.chunks))
            promotion = await self._promotion.promote(_promotion_command(command))
            return IndexingResult(
                command.job_id,
                command.indexed_document_version_id,
                "ready",
                len(processed.chunks),
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
            if target.parser_config in (
                DOCLING_MULTIMODAL_PARSER_CONFIG,
                DOCLING_MULTIMODAL_PARSER_CONFIG_V2,
            ) and (
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
    ) -> DoclingDocument:
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

    def _surface_labels(self, source: ParserSource) -> dict[int, str]:
        """Recover surface names Docling does not expose, such as sheet names."""

        try:
            return worksheet_labels(source)
        except ParserExecutionError as error:
            raise IndexingExecutionError(
                error.code,
                phase=IndexingPhase.PARSING,
                diagnostic=error.diagnostic,
            ) from error

    async def _chunks(
        self,
        command: IndexingCommand,
        target: IndexingTarget,
        document: DoclingDocument,
        strategy: ChunkingStrategyKind,
        surface_labels: Mapping[int, str],
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
            command, target, document, surface_labels
        )

    async def _semantic_chunks(
        self,
        command: IndexingCommand,
        target: IndexingTarget,
        document: DoclingDocument,
        surface_labels: Mapping[int, str],
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
        existing = await self._transaction(
            lambda uow: uow.indexing.get_chunk_plan(command)
        )
        if existing is not None:
            await asyncio.to_thread(
                validate_plan,
                existing, **plan_facts, units=units, sequence_hash=sequence_hash
            )
            return await asyncio.to_thread(
                self._assemble_semantic,
                document,
                units,
                existing,
                surface_labels,
            )

        requires_analysis = await asyncio.to_thread(
            _requires_semantic_analysis,
            units,
        )
        vectors = (
            await self._embed_analysis_units(target, units)
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
        winner = await self._transaction(
            lambda uow: uow.indexing.create_or_get_chunk_plan(command, proposed)
        )
        await asyncio.to_thread(
            validate_plan,
            winner,
            **plan_facts,
            units=units,
            sequence_hash=sequence_hash,
        )
        return await asyncio.to_thread(
            self._assemble_semantic,
            document,
            units,
            winner,
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

    def _require_multimodal_runtime(self, target: IndexingTarget):
        """Reject an unusable multimodal revision before converting its source."""

        if self._asset_store is None or self._multimodal_embedding_provider is None:
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
        FixedPgVectorSpace(cross_space).require_compatible(
            cross_space, self._multimodal_embedding_provider.embedding_space
        )
        return cross_space

    async def _execute_multimodal(
        self,
        command: IndexingCommand,
        target: IndexingTarget,
        source: ParserSource,
        document: DoclingDocument,
        assembled: tuple[ChunkAssemblyDraft, ...],
        cross_space,
        surface_labels: Mapping[int, str],
    ) -> int:
        assert self._asset_store is not None
        cross_space_id = target.embedding_space_ids["cross_modal_retrieval"]
        await self._set_phase(command, IndexingPhase.ASSET_EXTRACTION)
        assembly, extracted = self._composite_evidence(
            target, source, document, assembled, surface_labels
        )
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
        winner = await self._transaction(
            lambda uow: uow.indexing.create_or_get_artifact_manifest(
                command, proposed
            )
        )
        if winner != proposed:
            raise IndexingExecutionError(
                ErrorCode.INDEX_CHUNK_PLAN_MISMATCH,
                phase=IndexingPhase.PERSISTING,
                diagnostic={"check": "artifact_manifest_reuse"},
            )

        await self._embed_multimodal_representations(
            command, target, assets, units, chunks, planned, cross_space
        )
        changed = await self._transaction(
            lambda uow: uow.indexing.upsert_relations(command, relation_writes)
        )
        if not changed:
            raise IndexingCancelled
        await self._set_phase(command, IndexingPhase.VALIDATING)
        await self._complete(command, expected_chunks=len(chunks))
        return len(chunks)

    def _composite_evidence(
        self,
        target: IndexingTarget,
        source: ParserSource,
        document: DoclingDocument,
        assembled: tuple[ChunkAssemblyDraft, ...],
        surface_labels: Mapping[int, str],
    ) -> tuple[CompositeEvidenceDraft, tuple[ParsedAssetDraft, ...]]:
        """Derive assets, relations and evidence units from the one conversion."""

        try:
            assets = extract_docling_assets(
                document,
                self._parser_limits,
                page_image_surfaces=scanned_surfaces(source),
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
            unit.ordinal,
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

    async def _embed_multimodal_representations(
        self, command, target, extracted, units, chunks, planned, cross_space
    ) -> None:
        assets = {item.asset_key: item for item in extracted}
        units_by_id = {str(chunk.id): unit for chunk, unit in zip(chunks, units, strict=True)}
        chunks_by_id = {str(chunk.id): chunk for chunk in chunks}
        text_items = tuple(
            item for item in planned if item["space_role"] == "text_retrieval"
        )
        text_batch_size = self._embedding_provider.max_batch_size
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
                embedded = await self._embedding_provider.embed_documents(
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

        provider = self._multimodal_embedding_provider
        assert provider is not None
        image_items = tuple(
            item
            for item in planned
            if item["space_role"] == "cross_modal_retrieval"
        )
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
    ) -> tuple[tuple[float, ...], ...]:
        vectors: list[tuple[float, ...]] = []
        batch_size = self._embedding_provider.max_batch_size
        for offset in range(0, len(units), batch_size):
            batch = units[offset : offset + batch_size]
            try:
                embedded = await self._embedding_provider.embed_documents(
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

    @staticmethod
    def _writes(target, drafts, embeddings):
        chunks: list[IndexChunkWrite] = []
        vectors: list[VectorRecordWrite] = []
        for draft, embedding in zip(drafts, embeddings, strict=True):
            chunk_id = stable_chunk_id(target.indexed_document_version_id, draft.ordinal)
            chunks.append(
                IndexChunkWrite(
                    id=chunk_id,
                    ordinal=draft.ordinal,
                    content=draft.text,
                    content_hash=draft.content_sha256,
                    token_count=draft.token_count,
                    source_location=dict(draft.source_location),
                    hierarchy=dict(draft.hierarchy),
                    source_metadata={
                        "document_id": str(target.document_id),
                        "document_version_id": str(target.document_version_id),
                        "original_filename": target.original_filename,
                        "media_type": target.media_type,
                        "checksum_sha256": target.checksum_sha256,
                        "processing": dict(draft.processing_metadata),
                    },
                )
            )
            vectors.append(
                VectorRecordWrite(
                    id=stable_vector_id(target.embedding_space_id, chunk_id),
                    index_chunk_id=chunk_id,
                    embedding_space_id=target.embedding_space_id,
                    embedding=embedding,
                )
            )
        return tuple(chunks), tuple(vectors)


def _requires_semantic_analysis(units: tuple[SemanticUnit, ...]) -> bool:
    return (
        count_chunk_tokens("\n\n".join(unit.text for unit in units))
        > int(SEMANTIC_CHUNKING_CONFIG["max_chunk_tokens"])
        or any(unit.hard_boundary_before for unit in units[1:])
    )


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
