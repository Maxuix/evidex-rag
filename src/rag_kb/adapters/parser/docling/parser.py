"""Thin, parse-once gateway from bounded bytes to ``DoclingDocument``."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import multiprocessing
from multiprocessing.connection import Connection
import os
from pathlib import Path, PurePath
import re
import signal
import shutil
from tempfile import TemporaryDirectory
import threading
import time
from typing import Any
from uuid import UUID
import warnings
from zipfile import BadZipFile, ZipFile

from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    DocumentStream,
    FailureCategory,
    InputFormat,
)
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DocItemLabel, DoclingDocument
from PIL import Image as PillowImage

from rag_kb.adapters.parser.docling.artifacts import (
    ArtifactManifestError,
    verify_docling_artifacts,
)
from rag_kb.adapters.parser.docling.factory import build_docling_converter
from rag_kb.adapters.parser.docling.progress_pipeline import emit_pdf_progress
from rag_kb.adapters.parser.ooxml_metadata import worksheet_labels
from rag_kb.adapters.parser.scanned_pages import probe_pdf
from rag_kb.domain import (
    ErrorCode,
    FileAdmissionError,
    ParserExecutionError,
    ParserLimits,
    ParserProfile,
    ParserProgress,
    ParserSource,
    ParsingPreset,
)
from rag_kb.document_processing.markdown_bundle import (
    MARKDOWN_BUNDLE_MEDIA_TYPE,
    read_normalized_markdown_bundle,
)
from rag_kb.document_processing.resource_preflight import (
    ResourcePreflightContentError,
    ResourcePreflightLimitError,
    validate_csv_structure,
    validate_ooxml_images,
)
from rag_kb.document_processing.xlsx_preflight import validate_xlsx_structure
from rag_kb.document_processing.docling.resources import image_usage, require_limit
from rag_kb.ports.parsing import (
    DocumentParseContinuation,
    DocumentParseResult,
    ParserProgressHandler,
)


_OOXML_PREFIX = "application/vnd.openxmlformats-officedocument"
_MEDIA_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
    ".csv": "text/csv",
    ".pdf": "application/pdf",
    ".docx": f"{_OOXML_PREFIX}.wordprocessingml.document",
    ".pptx": f"{_OOXML_PREFIX}.presentationml.presentation",
    ".xlsx": f"{_OOXML_PREFIX}.spreadsheetml.sheet",
}
DOCLING_DOCUMENT_VERSION = "1.10.0"
ConverterFactory = Callable[..., DocumentConverter]
ChildTarget = Callable[[Connection, ParserLimits, Path, Path], None]
_PROCESS_STOP_GRACE_SECONDS = 1.0
_MAX_IPC_RESPONSE_BYTES = 256 * 1024 * 1024
_SUCCESS_RESPONSE = b"O"
_PDF_PROBE_RESPONSE = b"M"
_PDF_PROBE_SCHEMA = "pdf_image_coverage_v2"
_ERROR_RESPONSE = b"E"
_PROGRESS_RESPONSE = b"P"
_CHECKPOINT_SCHEMA = "docling_page_range_json_v1"
_MAX_CHECKPOINT_MANIFEST_BYTES = 256 * 1024
_USAGE_KEYS = {
    "max_num_pages", "max_docling_items", "max_extracted_characters", "max_assets",
    "max_total_asset_bytes", "max_total_image_pixels", "max_checkpoint_bytes",
}
_PDF_PROGRESS_STAGE_NAMES = frozenset(
    {
        "page_parse",
        "ocr",
        "layout",
        "table_structure",
        "page_assembly",
        "document_assembly",
    }
)


@dataclass(frozen=True, slots=True)
class _RoundTripResult:
    response: bytes
    last_progress: dict[str, Any]


class DoclingParser:
    """Run serial Docling conversions in a reusable, killable child process."""

    def __init__(
        self,
        limits: ParserLimits,
        *,
        artifacts_path: Path,
        artifact_manifest_path: Path,
        checkpoint_root: Path | None = None,
        child_target: ChildTarget | None = None,
    ) -> None:
        self._limits = limits
        self._artifacts_path = artifacts_path
        self._artifact_manifest_path = artifact_manifest_path
        self._checkpoint_root = checkpoint_root
        self._child_target = child_target or _parser_child
        self._context = multiprocessing.get_context("spawn")
        self._process: multiprocessing.Process | None = None
        self._connection: Connection | None = None
        self._state_lock = threading.Lock()
        self._request_lock = asyncio.Lock()
        self._ipc_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rag-kb-docling-ipc",
        )
        self._closed = False

    async def parse(
        self,
        source: ParserSource,
        *,
        profile: ParserProfile | None = None,
        checkpoint_key: str | None = None,
        on_progress: ParserProgressHandler | None = None,
        preset: ParsingPreset | None = None,
    ) -> DocumentParseResult | DocumentParseContinuation:
        """Convert once and return the native model with bounded source metadata."""

        try:
            resolved_profile = _resolve_profile(profile=profile, preset=preset)
        except ValueError as error:
            raise ParserExecutionError(ErrorCode.PARSER_NOT_CONFIGURED) from error
        resolved_preset = resolved_profile.preset
        _validate_source(source, self._limits, resolved_preset)
        if (
            resolved_profile.uses_balanced_pdf_runtime
            and PurePath(source.original_filename).suffix.lower() == ".pdf"
        ):
            if checkpoint_key is None:
                raise ParserExecutionError(
                    ErrorCode.PARSER_NOT_CONFIGURED,
                    diagnostic={"check": "pdf_checkpoint_key"},
                )
            return await self._parse_segmented_pdf(
                source,
                profile=resolved_profile,
                checkpoint_key=checkpoint_key,
                on_progress=on_progress,
            )

        surfaces = frozenset()
        if source.media_type == "application/pdf":
            _, surfaces = await self._probe_pdf(source, resolved_profile)
        round_trip = await self._convert_once(
            source,
            profile=resolved_profile,
        )
        try:
            document = await asyncio.to_thread(
                _decode_response, round_trip.response, self._limits
            )
        except ParserExecutionError as error:
            if error.code is ErrorCode.PARSER_CRASHED:
                await self._reset_child()
            raise
        return await self._result(source, document, surfaces)

    async def _convert_once(
        self,
        source: ParserSource,
        *,
        profile: ParserProfile,
        page_range: tuple[int, int] | None = None,
        on_raw_progress: Callable[[dict[str, Any]], None] | None = None,
        operation: str = "parse",
    ) -> _RoundTripResult:
        async with self._request_lock:
            connection = self._ensure_child()
            loop = asyncio.get_running_loop()
            try:
                concurrent_future = self._ipc_executor.submit(
                    _round_trip,
                    connection,
                    source,
                    profile,
                    page_range,
                    on_raw_progress,
                    operation,
                )
            except RuntimeError as error:
                raise ParserExecutionError(
                    ErrorCode.PARSER_NOT_CONFIGURED,
                    diagnostic={"check": "docling_parser_closed"},
                ) from error
            wrapped = asyncio.wrap_future(concurrent_future, loop=loop)
            try:
                return await asyncio.shield(wrapped)
            except asyncio.CancelledError:
                wrapped.add_done_callback(_consume_cancelled_result)
                await self._reset_child()
                raise
            except (EOFError, BrokenPipeError, OSError) as error:
                exit_code = await self._reset_child()
                if exit_code == -signal.SIGKILL:
                    raise ParserExecutionError(
                        ErrorCode.PARSER_RESOURCE_LIMIT,
                        diagnostic={"limit_name": "process_memory"},
                    ) from error
                raise ParserExecutionError(
                    ErrorCode.PARSER_CRASHED,
                    diagnostic={"check": "docling_child_ipc"},
                ) from error


    async def _result(
        self,
        source: ParserSource,
        document: DoclingDocument,
        page_image_surfaces: frozenset[int] = frozenset(),
    ) -> DocumentParseResult:
        labels = await asyncio.to_thread(worksheet_labels, source)
        return DocumentParseResult(
            document=document,
            surface_labels=tuple(labels.items()),
            page_image_surfaces=page_image_surfaces,
        )

    async def _parse_segmented_pdf(
        self,
        source: ParserSource,
        *,
        profile: ParserProfile,
        checkpoint_key: str,
        on_progress: ParserProgressHandler | None,
    ) -> DocumentParseResult | DocumentParseContinuation:
        cached_probe = await asyncio.to_thread(
            self._cached_pdf_probe, checkpoint_key, source, profile
        )
        total_pages, surfaces = (
            cached_probe if cached_probe is not None
            else await self._probe_pdf(source, profile)
        )
        checkpoint = await asyncio.to_thread(
            self._load_or_create_checkpoint,
            checkpoint_key,
            source,
            profile,
            total_pages,
            surfaces,
        )

        pending_index = next(
            (
                index
                for index, segment in enumerate(checkpoint["segments"])
                if segment["status"] != "completed"
            ),
            None,
        )
        if pending_index is not None:
            segment = checkpoint["segments"][pending_index]
            started_at = time.monotonic()
            progress_state: dict[str, Any] = {
                "stage_pages": {},
                "ocr_pages": 0,
                "ocr_regions": 0,
                "table_candidates": 0,
                "child_peak_rss_bytes": None,
                "last_emit_at": 0.0,
            }
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[ParserProgress | None] = asyncio.Queue()
            consumer = asyncio.create_task(
                _consume_progress(queue, on_progress)
            )

            def forward(raw: dict[str, Any]) -> None:
                stage = str(raw.get("stage") or "page_parse")
                stage_completed = _nonnegative_int(
                    raw.get("stage_completed_pages")
                )
                progress_state["stage_pages"][stage] = stage_completed
                for name in ("ocr_pages", "ocr_regions", "table_candidates"):
                    progress_state[name] = max(
                        progress_state[name], _nonnegative_int(raw.get(name))
                    )
                peak = raw.get("child_peak_rss_bytes")
                if isinstance(peak, int) and peak >= 0:
                    progress_state["child_peak_rss_bytes"] = max(
                        progress_state["child_peak_rss_bytes"] or 0,
                        peak,
                    )
                progress = _progress(
                    checkpoint,
                    pending_index,
                    stage=stage,
                    current_stage_pages=progress_state["stage_pages"],
                    current_ocr_pages=progress_state["ocr_pages"],
                    current_ocr_regions=progress_state["ocr_regions"],
                    current_table_candidates=progress_state[
                        "table_candidates"
                    ],
                    current_elapsed_ms=int(
                        (time.monotonic() - started_at) * 1000
                    ),
                    child_peak_rss_bytes=progress_state[
                        "child_peak_rss_bytes"
                    ],
                )
                now = time.monotonic()
                if now - progress_state["last_emit_at"] >= 1.0:
                    progress_state["last_emit_at"] = now
                    loop.call_soon_threadsafe(queue.put_nowait, progress)

            try:
                round_trip = await self._convert_once(
                    source,
                    profile=profile,
                    page_range=(segment["page_from"], segment["page_to"]),
                    on_raw_progress=forward,
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)
                await consumer

            try:
                document = await asyncio.to_thread(
                    _decode_response,
                    round_trip.response,
                    self._limits,
                    allow_empty=True,
                )
            except ParserExecutionError as error:
                if error.code is ErrorCode.PARSER_CRASHED:
                    await self._reset_child()
                raise
            elapsed_ms = int((time.monotonic() - started_at) * 1000)
            checkpoint = await asyncio.to_thread(
                self._complete_segment,
                checkpoint_key,
                checkpoint,
                pending_index,
                document,
                elapsed_ms,
                progress_state,
            )
            pending_index = next(
                (
                    index
                    for index, item in enumerate(checkpoint["segments"])
                    if item["status"] != "completed"
                ),
                None,
            )
            if pending_index is not None:
                progress = _progress(
                    checkpoint,
                    pending_index,
                    stage="segment_checkpointed",
                )
                await _publish_progress(on_progress, progress)
                return DocumentParseContinuation(progress)

        assembling = _progress(
            checkpoint,
            max(len(checkpoint["segments"]) - 1, 0),
            stage="document_assembly",
        )
        await _publish_progress(on_progress, assembling)
        document = await asyncio.to_thread(
            self._assemble_checkpoint,
            checkpoint_key,
            checkpoint,
        )
        completed = _progress(
            checkpoint,
            max(len(checkpoint["segments"]) - 1, 0),
            stage="completed",
        )
        await _publish_progress(on_progress, completed)
        return await self._result(source, document, surfaces)

    async def _probe_pdf(
        self, source: ParserSource, profile: ParserProfile,
    ) -> tuple[int, frozenset[int]]:
        result = await self._convert_once(source, profile=profile, operation="probe")
        try:
            if result.response[:1] == _ERROR_RESPONSE:
                _decode_response(result.response, self._limits)
            return _decode_pdf_probe(result.response, self._limits)
        except ParserExecutionError as error:
            if error.code is ErrorCode.PARSER_CRASHED:
                await self._reset_child()
            raise

    def _cached_pdf_probe(
        self, checkpoint_key: str, source: ParserSource, profile: ParserProfile,
    ) -> tuple[int, frozenset[int]] | None:
        path = self._checkpoint_directory(checkpoint_key) / "manifest.json"
        if not path.exists():
            return None
        checkpoint = _read_checkpoint_manifest(path)
        if (
            checkpoint.get("source_sha256") != hashlib.sha256(source.content).hexdigest()
            or checkpoint.get("profile") != profile.value
        ):
            raise ParserExecutionError(
                ErrorCode.PARSER_CRASHED,
                diagnostic={"check": "pdf_checkpoint_identity"},
            )
        probe = checkpoint.get("probe")
        if not isinstance(probe, dict) or probe.get("schema") != _PDF_PROBE_SCHEMA:
            return None
        return _decode_pdf_probe(
            _PDF_PROBE_RESPONSE + json.dumps(probe).encode(), self._limits
        )

    def discard_checkpoint(self, checkpoint_key: str) -> None:
        directory = self._checkpoint_directory(checkpoint_key)
        if not directory.exists():
            return
        if directory.is_symlink() or not directory.is_dir():
            raise ParserExecutionError(
                ErrorCode.PARSER_CRASHED,
                diagnostic={"check": "pdf_checkpoint_path"},
            )
        shutil.rmtree(directory)

    def _load_or_create_checkpoint(
        self,
        checkpoint_key: str,
        source: ParserSource,
        profile: ParserProfile,
        total_pages: int,
        surfaces: frozenset[int] = frozenset(),
    ) -> dict[str, Any]:
        directory = self._checkpoint_directory(checkpoint_key)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        root = self._checkpoint_root.resolve() if self._checkpoint_root else None
        if (
            root is None
            or directory.is_symlink()
            or not directory.is_dir()
            or not directory.resolve().is_relative_to(root)
        ):
            raise ParserExecutionError(
                ErrorCode.PARSER_CRASHED,
                diagnostic={"check": "pdf_checkpoint_path"},
            )
        manifest_path = directory / "manifest.json"
        source_sha256 = hashlib.sha256(source.content).hexdigest()
        if manifest_path.exists():
            checkpoint = _read_checkpoint_manifest(manifest_path)
            if (
                checkpoint.get("source_sha256") != source_sha256
                or checkpoint.get("profile") != profile.value
                or checkpoint.get("total_pages") != total_pages
            ):
                raise ParserExecutionError(
                    ErrorCode.PARSER_CRASHED,
                    diagnostic={"check": "pdf_checkpoint_identity"},
                )
            _validate_checkpoint(checkpoint, total_pages)
            checkpoint["probe"] = {"schema": _PDF_PROBE_SCHEMA, "total_pages": total_pages, "surfaces": sorted(surfaces)}
            self._checkpoint_usage(checkpoint_key, checkpoint)
            _write_private_json(manifest_path, checkpoint)
            return checkpoint
        segments = [
            {
                "page_from": page_from,
                "page_to": min(
                    page_from + self._limits.pdf_segment_pages - 1,
                    total_pages,
                ),
                "status": "pending",
            }
            for page_from in range(
                1,
                total_pages + 1,
                self._limits.pdf_segment_pages,
            )
        ]
        checkpoint = {
            "schema_version": _CHECKPOINT_SCHEMA,
            "source_sha256": source_sha256,
            "profile": profile.value,
            "total_pages": total_pages,
            "probe": {"schema": _PDF_PROBE_SCHEMA, "total_pages": total_pages, "surfaces": sorted(surfaces)},
            "elapsed_ms": 0,
            "segments": segments,
        }
        _write_private_json(manifest_path, checkpoint)
        return checkpoint

    def _complete_segment(
        self,
        checkpoint_key: str,
        checkpoint: dict[str, Any],
        segment_index: int,
        document: DoclingDocument,
        elapsed_ms: int,
        progress_state: dict[str, Any],
    ) -> dict[str, Any]:
        directory = self._checkpoint_directory(checkpoint_key)
        segment = checkpoint["segments"][segment_index]
        filename = f"pages-{segment['page_from']}-{segment['page_to']}.json"
        usage = _validate_document(document, self._limits, allow_empty=True)
        accumulated = self._checkpoint_usage(checkpoint_key, checkpoint)
        self._add_usage(accumulated, usage)
        payload = document.model_dump_json().encode("utf-8")
        usage["max_checkpoint_bytes"] = len(payload)
        require_limit(
            "max_checkpoint_bytes",
            accumulated.get("max_checkpoint_bytes", 0) + len(payload),
            self._limits,
        )
        _write_private_bytes(directory / filename, payload)
        completed = {
            **segment,
            "status": "completed",
            "filename": filename,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "usage": usage,
            "elapsed_ms": elapsed_ms,
            "stage_pages": {
                name: int(value)
                for name, value in sorted(progress_state["stage_pages"].items())
                if name in _PDF_PROGRESS_STAGE_NAMES
            },
            "ocr_pages": int(progress_state["ocr_pages"]),
            "ocr_regions": int(progress_state["ocr_regions"]),
            "table_candidates": int(progress_state["table_candidates"]),
            "child_peak_rss_bytes": progress_state["child_peak_rss_bytes"],
        }
        checkpoint = {
            **checkpoint,
            "elapsed_ms": int(checkpoint["elapsed_ms"]) + elapsed_ms,
            "segments": [
                completed if index == segment_index else item
                for index, item in enumerate(checkpoint["segments"])
            ],
        }
        _write_private_json(directory / "manifest.json", checkpoint)
        return checkpoint

    def _add_usage(
        self, accumulated: dict[str, int], usage: dict[str, int],
    ) -> dict[str, int]:
        for name, value in usage.items():
            if name not in _USAGE_KEYS or type(value) is not int or value < 0:
                raise ParserExecutionError(
                    ErrorCode.PARSER_CRASHED,
                    diagnostic={"check": "pdf_checkpoint_usage"},
                )
            accumulated[name] = accumulated.get(name, 0) + value
            require_limit(name, accumulated[name], self._limits)
        return accumulated

    def _read_segment(
        self, checkpoint_key: str, segment: dict[str, Any],
    ) -> tuple[DoclingDocument, dict[str, int]]:
        path = self._checkpoint_directory(checkpoint_key) / segment["filename"]
        payload = _read_private_bytes(path, self._limits.max_checkpoint_bytes)
        if hashlib.sha256(payload).hexdigest() != segment["sha256"]:
            raise ParserExecutionError(
                ErrorCode.PARSER_CRASHED,
                diagnostic={"check": "pdf_checkpoint_digest"},
            )
        try:
            document = DoclingDocument.model_validate_json(payload)
        except Exception as error:
            raise ParserExecutionError(
                ErrorCode.PARSER_CRASHED,
                diagnostic={"check": "pdf_checkpoint_document"},
            ) from error
        usage = _validate_document(document, self._limits, allow_empty=True)
        usage["max_checkpoint_bytes"] = len(payload)
        expected_pages = set(range(segment["page_from"], segment["page_to"] + 1))
        if set(document.pages) != expected_pages:
            raise ParserExecutionError(
                ErrorCode.PARSER_OUTPUT_INVALID,
                diagnostic={"check": "pdf_segment_pages"},
            )
        return document, usage

    def _checkpoint_usage(
        self, checkpoint_key: str, checkpoint: dict[str, Any],
    ) -> dict[str, int]:
        accumulated = {}
        for segment in checkpoint["segments"]:
            if segment["status"] != "completed":
                continue
            usage = segment.get("usage")
            if usage is None:
                # Old checkpoints are upgraded one segment at a time, before
                # any full-document allocation. Never trust absent counters.
                _, usage = self._read_segment(checkpoint_key, segment)
                segment["usage"] = usage
            if not isinstance(usage, dict) or set(usage) != _USAGE_KEYS:
                raise ParserExecutionError(
                    ErrorCode.PARSER_CRASHED,
                    diagnostic={"check": "pdf_checkpoint_usage"},
                )
            self._add_usage(accumulated, usage)
        return accumulated

    def _assemble_checkpoint(
        self, checkpoint_key: str, checkpoint: dict[str, Any],
    ) -> DoclingDocument:
        self._checkpoint_usage(checkpoint_key, checkpoint)
        origin = None

        def documents():
            nonlocal origin
            accumulated = {}
            for segment in checkpoint["segments"]:
                if segment["status"] != "completed":
                    raise ParserExecutionError(
                        ErrorCode.PARSER_CRASHED,
                        diagnostic={"check": "pdf_checkpoint_incomplete"},
                    )
                document, usage = self._read_segment(checkpoint_key, segment)
                self._add_usage(accumulated, usage)
                if origin is None:
                    origin = document.origin
                yield document

        # The pinned concatenate implementation consumes its argument once.
        # Yielding one document at a time avoids retaining every input alongside
        # the output copies. Recheck real usage, not just manifest counters.
        document = DoclingDocument.concatenate(documents())
        document.origin = origin
        _validate_document(document, self._limits)
        return document

    def _checkpoint_directory(self, checkpoint_key: str) -> Path:
        if self._checkpoint_root is None:
            raise ParserExecutionError(
                ErrorCode.PARSER_NOT_CONFIGURED,
                diagnostic={"check": "pdf_checkpoint_root"},
            )
        try:
            normalized = str(UUID(checkpoint_key))
        except (TypeError, ValueError, AttributeError) as error:
            raise ParserExecutionError(
                ErrorCode.PARSER_CRASHED,
                diagnostic={"check": "pdf_checkpoint_key"},
            ) from error
        if normalized != checkpoint_key:
            raise ParserExecutionError(
                ErrorCode.PARSER_CRASHED,
                diagnostic={"check": "pdf_checkpoint_key"},
            )
        root = self._checkpoint_root.resolve()
        directory = root / "pdf-checkpoints" / normalized
        if not directory.is_relative_to(root):
            raise ParserExecutionError(
                ErrorCode.PARSER_CRASHED,
                diagnostic={"check": "pdf_checkpoint_path"},
            )
        return directory

    def close(self) -> None:
        """Stop accepting conversions and terminate the owned child process."""

        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            resources = self._detach_child()
        _stop_child(*resources)
        self._ipc_executor.shutdown(wait=True, cancel_futures=True)

    def _ensure_child(self) -> Connection:
        with self._state_lock:
            if self._closed:
                raise ParserExecutionError(
                    ErrorCode.PARSER_NOT_CONFIGURED,
                    diagnostic={"check": "docling_parser_closed"},
                )
            if (
                self._process is not None
                and self._connection is not None
                and self._process.is_alive()
            ):
                return self._connection
            stale = self._detach_child()
            _stop_child(*stale)
            parent, child = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=self._child_target,
                args=(
                    child,
                    self._limits,
                    self._artifacts_path,
                    self._artifact_manifest_path,
                ),
                name="rag-kb-docling",
                daemon=True,
            )
            try:
                process.start()
            except Exception as error:
                parent.close()
                child.close()
                _stop_child(process, None)
                raise ParserExecutionError(
                    ErrorCode.PARSER_NOT_CONFIGURED,
                    diagnostic={"check": "docling_child_start"},
                ) from error
            child.close()
            self._process = process
            self._connection = parent
            return parent

    async def _reset_child(self) -> int | None:
        with self._state_lock:
            resources = self._detach_child()
        return await asyncio.shield(asyncio.to_thread(_stop_child, *resources))

    def _detach_child(
        self,
    ) -> tuple[multiprocessing.Process | None, Connection | None]:
        process = self._process
        connection = self._connection
        self._process = None
        self._connection = None
        return process, connection


class _DoclingRuntime:
    """Child-process runtime that owns converters and performs one conversion."""

    def __init__(
        self,
        limits: ParserLimits,
        *,
        artifacts_path: Path,
        artifact_manifest_path: Path,
        converter_factory: ConverterFactory = build_docling_converter,
    ) -> None:
        self._limits = limits
        self._artifacts_path = artifacts_path
        self._artifact_manifest_path = artifact_manifest_path
        self._converter_factory = converter_factory
        self._converters: dict[ParserProfile, DocumentConverter] = {}
        self._artifacts_verified = False

    def convert(
        self,
        source: ParserSource,
        profile: ParserProfile | ParsingPreset,
        *,
        page_range: tuple[int, int] | None = None,
        progress_emitter: Callable[[dict[str, Any]], None] | None = None,
    ) -> DoclingDocument:
        resolved_profile = _resolve_profile(profile=profile, preset=None)
        try:
            _preflight_conversion_source(source, self._limits)
            if source.media_type == "text/plain":
                document = DoclingDocument(
                    name=PurePath(source.original_filename).stem or "document"
                )
                try:
                    text = source.content.decode("utf-8-sig")
                except UnicodeDecodeError as error:
                    raise ParserExecutionError(
                        ErrorCode.FILE_CONTENT_INVALID,
                        diagnostic={"check": "plain_text_utf8"},
                    ) from error
                document.add_text(label=DocItemLabel.TEXT, text=text)
                _validate_document(document, self._limits)
                return document

            converter = self._get_converter(
                resolved_profile,
                require_artifacts=(
                    source.media_type == "application/pdf"
                    or PurePath(source.original_filename).suffix.lower() == ".pdf"
                ),
            )
            if source.media_type == MARKDOWN_BUNDLE_MEDIA_TYPE:
                result = self._convert_markdown_bundle(converter, source.content)
            else:
                stream = DocumentStream(
                    name=source.original_filename,
                    stream=BytesIO(source.content),
                )
                kwargs: dict[str, Any] = {
                    "raises_on_error": False,
                    "max_num_pages": self._limits.max_num_pages,
                    "max_file_size": self._limits.max_file_size,
                }
                if page_range is not None:
                    kwargs["page_range"] = page_range
                with emit_pdf_progress(progress_emitter):
                    result = converter.convert(stream, **kwargs)
        except ParserExecutionError:
            raise
        except (FileNotFoundError, ImportError, ModuleNotFoundError) as error:
            raise ParserExecutionError(
                ErrorCode.PARSER_NOT_CONFIGURED,
                diagnostic={"check": "docling_runtime"},
            ) from error
        except MemoryError as error:
            raise ParserExecutionError(
                ErrorCode.PARSER_RESOURCE_LIMIT,
                diagnostic={"limit_name": "process_memory"},
            ) from error
        except (
            PillowImage.DecompressionBombWarning,
            PillowImage.DecompressionBombError,
        ) as error:
            raise ParserExecutionError(
                ErrorCode.PARSER_RESOURCE_LIMIT,
                diagnostic={
                    "limit_name": "max_image_pixels",
                    "limit": self._limits.max_image_pixels,
                },
            ) from error
        except Exception as error:
            raise ParserExecutionError(
                ErrorCode.PARSER_CRASHED,
                diagnostic={"check": "docling_conversion"},
            ) from error
        return _validate_conversion_result(
            result,
            self._limits,
            allow_empty=page_range is not None,
        )

    def _convert_markdown_bundle(
        self,
        converter: DocumentConverter,
        content: bytes,
    ) -> Any:
        try:
            entrypoint, files = read_normalized_markdown_bundle(content)
        except FileAdmissionError as error:
            raise ParserExecutionError(error.code) from error
        with TemporaryDirectory(prefix="rag-kb-md-") as directory:
            root = Path(directory)
            for name, member in files.items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(member)
            markdown_path = root / entrypoint
            options = converter.format_to_options[InputFormat.MD].backend_options
            if options is None:
                raise ParserExecutionError(
                    ErrorCode.PARSER_NOT_CONFIGURED,
                    diagnostic={"check": "markdown_backend_options"},
                )
            options.source_uri = markdown_path
            return converter.convert(
                markdown_path,
                raises_on_error=False,
                max_num_pages=self._limits.max_num_pages,
                max_file_size=self._limits.max_file_size,
            )

    def _get_converter(
        self,
        profile: ParserProfile,
        *,
        require_artifacts: bool,
    ) -> DocumentConverter:
        # Markdown/CSV/office simple pipelines do not load the PDF
        # OCR/layout/table models.  Require the frozen model bundle only when
        # a PDF conversion can actually consume it.  Keep this check ahead of
        # the converter cache so a prior text conversion cannot bypass the PDF
        # artifact gate.
        if require_artifacts and not self._artifacts_verified:
            try:
                manifest = verify_docling_artifacts(
                    self._artifacts_path,
                    self._artifact_manifest_path,
                )
                if manifest.docling_document_version != DOCLING_DOCUMENT_VERSION:
                    raise ArtifactManifestError("document_version")
            except ArtifactManifestError as error:
                raise ParserExecutionError(
                    ErrorCode.PARSER_NOT_CONFIGURED,
                    diagnostic={"check": "docling_artifact_manifest"},
                ) from error
            self._artifacts_verified = True
        converter = self._converters.get(profile)
        if converter is not None:
            return converter
        try:
            factory_profile: ParserProfile | ParsingPreset = (
                profile
                if profile.uses_balanced_pdf_runtime
                else profile.preset
            )
            converter = self._converter_factory(
                factory_profile,
                artifacts_path=self._artifacts_path,
                limits=self._limits,
            )
        except (FileNotFoundError, ImportError, ModuleNotFoundError) as error:
            raise ParserExecutionError(
                ErrorCode.PARSER_NOT_CONFIGURED,
                diagnostic={"check": "docling_converter_factory"},
            ) from error
        self._converters[profile] = converter
        return converter


def _parser_child(
    connection: Connection,
    limits: ParserLimits,
    artifacts_path: Path,
    artifact_manifest_path: Path,
) -> None:
    _configure_pillow_resource_guard(limits)
    runtime = _DoclingRuntime(
        limits,
        artifacts_path=artifacts_path,
        artifact_manifest_path=artifact_manifest_path,
    )
    send_lock = threading.Lock()

    def send(payload: bytes) -> None:
        with send_lock:
            connection.send_bytes(payload)

    try:
        while True:
            try:
                request = connection.recv()
            except EOFError:
                return
            try:
                if len(request) == 3:
                    kind, source, profile_value = request
                    page_range = None
                else:
                    kind, source, profile_value, page_range = request
                if kind not in {"parse", "probe"} or not isinstance(source, ParserSource):
                    raise ValueError("invalid parser child request")
                profile = _resolve_profile_value(profile_value)
                if kind == "probe":
                    total_pages, surfaces = probe_pdf(
                        source, limits,
                        include_surfaces=profile.preset is ParsingPreset.MULTIMODAL_LOCAL_V2,
                    )
                    send(_PDF_PROBE_RESPONSE + json.dumps({
                        "schema": _PDF_PROBE_SCHEMA, "total_pages": total_pages,
                        "surfaces": sorted(surfaces),
                    }).encode())
                    continue
                started_at = time.monotonic()

                def emit(payload: dict[str, Any]) -> None:
                    send(
                        _encode_progress_response(
                            {
                                **payload,
                                "elapsed_ms": int(
                                    (time.monotonic() - started_at) * 1000
                                ),
                            }
                        )
                    )

                document = runtime.convert(
                    source,
                    profile,
                    page_range=page_range,
                    progress_emitter=emit,
                )
                send(
                    _SUCCESS_RESPONSE + document.model_dump_json().encode("utf-8")
                )
            except ParserExecutionError as error:
                send(
                    _encode_error_response(error.code, error.diagnostic)
                )
            except BaseException:
                try:
                    send(
                        _encode_error_response(
                            ErrorCode.PARSER_CRASHED,
                            {"check": "docling_child_runtime"},
                        )
                    )
                except BaseException:
                    pass
                return
    finally:
        connection.close()


def _configure_pillow_resource_guard(limits: ParserLimits) -> None:
    """Make Pillow's lazy decoder fail at the application's pixel budget."""

    PillowImage.MAX_IMAGE_PIXELS = limits.max_image_pixels
    warnings.filterwarnings(
        "error",
        category=PillowImage.DecompressionBombWarning,
    )


def _preflight_conversion_source(
    source: ParserSource,
    limits: ParserLimits,
) -> None:
    extension = PurePath(source.original_filename).suffix.lower()
    try:
        if extension == ".csv":
            text = source.content.decode("utf-8-sig", errors="strict")
            validate_csv_structure(
                text.replace("\r\n", "\n").replace("\r", "\n"),
                max_columns=limits.max_csv_columns,
                max_cells=limits.max_csv_cells,
            )
        elif extension in {".docx", ".pptx", ".xlsx"}:
            with ZipFile(BytesIO(source.content)) as archive:
                if extension == ".xlsx":
                    validate_xlsx_structure(
                        archive,
                        max_cells=limits.max_xlsx_cells,
                        max_columns=limits.max_xlsx_columns,
                        max_sheets=limits.max_xlsx_sheets,
                        max_xml_bytes=limits.max_xlsx_xml_bytes,
                    )
                validate_ooxml_images(
                    archive,
                    extension=extension,
                    max_images=limits.max_assets,
                    max_image_width=limits.max_image_width,
                    max_image_height=limits.max_image_height,
                    max_image_pixels=limits.max_image_pixels,
                    max_total_image_pixels=limits.max_total_image_pixels,
                )
    except ResourcePreflightLimitError as error:
        diagnostic = {
            "limit_name": error.limit_name,
            "limit": error.limit,
        }
        if error.observed is not None:
            diagnostic["observed"] = error.observed
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic=diagnostic,
        ) from error
    except (
        ResourcePreflightContentError,
        BadZipFile,
        OSError,
        UnicodeError,
        ValueError,
    ) as error:
        raise ParserExecutionError(
            ErrorCode.FILE_CONTENT_INVALID,
            diagnostic={"check": "resource_preflight"},
        ) from error


def _round_trip(
    connection: Connection,
    source: ParserSource,
    profile: ParserProfile,
    page_range: tuple[int, int] | None,
    on_progress: Callable[[dict[str, Any]], None] | None,
    operation: str = "parse",
) -> _RoundTripResult:
    request = (
        (operation, source, profile.value)
        if page_range is None and not profile.uses_balanced_pdf_runtime
        else (operation, source, profile.value, page_range)
    )
    connection.send(request)
    last_progress: dict[str, Any] = {}
    while True:
        response = connection.recv_bytes(_MAX_IPC_RESPONSE_BYTES)
        if response[:1] != _PROGRESS_RESPONSE:
            return _RoundTripResult(response, last_progress)
        progress = _decode_progress_response(response)
        last_progress = progress
        if on_progress is not None:
            on_progress(progress)


def _resolve_profile(
    *,
    profile: ParserProfile | ParsingPreset | None,
    preset: ParsingPreset | None,
) -> ParserProfile:
    if profile is not None:
        if isinstance(profile, ParsingPreset):
            preset = profile
        else:
            return ParserProfile(profile)
    if preset is None:
        raise ValueError("parser profile is required")
    resolved_preset = ParsingPreset(preset)
    return (
        ParserProfile.DOCLING_MULTIMODAL_LOCAL_V2
        if resolved_preset is ParsingPreset.MULTIMODAL_LOCAL_V2
        else ParserProfile.DOCLING_TEXT_LOCAL_V1
    )


def _resolve_profile_value(value: Any) -> ParserProfile:
    try:
        return ParserProfile(value)
    except ValueError:
        return _resolve_profile(profile=ParsingPreset(value), preset=None)


def _decode_pdf_probe(
    response: bytes, limits: ParserLimits,
) -> tuple[int, frozenset[int]]:
    try:
        if response[:1] != _PDF_PROBE_RESPONSE or len(response) > 65536:
            raise ValueError("invalid probe response")
        probe = json.loads(response[1:])
        count, surfaces = probe["total_pages"], probe["surfaces"]
        if (probe.get("schema") != _PDF_PROBE_SCHEMA or type(count) is not int or count < 1
                or not isinstance(surfaces, list) or len(surfaces) > count
                or any(type(n) is not int or not 1 <= n <= count for n in surfaces)
                or len(set(surfaces)) != len(surfaces)):
            raise ValueError("invalid probe metadata")
        require_limit("max_num_pages", count, limits)
        return count, frozenset(surfaces)
    except ParserExecutionError:
        raise
    except Exception as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_CRASHED,
            diagnostic={"check": "pdf_child_probe"},
        ) from error


async def _consume_progress(
    queue: asyncio.Queue[ParserProgress | None],
    handler: ParserProgressHandler | None,
) -> None:
    while True:
        progress = await queue.get()
        if progress is None:
            return
        await _publish_progress(handler, progress)


async def _publish_progress(
    handler: ParserProgressHandler | None,
    progress: ParserProgress,
) -> None:
    if handler is None:
        return
    try:
        await handler(progress)
    except Exception:
        # Progress is diagnostic. A transient status-write failure must not
        # corrupt or abort an otherwise healthy conversion.
        return


def _progress(
    checkpoint: dict[str, Any],
    segment_index: int,
    *,
    stage: str,
    current_stage_pages: dict[str, int] | None = None,
    current_ocr_pages: int = 0,
    current_ocr_regions: int = 0,
    current_table_candidates: int = 0,
    current_elapsed_ms: int = 0,
    child_peak_rss_bytes: int | None = None,
) -> ParserProgress:
    segments = checkpoint["segments"]
    safe_index = min(max(segment_index, 0), len(segments) - 1)
    segment = segments[safe_index]
    completed_segments = [
        item for item in segments if item["status"] == "completed"
    ]
    completed_pages = sum(
        item["page_to"] - item["page_from"] + 1
        for item in completed_segments
    )
    persisted_ocr_pages = sum(item.get("ocr_pages", 0) for item in completed_segments)
    persisted_ocr_regions = sum(
        item.get("ocr_regions", 0) for item in completed_segments
    )
    persisted_table_candidates = sum(
        item.get("table_candidates", 0) for item in completed_segments
    )
    persisted_stage_pages: dict[str, int] = {}
    for item in completed_segments:
        for name, value in item.get("stage_pages", {}).items():
            persisted_stage_pages[name] = persisted_stage_pages.get(name, 0) + value
    peaks = [
        item.get("child_peak_rss_bytes")
        for item in completed_segments
        if isinstance(item.get("child_peak_rss_bytes"), int)
    ]
    if child_peak_rss_bytes is not None:
        peaks.append(child_peak_rss_bytes)
    current_stage_pages = dict(current_stage_pages or {})
    stage_pages = dict(persisted_stage_pages)
    for name, value in current_stage_pages.items():
        stage_pages[name] = stage_pages.get(name, 0) + value
    current_assembled = current_stage_pages.get("page_assembly", 0)
    if stage == "completed":
        completed_pages = checkpoint["total_pages"]
    else:
        completed_pages = min(
            checkpoint["total_pages"], completed_pages + current_assembled
        )
    return ParserProgress(
        stage=stage,
        total_pages=checkpoint["total_pages"],
        completed_pages=completed_pages,
        segment_number=safe_index + 1,
        segment_count=len(segments),
        page_from=segment["page_from"],
        page_to=segment["page_to"],
        stage_pages=tuple(sorted(stage_pages.items())),
        ocr_pages=persisted_ocr_pages + current_ocr_pages,
        ocr_regions=persisted_ocr_regions + current_ocr_regions,
        table_candidates=(
            persisted_table_candidates + current_table_candidates
        ),
        elapsed_ms=int(checkpoint["elapsed_ms"]) + current_elapsed_ms,
        child_peak_rss_bytes=max(peaks) if peaks else None,
    )


def _nonnegative_int(value: Any) -> int:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else 0
    )


def _encode_progress_response(progress: dict[str, Any]) -> bytes:
    return _PROGRESS_RESPONSE + json.dumps(
        progress,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _decode_progress_response(response: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(response[1:])
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_CRASHED,
            diagnostic={"check": "docling_child_progress"},
        ) from error
    if not isinstance(payload, dict):
        raise ParserExecutionError(
            ErrorCode.PARSER_CRASHED,
            diagnostic={"check": "docling_child_progress"},
        )
    allowed = {
        "stage",
        "segment_total_pages",
        "stage_completed_pages",
        "ocr_pages",
        "ocr_regions",
        "table_candidates",
        "elapsed_ms",
        "child_peak_rss_bytes",
    }
    return {key: value for key, value in payload.items() if key in allowed}


def _read_checkpoint_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = _read_private_bytes(path, _MAX_CHECKPOINT_MANIFEST_BYTES)
        if len(payload) > _MAX_CHECKPOINT_MANIFEST_BYTES:
            raise ValueError("manifest too large")
        value = json.loads(payload)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_CRASHED,
            diagnostic={"check": "pdf_checkpoint_manifest"},
        ) from error
    if not isinstance(value, dict):
        raise ParserExecutionError(
            ErrorCode.PARSER_CRASHED,
            diagnostic={"check": "pdf_checkpoint_manifest"},
        )
    return value


def _validate_checkpoint(checkpoint: dict[str, Any], total_pages: int) -> None:
    segments = checkpoint.get("segments")
    if (
        checkpoint.get("schema_version") != _CHECKPOINT_SCHEMA
        or not isinstance(checkpoint.get("elapsed_ms"), int)
        or checkpoint["elapsed_ms"] < 0
        or not isinstance(segments, list)
        or not segments
    ):
        raise ParserExecutionError(
            ErrorCode.PARSER_CRASHED,
            diagnostic={"check": "pdf_checkpoint_manifest"},
        )
    expected_page = 1
    for segment in segments:
        if (
            not isinstance(segment, dict)
            or segment.get("page_from") != expected_page
            or not isinstance(segment.get("page_to"), int)
            or segment["page_to"] < segment["page_from"]
            or segment["page_to"] > total_pages
            or segment.get("status") not in {"pending", "completed"}
        ):
            raise ParserExecutionError(
                ErrorCode.PARSER_CRASHED,
                diagnostic={"check": "pdf_checkpoint_manifest"},
            )
        if segment["status"] == "completed" and not all(
            isinstance(segment.get(key), expected_type)
            for key, expected_type in (
                ("filename", str),
                ("sha256", str),
                ("elapsed_ms", int),
                ("stage_pages", dict),
                ("ocr_pages", int),
                ("ocr_regions", int),
                ("table_candidates", int),
            )
        ):
            raise ParserExecutionError(
                ErrorCode.PARSER_CRASHED,
                diagnostic={"check": "pdf_checkpoint_manifest"},
            )
        if segment["status"] == "completed":
            expected_filename = (
                f"pages-{segment['page_from']}-{segment['page_to']}.json"
            )
            segment_page_count = segment["page_to"] - segment["page_from"] + 1
            if (
                segment["filename"] != expected_filename
                or not re.fullmatch(r"[0-9a-f]{64}", segment["sha256"])
                or any(
                    name not in _PDF_PROGRESS_STAGE_NAMES
                    or not isinstance(value, int)
                    or isinstance(value, bool)
                    or not 0 <= value <= segment_page_count
                    for name, value in segment["stage_pages"].items()
                )
                or any(
                    segment[name] < 0
                    for name in (
                        "elapsed_ms",
                        "ocr_pages",
                        "ocr_regions",
                        "table_candidates",
                    )
                )
            ):
                raise ParserExecutionError(
                    ErrorCode.PARSER_CRASHED,
                    diagnostic={"check": "pdf_checkpoint_manifest"},
                )
        expected_page = segment["page_to"] + 1
    if expected_page != total_pages + 1:
        raise ParserExecutionError(
            ErrorCode.PARSER_CRASHED,
            diagnostic={"check": "pdf_checkpoint_manifest"},
        )


def _write_private_json(path: Path, value: dict[str, Any]) -> None:
    _write_private_bytes(
        path,
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8"),
    )


def _write_private_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _read_private_bytes(path: Path, maximum: int = _MAX_IPC_RESPONSE_BYTES) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ParserExecutionError(
            ErrorCode.PARSER_CRASHED,
            diagnostic={"check": "pdf_checkpoint_file"},
        )
    try:
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if size > maximum:
                raise ParserExecutionError(
                    ErrorCode.PARSER_RESOURCE_LIMIT,
                    diagnostic={"limit_name": "max_checkpoint_bytes", "limit": maximum},
                )
            # read(maximum) allocates the ceiling even for a tiny checkpoint.
            # Bound by the opened file's size, retaining one byte for growth detection.
            payload = stream.read(size + 1)
        if len(payload) != size:
            raise ParserExecutionError(
                ErrorCode.PARSER_CRASHED,
                diagnostic={"check": "pdf_checkpoint_file_changed"},
            )
        return payload
    except OSError as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_CRASHED,
            diagnostic={"check": "pdf_checkpoint_file"},
        ) from error


def _decode_response(
    response: Any,
    limits: ParserLimits,
    *,
    allow_empty: bool = False,
) -> DoclingDocument:
    if not isinstance(response, bytes) or len(response) < 2:
        raise ParserExecutionError(
            ErrorCode.PARSER_CRASHED,
            diagnostic={"check": "docling_child_protocol"},
        )
    kind = response[:1]
    payload = response[1:]
    if kind == _SUCCESS_RESPONSE:
        try:
            document = DoclingDocument.model_validate_json(payload)
            _validate_document(document, limits, allow_empty=allow_empty)
            return document
        except ParserExecutionError:
            raise
        except Exception as error:
            raise ParserExecutionError(
                ErrorCode.PARSER_CRASHED,
                diagnostic={"check": "docling_child_protocol"},
            ) from error
    if kind == _ERROR_RESPONSE:
        try:
            error_payload = json.loads(payload)
            code = ErrorCode(error_payload["code"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            code = ErrorCode.PARSER_CRASHED
            error_payload = {}
        diagnostic = error_payload.get("diagnostic")
        raise ParserExecutionError(
            code,
            diagnostic=diagnostic if isinstance(diagnostic, dict) else {},
        )
    raise ParserExecutionError(
        ErrorCode.PARSER_CRASHED,
        diagnostic={"check": "docling_child_protocol"},
    )


def _encode_error_response(code: ErrorCode, diagnostic: dict[str, Any]) -> bytes:
    payload = json.dumps(
        {"code": code.value, "diagnostic": diagnostic},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _ERROR_RESPONSE + payload


def _stop_child(
    process: multiprocessing.Process | None,
    connection: Connection | None,
) -> int | None:
    if connection is not None:
        try:
            connection.close()
        except OSError:
            pass
    if process is None:
        return None
    if process.pid is None:
        process.close()
        return None
    exit_code = process.exitcode
    try:
        if process.is_alive():
            process.terminate()
        process.join(timeout=_PROCESS_STOP_GRACE_SECONDS)
        if process.is_alive():
            process.kill()
            process.join(timeout=_PROCESS_STOP_GRACE_SECONDS)
        exit_code = process.exitcode
    finally:
        if not process.is_alive():
            process.close()
    return exit_code


def _validate_source(
    source: ParserSource,
    limits: ParserLimits,
    preset: ParsingPreset,
) -> None:
    extension = PurePath(source.original_filename).suffix.lower()
    if source.media_type == MARKDOWN_BUNDLE_MEDIA_TYPE:
        if (
            preset is not ParsingPreset.MULTIMODAL_LOCAL_V2
            or extension not in {".md", ".mdz"}
        ):
            raise ParserExecutionError(ErrorCode.PARSER_NOT_CONFIGURED)
        if len(source.content) > limits.max_markdown_bundle_size:
            raise ParserExecutionError(
                ErrorCode.PARSER_RESOURCE_LIMIT,
                diagnostic={
                    "limit_name": "max_markdown_bundle_size",
                    "limit": limits.max_markdown_bundle_size,
                },
            )
        return
    expected_media_type = _MEDIA_TYPES.get(extension)
    if expected_media_type is None:
        raise ParserExecutionError(ErrorCode.PARSER_NOT_CONFIGURED)
    if source.media_type != expected_media_type:
        raise ParserExecutionError(ErrorCode.FILE_MEDIA_TYPE_MISMATCH)
    if len(source.content) > limits.max_file_size:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic={
                "limit_name": "max_file_size",
                "limit": limits.max_file_size,
            },
        )


def _validate_conversion_result(
    result: Any,
    limits: ParserLimits,
    *,
    allow_empty: bool = False,
) -> DoclingDocument:
    status = getattr(result, "status", None)
    errors = getattr(result, "errors", ()) or ()
    diagnostic = {
        "check": "conversion_status",
        "status": getattr(status, "value", str(status)),
        "error_count": len(errors),
    }
    if status is ConversionStatus.PARTIAL_SUCCESS:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic=diagnostic,
        )
    if status is ConversionStatus.FAILURE:
        code = (
            ErrorCode.FILE_CONTENT_INVALID
            if _is_input_failure(result, errors)
            else ErrorCode.PARSER_OUTPUT_INVALID
        )
        raise ParserExecutionError(code, diagnostic=diagnostic)
    if status is not ConversionStatus.SUCCESS:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic=diagnostic,
        )
    document = getattr(result, "document", None)
    if not isinstance(document, DoclingDocument):
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "docling_document_type"},
        )
    _validate_document(document, limits, allow_empty=allow_empty)
    return document


def _is_input_failure(result: Any, errors: Any) -> bool:
    input_document = getattr(result, "input", None)
    if input_document is not None and getattr(input_document, "valid", True) is False:
        return True
    for item in errors:
        component = getattr(item, "component_type", None)
        category = getattr(item, "category", None)
        if component is DoclingComponentType.USER_INPUT:
            return True
        if category is FailureCategory.SOURCE_UNAVAILABLE:
            return True
    return False


def _validate_document(
    document: DoclingDocument,
    limits: ParserLimits,
    *,
    allow_empty: bool = False,
) -> dict[str, int]:
    if document.schema_name != "DoclingDocument":
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "docling_document_schema"},
        )
    if document.version != DOCLING_DOCUMENT_VERSION:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "docling_document_version"},
        )
    if len(document.pages) > limits.max_num_pages:
        _raise_limit("max_num_pages", limits.max_num_pages)

    item_count = 0
    character_count = 0
    for item, _level in document.iterate_items():
        item_count += 1
        if item_count > limits.max_docling_items:
            _raise_limit("max_docling_items", limits.max_docling_items)
        text = getattr(item, "text", None)
        if isinstance(text, str):
            character_count += len(text)
        data = getattr(item, "data", None)
        cells = getattr(data, "table_cells", ()) if data is not None else ()
        for cell in cells or ():
            cell_text = getattr(cell, "text", None)
            if isinstance(cell_text, str):
                character_count += len(cell_text)
        if character_count > limits.max_extracted_characters:
            _raise_limit(
                "max_extracted_characters",
                limits.max_extracted_characters,
            )
    if item_count == 0 and not allow_empty:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "non_empty_docling_items"},
        )

    return {
        "max_num_pages": len(document.pages),
        "max_docling_items": item_count,
        "max_extracted_characters": character_count,
        **image_usage(document, limits),
    }


def _raise_limit(limit_name: str, limit: int) -> None:
    raise ParserExecutionError(
        ErrorCode.PARSER_RESOURCE_LIMIT,
        diagnostic={"limit_name": limit_name, "limit": limit},
    )


def _consume_cancelled_result(future: asyncio.Future[Any]) -> None:
    if future.cancelled():
        return
    try:
        future.exception()
    except BaseException:
        return
