"""Thin, parse-once gateway from bounded bytes to ``DoclingDocument``."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import multiprocessing
from multiprocessing.connection import Connection
from pathlib import Path, PurePath
import re
from tempfile import TemporaryDirectory
import threading
from typing import Any

from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    DocumentStream,
    FailureCategory,
    InputFormat,
)
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument

from rag_kb.adapters.parser.docling.artifacts import (
    ArtifactManifestError,
    verify_docling_artifacts,
)
from rag_kb.adapters.parser.docling.factory import build_docling_converter
from rag_kb.domain import (
    ErrorCode,
    FileAdmissionError,
    ParserExecutionError,
    ParserLimits,
    ParserSource,
    ParsingPreset,
)
from rag_kb.document_processing.markdown_bundle import (
    MARKDOWN_BUNDLE_MEDIA_TYPE,
    read_normalized_markdown_bundle,
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
_DATA_URI_PREFIX = "data:"
_BASE64_MARKER = ";base64,"
_BASE64_PAYLOAD = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")
DOCLING_DOCUMENT_VERSION = "1.10.0"
ConverterFactory = Callable[..., DocumentConverter]
ChildTarget = Callable[[Connection, ParserLimits, Path, Path], None]
_PROCESS_STOP_GRACE_SECONDS = 1.0
_MAX_IPC_RESPONSE_BYTES = 256 * 1024 * 1024
_SUCCESS_RESPONSE = b"O"
_ERROR_RESPONSE = b"E"


class DoclingParser:
    """Run serial Docling conversions in a reusable, killable child process."""

    def __init__(
        self,
        limits: ParserLimits,
        *,
        artifacts_path: Path,
        artifact_manifest_path: Path,
        child_target: ChildTarget | None = None,
    ) -> None:
        self._limits = limits
        self._artifacts_path = artifacts_path
        self._artifact_manifest_path = artifact_manifest_path
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
        preset: ParsingPreset,
    ) -> DoclingDocument:
        """Convert one in-memory source exactly once and return its native model."""

        try:
            resolved_preset = ParsingPreset(preset)
        except ValueError as error:
            raise ParserExecutionError(ErrorCode.PARSER_NOT_CONFIGURED) from error
        _validate_source(source, self._limits, resolved_preset)
        async with self._request_lock:
            connection = self._ensure_child()
            loop = asyncio.get_running_loop()
            try:
                concurrent_future = self._ipc_executor.submit(
                    _round_trip,
                    connection,
                    source,
                    resolved_preset,
                )
            except RuntimeError as error:
                raise ParserExecutionError(
                    ErrorCode.PARSER_NOT_CONFIGURED,
                    diagnostic={"check": "docling_parser_closed"},
                ) from error
            wrapped = asyncio.wrap_future(concurrent_future, loop=loop)
            try:
                async with asyncio.timeout(self._limits.document_timeout_seconds):
                    response = await asyncio.shield(wrapped)
            except TimeoutError as error:
                wrapped.add_done_callback(_consume_cancelled_result)
                await self._reset_child()
                raise ParserExecutionError(
                    ErrorCode.PARSER_RESOURCE_LIMIT,
                    diagnostic={
                        "limit_name": "document_timeout",
                        "limit": self._limits.document_timeout_seconds,
                    },
                ) from error
            except asyncio.CancelledError:
                wrapped.add_done_callback(_consume_cancelled_result)
                await self._reset_child()
                raise
            except (EOFError, BrokenPipeError, OSError) as error:
                await self._reset_child()
                raise ParserExecutionError(
                    ErrorCode.PARSER_CRASHED,
                    diagnostic={"check": "docling_child_ipc"},
                ) from error

            try:
                return _decode_response(response, self._limits)
            except ParserExecutionError as error:
                if error.code is ErrorCode.PARSER_CRASHED:
                    await self._reset_child()
                raise

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

    async def _reset_child(self) -> None:
        with self._state_lock:
            resources = self._detach_child()
        await asyncio.shield(asyncio.to_thread(_stop_child, *resources))

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
        self._converters: dict[ParsingPreset, DocumentConverter] = {}
        self._artifacts_verified = False

    def convert(
        self,
        source: ParserSource,
        preset: ParsingPreset,
    ) -> DoclingDocument:
        try:
            converter = self._get_converter(preset)
            if source.media_type == MARKDOWN_BUNDLE_MEDIA_TYPE:
                result = self._convert_markdown_bundle(converter, source.content)
            else:
                stream = DocumentStream(
                    name=source.original_filename,
                    stream=BytesIO(source.content),
                )
                result = converter.convert(
                    stream,
                    raises_on_error=False,
                    max_num_pages=self._limits.max_num_pages,
                    max_file_size=self._limits.max_file_size,
                )
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
        except Exception as error:
            raise ParserExecutionError(
                ErrorCode.PARSER_CRASHED,
                diagnostic={"check": "docling_conversion"},
            ) from error
        return _validate_conversion_result(result, self._limits)

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

    def _get_converter(self, preset: ParsingPreset) -> DocumentConverter:
        converter = self._converters.get(preset)
        if converter is not None:
            return converter
        if not self._artifacts_verified:
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
        try:
            converter = self._converter_factory(
                preset,
                artifacts_path=self._artifacts_path,
                limits=self._limits,
            )
        except (FileNotFoundError, ImportError, ModuleNotFoundError) as error:
            raise ParserExecutionError(
                ErrorCode.PARSER_NOT_CONFIGURED,
                diagnostic={"check": "docling_converter_factory"},
            ) from error
        self._converters[preset] = converter
        return converter


def _parser_child(
    connection: Connection,
    limits: ParserLimits,
    artifacts_path: Path,
    artifact_manifest_path: Path,
) -> None:
    runtime = _DoclingRuntime(
        limits,
        artifacts_path=artifacts_path,
        artifact_manifest_path=artifact_manifest_path,
    )
    try:
        while True:
            try:
                request = connection.recv()
            except EOFError:
                return
            try:
                kind, source, preset_value = request
                if kind != "parse" or not isinstance(source, ParserSource):
                    raise ValueError("invalid parser child request")
                preset = ParsingPreset(preset_value)
                document = runtime.convert(source, preset)
                connection.send_bytes(
                    _SUCCESS_RESPONSE + document.model_dump_json().encode("utf-8")
                )
            except ParserExecutionError as error:
                connection.send_bytes(
                    _encode_error_response(error.code, error.diagnostic)
                )
            except BaseException:
                try:
                    connection.send_bytes(
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


def _round_trip(
    connection: Connection,
    source: ParserSource,
    preset: ParsingPreset,
) -> bytes:
    connection.send(("parse", source, preset.value))
    return connection.recv_bytes(_MAX_IPC_RESPONSE_BYTES)


def _decode_response(response: Any, limits: ParserLimits) -> DoclingDocument:
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
            _validate_document(document, limits)
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
) -> None:
    if connection is not None:
        try:
            connection.close()
        except OSError:
            pass
    if process is None:
        return
    if process.pid is None:
        process.close()
        return
    try:
        if process.is_alive():
            process.terminate()
        process.join(timeout=_PROCESS_STOP_GRACE_SECONDS)
        if process.is_alive():
            process.kill()
            process.join(timeout=_PROCESS_STOP_GRACE_SECONDS)
    finally:
        if not process.is_alive():
            process.close()


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
    _validate_document(document, limits)
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


def _validate_document(document: DoclingDocument, limits: ParserLimits) -> None:
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
    if item_count == 0:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "non_empty_docling_items"},
        )

    image_refs = [
        page.image for page in document.pages.values() if page.image is not None
    ]
    image_refs.extend(
        item.image for item in document.pictures if item.image is not None
    )
    image_refs.extend(item.image for item in document.tables if item.image is not None)
    if len(image_refs) > limits.max_assets:
        _raise_limit("max_assets", limits.max_assets)
    total_asset_bytes = 0
    for image_ref in image_refs:
        width = int(image_ref.size.width)
        height = int(image_ref.size.height)
        if width > limits.max_image_width:
            _raise_limit("max_image_width", limits.max_image_width)
        if height > limits.max_image_height:
            _raise_limit("max_image_height", limits.max_image_height)
        if width * height > limits.max_image_pixels:
            _raise_limit("max_image_pixels", limits.max_image_pixels)
        total_asset_bytes += _local_image_bytes(image_ref.uri)
        if total_asset_bytes > limits.max_total_asset_bytes:
            _raise_limit("max_total_asset_bytes", limits.max_total_asset_bytes)


def _local_image_bytes(uri: Any) -> int:
    if isinstance(uri, Path):
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "docling_image_ref"},
        )
    serialized = str(uri)
    if not serialized.startswith(_DATA_URI_PREFIX) or _BASE64_MARKER not in serialized:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "docling_image_ref"},
        )
    encoded = serialized.split(_BASE64_MARKER, 1)[1]
    try:
        if len(encoded) % 4 or _BASE64_PAYLOAD.fullmatch(encoded) is None:
            raise ValueError("invalid base64 payload")
        padding = encoded.count("=")
        return (len(encoded) * 3 // 4) - padding
    except (TypeError, ValueError) as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "docling_image_ref"},
        ) from error


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
