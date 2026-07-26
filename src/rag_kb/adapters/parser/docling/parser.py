"""Thin, parse-once gateway from bounded bytes to ``DoclingDocument``."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path, PurePath
import re
from typing import Any

from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    DocumentStream,
    FailureCategory,
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
    ParserExecutionError,
    ParserLimits,
    ParserSource,
    ParsingPreset,
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


class DoclingParser:
    """Run one native Docling conversion at a time in a dedicated executor."""

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
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rag-kb-docling",
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
        _validate_source(source, self._limits)
        if self._closed:
            raise ParserExecutionError(
                ErrorCode.PARSER_NOT_CONFIGURED,
                diagnostic={"check": "docling_parser_closed"},
            )
        try:
            concurrent_future = self._executor.submit(
                self._convert_once,
                source,
                resolved_preset,
            )
        except RuntimeError as error:
            raise ParserExecutionError(
                ErrorCode.PARSER_NOT_CONFIGURED,
                diagnostic={"check": "docling_parser_closed"},
            ) from error
        wrapped = asyncio.wrap_future(concurrent_future)
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            wrapped.add_done_callback(_consume_cancelled_result)
            raise

    def close(self, *, wait: bool = True) -> None:
        """Stop accepting conversions and release the owned executor."""

        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _convert_once(
        self,
        source: ParserSource,
        preset: ParsingPreset,
    ) -> DoclingDocument:
        try:
            converter = self._get_converter(preset)
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


def _validate_source(source: ParserSource, limits: ParserLimits) -> None:
    extension = PurePath(source.original_filename).suffix.lower()
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


def _consume_cancelled_result(future: asyncio.Future[DoclingDocument]) -> None:
    if future.cancelled():
        return
    try:
        future.exception()
    except BaseException:
        return
