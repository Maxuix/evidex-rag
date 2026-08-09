from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import signal
import struct
import threading
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch
from uuid import uuid4
import warnings
import zlib
from zipfile import ZIP_DEFLATED, ZipFile

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    DocumentStream,
    ErrorItem,
    FailureCategory,
    InputFormat,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    ImageRef,
    Size,
    TableCell,
    TableData,
)
from docling_core.types.doc.common.origin import DocumentOrigin
from PIL import Image
from pypdf import PdfWriter

from rag_kb.adapters.parser.docling.artifacts import (
    ArtifactManifestError,
    verify_docling_artifacts,
)
from rag_kb.adapters.parser.docling.factory import build_docling_converter
from rag_kb.adapters.parser.docling.parser import (
    DoclingParser,
    _DoclingRuntime,
    _configure_pillow_resource_guard,
    _validate_source,
)
import rag_kb.adapters.parser.docling.parser as parser_module
from rag_kb.domain import (
    ErrorCode,
    ParserExecutionError,
    ParserLimits,
    ParserProfile,
    ParserSource,
    ParsingPreset,
)
from rag_kb.document_processing.markdown_bundle import (
    MARKDOWN_BUNDLE_MEDIA_TYPE,
)
from rag_kb.ports.parsing import DocumentParseContinuation, DocumentParseResult


def _document(*texts: str) -> DoclingDocument:
    document = DoclingDocument(name="fixture")
    for text in texts:
        document.add_text(label=DocItemLabel.TEXT, text=text)
    return document


def _result(
    status: ConversionStatus,
    *,
    document: DoclingDocument | None = None,
    errors: list[ErrorItem] | None = None,
    input_valid: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        document=document,
        errors=errors or [],
        input=SimpleNamespace(valid=input_valid),
    )


def _png_with_dimensions(width: int, height: int) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", checksum)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IEND", b"")


def _docx_with_image(image: bytes) -> bytes:
    target = BytesIO()
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", b"<Types/>")
        archive.writestr("word/document.xml", b"<document/>")
        archive.writestr("word/media/image.png", image)
    return target.getvalue()


def _pdf_with_pages(count: int) -> bytes:
    target = BytesIO()
    writer = PdfWriter()
    for _ in range(count):
        writer.add_blank_page(width=100, height=100)
    writer.write(target)
    return target.getvalue()


class _FakeConverter:
    def __init__(
        self,
        result: SimpleNamespace | None = None,
        *,
        failure: Exception | None = None,
    ) -> None:
        self.result = result or _result(
            ConversionStatus.SUCCESS,
            document=_document("body"),
        )
        self.failure = failure
        self.calls: list[tuple[object, dict[str, object]]] = []

    def convert(self, source: object, **kwargs: object) -> SimpleNamespace:
        self.calls.append((source, kwargs))
        if self.failure is not None:
            raise self.failure
        return self.result


class _InProcessParser:
    def __init__(self, runtime: _DoclingRuntime, limits: ParserLimits) -> None:
        self._runtime = runtime
        self._limits = limits

    async def parse(
        self,
        source: ParserSource,
        *,
        preset: ParsingPreset,
    ) -> DoclingDocument:
        resolved_preset = ParsingPreset(preset)
        _validate_source(source, self._limits, resolved_preset)
        return self._runtime.convert(source, resolved_preset)

    def close(self) -> None:
        pass


class _ParserHarness:
    def __init__(
        self,
        test_case: unittest.TestCase,
        converter: _FakeConverter,
        *,
        limits: ParserLimits | None = None,
    ) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        root = Path(self._temporary.name)
        artifact = root / "model.bin"
        artifact.write_bytes(b"model")
        manifest = root / "manifest.json"
        _write_manifest(manifest, artifact)
        calls: list[ParsingPreset] = []

        def factory(
            preset: ParsingPreset,
            *,
            artifacts_path: Path,
            limits: ParserLimits,
        ) -> _FakeConverter:
            test_case.assertEqual(artifacts_path, root)
            calls.append(preset)
            return converter

        self.calls = calls
        resolved_limits = limits or ParserLimits()
        runtime = _DoclingRuntime(
            resolved_limits,
            artifacts_path=root,
            artifact_manifest_path=manifest,
            converter_factory=factory,
        )
        self.parser = _InProcessParser(runtime, resolved_limits)

    def close(self) -> None:
        self.parser.close()
        self._temporary.cleanup()


class _ProcessHarness:
    def __init__(
        self,
        *,
        child_target,
        timeout: float,
    ) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        root = Path(self._temporary.name)
        self.root = root
        self.parser = DoclingParser(
            replace(ParserLimits(), document_timeout_seconds=timeout),
            artifacts_path=root,
            artifact_manifest_path=root / "manifest.json",
            checkpoint_root=root / "parser-temp",
            child_target=child_target,
        )

    def close(self) -> None:
        self.parser.close()
        self._temporary.cleanup()


def _echo_child(connection, limits, artifacts_path, artifact_manifest_path) -> None:
    del limits, artifacts_path, artifact_manifest_path
    try:
        while True:
            try:
                kind, source, _preset = connection.recv()
            except EOFError:
                return
            if kind != "parse":
                return
            document = _document(source.original_filename)
            document.name = str(os.getpid())
            connection.send_bytes(b"O" + document.model_dump_json().encode("utf-8"))
    finally:
        connection.close()


def _hang_once_child(connection, limits, artifacts_path, artifact_manifest_path) -> None:
    del limits, artifact_manifest_path
    marker = artifacts_path / "first-child-hung"
    try:
        kind, source, _preset = connection.recv()
        if kind != "parse":
            return
        if not marker.exists():
            marker.write_text(str(os.getpid()), encoding="utf-8")
            time.sleep(30)
            return
        document = _document(source.original_filename)
        connection.send_bytes(b"O" + document.model_dump_json().encode("utf-8"))
    finally:
        connection.close()


def _crash_child(connection, limits, artifacts_path, artifact_manifest_path) -> None:
    del limits, artifacts_path, artifact_manifest_path
    connection.recv()
    os._exit(17)


def _oom_killed_child(
    connection,
    limits,
    artifacts_path,
    artifact_manifest_path,
) -> None:
    del limits, artifacts_path, artifact_manifest_path
    connection.recv()
    os.kill(os.getpid(), signal.SIGKILL)


def _segmented_child(connection, limits, artifacts_path, artifact_manifest_path) -> None:
    del limits, artifacts_path, artifact_manifest_path
    try:
        while True:
            try:
                kind, source, _profile, page_range = connection.recv()
            except EOFError:
                return
            if kind != "parse" or page_range is None:
                return
            if source.original_filename == "budget.pdf":
                time.sleep(0.02)
            page_from, page_to = page_range
            connection.send_bytes(
                b"P"
                + json.dumps(
                    {
                        "stage": "layout",
                        "stage_completed_pages": page_to - page_from + 1,
                        "ocr_pages": 0,
                        "ocr_regions": 0,
                        "table_candidates": 0,
                        "elapsed_ms": 1,
                        "child_peak_rss_bytes": 1024,
                    }
                ).encode("utf-8")
            )
            document = _document(f"pages-{page_from}-{page_to}")
            document.origin = DocumentOrigin(
                mimetype="application/pdf",
                binary_hash=17,
                filename="segmented.pdf",
            )
            for page_no in range(page_from, page_to + 1):
                document.add_page(
                    page_no=page_no,
                    size=Size(width=100, height=100),
                )
            connection.send_bytes(
                b"O" + document.model_dump_json().encode("utf-8")
            )
    finally:
        connection.close()


def _write_manifest(path: Path, artifact: Path) -> None:
    content = artifact.read_bytes()
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "docling_native_v1",
                "docling_version": version("docling"),
                "docling_core_version": version("docling-core"),
                "docling_document_version": "1.10.0",
                "artifacts": [
                    {
                        "path": artifact.name,
                        "size": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


class DoclingArtifactManifestTests(unittest.TestCase):
    def test_verifies_exact_local_content_and_package_versions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.bin"
            artifact.write_bytes(b"model")
            manifest_path = root / "manifest.json"
            _write_manifest(manifest_path, artifact)

            manifest = verify_docling_artifacts(root, manifest_path)

            self.assertEqual(manifest.profile, "docling_native_v1")
            self.assertEqual(len(manifest.entries), 1)

    def test_rejects_digest_mismatch_and_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.bin"
            artifact.write_bytes(b"model")
            manifest_path = root / "manifest.json"
            _write_manifest(manifest_path, artifact)
            artifact.write_bytes(b"changed")
            with self.assertRaises(ArtifactManifestError):
                verify_docling_artifacts(root, manifest_path)

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["artifacts"][0]["path"] = "../model.bin"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ArtifactManifestError):
                verify_docling_artifacts(root, manifest_path)


class DoclingConverterFactoryTests(unittest.TestCase):
    def test_freezes_local_cpu_rapidocr_and_preset_images(self) -> None:
        limits = ParserLimits()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = build_docling_converter(
                ParsingPreset.TEXT_LOCAL_V1,
                artifacts_path=root,
                limits=limits,
            )
            multimodal = build_docling_converter(
                ParsingPreset.MULTIMODAL_LOCAL_V2,
                artifacts_path=root,
                limits=limits,
            )

        self.assertEqual(
            set(text.allowed_formats),
            {
                InputFormat.MD,
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.CSV,
                InputFormat.PPTX,
                InputFormat.XLSX,
            },
        )
        text_options = text.format_to_options[InputFormat.PDF].pipeline_options
        multimodal_options = (
            multimodal.format_to_options[InputFormat.PDF].pipeline_options
        )
        self.assertEqual(text_options.document_timeout, 600)
        self.assertFalse(text_options.enable_remote_services)
        self.assertFalse(text_options.allow_external_plugins)
        self.assertTrue(text_options.do_ocr)
        self.assertEqual(text_options.ocr_options.kind, "rapidocr")
        self.assertEqual(text_options.ocr_options.lang, ["chinese"])
        self.assertEqual(text_options.ocr_options.backend, "onnxruntime")
        self.assertFalse(text_options.ocr_options.force_full_page_ocr)
        self.assertEqual(
            text_options.accelerator_options.device,
            AcceleratorDevice.CPU,
        )
        self.assertEqual(text_options.accelerator_options.num_threads, 1)
        self.assertFalse(text_options.generate_page_images)
        self.assertFalse(text_options.generate_picture_images)
        self.assertTrue(multimodal_options.generate_page_images)
        self.assertTrue(multimodal_options.generate_picture_images)
        self.assertIs(
            text.format_to_options[InputFormat.PDF].pipeline_cls,
            StandardPdfPipeline,
        )
        for input_format in (
            InputFormat.MD,
            InputFormat.HTML,
            InputFormat.CSV,
            InputFormat.DOCX,
            InputFormat.PPTX,
            InputFormat.XLSX,
        ):
            options = text.format_to_options[input_format].pipeline_options
            self.assertIs(
                text.format_to_options[input_format].pipeline_cls,
                SimplePipeline,
            )
            self.assertEqual(options.document_timeout, 600)
            self.assertFalse(options.enable_remote_services)
            self.assertFalse(options.allow_external_plugins)
            self.assertEqual(options.accelerator_options.device, AcceleratorDevice.CPU)

    def test_balanced_profile_uses_benchmarked_batches_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            converter = build_docling_converter(
                ParserProfile.DOCLING_TEXT_LOCAL_V2,
                artifacts_path=Path(directory),
                limits=ParserLimits(),
            )

        options = converter.format_to_options[InputFormat.PDF].pipeline_options
        self.assertEqual(options.accelerator_options.num_threads, 1)
        self.assertEqual(options.ocr_batch_size, 1)
        self.assertEqual(options.layout_batch_size, 1)
        self.assertEqual(options.table_batch_size, 1)
        self.assertEqual(options.document_timeout, 180)
        self.assertEqual(
            converter.format_to_options[InputFormat.PDF].pipeline_cls.__name__,
            "ProgressStandardPdfPipeline",
        )

    def test_docling_document_exposes_page_picture_and_table_images(self) -> None:
        document = _document("body")
        image = Image.new("RGB", (8, 6), "white")
        image_ref = ImageRef.from_pil(image, dpi=72)
        page = document.add_page(page_no=1, size=Size(width=8, height=6), image=image_ref)
        picture = document.add_picture(image=image_ref)
        table = document.add_table(
            data=TableData(
                num_rows=1,
                num_cols=1,
                table_cells=[
                    TableCell(
                        start_row_offset_idx=0,
                        end_row_offset_idx=1,
                        start_col_offset_idx=0,
                        end_col_offset_idx=1,
                        text="cell",
                    )
                ],
            )
        )

        self.assertEqual(page.image.pil_image.size, image.size)
        self.assertEqual(picture.get_image(document).size, image.size)
        self.assertEqual(document.pictures, [picture])
        self.assertEqual(document.tables, [table])
        self.assertTrue(callable(table.get_image))


class DoclingParserTests(unittest.IsolatedAsyncioTestCase):
    async def test_balanced_pdf_yields_segments_and_reassembles_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parser = DoclingParser(
                replace(
                    ParserLimits(),
                    pdf_segment_pages=2,
                    pdf_segment_timeout_seconds=5,
                    pdf_total_timeout_seconds=30,
                ),
                artifacts_path=root,
                artifact_manifest_path=root / "manifest.json",
                checkpoint_root=root / "parser-temp",
                child_target=_segmented_child,
            )
            self.addCleanup(parser.close)
            checkpoint_key = str(uuid4())
            source = ParserSource(
                "three-pages.pdf",
                "application/pdf",
                _pdf_with_pages(3),
            )
            observed = []

            async def on_progress(progress):
                observed.append(progress)

            first = await parser.parse(
                source,
                profile=ParserProfile.DOCLING_TEXT_LOCAL_V2,
                checkpoint_key=checkpoint_key,
                on_progress=on_progress,
            )
            second = await parser.parse(
                source,
                profile=ParserProfile.DOCLING_TEXT_LOCAL_V2,
                checkpoint_key=checkpoint_key,
                on_progress=on_progress,
            )

            self.assertIsInstance(first, DocumentParseContinuation)
            self.assertIsInstance(second, DocumentParseResult)
            assert isinstance(first, DocumentParseContinuation)
            assert isinstance(second, DocumentParseResult)
            self.assertEqual(tuple(second.document.pages), (1, 2, 3))
            self.assertEqual(second.document.origin.mimetype, "application/pdf")
            self.assertEqual(first.progress.completed_pages, 2)
            self.assertEqual(first.progress.stage_pages, (("layout", 2),))
            self.assertTrue(any(item.stage == "layout" for item in observed))
            self.assertEqual(observed[-1].stage, "completed")
            self.assertEqual(observed[-1].stage_pages, (("layout", 3),))
            checkpoint = (
                root / "parser-temp" / "pdf-checkpoints" / checkpoint_key
            )
            self.assertTrue(checkpoint.is_dir())
            parser.discard_checkpoint(checkpoint_key)
            self.assertFalse(checkpoint.exists())

    async def test_pdf_total_budget_is_checked_after_segment_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parser = DoclingParser(
                replace(
                    ParserLimits(),
                    pdf_segment_pages=2,
                    pdf_segment_timeout_seconds=5,
                    pdf_total_timeout_seconds=5,
                ),
                artifacts_path=root,
                artifact_manifest_path=root / "manifest.json",
                checkpoint_root=root / "parser-temp",
                child_target=_segmented_child,
            )
            self.addCleanup(parser.close)
            checkpoint_key = str(uuid4())
            source = ParserSource(
                "budget.pdf",
                "application/pdf",
                _pdf_with_pages(3),
            )

            first = await parser.parse(
                source,
                profile=ParserProfile.DOCLING_TEXT_LOCAL_V2,
                checkpoint_key=checkpoint_key,
            )
            self.assertIsInstance(first, DocumentParseContinuation)
            manifest_path = (
                root
                / "parser-temp"
                / "pdf-checkpoints"
                / checkpoint_key
                / "manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["elapsed_ms"] = 4_990
            manifest_path.write_text(
                json.dumps(manifest, separators=(",", ":"), sort_keys=True),
                encoding="utf-8",
            )

            with self.assertRaises(ParserExecutionError) as raised:
                await parser.parse(
                    source,
                    profile=ParserProfile.DOCLING_TEXT_LOCAL_V2,
                    checkpoint_key=checkpoint_key,
                )

            self.assertEqual(
                raised.exception.diagnostic["limit_name"],
                "pdf_total_timeout",
            )
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(
                all(
                    segment["status"] == "completed"
                    for segment in saved["segments"]
                )
            )

    async def test_markdown_bundle_uses_its_distinct_source_limit(self) -> None:
        limits = ParserLimits(
            max_file_size=32,
            max_markdown_bundle_size=64,
        )
        _validate_source(
            ParserSource(
                "guide.mdz",
                MARKDOWN_BUNDLE_MEDIA_TYPE,
                b"x" * 64,
            ),
            limits,
            ParsingPreset.MULTIMODAL_LOCAL_V2,
        )
        with self.assertRaises(ParserExecutionError) as raised:
            _validate_source(
                ParserSource(
                    "guide.mdz",
                    MARKDOWN_BUNDLE_MEDIA_TYPE,
                    b"x" * 65,
                ),
                limits,
                ParsingPreset.MULTIMODAL_LOCAL_V2,
            )
        self.assertEqual(
            raised.exception.diagnostic["limit_name"],
            "max_markdown_bundle_size",
        )

    async def test_uses_memory_stream_and_converts_each_source_once(self) -> None:
        converter = _FakeConverter()
        harness = _ParserHarness(self, converter)
        self.addCleanup(harness.close)
        source = ParserSource("guide.md", "text/markdown", b"# Guide")

        document = await harness.parser.parse(
            source,
            preset=ParsingPreset.TEXT_LOCAL_V1,
        )

        self.assertIs(document, converter.result.document)
        self.assertEqual(harness.calls, [ParsingPreset.TEXT_LOCAL_V1])
        self.assertEqual(len(converter.calls), 1)
        stream, kwargs = converter.calls[0]
        self.assertIsInstance(stream, DocumentStream)
        self.assertEqual(stream.name, "guide.md")
        self.assertEqual(stream.stream.getvalue(), source.content)
        self.assertFalse(kwargs["raises_on_error"])
        self.assertEqual(kwargs["max_num_pages"], 500)
        self.assertEqual(kwargs["max_file_size"], 10_485_760)

    async def test_initializes_both_presets_lazily(self) -> None:
        converter = _FakeConverter()
        harness = _ParserHarness(self, converter)
        self.addCleanup(harness.close)
        source = ParserSource("guide.txt", "text/plain", b"Guide")

        await harness.parser.parse(source, preset=ParsingPreset.TEXT_LOCAL_V1)
        await harness.parser.parse(source, preset=ParsingPreset.TEXT_LOCAL_V1)
        await harness.parser.parse(
            source,
            preset=ParsingPreset.MULTIMODAL_LOCAL_V2,
        )

        self.assertEqual(
            harness.calls,
            [
                ParsingPreset.TEXT_LOCAL_V1,
                ParsingPreset.MULTIMODAL_LOCAL_V2,
            ],
        )
        self.assertEqual(len(converter.calls), 3)

    async def test_rejects_partial_failure_and_invalid_document(self) -> None:
        cases = (
            (
                _result(
                    ConversionStatus.PARTIAL_SUCCESS,
                    document=_document("partial"),
                ),
                ErrorCode.PARSER_OUTPUT_INVALID,
            ),
            (
                _result(
                    ConversionStatus.FAILURE,
                    document=_document("failed"),
                    input_valid=False,
                ),
                ErrorCode.FILE_CONTENT_INVALID,
            ),
            (
                _result(
                    ConversionStatus.FAILURE,
                    document=_document("failed"),
                ),
                ErrorCode.PARSER_OUTPUT_INVALID,
            ),
            (
                _result(
                    ConversionStatus.SUCCESS,
                    document=DoclingDocument(name="empty"),
                ),
                ErrorCode.PARSER_OUTPUT_INVALID,
            ),
        )
        for result, expected in cases:
            with self.subTest(expected=expected, status=result.status):
                harness = _ParserHarness(self, _FakeConverter(result))
                try:
                    with self.assertRaises(ParserExecutionError) as raised:
                        await harness.parser.parse(
                            ParserSource("guide.pdf", "application/pdf", b"%PDF"),
                            preset=ParsingPreset.TEXT_LOCAL_V1,
                        )
                    self.assertEqual(raised.exception.code, expected)
                    self.assertNotIn("failed", str(raised.exception.diagnostic))
                finally:
                    harness.close()

    async def test_uses_structured_failure_category_without_error_text(self) -> None:
        result = _result(
            ConversionStatus.FAILURE,
            document=_document("failed"),
            errors=[
                ErrorItem(
                    component_type=DoclingComponentType.USER_INPUT,
                    module_name="backend",
                    error_message="secret source content",
                    category=FailureCategory.UNKNOWN,
                )
            ],
        )
        harness = _ParserHarness(self, _FakeConverter(result))
        self.addCleanup(harness.close)

        with self.assertRaises(ParserExecutionError) as raised:
            await harness.parser.parse(
                ParserSource("guide.pdf", "application/pdf", b"%PDF"),
                preset=ParsingPreset.TEXT_LOCAL_V1,
            )

        self.assertEqual(raised.exception.code, ErrorCode.FILE_CONTENT_INVALID)
        self.assertEqual(raised.exception.diagnostic["error_count"], 1)
        self.assertNotIn("secret", str(raised.exception.diagnostic))

    async def test_maps_configuration_resource_and_crash_failures(self) -> None:
        for failure, expected in (
            (ImportError("missing"), ErrorCode.PARSER_NOT_CONFIGURED),
            (FileNotFoundError("artifact"), ErrorCode.PARSER_NOT_CONFIGURED),
            (MemoryError(), ErrorCode.PARSER_RESOURCE_LIMIT),
            (
                Image.DecompressionBombWarning("image"),
                ErrorCode.PARSER_RESOURCE_LIMIT,
            ),
            (
                Image.DecompressionBombError("image"),
                ErrorCode.PARSER_RESOURCE_LIMIT,
            ),
            (RuntimeError("boom"), ErrorCode.PARSER_CRASHED),
        ):
            with self.subTest(expected=expected):
                harness = _ParserHarness(
                    self,
                    _FakeConverter(failure=failure),
                )
                try:
                    with self.assertRaises(ParserExecutionError) as raised:
                        await harness.parser.parse(
                            ParserSource("guide.md", "text/markdown", b"body"),
                            preset=ParsingPreset.TEXT_LOCAL_V1,
                        )
                    self.assertEqual(raised.exception.code, expected)
                finally:
                    harness.close()

    async def test_csv_and_ooxml_preflight_reject_before_converter_call(
        self,
    ) -> None:
        cases = (
            (
                ParserLimits(max_csv_cells=3),
                ParserSource("guide.csv", "text/csv", b"a,b\n1,2\n"),
                "max_csv_cells",
            ),
            (
                ParserLimits(max_image_pixels=8),
                ParserSource(
                    "guide.docx",
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document",
                    _docx_with_image(_png_with_dimensions(3, 3)),
                ),
                "max_image_pixels",
            ),
        )
        for limits, source, limit_name in cases:
            with self.subTest(limit_name=limit_name):
                converter = _FakeConverter()
                harness = _ParserHarness(self, converter, limits=limits)
                try:
                    with self.assertRaises(ParserExecutionError) as raised:
                        await harness.parser.parse(
                            source,
                            preset=ParsingPreset.TEXT_LOCAL_V1,
                        )
                    self.assertEqual(
                        raised.exception.code,
                        ErrorCode.PARSER_RESOURCE_LIMIT,
                    )
                    self.assertEqual(
                        raised.exception.diagnostic["limit_name"],
                        limit_name,
                    )
                    self.assertEqual(converter.calls, [])
                finally:
                    harness.close()

    async def test_child_pillow_guard_uses_application_pixel_limit(self) -> None:
        previous = Image.MAX_IMAGE_PIXELS
        try:
            with warnings.catch_warnings():
                _configure_pillow_resource_guard(
                    replace(ParserLimits(), max_image_pixels=123)
                )
                self.assertEqual(Image.MAX_IMAGE_PIXELS, 123)
                with self.assertRaises(Image.DecompressionBombWarning):
                    warnings.warn(
                        "bounded",
                        Image.DecompressionBombWarning,
                        stacklevel=1,
                    )
        finally:
            Image.MAX_IMAGE_PIXELS = previous

    async def test_enforces_source_item_page_character_and_image_limits(self) -> None:
        oversized_source = _ParserHarness(
            self,
            _FakeConverter(),
            limits=replace(ParserLimits(), max_file_size=3),
        )
        try:
            with self.assertRaises(ParserExecutionError) as raised:
                await oversized_source.parser.parse(
                    ParserSource("guide.txt", "text/plain", b"four"),
                    preset=ParsingPreset.TEXT_LOCAL_V1,
                )
            self.assertEqual(raised.exception.code, ErrorCode.PARSER_RESOURCE_LIMIT)
            self.assertEqual(raised.exception.diagnostic["limit_name"], "max_file_size")
        finally:
            oversized_source.close()

        item_harness = _ParserHarness(
            self,
            _FakeConverter(
                _result(
                    ConversionStatus.SUCCESS,
                    document=_document("one", "two"),
                )
            ),
            limits=replace(ParserLimits(), max_docling_items=1),
        )
        try:
            with self.assertRaises(ParserExecutionError) as raised:
                await item_harness.parser.parse(
                    ParserSource("guide.txt", "text/plain", b"body"),
                    preset=ParsingPreset.TEXT_LOCAL_V1,
                )
            self.assertEqual(raised.exception.diagnostic["limit_name"], "max_docling_items")
        finally:
            item_harness.close()

        character_harness = _ParserHarness(
            self,
            _FakeConverter(
                _result(
                    ConversionStatus.SUCCESS,
                    document=_document("four"),
                )
            ),
            limits=replace(ParserLimits(), max_extracted_characters=3),
        )
        try:
            with self.assertRaises(ParserExecutionError) as raised:
                await character_harness.parser.parse(
                    ParserSource("guide.txt", "text/plain", b"body"),
                    preset=ParsingPreset.TEXT_LOCAL_V1,
                )
            self.assertEqual(
                raised.exception.diagnostic["limit_name"],
                "max_extracted_characters",
            )
        finally:
            character_harness.close()

        paged_document = _document("body")
        paged_document.add_page(page_no=1, size=Size(width=10, height=10))
        paged_document.add_page(page_no=2, size=Size(width=10, height=10))
        page_harness = _ParserHarness(
            self,
            _FakeConverter(
                _result(ConversionStatus.SUCCESS, document=paged_document)
            ),
            limits=replace(ParserLimits(), max_num_pages=1),
        )
        try:
            with self.assertRaises(ParserExecutionError) as raised:
                await page_harness.parser.parse(
                    ParserSource("guide.pdf", "application/pdf", b"%PDF"),
                    preset=ParsingPreset.TEXT_LOCAL_V1,
                )
            self.assertEqual(raised.exception.diagnostic["limit_name"], "max_num_pages")
        finally:
            page_harness.close()

        image_document = _document("body")
        image_document.add_page(
            page_no=1,
            size=Size(width=3, height=3),
            image=ImageRef.from_pil(Image.new("RGB", (3, 3)), dpi=72),
        )
        image_harness = _ParserHarness(
            self,
            _FakeConverter(
                _result(ConversionStatus.SUCCESS, document=image_document)
            ),
            limits=replace(ParserLimits(), max_image_pixels=8),
        )
        try:
            with self.assertRaises(ParserExecutionError) as raised:
                await image_harness.parser.parse(
                    ParserSource("guide.pdf", "application/pdf", b"%PDF"),
                    preset=ParsingPreset.MULTIMODAL_LOCAL_V2,
                )
            self.assertEqual(
                raised.exception.diagnostic["limit_name"],
                "max_image_pixels",
            )
        finally:
            image_harness.close()

    async def test_rejects_mime_mismatch_before_conversion(self) -> None:
        converter = _FakeConverter()
        harness = _ParserHarness(self, converter)
        self.addCleanup(harness.close)

        with self.assertRaises(ParserExecutionError) as raised:
            await harness.parser.parse(
                ParserSource("guide.md", "text/plain", b"body"),
                preset=ParsingPreset.TEXT_LOCAL_V1,
            )

        self.assertEqual(raised.exception.code, ErrorCode.FILE_MEDIA_TYPE_MISMATCH)
        self.assertEqual(converter.calls, [])

    async def test_child_process_is_reused_for_serial_conversions(self) -> None:
        harness = _ProcessHarness(child_target=_echo_child, timeout=5.0)
        self.addCleanup(harness.close)
        source = ParserSource("guide.txt", "text/plain", b"body")

        first = await harness.parser.parse(
            source,
            preset=ParsingPreset.TEXT_LOCAL_V1,
        )
        second = await harness.parser.parse(
            source,
            preset=ParsingPreset.TEXT_LOCAL_V1,
        )

        self.assertEqual(first.document.name, second.document.name)

    async def test_scanned_surface_probe_does_not_block_event_loop(self) -> None:
        harness = _ProcessHarness(child_target=_echo_child, timeout=5.0)
        self.addCleanup(harness.close)
        source = ParserSource("guide.pdf", "application/pdf", b"%PDF-1.7")
        loop_thread = threading.get_ident()
        probe_entered = threading.Event()
        probe_release = threading.Event()
        probe_thread: list[int] = []

        def blocked_probe(probe_source):
            self.assertIs(probe_source, source)
            probe_thread.append(threading.get_ident())
            probe_entered.set()
            if not probe_release.wait(timeout=2):
                raise AssertionError("event loop did not release scanned-page probe")
            return frozenset({1})

        async def release_after_probe_starts() -> None:
            while not probe_entered.is_set():
                await asyncio.sleep(0)
            probe_release.set()

        release_task = asyncio.create_task(release_after_probe_starts())
        try:
            with patch.object(
                parser_module,
                "scanned_surfaces",
                side_effect=blocked_probe,
            ):
                parsed = await harness.parser.parse(
                    source,
                    preset=ParsingPreset.MULTIMODAL_LOCAL_V2,
                )
            await release_task
        finally:
            probe_release.set()
            if not release_task.done():
                release_task.cancel()
                await asyncio.gather(release_task, return_exceptions=True)

        self.assertEqual(parsed.page_image_surfaces, frozenset({1}))
        self.assertEqual(len(probe_thread), 1)
        self.assertNotEqual(probe_thread[0], loop_thread)

    async def test_timeout_kills_child_and_next_conversion_uses_clean_process(
        self,
    ) -> None:
        harness = _ProcessHarness(child_target=_hang_once_child, timeout=6.0)
        self.addCleanup(harness.close)
        source = ParserSource("guide.html", "text/html", b"<p>body</p>")

        with self.assertRaises(ParserExecutionError) as raised:
            await harness.parser.parse(
                source,
                preset=ParsingPreset.TEXT_LOCAL_V1,
            )

        self.assertEqual(raised.exception.code, ErrorCode.PARSER_RESOURCE_LIMIT)
        self.assertEqual(
            raised.exception.diagnostic,
            {"limit_name": "document_timeout", "limit": 6.0},
        )
        document = await harness.parser.parse(
            source,
            preset=ParsingPreset.TEXT_LOCAL_V1,
        )
        self.assertEqual(document.document.texts[0].text, "guide.html")

    async def test_cancellation_kills_child_and_next_conversion_recovers(self) -> None:
        harness = _ProcessHarness(child_target=_hang_once_child, timeout=10.0)
        self.addCleanup(harness.close)
        source = ParserSource("guide.csv", "text/csv", b"a,b")
        first = asyncio.create_task(
            harness.parser.parse(source, preset=ParsingPreset.TEXT_LOCAL_V1)
        )
        for _ in range(500):
            if (harness.root / "first-child-hung").exists():
                break
            await asyncio.sleep(0.01)
        self.assertTrue((harness.root / "first-child-hung").exists())

        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        document = await harness.parser.parse(
            source,
            preset=ParsingPreset.TEXT_LOCAL_V1,
        )
        self.assertEqual(document.document.texts[0].text, "guide.csv")

    async def test_abnormal_child_exit_is_redacted_and_recoverable(self) -> None:
        harness = _ProcessHarness(child_target=_crash_child, timeout=5.0)
        self.addCleanup(harness.close)

        with self.assertRaises(ParserExecutionError) as raised:
            await harness.parser.parse(
                ParserSource(
                    "guide.docx",
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document",
                    b"secret-source",
                ),
                preset=ParsingPreset.TEXT_LOCAL_V1,
            )

        self.assertEqual(raised.exception.code, ErrorCode.PARSER_CRASHED)
        self.assertEqual(
            raised.exception.diagnostic,
            {"check": "docling_child_ipc"},
        )
        self.assertNotIn("secret-source", str(raised.exception.diagnostic))

    async def test_sigkill_child_is_a_non_retryable_memory_resource_failure(
        self,
    ) -> None:
        harness = _ProcessHarness(child_target=_oom_killed_child, timeout=5.0)
        self.addCleanup(harness.close)

        with self.assertRaises(ParserExecutionError) as raised:
            await harness.parser.parse(
                ParserSource("guide.csv", "text/csv", b"a,b\n"),
                preset=ParsingPreset.TEXT_LOCAL_V1,
            )

        self.assertEqual(
            raised.exception.code,
            ErrorCode.PARSER_RESOURCE_LIMIT,
        )
        self.assertEqual(
            raised.exception.diagnostic,
            {"limit_name": "process_memory"},
        )

    async def test_close_terminates_and_reaps_active_conversion_child(self) -> None:
        harness = _ProcessHarness(child_target=_hang_once_child, timeout=10.0)
        self.addCleanup(harness.close)
        execution = asyncio.create_task(
            harness.parser.parse(
                ParserSource("guide.txt", "text/plain", b"body"),
                preset=ParsingPreset.TEXT_LOCAL_V1,
            )
        )
        marker = harness.root / "first-child-hung"
        for _ in range(500):
            if marker.exists():
                break
            await asyncio.sleep(0.01)
        self.assertTrue(marker.exists())
        child_pid = int(marker.read_text(encoding="utf-8"))

        await asyncio.to_thread(harness.parser.close)

        with self.assertRaises(ProcessLookupError):
            os.kill(child_pid, 0)
        with self.assertRaises(ParserExecutionError):
            await execution


if __name__ == "__main__":
    unittest.main()
