from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    DocumentStream,
    ErrorItem,
    FailureCategory,
    InputFormat,
)
from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    ImageRef,
    Size,
    TableCell,
    TableData,
)
from PIL import Image

from rag_kb.adapters.parser.docling import (
    ArtifactManifestError,
    DoclingParser,
    build_docling_converter,
    verify_docling_artifacts,
)
from rag_kb.domain import (
    ErrorCode,
    ParserExecutionError,
    ParserLimits,
    ParserSource,
    ParsingPreset,
)


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


class _BlockingConverter(_FakeConverter):
    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()
        self.started = threading.Event()
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def convert(self, source: object, **kwargs: object) -> SimpleNamespace:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.started.set()
        self.release.wait(timeout=5)
        with self.lock:
            self.active -= 1
        return super().convert(source, **kwargs)


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
        self.parser = DoclingParser(
            limits or ParserLimits(),
            artifacts_path=root,
            artifact_manifest_path=manifest,
            converter_factory=factory,
        )

    def close(self) -> None:
        self.parser.close()
        self._temporary.cleanup()


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
                ParsingPreset.MULTIMODAL_LOCAL_V1,
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
        for input_format in (InputFormat.MD, InputFormat.DOCX):
            options = text.format_to_options[input_format].pipeline_options
            self.assertEqual(options.document_timeout, 600)
            self.assertFalse(options.enable_remote_services)
            self.assertFalse(options.allow_external_plugins)
            self.assertEqual(options.accelerator_options.device, AcceleratorDevice.CPU)

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
            preset=ParsingPreset.MULTIMODAL_LOCAL_V1,
        )

        self.assertEqual(
            harness.calls,
            [
                ParsingPreset.TEXT_LOCAL_V1,
                ParsingPreset.MULTIMODAL_LOCAL_V1,
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
                    preset=ParsingPreset.MULTIMODAL_LOCAL_V1,
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

    async def test_cancelled_wait_keeps_single_conversion_lane_occupied(self) -> None:
        converter = _BlockingConverter()
        harness = _ParserHarness(self, converter)
        self.addCleanup(harness.close)
        source = ParserSource("guide.txt", "text/plain", b"body")
        first = asyncio.create_task(
            harness.parser.parse(source, preset=ParsingPreset.TEXT_LOCAL_V1)
        )
        started = await asyncio.to_thread(converter.started.wait, 2)
        self.assertTrue(started)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first

        second = asyncio.create_task(
            harness.parser.parse(source, preset=ParsingPreset.TEXT_LOCAL_V1)
        )
        await asyncio.sleep(0.05)
        self.assertFalse(second.done())
        self.assertEqual(converter.max_active, 1)
        converter.release.set()
        await second
        self.assertEqual(converter.max_active, 1)
        self.assertEqual(len(converter.calls), 2)


if __name__ == "__main__":
    unittest.main()
