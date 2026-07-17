from __future__ import annotations

import io
import os
import signal
import sys
import time
import unittest
from zipfile import ZIP_DEFLATED, ZipFile

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from rag_kb.adapters import DocumentProcessor, IsolatedUnstructuredProcessor
from rag_kb.adapters.parser.langchain_unstructured import process_with_unstructured
from rag_kb.domain import (
    AdmissionLimits,
    ErrorCode,
    FileAdmissionError,
    ParserExecutionError,
    ParserLimits,
    ParserSource,
)
from rag_kb.services import FileAdmissionService


def _hang_child(connection, source, limits) -> None:
    del connection, source, limits
    time.sleep(60)


def _crash_child(connection, source, limits) -> None:
    del connection, source, limits
    os._exit(7)


def _resource_child(connection, source, limits) -> None:
    del connection, source, limits
    os.kill(os.getpid(), signal.SIGKILL)


def _cpu_hog_child(connection, source, limits) -> None:
    del connection, source
    import resource

    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, hard))
    while True:
        pass


def _memory_hog_child(connection, source, limits) -> None:
    del source
    import resource

    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (limits.memory_bytes, hard))
    try:
        bytearray(limits.memory_bytes * 4)
    except MemoryError:
        connection.send(
            (
                "error",
                {
                    "code": ErrorCode.PARSER_RESOURCE_LIMIT.value,
                    "diagnostic": {
                        "limit_name": "memory_bytes",
                        "limit": limits.memory_bytes,
                    },
                },
            )
        )
    finally:
        connection.close()


def _minimal_docx() -> bytes:
    target = io.BytesIO()
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/word/document.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                "</Types>"
            ),
        )
        archive.writestr(
            "_rels/.rels",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                'Target="word/document.xml"/>'
                "</Relationships>"
            ),
        )
        archive.writestr(
            "word/document.xml",
            (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body>"
                '<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr><w:r><w:t>Overview</w:t></w:r></w:p>'
                "<w:p><w:r><w:t>DOCX body evidence.</w:t></w:r></w:p>"
                "</w:body></w:document>"
            ),
        )
    return target.getvalue()


def _minimal_pdf() -> bytes:
    target = io.BytesIO()
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    stream = DecodedStreamObject()
    stream.set_data(
        b"BT /F1 18 Tf 72 720 Td (Overview) Tj "
        b"0 -24 Td (PDF body evidence.) Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(target)
    return target.getvalue()


class FileAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FileAdmissionService(
            AdmissionLimits(
                max_bytes=256 * 1024,
                max_lines=3,
                max_archive_entries=20,
                max_expanded_bytes=128 * 1024,
            )
        )

    def test_all_supported_formats_are_admitted_with_bounded_inspection(self) -> None:
        cases = (
            ("guide.TXT", "text/plain; charset=UTF-8", b"\xef\xbb\xbfhello\r\nworld\n", 2),
            ("guide.md", "text/markdown", "标题\r正文".encode(), 2),
            ("guide.pdf", "application/pdf", _minimal_pdf(), None),
            (
                "guide.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                _minimal_docx(),
                None,
            ),
        )
        for filename, media_type, content, lines in cases:
            with self.subTest(filename=filename):
                source = io.BytesIO(content)
                admitted = self.service.validate(
                    source, original_filename=filename, media_type=media_type
                )
                self.assertEqual(admitted.line_count, lines)
                self.assertEqual(source.tell(), 0)

    def test_unicode_filename_is_normalized_and_control_characters_are_rejected(
        self,
    ) -> None:
        admitted = self.service.validate(
            io.BytesIO(b"ok"),
            original_filename="re\u0301sume\u0301-报告-📄.md",
            media_type="text/markdown",
        )
        self.assertEqual(admitted.original_filename, "résumé-报告-📄.md")

        for filename in ("bad\nname.txt", "bad\u007fname.txt", "bad\ud800name.txt"):
            with self.subTest(filename=repr(filename)), self.assertRaises(
                FileAdmissionError
            ) as raised:
                self.service.validate(
                    io.BytesIO(b"ok"),
                    original_filename=filename,
                    media_type="text/plain",
                )
            self.assertEqual(raised.exception.code, ErrorCode.FILE_NAME_INVALID)

    def test_format_media_content_size_and_line_failures_are_stable(self) -> None:
        oversized = b"x" * (256 * 1024 + 1)
        cases = (
            ("../guide.txt", "text/plain", b"ok", ErrorCode.FILE_NAME_INVALID),
            ("guide.rtf", "application/rtf", b"ok", ErrorCode.PARSER_NOT_CONFIGURED),
            ("guide.pdf", "application/pdf", b"ok", ErrorCode.FILE_CONTENT_INVALID),
            (
                "guide.pdf",
                "application/pdf; charset=utf-8",
                _minimal_pdf(),
                ErrorCode.FILE_MEDIA_TYPE_UNSUPPORTED,
            ),
            (
                "guide.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                b"not-a-zip",
                ErrorCode.FILE_CONTENT_INVALID,
            ),
            ("guide.txt", "application/json", b"ok", ErrorCode.FILE_MEDIA_TYPE_UNSUPPORTED),
            ("guide.txt", "text/markdown", b"ok", ErrorCode.FILE_MEDIA_TYPE_MISMATCH),
            ("guide.txt", "text/plain", oversized, ErrorCode.FILE_TOO_LARGE),
            ("guide.txt", "text/plain", b"\xff", ErrorCode.FILE_INVALID_UTF8),
            ("guide.txt", "text/plain", b"1\n2\n3\n4", ErrorCode.FILE_LINE_LIMIT_EXCEEDED),
        )
        for filename, media_type, content, code in cases:
            with self.subTest(code=code), self.assertRaises(FileAdmissionError) as raised:
                self.service.validate(
                    io.BytesIO(content),
                    original_filename=filename,
                    media_type=media_type,
                )
            self.assertEqual(raised.exception.code, code)
            self.assertNotIn(content[:64].decode("latin-1"), str(raised.exception))

    def test_docx_archive_expansion_limit_is_enforced(self) -> None:
        service = FileAdmissionService(
            AdmissionLimits(
                max_bytes=256 * 1024,
                max_archive_entries=20,
                max_expanded_bytes=32,
            )
        )
        with self.assertRaises(FileAdmissionError) as raised:
            service.validate(
                io.BytesIO(_minimal_docx()),
                original_filename="guide.docx",
                media_type=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
            )
        self.assertEqual(
            raised.exception.code, ErrorCode.FILE_ARCHIVE_LIMIT_EXCEEDED
        )


class UnstructuredParserTests(unittest.TestCase):
    def test_supported_formats_produce_stable_bounded_chunk_contracts(self) -> None:
        cases = (
            ParserSource("guide.txt", "text/plain", b"Overview\n\nText body evidence."),
            ParserSource(
                "guide.md",
                "text/markdown",
                "# 概览\r\ne\u0301vidence\r\n\r\n## 细节\rvalue".encode(),
            ),
            ParserSource("guide.pdf", "application/pdf", _minimal_pdf()),
            ParserSource(
                "guide.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                _minimal_docx(),
            ),
        )
        limits = ParserLimits(wall_seconds=60, cpu_seconds=45)
        for source in cases:
            with self.subTest(filename=source.original_filename):
                first = process_with_unstructured(source, limits)
                second = process_with_unstructured(source, limits)
                self.assertEqual(first, second)
                self.assertEqual(
                    [chunk.ordinal for chunk in first.chunks],
                    list(range(len(first.chunks))),
                )
                self.assertTrue(all(chunk.text for chunk in first.chunks))
                self.assertTrue(
                    all(
                        chunk.processing_metadata["integration"]
                        == "langchain-unstructured"
                        for chunk in first.chunks
                    )
                )
                self.assertEqual(
                    first.extracted_character_count,
                    sum(len(chunk.text) for chunk in first.chunks),
                )

    def test_unicode_normalization_and_output_limits_fail_closed(self) -> None:
        source = ParserSource(
            "guide.md",
            "text/markdown",
            "# 概览\r\ne\u0301vidence".encode(),
        )
        result = process_with_unstructured(source, ParserLimits())
        self.assertNotIn("\r", result.chunks[0].text)
        self.assertIn("évidence", result.chunks[0].text)

        with self.assertRaises(ParserExecutionError) as raised:
            process_with_unstructured(
                source,
                ParserLimits(max_extracted_characters=2),
            )
        self.assertEqual(raised.exception.code, ErrorCode.PARSER_RESOURCE_LIMIT)
        self.assertEqual(
            raised.exception.diagnostic["limit_name"],
            "max_extracted_characters",
        )


class IsolatedParserTests(unittest.IsolatedAsyncioTestCase):
    async def test_supported_input_runs_through_the_isolated_contract(self) -> None:
        processor = IsolatedUnstructuredProcessor(
            ParserLimits(
                max_chunks=20,
                wall_seconds=60,
                cpu_seconds=45,
                memory_bytes=4 * 1024 * 1024 * 1024,
            )
        )
        self.assertIsInstance(processor, DocumentProcessor)
        result = await processor.process(
            ParserSource("guide.txt", "text/plain", b"isolated parser")
        )
        self.assertEqual(result.chunks[0].text, "isolated parser")

    async def test_timeout_crash_and_resource_exit_are_distinct_and_redacted(self) -> None:
        cases = (
            (_hang_child, ErrorCode.PARSER_TIMEOUT, 0.05),
            (_crash_child, ErrorCode.PARSER_CRASHED, 5.0),
            (_resource_child, ErrorCode.PARSER_RESOURCE_LIMIT, 5.0),
        )
        os.environ["RAG_KB_SECRET_TEST"] = "credential-must-not-leak"
        try:
            for target, code, wall in cases:
                processor = IsolatedUnstructuredProcessor(
                    ParserLimits(
                        max_chunks=20,
                        wall_seconds=wall,
                        cpu_seconds=1,
                        memory_bytes=512 * 1024 * 1024,
                    ),
                    child_target=target,
                )
                with self.subTest(code=code), self.assertRaises(
                    ParserExecutionError
                ) as raised:
                    await processor.process(
                        ParserSource(
                            "guide.txt",
                            "text/plain",
                            b"secret-source-must-not-leak",
                        )
                    )
                self.assertEqual(raised.exception.code, code)
                rendered = f"{raised.exception} {raised.exception.diagnostic}"
                self.assertNotIn("credential-must-not-leak", rendered)
                self.assertNotIn("secret-source-must-not-leak", rendered)
        finally:
            del os.environ["RAG_KB_SECRET_TEST"]

    async def test_cpu_budget_is_enforced_by_the_child_kernel_limit(self) -> None:
        processor = IsolatedUnstructuredProcessor(
            ParserLimits(
                max_chunks=20,
                wall_seconds=4,
                cpu_seconds=1,
                memory_bytes=512 * 1024 * 1024,
            ),
            child_target=_cpu_hog_child,
        )
        with self.assertRaises(ParserExecutionError) as raised:
            await processor.process(ParserSource("guide.txt", "text/plain", b"cpu"))
        self.assertEqual(raised.exception.code, ErrorCode.PARSER_RESOURCE_LIMIT)

    @unittest.skipIf(sys.platform == "darwin", "Darwin rejects lowering RLIMIT_AS")
    async def test_memory_budget_is_enforced_by_the_child_kernel_limit(self) -> None:
        processor = IsolatedUnstructuredProcessor(
            ParserLimits(
                max_chunks=20,
                wall_seconds=4,
                cpu_seconds=2,
                memory_bytes=128 * 1024 * 1024,
            ),
            child_target=_memory_hog_child,
        )
        with self.assertRaises(ParserExecutionError) as raised:
            await processor.process(
                ParserSource("guide.txt", "text/plain", b"memory")
            )
        self.assertEqual(raised.exception.code, ErrorCode.PARSER_RESOURCE_LIMIT)
        self.assertEqual(raised.exception.diagnostic["limit_name"], "memory_bytes")


if __name__ == "__main__":
    unittest.main()
