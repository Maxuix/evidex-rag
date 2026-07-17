from __future__ import annotations

import asyncio
import io
import os
import signal
import sys
import time
import unittest

from rag_kb.adapters import (
    DocumentProcessor,
    IsolatedPlainTextProcessor,
    PlainTextTestParser,
)
from rag_kb.domain import (
    AdmissionLimits,
    ErrorCode,
    FileAdmissionError,
    ParserExecutionError,
    ParserLimits,
    ParserSource,
)
from rag_kb.indexing import process_plain_text
from rag_kb.services import FileAdmissionService


def _hang_child(connection, source, limits, maximum, overlap) -> None:
    del connection, source, limits, maximum, overlap
    time.sleep(60)


def _crash_child(connection, source, limits, maximum, overlap) -> None:
    del connection, source, limits, maximum, overlap
    os._exit(7)


def _resource_child(connection, source, limits, maximum, overlap) -> None:
    del connection, source, limits, maximum, overlap
    os.kill(os.getpid(), signal.SIGKILL)


def _cpu_hog_child(connection, source, limits, maximum, overlap) -> None:
    del connection, source, maximum, overlap
    import resource

    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, hard))
    while True:
        pass


def _memory_hog_child(connection, source, limits, maximum, overlap) -> None:
    del source, maximum, overlap
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


class FileAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FileAdmissionService(AdmissionLimits(max_bytes=16, max_lines=3))

    def test_txt_and_markdown_admit_strict_utf8_with_optional_bom(self) -> None:
        cases = (
            ("guide.TXT", "text/plain; charset=UTF-8", b"\xef\xbb\xbfhello\r\nworld\n", 2),
            ("guide.md", "text/markdown", "标题\r正文".encode(), 2),
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

    def test_format_media_utf8_size_and_line_failures_are_stable(self) -> None:
        cases = (
            ("../guide.txt", "text/plain", b"ok", ErrorCode.FILE_NAME_INVALID),
            ("guide.pdf", "application/pdf", b"ok", ErrorCode.PARSER_NOT_CONFIGURED),
            ("guide.txt", "application/json", b"ok", ErrorCode.FILE_MEDIA_TYPE_UNSUPPORTED),
            ("guide.txt", "text/markdown", b"ok", ErrorCode.FILE_MEDIA_TYPE_MISMATCH),
            ("guide.txt", "text/plain", b"x" * 17, ErrorCode.FILE_TOO_LARGE),
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
            self.assertNotIn(content.decode("latin-1"), str(raised.exception))


class PlainTextParserTests(unittest.TestCase):
    def test_parser_contract_returns_parser_neutral_blocks(self) -> None:
        parsed = PlainTextTestParser().parse(
            ParserSource("guide.txt", "text/plain", b"first\n\nsecond")
        )
        self.assertEqual([block.text for block in parsed.blocks], ["first", "second"])

    def test_markdown_normalizes_and_produces_stable_heading_chunks(self) -> None:
        result = process_plain_text(
            ParserSource(
                "guide.md",
                "text/markdown",
                "# 概览\r\ne\u0301vidence\r\n\r\n## 细节\rvalue".encode(),
            ),
            max_characters=12,
            overlap_characters=2,
            max_chunks=20,
        )
        self.assertNotIn("\r", result.parsed.canonical_text)
        self.assertIn("évidence", result.parsed.canonical_text)
        self.assertEqual([chunk.ordinal for chunk in result.chunks], list(range(len(result.chunks))))
        self.assertIn(("概览", "细节"), [chunk.heading_hierarchy for chunk in result.chunks])
        self.assertTrue(all(len(chunk.text) <= 12 for chunk in result.chunks))

    def test_chunk_limit_rejects_without_returning_a_partial_result(self) -> None:
        with self.assertRaises(ParserExecutionError) as raised:
            process_plain_text(
                ParserSource("a.txt", "text/plain", b"abcdefghij"),
                max_characters=3,
                overlap_characters=1,
                max_chunks=2,
            )
        self.assertEqual(raised.exception.code, ErrorCode.PARSER_CHUNK_LIMIT_EXCEEDED)
        self.assertEqual(raised.exception.phase, "parsing")


class IsolatedParserTests(unittest.IsolatedAsyncioTestCase):
    async def test_supported_input_runs_through_the_isolated_contract(self) -> None:
        processor = IsolatedPlainTextProcessor(
            ParserLimits(max_chunks=20, wall_seconds=5, cpu_seconds=2, memory_bytes=512 * 1024 * 1024),
            max_characters=20,
            overlap_characters=2,
        )
        self.assertIsInstance(processor, DocumentProcessor)
        result = await processor.process(
            ParserSource("guide.txt", "text/plain", b"isolated parser")
        )
        self.assertEqual(result.chunks[0].text, "isolated parser")

    async def test_timeout_crash_and_resource_exit_are_distinct_and_redacted(self) -> None:
        cases = (
            (_hang_child, ErrorCode.PARSER_TIMEOUT, 0.05),
            (_crash_child, ErrorCode.PARSER_CRASHED, 2.0),
            (_resource_child, ErrorCode.PARSER_RESOURCE_LIMIT, 2.0),
        )
        os.environ["RAG_KB_SECRET_TEST"] = "credential-must-not-leak"
        try:
            for target, code, wall in cases:
                processor = IsolatedPlainTextProcessor(
                    ParserLimits(max_chunks=20, wall_seconds=wall, cpu_seconds=1, memory_bytes=512 * 1024 * 1024),
                    child_target=target,
                )
                with self.subTest(code=code), self.assertRaises(ParserExecutionError) as raised:
                    await processor.process(
                        ParserSource("guide.txt", "text/plain", b"secret-source-must-not-leak")
                    )
                self.assertEqual(raised.exception.code, code)
                rendered = f"{raised.exception} {raised.exception.diagnostic}"
                self.assertNotIn("credential-must-not-leak", rendered)
                self.assertNotIn("secret-source-must-not-leak", rendered)
        finally:
            del os.environ["RAG_KB_SECRET_TEST"]

    async def test_cpu_budget_is_enforced_by_the_child_kernel_limit(self) -> None:
        processor = IsolatedPlainTextProcessor(
            ParserLimits(max_chunks=20, wall_seconds=4, cpu_seconds=1, memory_bytes=512 * 1024 * 1024),
            child_target=_cpu_hog_child,
        )
        with self.assertRaises(ParserExecutionError) as raised:
            await processor.process(ParserSource("guide.txt", "text/plain", b"cpu"))
        self.assertEqual(raised.exception.code, ErrorCode.PARSER_RESOURCE_LIMIT)

    @unittest.skipIf(sys.platform == "darwin", "Darwin rejects lowering RLIMIT_AS")
    async def test_memory_budget_is_enforced_by_the_child_kernel_limit(self) -> None:
        processor = IsolatedPlainTextProcessor(
            ParserLimits(max_chunks=20, wall_seconds=4, cpu_seconds=2, memory_bytes=128 * 1024 * 1024),
            child_target=_memory_hog_child,
        )
        with self.assertRaises(ParserExecutionError) as raised:
            await processor.process(ParserSource("guide.txt", "text/plain", b"memory"))
        self.assertEqual(raised.exception.code, ErrorCode.PARSER_RESOURCE_LIMIT)
        self.assertEqual(raised.exception.diagnostic["limit_name"], "memory_bytes")


if __name__ == "__main__":
    unittest.main()
