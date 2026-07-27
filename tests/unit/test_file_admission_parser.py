from __future__ import annotations

import io
import os
import socket
import unittest
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from rag_kb.domain import (
    AdmissionLimits,
    ErrorCode,
    FileAdmissionError,
)
from rag_kb.services import FileAdmissionService
from rag_kb.document_processing.markdown_bundle import (
    MARKDOWN_BUNDLE_MEDIA_TYPE,
)

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

    def test_markdown_bundle_is_admitted_as_a_versioned_archive(self) -> None:
        target = io.BytesIO()
        with ZipFile(target, "w", ZIP_DEFLATED) as archive:
            archive.writestr(
                "manifest.json",
                '{"version":1,"entrypoint":"guide.md"}',
            )
            archive.writestr("guide.md", "# Guide\n")

        admitted = self.service.validate(
            io.BytesIO(target.getvalue()),
            original_filename="guide.mdz",
            media_type=MARKDOWN_BUNDLE_MEDIA_TYPE,
        )

        self.assertEqual(admitted.extension, ".mdz")
        self.assertEqual(admitted.media_type, MARKDOWN_BUNDLE_MEDIA_TYPE)
