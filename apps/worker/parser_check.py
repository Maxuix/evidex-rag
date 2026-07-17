"""One-shot Linux parser-isolation self-check used by delivery validation."""

from __future__ import annotations

import asyncio
import io
import sys
from zipfile import ZIP_DEFLATED, ZipFile

from rag_kb.adapters import IsolatedUnstructuredProcessor
from rag_kb.services import (
    ErrorCode,
    ParserExecutionError,
    ParserLimits,
    ParserSource,
)


def _memory_probe(connection, source, limits) -> None:
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
                    "diagnostic": {"limit_name": "memory_bytes"},
                },
            )
        )
    finally:
        connection.close()


async def check_parser() -> None:
    processor = IsolatedUnstructuredProcessor(ParserLimits())
    sources = (
        (
            ParserSource(
                "isolation-check.txt",
                "text/plain",
                b"isolated parser ready",
            ),
            "isolated parser ready",
        ),
        (
            ParserSource(
                "isolation-check.md",
                "text/markdown",
                b"# Overview\n\nMarkdown parser ready",
            ),
            "Markdown parser ready",
        ),
        (
            ParserSource(
                "isolation-check.pdf",
                "application/pdf",
                _minimal_pdf(),
            ),
            "PDF parser ready",
        ),
        (
            ParserSource(
                "isolation-check.docx",
                (
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
                _minimal_docx(),
            ),
            "DOCX parser ready",
        ),
    )
    for source, expected in sources:
        result = await processor.process(source)
        if (
            not result.chunks
            or [chunk.ordinal for chunk in result.chunks]
            != list(range(len(result.chunks)))
            or expected not in " ".join(chunk.text for chunk in result.chunks)
        ):
            raise RuntimeError(
                "isolated parser self-check returned an invalid result"
            )
    if sys.platform.startswith("linux"):
        memory_probe = IsolatedUnstructuredProcessor(
            ParserLimits(
                max_chunks=1,
                wall_seconds=5,
                cpu_seconds=2,
                memory_bytes=128 * 1024 * 1024,
            ),
            child_target=_memory_probe,
        )
        try:
            await memory_probe.process(
                ParserSource("memory-check.txt", "text/plain", b"memory")
            )
        except ParserExecutionError as error:
            if error.code is ErrorCode.PARSER_RESOURCE_LIMIT:
                return
            raise
        raise RuntimeError("parser memory limit was not enforced")


def _minimal_pdf() -> bytes:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

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
    stream.set_data(b"BT /F1 18 Tf 72 720 Td (PDF parser ready) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(target)
    return target.getvalue()


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
                "<w:body><w:p><w:r><w:t>DOCX parser ready</w:t></w:r></w:p>"
                "</w:body></w:document>"
            ),
        )
    return target.getvalue()


def main() -> int:
    asyncio.run(check_parser())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
