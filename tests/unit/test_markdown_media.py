from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZIP_STORED, ZipFile

from PIL import Image

from rag_kb.adapters import FetchedImage, PublicHttpImageFetcher
from rag_kb.adapters.parser.docling import DoclingParser
from rag_kb.domain import ErrorCode, FileAdmissionError
from rag_kb.domain import ParserLimits, ParserSource, ParsingPreset
from rag_kb.document_processing.markdown_bundle import (
    MARKDOWN_BUNDLE_MEDIA_TYPE,
    read_normalized_markdown_bundle,
)
from rag_kb.document_processing.docling import extract_docling_assets
from rag_kb.services.markdown_media import MarkdownMediaNormalizer


def _image_bytes(image_format: str = "PNG") -> bytes:
    target = BytesIO()
    Image.new("RGB", (96, 80), "navy").save(target, format=image_format)
    return target.getvalue()


def _bundle(
    markdown: str,
    *,
    files: dict[str, bytes] | None = None,
    entrypoint: str = "docs/guide.md",
) -> bytes:
    target = BytesIO()
    with ZipFile(target, "w", ZIP_STORED) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps({"version": 1, "entrypoint": entrypoint}),
        )
        archive.writestr(entrypoint, markdown)
        for name, content in (files or {}).items():
            archive.writestr(name, content)
    return target.getvalue()


class _Fetcher:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.calls: list[str] = []

    def fetch(self, url: str, *, max_bytes: int) -> FetchedImage:
        self.calls.append(url)
        if len(self.content) > max_bytes:
            raise AssertionError("fixture exceeded fetch budget")
        return FetchedImage(content=self.content, final_url=url)


class MarkdownMediaNormalizerTests(unittest.IsolatedAsyncioTestCase):
    async def test_snapshots_remote_image_and_is_deterministic(self) -> None:
        fetcher = _Fetcher(_image_bytes())
        normalizer = MarkdownMediaNormalizer(fetcher)
        source = b"# Evidence\n\n![chart](https://example.com/chart.png)\n"

        first = await normalizer.normalize(
            source,
            original_filename="evidence.md",
            media_type="text/markdown",
        )
        second = await normalizer.normalize(
            source,
            original_filename="evidence.md",
            media_type="text/markdown",
        )

        self.assertEqual(first, second)
        self.assertEqual(
            fetcher.calls,
            [
                "https://example.com/chart.png",
                "https://example.com/chart.png",
            ],
        )
        entrypoint, files = read_normalized_markdown_bundle(first)
        markdown = files[entrypoint].decode()
        self.assertIn("![chart](.rag-media/", markdown)
        self.assertEqual(
            len([name for name in files if name.startswith(".rag-media/")]),
            1,
        )

    async def test_resolves_local_bundle_and_data_uri_with_content_dedup(self) -> None:
        image = _image_bytes("JPEG")
        data_uri = (
            "data:image/jpeg;base64,"
            + base64.b64encode(image).decode("ascii")
        )
        source = _bundle(
            f"![local](../images/chart.jpg)\n\n![inline]({data_uri})\n",
            files={"images/chart.jpg": image},
        )

        normalized = await MarkdownMediaNormalizer(_Fetcher(b"unused")).normalize(
            source,
            original_filename="evidence.mdz",
            media_type=MARKDOWN_BUNDLE_MEDIA_TYPE,
        )

        entrypoint, files = read_normalized_markdown_bundle(normalized)
        markdown = files[entrypoint].decode()
        self.assertEqual(markdown.count(".rag-media/"), 2)
        self.assertEqual(
            len([name for name in files if name.startswith(".rag-media/")]),
            1,
        )

    async def test_fails_closed_for_missing_local_and_unsupported_media(self) -> None:
        cases = (
            (
                _bundle("![missing](../images/missing.png)\n"),
                MARKDOWN_BUNDLE_MEDIA_TYPE,
                ErrorCode.MARKDOWN_MEDIA_UNRESOLVED,
            ),
            (
                b"![file](file:///tmp/private.png)\n",
                "text/markdown",
                ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
            ),
            (
                b"![gif](data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==)\n",
                "text/markdown",
                ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
            ),
            (
                b'<img src="https://example.com/chart.png">\n',
                "text/markdown",
                ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
            ),
        )
        for content, media_type, code in cases:
            with self.subTest(code=code), self.assertRaises(
                FileAdmissionError
            ) as raised:
                await MarkdownMediaNormalizer(_Fetcher(b"unused")).normalize(
                    content,
                    original_filename=(
                        "evidence.mdz"
                        if media_type == MARKDOWN_BUNDLE_MEDIA_TYPE
                        else "evidence.md"
                    ),
                    media_type=media_type,
                )
            self.assertEqual(raised.exception.code, code)

    async def test_rejects_bundle_path_traversal(self) -> None:
        source = _bundle(
            "![escape](image.png)\n",
            files={"../image.png": _image_bytes()},
        )

        with self.assertRaises(FileAdmissionError) as raised:
            await MarkdownMediaNormalizer(_Fetcher(b"unused")).normalize(
                source,
                original_filename="evidence.mdz",
                media_type=MARKDOWN_BUNDLE_MEDIA_TYPE,
            )

        self.assertEqual(
            raised.exception.code,
            ErrorCode.MARKDOWN_BUNDLE_INVALID,
        )

    async def test_img_example_inside_code_is_not_treated_as_media(self) -> None:
        normalized = await MarkdownMediaNormalizer(_Fetcher(b"unused")).normalize(
            b'`<img src="https://example.com/chart.png">`\n\n'
            b"```html\n<img src=\"chart.png\">\n```\n",
            original_filename="example.md",
            media_type="text/markdown",
        )

        entrypoint, files = read_normalized_markdown_bundle(normalized)
        self.assertIn("<img src=", files[entrypoint].decode())

    def test_remote_fetcher_rejects_private_network_targets(self) -> None:
        with self.assertRaises(FileAdmissionError) as raised:
            PublicHttpImageFetcher().fetch(
                "http://127.0.0.1/image.png",
                max_bytes=1024,
            )
        self.assertEqual(
            raised.exception.code,
            ErrorCode.MARKDOWN_MEDIA_FETCH_FAILED,
        )

    async def test_docling_v2_loads_bundle_image_as_pixels(self) -> None:
        image = _image_bytes()
        normalized = await MarkdownMediaNormalizer(_Fetcher(image)).normalize(
            b"# Evidence\n\n![chart](https://example.com/chart.png)\n",
            original_filename="evidence.md",
            media_type="text/markdown",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.bin"
            artifact.write_bytes(b"model")
            manifest = root / "manifest.json"
            manifest.write_text(
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
                                "size": artifact.stat().st_size,
                                "sha256": hashlib.sha256(
                                    artifact.read_bytes()
                                ).hexdigest(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            parser = DoclingParser(
                ParserLimits(),
                artifacts_path=root,
                artifact_manifest_path=manifest,
            )
            try:
                document = await parser.parse(
                    ParserSource(
                        "evidence.md",
                        MARKDOWN_BUNDLE_MEDIA_TYPE,
                        normalized,
                    ),
                    preset=ParsingPreset.MULTIMODAL_LOCAL_V2,
                )
            finally:
                parser.close()

        self.assertEqual(len(document.pictures), 1)
        self.assertIsNotNone(document.pictures[0].image)
        self.assertEqual(document.pictures[0].image.pil_image.size, (96, 80))
        assets = extract_docling_assets(document, ParserLimits())
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].media_type, "image/png")
        self.assertEqual((assets[0].width, assets[0].height), (96, 80))


if __name__ == "__main__":
    unittest.main()
