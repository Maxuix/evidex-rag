from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
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


def _animated_gif_bytes() -> bytes:
    target = BytesIO()
    frames = [
        Image.new("RGB", (96, 80), color)
        for color in ("navy", "gold")
    ]
    frames[0].save(
        target,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
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

    async def test_normalizes_html_images_static_gif_and_protocol_relative_url(
        self,
    ) -> None:
        gif = _image_bytes("GIF")
        data_uri = (
            "data:image/gif;base64,"
            + base64.b64encode(gif).decode("ascii")
        )
        source = (
            'Before <img src="//example.com/inline.gif" alt="inline" '
            'width="80" height="80"> after\n\n'
            "<figure>\n"
            f'<img src="{data_uri}" alt="block">\n'
            "<figcaption>Important chart</figcaption>\n"
            "</figure>\n"
        ).encode()
        fetcher = _Fetcher(gif)

        normalized = await MarkdownMediaNormalizer(fetcher).normalize(
            source,
            original_filename="evidence.md",
            media_type="text/markdown",
        )

        self.assertEqual(fetcher.calls, ["https://example.com/inline.gif"])
        entrypoint, files = read_normalized_markdown_bundle(normalized)
        markdown = files[entrypoint].decode()
        self.assertNotIn("<img", markdown)
        self.assertIn("Before ![inline](.rag-media/", markdown)
        self.assertIn("![block](.rag-media/", markdown)
        self.assertIn("Important chart", markdown)
        media = [
            (name, content)
            for name, content in files.items()
            if name.startswith(".rag-media/")
        ]
        self.assertEqual(len(media), 1)
        self.assertTrue(media[0][0].endswith(".png"))
        with Image.open(BytesIO(media[0][1])) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.size, (96, 80))

    async def test_ignores_small_decorative_avatar_in_safe_phrasing_content(
        self,
    ) -> None:
        source = b"""<div align="center">
  <img src="https://example.com/avatar.png" width="80" height="80"
       style="border-radius: 50%;" />
  <br />
  <strong>Author:</strong>
  <a href="https://example.com">Example</a>
  <br />
  <em>Original tutorial | Complete guide</em>
</div>
"""
        fetcher = _Fetcher(_image_bytes())

        normalized = await MarkdownMediaNormalizer(fetcher).normalize(
            source,
            original_filename="guide.md",
            media_type="text/markdown",
        )

        self.assertEqual(fetcher.calls, [])
        entrypoint, files = read_normalized_markdown_bundle(normalized)
        markdown = files[entrypoint].decode()
        self.assertNotIn("![", markdown)
        self.assertNotIn(".rag-media/", markdown)
        self.assertIn("Author:", markdown)
        self.assertIn("Example", markdown)
        self.assertIn("Original tutorial | Complete guide", markdown)
        self.assertNotIn("<strong>", markdown)
        self.assertNotIn("<em>", markdown)

    async def test_converts_supported_static_raster_formats_to_png(self) -> None:
        cases = (
            ("GIF", "gif"),
            ("BMP", "bmp"),
            ("TIFF", "tiff"),
            ("AVIF", "avif"),
        )
        for image_format, extension in cases:
            with self.subTest(image_format=image_format):
                source_image = _image_bytes(image_format)
                source = _bundle(
                    f"![chart](../images/chart.{extension})\n",
                    files={f"images/chart.{extension}": source_image},
                )

                normalized = await MarkdownMediaNormalizer(
                    _Fetcher(b"unused")
                ).normalize(
                    source,
                    original_filename="evidence.mdz",
                    media_type=MARKDOWN_BUNDLE_MEDIA_TYPE,
                )

                entrypoint, files = read_normalized_markdown_bundle(normalized)
                markdown = files[entrypoint].decode()
                self.assertIn("![chart](.rag-media/", markdown)
                media = [
                    (name, content)
                    for name, content in files.items()
                    if name.startswith(".rag-media/")
                ]
                self.assertEqual(len(media), 1)
                self.assertTrue(media[0][0].endswith(".png"))
                with Image.open(BytesIO(media[0][1])) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertEqual(image.size, (96, 80))

    async def test_large_data_uri_is_not_limited_as_a_url(self) -> None:
        bmp = _image_bytes("BMP")
        data_uri = (
            "data:image/bmp;base64,"
            + base64.b64encode(bmp).decode("ascii")
        )
        self.assertGreater(len(data_uri), 4096)

        normalized = await MarkdownMediaNormalizer(
            _Fetcher(b"unused")
        ).normalize(
            f"![chart]({data_uri})\n".encode(),
            original_filename="evidence.md",
            media_type="text/markdown",
        )

        entrypoint, files = read_normalized_markdown_bundle(normalized)
        self.assertIn("![chart](.rag-media/", files[entrypoint].decode())

    async def test_repeated_references_are_cached_and_processing_is_bounded(
        self,
    ) -> None:
        repeated = "\n".join(
            "![chart](https://example.com/chart.png)"
            for _ in range(64)
        ).encode()
        repeated_fetcher = _Fetcher(_image_bytes())

        await MarkdownMediaNormalizer(repeated_fetcher).normalize(
            repeated,
            original_filename="evidence.md",
            media_type="text/markdown",
        )

        self.assertEqual(
            repeated_fetcher.calls,
            ["https://example.com/chart.png"],
        )

        pixel_fetcher = _Fetcher(_image_bytes())
        with patch(
            "rag_kb.services.markdown_media._MAX_TOTAL_IMAGE_PIXELS",
            10_000,
        ), self.assertRaises(FileAdmissionError) as pixel_error:
            await MarkdownMediaNormalizer(pixel_fetcher).normalize(
                b"![one](https://example.com/same.png)\n"
                b"![two](https://example.com/same.png)\n",
                original_filename="evidence.md",
                media_type="text/markdown",
            )
        self.assertEqual(
            pixel_error.exception.code,
            ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
        )
        self.assertEqual(
            pixel_error.exception.check,
            "image_total_pixels",
        )
        self.assertEqual(
            pixel_fetcher.calls,
            ["https://example.com/same.png"],
        )

        large_bmp = BytesIO()
        Image.new("RGB", (1024, 1024), "navy").save(
            large_bmp,
            format="BMP",
        )
        distinct_fetcher = _Fetcher(large_bmp.getvalue())
        distinct = "\n".join(
            f"![chart](https://example.com/chart-{index}.bmp)"
            for index in range(8)
        ).encode()

        with self.assertRaises(FileAdmissionError) as raised:
            await MarkdownMediaNormalizer(distinct_fetcher).normalize(
                distinct,
                original_filename="evidence.md",
                media_type="text/markdown",
            )

        self.assertEqual(raised.exception.code, ErrorCode.FILE_TOO_LARGE)
        self.assertEqual(len(distinct_fetcher.calls), 7)

    async def test_rejects_complex_html_blocks_and_attribute_injection(
        self,
    ) -> None:
        fetcher = _Fetcher(_image_bytes())
        cases = (
            (
                b"<table><tr><td><img "
                b'src="https://example.com/chart.png"></td></tr></table>\n'
            ),
            (
                b'<img src="https://example.com/a.png&#10;'
                b'![evil](https://example.com/b.png)">\n'
            ),
        )
        for source in cases:
            with self.subTest(source=source), self.assertRaises(
                FileAdmissionError
            ) as raised:
                await MarkdownMediaNormalizer(fetcher).normalize(
                    source,
                    original_filename="evidence.md",
                    media_type="text/markdown",
                )
            self.assertEqual(
                raised.exception.code,
                ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
            )
            self.assertIn(
                raised.exception.check,
                {"html_image_attribute", "html_image_structure"},
            )
        self.assertEqual(fetcher.calls, [])

    async def test_fails_closed_for_missing_and_unsafe_media(self) -> None:
        animated_gif = base64.b64encode(_animated_gif_bytes()).decode("ascii")
        cases = (
            (
                _bundle("![missing](../images/missing.png)\n"),
                MARKDOWN_BUNDLE_MEDIA_TYPE,
                ErrorCode.MARKDOWN_MEDIA_UNRESOLVED,
                None,
            ),
            (
                b"![file](file:///tmp/private.png)\n",
                "text/markdown",
                ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
                "reference_scheme",
            ),
            (
                (
                    f"![animated](data:image/gif;base64,{animated_gif})\n"
                ).encode(),
                "text/markdown",
                ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
                "image_animated",
            ),
            (
                b"![svg](data:image/svg+xml;base64,PHN2Zy8+)\n",
                "text/markdown",
                ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
                "data_uri",
            ),
            (
                b'<img alt="missing source">\n',
                "text/markdown",
                ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
                "html_image_missing_src",
            ),
            (
                b"![broken](data:image/png;base64,bm90LWltYWdl)\n",
                "text/markdown",
                ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
                "image_decode",
            ),
        )
        for content, media_type, code, check in cases:
            with self.subTest(code=code, check=check), self.assertRaises(
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
            self.assertEqual(raised.exception.check, check)

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
            b'# Evidence\n\n<img src="https://example.com/chart.png" '
            b'alt="chart">\n',
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
