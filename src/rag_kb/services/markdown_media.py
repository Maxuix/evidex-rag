"""Admission-time normalization for self-contained Markdown media bundles."""

from __future__ import annotations

import asyncio
import base64
import binascii
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import posixpath
import re
from urllib.parse import unquote, urlsplit
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import marko
from marko.block import HTMLBlock
from marko.inline import Image, InlineHTML
from marko.md_renderer import MarkdownRenderer
from PIL import Image as PillowImage
from PIL import UnidentifiedImageError

from rag_kb.adapters.markdown_media import RemoteImageFetcher
from rag_kb.domain import ErrorCode, FileAdmissionError
from rag_kb.document_processing.markdown_bundle import (
    MARKDOWN_BUNDLE_ENTRYPOINT,
    MARKDOWN_BUNDLE_MANIFEST,
    MARKDOWN_BUNDLE_MEDIA_TYPE,
    MARKDOWN_BUNDLE_VERSION,
    MarkdownBundle,
    read_markdown_bundle,
    safe_relative_path,
)


_MEDIA_DIRECTORY = ".rag-media"
_ALLOWED_IMAGE_FORMATS = {
    "PNG": ("image/png", ".png"),
    "JPEG": ("image/jpeg", ".jpg"),
    "WEBP": ("image/webp", ".webp"),
}
_MAX_REFERENCES = 64
_MAX_REFERENCE_LENGTH = 4096
_MAX_IMAGE_BYTES = 8 * 1024 * 1024
_MAX_TOTAL_IMAGE_BYTES = 9 * 1024 * 1024
_MAX_BUNDLE_BYTES = 10 * 1024 * 1024
_MAX_IMAGE_PIXELS = 40_000_000
_MAX_IMAGE_WIDTH = 16_384
_MAX_IMAGE_HEIGHT = 16_384
_DATA_PREFIX = "data:"
_HTML_IMAGE = re.compile(r"<\s*img\b", re.IGNORECASE)
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class _NormalizedImage:
    content: bytes
    media_type: str
    extension: str
    checksum_sha256: str


class MarkdownMediaNormalizer:
    """Resolve every Markdown image and emit one deterministic local-only ZIP."""

    def __init__(
        self,
        fetcher: RemoteImageFetcher,
    ) -> None:
        self._fetcher = fetcher

    async def normalize(
        self,
        content: bytes,
        *,
        original_filename: str,
        media_type: str,
    ) -> bytes:
        return await asyncio.to_thread(
            self._normalize,
            content,
            original_filename=original_filename,
            media_type=media_type,
        )

    def _normalize(
        self,
        content: bytes,
        *,
        original_filename: str,
        media_type: str,
    ) -> bytes:
        del original_filename
        if media_type == "text/markdown":
            bundle_input = _raw_markdown_input(content)
        elif media_type == MARKDOWN_BUNDLE_MEDIA_TYPE:
            bundle_input = read_markdown_bundle(content)
        else:
            raise FileAdmissionError(ErrorCode.FILE_MEDIA_TYPE_UNSUPPORTED)
        parser = marko.Markdown(renderer=MarkdownRenderer)
        document = parser.parse(bundle_input.markdown)
        if _has_html_image(document):
            raise FileAdmissionError(ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED)
        images = tuple(_walk_images(document))
        if len(images) > _MAX_REFERENCES:
            raise FileAdmissionError(
                ErrorCode.FILE_ARCHIVE_LIMIT_EXCEEDED,
                limit=_MAX_REFERENCES,
                observed=len(images),
            )

        normalized_files: dict[str, bytes] = {}
        manifest_media: list[dict[str, str]] = []
        total_image_bytes = 0
        for image in images:
            reference = image.dest.strip()
            if not reference or len(reference) > _MAX_REFERENCE_LENGTH:
                raise FileAdmissionError(ErrorCode.MARKDOWN_MEDIA_UNRESOLVED)
            content_bytes, source_kind, provenance = self._resolve(
                reference,
                bundle_input,
            )
            normalized = _validate_image(content_bytes)
            total_image_bytes += len(normalized.content)
            if total_image_bytes > _MAX_TOTAL_IMAGE_BYTES:
                raise FileAdmissionError(
                    ErrorCode.FILE_TOO_LARGE,
                    limit=_MAX_TOTAL_IMAGE_BYTES,
                    observed=total_image_bytes,
                )
            path = (
                f"{_MEDIA_DIRECTORY}/{normalized.checksum_sha256}"
                f"{normalized.extension}"
            )
            normalized_files.setdefault(path, normalized.content)
            image.dest = path
            manifest_media.append(
                {
                    "path": path,
                    "media_type": normalized.media_type,
                    "sha256": normalized.checksum_sha256,
                    "source": source_kind,
                    "reference": provenance,
                }
            )

        rendered = parser.render(document)
        if not rendered.endswith("\n"):
            rendered += "\n"
        manifest = {
            "version": MARKDOWN_BUNDLE_VERSION,
            "entrypoint": MARKDOWN_BUNDLE_ENTRYPOINT,
            "media": manifest_media,
        }
        entries = {
            MARKDOWN_BUNDLE_MANIFEST: _canonical_json(manifest),
            MARKDOWN_BUNDLE_ENTRYPOINT: rendered.encode("utf-8"),
            **normalized_files,
        }
        result = _write_deterministic_zip(entries)
        if len(result) > _MAX_BUNDLE_BYTES:
            raise FileAdmissionError(
                ErrorCode.FILE_TOO_LARGE,
                limit=_MAX_BUNDLE_BYTES,
                observed=len(result),
            )
        return result

    def _resolve(
        self,
        reference: str,
        bundle_input: MarkdownBundle,
    ) -> tuple[bytes, str, str]:
        if reference.startswith(_DATA_PREFIX):
            return _decode_data_uri(reference), "data", "data-uri"
        parsed = urlsplit(reference)
        if parsed.scheme in {"http", "https"}:
            fetched = self._fetcher.fetch(reference, max_bytes=_MAX_IMAGE_BYTES)
            return fetched.content, "remote", fetched.final_url
        if parsed.scheme or parsed.netloc:
            raise FileAdmissionError(ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED)
        if not bundle_input.files:
            raise FileAdmissionError(ErrorCode.MARKDOWN_MEDIA_UNRESOLVED)
        local_path = _resolve_local_path(
            bundle_input.entrypoint,
            unquote(parsed.path),
        )
        content = bundle_input.files.get(local_path)
        if content is None:
            raise FileAdmissionError(ErrorCode.MARKDOWN_MEDIA_UNRESOLVED)
        return content, "local", local_path


def _raw_markdown_input(content: bytes) -> MarkdownBundle:
    try:
        markdown = content.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise FileAdmissionError(ErrorCode.FILE_INVALID_UTF8) from error
    return MarkdownBundle(
        markdown=markdown,
        entrypoint=MARKDOWN_BUNDLE_ENTRYPOINT,
        files={},
    )


def _walk_images(element: object):
    if isinstance(element, Image):
        yield element
    children = getattr(element, "children", None)
    if isinstance(children, list):
        for child in children:
            yield from _walk_images(child)


def _has_html_image(element: object) -> bool:
    if isinstance(element, HTMLBlock):
        return bool(_HTML_IMAGE.search(element.body))
    if isinstance(element, InlineHTML):
        return bool(_HTML_IMAGE.search(element.children))
    children = getattr(element, "children", None)
    return isinstance(children, list) and any(
        _has_html_image(child) for child in children
    )


def _validate_image(content: bytes) -> _NormalizedImage:
    if not content or len(content) > _MAX_IMAGE_BYTES:
        raise FileAdmissionError(ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED)
    try:
        with PillowImage.open(BytesIO(content)) as image:
            image_format = image.format
            if image_format not in _ALLOWED_IMAGE_FORMATS:
                raise FileAdmissionError(ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED)
            if getattr(image, "n_frames", 1) != 1 or getattr(
                image, "is_animated", False
            ):
                raise FileAdmissionError(ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED)
            width, height = image.size
            if (
                width <= 0
                or height <= 0
                or width > _MAX_IMAGE_WIDTH
                or height > _MAX_IMAGE_HEIGHT
                or width * height > _MAX_IMAGE_PIXELS
            ):
                raise FileAdmissionError(ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED)
            image.load()
    except FileAdmissionError:
        raise
    except (
        PillowImage.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise FileAdmissionError(
            ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED
        ) from error
    media_type, extension = _ALLOWED_IMAGE_FORMATS[image_format]
    return _NormalizedImage(
        content=content,
        media_type=media_type,
        extension=extension,
        checksum_sha256=hashlib.sha256(content).hexdigest(),
    )


def _decode_data_uri(value: str) -> bytes:
    header, separator, payload = value.partition(",")
    if (
        not separator
        or not header.lower().endswith(";base64")
        or header[5:].split(";", 1)[0].lower()
        not in {"image/png", "image/jpeg", "image/webp"}
    ):
        raise FileAdmissionError(ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED)
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise FileAdmissionError(
            ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED
        ) from error
    if len(decoded) > _MAX_IMAGE_BYTES:
        raise FileAdmissionError(ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED)
    return decoded


def _resolve_local_path(entrypoint: str, reference: str) -> str:
    if not reference or reference.startswith("/"):
        raise FileAdmissionError(ErrorCode.MARKDOWN_MEDIA_UNRESOLVED)
    combined = posixpath.normpath(
        posixpath.join(posixpath.dirname(entrypoint), reference)
    )
    return safe_relative_path(combined)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_deterministic_zip(entries: dict[str, bytes]) -> bytes:
    target = BytesIO()
    with ZipFile(target, mode="w", compression=ZIP_STORED) as archive:
        for name in sorted(entries):
            info = ZipInfo(filename=name, date_time=_FIXED_ZIP_TIME)
            info.compress_type = ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100600 << 16
            archive.writestr(info, entries[name])
    return target.getvalue()
