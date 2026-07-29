"""Admission-time normalization for self-contained Markdown media bundles."""

from __future__ import annotations

import asyncio
import base64
import binascii
from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
from io import BytesIO
import json
import posixpath
import re
import unicodedata
from urllib.parse import unquote, urlsplit
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import marko
from marko.block import HTMLBlock
from marko.inline import Image, InlineHTML, RawText
from marko.md_renderer import MarkdownRenderer
from PIL import Image as PillowImage, ImageOps
from PIL import UnidentifiedImageError

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
from rag_kb.ports.markdown_media import RemoteImageFetcher


_MEDIA_DIRECTORY = ".rag-media"
_ALLOWED_IMAGE_FORMATS = {
    "PNG": ("image/png", ".png"),
    "JPEG": ("image/jpeg", ".jpg"),
    "WEBP": ("image/webp", ".webp"),
}
_CONVERTIBLE_IMAGE_FORMATS = frozenset({"GIF", "BMP", "TIFF", "AVIF"})
_DATA_IMAGE_MEDIA_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
        "image/bmp",
        "image/tiff",
        "image/avif",
    }
)
_MAX_REFERENCES = 64
_MAX_REFERENCE_LENGTH = 4096
_MAX_IMAGE_BYTES = 8 * 1024 * 1024
_MAX_TOTAL_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_BUNDLE_BYTES = 20 * 1024 * 1024
_MAX_IMAGE_PIXELS = 40_000_000
_MAX_TOTAL_IMAGE_PIXELS = 80_000_000
_MAX_IMAGE_WIDTH = 16_384
_MAX_IMAGE_HEIGHT = 16_384
_MAX_DECORATIVE_HTML_IMAGE_DIMENSION = 128
_MAX_DATA_URI_REFERENCE_LENGTH = ((_MAX_IMAGE_BYTES + 2) // 3) * 4 + 128
_DATA_PREFIX = "data:"
_HTML_IMAGE = re.compile(r"<\s*img\b", re.IGNORECASE)
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_HTML_IMAGE_BLOCK_TAGS = frozenset(
    {
        "a",
        "abbr",
        "b",
        "br",
        "center",
        "cite",
        "code",
        "del",
        "div",
        "em",
        "figcaption",
        "figure",
        "i",
        "ins",
        "kbd",
        "mark",
        "p",
        "picture",
        "q",
        "s",
        "samp",
        "small",
        "source",
        "span",
        "strong",
        "sub",
        "sup",
        "time",
        "u",
        "var",
        "wbr",
    }
)
_HTML_BLOCK_BREAKS = frozenset(
    {"br", "center", "div", "figcaption", "figure", "p"}
)
_HTML_SUPPRESSED_CONTENT = frozenset({"script", "style"})


@dataclass(frozen=True, slots=True)
class _NormalizedImage:
    content: bytes
    media_type: str
    extension: str
    checksum_sha256: str
    width: int
    height: int


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
        _normalize_html_images(document, parser)
        if _has_html_image(document):
            raise FileAdmissionError(
                ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
                check="html_image",
            )
        images = tuple(_walk_images(document))
        if len(images) > _MAX_REFERENCES:
            raise FileAdmissionError(
                ErrorCode.FILE_ARCHIVE_LIMIT_EXCEEDED,
                limit=_MAX_REFERENCES,
                observed=len(images),
            )

        normalized_files: dict[str, bytes] = {}
        manifest_media: list[dict[str, str]] = []
        resolved_references: dict[
            str,
            tuple[_NormalizedImage, str, str],
        ] = {}
        normalized_content: dict[str, _NormalizedImage] = {}
        total_input_bytes = 0
        total_output_bytes = 0
        total_image_pixels = 0
        for image in images:
            reference = image.dest.strip()
            if not reference:
                raise FileAdmissionError(ErrorCode.MARKDOWN_MEDIA_UNRESOLVED)
            if _is_data_reference(reference):
                if len(reference) > _MAX_DATA_URI_REFERENCE_LENGTH:
                    raise FileAdmissionError(
                        ErrorCode.FILE_TOO_LARGE,
                        limit=_MAX_IMAGE_BYTES,
                    )
            elif len(reference) > _MAX_REFERENCE_LENGTH:
                raise FileAdmissionError(ErrorCode.MARKDOWN_MEDIA_UNRESOLVED)
            cached_reference = resolved_references.get(reference)
            if cached_reference is None:
                content_bytes, source_kind, provenance = self._resolve(
                    reference,
                    bundle_input,
                )
                total_input_bytes += len(content_bytes)
                if total_input_bytes > _MAX_TOTAL_IMAGE_BYTES:
                    raise FileAdmissionError(
                        ErrorCode.FILE_TOO_LARGE,
                        limit=_MAX_TOTAL_IMAGE_BYTES,
                        observed=total_input_bytes,
                    )
                content_checksum = hashlib.sha256(content_bytes).hexdigest()
                normalized = normalized_content.get(content_checksum)
                if normalized is None:
                    normalized = _validate_image(content_bytes)
                    normalized_content[content_checksum] = normalized
                cached_reference = normalized, source_kind, provenance
                resolved_references[reference] = cached_reference
            normalized, source_kind, provenance = cached_reference
            total_image_pixels += normalized.width * normalized.height
            if total_image_pixels > _MAX_TOTAL_IMAGE_PIXELS:
                raise FileAdmissionError(
                    ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
                    check="image_total_pixels",
                )
            path = (
                f"{_MEDIA_DIRECTORY}/{normalized.checksum_sha256}"
                f"{normalized.extension}"
            )
            if path not in normalized_files:
                total_output_bytes += len(normalized.content)
                if total_output_bytes > _MAX_TOTAL_IMAGE_BYTES:
                    raise FileAdmissionError(
                        ErrorCode.FILE_TOO_LARGE,
                        limit=_MAX_TOTAL_IMAGE_BYTES,
                        observed=total_output_bytes,
                    )
                normalized_files[path] = normalized.content
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
        if _is_data_reference(reference):
            return _decode_data_uri(reference), "data", "data-uri"
        if reference.startswith("//"):
            fetched = self._fetcher.fetch(
                f"https:{reference}",
                max_bytes=_MAX_IMAGE_BYTES,
            )
            return fetched.content, "remote", fetched.final_url
        parsed = urlsplit(reference)
        if parsed.scheme in {"http", "https"}:
            fetched = self._fetcher.fetch(reference, max_bytes=_MAX_IMAGE_BYTES)
            return fetched.content, "remote", fetched.final_url
        if parsed.scheme or parsed.netloc:
            raise FileAdmissionError(
                ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
                check="reference_scheme",
            )
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


def _normalize_html_images(
    element: object,
    parser: marko.Markdown,
) -> None:
    children = getattr(element, "children", None)
    if not isinstance(children, list):
        return
    normalized: list[object] = []
    for child in children:
        if isinstance(child, HTMLBlock) and _HTML_IMAGE.search(child.body):
            markdown = _html_block_to_markdown(child.body)
            replacement = parser.parse(markdown)
            _normalize_html_images(replacement, parser)
            normalized.extend(replacement.children)
            continue
        if isinstance(child, InlineHTML) and _HTML_IMAGE.search(child.children):
            normalized.append(_inline_html_image(child.children))
            continue
        _normalize_html_images(child, parser)
        normalized.append(child)
    children[:] = normalized


def _inline_html_image(value: str) -> Image | RawText:
    parser = _SingleImageTagParser()
    parser.feed(value)
    parser.close()
    if not parser.saw_image or parser.invalid:
        raise FileAdmissionError(
            ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
            check="html_image_structure",
        )
    if parser.image is None:
        return RawText("")
    reference, alt, title = parser.image
    return _new_image(reference, alt=alt, title=title)


def _html_block_to_markdown(value: str) -> str:
    parser = _HtmlImageBlockParser()
    parser.feed(value)
    parser.close()
    if parser.invalid or parser.image_count == 0:
        raise FileAdmissionError(
            ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
            check="html_image_structure",
        )
    rendered = "".join(parser.output)
    return re.sub(r"\n{3,}", "\n\n", rendered).strip() + "\n"


class _SingleImageTagParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.image: tuple[str, str, str | None] | None = None
        self.saw_image = False
        self.invalid = False

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.lower() != "img" or self.saw_image:
            self.invalid = True
            return
        self.saw_image = True
        image = _image_attributes(attrs)
        if not _is_decorative_html_image(attrs):
            self.image = image

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attrs)

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.invalid = True


class _HtmlImageBlockParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.image_count = 0
        self.invalid = False
        self.output: list[str] = []
        self._suppressed_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        normalized_tag = tag.lower()
        if normalized_tag in _HTML_SUPPRESSED_CONTENT:
            self.invalid = True
            self._suppressed_depth += 1
            return
        if self._suppressed_depth:
            return
        if normalized_tag == "img":
            image = _image_attributes(attrs)
            self.image_count += 1
            if _is_decorative_html_image(attrs):
                return
            reference, alt, title = image
            self.output.append(
                f"\n\n{_markdown_image(reference, alt=alt, title=title)}\n\n"
            )
        elif normalized_tag not in _HTML_IMAGE_BLOCK_TAGS:
            self.invalid = True
        elif normalized_tag in _HTML_BLOCK_BREAKS:
            self.output.append("\n")

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() in _HTML_SUPPRESSED_CONTENT and self._suppressed_depth:
            self._suppressed_depth -= 1

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()
        if normalized_tag in _HTML_SUPPRESSED_CONTENT:
            if self._suppressed_depth:
                self._suppressed_depth -= 1
            return
        if normalized_tag not in _HTML_IMAGE_BLOCK_TAGS:
            self.invalid = True
            return
        if not self._suppressed_depth and normalized_tag in _HTML_BLOCK_BREAKS:
            self.output.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._suppressed_depth:
            self.output.append(data)


def _image_attributes(
    attrs: list[tuple[str, str | None]],
) -> tuple[str, str, str | None]:
    values = {
        name.lower(): value
        for name, value in attrs
        if value is not None
    }
    reference = values.get("src", "").strip()
    alt = values.get("alt", "")
    title = values.get("title")
    if not reference:
        raise FileAdmissionError(
            ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
            check="html_image_missing_src",
        )
    if (
        _has_control_character(reference)
        or _has_control_character(alt)
        or (title is not None and _has_control_character(title))
    ):
        raise FileAdmissionError(
            ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
            check="html_image_attribute",
        )
    if _is_data_reference(reference):
        if len(reference) > _MAX_DATA_URI_REFERENCE_LENGTH:
            raise FileAdmissionError(
                ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
                check="html_image_attribute",
            )
    elif len(reference) > _MAX_REFERENCE_LENGTH:
        raise FileAdmissionError(
            ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
            check="html_image_attribute",
        )
    return reference, alt, title


def _is_decorative_html_image(
    attrs: list[tuple[str, str | None]],
) -> bool:
    values = {
        name.lower(): value
        for name, value in attrs
        if value is not None
    }
    if "alt" in values:
        return not values["alt"].strip()
    width = _declared_html_dimension(values.get("width"))
    height = _declared_html_dimension(values.get("height"))
    return (
        width is not None
        and height is not None
        and width <= _MAX_DECORATIVE_HTML_IMAGE_DIMENSION
        and height <= _MAX_DECORATIVE_HTML_IMAGE_DIMENSION
    )


def _declared_html_dimension(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized.isascii() or not normalized.isdecimal():
        return None
    dimension = int(normalized)
    return dimension if dimension > 0 else None


def _new_image(
    reference: str,
    *,
    alt: str,
    title: str | None,
) -> Image:
    image = object.__new__(Image)
    image.dest = reference
    image.title = title
    image.children = [RawText(_escape_markdown_alt(alt))]
    return image


def _markdown_image(
    reference: str,
    *,
    alt: str,
    title: str | None,
) -> str:
    safe_reference = reference.replace("<", "%3C").replace(">", "%3E")
    title_suffix = ""
    if title:
        safe_title = title.replace("\\", "\\\\").replace('"', '\\"')
        title_suffix = f' "{safe_title}"'
    return (
        f"![{_escape_markdown_alt(alt)}]"
        f"(<{safe_reference}>{title_suffix})"
    )


def _escape_markdown_alt(value: str) -> str:
    return (
        value.replace("\r", " ")
        .replace("\n", " ")
        .replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def _has_control_character(value: str) -> bool:
    return any(
        unicodedata.category(character) in {"Cc", "Cs"}
        for character in value
    )


def _is_data_reference(value: str) -> bool:
    return value[: len(_DATA_PREFIX)].lower() == _DATA_PREFIX


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
    if not content:
        raise FileAdmissionError(
            ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
            check="image_empty",
        )
    if len(content) > _MAX_IMAGE_BYTES:
        raise FileAdmissionError(
            ErrorCode.FILE_TOO_LARGE,
            limit=_MAX_IMAGE_BYTES,
            observed=len(content),
        )
    try:
        with PillowImage.open(BytesIO(content)) as image:
            image_format = image.format
            if (
                image_format not in _ALLOWED_IMAGE_FORMATS
                and image_format not in _CONVERTIBLE_IMAGE_FORMATS
            ):
                raise FileAdmissionError(
                    ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
                    check="image_format",
                )
            if getattr(image, "n_frames", 1) != 1 or getattr(
                image, "is_animated", False
            ):
                raise FileAdmissionError(
                    ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
                    check="image_animated",
                )
            width, height = image.size
            if (
                width <= 0
                or height <= 0
                or width > _MAX_IMAGE_WIDTH
                or height > _MAX_IMAGE_HEIGHT
                or width * height > _MAX_IMAGE_PIXELS
            ):
                raise FileAdmissionError(
                    ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
                    check="image_dimensions",
                )
            image.load()
            if image_format in _ALLOWED_IMAGE_FORMATS:
                normalized_content = content
                media_type, extension = _ALLOWED_IMAGE_FORMATS[image_format]
            else:
                normalized_content = _convert_to_png(image)
                media_type, extension = _ALLOWED_IMAGE_FORMATS["PNG"]
    except FileAdmissionError:
        raise
    except (
        PillowImage.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise FileAdmissionError(
            ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
            check="image_decode",
        ) from error
    if len(normalized_content) > _MAX_IMAGE_BYTES:
        raise FileAdmissionError(
            ErrorCode.FILE_TOO_LARGE,
            limit=_MAX_IMAGE_BYTES,
            observed=len(normalized_content),
        )
    return _NormalizedImage(
        content=normalized_content,
        media_type=media_type,
        extension=extension,
        checksum_sha256=hashlib.sha256(normalized_content).hexdigest(),
        width=width,
        height=height,
    )


def _convert_to_png(image: PillowImage.Image) -> bytes:
    oriented = ImageOps.exif_transpose(image)
    has_alpha = (
        oriented.mode in {"LA", "PA", "RGBA"}
        or "transparency" in oriented.info
    )
    converted = oriented.convert("RGBA" if has_alpha else "RGB")
    target = BytesIO()
    converted.save(target, format="PNG", compress_level=9)
    return target.getvalue()


def _decode_data_uri(value: str) -> bytes:
    header, separator, payload = value.partition(",")
    if (
        not separator
        or not header.lower().endswith(";base64")
        or header[5:].split(";", 1)[0].lower() not in _DATA_IMAGE_MEDIA_TYPES
    ):
        raise FileAdmissionError(
            ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
            check="data_uri",
        )
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise FileAdmissionError(
            ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
            check="data_uri",
        ) from error
    if len(decoded) > _MAX_IMAGE_BYTES:
        raise FileAdmissionError(
            ErrorCode.FILE_TOO_LARGE,
            limit=_MAX_IMAGE_BYTES,
            observed=len(decoded),
        )
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
