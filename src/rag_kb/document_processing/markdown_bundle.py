"""Pure validation and reading for the versioned Markdown ZIP boundary."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import PurePosixPath
import stat
from zipfile import BadZipFile, ZipFile, ZipInfo

from rag_kb.domain import ErrorCode, FileAdmissionError


MARKDOWN_BUNDLE_MEDIA_TYPE = "application/vnd.rag-kb.markdown-bundle+zip"
MARKDOWN_BUNDLE_EXTENSION = ".mdz"
MARKDOWN_BUNDLE_VERSION = 1
MARKDOWN_BUNDLE_ENTRYPOINT = "document.md"
MARKDOWN_BUNDLE_MANIFEST = "manifest.json"
MAX_ARCHIVE_ENTRIES = 1_000
MAX_ARCHIVE_EXPANDED_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class MarkdownBundle:
    markdown: str
    entrypoint: str
    files: dict[str, bytes]


def read_markdown_bundle(content: bytes) -> MarkdownBundle:
    try:
        with ZipFile(BytesIO(content)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_ARCHIVE_ENTRIES:
                raise FileAdmissionError(ErrorCode.FILE_ARCHIVE_LIMIT_EXCEEDED)
            if (
                sum(info.file_size for info in infos)
                > MAX_ARCHIVE_EXPANDED_BYTES
            ):
                raise FileAdmissionError(ErrorCode.FILE_ARCHIVE_LIMIT_EXCEEDED)
            files: dict[str, bytes] = {}
            for info in infos:
                name = _safe_archive_name(info)
                if not info.is_dir() and name in files:
                    raise FileAdmissionError(ErrorCode.MARKDOWN_BUNDLE_INVALID)
                if info.is_dir():
                    continue
                files[name] = archive.read(info)
    except FileAdmissionError:
        raise
    except (BadZipFile, KeyError, OSError, ValueError) as error:
        raise FileAdmissionError(ErrorCode.MARKDOWN_BUNDLE_INVALID) from error

    manifest_bytes = files.get(MARKDOWN_BUNDLE_MANIFEST)
    if manifest_bytes is None:
        raise FileAdmissionError(ErrorCode.MARKDOWN_BUNDLE_INVALID)
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FileAdmissionError(ErrorCode.MARKDOWN_BUNDLE_INVALID) from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != MARKDOWN_BUNDLE_VERSION
        or not isinstance(manifest.get("entrypoint"), str)
    ):
        raise FileAdmissionError(ErrorCode.MARKDOWN_BUNDLE_INVALID)
    entrypoint = safe_relative_path(manifest["entrypoint"])
    if PurePosixPath(entrypoint).suffix.lower() != ".md":
        raise FileAdmissionError(ErrorCode.MARKDOWN_BUNDLE_INVALID)
    markdown_bytes = files.get(entrypoint)
    if markdown_bytes is None:
        raise FileAdmissionError(ErrorCode.MARKDOWN_BUNDLE_INVALID)
    try:
        markdown = markdown_bytes.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise FileAdmissionError(ErrorCode.FILE_INVALID_UTF8) from error
    return MarkdownBundle(
        markdown=markdown,
        entrypoint=entrypoint,
        files=files,
    )


def read_normalized_markdown_bundle(
    content: bytes,
) -> tuple[str, dict[str, bytes]]:
    bundle = read_markdown_bundle(content)
    if bundle.entrypoint != MARKDOWN_BUNDLE_ENTRYPOINT:
        raise FileAdmissionError(ErrorCode.MARKDOWN_BUNDLE_INVALID)
    return bundle.entrypoint, bundle.files


def safe_relative_path(value: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        raise FileAdmissionError(ErrorCode.MARKDOWN_BUNDLE_INVALID)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.startswith("./"):
        raise FileAdmissionError(ErrorCode.MARKDOWN_BUNDLE_INVALID)
    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise FileAdmissionError(ErrorCode.MARKDOWN_BUNDLE_INVALID)
    return normalized


def _safe_archive_name(info: ZipInfo) -> str:
    if info.flag_bits & 0x1:
        raise FileAdmissionError(ErrorCode.MARKDOWN_BUNDLE_INVALID)
    mode = info.external_attr >> 16
    if mode and stat.S_ISLNK(mode):
        raise FileAdmissionError(ErrorCode.MARKDOWN_BUNDLE_INVALID)
    value = info.filename.rstrip("/") if info.is_dir() else info.filename
    return safe_relative_path(value)
