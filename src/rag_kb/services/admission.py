"""Bounded admission for the P1A plain-text upload surface."""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import PurePath, PurePosixPath
from typing import BinaryIO
import unicodedata
from zipfile import BadZipFile, ZipFile

from rag_kb.domain import (
    AdmittedFile,
    AdmissionLimits,
    ErrorCode,
    FileAdmissionError,
)


_FILENAME = re.compile(r"^[^/\\\x00]{1,255}$")
_MEDIA_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
_TEXT_EXTENSIONS = {".txt", ".md"}


class FileAdmissionService:
    def __init__(self, limits: AdmissionLimits) -> None:
        self.limits = limits

    def validate(
        self,
        source: BinaryIO,
        *,
        original_filename: str,
        media_type: str,
    ) -> AdmittedFile:
        filename = unicodedata.normalize("NFC", original_filename.strip())
        if (
            not _FILENAME.fullmatch(filename)
            or PurePath(filename).name != filename
            or filename in {".", ".."}
            or any(
                unicodedata.category(character) in {"Cc", "Cs"}
                for character in filename
            )
        ):
            raise FileAdmissionError(ErrorCode.FILE_NAME_INVALID)

        extension = PurePath(filename).suffix.lower()
        expected_media_type = _MEDIA_TYPES.get(extension)
        if expected_media_type is None:
            raise FileAdmissionError(ErrorCode.PARSER_NOT_CONFIGURED)

        normalized_media_type, charset = _parse_media_type(media_type)
        if normalized_media_type not in set(_MEDIA_TYPES.values()):
            raise FileAdmissionError(ErrorCode.FILE_MEDIA_TYPE_UNSUPPORTED)
        if normalized_media_type != expected_media_type:
            raise FileAdmissionError(ErrorCode.FILE_MEDIA_TYPE_MISMATCH)
        if charset is not None and (
            extension not in _TEXT_EXTENSIONS
            or charset not in {"utf-8", "utf8"}
        ):
            raise FileAdmissionError(ErrorCode.FILE_MEDIA_TYPE_UNSUPPORTED)

        source.seek(0)
        content = source.read(self.limits.max_bytes + 1)
        if not isinstance(content, bytes):
            raise TypeError("source file must yield bytes")
        if len(content) > self.limits.max_bytes:
            raise FileAdmissionError(
                ErrorCode.FILE_TOO_LARGE,
                limit=self.limits.max_bytes,
                observed=len(content),
            )
        line_count = None
        if extension in _TEXT_EXTENSIONS:
            line_count = self._validate_text(content)
        elif extension == ".pdf":
            self._validate_pdf(content)
        elif extension == ".docx":
            self._validate_docx(content)
        source.seek(0)
        return AdmittedFile(
            original_filename=filename,
            extension=extension,
            media_type=normalized_media_type,
            size_bytes=len(content),
            line_count=line_count,
        )

    def _validate_text(self, content: bytes) -> int:
        try:
            text = content.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError as error:
            raise FileAdmissionError(ErrorCode.FILE_INVALID_UTF8) from error
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        line_count = _line_count(normalized)
        if line_count > self.limits.max_lines:
            raise FileAdmissionError(
                ErrorCode.FILE_LINE_LIMIT_EXCEEDED,
                limit=self.limits.max_lines,
                observed=line_count,
            )
        return line_count

    @staticmethod
    def _validate_pdf(content: bytes) -> None:
        if b"%PDF-" not in content[:1024]:
            raise FileAdmissionError(ErrorCode.FILE_CONTENT_INVALID)

    def _validate_docx(self, content: bytes) -> None:
        try:
            with ZipFile(BytesIO(content)) as archive:
                entries = archive.infolist()
                if len(entries) > self.limits.max_archive_entries:
                    raise FileAdmissionError(
                        ErrorCode.FILE_ARCHIVE_LIMIT_EXCEEDED,
                        limit=self.limits.max_archive_entries,
                        observed=len(entries),
                    )
                expanded_bytes = sum(item.file_size for item in entries)
                if expanded_bytes > self.limits.max_expanded_bytes:
                    raise FileAdmissionError(
                        ErrorCode.FILE_ARCHIVE_LIMIT_EXCEEDED,
                        limit=self.limits.max_expanded_bytes,
                        observed=expanded_bytes,
                    )
                names = {item.filename for item in entries}
                if (
                    "[Content_Types].xml" not in names
                    or "word/document.xml" not in names
                    or any(
                        item.flag_bits & 0x1 or _unsafe_archive_name(item.filename)
                        for item in entries
                    )
                ):
                    raise FileAdmissionError(ErrorCode.FILE_CONTENT_INVALID)
        except FileAdmissionError:
            raise
        except (BadZipFile, OSError, ValueError) as error:
            raise FileAdmissionError(ErrorCode.FILE_CONTENT_INVALID) from error


def _parse_media_type(value: str) -> tuple[str, str | None]:
    parts = [part.strip() for part in value.split(";")]
    media_type = parts[0].lower()
    charset = None
    for parameter in parts[1:]:
        name, separator, parameter_value = parameter.partition("=")
        if separator and name.strip().lower() == "charset":
            charset = parameter_value.strip().strip('"').lower()
    return media_type, charset


def _line_count(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _unsafe_archive_name(value: str) -> bool:
    path = PurePosixPath(value)
    return path.is_absolute() or ".." in path.parts or "\\" in value
