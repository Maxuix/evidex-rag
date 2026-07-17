"""Bounded admission for the P1A plain-text upload surface."""

from __future__ import annotations

import re
from pathlib import PurePath
from typing import BinaryIO
import unicodedata

from rag_kb.domain import (
    AdmittedFile,
    AdmissionLimits,
    ErrorCode,
    FileAdmissionError,
)


_FILENAME = re.compile(r"^[^/\\\x00]{1,255}$")
_MEDIA_TYPES = {".txt": "text/plain", ".md": "text/markdown"}


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
        if charset is not None and charset not in {"utf-8", "utf8"}:
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
        source.seek(0)
        return AdmittedFile(
            original_filename=filename,
            extension=extension,
            media_type=normalized_media_type,
            size_bytes=len(content),
            line_count=line_count,
        )


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
