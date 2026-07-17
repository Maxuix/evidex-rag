"""Bounded decoding for browser-safe Unicode upload metadata."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import json
import re
from typing import Any, NoReturn

from apps.api.errors import ApiProblem
from rag_kb.schemas import ErrorCode, FieldViolation


_BASE64URL = re.compile(r"^[A-Za-z0-9_-]+$")
_MAX_DECODED_BYTES = 3072
_METADATA_KEYS = frozenset({"v", "filename", "display_name"})


@dataclass(frozen=True, slots=True)
class UploadMetadata:
    original_filename: str
    display_name: str | None


def resolve_upload_metadata(
    *,
    encoded_metadata: str | None,
    legacy_filename: str | None,
    legacy_display_name: str | None,
) -> UploadMetadata:
    """Accept exactly one version of the upload metadata transport."""

    if encoded_metadata is not None:
        if legacy_filename is not None or legacy_display_name is not None:
            _invalid_metadata(
                "Encoded and legacy document metadata headers cannot be combined.",
                error_type="upload_metadata_conflict",
            )
        return _decode_metadata(encoded_metadata)
    if legacy_filename is None:
        _invalid_metadata(
            "X-Document-Metadata or X-Document-Filename is required.",
            error_type="missing",
        )
    return UploadMetadata(
        original_filename=legacy_filename,
        display_name=legacy_display_name,
    )


def _decode_metadata(value: str) -> UploadMetadata:
    if not _BASE64URL.fullmatch(value):
        _invalid_metadata(
            "X-Document-Metadata must use unpadded Base64URL encoding.",
            error_type="upload_metadata_encoding",
        )
    padding = "=" * (-len(value) % 4)
    try:
        raw = base64.b64decode(
            f"{value}{padding}",
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError):
        _invalid_metadata(
            "X-Document-Metadata is not valid Base64URL.",
            error_type="upload_metadata_encoding",
        )
    if len(raw) > _MAX_DECODED_BYTES:
        _invalid_metadata(
            "X-Document-Metadata exceeds the decoded size limit.",
            error_type="upload_metadata_too_large",
        )
    try:
        decoded = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        _invalid_metadata(
            "X-Document-Metadata is not valid UTF-8.",
            error_type="upload_metadata_utf8",
        )
    try:
        payload: Any = json.loads(decoded)
    except (json.JSONDecodeError, RecursionError):
        _invalid_metadata(
            "X-Document-Metadata is not valid JSON.",
            error_type="upload_metadata_json",
        )
    if not isinstance(payload, dict) or set(payload) - _METADATA_KEYS:
        _invalid_metadata(
            "X-Document-Metadata must contain only supported fields.",
            error_type="upload_metadata_schema",
        )
    if type(payload.get("v")) is not int or payload["v"] != 1:
        _invalid_metadata(
            "X-Document-Metadata uses an unsupported version.",
            error_type="upload_metadata_version",
        )
    filename = payload.get("filename")
    display_name = payload.get("display_name")
    if not isinstance(filename, str) or (
        display_name is not None and not isinstance(display_name, str)
    ):
        _invalid_metadata(
            "X-Document-Metadata contains invalid field types.",
            error_type="upload_metadata_schema",
        )
    return UploadMetadata(
        original_filename=filename,
        display_name=display_name,
    )


def _invalid_metadata(message: str, *, error_type: str) -> NoReturn:
    raise ApiProblem(
        code=ErrorCode.REQUEST_VALIDATION_FAILED,
        status=422,
        title="Request validation failed",
        detail="The document upload metadata is invalid.",
        errors=(
            FieldViolation(
                location=("header", "x-document-metadata"),
                message=message,
                error_type=error_type,
            ),
        ),
    )
