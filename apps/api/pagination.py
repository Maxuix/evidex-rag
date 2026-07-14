"""Opaque cursor serialization shared by versioned API endpoints."""

from __future__ import annotations

import base64
import binascii
import json

from pydantic import ValidationError

from apps.api.errors import ApiProblem
from rag_kb.schemas import CursorPayload, ErrorCode


def encode_cursor(payload: CursorPayload) -> str:
    raw = payload.model_dump_json(exclude_none=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def decode_cursor(value: str) -> CursorPayload:
    try:
        if not value or len(value) > 2048:
            raise ValueError("cursor length is invalid")
        padding = "=" * (-len(value) % 4)
        raw = base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        )
        decoded = json.loads(raw.decode("utf-8"))
        return CursorPayload.model_validate(decoded)
    except (
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        binascii.Error,
        ValidationError,
    ):
        raise ApiProblem(
            code=ErrorCode.INVALID_CURSOR,
            status=400,
            title="Invalid cursor",
            detail="The pagination cursor is malformed or incompatible.",
            retryable=False,
        ) from None
