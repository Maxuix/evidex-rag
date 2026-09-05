"""Shared budgets for serialized Docling images, before pixel decoding."""

from __future__ import annotations

import base64
from io import BytesIO
import math
from pathlib import Path
import re
import warnings

from docling_core.types.doc import DoclingDocument
from PIL import Image

from rag_kb.domain import ErrorCode, ParserExecutionError, ParserLimits

_BASE64 = re.compile(r"[A-Za-z0-9+/]*={0,2}\Z")


def require_limit(name: str, value: int, limits: ParserLimits) -> None:
    maximum = getattr(limits, name)
    if value > maximum:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic={"limit_name": name, "limit": maximum},
        )


def image_dimensions(
    width: float, height: float, limits: ParserLimits,
) -> tuple[int, int]:
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value) and value >= 1
        for value in (width, height)
    ):
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "docling_image_dimensions"},
        )
    width, height = math.ceil(width), math.ceil(height)
    for name, value in (
        ("max_image_width", width),
        ("max_image_height", height),
        ("max_image_pixels", width * height),
    ):
        require_limit(name, value, limits)
    return width, height


def image_payload(uri: object) -> str:
    value = str(uri)
    if (
        isinstance(uri, Path)
        or not value.startswith("data:image/")
        or ";base64," not in value
    ):
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "docling_image_ref"},
        )
    encoded = value.split(";base64,", 1)[1]
    if len(encoded) % 4 or not _BASE64.fullmatch(encoded):
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "docling_image_ref"},
        )
    return encoded


def image_usage(
    document: DoclingDocument,
    limits: ParserLimits,
    *,
    verify_headers: bool = False,
) -> dict[str, int]:
    refs = [page.image for page in document.pages.values() if page.image is not None]
    refs.extend(
        item.image for item in (*document.pictures, *document.tables)
        if item.image is not None
    )
    usage = {
        "max_assets": len(refs),
        "max_total_asset_bytes": 0,
        "max_total_image_pixels": 0,
    }
    require_limit("max_assets", len(refs), limits)
    for ref in refs:
        width, height = image_dimensions(ref.size.width, ref.size.height, limits)
        usage["max_total_image_pixels"] += width * height
        require_limit("max_total_image_pixels", usage["max_total_image_pixels"], limits)
        encoded = image_payload(ref.uri)
        usage["max_total_asset_bytes"] += len(encoded) * 3 // 4 - encoded.count("=")
        require_limit("max_total_asset_bytes", usage["max_total_asset_bytes"], limits)
    if verify_headers:
        for ref in refs:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    encoded = image_payload(ref.uri)
                    with Image.open(
                        BytesIO(base64.b64decode(encoded, validate=True))
                    ) as header:
                        dimensions = image_dimensions(*header.size, limits)
                        if dimensions != (ref.size.width, ref.size.height):
                            raise ValueError("inconsistent dimensions")
            except ParserExecutionError:
                raise
            except (
                Image.DecompressionBombWarning, Image.DecompressionBombError,
            ) as error:
                raise ParserExecutionError(
                    ErrorCode.PARSER_RESOURCE_LIMIT,
                    diagnostic={
                        "limit_name": "max_image_pixels",
                        "limit": limits.max_image_pixels,
                    },
                ) from error
            except Exception as error:
                raise ParserExecutionError(
                    ErrorCode.PARSER_OUTPUT_INVALID,
                    diagnostic={"check": "docling_image_header"},
                ) from error
    return usage
