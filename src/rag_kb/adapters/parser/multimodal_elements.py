"""Safe normalization for local multimodal parser outputs."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from typing import Any

from PIL import Image, UnidentifiedImageError

from rag_kb.domain import ErrorCode, ParsedAssetDraft, ParserExecutionError, ParserLimits


_MEDIA_BY_FORMAT = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
}


def bounded_image_asset(
    content: bytes,
    *,
    kind: str,
    source_location: dict[str, Any],
    limits: ParserLimits,
) -> ParsedAssetDraft:
    try:
        with Image.open(BytesIO(content)) as image:
            media_type = _MEDIA_BY_FORMAT.get(image.format or "")
            width, height = image.size
            frames = getattr(image, "n_frames", 1)
            if media_type is None or frames != 1:
                raise ValueError("unsupported image format or frame count")
            if (
                width < 1
                or height < 1
                or width > limits.max_image_width
                or height > limits.max_image_height
                or width * height > limits.max_image_pixels
            ):
                raise ParserExecutionError(
                    ErrorCode.PARSER_RESOURCE_LIMIT,
                    diagnostic={
                        "limit_name": "max_image_pixels",
                        "limit": limits.max_image_pixels,
                    },
                )
            image.verify()
    except ParserExecutionError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "image_format"},
        ) from error
    digest = hashlib.sha256(content).hexdigest()
    location = json.dumps(
        source_location, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    asset_key = hashlib.sha256(
        f"{kind}\x1f{location}\x1f{digest}".encode("utf-8")
    ).hexdigest()
    return ParsedAssetDraft(
        asset_key=asset_key,
        kind=kind,
        media_type=media_type,
        content=content,
        content_sha256=digest,
        width=width,
        height=height,
        source_location=dict(source_location),
        processing_metadata={"format": media_type.removeprefix("image/")},
    )


def stable_element_key(
    source_checksum: str,
    profile: str,
    ordinal: int,
    category: str,
    source_location: dict[str, Any],
    text: str,
    asset_key: str | None,
) -> str:
    payload = {
        "source_checksum": source_checksum,
        "profile": profile,
        "ordinal": ordinal,
        "category": category,
        "source_location": source_location,
        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "asset_key": asset_key,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
