"""Cheap structural resource checks that run before Docling conversion."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import PurePath
import warnings
from zipfile import ZipFile

from PIL import Image as PillowImage
from PIL import UnidentifiedImageError


_CSV_DELIMITERS = ",;\t|:"
_OOXML_MEDIA_PREFIXES = {
    ".docx": "word/media/",
    ".pptx": "ppt/media/",
    ".xlsx": "xl/media/",
}
_RASTER_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class ResourcePreflightLimitError(ValueError):
    """A content-safe deterministic resource-budget violation."""

    def __init__(
        self,
        limit_name: str,
        limit: int,
        *,
        observed: int | None = None,
    ) -> None:
        super().__init__(limit_name)
        self.limit_name = limit_name
        self.limit = limit
        self.observed = observed


class ResourcePreflightContentError(ValueError):
    """The bounded structural probe could not parse the declared input."""


def validate_csv_structure(
    text: str,
    *,
    max_columns: int,
    max_cells: int,
) -> None:
    """Bound CSV object growth using the locked Docling dialect semantics."""

    content = StringIO(text)
    head = content.readline()
    try:
        dialect: type[csv.Dialect] = csv.Sniffer().sniff(
            head,
            _CSV_DELIMITERS,
        )
        if dialect.delimiter not in set(_CSV_DELIMITERS):
            raise csv.Error("unsupported delimiter")
    except csv.Error:
        dialect = csv.excel

    content.seek(0)
    cell_count = 0
    try:
        for row in csv.reader(content, dialect=dialect, strict=True):
            column_count = len(row)
            if column_count > max_columns:
                raise ResourcePreflightLimitError(
                    "max_csv_columns",
                    max_columns,
                    observed=column_count,
                )
            cell_count += column_count
            if cell_count > max_cells:
                raise ResourcePreflightLimitError(
                    "max_csv_cells",
                    max_cells,
                    observed=cell_count,
                )
    except ResourcePreflightLimitError:
        raise
    except (csv.Error, UnicodeError, ValueError) as error:
        raise ResourcePreflightContentError("csv_structure") from error


def validate_ooxml_images(
    archive: ZipFile,
    *,
    extension: str,
    max_images: int,
    max_image_width: int,
    max_image_height: int,
    max_image_pixels: int,
    max_total_image_pixels: int,
) -> None:
    """Inspect OOXML raster headers without decoding their pixel payloads."""

    prefix = _OOXML_MEDIA_PREFIXES.get(extension.lower())
    if prefix is None:
        return

    image_count = 0
    total_pixels = 0
    for entry in archive.infolist():
        if entry.is_dir() or not entry.filename.lower().startswith(prefix):
            continue
        suffix = PurePath(entry.filename).suffix.lower()
        try:
            with archive.open(entry) as member, warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore",
                    PillowImage.DecompressionBombWarning,
                )
                with PillowImage.open(member) as image:
                    width, height = image.size
        except PillowImage.DecompressionBombError as error:
            raise ResourcePreflightLimitError(
                "max_image_pixels",
                max_image_pixels,
            ) from error
        except (UnidentifiedImageError, OSError, ValueError) as error:
            if suffix in _RASTER_EXTENSIONS:
                raise ResourcePreflightContentError("ooxml_image_header") from error
            continue

        if width <= 0 or height <= 0:
            raise ResourcePreflightContentError("ooxml_image_dimensions")
        if width > max_image_width:
            raise ResourcePreflightLimitError(
                "max_image_width",
                max_image_width,
                observed=width,
            )
        if height > max_image_height:
            raise ResourcePreflightLimitError(
                "max_image_height",
                max_image_height,
                observed=height,
            )
        pixels = width * height
        if pixels > max_image_pixels:
            raise ResourcePreflightLimitError(
                "max_image_pixels",
                max_image_pixels,
                observed=pixels,
            )
        image_count += 1
        if image_count > max_images:
            raise ResourcePreflightLimitError(
                "max_assets",
                max_images,
                observed=image_count,
            )
        total_pixels += pixels
        if total_pixels > max_total_image_pixels:
            raise ResourcePreflightLimitError(
                "max_total_image_pixels",
                max_total_image_pixels,
                observed=total_pixels,
            )
