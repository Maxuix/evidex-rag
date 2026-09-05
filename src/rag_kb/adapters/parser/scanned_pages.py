"""Bounded PDF page facts, executed only in the owned parser child.

Image coverage is measured from drawing operations, so a page number or a stale
OCR text layer does not hide a scan. No text extraction or image decoding runs.
"""

from __future__ import annotations

from contextlib import contextmanager
from io import BytesIO
import math
from pathlib import PurePath

from pypdf import PdfReader, filters
from pypdf.errors import LimitReachedError
from pypdf.generic import ArrayObject, ContentStream, DecodedStreamObject

from rag_kb.domain import ErrorCode, ParserExecutionError, ParserLimits, ParserSource
from rag_kb.document_processing.docling.resources import require_limit

_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
_ALLOWED_FILTERS = frozenset({
    "/FlateDecode", "/Fl", "/LZWDecode", "/LZW", "/ASCII85Decode", "/A85",
    "/ASCIIHexDecode", "/AHx", "/RunLengthDecode", "/RL",
})


@contextmanager
def _decode_limits(maximum):
    # pypdf exposes these guards; the parser child executes requests serially.
    names = ("ZLIB_MAX_OUTPUT_LENGTH", "LZW_MAX_OUTPUT_LENGTH", "RUN_LENGTH_MAX_OUTPUT_LENGTH")
    previous = {name: getattr(filters, name) for name in names}
    try:
        for name, value in previous.items():
            setattr(filters, name, min(maximum, value) if value else maximum)
        yield
    finally:
        for name, value in previous.items():
            setattr(filters, name, value)


def probe_pdf(
    source: ParserSource, limits: ParserLimits, *, include_surfaces: bool,
) -> tuple[int, frozenset[int]]:
    try:
        reader = PdfReader(BytesIO(source.content), strict=False)
        count = len(reader.pages)
        if count < 1:
            raise ValueError("empty PDF")
        require_limit("max_num_pages", count, limits)
        if not include_surfaces:
            return count, frozenset()
        budget = {"bytes": 0, "operators": 0, "images": 0}
        surfaces = set()
        with _decode_limits(limits.max_pdf_content_bytes):
            for number, page in enumerate(reader.pages, start=1):
                page_box = tuple(float(v) for v in page.cropbox)
                area = _area(page_box)
                if not math.isfinite(area) or area <= 0:
                    raise ValueError("invalid page box")
                images = []
                text = [False]
                _walk(page.get("/Contents"), page.get("/Resources", {}), reader, _IDENTITY,
                      page_box, set(), 0, images, text, budget, limits)
                # Preserve image-only pages and scans with overlaid page numbers,
                # watermarks or OCR text. Small logos beside native text stay out.
                if images and (not text[0] or _union_area(images) / area >= 0.5):
                    surfaces.add(number)
        return count, frozenset(surfaces)
    except ParserExecutionError:
        raise
    except LimitReachedError as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic={"limit_name": "max_pdf_content_bytes", "limit": limits.max_pdf_content_bytes},
        ) from error
    except Exception as error:
        raise ParserExecutionError(
            ErrorCode.FILE_CONTENT_INVALID,
            diagnostic={"check": "pdf_page_probe"},
        ) from error


def scanned_surfaces(
    source: ParserSource, limits: ParserLimits | None = None,
) -> frozenset[int]:
    if PurePath(source.original_filename).suffix.lower() != ".pdf":
        return frozenset()
    return probe_pdf(source, limits or ParserLimits(), include_surfaces=True)[1]


def _decoded(contents, budget, limits):
    if contents is None:
        return b""
    obj = contents.get_object()
    if isinstance(obj, ArrayObject):
        return b"\n".join(_decoded(item, budget, limits) for item in obj)
    codecs = obj.get("/Filter", [])
    if not isinstance(codecs, (list, ArrayObject)):
        codecs = [codecs]
    if any(str(codec) not in _ALLOWED_FILTERS for codec in codecs):
        raise ValueError("unsupported content stream filter")
    data = obj.get_data()
    budget["bytes"] += len(data)
    require_limit("max_pdf_content_bytes", budget["bytes"], limits)
    return data


def _walk(contents, resources, reader, matrix, clip, active, depth, images, text, budget, limits):
    require_limit("max_pdf_form_depth", depth, limits)
    stream = DecodedStreamObject()
    stream.set_data(_decoded(contents, budget, limits))
    operations = ContentStream(stream, reader).operations
    budget["operators"] += len(operations)
    require_limit("max_pdf_operators", budget["operators"], limits)
    resources = resources.get_object() if hasattr(resources, "get_object") else resources
    objects = resources.get("/XObject", {})
    objects = objects.get_object() if hasattr(objects, "get_object") else objects
    stack = []
    for operands, operator in operations:
        if operator == b"q":
            stack.append(matrix)
            require_limit("max_pdf_form_depth", depth + len(stack), limits)
        elif operator == b"Q":
            if not stack:
                raise ValueError("unbalanced graphics state")
            matrix = stack.pop()
        elif operator == b"cm":
            matrix = _compose(tuple(float(v) for v in operands), matrix)
        elif operator in (b"Tj", b"TJ", b"'", b'"'):
            text[0] = True
        elif operator == b"INLINE IMAGE":
            _image(matrix, clip, images, budget, limits)
        elif operator == b"Do":
            target = objects[operands[0]].get_object()
            subtype = target.get("/Subtype")
            if subtype == "/Image":
                _image(matrix, clip, images, budget, limits)
            elif subtype == "/Form":
                identity = id(target)
                if identity in active:
                    raise ValueError("cyclic form")
                transform = _compose(tuple(float(v) for v in target.get("/Matrix", _IDENTITY)), matrix)
                form_clip = (
                    _intersect(clip, _rectangle(target["/BBox"], transform))
                    if "/BBox" in target else clip
                )
                _walk(target, target.get("/Resources", resources), reader, transform, form_clip,
                      active | {identity}, depth + 1, images, text, budget, limits)


def _compose(a, b):
    if len(a) != 6 or not all(math.isfinite(v) for v in (*a, *b)):
        raise ValueError("invalid matrix")
    return (
        a[0]*b[0] + a[1]*b[2], a[0]*b[1] + a[1]*b[3],
        a[2]*b[0] + a[3]*b[2], a[2]*b[1] + a[3]*b[3],
        a[4]*b[0] + a[5]*b[2] + b[4], a[4]*b[1] + a[5]*b[3] + b[5],
    )


def _rectangle(box, matrix):
    left, bottom, right, top = map(float, box)
    points = [
        (matrix[0]*x + matrix[2]*y + matrix[4], matrix[1]*x + matrix[3]*y + matrix[5])
        for x, y in ((left, bottom), (left, top), (right, bottom), (right, top))
    ]
    if not all(math.isfinite(v) for point in points for v in point):
        raise ValueError("invalid rectangle")
    return (
        min(p[0] for p in points), min(p[1] for p in points),
        max(p[0] for p in points), max(p[1] for p in points),
    )


def _intersect(a, b):
    return max(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3])


def _area(rect):
    return max(0., rect[2]-rect[0]) * max(0., rect[3]-rect[1])


def _image(matrix, clip, images, budget, limits):
    budget["images"] += 1
    require_limit("max_assets", budget["images"], limits)
    rect = _intersect(_rectangle((0, 0, 1, 1), matrix), clip)
    if _area(rect) > 0:
        images.append(rect)


def _union_area(rectangles):
    xs = sorted({r[i] for r in rectangles for i in (0, 2)})
    total = 0.
    for left, right in zip(xs, xs[1:]):
        ranges = sorted((r[1], r[3]) for r in rectangles if r[0] < right and r[2] > left)
        end = float("-inf")
        height = 0.
        for bottom, top in ranges:
            height += max(0., top - max(bottom, end))
            end = max(end, top)
        total += (right-left) * height
    return total
