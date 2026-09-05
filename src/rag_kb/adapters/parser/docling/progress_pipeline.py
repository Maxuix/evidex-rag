"""Content-safe progress hooks for the frozen Docling PDF pipeline."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import resource
import sys
import threading
import time
from typing import Any, Iterator

from docling.datamodel.base_models import DocItemLabel
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline


ProgressEmitter = Callable[[dict[str, Any]], None]
_EMITTER_LOCK = threading.Lock()
_ACTIVE_EMITTER: ProgressEmitter | None = None
_PROGRESS_PAGE_INTERVAL = 1
_PROGRESS_TIME_INTERVAL_SECONDS = 1.0


@contextmanager
def emit_pdf_progress(emitter: ProgressEmitter | None) -> Iterator[None]:
    """Install one child-process emitter for the current serial conversion."""

    global _ACTIVE_EMITTER
    with _EMITTER_LOCK:
        if _ACTIVE_EMITTER is not None:
            raise RuntimeError("a PDF progress emitter is already active")
        _ACTIVE_EMITTER = emitter
    try:
        yield
    finally:
        with _EMITTER_LOCK:
            _ACTIVE_EMITTER = None


class ProgressStandardPdfPipeline(StandardPdfPipeline):
    """StandardPdfPipeline with throttled counters and unchanged document output."""

    def _build_document(self, conv_res):
        expected = self._get_expected_page_nos(conv_res)
        _emit(
            {
                "stage": "page_parse",
                "segment_total_pages": len(expected),
                "stage_completed_pages": 0,
                "ocr_pages": 0,
                "ocr_regions": 0,
                "table_candidates": 0,
            }
        )
        return super()._build_document(conv_res)

    def _create_run_ctx(self):
        ctx = super()._create_run_ctx()
        counters: dict[str, int] = {stage.name: 0 for stage in ctx.stages}
        last_emitted: dict[str, tuple[int, float]] = {
            stage.name: (0, 0.0) for stage in ctx.stages
        }
        metrics = {"ocr_pages": 0, "ocr_regions": 0, "table_candidates": 0}
        metrics_lock = threading.Lock()

        for stage in ctx.stages:
            original = stage._postprocess

            def report(item, *, name=stage.name, postprocess=original):
                payload = item.payload
                with metrics_lock:
                    counters[name] += 1
                    if name == "ocr" and payload is not None:
                        parsed_page = getattr(payload, "parsed_page", None)
                        cells = getattr(parsed_page, "word_cells", ()) or ()
                        ocr_regions = sum(
                            bool(getattr(cell, "from_ocr", False)) for cell in cells
                        )
                        if ocr_regions:
                            metrics["ocr_pages"] += 1
                            metrics["ocr_regions"] += ocr_regions
                    elif name == "layout" and payload is not None:
                        prediction = getattr(payload.predictions, "layout", None)
                        clusters = getattr(prediction, "clusters", ()) or ()
                        metrics["table_candidates"] += sum(
                            getattr(cluster, "label", None) is DocItemLabel.TABLE
                            for cluster in clusters
                        )
                    completed = counters[name]
                    previous, emitted_at = last_emitted[name]
                    now = time.monotonic()
                    should_emit = (
                        completed - previous >= _PROGRESS_PAGE_INTERVAL
                        or now - emitted_at >= _PROGRESS_TIME_INTERVAL_SECONDS
                    )
                    if should_emit:
                        last_emitted[name] = (completed, now)
                        _emit(
                            {
                                "stage": _public_stage(name),
                                "stage_completed_pages": completed,
                                **metrics,
                            }
                        )
                if postprocess is not None:
                    postprocess(item)

            stage._postprocess = report
        return ctx

    def _assemble_document(self, conv_res):
        _emit({"stage": "document_assembly", "stage_completed_pages": 0})
        result = super()._assemble_document(conv_res)
        _emit(
            {
                "stage": "document_assembly",
                "stage_completed_pages": len(conv_res.pages),
            }
        )
        return result


def _public_stage(stage: str) -> str:
    return {
        "preprocess": "page_parse",
        "ocr": "ocr",
        "layout": "layout",
        "table": "table_structure",
        "assemble": "page_assembly",
    }.get(stage, stage)


def _emit(payload: dict[str, Any]) -> None:
    with _EMITTER_LOCK:
        emitter = _ACTIVE_EMITTER
    if emitter is None:
        return
    enriched = {
        **payload,
        "child_peak_rss_bytes": _child_peak_rss_bytes(),
    }
    emitter(enriched)


def _child_peak_rss_bytes() -> int:
    # Worker containers run Linux, where ru_maxrss is expressed in KiB.
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024
