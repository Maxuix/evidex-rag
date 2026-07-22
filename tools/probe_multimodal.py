#!/usr/bin/env python3
"""Read-only capability probe for the locked local multimodal stack."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import resource
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--docx", type=Path)
    parser.add_argument(
        "--exercise-parser",
        action="store_true",
        help="parse the supplied PDF twice and report deterministic, bounded facts",
    )
    parser.add_argument(
        "--expect-ocr",
        help="require this text in OCR output without printing the extracted text",
    )
    arguments = parser.parse_args()
    if arguments.exercise_parser and arguments.pdf is None:
        parser.error("--exercise-parser requires --pdf")
    if arguments.expect_ocr and not arguments.exercise_parser:
        parser.error("--expect-ocr requires --exercise-parser")

    cache_root = Path(tempfile.gettempdir()) / "rag-kb-capability-probe-cache"
    cache_root.mkdir(mode=0o700, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

    from unstructured.partition.docx import partition_docx, register_picture_partitioner
    from unstructured.partition.pdf import partition_pdf

    pdf_parameters = inspect.signature(partition_pdf).parameters
    report: dict[str, object] = {
        "unstructured_version": importlib.metadata.version("unstructured"),
        "pillow_version": importlib.metadata.version("pillow"),
        "executables": {
            name: shutil.which(name) is not None for name in ("pdftoppm", "tesseract")
        },
        "pdf_contract": {
            name: name in pdf_parameters
            for name in (
                "strategy",
                "infer_table_structure",
                "extract_image_block_types",
                "extract_image_block_output_dir",
            )
        },
        "docx_contract": {
            "partition_docx": callable(partition_docx),
            "register_picture_partitioner": callable(register_picture_partitioner),
        },
    }
    if arguments.pdf is not None:
        report["pdf_fixture"] = _fixture(arguments.pdf, ".pdf")
        if arguments.exercise_parser:
            report["pdf_parse"] = _parse_pdf(arguments.pdf, arguments.expect_ocr)
    if arguments.docx is not None:
        report["docx_fixture"] = _fixture(arguments.docx, ".docx")
    print(json.dumps(report, indent=2, sort_keys=True))

    tools = report["executables"]
    pdf_contract = report["pdf_contract"]
    assert isinstance(tools, dict) and isinstance(pdf_contract, dict)
    succeeded = all(tools.values()) and all(pdf_contract.values())
    parse_report = report.get("pdf_parse")
    if isinstance(parse_report, dict):
        succeeded = succeeded and bool(parse_report["repeatable"])
        succeeded = succeeded and bool(
            parse_report.get("ocr_expectation_matched", True)
        )
    return 0 if succeeded else 1


def _fixture(path: Path, suffix: str) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if resolved.suffix.lower() != suffix or not resolved.is_file():
        raise ValueError(f"fixture must be a {suffix} file")
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size}


def _parse_pdf(path: Path, expected_ocr: str | None) -> dict[str, object]:
    from rag_kb.adapters.parser.multimodal import (
        partition_multimodal_with_unstructured,
    )
    from rag_kb.document_processing import (
        assemble_multimodal_units,
        asset_manifest_hash,
        element_sequence_hash,
    )
    from rag_kb.domain import ParserLimits, ParserSource

    resolved = path.resolve(strict=True)
    source = ParserSource(resolved.name, "application/pdf", resolved.read_bytes())
    peak_temp_bytes = 0
    stop_sampling = threading.Event()

    with tempfile.TemporaryDirectory(prefix="rag-kb-capability-parse-") as root:
        temp_root = Path(root)

        def sample_temp() -> None:
            nonlocal peak_temp_bytes
            while not stop_sampling.wait(0.02):
                peak_temp_bytes = max(peak_temp_bytes, _tree_bytes(temp_root))

        sampler = threading.Thread(target=sample_temp, daemon=True)
        sampler.start()
        rss_before = _peak_rss_bytes()
        durations: list[float] = []
        parsed = []
        try:
            for _ in range(2):
                started = time.perf_counter()
                parsed.append(
                    partition_multimodal_with_unstructured(
                        source,
                        ParserLimits(),
                        temp_root=temp_root,
                    )
                )
                durations.append(time.perf_counter() - started)
        finally:
            peak_temp_bytes = max(peak_temp_bytes, _tree_bytes(temp_root))
            stop_sampling.set()
            sampler.join(timeout=1)
        rss_after = _peak_rss_bytes()
        temp_residue_bytes = _tree_bytes(temp_root)

    first, second = parsed
    first_element_hash = element_sequence_hash(first)
    second_element_hash = element_sequence_hash(second)
    first_asset_hash = asset_manifest_hash(first.assets)
    second_asset_hash = asset_manifest_hash(second.assets)
    units = assemble_multimodal_units(first)
    ocr_text = "\n".join(
        item.text for item in first.elements if item.category == "OCRText"
    )
    report: dict[str, object] = {
        "durations_seconds": [round(value, 3) for value in durations],
        "element_count": len(first.elements),
        "element_categories": [item.category for item in first.elements],
        "asset_count": len(first.assets),
        "asset_bytes": sum(len(item.content) for item in first.assets),
        "page_image_count": sum(item.kind == "page_image" for item in first.assets),
        "ocr_text_characters": len(ocr_text),
        "ocr_text_sha256": hashlib.sha256(ocr_text.encode()).hexdigest(),
        "unit_count": len(units),
        "unit_modalities": [item.modality.value for item in units],
        "required_representations": [
            list(item.required_representations) for item in units
        ],
        "element_sequence_hash": first_element_hash,
        "asset_manifest_hash": first_asset_hash,
        "repeatable": (
            first_element_hash == second_element_hash
            and first_asset_hash == second_asset_hash
        ),
        "peak_rss_bytes": rss_after,
        "peak_rss_growth_bytes": max(0, rss_after - rss_before),
        "peak_temp_bytes": peak_temp_bytes,
        "temp_residue_bytes": temp_residue_bytes,
    }
    if expected_ocr is not None:
        report["ocr_expectation_matched"] = expected_ocr in ocr_text
    return report


def _tree_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


if __name__ == "__main__":
    raise SystemExit(main())
