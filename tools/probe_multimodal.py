#!/usr/bin/env python3
"""Read-only capability probe for the locked local multimodal stack."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import shutil
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--docx", type=Path)
    arguments = parser.parse_args()

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
    if arguments.docx is not None:
        report["docx_fixture"] = _fixture(arguments.docx, ".docx")
    print(json.dumps(report, indent=2, sort_keys=True))

    tools = report["executables"]
    pdf_contract = report["pdf_contract"]
    assert isinstance(tools, dict) and isinstance(pdf_contract, dict)
    return 0 if all(tools.values()) and all(pdf_contract.values()) else 1


def _fixture(path: Path, suffix: str) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if resolved.suffix.lower() != suffix or not resolved.is_file():
        raise ValueError(f"fixture must be a {suffix} file")
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size}


if __name__ == "__main__":
    raise SystemExit(main())
