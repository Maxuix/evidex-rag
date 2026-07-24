#!/usr/bin/env python3
"""Download and verify the frozen offline Docling parser artifact bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from docling.datamodel.pipeline_options import LayoutOptions
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel
from huggingface_hub import snapshot_download

try:
    from docling_artifacts_verifier import verify_docling_artifacts
except ImportError:
    from rag_kb.adapters.parser.docling.artifacts import verify_docling_artifacts


_LAYOUT_REPOSITORY = "docling-project/docling-layout-heron"
_LAYOUT_REVISION = "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8"
_TABLE_REPOSITORY = "docling-project/docling-models"
_TABLE_REVISION = "fc0f2d45e2218ea24bce5045f58a389aed16dc23"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-path", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--download", action="store_true")
    arguments = parser.parse_args()
    artifacts_path = arguments.artifacts_path.resolve()
    if arguments.download:
        artifacts_path.mkdir(parents=True, exist_ok=True)
        _download(artifacts_path)
    manifest = verify_docling_artifacts(
        artifacts_path,
        arguments.manifest.resolve(),
    )
    print(
        f"verified {len(manifest.entries)} artifacts for {manifest.profile}"
    )
    return 0


def _download(artifacts_path: Path) -> None:
    layout_folder = LayoutOptions().model_spec.model_repo_folder
    snapshot_download(
        repo_id=_LAYOUT_REPOSITORY,
        revision=_LAYOUT_REVISION,
        local_dir=artifacts_path / layout_folder,
        allow_patterns=(
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
        ),
    )
    snapshot_download(
        repo_id=_TABLE_REPOSITORY,
        revision=_TABLE_REVISION,
        local_dir=artifacts_path / "docling-project--docling-models",
        allow_patterns=(
            "model_artifacts/tableformer/accurate/tableformer_accurate.safetensors",
            "model_artifacts/tableformer/accurate/tm_config.json",
        ),
    )
    RapidOcrModel.download_models(
        backend="onnxruntime",
        local_dir=artifacts_path / RapidOcrModel._model_repo_folder,
        lang="chinese",
    )


if __name__ == "__main__":
    raise SystemExit(main())
