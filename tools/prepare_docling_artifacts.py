#!/usr/bin/env python3
"""Download and verify the frozen offline Docling parser artifact bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from docling.datamodel.pipeline_options import LayoutOptions
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel
from huggingface_hub import snapshot_download

try:
    from tools.artifact_download import retry_transient_download
except ModuleNotFoundError:
    from artifact_download import retry_transient_download

try:
    from docling_artifacts_verifier import verify_docling_artifacts
except ImportError:
    from rag_kb.adapters.parser.docling.artifacts import verify_docling_artifacts


_LAYOUT_REPOSITORY = "docling-project/docling-layout-heron"
_LAYOUT_REVISION = "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8"
_TABLE_REPOSITORY = "docling-project/docling-models"
_TABLE_REVISION = "fc0f2d45e2218ea24bce5045f58a389aed16dc23"
_LAYOUT_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
)
_TABLE_FILES = (
    "model_artifacts/tableformer/accurate/tableformer_accurate.safetensors",
    "model_artifacts/tableformer/accurate/tm_config.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-path", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--cache-path",
        type=Path,
        help="persistent build cache used before materializing verified artifacts",
    )
    arguments = parser.parse_args()
    artifacts_path = arguments.artifacts_path.resolve()
    if arguments.download:
        artifacts_path.mkdir(parents=True, exist_ok=True)
        cache_path = (
            arguments.cache_path.resolve()
            if arguments.cache_path is not None
            else None
        )
        _download(artifacts_path, cache_path=cache_path)
    manifest = verify_docling_artifacts(
        artifacts_path,
        arguments.manifest.resolve(),
    )
    print(
        f"verified {len(manifest.entries)} artifacts for {manifest.profile}"
    )
    return 0


def _download(artifacts_path: Path, *, cache_path: Path | None) -> None:
    layout_folder = LayoutOptions().model_spec.model_repo_folder
    _download_snapshot(
        description="Docling layout model",
        repo_id=_LAYOUT_REPOSITORY,
        revision=_LAYOUT_REVISION,
        destination=artifacts_path / layout_folder,
        files=_LAYOUT_FILES,
        cache_path=cache_path / "huggingface" if cache_path else None,
    )
    _download_snapshot(
        description="Docling table model",
        repo_id=_TABLE_REPOSITORY,
        revision=_TABLE_REVISION,
        destination=artifacts_path / "docling-project--docling-models",
        files=_TABLE_FILES,
        cache_path=cache_path / "huggingface" if cache_path else None,
    )
    rapid_ocr_destination = artifacts_path / RapidOcrModel._model_repo_folder
    rapid_ocr_download = (
        cache_path / "rapidocr" / RapidOcrModel._model_repo_folder
        if cache_path
        else rapid_ocr_destination
    )
    retry_transient_download(
        lambda: RapidOcrModel.download_models(
            backend="onnxruntime",
            local_dir=rapid_ocr_download,
            lang="chinese",
        ),
        description="RapidOCR models",
    )
    if rapid_ocr_download != rapid_ocr_destination:
        shutil.copytree(
            rapid_ocr_download,
            rapid_ocr_destination,
            dirs_exist_ok=True,
        )


def _download_snapshot(
    *,
    description: str,
    repo_id: str,
    revision: str,
    destination: Path,
    files: tuple[str, ...],
    cache_path: Path | None,
) -> None:
    if cache_path is None:
        retry_transient_download(
            lambda: snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=destination,
                allow_patterns=files,
            ),
            description=description,
        )
        return

    snapshot_path = retry_transient_download(
        lambda: snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_path,
            allow_patterns=files,
        ),
        description=description,
    )
    _copy_snapshot_files(Path(snapshot_path), destination, files)


def _copy_snapshot_files(
    snapshot_path: Path,
    destination: Path,
    files: tuple[str, ...],
) -> None:
    for relative_path in files:
        source = snapshot_path / relative_path
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


if __name__ == "__main__":
    raise SystemExit(main())
