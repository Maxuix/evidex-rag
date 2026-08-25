#!/usr/bin/env python3
"""Download and verify the fixed offline MiniLM reranker bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from huggingface_hub import snapshot_download

try:
    from tools.artifact_download import retry_transient_download
except ModuleNotFoundError:
    from artifact_download import retry_transient_download

try:
    from local_reranker_artifacts_verifier import verify_local_reranker_artifacts
except ImportError:
    from rag_kb.adapters.local_reranker_artifacts import (
        verify_local_reranker_artifacts,
    )


_REPOSITORY = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
_FILES = (
    "README.md",
    "config.json",
    "onnx/model_qint8_arm64.onnx",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
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
    parser.add_argument(
        "--skip-runtime-check",
        action="store_true",
        help="verify files without checking this host's architecture/packages",
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
    manifest = verify_local_reranker_artifacts(
        artifacts_path,
        arguments.manifest.resolve(),
        verify_runtime=not arguments.skip_runtime_check,
    )
    print(f"verified {len(manifest.entries)} artifacts for {manifest.profile}")
    return 0


def _download(artifacts_path: Path, *, cache_path: Path | None) -> None:
    if cache_path is None:
        retry_transient_download(
            lambda: snapshot_download(
                repo_id=_REPOSITORY,
                revision=_REVISION,
                local_dir=artifacts_path,
                allow_patterns=_FILES,
            ),
            description="local reranker model",
        )
        return

    snapshot_path = retry_transient_download(
        lambda: snapshot_download(
            repo_id=_REPOSITORY,
            revision=_REVISION,
            cache_dir=cache_path / "huggingface",
            allow_patterns=_FILES,
        ),
        description="local reranker model",
    )
    for relative_path in _FILES:
        source = Path(snapshot_path) / relative_path
        target = artifacts_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


if __name__ == "__main__":
    raise SystemExit(main())
