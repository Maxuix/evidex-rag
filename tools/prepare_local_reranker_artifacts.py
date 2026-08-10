#!/usr/bin/env python3
"""Download and verify the fixed offline MiniLM reranker bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

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
        "--skip-runtime-check",
        action="store_true",
        help="verify files without checking this host's architecture/packages",
    )
    arguments = parser.parse_args()
    artifacts_path = arguments.artifacts_path.resolve()
    if arguments.download:
        artifacts_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=_REPOSITORY,
            revision=_REVISION,
            local_dir=artifacts_path,
            allow_patterns=_FILES,
        )
    manifest = verify_local_reranker_artifacts(
        artifacts_path,
        arguments.manifest.resolve(),
        verify_runtime=not arguments.skip_runtime_check,
    )
    print(f"verified {len(manifest.entries)} artifacts for {manifest.profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
