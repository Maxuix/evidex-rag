from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from rag_kb.adapters.local_reranker_artifacts import (
    LocalRerankerArtifactError,
    verify_local_reranker_artifacts,
)


class LocalRerankerArtifactTests(unittest.TestCase):
    def test_verifies_complete_content_addressed_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifact = root / "onnx" / "model.onnx"
            artifact.parent.mkdir()
            artifact.write_bytes(b"model")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(_manifest("onnx/model.onnx", b"model")),
                encoding="utf-8",
            )

            verified = verify_local_reranker_artifacts(
                root,
                manifest,
                verify_runtime=False,
            )

            self.assertEqual(verified.profile, "local_minilm_v1")
            self.assertEqual(len(verified.entries), 1)

    def test_rejects_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifact = root / "model.onnx"
            artifact.write_bytes(b"changed")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(_manifest("model.onnx", b"expected")),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                LocalRerankerArtifactError,
                "artifact_size|artifact_digest",
            ):
                verify_local_reranker_artifacts(
                    root,
                    manifest,
                    verify_runtime=False,
                )

    def test_rejects_parent_path(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(_manifest("../model.onnx", b"model")),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                LocalRerankerArtifactError,
                "manifest_entry_path",
            ):
                verify_local_reranker_artifacts(
                    root,
                    manifest,
                    verify_runtime=False,
                )


def _manifest(path: str, content: bytes) -> dict:
    return {
        "schema_version": 1,
        "profile": "local_minilm_v1",
        "repository": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "revision": "revision",
        "runtime": {
            "architecture": ["arm64", "aarch64"],
            "onnxruntime_version": "1.27.0",
            "tokenizers_version": "0.22.2",
        },
        "artifacts": [
            {
                "path": path,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
