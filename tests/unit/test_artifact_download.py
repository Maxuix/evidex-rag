from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from tools.artifact_download import retry_transient_download


class ArtifactDownloadRetryTests(unittest.TestCase):
    @patch("tools.artifact_download.time.sleep")
    def test_retries_nested_transient_transport_failures(self, sleep) -> None:
        calls = 0

        def operation() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                try:
                    raise httpx.ConnectError("temporary TLS failure")
                except httpx.ConnectError as error:
                    raise FileNotFoundError("snapshot unavailable") from error
            return "ready"

        result = retry_transient_download(operation, description="test model")

        self.assertEqual(result, "ready")
        self.assertEqual(calls, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])

    @patch("tools.artifact_download.time.sleep")
    def test_does_not_retry_non_network_failures(self, sleep) -> None:
        with self.assertRaisesRegex(ValueError, "invalid manifest"):
            retry_transient_download(
                lambda: (_ for _ in ()).throw(ValueError("invalid manifest")),
                description="test model",
            )

        sleep.assert_not_called()

    @patch("tools.artifact_download.time.sleep")
    def test_retry_count_is_bounded(self, sleep) -> None:
        calls = 0

        def operation() -> None:
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("still unavailable")

        with self.assertRaises(httpx.ReadTimeout):
            retry_transient_download(operation, description="test model")

        self.assertEqual(calls, 7)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [2, 4, 8, 16, 30, 30],
        )


class ArtifactCacheMaterializationTests(unittest.TestCase):
    def test_docling_snapshot_cache_materializes_only_requested_files(
        self,
    ) -> None:
        from tools import prepare_docling_artifacts

        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            snapshot = root / "snapshot"
            requested = ("config.json", "nested/model.bin")
            for relative_path in (*requested, "ignored.bin"):
                path = snapshot / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative_path, encoding="utf-8")
            destination = root / "destination"
            cache = root / "cache"

            with patch.object(
                prepare_docling_artifacts,
                "snapshot_download",
                return_value=str(snapshot),
            ) as download:
                prepare_docling_artifacts._download_snapshot(
                    description="test snapshot",
                    repo_id="example/repository",
                    revision="fixed-revision",
                    destination=destination,
                    files=requested,
                    cache_path=cache,
                )

            self.assertEqual(
                (destination / "config.json").read_text(encoding="utf-8"),
                "config.json",
            )
            self.assertEqual(
                (destination / "nested/model.bin").read_text(encoding="utf-8"),
                "nested/model.bin",
            )
            self.assertFalse((destination / "ignored.bin").exists())
            download.assert_called_once_with(
                repo_id="example/repository",
                revision="fixed-revision",
                cache_dir=cache,
                allow_patterns=requested,
            )

    def test_reranker_snapshot_cache_materializes_locked_file_set(self) -> None:
        from tools import prepare_local_reranker_artifacts

        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            snapshot = root / "snapshot"
            for relative_path in prepare_local_reranker_artifacts._FILES:
                path = snapshot / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative_path, encoding="utf-8")
            destination = root / "destination"
            cache = root / "cache"

            with patch.object(
                prepare_local_reranker_artifacts,
                "snapshot_download",
                return_value=str(snapshot),
            ) as download:
                prepare_local_reranker_artifacts._download(
                    destination,
                    cache_path=cache,
                )

            self.assertEqual(
                sorted(
                    path.relative_to(destination).as_posix()
                    for path in destination.rglob("*")
                    if path.is_file()
                ),
                sorted(prepare_local_reranker_artifacts._FILES),
            )
            download.assert_called_once_with(
                repo_id=prepare_local_reranker_artifacts._REPOSITORY,
                revision=prepare_local_reranker_artifacts._REVISION,
                cache_dir=cache / "huggingface",
                allow_patterns=prepare_local_reranker_artifacts._FILES,
            )
