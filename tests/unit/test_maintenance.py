from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID, uuid4

from rag_kb.adapters.file_store.assets import LocalIndexAssetStore
from rag_kb.config import DeploymentProfile
from rag_kb.domain import (
    FileReconciliationResult,
    IndexAssetIdentity,
    IndexAssetSnapshot,
    IndexCleanupResult,
    RetiredIndexTargetAssets,
    SourceFileMissingError,
)
from rag_kb.ports.files import IndexAssetStore
from rag_kb.services.maintenance import MaintenanceCleanupService
from tools.reset_local import (
    CONFIRMATION,
    VolumeTargets,
    compose_down_command,
    main,
    resolve_volume_targets,
    volume_remove_command,
)


_WORKSPACE_ID = UUID("01900000-0000-7000-8000-000000000601")
_NOW = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)


class LocalResetTests(unittest.TestCase):
    def test_commands_are_project_scoped_and_preserve_model_cache(self) -> None:
        self.assertEqual(
            compose_down_command(
                env_file="local.env",
                project_name="rag-kb-local",
            ),
            [
                "docker",
                "compose",
                "--env-file",
                "local.env",
                "--profile",
                "tools",
                "--project-name",
                "rag-kb-local",
                "down",
                "--remove-orphans",
            ],
        )
        self.assertEqual(
            volume_remove_command(
                ("rag-kb-local_postgres-data", "rag-kb-local_source-data")
            ),
            [
                "docker",
                "volume",
                "rm",
                "rag-kb-local_postgres-data",
                "rag-kb-local_source-data",
            ],
        )

    def test_wrong_confirmation_stops_before_external_action(self) -> None:
        with (
            patch.object(
                sys,
                "argv",
                [
                    "reset_local.py",
                    "--project-name",
                    "rag-kb-local",
                    "--confirm",
                    "wrong",
                ],
            ),
            patch("tools.reset_local.load_settings") as load,
            patch("tools.reset_local.subprocess.run") as run,
            self.assertRaises(SystemExit),
        ):
            main()
        load.assert_not_called()
        run.assert_not_called()

    def test_exact_confirmation_allows_only_development(self) -> None:
        settings = SimpleNamespace(
            app=SimpleNamespace(deployment_profile=DeploymentProfile.DEVELOPMENT)
        )
        runtime = SimpleNamespace(
            manifest=Path("local.env"),
            compose_project="rag",
            is_primary_checkout=True,
            is_canonical_manifest=True,
        )
        with (
            patch.object(
                sys,
                "argv",
                [
                    "reset_local.py",
                    "--project-name",
                    "rag",
                    "--confirm",
                    CONFIRMATION,
                ],
            ),
            patch("tools.reset_local.load_settings", return_value=settings),
            patch("tools.reset_local.resolve_local_runtime", return_value=runtime),
            patch(
                "tools.reset_local.resolve_volume_targets",
                side_effect=(
                    VolumeTargets(
                        project_name="rag",
                        remove=(
                            "rag_postgres-data",
                            "rag_source-data",
                        ),
                        preserve=("rag_inference-model-cache",),
                    ),
                    VolumeTargets(
                        project_name="rag",
                        remove=(),
                        preserve=("rag_inference-model-cache",),
                    ),
                ),
            ),
            patch("tools.reset_local.subprocess.run") as run,
        ):
            self.assertEqual(main(), 0)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            run.call_args_list[0].args[0][-2:],
            ["down", "--remove-orphans"],
        )
        self.assertEqual(
            run.call_args_list[1].args[0],
            [
                "docker",
                "volume",
                "rm",
                "rag_postgres-data",
                "rag_source-data",
            ],
        )

    def test_one_manifest_is_used_for_compose_and_application_settings(self) -> None:
        settings = SimpleNamespace(
            app=SimpleNamespace(deployment_profile=DeploymentProfile.DEVELOPMENT)
        )
        runtime = SimpleNamespace(
            manifest=Path("one.env"),
            compose_project="rag",
            is_primary_checkout=True,
            is_canonical_manifest=True,
        )
        with (
            patch.object(
                sys,
                "argv",
                [
                    "reset_local.py",
                    "--env-file",
                    "one.env",
                    "--project-name",
                    "rag",
                    "--inspect-only",
                    "--confirm",
                    CONFIRMATION,
                ],
            ),
            patch(
                "tools.reset_local.load_settings",
                return_value=settings,
            ) as load,
            patch(
                "tools.reset_local.resolve_local_runtime",
                return_value=runtime,
            ) as resolve,
            patch(
                "tools.reset_local.resolve_volume_targets",
                return_value=VolumeTargets(
                    project_name="rag",
                    remove=("rag_postgres-data",),
                    preserve=("rag_inference-model-cache",),
                ),
            ),
            patch("tools.reset_local.subprocess.run") as run,
        ):
            self.assertEqual(main(), 0)

        resolve.assert_called_once_with(
            env_file=Path("one.env"),
            require_manifest=True,
        )
        load.assert_called_once_with(env_file=Path("one.env"))
        run.assert_not_called()

    def test_reset_rejects_noncanonical_project_before_volume_resolution(self) -> None:
        runtime = SimpleNamespace(
            manifest=Path("one.env"),
            compose_project="rag",
            is_primary_checkout=True,
            is_canonical_manifest=True,
        )
        with (
            patch.object(
                sys,
                "argv",
                [
                    "reset_local.py",
                    "--project-name",
                    "rag-kb-p6-routing",
                    "--confirm",
                    CONFIRMATION,
                ],
            ),
            patch("tools.reset_local.resolve_local_runtime", return_value=runtime),
            patch("tools.reset_local.resolve_volume_targets") as resolve_targets,
            self.assertRaises(SystemExit),
        ):
            main()
        resolve_targets.assert_not_called()

    def test_volume_resolution_uses_exact_compose_labels(self) -> None:
        responses = (
            SimpleNamespace(stdout="rag-kb-local_postgres-data\n"),
            SimpleNamespace(stdout="rag-kb-local\tpostgres-data\n"),
            SimpleNamespace(stdout="rag-kb-local_source-data\n"),
            SimpleNamespace(stdout="rag-kb-local\tsource-data\n"),
            SimpleNamespace(stdout="rag-kb-local_inference-model-cache\n"),
            SimpleNamespace(stdout="rag-kb-local\tinference-model-cache\n"),
        )
        with patch(
            "tools.reset_local.subprocess.run",
            side_effect=responses,
        ) as run:
            targets = resolve_volume_targets("rag-kb-local")

        self.assertEqual(
            targets,
            VolumeTargets(
                project_name="rag-kb-local",
                remove=(
                    "rag-kb-local_postgres-data",
                    "rag-kb-local_source-data",
                ),
                preserve=("rag-kb-local_inference-model-cache",),
            ),
        )
        self.assertEqual(run.call_count, 6)
        for volume_key, call in zip(
            (
                "postgres-data",
                "source-data",
                "inference-model-cache",
            ),
            run.call_args_list[::2],
            strict=True,
        ):
            self.assertIn(
                f"label=com.docker.compose.volume={volume_key}",
                call.args[0],
            )


class MaintenanceCleanupServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_size_counts_targets_and_all_target_assets_are_deleted(self) -> None:
        first_target = uuid4()
        second_target = uuid4()
        ignored_target = uuid4()
        repository = _IndexingRepository(
            (
                _target(first_target, asset_count=3),
                _target(second_target),
                _target(ignored_target, asset_count=1),
            )
        )
        store = _AssetStore()
        service = _service(repository, asset_store=store, batch_size=2)

        result = await service.run_once(now=_NOW)

        self.assertEqual(repository.list_limits, [2])
        self.assertEqual(
            repository.cleanup_target_ids,
            [(first_target, second_target)],
        )
        self.assertEqual(
            [identity.indexed_document_version_id for identity in store.deletes],
            [first_target, first_target, first_target],
        )
        self.assertEqual(result.index.retired_targets_cleaned, 2)
        cleanup_position = repository.events.index("database_cleanup")
        self.assertTrue(
            all(
                repository.events.index(f"delete:{identity.asset_key}")
                < cleanup_position
                for identity in store.deletes
            )
        )

    async def test_file_failure_preserves_target_and_retention_still_runs(self) -> None:
        failed_target = uuid4()
        successful_target = uuid4()
        failed = _target(failed_target, asset_count=3)
        successful = _target(successful_target, asset_count=1)
        failed_key = failed.assets[1].storage_uri.rsplit("/", 1)[-1]
        repository = _IndexingRepository((failed, successful))
        store = _AssetStore(failed_keys={failed_key})
        service = _service(repository, asset_store=store)

        result = await service.run_once(now=_NOW)

        self.assertEqual(
            [identity.indexed_document_version_id for identity in store.deletes],
            [failed_target, failed_target, failed_target, successful_target],
        )
        self.assertEqual(repository.cleanup_target_ids, [(successful_target,)])
        self.assertEqual(repository.cleanup_record_deletions, 1)
        self.assertEqual(result.index.jobs_deleted, 1)
        self.assertEqual(result.index.file_cleanup_records_deleted, 1)

    async def test_without_asset_store_only_asset_free_target_is_cleaned(self) -> None:
        asset_target = uuid4()
        text_target = uuid4()
        repository = _IndexingRepository(
            (
                _target(asset_target, asset_count=1),
                _target(text_target),
            )
        )
        service = _service(repository, asset_store=None)

        result = await service.run_once(now=_NOW)

        self.assertEqual(repository.cleanup_target_ids, [(text_target,)])
        self.assertEqual(result.index.retired_targets_cleaned, 1)
        self.assertEqual(result.index.jobs_deleted, 1)
        self.assertEqual(result.index.file_cleanup_records_deleted, 1)

    async def test_database_failure_replays_already_deleted_files_idempotently(self) -> None:
        target_id = uuid4()
        repository = _IndexingRepository(
            (_target(target_id, asset_count=1),),
            cleanup_failures=1,
        )
        store = _AssetStore()
        service = _service(repository, asset_store=store)

        with self.assertRaisesRegex(RuntimeError, "database cleanup failed"):
            await service.run_once(now=_NOW)
        result = await service.run_once(now=_NOW)

        self.assertEqual(len(store.deletes), 2)
        self.assertEqual(
            [identity.indexed_document_version_id for identity in store.deletes],
            [target_id, target_id],
        )
        self.assertEqual(repository.cleanup_target_ids, [(target_id,), (target_id,)])
        self.assertEqual(result.index.retired_targets_cleaned, 1)

    async def test_real_local_asset_is_deleted_before_retired_target_cleanup(
        self,
    ) -> None:
        target_id = uuid4()
        content = b"retired derived image"
        checksum = hashlib.sha256(content).hexdigest()
        identity = IndexAssetIdentity(_WORKSPACE_ID, target_id, checksum)
        snapshot = _asset_snapshot(target_id, identity.storage_uri, checksum)
        repository = _IndexingRepository(
            (RetiredIndexTargetAssets(target_id, (snapshot,)),)
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "asset-staging"
            final = root / "assets"
            staging.mkdir()
            final.mkdir()
            store = LocalIndexAssetStore(staging, final)
            await store.put(identity, content, checksum)

            result = await _service(repository, asset_store=store).run_once(
                now=_NOW,
            )

            with self.assertRaises(SourceFileMissingError):
                await store.read(identity)

        self.assertEqual(repository.cleanup_target_ids, [(target_id,)])
        self.assertEqual(result.index.retired_targets_cleaned, 1)

    async def test_invalid_local_asset_uri_preserves_target_and_other_files(
        self,
    ) -> None:
        target_id = uuid4()
        retained_target_id = uuid4()
        content = b"retained derived image"
        checksum = hashlib.sha256(content).hexdigest()
        retained = IndexAssetIdentity(
            _WORKSPACE_ID,
            retained_target_id,
            checksum,
        )
        invalid = _asset_snapshot(
            target_id,
            (
                f"local-index-asset://{_WORKSPACE_ID}/{target_id}/"
                "../escaped"
            ),
            checksum,
        )
        repository = _IndexingRepository(
            (RetiredIndexTargetAssets(target_id, (invalid,)),)
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "asset-staging"
            final = root / "assets"
            staging.mkdir()
            final.mkdir()
            store = LocalIndexAssetStore(staging, final)
            await store.put(retained, content, checksum)

            result = await _service(repository, asset_store=store).run_once(
                now=_NOW,
            )

            self.assertEqual(await store.read(retained), content)

        self.assertEqual(repository.cleanup_target_ids, [()])
        self.assertEqual(result.index.retired_targets_cleaned, 0)


class _FileReconciliation:
    async def run_once(self, *, now):
        return FileReconciliationResult()


class _AssetStore:
    def __init__(self, *, failed_keys: set[str] | None = None) -> None:
        self.failed_keys = failed_keys or set()
        self.deletes: list[IndexAssetIdentity] = []
        self.events: list[str] | None = None

    @staticmethod
    def parse_uri(storage_uri: str) -> IndexAssetIdentity:
        prefix = "local-index-asset://"
        workspace_id, target_id, asset_key = storage_uri.removeprefix(prefix).split("/")
        return IndexAssetIdentity(
            workspace_id=UUID(workspace_id),
            indexed_document_version_id=UUID(target_id),
            asset_key=asset_key,
        )

    async def delete(self, identity: IndexAssetIdentity) -> None:
        self.deletes.append(identity)
        if self.events is not None:
            self.events.append(f"delete:{identity.asset_key}")
        if identity.asset_key in self.failed_keys:
            raise OSError("simulated asset deletion failure")


class _IndexingRepository:
    def __init__(
        self,
        targets: tuple[RetiredIndexTargetAssets, ...],
        *,
        cleanup_failures: int = 0,
    ) -> None:
        self.targets = targets
        self.cleanup_failures = cleanup_failures
        self.list_limits: list[int] = []
        self.cleanup_target_ids: list[tuple[UUID, ...]] = []
        self.cleanup_record_deletions = 0
        self.events: list[str] = []

    async def list_retired_target_assets(self, *, data_before, limit):
        self.list_limits.append(limit)
        return self.targets[:limit]

    async def cleanup_retired(
        self,
        *,
        target_ids,
        data_before,
        tasks_before,
        limit,
    ):
        self.cleanup_target_ids.append(target_ids)
        self.events.append("database_cleanup")
        if self.cleanup_failures:
            self.cleanup_failures -= 1
            raise RuntimeError("database cleanup failed")
        return IndexCleanupResult(
            retired_targets_cleaned=len(target_ids),
            jobs_deleted=1,
        )


class _FileConsistencyRepository:
    def __init__(self, indexing: _IndexingRepository) -> None:
        self._indexing = indexing

    async def delete_expired_cleanup_records(self, *, before, limit):
        self._indexing.cleanup_record_deletions += 1
        return 1


class _UnitOfWork:
    def __init__(self, indexing: _IndexingRepository) -> None:
        self.workspace_id = _WORKSPACE_ID
        self.indexing = indexing
        self.file_consistency = _FileConsistencyRepository(indexing)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exception_type, exception, traceback):
        return None

    async def commit(self):
        return None


class _UnitOfWorkFactory:
    def __init__(self, indexing: _IndexingRepository) -> None:
        self._indexing = indexing

    def __call__(self, *, mode=None):
        del mode
        return _UnitOfWork(self._indexing)


def _service(
    repository: _IndexingRepository,
    *,
    asset_store: IndexAssetStore | None,
    batch_size: int = 10,
) -> MaintenanceCleanupService:
    if isinstance(asset_store, _AssetStore):
        asset_store.events = repository.events
    return MaintenanceCleanupService(
        _UnitOfWorkFactory(repository),
        _FileReconciliation(),
        _WORKSPACE_ID,
        batch_size=batch_size,
        retired_data_grace_seconds=300,
        task_retention_seconds=600,
        asset_store=asset_store,
    )


def _target(
    target_id: UUID,
    *,
    asset_count: int = 0,
) -> RetiredIndexTargetAssets:
    return RetiredIndexTargetAssets(
        indexed_document_version_id=target_id,
        assets=tuple(_asset(target_id, ordinal) for ordinal in range(asset_count)),
    )


def _asset(target_id: UUID, ordinal: int) -> IndexAssetSnapshot:
    asset_key = f"{ordinal + 1:064x}"
    return IndexAssetSnapshot(
        id=uuid4(),
        workspace_id=_WORKSPACE_ID,
        kb_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        indexed_document_version_id=target_id,
        storage_uri=(
            f"local-index-asset://{_WORKSPACE_ID}/{target_id}/{asset_key}"
        ),
        media_type="image/png",
        checksum_sha256=asset_key,
    )


def _asset_snapshot(
    target_id: UUID,
    storage_uri: str,
    checksum_sha256: str,
) -> IndexAssetSnapshot:
    return IndexAssetSnapshot(
        id=uuid4(),
        workspace_id=_WORKSPACE_ID,
        kb_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        indexed_document_version_id=target_id,
        storage_uri=storage_uri,
        media_type="image/png",
        checksum_sha256=checksum_sha256,
    )


if __name__ == "__main__":
    unittest.main()
