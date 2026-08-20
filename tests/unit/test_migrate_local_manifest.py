from __future__ import annotations

import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from tools.local_runtime import LocalRuntimeError, _parse_manifest
from tools.migrate_local_manifest import (
    BACKUP_NAME,
    apply_migration,
    build_migration,
    main,
)


ROOT = Path(__file__).resolve().parents[2]


class MigrateLocalManifestTests(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, Path]:
        manifest = root / ".env.local"
        manifest.write_text(
            "RAG_KB_ENV_FILE=/legacy/.env\n"
            "POSTGRES_ADMIN_PASSWORD=admin-state\n"
            "RAG_KB_MIGRATION_PASSWORD=migration-state\n"
            "RAG_KB_RUNTIME_PASSWORD=runtime-state\n",
            encoding="utf-8",
        )
        manifest.chmod(0o600)
        legacy = root / ".env"
        legacy.write_text(
            "RAG_KB__DATABASE__RUNTIME_DSN="
            "postgresql+asyncpg://rag_kb_runtime:old@postgres:5432/rag_kb\n"
            "RAG_KB__DATABASE__MIGRATION_DSN="
            "postgresql+asyncpg://rag_kb_migration:old@postgres:5432/rag_kb\n",
            encoding="utf-8",
        )
        legacy.chmod(0o600)
        (root / ".env.example").write_bytes((ROOT / ".env.example").read_bytes())
        return manifest, legacy

    def test_preview_is_read_only_and_contains_no_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self._inputs(root)
            before = manifest.read_bytes()

            migration = build_migration(root, canonical_checkout=root)
            summary = migration.safe_summary(applied=False)

            self.assertEqual(manifest.read_bytes(), before)
            self.assertFalse((root / BACKUP_NAME).exists())
            rendered = json.dumps(summary, sort_keys=True)
            self.assertEqual(summary["status"], "ready")
            self.assertNotIn("runtime-state", rendered)
            self.assertNotIn("old", rendered)
            self.assertNotIn(str(root), rendered)

    def test_apply_creates_private_backup_and_aligned_single_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, legacy = self._inputs(root)
            before = manifest.read_bytes()
            migration = build_migration(root, canonical_checkout=root)

            apply_migration(migration)

            backup = root / BACKUP_NAME
            self.assertEqual(backup.read_bytes(), before)
            self.assertEqual(stat.S_IMODE(backup.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(manifest.stat().st_mode), 0o600)
            values = _parse_manifest(manifest)
            self.assertNotIn("RAG_KB_ENV_FILE", values)
            self.assertEqual(values["COMPOSE_PROJECT_NAME"], "rag")
            self.assertIn("runtime-state", values["RAG_KB__DATABASE__RUNTIME_DSN"])
            self.assertIn(
                "migration-state",
                values["RAG_KB__DATABASE__MIGRATION_DSN"],
            )
            self.assertTrue(legacy.exists())

    def test_existing_backup_blocks_apply_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self._inputs(root)
            migration = build_migration(root, canonical_checkout=root)
            backup = root / BACKUP_NAME
            backup.write_text("existing\n", encoding="utf-8")
            before = manifest.read_bytes()

            with self.assertRaises(FileExistsError):
                apply_migration(migration)

            self.assertEqual(manifest.read_bytes(), before)
            self.assertEqual(backup.read_text(encoding="utf-8"), "existing\n")

    def test_legacy_non_application_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, legacy = self._inputs(root)
            legacy.write_text("PROVIDER_KEY=must-not-migrate\n", encoding="utf-8")

            with self.assertRaises(LocalRuntimeError):
                build_migration(root, canonical_checkout=root)

    def test_cli_failure_is_generic_and_content_safe(self) -> None:
        stdout = io.StringIO()
        with (
            patch(
                "tools.migrate_local_manifest.build_migration",
                side_effect=LocalRuntimeError("secret-context"),
            ),
            patch("sys.argv", ["migrate-local-manifest"]),
            patch("sys.stdout", stdout),
        ):
            self.assertEqual(main(), 1)

        rendered = stdout.getvalue()
        self.assertNotIn("secret-context", rendered)
        self.assertEqual(json.loads(rendered)["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
