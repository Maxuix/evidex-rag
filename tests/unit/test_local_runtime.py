from __future__ import annotations

import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from tools.local_runtime import (
    CANONICAL_COMPOSE_PROJECT,
    LocalRuntimeError,
    build_doctor_report,
    main,
    resolve_local_runtime,
)


_COMPLETE_MANIFEST = """\
COMPOSE_PROJECT_NAME=rag
RAG_KB_API_PORT=18000
RAG_KB_FRONTEND_PORT=13000
RAG_KB_POSTGRES_PORT=15432
RAG_KB_FALKORDB_PORT=16379
POSTGRES_ADMIN_PASSWORD=admin-secret
RAG_KB_MIGRATION_PASSWORD=migration-secret
RAG_KB_RUNTIME_PASSWORD=runtime-secret
RAG_KB__APP__BIND_HOST=127.0.0.1
RAG_KB__IDENTITY__PRINCIPAL_ID=development-principal
RAG_KB__IDENTITY__CLIENT_ID=development-web
RAG_KB__IDENTITY__WORKSPACE_ID=01900000-0000-7000-8000-000000000001
RAG_KB__DATABASE__RUNTIME_DSN=postgresql+asyncpg://user:secret@postgres/rag_kb
RAG_KB__DATABASE__MIGRATION_DSN=postgresql+asyncpg://user:secret@postgres/rag_kb
"""


class LocalRuntimeTests(unittest.TestCase):
    def _manifest(self, root: Path, content: str = _COMPLETE_MANIFEST) -> Path:
        path = root / ".env.local"
        path.write_text(content, encoding="utf-8")
        path.chmod(0o600)
        return path

    def test_resolves_only_safe_identity_from_one_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)

            runtime = resolve_local_runtime(
                checkout=root,
                canonical_checkout=root,
                env_file=manifest,
                require_manifest=True,
            )

        self.assertEqual(runtime.compose_project, CANONICAL_COMPOSE_PROJECT)
        self.assertEqual(runtime.api_port, 18000)
        self.assertEqual(runtime.frontend_port, 13000)
        self.assertEqual(runtime.postgres_port, 15432)
        self.assertEqual(runtime.falkordb_port, 16379)
        self.assertNotIn("secret", repr(runtime))

    def test_rejects_project_switch_and_duplicate_or_invalid_ports(self) -> None:
        cases = (
            _COMPLETE_MANIFEST.replace("COMPOSE_PROJECT_NAME=rag", "COMPOSE_PROJECT_NAME=p6"),
            _COMPLETE_MANIFEST + "RAG_KB_API_PORT=8001\n",
            _COMPLETE_MANIFEST.replace("RAG_KB_API_PORT=18000", "RAG_KB_API_PORT=not-a-port"),
            _COMPLETE_MANIFEST.replace("RAG_KB_API_PORT=18000", "RAG_KB_API_PORT=13000"),
        )
        for content in cases:
            with self.subTest(content=content.splitlines()[-1]):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    manifest = self._manifest(root, content)
                    with self.assertRaises(LocalRuntimeError):
                        resolve_local_runtime(
                            checkout=root,
                            canonical_checkout=root,
                            env_file=manifest,
                            require_manifest=True,
                        )

    def test_doctor_reports_drift_without_values_or_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(
                root,
                "POSTGRES_ADMIN_PASSWORD=admin-secret\n"
                "RAG_KB_MIGRATION_PASSWORD=migration-secret\n"
                "RAG_KB_RUNTIME_PASSWORD=runtime-secret\n"
                "RAG_KB_ENV_FILE=/private/legacy.env\n",
            )
            (root / ".env").write_text("TOKEN=legacy-secret\n", encoding="utf-8")
            runtime_dir = root / ".runtime"
            runtime_dir.mkdir()
            (runtime_dir / "routing-rag-current-source.override.yaml").write_text(
                "services: {}\n",
                encoding="utf-8",
            )
            linked = root / "linked"
            runtime = resolve_local_runtime(
                checkout=linked,
                canonical_checkout=root,
                env_file=manifest,
            )

            report = build_doctor_report(
                runtime,
                active_compose_projects=("rag", "rag-kb-p6-routing"),
            )

        rendered = json.dumps(report, sort_keys=True)
        self.assertEqual(report["status"], "fail")
        self.assertIn("linked_worktree", rendered)
        self.assertIn("legacy_app_env_present", rendered)
        self.assertIn("legacy_state_indirection", rendered)
        self.assertIn("noncanonical_compose_project_active", rendered)
        self.assertNotIn("admin-secret", rendered)
        self.assertNotIn("legacy-secret", rendered)
        self.assertNotIn("private", rendered)
        self.assertNotIn(str(root), rendered)

    def test_doctor_passes_complete_private_primary_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = resolve_local_runtime(
                checkout=root,
                canonical_checkout=root,
                env_file=self._manifest(root),
                require_manifest=True,
            )

            report = build_doctor_report(
                runtime,
                active_compose_projects=("rag",),
            )

        self.assertEqual(report["status"], "pass")
        self.assertTrue(all(item["status"] == "pass" for item in report["checks"]))

    def test_cli_output_is_content_safe_when_manifest_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root, "TOKEN=do-not-print\n")
            stdout = io.StringIO()
            with (
                patch("tools.local_runtime.PROJECT_ROOT", root),
                patch("tools.local_runtime.discover_canonical_checkout", return_value=root),
                patch("tools.local_runtime.collect_active_compose_projects", return_value=()),
                patch("sys.argv", ["local-runtime", "doctor", "--env-file", str(manifest)]),
                patch("sys.stdout", stdout),
            ):
                self.assertEqual(main(), 1)

        rendered = stdout.getvalue()
        self.assertNotIn("do-not-print", rendered)
        self.assertNotIn(str(root), rendered)
        self.assertEqual(json.loads(rendered)["status"], "fail")

    def test_manifest_must_be_private_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            manifest.chmod(0o644)
            runtime = resolve_local_runtime(
                checkout=root,
                canonical_checkout=root,
                env_file=manifest,
            )
            report = build_doctor_report(runtime, active_compose_projects=())
            permissions = next(
                item for item in report["checks"] if item["name"] == "manifest_permissions"
            )
            self.assertEqual(permissions["status"], "fail")
            self.assertEqual(stat.S_IMODE(manifest.stat().st_mode), 0o644)

    def test_manifest_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.env"
            target.write_text(_COMPLETE_MANIFEST, encoding="utf-8")
            target.chmod(0o600)
            manifest = root / ".env.local"
            manifest.symlink_to(target)

            with self.assertRaises(LocalRuntimeError):
                resolve_local_runtime(
                    checkout=root,
                    canonical_checkout=root,
                    env_file=manifest,
                    require_manifest=True,
                )


if __name__ == "__main__":
    unittest.main()
