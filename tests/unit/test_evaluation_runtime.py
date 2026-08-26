from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import stat
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from uuid import UUID

from tools import evaluation_runtime as module
from tools.evaluation_runtime import (
    AdaptiveGraphIdentity,
    EvaluationRuntime,
    EvaluationRuntimeError,
    compose_command,
    load_evaluation_runtime,
)
from tools.evaluate_agent_complex_qa import _parser as complex_parser
from tools.evaluate_multimodal_real import _parser as multimodal_parser
from tools.local_runtime import _parse_manifest
from tools.run_adaptive_graph_r4 import _parser as r4_parser
from tools.run_adaptive_graph_r7_stage_a import _parser as r7_parser


_OWNER = "0123456789abcdef0123456789abcdef"
_IDENTITY = {
    "workspace_id": "01900000-0000-7000-8000-000000000001",
    "knowledge_base_id": "01900000-0000-7000-8000-000000000002",
    "index_revision_id": "01900000-0000-7000-8000-000000000003",
    "graph_build_id": "01900000-0000-7000-8000-000000000004",
    "answer_profile_revision_id": "01900000-0000-7000-8000-000000000005",
    "judge_profile_revision_id": "01900000-0000-7000-8000-000000000006",
}
_PROFILE_IDENTITY = {
    "schema_profile_key": "software_knowledge_v1",
    "schema_profile_digest": "6cae93809f060d21f0c85ba04cde955abdc5259fd93e1b1757fb7445d51eaf38",
    "extractor_version": "graphiti_v4",
}


class EvaluationRuntimeTests(unittest.TestCase):
    def _runtime_files(
        self,
        root: Path,
        *,
        changes: dict[str, object] | None = None,
        mode: int = 0o600,
    ) -> Path:
        root.mkdir(mode=0o700)
        root.chmod(0o700)
        env = root / "runtime.env"
        env.write_text("COMPOSE_PROJECT_NAME=rag-eval\n", encoding="utf-8")
        env.chmod(0o600)
        compose_env = root / "compose.env"
        compose_env.write_text("COMPOSE_PROJECT_NAME=rag-eval\n", encoding="utf-8")
        compose_env.chmod(0o600)
        value: dict[str, object] = {
            "schema_version": 1,
            "compose_project": "rag-eval",
            "owner": _OWNER,
            "build_revision": "a" * 40,
            "env_file": "runtime.env",
            "compose_env_file": "compose.env",
            "api_base_url": "http://127.0.0.1:28000/api/v1",
            "ports": {
                "api": 28000,
                "frontend": 23000,
                "postgres": 25432,
                "falkordb": 26379,
            },
            "adaptive_graph": {**_IDENTITY, **_PROFILE_IDENTITY},
        }
        value.update(changes or {})
        manifest = root / "runtime.json"
        manifest.write_text(json.dumps(value), encoding="utf-8")
        manifest.chmod(mode)
        return manifest

    def test_loads_private_closed_isolated_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rag-eval"
            manifest = self._runtime_files(root)
            with patch.object(module, "DEFAULT_RUNTIME_ROOT", root):
                runtime = load_evaluation_runtime(
                    manifest,
                    require_adaptive_graph=True,
                )

        self.assertEqual(runtime.owner, _OWNER)
        self.assertEqual(runtime.api_base_url, "http://127.0.0.1:28000/api/v1")
        self.assertEqual(runtime.adaptive_graph.knowledge_base_id, UUID(_IDENTITY["knowledge_base_id"]))
        self.assertNotIn("secret", repr(runtime))

    def test_read_only_loader_can_accept_git_canonical_checkout_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "primary" / ".runtime/evaluations/rag-eval"
            root.parent.mkdir(parents=True)
            manifest = self._runtime_files(root)
            with (
                patch.object(module, "DEFAULT_RUNTIME_ROOT", Path(directory) / "linked"),
                patch.object(
                    module,
                    "canonical_evaluation_runtime_manifest",
                    return_value=manifest,
                ),
            ):
                with self.assertRaises(EvaluationRuntimeError):
                    load_evaluation_runtime(manifest)
                runtime = load_evaluation_runtime(
                    manifest,
                    require_adaptive_graph=True,
                    allow_canonical_checkout=True,
                )

        self.assertEqual(runtime.owner, _OWNER)

    def test_rejects_personal_project_ports_api_and_open_permissions(self) -> None:
        changes = (
            {"compose_project": "rag"},
            {
                "ports": {
                    "api": 8000,
                    "frontend": 23000,
                    "postgres": 25432,
                    "falkordb": 26379,
                },
                "api_base_url": "http://127.0.0.1:8000/api/v1",
            },
            {"api_base_url": "http://example.com:28000/api/v1"},
            {"owner": "not-an-owner"},
        )
        for change in changes:
            with self.subTest(change=change):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory) / "rag-eval"
                    manifest = self._runtime_files(root, changes=change)
                    with patch.object(module, "DEFAULT_RUNTIME_ROOT", root):
                        with self.assertRaises(EvaluationRuntimeError):
                            load_evaluation_runtime(manifest)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rag-eval"
            manifest = self._runtime_files(root, mode=0o644)
            with patch.object(module, "DEFAULT_RUNTIME_ROOT", root):
                with self.assertRaises(EvaluationRuntimeError):
                    load_evaluation_runtime(manifest)

    def test_rejects_symlink_manifest_and_missing_adaptive_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rag-eval"
            target = self._runtime_files(root)
            link = root / "linked.json"
            link.symlink_to(target)
            with patch.object(module, "DEFAULT_RUNTIME_ROOT", root):
                with self.assertRaises(EvaluationRuntimeError):
                    load_evaluation_runtime(link)
                target.write_text(
                    json.dumps(
                        {
                            **json.loads(target.read_text(encoding="utf-8")),
                            "adaptive_graph": None,
                        }
                    ),
                    encoding="utf-8",
                )
                target.chmod(0o600)
                with self.assertRaises(EvaluationRuntimeError):
                    load_evaluation_runtime(target, require_adaptive_graph=True)

    def test_rejects_unknown_graph_schema_profile_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rag-eval"
            manifest = self._runtime_files(
                root,
                changes={
                    "adaptive_graph": {
                        **_IDENTITY,
                        **_PROFILE_IDENTITY,
                        "schema_profile_key": "unknown_profile",
                    }
                },
            )
            with patch.object(module, "DEFAULT_RUNTIME_ROOT", root):
                with self.assertRaises(EvaluationRuntimeError):
                    load_evaluation_runtime(manifest, require_adaptive_graph=True)

    def test_compose_command_fixes_project_and_both_files(self) -> None:
        runtime = EvaluationRuntime(
            manifest=Path("/private/runtime.json"),
            runtime_root=Path("/private"),
            env_file=Path("/private/runtime.env"),
            compose_env_file=Path("/private/compose.env"),
            owner=_OWNER,
            build_revision="a" * 40,
            api_base_url="http://127.0.0.1:28000/api/v1",
            ports=dict(module.EVALUATION_PORTS),
            adaptive_graph=AdaptiveGraphIdentity(
                **{name: UUID(value) for name, value in _IDENTITY.items()}
            ),
        )

        command = compose_command(runtime, "up", "api")

        self.assertEqual(command[3], "/private/compose.env")
        self.assertEqual(command[-4:], ["--project-name", "rag-eval", "up", "api"])
        self.assertEqual(command.count("--file"), 2)
        self.assertEqual(command[-3], "rag-eval")

    def test_evaluator_clis_expose_only_runtime_or_offline_entrypoints(self) -> None:
        for parser in (r4_parser(), r7_parser(), complex_parser(), multimodal_parser()):
            with self.subTest(program=parser.prog):
                help_text = parser.format_help()
                self.assertIn("--evaluation-runtime", help_text)
                self.assertIn("--dry-run", help_text)
                self.assertNotIn("--api", help_text)
                self.assertNotIn("--env-file", help_text)
        self.assertTrue(r7_parser().parse_args(["--host-worker"]).host_worker)

    def test_eval_env_replaces_database_identity_and_drops_legacy_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / ".env.local"
            canonical.write_text(
                "COMPOSE_PROJECT_NAME=rag\n"
                "POSTGRES_ADMIN_PASSWORD=personal-admin\n"
                "RAG_KB_MIGRATION_PASSWORD=personal-migration\n"
                "RAG_KB_RUNTIME_PASSWORD=personal-runtime\n"
                "RAG_KB__DATABASE__RUNTIME_DSN=personal-runtime-dsn\n"
                "RAG_KB__DATABASE__MIGRATION_DSN=personal-migration-dsn\n"
                "RAG_KB__IDENTITY__WORKSPACE_ID=01900000-0000-7000-8000-000000000099\n"
                "RAG_KB__MODEL_PROVIDER__API_KEY=legacy-provider-secret\n"
                "RAG_KB__RETRIEVAL__MAX_CANDIDATE_COUNT=80\n",
                encoding="utf-8",
            )
            canonical.chmod(0o600)
            output = root / "runtime.env"
            with patch.object(module, "CANONICAL_MANIFEST", canonical):
                host_rendered = module._evaluation_env(
                    owner=_OWNER,
                    workspace_id=UUID(_IDENTITY["workspace_id"]),
                    host_access=True,
                )
                compose_rendered = module._evaluation_env(
                    owner=_OWNER,
                    workspace_id=UUID(_IDENTITY["workspace_id"]),
                    host_access=False,
                )
                output.write_bytes(host_rendered)
            host_values = _parse_manifest(output)
            output.write_bytes(compose_rendered)
            compose_values = _parse_manifest(output)

        self.assertEqual(host_values["COMPOSE_PROJECT_NAME"], "rag-eval")
        self.assertEqual(host_values["RAG_KB__IDENTITY__WORKSPACE_ID"], _IDENTITY["workspace_id"])
        self.assertEqual(host_values["RAG_KB__IDENTITY__PRINCIPAL_ID"], f"eval-{_OWNER}")
        self.assertNotIn("RAG_KB__MODEL_PROVIDER__API_KEY", host_values)
        self.assertNotIn("TIKTOKEN_CACHE_DIR", host_values)
        self.assertNotIn(b"TIKTOKEN_CACHE_DIR", host_rendered)
        self.assertNotIn(b"personal-", host_rendered)
        self.assertEqual(host_values["RAG_KB__RETRIEVAL__MAX_CANDIDATE_COUNT"], "80")
        self.assertIn("127.0.0.1:25432", host_values["RAG_KB__DATABASE__RUNTIME_DSN"])
        self.assertIn("postgres:5432", compose_values["RAG_KB__DATABASE__RUNTIME_DSN"])
        self.assertNotIn("RAG_KB__FILE_STORE__ROOT_PATH", compose_values)

    def test_private_archive_extracts_regular_files_and_rejects_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            safe_archive = root / "safe.tar.gz"
            with tarfile.open(safe_archive, "w:gz") as archive:
                member = tarfile.TarInfo("nested/payload")
                member.size = 4
                archive.addfile(member, BytesIO(b"safe"))
            target = root / "safe"
            module._extract_private_archive(safe_archive, target)
            payload = target / "nested/payload"
            self.assertEqual(payload.read_bytes(), b"safe")
            self.assertEqual(stat.S_IMODE(payload.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(payload.parent.stat().st_mode), 0o700)

            unsafe_archive = root / "unsafe.tar.gz"
            with tarfile.open(unsafe_archive, "w:gz") as archive:
                link = tarfile.TarInfo("linked")
                link.type = tarfile.SYMTYPE
                link.linkname = "/tmp/outside"
                archive.addfile(link)
            unsafe_target = root / "unsafe"
            with self.assertRaises(EvaluationRuntimeError):
                module._extract_private_archive(unsafe_archive, unsafe_target)
            self.assertFalse(unsafe_target.exists())

    def test_seed_adaptive_identity_comes_from_frozen_r7_result(self) -> None:
        for schema_version in (
            "adaptive_graph_r7_stage_a_v2",
            "adaptive_graph_r7_stage_a_v3",
        ):
            with self.subTest(schema_version=schema_version):
                with tempfile.TemporaryDirectory() as directory:
                    seed = Path(directory)
                    archive_path = seed / "adaptive-graph-route-v2.tar.gz"
                    payload = json.dumps(
                        {
                            "status": "completed",
                            "runtime": {
                                "schema_version": schema_version,
                                "dataset_id": "routing-rag-v2",
                                "knowledge_base_id": _IDENTITY["knowledge_base_id"],
                                "index_revision_id": _IDENTITY["index_revision_id"],
                                "graph_build_id": _IDENTITY["graph_build_id"],
                                "answer_profile_revision_id": _IDENTITY[
                                    "answer_profile_revision_id"
                                ],
                                "judge_source_profile_revision_id": _IDENTITY[
                                    "judge_profile_revision_id"
                                ],
                            },
                        }
                    ).encode()
                    with tarfile.open(archive_path, "w:gz") as archive:
                        member = tarfile.TarInfo(module.FROZEN_ADAPTIVE_RESULT)
                        member.size = len(payload)
                        archive.addfile(member, BytesIO(payload))
                    archive_path.chmod(0o600)

                    identity = module._seed_adaptive_identity(seed)

                self.assertEqual(
                    identity.knowledge_base_id,
                    UUID(_IDENTITY["knowledge_base_id"]),
                )
                self.assertEqual(
                    identity.answer_profile_revision_id,
                    UUID(_IDENTITY["answer_profile_revision_id"]),
                )
                self.assertEqual(
                    identity.judge_profile_revision_id,
                    UUID(_IDENTITY["judge_profile_revision_id"]),
                )

    def test_adaptive_identity_validates_exact_frozen_database_relation(self) -> None:
        runtime = EvaluationRuntime(
            manifest=Path("runtime.json"),
            runtime_root=Path("."),
            env_file=Path("runtime.env"),
            compose_env_file=Path("compose.env"),
            owner=_OWNER,
            build_revision="a" * 40,
            api_base_url="http://127.0.0.1:28000/api/v1",
            ports=dict(module.EVALUATION_PORTS),
            adaptive_graph=None,
        )
        frozen = module.FrozenAdaptiveGraphIdentity(
            knowledge_base_id=UUID(_IDENTITY["knowledge_base_id"]),
            index_revision_id=UUID(_IDENTITY["index_revision_id"]),
            graph_build_id=UUID(_IDENTITY["graph_build_id"]),
            answer_profile_revision_id=UUID(_IDENTITY["answer_profile_revision_id"]),
            judge_profile_revision_id=UUID(_IDENTITY["judge_profile_revision_id"]),
        )
        output = "\t".join(
            _IDENTITY[name]
            for name in (
                "workspace_id",
                "knowledge_base_id",
                "index_revision_id",
                "graph_build_id",
                "answer_profile_revision_id",
                "judge_profile_revision_id",
            )
        ) + "\t" + "\t".join(
            _PROFILE_IDENTITY[name]
            for name in (
                "schema_profile_key",
                "schema_profile_digest",
                "extractor_version",
            )
        )
        with (
            patch.object(module, "_container_id", return_value="postgres-container"),
            patch.object(module, "_run", return_value=output) as run,
        ):
            identity = module._adaptive_identity(runtime, frozen=frozen)

        query = run.call_args.args[0][-1]
        self.assertIn(str(frozen.knowledge_base_id), query)
        self.assertIn(str(frozen.graph_build_id), query)
        self.assertIn(str(frozen.judge_profile_revision_id), query)
        self.assertEqual(identity.answer_profile_revision_id, frozen.answer_profile_revision_id)
        self.assertEqual(identity.judge_profile_revision_id, frozen.judge_profile_revision_id)

    def test_eval_images_reuse_only_source_equivalent_local_images(self) -> None:
        revision = "a" * 40
        with patch.object(
            module,
            "_run",
            side_effect=(revision, "", revision, "", "", ""),
        ) as run:
            self.assertEqual(module._reuse_local_images(), revision)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(
            commands[-2:],
            [
                ["docker", "image", "tag", "rag-kb-app:local", "rag-kb-app:eval"],
                [
                    "docker",
                    "image",
                    "tag",
                    "rag-kb-user-frontend:local",
                    "rag-kb-user-frontend:eval",
                ],
            ],
        )

    def test_eval_image_reuse_rejects_source_drift_before_tagging(self) -> None:
        with patch.object(
            module,
            "_run",
            side_effect=("a" * 40, subprocess.CalledProcessError(1, ["git"])),
        ) as run:
            with self.assertRaises(EvaluationRuntimeError):
                module._reuse_local_images()
        self.assertEqual(run.call_count, 2)

    def test_database_restore_preserves_dump_ownership(self) -> None:
        runtime = EvaluationRuntime(
            manifest=Path("runtime.json"),
            runtime_root=Path("."),
            env_file=Path("runtime.env"),
            compose_env_file=Path("compose.env"),
            owner=_OWNER,
            build_revision="a" * 40,
            api_base_url="http://127.0.0.1:28000/api/v1",
            ports=dict(module.EVALUATION_PORTS),
            adaptive_graph=None,
        )
        with (
            patch.object(module, "_container_id", return_value="postgres-container"),
            patch.object(module, "_run", return_value="") as run,
        ):
            module._restore_database(runtime, seed=Path("/private/seed"))
        restore_command = run.call_args_list[1].args[0]
        self.assertEqual(restore_command[0:3], ["docker", "exec", "postgres-container"])
        self.assertNotIn("--no-owner", restore_command)
        self.assertNotIn("--no-privileges", restore_command)
        self.assertNotIn("--role", restore_command)

    def test_eval_runtime_replays_existing_baseline_grants_after_restore(self) -> None:
        runtime = EvaluationRuntime(
            manifest=Path("runtime.json"),
            runtime_root=Path("."),
            env_file=Path("runtime.env"),
            compose_env_file=Path("compose.env"),
            owner=_OWNER,
            build_revision="a" * 40,
            api_base_url="http://127.0.0.1:28000/api/v1",
            ports=dict(module.EVALUATION_PORTS),
            adaptive_graph=None,
        )
        with (
            patch.object(module, "_container_id", return_value="postgres-container"),
            patch.object(module, "_run", side_effect=("", "t")) as run,
        ):
            module._reconcile_runtime_grants(runtime)

        command = run.call_args_list[0].args[0]
        check_command = run.call_args_list[1].args[0]
        self.assertEqual(command[:3], ["docker", "exec", "postgres-container"])
        self.assertIn("GRANT SELECT, INSERT, UPDATE, DELETE", command[-1])
        self.assertIn("REVOKE INSERT, UPDATE, DELETE ON TABLE alembic_version", command[-1])
        self.assertIn("REVOKE UPDATE ON TABLE index_chunk_plan", command[-1])
        self.assertEqual(command[-1], module.EVALUATION_RUNTIME_GRANTS)
        self.assertEqual(check_command[-1], module.EVALUATION_RUNTIME_GRANTS_CHECK)
        self.assertTrue(run.call_args_list[1].kwargs["capture"])

    def test_owner_guard_rejects_unknown_service_or_incomplete_volumes(self) -> None:
        runtime = EvaluationRuntime(
            manifest=Path("runtime.json"),
            runtime_root=Path("."),
            env_file=Path("runtime.env"),
            compose_env_file=Path("compose.env"),
            owner=_OWNER,
            build_revision="a" * 40,
            api_base_url="http://127.0.0.1:28000/api/v1",
            ports=dict(module.EVALUATION_PORTS),
            adaptive_graph=None,
        )
        with (
            patch.object(module, "_docker_project_objects", return_value=(("container",), (), ())),
            patch.object(module, "_run", return_value=f"{_OWNER}\tunknown"),
        ):
            with self.assertRaises(EvaluationRuntimeError):
                module._owned_project_objects(runtime)
        with (
            patch.object(module, "_docker_project_objects", return_value=((), ("volume",), ())),
            patch.object(module, "_run", return_value=f"{_OWNER}\tsource-data"),
        ):
            with self.assertRaises(EvaluationRuntimeError):
                module._owned_project_objects(runtime)

    def test_owner_guard_allows_known_partial_objects_only_for_cleanup(self) -> None:
        runtime = EvaluationRuntime(
            manifest=Path("runtime.json"),
            runtime_root=Path("."),
            env_file=Path("runtime.env"),
            compose_env_file=Path("compose.env"),
            owner=_OWNER,
            build_revision="a" * 40,
            api_base_url="http://127.0.0.1:28000/api/v1",
            ports=dict(module.EVALUATION_PORTS),
            adaptive_graph=None,
        )
        with (
            patch.object(module, "_docker_project_objects", return_value=((), ("volume",), ())),
            patch.object(module, "_run", return_value=f"{_OWNER}\tsource-data"),
        ):
            containers, volumes, networks = module._owned_project_objects(
                runtime,
                require_complete=False,
            )
        self.assertEqual(containers, ())
        self.assertEqual(volumes, ("volume",))
        self.assertEqual(networks, ())

    def test_inspect_requires_complete_healthy_service_set(self) -> None:
        runtime = EvaluationRuntime(
            manifest=Path("runtime.json"),
            runtime_root=Path("."),
            env_file=Path("runtime.env"),
            compose_env_file=Path("compose.env"),
            owner=_OWNER,
            build_revision="a" * 40,
            api_base_url="http://127.0.0.1:28000/api/v1",
            ports=dict(module.EVALUATION_PORTS),
            adaptive_graph=AdaptiveGraphIdentity(
                **{name: UUID(value) for name, value in _IDENTITY.items()}
            ),
        )
        complete_objects = (
            tuple(f"container-{index}" for index in range(len(module.READY_SERVICES))),
            tuple(f"volume-{index}" for index in range(len(module.EXPECTED_VOLUMES))),
            ("network",),
        )
        with (
            patch.object(module, "_owned_project_objects", return_value=complete_objects),
            patch.object(module, "_services_ready", return_value=True),
            patch.object(module, "_falkor_restored", return_value=True),
        ):
            self.assertEqual(module.inspect_runtime(runtime)["status"], "ready")
        with (
            patch.object(module, "_owned_project_objects", return_value=complete_objects),
            patch.object(module, "_services_ready", return_value=False),
            patch.object(module, "_falkor_restored", return_value=True),
        ):
            self.assertEqual(module.inspect_runtime(runtime)["status"], "incomplete")

    def test_inspect_requires_nonempty_restored_falkor_database(self) -> None:
        runtime = EvaluationRuntime(
            manifest=Path("runtime.json"),
            runtime_root=Path("."),
            env_file=Path("runtime.env"),
            compose_env_file=Path("compose.env"),
            owner=_OWNER,
            build_revision="a" * 40,
            api_base_url="http://127.0.0.1:28000/api/v1",
            ports=dict(module.EVALUATION_PORTS),
            adaptive_graph=None,
        )
        with (
            patch.object(module, "_container_id", return_value="falkor-container"),
            patch.object(module, "_run", return_value="10"),
        ):
            self.assertTrue(module._falkor_restored(runtime))
        with (
            patch.object(module, "_container_id", return_value="falkor-container"),
            patch.object(module, "_run", return_value="0"),
        ):
            self.assertFalse(module._falkor_restored(runtime))


if __name__ == "__main__":
    unittest.main()
