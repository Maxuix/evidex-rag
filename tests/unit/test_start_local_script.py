from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess
import tempfile
import textwrap
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "start-local.sh"
COMPOSE = ROOT / "compose.yaml"
DOCKERFILE = ROOT / "Dockerfile"


def _seconds(value: str | int) -> int:
    return int(value) if isinstance(value, int) else int(str(value).rstrip("s"))


class StartLocalScriptTests(unittest.TestCase):
    def test_python_image_reuses_downloads_across_network_retries(self) -> None:
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")

        self.assertIn("PIP_DEFAULT_TIMEOUT=120", dockerfile)
        self.assertIn("PIP_RETRIES=10", dockerfile)
        self.assertIn(
            "--mount=type=cache,id=rag-kb-pip-v1,"
            "target=/root/.cache/pip,sharing=locked",
            dockerfile,
        )
        self.assertIn("pip install --require-hashes", dockerfile)
        self.assertNotIn("pip install --no-cache-dir", dockerfile)

    def test_application_build_uses_single_overridable_china_mirrors(self) -> None:
        configuration = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
        arguments = configuration["x-app-image"]["build"]["args"]
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")

        self.assertEqual(
            arguments["RAG_KB_BUILD_DEBIAN_MIRROR"],
            (
                "${RAG_KB_BUILD_DEBIAN_MIRROR:-"
                "https://mirrors.tuna.tsinghua.edu.cn/debian}"
            ),
        )
        self.assertEqual(
            arguments["RAG_KB_BUILD_PYPI_INDEX_URL"],
            (
                "${RAG_KB_BUILD_PYPI_INDEX_URL:-"
                "https://pypi.tuna.tsinghua.edu.cn/simple}"
            ),
        )
        self.assertIn(
            'PIP_INDEX_URL="${RAG_KB_BUILD_PYPI_INDEX_URL}"',
            dockerfile,
        )
        self.assertIn(
            'grep -F "URIs: http://deb.debian.org/debian-security"',
            dockerfile,
        )
        self.assertNotIn("extra-index-url", dockerfile.lower())

    def test_compose_allows_bounded_cold_application_startup(self) -> None:
        configuration = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))

        self.assertEqual(
            configuration["services"]["api"]["image"],
            configuration["services"]["worker"]["image"],
        )
        self.assertEqual(
            configuration["services"]["api"]["build"],
            configuration["services"]["worker"]["build"],
        )
        self.assertEqual(
            configuration["services"]["api"]["healthcheck"]["retries"], 40
        )
        worker = configuration["services"]["worker"]["healthcheck"]
        # A fresh heartbeat still proceeds through the real dependency graph
        # and pays the native Docling import cost on every run.
        self.assertGreaterEqual(_seconds(worker["timeout"]), 15)
        self.assertEqual(worker["retries"], 3)
        self.assertLessEqual(
            _seconds(worker["interval"]) * worker["retries"],
            30,
        )

    def test_compose_persists_worker_inference_models_separately(self) -> None:
        configuration = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))

        cache_mount = "inference-model-cache:/var/lib/rag-kb/model-cache"
        self.assertIn(cache_mount, configuration["services"]["storage-init"]["volumes"])
        self.assertIn(cache_mount, configuration["services"]["worker"]["volumes"])
        self.assertNotIn(cache_mount, configuration["services"]["api"]["volumes"])
        self.assertEqual(configuration["volumes"]["inference-model-cache"], None)
        self.assertIn(
            "/var/lib/rag-kb/model-cache/huggingface",
            " ".join(configuration["services"]["storage-init"]["command"]),
        )
        self.assertIn(
            "HF_HOME=/var/lib/rag-kb/model-cache/huggingface",
            DOCKERFILE.read_text(encoding="utf-8"),
        )

    def test_storage_init_creates_runtime_owned_worker_heartbeat_directory(
        self,
    ) -> None:
        configuration = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
        command = " ".join(configuration["services"]["storage-init"]["command"])

        self.assertIn(
            "install -d -o 10001 -g 10001",
            command,
        )
        self.assertIn(
            "/var/lib/rag-kb/sources/.worker-runtime",
            command,
        )

    def test_worker_has_hard_memory_and_pid_limits(self) -> None:
        configuration = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
        worker = configuration["services"]["worker"]

        self.assertEqual(worker["mem_limit"], "6g")
        self.assertEqual(worker["pids_limit"], 256)

    def test_compose_runs_user_and_diagnostic_frontends_on_separate_ports(self) -> None:
        configuration = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))

        user_frontend = configuration["services"]["frontend"]
        diagnostic_frontend = configuration["services"]["frontend-diagnostic"]
        self.assertEqual(
            user_frontend["ports"],
            ["127.0.0.1:${RAG_KB_FRONTEND_PORT:-3000}:3000"],
        )
        self.assertEqual(
            diagnostic_frontend["ports"],
            [
                "127.0.0.1:${RAG_KB_DIAGNOSTIC_FRONTEND_PORT:-3001}:3000",
            ],
        )
        origins = configuration["x-runtime-environment"][
            "RAG_KB__SECURITY__ALLOWED_CORS_ORIGINS"
        ]
        self.assertIn("${RAG_KB_FRONTEND_PORT:-3000}", origins)
        self.assertIn("${RAG_KB_DIAGNOSTIC_FRONTEND_PORT:-3001}", origins)

    def _run(
        self,
        directory: Path,
        *,
        container: bool,
        volume: bool = False,
        credentials: dict[str, str] | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
        fake_bin = directory / "bin"
        fake_bin.mkdir()
        log = directory / "docker.log"
        state_file = directory / ".env.local"
        app_env = directory / ".env"
        app_env.write_text("RAG_KB__APP__BIND_HOST=127.0.0.1\n")
        docker = fake_bin / "docker"
        docker.write_text(
            textwrap.dedent(
                """+                #!/bin/sh
                printf '%s\n' "$*" >>"$FAKE_DOCKER_LOG"
                if [ "$1 $2 $3 $4" = "compose ps -aq postgres" ]; then
                  [ "$FAKE_CONTAINER" = "1" ] && printf 'postgres-container\n'
                  exit 0
                fi
                if [ "$1" = "inspect" ]; then
                  printf 'POSTGRES_PASSWORD=admin-existing\n'
                  printf 'RAG_KB_MIGRATION_PASSWORD=migration-existing\n'
                  printf 'RAG_KB_RUNTIME_PASSWORD=runtime-existing\n'
                  exit 0
                fi
                if [ "$1 $2" = "volume inspect" ]; then
                  [ "$FAKE_VOLUME" = "1" ] && exit 0
                  exit 1
                fi
                exit 0
                """
            ),
            encoding="utf-8",
        )
        docker.chmod(0o755)
        environment = {
            **os.environ,
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "FAKE_DOCKER_LOG": str(log),
            "FAKE_CONTAINER": "1" if container else "0",
            "FAKE_VOLUME": "1" if volume else "0",
            "RAG_KB_LOCAL_COMPOSE_ENV_FILE": str(state_file),
            "RAG_KB_LOCAL_APP_ENV_FILE": str(app_env),
        }
        for name in (
            "POSTGRES_ADMIN_PASSWORD",
            "RAG_KB_MIGRATION_PASSWORD",
            "RAG_KB_RUNTIME_PASSWORD",
            "RAG_KB_LOCAL_DATABASE_PASSWORD",
        ):
            environment.pop(name, None)
        if credentials is not None:
            environment.update(credentials)
        completed = subprocess.run(
            (str(SCRIPT),),
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        return completed, state_file, log

    def test_imports_existing_container_credentials_without_printing_them(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            completed, state_file, log = self._run(
                Path(raw_directory), container=True
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(stat.S_IMODE(state_file.stat().st_mode), 0o600)
            state = state_file.read_text(encoding="utf-8")
            self.assertIn("POSTGRES_ADMIN_PASSWORD=admin-existing", state)
            self.assertIn("RAG_KB_MIGRATION_PASSWORD=migration-existing", state)
            self.assertIn("RAG_KB_RUNTIME_PASSWORD=runtime-existing", state)
            self.assertNotIn("admin-existing", completed.stdout + completed.stderr)
            calls = log.read_text(encoding="utf-8")
            self.assertIn("compose ps -aq postgres", calls)
            self.assertIn("inspect --format", calls)
            self.assertIn(
                f"compose --env-file {state_file} up -d --wait postgres",
                calls,
            )
            build_call = (
                f"compose --env-file {state_file} build api frontend "
                "frontend-diagnostic"
            )
            self.assertIn(build_call, calls)
            self.assertNotIn("build api worker", calls)
            self.assertIn(
                f"compose --env-file {state_file} up storage-init",
                calls,
            )
            self.assertNotIn("wait storage-init", calls)
            self.assertIn(
                f"compose --env-file {state_file} --profile tools run --rm migrate",
                calls,
            )
            self.assertIn(
                f"compose --env-file {state_file} up -d --wait api worker "
                "frontend frontend-diagnostic",
                calls,
            )
            self.assertLess(
                calls.index(build_call),
                calls.index(
                    f"compose --env-file {state_file} --profile tools run "
                    "--rm migrate"
                ),
            )
            self.assertIn("User Chat: http://127.0.0.1:3000", completed.stdout)
            self.assertIn(
                "Diagnostic UI: http://127.0.0.1:3001",
                completed.stdout,
            )

    def test_persists_three_shell_credentials_and_reuses_them(self) -> None:
        credentials = {
            "POSTGRES_ADMIN_PASSWORD": "admin-shell",
            "RAG_KB_MIGRATION_PASSWORD": "migration-shell",
            "RAG_KB_RUNTIME_PASSWORD": "runtime-shell",
        }
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            first, state_file, _ = self._run(
                directory, container=False, credentials=credentials
            )
            self.assertEqual(first.returncode, 0, first.stderr)

            second = subprocess.run(
                (str(SCRIPT),),
                cwd=ROOT,
                env={
                    **os.environ,
                    "PATH": f"{directory / 'bin'}:/usr/bin:/bin",
                    "FAKE_DOCKER_LOG": str(directory / "docker.log"),
                    "FAKE_CONTAINER": "0",
                    "FAKE_VOLUME": "0",
                    "RAG_KB_LOCAL_COMPOSE_ENV_FILE": str(state_file),
                    "RAG_KB_LOCAL_APP_ENV_FILE": str(directory / ".env"),
                },
                capture_output=True,
                text=True,
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("saved local credentials", second.stdout)

    def test_refuses_to_guess_credentials_for_an_existing_volume(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            completed, state_file, _ = self._run(
                Path(raw_directory), container=False, volume=True
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(state_file.exists())
            self.assertIn(
                "existing database volume has no recoverable credentials",
                completed.stderr,
            )

    def test_accepts_one_shared_local_password(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            completed, state_file, _ = self._run(
                Path(raw_directory),
                container=False,
                credentials={"RAG_KB_LOCAL_DATABASE_PASSWORD": "one-local-password"},
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            state = state_file.read_text(encoding="utf-8")
            self.assertEqual(state.count("=one-local-password"), 3)
            self.assertNotIn("one-local-password", completed.stdout + completed.stderr)

    def test_rejects_partial_role_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            completed, state_file, _ = self._run(
                Path(raw_directory),
                container=False,
                credentials={"POSTGRES_ADMIN_PASSWORD": "only-one"},
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(state_file.exists())


if __name__ == "__main__":
    unittest.main()
