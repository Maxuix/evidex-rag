from __future__ import annotations

import os
from pathlib import Path
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
        self.assertIn("python -m pip check", dockerfile)
        self.assertIn("python -m pip uninstall --yes pip", dockerfile)
        self.assertIn("find_spec('pip') is None", dockerfile)
        self.assertLess(
            dockerfile.index("python -m pip uninstall --yes pip"),
            dockerfile.index("FROM python-dependencies AS runtime"),
        )
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

    def test_compose_runs_user_frontend_on_loopback_port(self) -> None:
        configuration = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))

        user_frontend = configuration["services"]["frontend"]
        self.assertEqual(
            user_frontend["ports"],
            ["127.0.0.1:${RAG_KB_FRONTEND_PORT:-3000}:3000"],
        )
        self.assertNotIn("frontend-diagnostic", configuration["services"])
        origins = configuration["x-runtime-environment"][
            "RAG_KB__SECURITY__ALLOWED_CORS_ORIGINS"
        ]
        self.assertIn("${RAG_KB_FRONTEND_PORT:-3000}", origins)
        self.assertIn("http://localhost:${RAG_KB_FRONTEND_PORT:-3000}", origins)
        self.assertNotIn("DIAGNOSTIC_FRONTEND_PORT", origins)
        self.assertIn("http://127.0.0.1:5173", origins)
        self.assertIn("http://localhost:5173", origins)

    def test_compose_uses_one_manifest_and_revision_labels(self) -> None:
        configuration = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))

        self.assertEqual(configuration["x-runtime-service"]["env_file"], [".env.local"])
        self.assertEqual(
            configuration["x-app-image"]["build"]["args"]["RAG_KB_BUILD_REVISION"],
            "${RAG_KB_BUILD_REVISION:-unknown}",
        )
        self.assertEqual(
            configuration["x-user-frontend-image"]["build"]["args"]
            ["RAG_KB_BUILD_REVISION"],
            "${RAG_KB_BUILD_REVISION:-unknown}",
        )
        self.assertIn(
            'LABEL org.opencontainers.image.revision="${RAG_KB_BUILD_REVISION}"',
            DOCKERFILE.read_text(encoding="utf-8"),
        )
        self.assertIn(
            'LABEL org.opencontainers.image.revision="${RAG_KB_BUILD_REVISION}"',
            (ROOT / "apps/web-chat/Dockerfile").read_text(encoding="utf-8"),
        )

    def _run(
        self,
        directory: Path,
        *,
        linked: bool = False,
        doctor_ok: bool = True,
        extra_environment: dict[str, str] | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
        primary = directory / "primary"
        checkout = directory / "linked" if linked else primary
        checkout.mkdir(parents=True)
        primary.mkdir(parents=True, exist_ok=True)
        (primary / ".git").mkdir()
        script = checkout / "start-local.sh"
        script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
        script.chmod(0o755)
        manifest = checkout / ".env.local"
        manifest.write_text(
            "COMPOSE_PROJECT_NAME=rag\n"
            "RAG_KB_API_PORT=18000\n"
            "RAG_KB_FRONTEND_PORT=13000\n",
            encoding="utf-8",
        )
        manifest.chmod(0o600)

        fake_bin = directory / "bin"
        fake_bin.mkdir()
        log = directory / "commands.log"
        git = fake_bin / "git"
        git.write_text(
            textwrap.dedent(
                """\
                #!/bin/sh
                case "$*" in
                  *--git-common-dir*) printf '%s\\n' "$FAKE_GIT_COMMON_DIR" ;;
                  *'rev-parse HEAD'*) printf '0123456789abcdef\\n' ;;
                  *) exit 1 ;;
                esac
                """
            ),
            encoding="utf-8",
        )
        git.chmod(0o755)
        python = fake_bin / "python3"
        python.write_text(
            "#!/bin/sh\nprintf 'python3 %s\\n' \"$*\" >>\"$FAKE_COMMAND_LOG\"\n"
            "exit \"$FAKE_DOCTOR_EXIT\"\n",
            encoding="utf-8",
        )
        python.chmod(0o755)
        docker = fake_bin / "docker"
        docker.write_text(
            "#!/bin/sh\nprintf 'project=%s revision=%s docker %s\\n' "
            '"$COMPOSE_PROJECT_NAME" "$RAG_KB_BUILD_REVISION" "$*" '
            '>>"$FAKE_COMMAND_LOG"\nexit 0\n',
            encoding="utf-8",
        )
        docker.chmod(0o755)

        environment = {
            **os.environ,
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "FAKE_GIT_COMMON_DIR": str(primary / ".git"),
            "FAKE_COMMAND_LOG": str(log),
            "FAKE_DOCTOR_EXIT": "0" if doctor_ok else "1",
        }
        for name in (
            "COMPOSE_PROJECT_NAME",
            "RAG_KB_LOCAL_COMPOSE_ENV_FILE",
            "RAG_KB_LOCAL_APP_ENV_FILE",
            "RAG_KB_BUILD_REVISION",
        ):
            environment.pop(name, None)
        if extra_environment:
            environment.update(extra_environment)
        completed = subprocess.run(
            (str(script),),
            cwd=checkout,
            env=environment,
            capture_output=True,
            text=True,
        )
        return completed, manifest, log

    def test_primary_start_uses_fixed_project_manifest_and_revision(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            completed, manifest, log = self._run(Path(raw_directory))

            self.assertEqual(completed.returncode, 0, completed.stderr)
            calls = log.read_text(encoding="utf-8")
            prefix = f"docker compose --env-file {manifest} --project-name rag"
            self.assertIn(f"project=rag revision=0123456789abcdef {prefix} up -d --wait postgres", calls)
            self.assertIn(f"{prefix} exec -T postgres", calls)
            self.assertIn(f"{prefix} build api frontend", calls)
            self.assertIn(f"{prefix} --profile tools run --rm migrate", calls)
            self.assertNotIn("inspect", calls)
            self.assertNotIn("ps -aq", calls)
            self.assertIn("User Chat: http://127.0.0.1:13000", completed.stdout)
            self.assertIn("API docs: http://127.0.0.1:18000", completed.stdout)

    def test_linked_worktree_stops_before_docker_or_doctor(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            completed, _, log = self._run(Path(raw_directory), linked=True)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("linked worktrees are read-only", completed.stderr)
            self.assertFalse(log.exists())

    def test_doctor_failure_stops_before_compose(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            completed, _, log = self._run(
                Path(raw_directory),
                doctor_ok=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("doctor found blocking drift", completed.stderr)
            calls = log.read_text(encoding="utf-8")
            self.assertIn("python3", calls)
            self.assertIn("docker info", calls)
            self.assertNotIn("docker compose", calls)

    def test_retired_env_override_stops_before_external_action(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            completed, _, log = self._run(
                Path(raw_directory),
                extra_environment={"RAG_KB_LOCAL_APP_ENV_FILE": "legacy.env"},
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("is retired", completed.stderr)
            self.assertFalse(log.exists())


if __name__ == "__main__":
    unittest.main()
