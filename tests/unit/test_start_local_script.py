from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "start-local.sh"


class StartLocalScriptTests(unittest.TestCase):
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
        app_env.write_text("RAG_KB__APP__DEPLOYMENT_PROFILE=development\n")
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
            self.assertIn(
                f"compose --env-file {state_file} up -d storage-init",
                calls,
            )
            self.assertIn(
                f"compose --env-file {state_file} wait storage-init",
                calls,
            )
            self.assertIn(
                f"compose --env-file {state_file} --profile tools run --rm migrate",
                calls,
            )
            self.assertIn(
                f"compose --env-file {state_file} up -d --wait api worker frontend",
                calls,
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
