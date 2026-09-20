from __future__ import annotations

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = ROOT / "Makefile"
COMPOSE = ROOT / "compose.yaml"
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"


def _seconds(value: str | int) -> int:
    return int(value) if isinstance(value, int) else int(str(value).rstrip("s"))


class MakefileStartupTests(unittest.TestCase):
    def test_makefile_is_the_canonical_local_entrypoint(self) -> None:
        makefile = MAKEFILE.read_text(encoding="utf-8")

        self.assertFalse((ROOT / "start-local.sh").exists())
        self.assertIn("COMPOSE_PROJECT := rag", makefile)
        self.assertIn("MANIFEST := .env.local", makefile)
        self.assertIn(
            "docker compose --env-file $(MANIFEST) --project-name $(COMPOSE_PROJECT)",
            makefile,
        )

    def test_up_runs_doctor_then_the_ordered_compose_sequence(self) -> None:
        makefile = MAKEFILE.read_text(encoding="utf-8")

        self.assertIn("up: doctor prepare frontend-dist", makefile)
        self.assertIn("python3 tools/local_runtime.py doctor", makefile)
        steps = [
            "up -d --wait postgres",
            "exec -T postgres /docker-entrypoint-initdb.d/10-init-runtime.sh",
            "build api frontend",
            "up storage-init",
            "--profile tools run --rm migrate",
            "up -d --wait api worker frontend",
        ]
        positions = [makefile.index(step) for step in steps]
        self.assertEqual(positions, sorted(positions))

    def test_revision_and_model_asset_wiring_is_preserved(self) -> None:
        makefile = MAKEFILE.read_text(encoding="utf-8")

        self.assertIn("export RAG_KB_BUILD_REVISION", makefile)
        self.assertIn("git rev-parse HEAD", makefile)
        self.assertIn("docker image inspect rag-kb-app:local", makefile)
        self.assertIn(
            "RAG_KB_BUILD_MODEL_ASSET_CONTEXT := docker-image://rag-kb-app:local",
            makefile,
        )

    def test_identity_guards_are_preserved(self) -> None:
        makefile = MAKEFILE.read_text(encoding="utf-8")

        self.assertIn("RAG_KB_LOCAL_COMPOSE_ENV_FILE is retired", makefile)
        self.assertIn("RAG_KB_LOCAL_APP_ENV_FILE is retired", makefile)
        self.assertIn("the personal Compose project is fixed to", makefile)
        self.assertIn("linked worktrees are read-only for the personal runtime", makefile)

    def test_ready_banner_reads_ports_from_the_manifest(self) -> None:
        makefile = MAKEFILE.read_text(encoding="utf-8")

        self.assertIn("manifest_value", makefile)
        self.assertIn("RAG_KB_API_PORT", makefile)
        self.assertIn("RAG_KB_FRONTEND_PORT", makefile)
        self.assertIn("User Chat: http://127.0.0.1:", makefile)
        self.assertIn("API docs: http://127.0.0.1:", makefile)


class ComposeBuildContractTests(unittest.TestCase):
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

    def test_model_downloads_use_persistent_build_cache(self) -> None:
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        dockerignore = DOCKERIGNORE.read_text(encoding="utf-8")

        cache_mount = (
            "--mount=type=cache,id=rag-kb-build-models-v1,"
            "target=/var/cache/rag-kb-build-models,sharing=locked"
        )
        seed_mount = (
            "--mount=type=bind,from=model-assets,source=/,"
            "target=/mnt/rag-kb-model-assets,ro"
        )
        self.assertEqual(dockerfile.count(cache_mount), 2)
        self.assertEqual(dockerfile.count(seed_mount), 2)
        self.assertEqual(
            dockerfile.count("--cache-path /var/cache/rag-kb-build-models"),
            2,
        )
        self.assertIn(
            "COPY tools/artifact_download.py /app/tools/artifact_download.py",
            dockerfile,
        )
        self.assertIn("!tools/artifact_download.py", dockerignore.splitlines())

    def test_tokenizer_asset_is_in_source_image_and_has_a_build_gate(self) -> None:
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        compose = COMPOSE.read_text(encoding="utf-8")

        self.assertIn("COPY src /app/src", dockerfile)
        self.assertIn(
            "from rag_kb.tokenizer import preflight_tokenizer; preflight_tokenizer()",
            dockerfile,
        )
        self.assertLess(
            dockerfile.index("preflight_tokenizer()"),
            dockerfile.index("USER 10001:10001"),
        )
        self.assertNotIn("TIKTOKEN_CACHE_DIR", dockerfile)
        self.assertNotIn("TIKTOKEN_CACHE_DIR", compose)
        self.assertNotIn(".tiktoken-cache", compose)

    def test_compose_reuses_verified_assets_from_the_current_image(self) -> None:
        configuration = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
        build = configuration["x-app-image"]["build"]
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")

        self.assertIn("model-assets", build["additional_contexts"])
        self.assertIn(
            "RAG_KB_BUILD_MODEL_ASSET_CONTEXT",
            build["additional_contexts"]["model-assets"],
        )
        self.assertIn(
            "/mnt/rag-kb-model-assets/opt/rag-kb/docling-artifacts",
            dockerfile,
        )
        self.assertIn(
            "/mnt/rag-kb-model-assets/opt/rag-kb/local-reranker",
            dockerfile,
        )
        self.assertIn(
            "--artifacts-path "
            "/mnt/rag-kb-model-assets/opt/rag-kb/docling-artifacts",
            dockerfile,
        )
        self.assertIn(
            "--artifacts-path "
            "/mnt/rag-kb-model-assets/opt/rag-kb/local-reranker",
            dockerfile,
        )

    def test_application_build_uses_single_overridable_china_mirrors(self) -> None:
        configuration = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
        arguments = configuration["x-app-image"]["build"]["args"]
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        frontend_arguments = configuration["x-user-frontend-image"]["build"]["args"]
        frontend_dockerfile = (ROOT / "apps/web-chat/Dockerfile").read_text(
            encoding="utf-8"
        )

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
        self.assertEqual(
            arguments["RAG_KB_BUILD_HF_ENDPOINT"],
            (
                "${RAG_KB_BUILD_HF_ENDPOINT:-"
                "https://hf-mirror.com}"
            ),
        )
        self.assertEqual(
            arguments["RAG_KB_BUILD_HF_DISABLE_XET"],
            "${RAG_KB_BUILD_HF_DISABLE_XET:-1}",
        )
        self.assertIn(
            'PIP_INDEX_URL="${RAG_KB_BUILD_PYPI_INDEX_URL}"',
            dockerfile,
        )
        self.assertEqual(
            dockerfile.count('HF_ENDPOINT="${RAG_KB_BUILD_HF_ENDPOINT}"'),
            2,
        )
        self.assertEqual(
            dockerfile.count(
                'HF_HUB_DISABLE_XET="${RAG_KB_BUILD_HF_DISABLE_XET}"'
            ),
            2,
        )
        self.assertEqual(
            arguments["RAG_KB_BUILD_DEBIAN_SECURITY_MIRROR"],
            (
                "${RAG_KB_BUILD_DEBIAN_SECURITY_MIRROR:-"
                "https://mirrors.tuna.tsinghua.edu.cn/debian-security}"
            ),
        )
        self.assertIn(
            'grep -F "URIs: ${RAG_KB_BUILD_DEBIAN_SECURITY_MIRROR}"',
            dockerfile,
        )
        self.assertNotIn("extra-index-url", dockerfile.lower())
        self.assertEqual(
            frontend_arguments["RAG_KB_BUILD_NPM_REGISTRY"],
            (
                "${RAG_KB_BUILD_NPM_REGISTRY:-"
                "https://registry.npmmirror.com}"
            ),
        )
        self.assertIn(
            '--registry="${RAG_KB_BUILD_NPM_REGISTRY}"',
            frontend_dockerfile,
        )
        self.assertIn("--replace-registry-host=always", frontend_dockerfile)
        self.assertIn("--fetch-retries=6", frontend_dockerfile)
        self.assertIn("--fetch-timeout=120000", frontend_dockerfile)
        self.assertIn(
            "--mount=type=cache,id=rag-kb-npm-v1,"
            "target=/root/.npm,sharing=locked",
            frontend_dockerfile,
        )

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

    def test_runtime_image_normalizes_code_read_permissions(self) -> None:
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")

        self.assertIn("chmod a+r /app/alembic.ini", dockerfile)
        self.assertIn("chmod -R a+rX /app/apps /app/src", dockerfile)
        self.assertLess(
            dockerfile.index("chmod -R a+rX /app/apps /app/src"),
            dockerfile.index("USER 10001:10001"),
        )

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

    def test_frontend_can_use_verified_host_build_without_container_network(
        self,
    ) -> None:
        configuration = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
        build = configuration["x-user-frontend-image"]["build"]
        dockerfile = (ROOT / "apps/web-chat/Dockerfile").read_text(
            encoding="utf-8"
        )
        makefile = MAKEFILE.read_text(encoding="utf-8")

        self.assertIn("RAG_KB_BUILD_FRONTEND_TARGET", build["target"])
        self.assertIn("frontend-dist", build["additional_contexts"])
        self.assertIn(
            "RAG_KB_BUILD_FRONTEND_DIST_CONTEXT",
            build["additional_contexts"]["frontend-dist"],
        )
        self.assertIn("FROM runtime-base AS prebuilt-runtime", dockerfile)
        self.assertIn("COPY --from=frontend-dist / /app/dist", dockerfile)
        self.assertIn("npm --prefix $(FRONTEND_DIR) ls --all", makefile)
        self.assertIn("npm --prefix $(FRONTEND_DIR) run build", makefile)
        self.assertIn(
            "RAG_KB_BUILD_FRONTEND_TARGET := prebuilt-runtime",
            makefile,
        )

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


if __name__ == "__main__":
    unittest.main()
