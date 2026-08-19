from __future__ import annotations

from pathlib import Path
import unittest

from tools.run_database_tests import (
    CONTAINER_LABEL,
    ContainerIdentity,
    create_container_identity,
    database_environment,
    docker_run_command,
    parse_published_port,
)


class DatabaseTestRunnerTests(unittest.TestCase):
    def test_container_identity_is_scoped_and_unique(self) -> None:
        first = create_container_identity(Path("/tmp/example-worktree"))
        second = create_container_identity(Path("/tmp/example-worktree"))

        self.assertRegex(first.name, r"^rag-kb-db-test-[0-9a-f]{8}-[0-9a-f]{8}$")
        self.assertNotEqual(first, second)
        self.assertEqual(len(first.owner), 32)

    def test_published_port_must_be_one_loopback_endpoint(self) -> None:
        self.assertEqual(parse_published_port("127.0.0.1:49152\n"), 49_152)

        for invalid in (
            "",
            "0.0.0.0:49152",
            "[::]:49152",
            "127.0.0.1:49152\n127.0.0.1:49153",
            "127.0.0.1:70000",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(RuntimeError):
                    parse_published_port(invalid)

    def test_database_dsns_are_passwordless_and_use_dynamic_port(self) -> None:
        environment = database_environment(49_152)

        self.assertEqual(
            set(environment),
            {
                "RAG_KB__DATABASE__MIGRATION_DSN",
                "RAG_KB_TEST_MIGRATION_DSN",
                "RAG_KB_TEST_RUNTIME_DSN",
                "RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN",
            },
        )
        self.assertTrue(
            all(
                "@127.0.0.1:49152/rag_kb" in value
                for value in environment.values()
            )
        )
        self.assertTrue(
            all("://rag_kb_" in value for value in environment.values())
        )
        self.assertTrue(
            all(
                ":isolated-test-only@" not in value
                for value in environment.values()
            )
        )

    def test_docker_container_is_loopback_only_trust_and_tmpfs(self) -> None:
        identity = ContainerIdentity(name="rag-kb-db-test-fixed", owner="owner")
        command = docker_run_command(image="postgres-image", identity=identity)
        rendered = " ".join(command)

        self.assertIn("--pull never", rendered)
        self.assertIn("127.0.0.1::5432", command)
        self.assertIn("POSTGRES_HOST_AUTH_METHOD=trust", command)
        self.assertIn("/var/lib/postgresql:rw,nosuid,size=1024m", command)
        self.assertIn(f"{CONTAINER_LABEL}=owner", command)
        self.assertNotIn(".env", rendered)


if __name__ == "__main__":
    unittest.main()
