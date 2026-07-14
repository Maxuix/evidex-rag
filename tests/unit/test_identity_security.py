from __future__ import annotations

import unittest
from uuid import UUID

from rag_kb.auth import (
    AccessDeniedError,
    AuthContext,
    DevelopmentAuthProvider,
    SingleWorkspaceAccessPolicy,
)


WORKSPACE = UUID("01900000-0000-7000-8000-000000000001")
OTHER_WORKSPACE = UUID("01900000-0000-7000-8000-000000000002")


class IdentitySecurityTests(unittest.TestCase):
    def test_development_provider_returns_one_fixed_non_null_context(self) -> None:
        provider = DevelopmentAuthProvider(
            deployment_profile="development",
            principal_id="development-principal",
            client_id="development-web",
            workspace_id=WORKSPACE,
        )

        first = provider.get_context()
        second = provider.get_context()

        self.assertIs(first, second)
        self.assertEqual(first.workspace_id, WORKSPACE)
        self.assertEqual(first.principal_id, "development-principal")
        self.assertEqual(first.client_id, "development-web")

    def test_development_provider_cannot_run_in_another_profile(self) -> None:
        with self.assertRaises(ValueError):
            DevelopmentAuthProvider(
                deployment_profile="production",
                principal_id="principal",
                client_id="client",
                workspace_id=WORKSPACE,
            )

    def test_auth_context_requires_non_empty_actor_identifiers(self) -> None:
        for principal_id, client_id in (
            ("", "client"),
            ("principal", ""),
            ("  ", "client"),
            ("principal", "  "),
        ):
            with self.subTest(
                principal_id=principal_id,
                client_id=client_id,
            ), self.assertRaises(ValueError):
                AuthContext(principal_id, client_id, WORKSPACE)

    def test_single_workspace_policy_derives_mandatory_metadata_filter(self) -> None:
        policy = SingleWorkspaceAccessPolicy(WORKSPACE)
        context = AuthContext("principal", "client", WORKSPACE)

        metadata_filter = policy.metadata_filter(context)

        self.assertEqual(metadata_filter.workspace_id, WORKSPACE)
        self.assertEqual(
            policy.require_workspace(context, WORKSPACE),
            metadata_filter,
        )

    def test_single_workspace_policy_rejects_absent_or_cross_workspace_context(self) -> None:
        policy = SingleWorkspaceAccessPolicy(WORKSPACE)
        wrong_context = AuthContext("principal", "client", OTHER_WORKSPACE)

        with self.assertRaises(AccessDeniedError):
            policy.metadata_filter(None)  # type: ignore[arg-type]
        with self.assertRaises(AccessDeniedError):
            policy.metadata_filter(wrong_context)
        with self.assertRaises(AccessDeniedError):
            policy.require_workspace(
                AuthContext("principal", "client", WORKSPACE),
                OTHER_WORKSPACE,
            )


if __name__ == "__main__":
    unittest.main()
