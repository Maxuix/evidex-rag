"""Fixed server-side identity provider for the local development profile."""

from __future__ import annotations

from uuid import UUID

from rag_kb.auth.context import AuthContext


class DevelopmentAuthProvider:
    def __init__(
        self,
        *,
        deployment_profile: str,
        principal_id: str,
        client_id: str,
        workspace_id: UUID,
    ) -> None:
        if deployment_profile != "development":
            raise ValueError(
                "DevelopmentAuthProvider is restricted to the development profile"
            )
        self._context = AuthContext(
            principal_id=principal_id,
            client_id=client_id,
            workspace_id=workspace_id,
        )

    def get_context(self) -> AuthContext:
        """Return the fixed identity without consulting request-controlled data."""

        return self._context
