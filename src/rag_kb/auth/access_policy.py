"""Single-workspace access-policy contract and P0/P1A implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from rag_kb.auth.context import AuthContext, MetadataFilter


class AccessDeniedError(RuntimeError):
    """The supplied identity is absent or outside the configured workspace."""


@runtime_checkable
class AccessPolicy(Protocol):
    def metadata_filter(self, context: AuthContext) -> MetadataFilter: ...

    def authorize_retrieval_debug(self, context: AuthContext) -> None: ...

    def require_workspace(
        self,
        context: AuthContext,
        workspace_id: UUID,
    ) -> MetadataFilter: ...


class SingleWorkspaceAccessPolicy:
    def __init__(self, workspace_id: UUID) -> None:
        self._workspace_id = workspace_id

    def metadata_filter(self, context: AuthContext) -> MetadataFilter:
        if context is None:  # type: ignore[comparison-overlap]
            raise AccessDeniedError("AuthContext is required")
        if context.workspace_id != self._workspace_id:
            raise AccessDeniedError("identity is outside the configured workspace")
        return MetadataFilter(workspace_id=self._workspace_id)

    def require_workspace(
        self,
        context: AuthContext,
        workspace_id: UUID,
    ) -> MetadataFilter:
        metadata_filter = self.metadata_filter(context)
        if workspace_id != metadata_filter.workspace_id:
            raise AccessDeniedError("requested workspace is not authorized")
        return metadata_filter

    def authorize_retrieval_debug(self, context: AuthContext) -> None:
        """The one authenticated development principal may inspect safe plans."""

        self.metadata_filter(context)
