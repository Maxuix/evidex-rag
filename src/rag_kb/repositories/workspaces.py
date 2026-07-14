"""Workspace repository contract and SQLAlchemy implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from rag_kb.domain import Workspace


@runtime_checkable
class WorkspaceRepository(Protocol):
    """Asynchronous persistence contract exposed to application services."""

    async def add(self, name: str) -> Workspace: ...

    async def get(self, workspace_id: UUID) -> Workspace | None: ...
