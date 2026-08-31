"""Workspace repository contract and SQLAlchemy implementation."""

from __future__ import annotations

from typing import Protocol
from rag_kb.domain import Workspace


class WorkspaceRepository(Protocol):
    """Asynchronous persistence contract exposed to application services."""

    async def add(self, name: str) -> Workspace: ...

    async def get(self) -> Workspace | None: ...

    async def rename(self, name: str) -> Workspace | None: ...
