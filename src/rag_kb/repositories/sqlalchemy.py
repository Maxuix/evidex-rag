"""SQLAlchemy implementations of relational repository contracts."""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from rag_kb.db.models import Workspace as WorkspaceRow
from rag_kb.domain import Workspace


class SqlAlchemyWorkspaceRepository:
    """Session-bound workspace persistence without transaction ownership."""

    def __init__(
        self,
        session: AsyncSession,
        workspace_id: UUID,
        ensure_active: Callable[[], None],
    ) -> None:
        self._session = session
        self._workspace_id = workspace_id
        self._ensure_active = ensure_active

    async def add(self, name: str) -> Workspace:
        self._ensure_active()
        row = WorkspaceRow(id=self._workspace_id, name=name)
        self._session.add(row)
        await self._session.flush()
        return _to_domain(row)

    async def get(self) -> Workspace | None:
        self._ensure_active()
        row = await self._session.get(WorkspaceRow, self._workspace_id)
        return _to_domain(row) if row is not None else None

    async def rename(self, name: str) -> Workspace | None:
        self._ensure_active()
        row = await self._session.get(WorkspaceRow, self._workspace_id)
        if row is None:
            return None
        row.name = name
        await self._session.flush()
        await self._session.refresh(row)
        return _to_domain(row)


def _to_domain(row: WorkspaceRow) -> Workspace:
    return Workspace(
        id=row.id,
        name=row.name,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
