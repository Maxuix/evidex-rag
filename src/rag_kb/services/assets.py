"""Authorized derived-asset reads without exposing storage identities."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING
from uuid import UUID

from rag_kb.domain import (
    IndexAssetContent,
    ResourceNotFoundError,
    SourceFileIntegrityError,
)
from rag_kb.ports.files import IndexAssetStore
from rag_kb.uow import execute_in_transaction

if TYPE_CHECKING:
    from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWork, SqlAlchemyUnitOfWorkFactory


class IndexAssetService:
    def __init__(
        self,
        unit_of_work: SqlAlchemyUnitOfWorkFactory,
        asset_store: IndexAssetStore,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._asset_store = asset_store

    async def read(self, asset_id: UUID) -> IndexAssetContent:
        async def load(uow: SqlAlchemyUnitOfWork):
            return await uow.indexing.get_asset(asset_id)

        snapshot = await execute_in_transaction(
            self._unit_of_work, load
        )
        if snapshot is None:
            raise ResourceNotFoundError("index asset was not found")
        identity = self._asset_store.parse_uri(snapshot.storage_uri)
        if (
            identity.workspace_id != snapshot.workspace_id
            or identity.indexed_document_version_id
            != snapshot.indexed_document_version_id
        ):
            raise SourceFileIntegrityError("index asset identity differs")
        content = await self._asset_store.read(identity)
        if hashlib.sha256(content).hexdigest() != snapshot.checksum_sha256:
            raise SourceFileIntegrityError("index asset checksum differs")
        return IndexAssetContent(snapshot, content)
