"""Immutable identity and metadata-filter contracts."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class AuthContext:
    principal_id: str
    client_id: str
    workspace_id: UUID

    def __post_init__(self) -> None:
        if not self.principal_id.strip():
            raise ValueError("principal_id must not be empty")
        if not self.client_id.strip():
            raise ValueError("client_id must not be empty")


@dataclass(frozen=True, slots=True)
class MetadataFilter:
    workspace_id: UUID
