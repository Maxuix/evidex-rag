"""Framework-independent workspace facts used by persistence contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, slots=True)
class Workspace:
    """A persisted workspace without ORM state or persistence behavior."""

    id: UUID
    name: str
    created_at: datetime
    updated_at: datetime
