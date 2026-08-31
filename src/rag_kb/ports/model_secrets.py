"""Port for process-shared local provider credentials."""

from __future__ import annotations

from typing import Protocol

from rag_kb.domain import ModelSecretEntry


class ModelSecretStore(Protocol):
    def write(self, secret: str) -> str: ...

    def read(self, reference: str) -> str: ...

    def delete(self, reference: str) -> None: ...

    def list_entries(self) -> tuple[ModelSecretEntry, ...]: ...

    def delete_entry(self, entry: ModelSecretEntry) -> None: ...
