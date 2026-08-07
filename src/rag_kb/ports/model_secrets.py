"""Port for process-shared local provider credentials."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelSecretStore(Protocol):
    def write(self, secret: str) -> str: ...

    def read(self, reference: str) -> str: ...

    def delete(self, reference: str) -> None: ...
