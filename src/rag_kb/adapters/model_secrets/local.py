"""Mode-0600 atomic local storage for provider API keys."""

from __future__ import annotations

import os
from pathlib import Path
import secrets
import stat
import re
from datetime import UTC, datetime
from uuid import UUID, uuid4

from rag_kb.domain import ModelSecretEntry


class LocalModelSecretStore:
    def __init__(self, root: Path) -> None:
        resolved = root.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError("model-secret root must be an existing directory")
        self._root = resolved

    def write(self, secret: str) -> str:
        if not secret:
            raise ValueError("model-provider API key must not be empty")
        reference = str(uuid4())
        destination = self._path(reference)
        temporary = self._root / f".{reference}.{secrets.token_hex(8)}.tmp"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(secret)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            os.chmod(destination, 0o600)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
        return reference

    def read(self, reference: str) -> str:
        path = self._path(reference)
        value = path.read_text(encoding="utf-8")
        if not value:
            raise ValueError("stored model-provider API key is empty")
        return value

    def delete(self, reference: str) -> None:
        try:
            self._path(reference).unlink()
        except FileNotFoundError:
            pass

    def list_entries(self) -> tuple[ModelSecretEntry, ...]:
        entries: list[ModelSecretEntry] = []
        for path in sorted(self._root.iterdir(), key=lambda item: item.name):
            try:
                info = path.lstat()
            except FileNotFoundError:
                # Reconciliation is tolerant of an entry removed concurrently.
                continue
            name = path.name
            canonical_reference = None
            try:
                candidate = str(UUID(name))
                if candidate == name:
                    canonical_reference = candidate
            except ValueError:
                pass
            temporary = bool(
                re.fullmatch(
                    r"\.([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.[0-9a-f]{16}\.tmp",
                    name,
                )
            )
            entries.append(
                ModelSecretEntry(
                    canonical_reference=canonical_reference,
                    opaque_name=name,
                    is_regular_file=stat.S_ISREG(info.st_mode),
                    modified_at=datetime.fromtimestamp(info.st_mtime, UTC),
                    is_temporary=temporary,
                )
            )
        return tuple(entries)

    def delete_entry(self, entry: ModelSecretEntry) -> None:
        name = entry.opaque_name
        if not name or Path(name).name != name:
            raise ValueError("invalid model-secret entry name")
        path = self._root / name
        if path.parent != self._root:
            raise ValueError("model-secret entry escaped configured root")
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("model-secret entry is not a regular file")
        path.unlink()

    def _path(self, reference: str) -> Path:
        try:
            normalized = str(UUID(reference))
        except ValueError as error:
            raise ValueError("invalid model-secret reference") from error
        return self._root / normalized
