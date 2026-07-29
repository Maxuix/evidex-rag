"""Application-facing remote media contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class FetchedImage:
    content: bytes
    final_url: str


class RemoteImageFetcher(Protocol):
    def fetch(self, url: str, *, max_bytes: int) -> FetchedImage: ...
