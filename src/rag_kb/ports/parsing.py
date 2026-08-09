"""Application-facing parser contract and immutable parse result."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from rag_kb.domain import ParserProfile, ParserProgress, ParserSource

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument


@dataclass(frozen=True, slots=True)
class DocumentParseResult:
    document: DoclingDocument
    surface_labels: tuple[tuple[int, str], ...] = ()
    page_image_surfaces: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        surface_ordinals = tuple(ordinal for ordinal, _ in self.surface_labels)
        if (
            any(ordinal < 1 or not label.strip() for ordinal, label in self.surface_labels)
            or len(set(surface_ordinals)) != len(surface_ordinals)
            or any(surface < 1 for surface in self.page_image_surfaces)
        ):
            raise ValueError("parse result surface metadata is invalid")


@dataclass(frozen=True, slots=True)
class DocumentParseContinuation:
    progress: ParserProgress


ParserProgressHandler = Callable[[ParserProgress], Awaitable[None]]


@runtime_checkable
class DocumentParser(Protocol):
    async def parse(
        self,
        source: ParserSource,
        *,
        profile: ParserProfile,
        checkpoint_key: str,
        on_progress: ParserProgressHandler | None = None,
    ) -> DocumentParseResult | DocumentParseContinuation: ...

    def discard_checkpoint(self, checkpoint_key: str) -> None: ...
