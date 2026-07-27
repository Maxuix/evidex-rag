"""Markdown remote-media adapter boundary."""

from rag_kb.adapters.markdown_media.contracts import (
    FetchedImage,
    RemoteImageFetcher,
)
from rag_kb.adapters.markdown_media.http import PublicHttpImageFetcher

__all__ = [
    "FetchedImage",
    "PublicHttpImageFetcher",
    "RemoteImageFetcher",
]
