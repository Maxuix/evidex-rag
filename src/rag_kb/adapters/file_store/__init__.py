"""File-store adapter boundary."""

from rag_kb.adapters.file_store.contracts import IndexAssetStore, SourceFileStore
from rag_kb.adapters.file_store.assets import LocalIndexAssetStore
from rag_kb.adapters.file_store.local import LocalFileStore

__all__ = ["IndexAssetStore", "LocalFileStore", "LocalIndexAssetStore", "SourceFileStore"]
