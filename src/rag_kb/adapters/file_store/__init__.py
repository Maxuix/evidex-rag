"""File-store adapter boundary."""

from rag_kb.adapters.file_store.contracts import SourceFileStore
from rag_kb.adapters.file_store.local import LocalFileStore

__all__ = ["LocalFileStore", "SourceFileStore"]
