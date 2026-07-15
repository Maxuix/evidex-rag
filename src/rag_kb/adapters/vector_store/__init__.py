"""Fixed P1A vector-space and retrieval boundaries."""

from rag_kb.adapters.vector_store.contracts import VectorStore
from rag_kb.adapters.vector_store.fixed_pgvector import FixedPgVectorSpace

__all__ = ["FixedPgVectorSpace", "VectorStore"]
