"""Fixed P1A vector-space and retrieval boundaries."""

from rag_kb.adapters.vector_store.contracts import VectorStore
from rag_kb.adapters.vector_store.fixed_pgvector import FixedPgVectorSpace
from rag_kb.adapters.vector_store.pgvector import PgVectorStore

__all__ = ["FixedPgVectorSpace", "PgVectorStore", "VectorStore"]
