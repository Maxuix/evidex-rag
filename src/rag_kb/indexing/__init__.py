"""Document indexing capability boundary."""

from rag_kb.adapters.parser.plain_text import process_plain_text
from rag_kb.indexing.pipeline import IndexingPipeline

__all__ = ["IndexingPipeline", "process_plain_text"]
