"""Document indexing capability boundary."""

from rag_kb.indexing.pipeline import IndexingPipeline
from rag_kb.indexing.promotion import CandidatePromotionService

__all__ = ["CandidatePromotionService", "IndexingPipeline"]
