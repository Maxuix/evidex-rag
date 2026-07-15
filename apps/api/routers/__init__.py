"""Published business routers for the current eligible API surface."""

from apps.api.routers.documents import router as documents_router
from apps.api.routers.chat import router as chat_router
from apps.api.routers.indexing import router as indexing_router
from apps.api.routers.knowledge_bases import router as knowledge_bases_router
from apps.api.routers.retrieval import router as retrieval_router


BUSINESS_ROUTERS = (
    knowledge_bases_router,
    documents_router,
    indexing_router,
    retrieval_router,
    chat_router,
)

__all__ = ["BUSINESS_ROUTERS"]
