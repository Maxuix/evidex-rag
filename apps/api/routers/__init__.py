"""Published business routers for the current eligible API surface."""

from apps.api.routers.documents import router as documents_router
from apps.api.routers.chat import router as chat_router
from apps.api.routers.indexing import router as indexing_router
from apps.api.routers.knowledge_bases import router as knowledge_bases_router
from apps.api.routers.retrieval import router as retrieval_router
from apps.api.routers.assets import router as assets_router
from apps.api.routers.model_settings import router as model_settings_router
from apps.api.routers.graph import profile_router as graph_profile_router
from apps.api.routers.graph import router as graph_router


BUSINESS_ROUTERS = (
    knowledge_bases_router,
    documents_router,
    indexing_router,
    retrieval_router,
    assets_router,
    chat_router,
    model_settings_router,
    graph_router,
    graph_profile_router,
)

__all__ = ["BUSINESS_ROUTERS"]
