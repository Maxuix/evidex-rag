"""Public API data-transfer schemas, separate from domain and persistence models."""

from rag_kb.domain import ErrorCode
from rag_kb.schemas.common import (
    CursorPage,
    CursorPayload,
    FieldViolation,
    PaginationQuery,
    ProblemDetails,
)
from rag_kb.schemas.documents import (
    DocumentDeleteResponse,
    DocumentPage,
    DocumentResponse,
    DocumentVersionResponse,
)
from rag_kb.schemas.knowledge_bases import (
    KnowledgeBaseCreate,
    KnowledgeBasePage,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
    RetrievalDefaults,
)

__all__ = [
    "CursorPage",
    "CursorPayload",
    "DocumentDeleteResponse",
    "DocumentPage",
    "DocumentResponse",
    "DocumentVersionResponse",
    "ErrorCode",
    "FieldViolation",
    "KnowledgeBaseCreate",
    "KnowledgeBasePage",
    "KnowledgeBaseResponse",
    "KnowledgeBaseUpdate",
    "PaginationQuery",
    "ProblemDetails",
    "RetrievalDefaults",
]
