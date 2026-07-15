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
    DocumentUploadResponse,
    DocumentVersionResponse,
)
from rag_kb.schemas.indexing import IndexingErrorResponse, IndexingJobResponse
from rag_kb.schemas.knowledge_bases import (
    KnowledgeBaseCreate,
    KnowledgeBasePage,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
    RetrievalDefaults,
)
from rag_kb.schemas.retrieval import (
    EvidencePackResponse,
    EvidenceResponse,
    RetrievalDebugResponse,
    RetrievalQueryPlanResponse,
    RetrievalQueryRequest,
)

__all__ = [
    "CursorPage",
    "CursorPayload",
    "DocumentDeleteResponse",
    "DocumentPage",
    "DocumentResponse",
    "DocumentUploadResponse",
    "DocumentVersionResponse",
    "IndexingErrorResponse",
    "IndexingJobResponse",
    "ErrorCode",
    "FieldViolation",
    "KnowledgeBaseCreate",
    "KnowledgeBasePage",
    "KnowledgeBaseResponse",
    "KnowledgeBaseUpdate",
    "PaginationQuery",
    "ProblemDetails",
    "RetrievalDefaults",
    "EvidencePackResponse",
    "EvidenceResponse",
    "RetrievalDebugResponse",
    "RetrievalQueryPlanResponse",
    "RetrievalQueryRequest",
]
