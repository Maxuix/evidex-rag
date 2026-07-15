"""Application services that coordinate use cases through public contracts."""

from rag_kb.domain import (
    AdmissionLimits,
    Document,
    DocumentMutationResult,
    ErrorCode,
    FileAdmissionError,
    IdempotencyKeyReusedError,
    KnowledgeBase,
    ParserLimits,
    ParserExecutionError,
    ParserSource,
    ResourceNameConflictError,
    ResourceNotFoundError,
    ResourceStateConflictError,
)
from rag_kb.services.content import (
    ContentServices,
    DocumentService,
    KnowledgeBaseService,
    build_content_services,
    embedding_space_definition,
)
from rag_kb.services.evaluation import EvaluationPersistenceService
from rag_kb.services.files import FileReconciliationService, SourceFileService
from rag_kb.services.indexing import IndexingJobService
from rag_kb.services.maintenance import MaintenanceCleanupResult, MaintenanceCleanupService
from rag_kb.services.admission import FileAdmissionService

__all__ = [
    "AdmissionLimits",
    "ContentServices",
    "Document",
    "DocumentMutationResult",
    "DocumentService",
    "ErrorCode",
    "EvaluationPersistenceService",
    "FileReconciliationService",
    "IndexingJobService",
    "FileAdmissionService",
    "FileAdmissionError",
    "IdempotencyKeyReusedError",
    "KnowledgeBase",
    "KnowledgeBaseService",
    "MaintenanceCleanupResult",
    "MaintenanceCleanupService",
    "ParserLimits",
    "ParserExecutionError",
    "ParserSource",
    "ResourceNameConflictError",
    "ResourceNotFoundError",
    "ResourceStateConflictError",
    "SourceFileService",
    "build_content_services",
    "embedding_space_definition",
]
