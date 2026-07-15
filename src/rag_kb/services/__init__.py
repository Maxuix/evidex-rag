"""Application services that coordinate use cases through public contracts."""

from rag_kb.answering import (
    AnswerGenerationStep,
    AnswerStructureValidationStep,
    EvidenceAssessmentStep,
)

from rag_kb.domain import (
    AdmissionLimits,
    AnswerPolicyNotSupportedError,
    ChatMessage,
    ChatRun,
    ChatSession,
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
from rag_kb.services.chat import ChatService, chat_model_configuration
from rag_kb.services.chat_pipeline import (
    ChatEvidenceRetriever,
    ChatExecutionContextLoader,
    ChatPipelineStep,
    ChatRunCoordinator,
    DirectChatPipeline,
)
from rag_kb.services.evaluation import EvaluationPersistenceService
from rag_kb.services.files import FileReconciliationService, SourceFileService
from rag_kb.services.indexing import IndexingJobService
from rag_kb.services.maintenance import MaintenanceCleanupResult, MaintenanceCleanupService
from rag_kb.services.admission import FileAdmissionService

__all__ = [
    "AdmissionLimits",
    "AnswerPolicyNotSupportedError",
    "AnswerGenerationStep",
    "AnswerStructureValidationStep",
    "ContentServices",
    "ChatService",
    "ChatEvidenceRetriever",
    "ChatExecutionContextLoader",
    "ChatPipelineStep",
    "ChatRunCoordinator",
    "DirectChatPipeline",
    "ChatMessage",
    "ChatRun",
    "ChatSession",
    "Document",
    "DocumentMutationResult",
    "DocumentService",
    "ErrorCode",
    "EvidenceAssessmentStep",
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
    "chat_model_configuration",
    "embedding_space_definition",
]
