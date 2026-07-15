"""SQLAlchemy persistence models for the P0/P1A PostgreSQL schema."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PostgreSQLUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.schema import conv


NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class SourceChangeKind(StrEnum):
    UPSERT = "upsert"
    DELETE = "delete"


class DocumentSourceStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DELETED = "deleted"


class IndexRevisionStatus(StrEnum):
    BUILDING = "building"
    READY = "ready"
    ACTIVE = "active"
    RETIRED = "retired"
    FAILED = "failed"


class IndexBuildStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class IndexServingStatus(StrEnum):
    CANDIDATE = "candidate"
    SERVING = "serving"
    RETIRED = "retired"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ChatRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ChatMessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class AssistantMessageStatus(StrEnum):
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class EvalRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def enum_type(enum_class: type[StrEnum], name: str) -> Enum:
    return Enum(
        enum_class,
        name=name,
        values_callable=lambda members: [member.value for member in members],
    )


def uuid_primary_key():
    return mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuidv7()"),
    )


def created_timestamp():
    return mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


def updated_timestamp():
    return mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
        server_onupdate=text("now()"),
    )


class Workspace(Base):
    __tablename__ = "workspace"

    id: Mapped[UUID] = uuid_primary_key()
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    created_at: Mapped[datetime] = created_timestamp()
    updated_at: Mapped[datetime] = updated_timestamp()


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    __table_args__ = (
        UniqueConstraint("workspace_id", "name", name="uq_knowledge_base_workspace_name"),
        UniqueConstraint("workspace_id", "id", name="uq_knowledge_base_workspace_id"),
        ForeignKeyConstraint(
            ["id", "active_index_revision_id"],
            ["index_revision.kb_id", "index_revision.id"],
            name="fk_active_revision_same_kb",
            deferrable=True,
            initially="DEFERRED",
            use_alter=True,
        ),
        CheckConstraint(
            "source_change_seq >= 0", name="knowledge_base_source_change_seq_nonnegative"
        ),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_change_seq: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0")
    )
    active_index_revision_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=True
    )
    provisioned_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    retrieval_defaults: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    answer_policy_defaults: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text(
            "'{\"answer_style\": \"concise\", "
            "\"insufficiency_policy\": \"refuse\"}'::jsonb"
        ),
    )
    created_at: Mapped[datetime] = created_timestamp()
    updated_at: Mapped[datetime] = updated_timestamp()


class EmbeddingSpace(Base):
    __tablename__ = "embedding_space"
    __table_args__ = (
        UniqueConstraint("compatibility_fingerprint"),
        UniqueConstraint("workspace_id", "id", name="uq_embedding_space_workspace_id"),
        CheckConstraint("dimension > 0", name="embedding_space_dimension_positive"),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    provider_identity: Mapped[str] = mapped_column(String(255), nullable=False)
    endpoint_identity: Mapped[str] = mapped_column(String(255), nullable=False)
    requested_model: Mapped[str] = mapped_column(String(255), nullable=False)
    resolved_model: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(255), nullable=False)
    deployment_revision: Mapped[str | None] = mapped_column(String(255), nullable=True)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    distance_metric: Mapped[str] = mapped_column(String(32), nullable=False)
    vector_data_type: Mapped[str] = mapped_column(String(32), nullable=False)
    normalization: Mapped[str] = mapped_column(String(32), nullable=False)
    configuration_fingerprint: Mapped[str] = mapped_column(String(80), nullable=False)
    tokenizer_fingerprint: Mapped[str | None] = mapped_column(String(80), nullable=True)
    compatibility_fingerprint: Mapped[str] = mapped_column(String(80), nullable=False)
    created_at: Mapped[datetime] = created_timestamp()


class Document(Base):
    __tablename__ = "document"
    __table_args__ = (
        UniqueConstraint("kb_id", "id", name="uq_document_kb_id"),
        ForeignKeyConstraint(
            ["id", "current_version_id"],
            ["document_version.document_id", "document_version.id"],
            name="fk_document_current_version",
            deferrable=True,
            initially="DEFERRED",
            use_alter=True,
        ),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    kb_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_base.id", ondelete="CASCADE"), nullable=False, index=True
    )
    current_version_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=True
    )
    display_name: Mapped[str] = mapped_column(String(512), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = created_timestamp()
    updated_at: Mapped[datetime] = updated_timestamp()


class DocumentVersion(Base):
    __tablename__ = "document_version"
    __table_args__ = (
        UniqueConstraint("document_id", "version_number"),
        UniqueConstraint("document_id", "id", name="uq_document_version_document_id"),
        UniqueConstraint("kb_id", "id", name="uq_document_version_kb_id"),
        ForeignKeyConstraint(
            ["kb_id", "document_id"],
            ["document.kb_id", "document.id"],
            name="fk_document_version_same_kb",
        ),
        CheckConstraint("version_number > 0", name="document_version_number_positive"),
        CheckConstraint("size_bytes >= 0", name="document_version_size_nonnegative"),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    kb_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_base.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    source_status: Mapped[DocumentSourceStatus] = mapped_column(
        enum_type(DocumentSourceStatus, "document_source_status"), nullable=False
    )
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    original_filename: Mapped[str] = mapped_column(String(1024), nullable=False)
    media_type: Mapped[str] = mapped_column(String(255), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = created_timestamp()


class SourceChange(Base):
    __tablename__ = "source_change"
    __table_args__ = (
        UniqueConstraint("kb_id", "source_change_seq", name="uq_source_change_sequence"),
        ForeignKeyConstraint(
            ["kb_id", "document_id"],
            ["document.kb_id", "document.id"],
            name="fk_source_change_same_kb_document",
        ),
        ForeignKeyConstraint(
            ["kb_id", "document_version_id"],
            ["document_version.kb_id", "document_version.id"],
            name="fk_source_change_same_kb_version",
        ),
        CheckConstraint("source_change_seq > 0", name="source_change_seq_positive"),
        CheckConstraint(
            "(change_kind = 'delete' AND document_version_id IS NULL) OR "
            "(change_kind = 'upsert' AND document_version_id IS NOT NULL)",
            name="source_change_version_matches_kind",
        ),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    kb_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_base.id", ondelete="CASCADE"), nullable=False
    )
    source_change_seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    document_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    document_version_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=True
    )
    change_kind: Mapped[SourceChangeKind] = mapped_column(
        enum_type(SourceChangeKind, "source_change_kind"), nullable=False
    )
    created_at: Mapped[datetime] = created_timestamp()


class ContentMutation(Base):
    """Lifecycle-specific idempotency record committed with business state."""

    __tablename__ = "content_mutation"
    __table_args__ = (
        UniqueConstraint(
            "principal_id",
            "client_id",
            "endpoint",
            "idempotency_key",
            name="uq_content_mutation_idempotency_scope",
        ),
        CheckConstraint(
            "status IN ('pending', 'completed')",
            name="content_mutation_status_supported",
        ),
        CheckConstraint(
            "kb_id IS NOT NULL OR document_id IS NOT NULL",
            name="content_mutation_has_result",
        ),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    principal_id: Mapped[str] = mapped_column(String(255), nullable=False)
    client_id: Mapped[str] = mapped_column(String(255), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(255), nullable=False)
    idempotency_key: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=False
    )
    request_hash: Mapped[str] = mapped_column(String(71), nullable=False)
    operation: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    kb_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("knowledge_base.id", ondelete="CASCADE"), nullable=True
    )
    document_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), nullable=True
    )
    document_version_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("document_version.id", ondelete="CASCADE"), nullable=True
    )
    source_change_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("source_change.id", ondelete="CASCADE"), nullable=True
    )
    indexed_document_version_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("indexed_document_version.id", ondelete="CASCADE"), nullable=True
    )
    index_revision_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("index_revision.id", ondelete="CASCADE"), nullable=True
    )
    job_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("indexing_job.id", ondelete="CASCADE"), nullable=True
    )
    created_at: Mapped[datetime] = created_timestamp()
    updated_at: Mapped[datetime] = updated_timestamp()


class SourceFileCleanup(Base):
    """Durable, idempotent physical deletion work for one source version."""

    __tablename__ = "source_file_cleanup"
    __table_args__ = (
        UniqueConstraint("document_version_id"),
        CheckConstraint(
            "status IN ('pending', 'completed', 'failed')",
            name="source_file_cleanup_status_supported",
        ),
        CheckConstraint(
            "attempt_count >= 0",
            name="source_file_cleanup_attempt_nonnegative",
        ),
        Index(
            "ix_source_file_cleanup_due",
            "workspace_id",
            "status",
            "next_attempt_at",
        ),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    document_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("document_version.id", ondelete="CASCADE"), nullable=False
    )
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'pending'")
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    next_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    last_error_code: Mapped[str | None] = mapped_column(String(128))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = created_timestamp()
    updated_at: Mapped[datetime] = updated_timestamp()


class IndexRevision(Base):
    __tablename__ = "index_revision"
    __table_args__ = (
        UniqueConstraint("kb_id", "id", name="uq_index_revision_kb_id_id"),
        ForeignKeyConstraint(
            ["workspace_id", "embedding_space_id"],
            ["embedding_space.workspace_id", "embedding_space.id"],
            name="fk_index_revision_same_workspace_embedding",
        ),
        CheckConstraint("source_snapshot_seq >= 0", name="index_revision_snapshot_nonnegative"),
        Index(
            "uq_one_active_revision_per_kb",
            "kb_id",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    kb_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_base.id", ondelete="CASCADE"), nullable=False, index=True
    )
    embedding_space_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=False
    )
    status: Mapped[IndexRevisionStatus] = mapped_column(
        enum_type(IndexRevisionStatus, "index_revision_status"), nullable=False
    )
    source_snapshot_seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    parser_config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    chunking_config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_detail: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = created_timestamp()
    updated_at: Mapped[datetime] = updated_timestamp()


class IndexedDocumentVersion(Base):
    __tablename__ = "indexed_document_version"
    __table_args__ = (
        UniqueConstraint("document_version_id", "index_revision_id"),
        ForeignKeyConstraint(
            ["kb_id", "document_id"],
            ["document.kb_id", "document.id"],
            name="fk_indexed_version_same_kb_document",
        ),
        ForeignKeyConstraint(
            ["kb_id", "document_version_id"],
            ["document_version.kb_id", "document_version.id"],
            name="fk_indexed_version_same_kb_version",
        ),
        ForeignKeyConstraint(
            ["document_id", "document_version_id"],
            ["document_version.document_id", "document_version.id"],
            name="fk_indexed_version_same_document_version",
        ),
        ForeignKeyConstraint(
            ["kb_id", "index_revision_id"],
            ["index_revision.kb_id", "index_revision.id"],
            name="fk_indexed_version_same_kb_revision",
        ),
        CheckConstraint("source_change_seq > 0", name="indexed_version_seq_positive"),
        CheckConstraint(
            "serving_status <> 'serving' OR build_status = 'ready'",
            name=conv("ck_serving_requires_ready"),
        ),
        Index(
            "uq_one_serving_version_per_document_revision",
            "document_id",
            "index_revision_id",
            unique=True,
            postgresql_where=text(
                "build_status = 'ready' AND serving_status = 'serving'"
            ),
        ),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    kb_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_base.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    document_version_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=False
    )
    index_revision_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=False
    )
    source_change_seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    build_status: Mapped[IndexBuildStatus] = mapped_column(
        enum_type(IndexBuildStatus, "index_build_status"), nullable=False
    )
    serving_status: Mapped[IndexServingStatus] = mapped_column(
        enum_type(IndexServingStatus, "index_serving_status"), nullable=False
    )
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_detail: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = created_timestamp()
    updated_at: Mapped[datetime] = updated_timestamp()


class IndexingJob(Base):
    __tablename__ = "indexing_job"
    __table_args__ = (
        UniqueConstraint("indexed_document_version_id"),
        CheckConstraint("attempt >= 0", name="indexing_job_attempt_nonnegative"),
        Index("ix_indexing_job_claim", "status", "next_attempt_at", "created_at"),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    kb_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_base.id", ondelete="CASCADE"), nullable=False, index=True
    )
    indexed_document_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("indexed_document_version.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[JobStatus] = mapped_column(
        enum_type(JobStatus, "job_status"), nullable=False
    )
    phase: Mapped[str] = mapped_column(String(64), nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    claimed_by: Mapped[str | None] = mapped_column(String(255))
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_detail: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = created_timestamp()
    updated_at: Mapped[datetime] = updated_timestamp()


class IndexChunk(Base):
    __tablename__ = "index_chunk"
    __table_args__ = (
        UniqueConstraint("indexed_document_version_id", "ordinal"),
        UniqueConstraint("kb_id", "id", name="uq_index_chunk_kb_id"),
        CheckConstraint("ordinal >= 0", name="index_chunk_ordinal_nonnegative"),
        CheckConstraint("token_count >= 0", name="index_chunk_tokens_nonnegative"),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    kb_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_base.id", ondelete="CASCADE"), nullable=False, index=True
    )
    indexed_document_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("indexed_document_version.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    source_location: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    hierarchy: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    source_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = created_timestamp()


class VectorRecord(Base):
    __tablename__ = "vector_record_1024"
    __table_args__ = (
        UniqueConstraint("index_chunk_id"),
        ForeignKeyConstraint(
            ["kb_id", "index_chunk_id"],
            ["index_chunk.kb_id", "index_chunk.id"],
            name="fk_vector_record_same_kb_chunk",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "embedding_space_id"],
            ["embedding_space.workspace_id", "embedding_space.id"],
            name="fk_vector_record_same_workspace_embedding",
        ),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    kb_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_base.id", ondelete="CASCADE"), nullable=False, index=True
    )
    index_chunk_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=False
    )
    embedding_space_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=False, index=True
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(1024), nullable=False)
    created_at: Mapped[datetime] = created_timestamp()


class ChatSession(Base):
    __tablename__ = "chat_session"
    __table_args__ = (
        UniqueConstraint("workspace_id", "id", name="uq_chat_session_workspace_id"),
        ForeignKeyConstraint(
            ["workspace_id", "kb_id"],
            ["knowledge_base.workspace_id", "knowledge_base.id"],
            name="fk_chat_session_same_workspace_kb",
        ),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    kb_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    principal_id: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = created_timestamp()
    updated_at: Mapped[datetime] = updated_timestamp()


class ChatMessage(Base):
    __tablename__ = "chat_message"
    __table_args__ = (
        UniqueConstraint("session_id", "client_request_id"),
        ForeignKeyConstraint(
            ["workspace_id", "session_id"],
            ["chat_session.workspace_id", "chat_session.id"],
            name="fk_chat_message_same_workspace_session",
        ),
        ForeignKeyConstraint(
            ["chat_run_id"],
            ["chat_run.id"],
            name="fk_chat_message_run",
            use_alter=True,
        ),
        CheckConstraint(
            "(role = 'user' AND assistant_status IS NULL) OR "
            "(role = 'assistant' AND assistant_status IS NOT NULL)",
            name="chat_message_status_matches_role",
        ),
        Index(
            "uq_assistant_message_chat_run",
            "chat_run_id",
            unique=True,
            postgresql_where=text("role = 'assistant' AND chat_run_id IS NOT NULL"),
        ),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    session_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    chat_run_id: Mapped[UUID | None] = mapped_column(PostgreSQLUUID(as_uuid=True))
    role: Mapped[ChatMessageRole] = mapped_column(
        enum_type(ChatMessageRole, "chat_message_role"), nullable=False
    )
    assistant_status: Mapped[AssistantMessageStatus | None] = mapped_column(
        enum_type(AssistantMessageStatus, "assistant_message_status")
    )
    client_request_id: Mapped[UUID | None] = mapped_column(PostgreSQLUUID(as_uuid=True))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = created_timestamp()


class ChatRun(Base):
    __tablename__ = "chat_run"
    __table_args__ = (
        UniqueConstraint(
            "principal_id",
            "client_id",
            "endpoint",
            "idempotency_key",
            name="uq_chat_run_idempotency_scope",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "session_id"],
            ["chat_session.workspace_id", "chat_session.id"],
            name="fk_chat_run_same_workspace_session",
        ),
        CheckConstraint("attempt >= 0", name="chat_run_attempt_nonnegative"),
        Index("ix_chat_run_claim", "status", "next_attempt_at", "created_at"),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    kb_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_base.id", ondelete="CASCADE"), nullable=False, index=True
    )
    session_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    user_message_id: Mapped[UUID] = mapped_column(
        ForeignKey("chat_message.id", ondelete="RESTRICT"), nullable=False
    )
    index_revision_id: Mapped[UUID] = mapped_column(
        ForeignKey("index_revision.id", ondelete="RESTRICT"), nullable=False
    )
    status: Mapped[ChatRunStatus] = mapped_column(
        enum_type(ChatRunStatus, "chat_run_status"), nullable=False
    )
    principal_id: Mapped[str] = mapped_column(String(255), nullable=False)
    client_id: Mapped[str] = mapped_column(String(255), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(255), nullable=False)
    idempotency_key: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(71), nullable=False)
    requested_policy: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    effective_policy: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    retrieval_strategy: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    model_configuration: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    claimed_by: Mapped[str | None] = mapped_column(String(255))
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_detail: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error_retryable: Mapped[bool | None] = mapped_column(Boolean)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    timing: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = created_timestamp()
    updated_at: Mapped[datetime] = updated_timestamp()
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Citation(Base):
    __tablename__ = "citation"
    __table_args__ = (
        UniqueConstraint("assistant_message_id", "ordinal"),
        CheckConstraint("ordinal >= 0", name="citation_ordinal_nonnegative"),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    assistant_message_id: Mapped[UUID] = mapped_column(
        ForeignKey("chat_message.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    index_chunk_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("index_chunk.id", ondelete="SET NULL")
    )
    document_id_snapshot: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    document_version_id_snapshot: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=False
    )
    quoted_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_location: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    score: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = created_timestamp()


class EvalDataset(Base):
    __tablename__ = "eval_dataset"
    __table_args__ = (
        UniqueConstraint("workspace_id", "name", "version"),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = created_timestamp()


class EvalCase(Base):
    __tablename__ = "eval_case"
    __table_args__ = (UniqueConstraint("dataset_id", "case_key"),)

    id: Mapped[UUID] = uuid_primary_key()
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("eval_dataset.id", ondelete="CASCADE"), nullable=False, index=True
    )
    case_key: Mapped[str] = mapped_column(String(255), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    tags: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    created_at: Mapped[datetime] = created_timestamp()


class EvalRun(Base):
    __tablename__ = "eval_run"

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    kb_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_base.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("eval_dataset.id", ondelete="RESTRICT"), nullable=False
    )
    index_revision_id: Mapped[UUID] = mapped_column(
        ForeignKey("index_revision.id", ondelete="RESTRICT"), nullable=False
    )
    status: Mapped[EvalRunStatus] = mapped_column(
        enum_type(EvalRunStatus, "eval_run_status"), nullable=False
    )
    run_config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_detail: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = created_timestamp()
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EvalResult(Base):
    __tablename__ = "eval_result"
    __table_args__ = (UniqueConstraint("eval_run_id", "eval_case_id"),)

    id: Mapped[UUID] = uuid_primary_key()
    eval_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("eval_run.id", ondelete="CASCADE"), nullable=False, index=True
    )
    eval_case_id: Mapped[UUID] = mapped_column(
        ForeignKey("eval_case.id", ondelete="CASCADE"), nullable=False
    )
    generated_answer: Mapped[str | None] = mapped_column(Text)
    evidence: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_detail: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = created_timestamp()
