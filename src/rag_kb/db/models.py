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
    Computed,
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
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID as PostgreSQLUUID
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


class ModelProvider(Base):
    __tablename__ = "model_provider"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id", "name", name="uq_model_provider_workspace_name"
        ),
        UniqueConstraint(
            "workspace_id", "id", name="uq_model_provider_workspace_id"
        ),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    created_at: Mapped[datetime] = created_timestamp()
    updated_at: Mapped[datetime] = updated_timestamp()


class ModelProviderRevision(Base):
    __tablename__ = "model_provider_revision"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id", "id", name="uq_model_provider_revision_workspace_id"
        ),
        UniqueConstraint(
            "provider_id", "revision", name="uq_model_provider_revision_number"
        ),
        ForeignKeyConstraint(
            ["workspace_id", "provider_id"],
            ["model_provider.workspace_id", "model_provider.id"],
            name="fk_model_provider_revision_same_workspace_provider",
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "revision > 0", name="model_provider_revision_positive"
        ),
        CheckConstraint(
            "timeout_seconds > 0 AND max_retries >= 0 AND max_concurrency > 0",
            name="model_provider_revision_limits_valid",
        ),
        CheckConstraint(
            "protocol IN ('openai_compatible','tongyi_multimodal')",
            name="model_provider_revision_protocol_supported",
        ),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    provider_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=False, index=True
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    protocol: Mapped[str] = mapped_column(String(64), nullable=False)
    base_url: Mapped[str] = mapped_column(Text, nullable=False)
    secret_reference: Mapped[str] = mapped_column(String(64), nullable=False)
    timeout_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    max_retries: Mapped[int] = mapped_column(Integer, nullable=False)
    max_concurrency: Mapped[int] = mapped_column(Integer, nullable=False)
    configuration_fingerprint: Mapped[str] = mapped_column(String(80), nullable=False)
    created_at: Mapped[datetime] = created_timestamp()


class ModelProfile(Base):
    __tablename__ = "model_profile"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id", "name", name="uq_model_profile_workspace_name"
        ),
        UniqueConstraint(
            "workspace_id", "id", name="uq_model_profile_workspace_id"
        ),
        ForeignKeyConstraint(
            ["workspace_id", "provider_id"],
            ["model_provider.workspace_id", "model_provider.id"],
            name="fk_model_profile_same_workspace_provider",
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "kind IN ('chat','text_embedding','multimodal_embedding')",
            name="model_profile_kind_supported",
        ),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    provider_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    created_at: Mapped[datetime] = created_timestamp()
    updated_at: Mapped[datetime] = updated_timestamp()


class ModelProfileRevision(Base):
    __tablename__ = "model_profile_revision"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id", "id", name="uq_model_profile_revision_workspace_id"
        ),
        UniqueConstraint(
            "profile_id", "revision", name="uq_model_profile_revision_number"
        ),
        ForeignKeyConstraint(
            ["workspace_id", "profile_id"],
            ["model_profile.workspace_id", "model_profile.id"],
            name="fk_model_profile_revision_same_workspace_profile",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "provider_revision_id"],
            [
                "model_provider_revision.workspace_id",
                "model_provider_revision.id",
            ],
            name="fk_model_profile_revision_same_workspace_provider_revision",
            ondelete="RESTRICT",
        ),
        CheckConstraint("revision > 0", name="model_profile_revision_positive"),
        CheckConstraint(
            "validation_status IN ('unverified','valid','invalid')",
            name="model_profile_revision_validation_status_supported",
        ),
        CheckConstraint(
            "jsonb_typeof(configuration) = 'object' "
            "AND pg_column_size(configuration) <= 65536",
            name="model_profile_revision_configuration_object",
        ),
        CheckConstraint(
            "validation_snapshot IS NULL OR "
            "(jsonb_typeof(validation_snapshot) = 'object' "
            "AND pg_column_size(validation_snapshot) <= 65536)",
            name="model_profile_revision_validation_snapshot_object",
        ),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    profile_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=False, index=True
    )
    provider_revision_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=False, index=True
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    configuration: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    configuration_fingerprint: Mapped[str] = mapped_column(String(80), nullable=False)
    capability_fingerprint: Mapped[str] = mapped_column(String(80), nullable=False)
    compatibility_fingerprint: Mapped[str | None] = mapped_column(
        String(80), nullable=True
    )
    validation_status: Mapped[str] = mapped_column(String(32), nullable=False)
    validation_error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    validation_snapshot: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB(none_as_null=True), nullable=True
    )
    validated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = created_timestamp()


class ModelSelection(Base):
    __tablename__ = "model_selection"
    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_id", "chat_profile_revision_id"],
            ["model_profile_revision.workspace_id", "model_profile_revision.id"],
            name="fk_model_selection_same_workspace_chat_profile",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "text_embedding_profile_revision_id"],
            ["model_profile_revision.workspace_id", "model_profile_revision.id"],
            name="fk_model_selection_same_workspace_text_embedding_profile",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "multimodal_embedding_profile_revision_id"],
            ["model_profile_revision.workspace_id", "model_profile_revision.id"],
            name="fk_model_selection_same_workspace_multimodal_embedding_profile",
            ondelete="RESTRICT",
        ),
    )

    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="CASCADE"), primary_key=True
    )
    chat_profile_revision_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=True
    )
    text_embedding_profile_revision_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=True
    )
    multimodal_embedding_profile_revision_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=True
    )
    updated_at: Mapped[datetime] = updated_timestamp()


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    __table_args__ = (
        Index(
            "uq_knowledge_base_workspace_active_name",
            "workspace_id",
            "name",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
        Index("ix_knowledge_base_workspace_deleted_at", "workspace_id", "deleted_at"),
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
            "\"insufficiency_policy\": \"partial_answer\"}'::jsonb"
        ),
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = created_timestamp()
    updated_at: Mapped[datetime] = updated_timestamp()


class KnowledgeBaseGraphConfig(Base):
    __tablename__ = "knowledge_base_graph_config"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id", "kb_id", name="uq_graph_config_workspace_kb"
        ),
        UniqueConstraint(
            "workspace_id", "kb_id", "build_id", name="uq_graph_config_build"
        ),
        ForeignKeyConstraint(
            ["workspace_id", "kb_id"],
            ["knowledge_base.workspace_id", "knowledge_base.id"],
            name="fk_graph_config_same_workspace_kb",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "chat_profile_revision_id"],
            ["model_profile_revision.workspace_id", "model_profile_revision.id"],
            name="fk_graph_config_same_workspace_chat_profile",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "kb_id", "active_build_id"],
            [
                "graphiti_graph_build.workspace_id",
                "graphiti_graph_build.kb_id",
                "graphiti_graph_build.build_id",
            ],
            name="fk_graph_config_active_graphiti_build",
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "status IN ('disabled','building','ready','failed')",
            name="graph_config_status_supported",
        ),
        CheckConstraint(
            "(status = 'disabled' AND chat_profile_revision_id IS NULL) "
            "OR (status <> 'disabled' AND chat_profile_revision_id IS NOT NULL)",
            name="graph_config_profile_matches_status",
        ),
        CheckConstraint(
            "length(btrim(extractor_version)) > 0",
            name="graph_config_extractor_version_nonempty",
        ),
        CheckConstraint(
            "length(btrim(schema_profile_key)) > 0",
            name="graph_config_schema_profile_key_nonempty",
        ),
        CheckConstraint(
            "schema_profile_digest ~ '^[0-9a-f]{64}$'",
            name="graph_config_schema_profile_digest_valid",
        ),
    )

    kb_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), primary_key=True
    )
    workspace_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'disabled'")
    )
    build_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=False, server_default=text("uuidv7()")
    )
    active_build_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=True
    )
    chat_profile_revision_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=True
    )
    extractor_version: Mapped[str] = mapped_column(
        String(128), nullable=False, server_default=text("'graphiti_v1'")
    )
    schema_profile_key: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        server_default=text("'generic_open_domain_v1'"),
    )
    schema_profile_digest: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        server_default=text(
            "'3b351f4e2c601226f922d12b60d4c9f98a4770f4ec04e94b08f5a3f0d021eaf0'"
        ),
    )
    preflight_extractor_version: Mapped[str | None] = mapped_column(String(128))
    last_error_code: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = created_timestamp()
    updated_at: Mapped[datetime] = updated_timestamp()


class GraphitiGraphBuild(Base):
    __tablename__ = "graphiti_graph_build"
    __table_args__ = (
        UniqueConstraint("workspace_id", "kb_id", "build_id", name="uq_graphiti_build_scope"),
        UniqueConstraint("group_id", name="uq_graphiti_build_group"),
        ForeignKeyConstraint(
            ["workspace_id", "kb_id"],
            ["knowledge_base.workspace_id", "knowledge_base.id"],
            name="fk_graphiti_build_same_workspace_kb",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "chat_profile_revision_id"],
            ["model_profile_revision.workspace_id", "model_profile_revision.id"],
            name="fk_graphiti_build_same_workspace_chat_profile",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "embedding_profile_revision_id"],
            ["model_profile_revision.workspace_id", "model_profile_revision.id"],
            name="fk_graphiti_build_same_workspace_embedding_profile",
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "status IN ('building','ready','failed','superseded')",
            name="graphiti_build_status_supported",
        ),
        CheckConstraint(
            "expected_episode_count >= 0 AND embedding_dimension BETWEEN 64 AND 4096",
            name="graphiti_build_counts_valid",
        ),
        CheckConstraint(
            "length(btrim(schema_profile_key)) > 0",
            name="graphiti_build_schema_profile_key_nonempty",
        ),
        CheckConstraint(
            "schema_profile_digest ~ '^[0-9a-f]{64}$'",
            name="graphiti_build_schema_profile_digest_valid",
        ),
    )

    build_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    workspace_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    kb_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    group_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    index_revision_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    serving_chunk_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    expected_episode_count: Mapped[int] = mapped_column(Integer, nullable=False)
    chat_profile_revision_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    embedding_profile_revision_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    extractor_version: Mapped[str] = mapped_column(String(128), nullable=False)
    schema_profile_key: Mapped[str] = mapped_column(String(128), nullable=False)
    schema_profile_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    superseded_by: Mapped[UUID | None] = mapped_column(PostgreSQLUUID(as_uuid=True))
    last_error_code: Mapped[str | None] = mapped_column(String(128))
    started_at: Mapped[datetime] = created_timestamp()
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class GraphitiGraphWorkLease(Base):
    __tablename__ = "graphiti_graph_work_lease"
    __table_args__ = (
        UniqueConstraint("lease_token", name="uq_graphiti_work_lease_token"),
        ForeignKeyConstraint(
            ["workspace_id", "kb_id", "build_id"],
            [
                "graphiti_graph_build.workspace_id",
                "graphiti_graph_build.kb_id",
                "graphiti_graph_build.build_id",
            ],
            name="fk_graphiti_work_lease_build",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "kb_id", "index_chunk_id"],
            ["index_chunk.workspace_id", "index_chunk.kb_id", "index_chunk.id"],
            name="fk_graphiti_work_lease_chunk",
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "work_kind IN ('preflight','chunk','finalize')",
            name="graphiti_work_lease_kind_supported",
        ),
        Index(
            "ix_graphiti_work_lease_expiry",
            "lease_expires_at",
        ),
    )

    build_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    workspace_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    kb_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    lease_token: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    claimed_by: Mapped[str] = mapped_column(String(255), nullable=False)
    claimed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    lease_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    work_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    index_chunk_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=True
    )


class GraphitiEpisodeChunk(Base):
    __tablename__ = "graphiti_episode_chunk"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id", "kb_id", "build_id", "index_chunk_id",
            name="uq_graphiti_episode_build_chunk",
        ),
        UniqueConstraint("group_id", "episode_uuid", name="uq_graphiti_episode_group_uuid"),
        ForeignKeyConstraint(
            ["workspace_id", "kb_id", "build_id"],
            ["graphiti_graph_build.workspace_id", "graphiti_graph_build.kb_id", "graphiti_graph_build.build_id"],
            name="fk_graphiti_episode_build",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "kb_id", "index_chunk_id"],
            ["index_chunk.workspace_id", "index_chunk.kb_id", "index_chunk.id"],
            name="fk_graphiti_episode_chunk",
            ondelete="CASCADE",
        ),
        CheckConstraint("status IN ('completed')", name="graphiti_episode_status_supported"),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    kb_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    build_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    group_id: Mapped[str] = mapped_column(String(255), nullable=False)
    episode_uuid: Mapped[str] = mapped_column(String(64), nullable=False)
    index_revision_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    index_chunk_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default=text("'completed'"))
    created_at: Mapped[datetime] = created_timestamp()


class EmbeddingSpace(Base):
    __tablename__ = "embedding_space"
    __table_args__ = (
        UniqueConstraint("compatibility_fingerprint"),
        UniqueConstraint("workspace_id", "id", name="uq_embedding_space_workspace_id"),
        UniqueConstraint(
            "workspace_id",
            "id",
            "dimension",
            name="uq_embedding_space_workspace_id_dimension",
        ),
        CheckConstraint(
            "dimension BETWEEN 64 AND 4096",
            name="embedding_space_dimension_supported",
        ),
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
    model_profile_revision_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("model_profile_revision.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
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
            "endpoint",
            "idempotency_key",
            name="uq_content_mutation_idempotency_scope",
        ),
        CheckConstraint(
            "status IN ('pending', 'completed', 'failed')",
            name="content_mutation_status_supported",
        ),
        CheckConstraint(
            "(status = 'failed' AND failure_code IS NOT NULL AND failed_at IS NOT NULL) OR "
            "(status <> 'failed' AND failure_code IS NULL AND failed_at IS NULL)",
            name="content_mutation_failure_facts_match_status",
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
    endpoint: Mapped[str] = mapped_column(String(255), nullable=False)
    idempotency_key: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=False
    )
    request_hash: Mapped[str] = mapped_column(String(71), nullable=False)
    operation: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    failure_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
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
        UniqueConstraint("workspace_id", "id", name="uq_index_revision_workspace_id"),
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
    enrichment_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    representation_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_detail: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = created_timestamp()
    updated_at: Mapped[datetime] = updated_timestamp()


class IndexRevisionEmbeddingSpace(Base):
    __tablename__ = "index_revision_embedding_space"
    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_id", "index_revision_id"],
            ["index_revision.workspace_id", "index_revision.id"],
            name="fk_revision_space_same_workspace_revision",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "embedding_space_id"],
            ["embedding_space.workspace_id", "embedding_space.id"],
            name="fk_revision_space_same_workspace_embedding",
        ),
        CheckConstraint(
            "role IN ('text_retrieval', 'semantic_analysis', 'cross_modal_retrieval')",
            name="revision_space_role_supported",
        ),
        CheckConstraint(
            "retrieval_weight_micros IS NULL OR retrieval_weight_micros > 0",
            name="revision_space_weight_positive",
        ),
    )

    index_revision_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), primary_key=True
    )
    role: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    embedding_space_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=False, index=True
    )
    required: Mapped[bool] = mapped_column(Boolean, nullable=False)
    retrieval_weight_micros: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = created_timestamp()


class IndexedDocumentVersion(Base):
    __tablename__ = "indexed_document_version"
    __table_args__ = (
        UniqueConstraint("document_version_id", "index_revision_id"),
        UniqueConstraint(
            "workspace_id",
            "kb_id",
            "id",
            name="uq_indexed_version_workspace_kb_id",
        ),
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


class IndexChunkPlan(Base):
    __tablename__ = "index_chunk_plan"
    __table_args__ = (
        CheckConstraint("unit_count > 0", name="index_chunk_plan_unit_count_positive"),
        CheckConstraint("chunk_count > 0", name="index_chunk_plan_chunk_count_positive"),
        CheckConstraint(
            "jsonb_typeof(boundaries) = 'array'",
            name="index_chunk_plan_boundaries_array",
        ),
        CheckConstraint(
            "jsonb_array_length(boundaries) = chunk_count - 1",
            name="index_chunk_plan_boundary_count",
        ),
    )

    indexed_document_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("indexed_document_version.id", ondelete="CASCADE"),
        primary_key=True,
    )
    source_checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    profile_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    unit_sequence_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    unit_count: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    boundaries: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    plan_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = created_timestamp()


class IndexAsset(Base):
    __tablename__ = "index_asset"
    __table_args__ = (
        UniqueConstraint("indexed_document_version_id", "asset_key"),
        UniqueConstraint(
            "indexed_document_version_id", "id", name="uq_index_asset_target_id"
        ),
        ForeignKeyConstraint(
            ["kb_id", "document_id"],
            ["document.kb_id", "document.id"],
            name="fk_index_asset_same_kb_document",
        ),
        ForeignKeyConstraint(
            ["document_id", "document_version_id"],
            ["document_version.document_id", "document_version.id"],
            name="fk_index_asset_same_document_version",
        ),
        CheckConstraint("width IS NULL OR width > 0", name="index_asset_width_positive"),
        CheckConstraint("height IS NULL OR height > 0", name="index_asset_height_positive"),
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
    indexed_document_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("indexed_document_version.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    asset_key: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    media_type: Mapped[str] = mapped_column(String(255), nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    source_location: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    processing_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = created_timestamp()


class IndexArtifactManifest(Base):
    __tablename__ = "index_artifact_manifest"
    __table_args__ = (
        CheckConstraint("unit_count >= 0", name="artifact_manifest_units_nonnegative"),
        CheckConstraint("asset_count >= 0", name="artifact_manifest_assets_nonnegative"),
        CheckConstraint(
            "representation_count >= 0",
            name="artifact_manifest_representations_nonnegative",
        ),
        CheckConstraint("jsonb_typeof(unit_plan) = 'array'", name="artifact_manifest_unit_plan_array"),
        CheckConstraint(
            "jsonb_typeof(representation_matrix) = 'array'",
            name="artifact_manifest_representation_matrix_array",
        ),
        CheckConstraint(
            "jsonb_typeof(relation_plan) = 'array' AND relation_count >= 0 "
            "AND jsonb_array_length(relation_plan) = relation_count",
            name="artifact_manifest_relation_plan_consistent",
        ),
    )

    indexed_document_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("indexed_document_version.id", ondelete="CASCADE"), primary_key=True
    )
    source_checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    profile_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    element_sequence_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    asset_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    unit_plan: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    representation_matrix: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    unit_count: Mapped[int] = mapped_column(Integer, nullable=False)
    asset_count: Mapped[int] = mapped_column(Integer, nullable=False)
    representation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    relation_plan: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False
    )
    relation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    relation_manifest_hash: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = created_timestamp()


class IndexingJob(Base):
    __tablename__ = "indexing_job"
    __table_args__ = (
        UniqueConstraint("indexed_document_version_id"),
        CheckConstraint("attempt >= 0", name="indexing_job_attempt_nonnegative"),
        CheckConstraint(
            "continuation_count >= 0",
            name="indexing_job_continuation_count_nonnegative",
        ),
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
    progress: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    continuation_pending: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    continuation_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
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
        UniqueConstraint("indexed_document_version_id", "unit_key"),
        UniqueConstraint(
            "indexed_document_version_id", "id", name="uq_index_chunk_target_id"
        ),
        UniqueConstraint("kb_id", "id", name="uq_index_chunk_kb_id"),
        UniqueConstraint(
            "workspace_id", "kb_id", "id", name="uq_index_chunk_workspace_kb_id"
        ),
        CheckConstraint("ordinal >= 0", name="index_chunk_ordinal_nonnegative"),
        CheckConstraint(
            "length(btrim(unit_key)) > 0",
            name="index_chunk_unit_key_nonempty",
        ),
        CheckConstraint("token_count >= 0", name="index_chunk_tokens_nonnegative"),
        CheckConstraint(
            "modality IN ('text', 'image', 'table')",
            name="index_chunk_modality_supported",
        ),
        CheckConstraint(
            "(embedding_text IS NULL) = (embedding_text_hash IS NULL)",
            name="index_chunk_embedding_text_pair",
        ),
        ForeignKeyConstraint(
            ["indexed_document_version_id", "index_asset_id"],
            ["index_asset.indexed_document_version_id", "index_asset.id"],
            name="fk_index_chunk_same_target_asset",
        ),
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
    unit_key: Mapped[str] = mapped_column(String(255), nullable=False)
    modality: Mapped[str] = mapped_column(String(32), nullable=False)
    index_asset_id: Mapped[UUID | None] = mapped_column(PostgreSQLUUID(as_uuid=True))
    evidence_group_key: Mapped[str | None] = mapped_column(String(255))
    relations: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_text: Mapped[str | None] = mapped_column(Text)
    embedding_text_hash: Mapped[str | None] = mapped_column(String(64))
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    source_location: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    hierarchy: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    source_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    excluded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = created_timestamp()


class IndexChunkLexical(Base):
    __tablename__ = "index_chunk_lexical"
    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_id", "kb_id", "indexed_document_version_id"],
            [
                "indexed_document_version.workspace_id",
                "indexed_document_version.kb_id",
                "indexed_document_version.id",
            ],
            name="fk_chunk_lexical_same_scope_target",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["indexed_document_version_id", "index_chunk_id"],
            ["index_chunk.indexed_document_version_id", "index_chunk.id"],
            name="fk_chunk_lexical_same_target_chunk",
            ondelete="CASCADE",
        ),
        Index(
            "ix_index_chunk_lexical_scope",
            "workspace_id",
            "kb_id",
            "analyzer_version",
            "indexed_document_version_id",
        ),
        Index(
            "ix_index_chunk_lexical_tsv",
            "lexical_tsv",
            postgresql_using="gin",
        ),
    )

    index_chunk_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), primary_key=True
    )
    analyzer_version: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False
    )
    kb_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_base.id", ondelete="CASCADE"), nullable=False
    )
    indexed_document_version_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=False
    )
    lexical_text: Mapped[str] = mapped_column(Text, nullable=False)
    lexical_text_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    lexical_tsv: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('simple'::regconfig, lexical_text)",
            persisted=True,
        ),
        nullable=False,
    )
    created_at: Mapped[datetime] = created_timestamp()


class IndexLexicalManifest(Base):
    __tablename__ = "index_lexical_manifest"
    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_id", "kb_id", "indexed_document_version_id"],
            [
                "indexed_document_version.workspace_id",
                "indexed_document_version.kb_id",
                "indexed_document_version.id",
            ],
            name="fk_lexical_manifest_same_scope_target",
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "lexical_chunk_count >= 0",
            name="lexical_manifest_chunk_count_nonnegative",
        ),
    )

    indexed_document_version_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), primary_key=True
    )
    analyzer_version: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False
    )
    kb_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_base.id", ondelete="CASCADE"), nullable=False
    )
    lexical_chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    lexical_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = created_timestamp()


class IndexChunkAssetRelation(Base):
    __tablename__ = "index_chunk_asset_relation"
    __table_args__ = (
        UniqueConstraint(
            "indexed_document_version_id",
            "chunk_id",
            "asset_id",
            "relation_type",
            name="uq_chunk_asset_relation_stable_edge",
        ),
        CheckConstraint(
            "relation_type IN ("
            "'explicit_figure_reference', 'caption_of', 'inline_figure', "
            "'ocr_of', 'table_of', 'spatial_neighbor', 'same_page')",
            name="chunk_asset_relation_type_supported",
        ),
        CheckConstraint(
            "confidence_micros BETWEEN 0 AND 1000000",
            name="chunk_asset_relation_confidence_micros",
        ),
        CheckConstraint(
            "ordinal >= 0", name="chunk_asset_relation_ordinal_nonnegative"
        ),
        ForeignKeyConstraint(
            ["workspace_id", "kb_id", "indexed_document_version_id"],
            [
                "indexed_document_version.workspace_id",
                "indexed_document_version.kb_id",
                "indexed_document_version.id",
            ],
            name="fk_chunk_asset_relation_same_scope_target",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["indexed_document_version_id", "chunk_id"],
            ["index_chunk.indexed_document_version_id", "index_chunk.id"],
            name="fk_chunk_asset_relation_same_target_chunk",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["indexed_document_version_id", "visual_unit_id"],
            ["index_chunk.indexed_document_version_id", "index_chunk.id"],
            name="fk_chunk_asset_relation_same_target_visual",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["indexed_document_version_id", "asset_id"],
            ["index_asset.indexed_document_version_id", "index_asset.id"],
            name="fk_chunk_asset_relation_same_target_asset",
            ondelete="CASCADE",
        ),
        Index(
            "ix_chunk_asset_relation_chunk",
            "workspace_id",
            "kb_id",
            "indexed_document_version_id",
            "chunk_id",
        ),
        Index(
            "ix_chunk_asset_relation_asset",
            "workspace_id",
            "kb_id",
            "indexed_document_version_id",
            "asset_id",
        ),
    )

    id: Mapped[UUID] = uuid_primary_key()
    workspace_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    kb_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    indexed_document_version_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=False
    )
    chunk_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    visual_unit_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=False
    )
    asset_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    relation_type: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence_micros: Mapped[int] = mapped_column(Integer, nullable=False)
    figure_label: Mapped[str | None] = mapped_column(String(128))
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    provenance: Mapped[str] = mapped_column(String(128), nullable=False)
    evidence_group_key: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = created_timestamp()


class VectorRecord(Base):
    __tablename__ = "vector_record"
    __table_args__ = (
        UniqueConstraint(
            "index_chunk_id",
            "embedding_space_id",
            "representation_kind",
            name="uq_vector_record_chunk_space_representation",
        ),
        ForeignKeyConstraint(
            ["kb_id", "index_chunk_id"],
            ["index_chunk.kb_id", "index_chunk.id"],
            name="fk_vector_record_same_kb_chunk",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "embedding_space_id", "embedding_dimension"],
            [
                "embedding_space.workspace_id",
                "embedding_space.id",
                "embedding_space.dimension",
            ],
            name="fk_vector_record_same_workspace_embedding_dimension",
        ),
        CheckConstraint(
            "embedding_dimension BETWEEN 64 AND 4096",
            name="vector_record_dimension_supported",
        ),
        CheckConstraint(
            "vector_dims(embedding) = embedding_dimension",
            name="vector_record_dimension_matches_value",
        ),
        Index(
            "ix_vector_record_space_representation",
            "embedding_space_id",
            "representation_kind",
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
        PostgreSQLUUID(as_uuid=True), nullable=False
    )
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    representation_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
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
        Index(
            "uq_chat_run_session_nonterminal",
            "session_id",
            unique=True,
            postgresql_where=text("status IN ('queued', 'running')"),
        ),
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
    endpoint: Mapped[str] = mapped_column(String(255), nullable=False)
    idempotency_key: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(71), nullable=False)
    requested_policy: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    effective_policy: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    retrieval_strategy: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    model_configuration: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    agent_configuration: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text(
            "jsonb_build_object("
            "'version', 'native_tool_calling_agent_v3', "
            "'budget', jsonb_build_object("
            "'max_model_rounds', 8, 'max_graph_calls', 2, "
            "'max_total_tokens', 150000, 'max_evidence_items', 64, "
            "'max_retrieval_calls', 16))"
        ),
    )
    agent_trace: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    conversation_context: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    contextualized_query: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
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
    document_display_name_snapshot: Mapped[str] = mapped_column(
        String(512), nullable=False
    )
    document_original_filename_snapshot: Mapped[str] = mapped_column(
        String(1024), nullable=False
    )
    quoted_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_location: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    modality: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'text'")
    )
    asset_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    matched_representations: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    score: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = created_timestamp()
