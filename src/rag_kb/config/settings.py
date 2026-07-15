"""Typed, fail-closed application configuration."""

from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    IPvAnyAddress,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_kb.config.profiles import DeploymentProfile


PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]


def parse_environment_boolean(value: object) -> object:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return value


def parse_environment_integer(value: object) -> object:
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return value


EnabledFlag = Annotated[Literal[True], BeforeValidator(parse_environment_boolean)]
DisabledFlag = Annotated[Literal[False], BeforeValidator(parse_environment_boolean)]
FixedDimension = Annotated[Literal[1024], BeforeValidator(parse_environment_integer)]
FixedBatchSize = Annotated[Literal[10], BeforeValidator(parse_environment_integer)]
FixedMaxUploadBytes = Annotated[
    Literal[10_485_760], BeforeValidator(parse_environment_integer)
]
FixedMaxLines = Annotated[Literal[200_000], BeforeValidator(parse_environment_integer)]
FixedMaxChunks = Annotated[Literal[20_000], BeforeValidator(parse_environment_integer)]
FixedParserCpuSeconds = Annotated[Literal[20], BeforeValidator(parse_environment_integer)]
FixedParserMemoryBytes = Annotated[
    Literal[536_870_912], BeforeValidator(parse_environment_integer)
]


class StrictSettingsModel(BaseModel):
    """Shared strict and immutable behavior for nested settings groups."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AppSettings(StrictSettingsModel):
    deployment_profile: DeploymentProfile = DeploymentProfile.DEVELOPMENT
    bind_host: IPvAnyAddress = IPvAnyAddress("127.0.0.1")
    api_port: Annotated[int, Field(ge=1, le=65535)] = 8000
    api_prefix: Literal["/api/v1"] = "/api/v1"

    @model_validator(mode="after")
    def require_local_development(self) -> Self:
        if self.deployment_profile is not DeploymentProfile.DEVELOPMENT:
            raise ValueError("only DEPLOYMENT_PROFILE=development is enabled")
        if not self.bind_host.is_loopback:
            raise ValueError("development bind_host must be a loopback address")
        return self


class IdentitySettings(StrictSettingsModel):
    provider: Literal["development_fixed"] = "development_fixed"
    principal_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")] = (
        "development-principal"
    )
    client_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")] = (
        "development-web"
    )
    workspace_id: UUID = UUID("01900000-0000-7000-8000-000000000001")

    @field_validator("workspace_id")
    @classmethod
    def require_uuidv7_workspace(cls, value: UUID) -> UUID:
        if value.version != 7:
            raise ValueError("development workspace_id must be UUIDv7")
        return value


class SecuritySettings(StrictSettingsModel):
    allowed_cors_origins: tuple[str, ...] = ("http://127.0.0.1:3000",)
    cors_allow_credentials: DisabledFlag = False

    @field_validator("allowed_cors_origins")
    @classmethod
    def require_explicit_loopback_origins(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not value:
            raise ValueError("at least one CORS origin is required")
        if len(value) != len(set(value)):
            raise ValueError("CORS origins must be unique")
        for origin in value:
            parsed = urlsplit(origin)
            if origin == "*" or parsed.scheme not in {"http", "https"}:
                raise ValueError("CORS origins must be explicit HTTP(S) origins")
            if (
                not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
                or parsed.path
            ):
                raise ValueError("CORS origins cannot contain credentials or paths")
            try:
                loopback = ipaddress.ip_address(parsed.hostname).is_loopback
            except ValueError:
                loopback = parsed.hostname == "localhost"
            if not loopback:
                raise ValueError("development CORS origins must be loopback")
        return value


class DatabaseSettings(StrictSettingsModel):
    runtime_dsn: SecretStr
    migration_dsn: SecretStr
    runtime_role: Literal["rag_kb_runtime"] = "rag_kb_runtime"
    migration_role: Literal["rag_kb_migration"] = "rag_kb_migration"
    server_connection_limit: PositiveInt = 50
    reserved_connections: NonNegativeInt = 10
    api_pool_size: PositiveInt = 5
    api_max_overflow: NonNegativeInt = 5
    worker_pool_size: PositiveInt = 8
    worker_max_overflow: NonNegativeInt = 4

    @field_validator("runtime_dsn", "migration_dsn")
    @classmethod
    def require_async_postgresql(cls, value: SecretStr) -> SecretStr:
        raw_value = value.get_secret_value()
        if not raw_value.startswith("postgresql+asyncpg://"):
            raise ValueError("database DSNs must use postgresql+asyncpg")
        return value

    @model_validator(mode="after")
    def validate_roles_and_pool_budget(self) -> Self:
        if self.runtime_role == self.migration_role:
            raise ValueError("runtime_role and migration_role must be different")

        role_urls = (
            ("runtime_dsn", self.runtime_dsn, self.runtime_role),
            ("migration_dsn", self.migration_dsn, self.migration_role),
        )
        for field_name, secret_dsn, expected_role in role_urls:
            parsed_dsn = urlsplit(secret_dsn.get_secret_value())
            if parsed_dsn.username != expected_role:
                raise ValueError(
                    f"{field_name} username must match configured role {expected_role}"
                )
            if not parsed_dsn.password:
                raise ValueError(f"{field_name} must contain a non-empty password")

        if self.reserved_connections >= self.server_connection_limit:
            raise ValueError(
                "reserved_connections must be less than server_connection_limit"
            )
        if self.configured_pool_capacity > self.application_connection_budget:
            raise ValueError(
                "configured API and Worker pools exceed the application "
                "connection budget"
            )
        return self

    @property
    def configured_pool_capacity(self) -> int:
        return (
            self.api_pool_size
            + self.api_max_overflow
            + self.worker_pool_size
            + self.worker_max_overflow
        )

    @property
    def application_connection_budget(self) -> int:
        return self.server_connection_limit - self.reserved_connections

    @property
    def required_api_connections(self) -> int:
        # Ordinary/status request + one short SSE read + one safety slot.
        return 3


class JobPollerSettings(StrictSettingsModel):
    backend: Literal["postgresql"] = "postgresql"
    poll_interval_seconds: PositiveFloat = 1.0
    chat_concurrency: PositiveInt = 1
    indexing_concurrency: PositiveInt = 1
    chat_weight: PositiveInt = 3
    indexing_weight: PositiveInt = 1
    aging_seconds: PositiveFloat = 30.0
    chat_start_target_seconds: PositiveFloat = 2.0
    heartbeat_interval_seconds: PositiveFloat = 10.0
    stale_after_seconds: PositiveFloat = 120.0
    max_attempts: PositiveInt = 3
    retry_base_delay_seconds: PositiveFloat = 5.0
    retry_max_delay_seconds: PositiveFloat = 60.0
    chat_deadline_seconds: PositiveFloat = 120.0
    indexing_deadline_seconds: PositiveFloat = 900.0
    reconciliation_batch_size: PositiveInt = 100

    @model_validator(mode="after")
    def require_safe_stale_timeout(self) -> Self:
        if self.chat_start_target_seconds < self.poll_interval_seconds:
            raise ValueError(
                "chat_start_target_seconds must cover at least one poll interval"
            )
        if self.stale_after_seconds <= self.heartbeat_interval_seconds:
            raise ValueError(
                "stale_after_seconds must be greater than heartbeat_interval_seconds"
            )
        if self.retry_max_delay_seconds < self.retry_base_delay_seconds:
            raise ValueError(
                "retry_max_delay_seconds must be at least retry_base_delay_seconds"
            )
        if self.indexing_deadline_seconds <= self.heartbeat_interval_seconds:
            raise ValueError(
                "indexing_deadline_seconds must exceed heartbeat_interval_seconds"
            )
        if self.chat_deadline_seconds <= self.heartbeat_interval_seconds:
            raise ValueError(
                "chat_deadline_seconds must exceed heartbeat_interval_seconds"
            )
        return self

    @property
    def required_worker_connections(self) -> int:
        lane_capacity = self.chat_concurrency + self.indexing_concurrency
        return lane_capacity * 2 + 3


class ChatDeliverySettings(StrictSettingsModel):
    poll_interval_seconds: PositiveFloat = 1.0
    jitter_ratio: Annotated[float, Field(ge=0, le=0.5)] = 0.2
    max_connection_duration_seconds: PositiveFloat = 600.0
    max_connections_per_principal_run: PositiveInt = 2


class FileStoreSettings(StrictSettingsModel):
    backend: Literal["local"] = "local"
    root_path: Path = Path("/var/lib/rag-kb/sources")
    staging_path: Path = Path("/var/lib/rag-kb/sources/staging")
    final_path: Path = Path("/var/lib/rag-kb/sources/final")
    reconciliation_interval_seconds: PositiveFloat = 30.0
    orphan_grace_seconds: PositiveFloat = 300.0
    reconciliation_batch_size: PositiveInt = 100
    cleanup_max_attempts: PositiveInt = 5
    cleanup_base_delay_seconds: PositiveFloat = 5.0

    @model_validator(mode="after")
    def require_one_absolute_storage_tree(self) -> Self:
        if not all(
            path.is_absolute()
            for path in (self.root_path, self.staging_path, self.final_path)
        ):
            raise ValueError("file-store paths must be absolute")
        if self.staging_path == self.final_path:
            raise ValueError("staging_path and final_path must be different")
        for name, path in (
            ("staging_path", self.staging_path),
            ("final_path", self.final_path),
        ):
            if not path.is_relative_to(self.root_path):
                raise ValueError(f"{name} must be located beneath root_path")
        return self


class MaintenanceSettings(StrictSettingsModel):
    batch_size: PositiveInt = 100
    retired_data_grace_seconds: PositiveFloat = 300.0
    task_retention_seconds: PositiveFloat = 604_800.0

    @model_validator(mode="after")
    def require_task_retention_after_data_grace(self) -> Self:
        if self.task_retention_seconds <= self.retired_data_grace_seconds:
            raise ValueError(
                "task_retention_seconds must exceed retired_data_grace_seconds"
            )
        return self


class FileAdmissionSettings(StrictSettingsModel):
    max_bytes: FixedMaxUploadBytes = 10_485_760
    max_lines: FixedMaxLines = 200_000


class ParserSettings(StrictSettingsModel):
    profile: Literal["plain_text_test_v1"] = "plain_text_test_v1"
    max_chunks: FixedMaxChunks = 20_000
    wall_seconds: PositiveFloat = 30.0

    @field_validator("wall_seconds")
    @classmethod
    def require_fixed_wall_timeout(cls, value: float) -> float:
        if value != 30.0:
            raise ValueError("parser wall_seconds is fixed at 30")
        return value
    cpu_seconds: FixedParserCpuSeconds = 20
    memory_bytes: FixedParserMemoryBytes = 536_870_912


class VectorStoreSettings(StrictSettingsModel):
    backend: Literal["pgvector"] = "pgvector"
    exact_search: EnabledFlag = True
    hnsw_enabled: DisabledFlag = False


class ProviderSettings(StrictSettingsModel):
    base_url: AnyHttpUrl
    api_key: SecretStr
    logical_endpoint_identity: str
    model: str
    timeout_seconds: PositiveFloat = 30.0
    max_retries: NonNegativeInt = 2
    max_concurrency: PositiveInt = 2

    @field_validator("api_key")
    @classmethod
    def require_non_empty_api_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value():
            raise ValueError("provider api_key must not be empty")
        return value


class ChatProviderSettings(ProviderSettings):
    logical_endpoint_identity: Literal["deepseek-official-chat"] = (
        "deepseek-official-chat"
    )
    provider_identity: Literal["deepseek"] = "deepseek"
    model: Literal["deepseek-v4-flash"] = "deepseek-v4-flash"
    resolved_model: Literal["deepseek-v4-flash"] = "deepseek-v4-flash"
    model_version: Literal["DeepSeek-V4-Flash"] = "DeepSeek-V4-Flash"
    structured_output_mode: Literal["json_object"] = "json_object"
    configuration_fingerprint: Literal[
        "sha256:f65fee43dc8b5886c78ceb1116514e407d539d7f4b9378a8b8073e2de8620b04"
    ] = "sha256:f65fee43dc8b5886c78ceb1116514e407d539d7f4b9378a8b8073e2de8620b04"
    capability_fingerprint: Literal[
        "sha256:b09e82393dc80d4326ce8bf5040025bf046e31e7828d0ecdec76c02aefe11d76"
    ] = "sha256:b09e82393dc80d4326ce8bf5040025bf046e31e7828d0ecdec76c02aefe11d76"


class EmbeddingProviderSettings(ProviderSettings):
    logical_endpoint_identity: Literal[
        "alibaba-model-studio-beijing-embedding"
    ] = "alibaba-model-studio-beijing-embedding"
    provider_identity: Literal["alibaba-cloud-model-studio-qwen"] = (
        "alibaba-cloud-model-studio-qwen"
    )
    model: Literal["text-embedding-v4"] = "text-embedding-v4"
    resolved_model: Literal["text-embedding-v4"] = "text-embedding-v4"
    model_version: Literal["text-embedding-v4 (Qwen3-Embedding series)"] = (
        "text-embedding-v4 (Qwen3-Embedding series)"
    )
    dimension: FixedDimension = 1024
    metric: Literal["cosine"] = "cosine"
    vector_data_type: Literal["float32"] = "float32"
    normalization: Literal["l2"] = "l2"
    max_batch_size: FixedBatchSize = 10
    configuration_fingerprint: Literal[
        "sha256:c135eb852aefd97be80fd82dd168f7cf1ccff0c99eb4fcfdbbf3e337cedeee66"
    ] = "sha256:c135eb852aefd97be80fd82dd168f7cf1ccff0c99eb4fcfdbbf3e337cedeee66"
    compatibility_fingerprint: Literal[
        "sha256:7bd706a3642d7ee17a5a0112a3e0d7e50abaa26bf19c5511d240f222ae4d1153"
    ] = "sha256:7bd706a3642d7ee17a5a0112a3e0d7e50abaa26bf19c5511d240f222ae4d1153"


class ModelProviderSettings(StrictSettingsModel):
    chat: ChatProviderSettings
    embedding: EmbeddingProviderSettings
    rerank_enabled: DisabledFlag = False


class RetrievalSettings(StrictSettingsModel):
    strategy: Literal["exact_vector"] = "exact_vector"
    top_k: Annotated[int, Field(ge=1, le=100)] = 10
    hybrid_enabled: DisabledFlag = False
    rerank_enabled: DisabledFlag = False


class WorkflowSettings(StrictSettingsModel):
    runner: Literal["direct"] = "direct"
    langgraph_enabled: DisabledFlag = False
    checkpoint_recovery_enabled: DisabledFlag = False


class DeliveryReliabilitySettings(StrictSettingsModel):
    second_queue_enabled: DisabledFlag = False
    retained_event_replay_enabled: DisabledFlag = False
    multi_runner_recovery_enabled: DisabledFlag = False
    outbox_delivery_enabled: DisabledFlag = False


class ObservabilitySettings(StrictSettingsModel):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json"] = "json"
    include_content: DisabledFlag = False
    tracing_enabled: bool = False


class Settings(BaseSettings):
    """Complete P0/P1A process settings loaded from environment or `.env`."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_KB__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        frozen=True,
        nested_model_default_partial_update=True,
    )

    app: AppSettings = Field(default_factory=AppSettings)
    identity: IdentitySettings = Field(default_factory=IdentitySettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    database: DatabaseSettings
    job_poller: JobPollerSettings = Field(default_factory=JobPollerSettings)
    chat_delivery: ChatDeliverySettings = Field(default_factory=ChatDeliverySettings)
    file_store: FileStoreSettings = Field(default_factory=FileStoreSettings)
    maintenance: MaintenanceSettings = Field(default_factory=MaintenanceSettings)
    file_admission: FileAdmissionSettings = Field(default_factory=FileAdmissionSettings)
    parser: ParserSettings = Field(default_factory=ParserSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    model_provider: ModelProviderSettings
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    workflow: WorkflowSettings = Field(default_factory=WorkflowSettings)
    delivery_reliability: DeliveryReliabilitySettings = Field(
        default_factory=DeliveryReliabilitySettings
    )
    observability: ObservabilitySettings = Field(
        default_factory=ObservabilitySettings
    )

    @model_validator(mode="after")
    def require_worker_scheduling_budget(self) -> Self:
        api_capacity = self.database.api_pool_size + self.database.api_max_overflow
        if api_capacity < self.database.required_api_connections:
            raise ValueError(
                "API database pool cannot cover requests, SSE reads, and safety margin"
            )
        worker_capacity = (
            self.database.worker_pool_size + self.database.worker_max_overflow
        )
        if worker_capacity < self.job_poller.required_worker_connections:
            raise ValueError(
                "Worker database pool cannot cover lanes, heartbeats, polling, "
                "reconciliation, and safety margin"
            )
        longest_operation = max(
            self.parser.wall_seconds,
            self.model_provider.chat.timeout_seconds,
            self.model_provider.embedding.timeout_seconds,
        )
        if self.job_poller.stale_after_seconds <= longest_operation:
            raise ValueError(
                "stale_after_seconds must exceed every bounded indexing operation"
            )
        return self


def load_settings(*, env_file: str | Path | None = ".env") -> Settings:
    """Load a fresh immutable settings graph without global import-time state."""

    return Settings(_env_file=env_file)  # type: ignore[call-arg]
