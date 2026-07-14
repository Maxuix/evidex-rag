"""Typed, fail-closed application configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

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


class JobPollerSettings(StrictSettingsModel):
    backend: Literal["postgresql"] = "postgresql"
    poll_interval_seconds: PositiveFloat = 1.0
    chat_concurrency: PositiveInt = 1
    indexing_concurrency: PositiveInt = 1
    chat_weight: PositiveInt = 3
    indexing_weight: PositiveInt = 1
    aging_seconds: PositiveFloat = 30.0
    heartbeat_interval_seconds: PositiveFloat = 10.0
    stale_after_seconds: PositiveFloat = 120.0
    max_attempts: PositiveInt = 3

    @model_validator(mode="after")
    def require_safe_stale_timeout(self) -> Self:
        if self.stale_after_seconds <= self.heartbeat_interval_seconds:
            raise ValueError(
                "stale_after_seconds must be greater than heartbeat_interval_seconds"
            )
        return self


class FileStoreSettings(StrictSettingsModel):
    backend: Literal["local"] = "local"
    root_path: Path = Path("/var/lib/rag-kb/sources")
    staging_path: Path = Path("/var/lib/rag-kb/sources/staging")
    final_path: Path = Path("/var/lib/rag-kb/sources/final")

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
    database: DatabaseSettings
    job_poller: JobPollerSettings = Field(default_factory=JobPollerSettings)
    file_store: FileStoreSettings = Field(default_factory=FileStoreSettings)
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


def load_settings(*, env_file: str | Path | None = ".env") -> Settings:
    """Load a fresh immutable settings graph without global import-time state."""

    return Settings(_env_file=env_file)  # type: ignore[call-arg]
