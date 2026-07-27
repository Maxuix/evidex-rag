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
FixedCrossModalDimension = Annotated[
    Literal[768], BeforeValidator(parse_environment_integer)
]
FixedBatchSize = Annotated[Literal[10], BeforeValidator(parse_environment_integer)]
FixedMaxUploadBytes = Annotated[
    Literal[10_485_760], BeforeValidator(parse_environment_integer)
]
FixedMaxMarkdownBundleBytes = Annotated[
    Literal[20_971_520], BeforeValidator(parse_environment_integer)
]
FixedMaxLines = Annotated[Literal[200_000], BeforeValidator(parse_environment_integer)]
FixedMaxArchiveEntries = Annotated[
    Literal[10_000], BeforeValidator(parse_environment_integer)
]
FixedMaxExpandedBytes = Annotated[
    Literal[104_857_600], BeforeValidator(parse_environment_integer)
]
FixedMaxChunks = Annotated[Literal[20_000], BeforeValidator(parse_environment_integer)]
FixedMaxExtractedCharacters = Annotated[
    Literal[5_000_000], BeforeValidator(parse_environment_integer)
]
FixedMaxMetadataBytes = Annotated[
    Literal[65_536], BeforeValidator(parse_environment_integer)
]
FixedMaxDoclingPages = Annotated[
    Literal[500], BeforeValidator(parse_environment_integer)
]
FixedDoclingTimeoutSeconds = Annotated[
    Literal[600], BeforeValidator(parse_environment_integer)
]
FixedMaxDoclingItems = Annotated[
    Literal[20_000], BeforeValidator(parse_environment_integer)
]
FixedContextTurns = Annotated[Literal[6], BeforeValidator(parse_environment_integer)]
FixedContextTokens = Annotated[
    Literal[4000], BeforeValidator(parse_environment_integer)
]
FixedVisualImageCount = Annotated[
    Literal[2], BeforeValidator(parse_environment_integer)
]
FixedVisualImageBytes = Annotated[
    Literal[5_242_880], BeforeValidator(parse_environment_integer)
]
FixedVisualTotalBytes = Annotated[
    Literal[12_582_912], BeforeValidator(parse_environment_integer)
]
FixedVisualPixels = Annotated[
    Literal[16_000_000], BeforeValidator(parse_environment_integer)
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


class SessionContextSettings(StrictSettingsModel):
    strategy: Literal["recent_completed_turns_v1"] = "recent_completed_turns_v1"
    max_turns: FixedContextTurns = 6
    max_context_tokens: FixedContextTokens = 4000
    tokenizer: Literal["cl100k_base"] = "cl100k_base"
    query_schema: Literal["contextual_query_v2"] = "contextual_query_v2"
    configuration_fingerprint: Literal[
        "sha256:7de39f7456bdf97eb6c41deab2b916d4f77c072d175faf546f297137763a1d20"
    ] = "sha256:7de39f7456bdf97eb6c41deab2b916d4f77c072d175faf546f297137763a1d20"


class FileStoreSettings(StrictSettingsModel):
    backend: Literal["local"] = "local"
    root_path: Path = Path("/var/lib/rag-kb/sources")
    staging_path: Path = Path("/var/lib/rag-kb/sources/staging")
    final_path: Path = Path("/var/lib/rag-kb/sources/final")
    asset_staging_path: Path | None = None
    asset_final_path: Path | None = None
    parser_temp_path: Path | None = None
    reconciliation_interval_seconds: PositiveFloat = 30.0
    orphan_grace_seconds: PositiveFloat = 300.0
    reconciliation_batch_size: PositiveInt = 100
    cleanup_max_attempts: PositiveInt = 5
    cleanup_base_delay_seconds: PositiveFloat = 5.0

    @model_validator(mode="after")
    def require_one_absolute_storage_tree(self) -> Self:
        if self.asset_staging_path is None:
            object.__setattr__(self, "asset_staging_path", self.root_path / "asset-staging")
        if self.asset_final_path is None:
            object.__setattr__(self, "asset_final_path", self.root_path / "assets")
        if self.parser_temp_path is None:
            object.__setattr__(self, "parser_temp_path", self.root_path / "parser-temp")
        assert self.asset_staging_path is not None
        assert self.asset_final_path is not None
        assert self.parser_temp_path is not None
        if not all(
            path.is_absolute()
            for path in (
                self.root_path,
                self.staging_path,
                self.final_path,
                self.asset_staging_path,
                self.asset_final_path,
                self.parser_temp_path,
            )
        ):
            raise ValueError("file-store paths must be absolute")
        managed = {
            self.staging_path,
            self.final_path,
            self.asset_staging_path,
            self.asset_final_path,
            self.parser_temp_path,
        }
        if len(managed) != 5:
            raise ValueError("file-store managed paths must be different")
        for name, path in (
            ("staging_path", self.staging_path),
            ("final_path", self.final_path),
            ("asset_staging_path", self.asset_staging_path),
            ("asset_final_path", self.asset_final_path),
            ("parser_temp_path", self.parser_temp_path),
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
    max_markdown_bundle_bytes: FixedMaxMarkdownBundleBytes = 20_971_520
    max_lines: FixedMaxLines = 200_000
    max_archive_entries: FixedMaxArchiveEntries = 10_000
    max_expanded_bytes: FixedMaxExpandedBytes = 104_857_600


class ParserSettings(StrictSettingsModel):
    profile: Literal["docling_text_local_v1"] = "docling_text_local_v1"
    docling_artifacts_path: Path = Path("/opt/rag-kb/docling-artifacts")
    docling_artifact_manifest_path: Path = Path(
        "/app/config/docling-artifacts-v1.json"
    )
    max_file_size: FixedMaxUploadBytes = 10_485_760
    max_markdown_bundle_size: FixedMaxMarkdownBundleBytes = 20_971_520
    max_num_pages: FixedMaxDoclingPages = 500
    document_timeout_seconds: FixedDoclingTimeoutSeconds = 600
    max_docling_items: FixedMaxDoclingItems = 20_000
    max_chunks: FixedMaxChunks = 20_000
    max_extracted_characters: FixedMaxExtractedCharacters = 5_000_000
    max_metadata_bytes: FixedMaxMetadataBytes = 65_536
    max_assets: Annotated[int, Field(ge=1, le=10_000)] = 1_000
    max_total_asset_bytes: Annotated[int, Field(ge=1, le=1_073_741_824)] = 104_857_600
    max_image_pixels: Annotated[int, Field(ge=1, le=100_000_000)] = 40_000_000
    max_image_width: Annotated[int, Field(ge=1, le=32_768)] = 16_384
    max_image_height: Annotated[int, Field(ge=1, le=32_768)] = 16_384
    max_ocr_characters: Annotated[int, Field(ge=1, le=10_000_000)] = 2_000_000
    max_ocr_tokens: Annotated[int, Field(ge=1, le=1_000_000)] = 500_000
    max_caption_tokens: Annotated[int, Field(ge=1, le=4_096)] = 512
    max_table_html_bytes: Annotated[int, Field(ge=1, le=10_485_760)] = 1_048_576
    max_units: Annotated[int, Field(ge=1, le=100_000)] = 20_000
    max_representations: Annotated[int, Field(ge=1, le=300_000)] = 60_000


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
    logical_endpoint_identity: Literal[
        "alibaba-model-studio-beijing-chat"
    ] = (
        "alibaba-model-studio-beijing-chat"
    )
    provider_identity: Literal["alibaba-cloud-model-studio-qwen"] = (
        "alibaba-cloud-model-studio-qwen"
    )
    model: Literal["qwen3.7-plus"] = "qwen3.7-plus"
    resolved_model: Literal["qwen3.7-plus"] = "qwen3.7-plus"
    model_version: Literal["qwen3.7-plus"] = "qwen3.7-plus"
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.1
    max_tokens: PositiveInt = 2048
    structured_output_mode: Literal["json_object"] = "json_object"
    thinking_enabled: DisabledFlag = False
    vision_enabled: EnabledFlag = True
    max_visual_images: FixedVisualImageCount = 2
    max_visual_image_bytes: FixedVisualImageBytes = 5_242_880
    max_visual_total_bytes: FixedVisualTotalBytes = 12_582_912
    max_visual_pixels: FixedVisualPixels = 16_000_000
    visual_media_profile: Literal["jpeg_png_webp_v1"] = "jpeg_png_webp_v1"
    configuration_fingerprint: Literal[
        "sha256:85b03b3eececbb2fda11cadf77040999028b09e335b8ef5684c34a34601a71d2"
    ] = "sha256:85b03b3eececbb2fda11cadf77040999028b09e335b8ef5684c34a34601a71d2"
    capability_fingerprint: Literal[
        "sha256:c24fb9b08baf600afc8f4610f88412ca5c7a2979890647e1512e7cedb8066279"
    ] = "sha256:c24fb9b08baf600afc8f4610f88412ca5c7a2979890647e1512e7cedb8066279"


class EmbeddingProviderSettings(ProviderSettings):
    logical_endpoint_identity: Literal[
        "alibaba-model-studio-beijing-embedding"
    ] = "alibaba-model-studio-beijing-embedding"
    provider_identity: Literal["alibaba-cloud-model-studio-qwen"] = (
        "alibaba-cloud-model-studio-qwen"
    )
    model: Literal["qwen3.7-text-embedding"] = "qwen3.7-text-embedding"
    resolved_model: Literal["qwen3.7-text-embedding"] = (
        "qwen3.7-text-embedding"
    )
    model_version: Literal["qwen3.7-text-embedding"] = "qwen3.7-text-embedding"
    dimension: FixedDimension = 1024
    metric: Literal["cosine"] = "cosine"
    vector_data_type: Literal["float32"] = "float32"
    normalization: Literal["l2"] = "l2"
    max_batch_size: FixedBatchSize = 10
    configuration_fingerprint: Literal[
        "sha256:5f774411565f9aaef04c7a9762bf6e245589cff064c396a8c5b38eb9098ac18f"
    ] = "sha256:5f774411565f9aaef04c7a9762bf6e245589cff064c396a8c5b38eb9098ac18f"
    compatibility_fingerprint: Literal[
        "sha256:398af80b01c3e440c0edf5871de60f80fdab255f453bfa6c685c65e2f9be61c7"
    ] = "sha256:398af80b01c3e440c0edf5871de60f80fdab255f453bfa6c685c65e2f9be61c7"


class MultimodalEmbeddingProviderSettings(ProviderSettings):
    provider_identity: Literal["alibaba-cloud-model-studio-qwen"] = (
        "alibaba-cloud-model-studio-qwen"
    )
    logical_endpoint_identity: Literal[
        "alibaba-model-studio-beijing-multimodal-embedding"
    ] = "alibaba-model-studio-beijing-multimodal-embedding"
    model: Literal["tongyi-embedding-vision-flash-2026-03-06"] = (
        "tongyi-embedding-vision-flash-2026-03-06"
    )
    resolved_model: Literal["tongyi-embedding-vision-flash-2026-03-06"] = (
        "tongyi-embedding-vision-flash-2026-03-06"
    )
    model_version: Literal["tongyi-embedding-vision-flash-2026-03-06"] = (
        "tongyi-embedding-vision-flash-2026-03-06"
    )
    dimension: FixedCrossModalDimension = 768
    metric: Literal["cosine"] = "cosine"
    vector_data_type: Literal["float32"] = "float32"
    normalization: Literal["l2"] = "l2"
    max_batch_size: Annotated[int, Field(ge=1, le=20)] = 20
    text_query_template: Literal["query: {text}"] = "query: {text}"
    image_resize_policy: Literal["provider_res_level_1_no_crop_v1"] = (
        "provider_res_level_1_no_crop_v1"
    )
    color_space: Literal["RGB"] = "RGB"
    configuration_fingerprint: Literal[
        "sha256:a3c9bcf7f049967db37f8bb59bb16504a0f371a14d0adfc793338d0eeecd856b"
    ] = "sha256:a3c9bcf7f049967db37f8bb59bb16504a0f371a14d0adfc793338d0eeecd856b"
    compatibility_fingerprint: Literal[
        "sha256:953af16a5423f52cfdb65efb499a7181212be4760636f9a4ef428a959d9d9da2"
    ] = "sha256:953af16a5423f52cfdb65efb499a7181212be4760636f9a4ef428a959d9d9da2"


class ModelProviderSettings(StrictSettingsModel):
    chat: ChatProviderSettings
    embedding: EmbeddingProviderSettings
    multimodal_embedding: MultimodalEmbeddingProviderSettings | None = None
    rerank_enabled: DisabledFlag = False


class RetrievalSettings(StrictSettingsModel):
    strategy: Literal["exact_vector"] = "exact_vector"
    top_k: Annotated[int, Field(ge=1, le=100)] = 10
    min_cosine_similarity: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.35
    min_rerank_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.45
    candidate_multiplier: Annotated[int, Field(ge=2, le=8)] = 4
    max_candidate_count: Annotated[int, Field(ge=10, le=100)] = 40
    vector_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.65
    lexical_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.35
    mmr_lambda: Annotated[float, Field(gt=0.0, le=1.0)] = 0.75
    hybrid_enabled: DisabledFlag = False
    rerank_enabled: bool = True
    cross_modal_candidate_count: Annotated[int, Field(ge=1, le=100)] = 20
    cross_modal_min_cosine_similarity: Annotated[
        float, Field(ge=-1.0, le=1.0)
    ] = 0.25
    rrf_k: Annotated[int, Field(ge=1, le=1000)] = 60
    cross_modal_weight_micros: Annotated[int, Field(ge=1, le=10_000_000)] = 1_000_000

    @model_validator(mode="after")
    def require_rerank_weights_sum_to_one(self) -> Self:
        if abs(self.vector_weight + self.lexical_weight - 1.0) > 1e-9:
            raise ValueError("rerank weights must sum to one")
        return self


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
    session_context: SessionContextSettings = Field(
        default_factory=SessionContextSettings
    )
    file_store: FileStoreSettings = Field(default_factory=FileStoreSettings)
    maintenance: MaintenanceSettings = Field(default_factory=MaintenanceSettings)
    file_admission: FileAdmissionSettings = Field(default_factory=FileAdmissionSettings)
    parser: ParserSettings = Field(default_factory=ParserSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    model_provider: ModelProviderSettings
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
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
        operation_timeouts = [
            self.model_provider.chat.timeout_seconds,
            self.model_provider.embedding.timeout_seconds,
        ]
        if self.model_provider.multimodal_embedding is not None:
            operation_timeouts.append(
                self.model_provider.multimodal_embedding.timeout_seconds
            )
        longest_operation = max(operation_timeouts)
        if self.job_poller.stale_after_seconds <= longest_operation:
            raise ValueError(
                "stale_after_seconds must exceed every bounded indexing operation"
            )
        return self


def load_settings(*, env_file: str | Path | None = ".env") -> Settings:
    """Load a fresh immutable settings graph without global import-time state."""

    return Settings(_env_file=env_file)  # type: ignore[call-arg]
