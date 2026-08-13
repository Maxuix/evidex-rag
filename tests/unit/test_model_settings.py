from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import httpx
from pydantic import ValidationError

from apps.api.routers.model_settings import _profile_response
from apps.worker.dependencies import _chat_model_loader
from rag_kb.adapters.model_api.model_catalog import (
    OpenAICompatibleModelCatalogAdapter,
    parse_model_catalog,
)
from rag_kb.adapters.model_api.routing_chat import RoutingChatModelAdapter
from rag_kb.adapters.model_secrets.local import LocalModelSecretStore
from rag_kb.db.models import ModelProfileRevision as ModelProfileRevisionRow
from rag_kb.domain import (
    ChatModelExecutionError,
    ChatModelMessage,
    ChatModelRequest,
    ChatModelResponse,
    EmbeddingDimensionRequestMode,
    EmbeddingDimensionSelectionSource,
    EmbeddingExecutionMode,
    EmbeddingInputCapability,
    EmbeddingValidationSnapshot,
    ModelKind,
    ModelProfile,
    ModelProfileBundle,
    ModelProfileRevision,
    ModelProvider,
    ModelProviderBundle,
    ModelProviderProtocol,
    ModelProviderRevision,
    ModelValidationStatus,
    ResourceStateConflictError,
    derive_embedding_execution_mode,
    select_automatic_embedding_dimension,
)
from rag_kb.schemas.model_settings import ModelProfileCreate, ModelProviderCreate
from rag_kb.services.model_settings import (
    _require_parameters,
    embedding_capability_fingerprint,
    embedding_compatibility_fingerprint,
    model_fingerprints,
    provider_fingerprint,
)


class _ChatModel:
    def __init__(self, model: str) -> None:
        self.model = model

    async def complete(self, request: ChatModelRequest) -> ChatModelResponse:
        return ChatModelResponse(self.model, self.model, None, None, {})


class ModelSettingsTests(unittest.TestCase):
    def test_profile_response_ignores_retired_chat_configuration_fields(self) -> None:
        now = datetime.now(UTC)
        workspace_id = uuid4()
        provider_id = uuid4()
        profile_id = uuid4()
        provider_revision_id = uuid4()
        profile = ModelProfile(
            id=profile_id,
            workspace_id=workspace_id,
            provider_id=provider_id,
            name="Chat",
            kind=ModelKind.CHAT,
            enabled=True,
            created_at=now,
            updated_at=now,
        )
        revision = ModelProfileRevision(
            id=uuid4(),
            workspace_id=workspace_id,
            profile_id=profile_id,
            provider_revision_id=provider_revision_id,
            revision=1,
            model="chat-model",
            configuration={
                "type": "chat",
                "temperature": 0.45,
                "api_mode": "responses",
                "tool_choice_mode": "auto",
            },
            configuration_fingerprint="sha256:profile",
            capability_fingerprint="sha256:capability",
            compatibility_fingerprint=None,
            validation_status=ModelValidationStatus.VALID,
            validation_error_code=None,
            validation_snapshot=None,
            validated_at=now,
            created_at=now,
        )

        response = _profile_response(Mock(profile=profile, current_revision=revision))

        self.assertEqual(response.parameters.type, "chat")
        self.assertEqual(response.parameters.temperature, 0.45)
        self.assertNotIn("api_mode", response.parameters.model_dump(mode="json"))
        self.assertNotIn("tool_choice_mode", response.parameters.model_dump(mode="json"))

    def test_provider_create_defaults_are_chat_friendly_and_ui_aligned(self) -> None:
        provider = ModelProviderCreate.model_validate({
            "name": "Any compatible provider",
            "protocol": "openai_compatible",
            "base_url": "https://provider.invalid/v1",
            "api_key": "secret",
        })
        self.assertEqual(provider.timeout_seconds, 60)
        self.assertEqual(provider.max_retries, 1)
        self.assertEqual(provider.max_concurrency, 2)

        source = (
            Path(__file__).resolve().parents[2]
            / "apps"
            / "web-chat"
            / "src"
            / "ModelSettingsDialog.tsx"
        ).read_text(encoding="utf-8")
        self.assertIn("DEFAULT_PROVIDER_TIMEOUT_SECONDS = 60", source)
        self.assertIn("DEFAULT_PROVIDER_MAX_RETRIES = 1", source)
        self.assertIn("DEFAULT_PROVIDER_MAX_CONCURRENCY = 2", source)

    def test_dynamic_chat_retry_budget_is_provider_name_independent(self) -> None:
        now = datetime.now(UTC)

        def bundle(
            provider_name: str,
            *,
            timeout: float,
            retries: int,
        ) -> ModelProfileBundle:
            workspace_id = uuid4()
            provider_id = uuid4()
            profile_id = uuid4()
            provider_revision = ModelProviderRevision(
                id=uuid4(),
                workspace_id=workspace_id,
                provider_id=provider_id,
                revision=1,
                protocol=ModelProviderProtocol.OPENAI_COMPATIBLE,
                base_url="https://provider.invalid/v1",
                secret_reference=str(uuid4()),
                timeout_seconds=timeout,
                max_retries=retries,
                max_concurrency=2,
                configuration_fingerprint="sha256:provider",
                created_at=now,
            )
            return ModelProfileBundle(
                profile=ModelProfile(
                    id=profile_id,
                    workspace_id=workspace_id,
                    provider_id=provider_id,
                    name=f"{provider_name} chat",
                    kind=ModelKind.CHAT,
                    enabled=True,
                    created_at=now,
                    updated_at=now,
                ),
                current_revision=ModelProfileRevision(
                    id=uuid4(),
                    workspace_id=workspace_id,
                    profile_id=profile_id,
                    provider_revision_id=provider_revision.id,
                    revision=1,
                    model="chat-model",
                    configuration={"type": "chat"},
                    configuration_fingerprint="sha256:profile",
                    capability_fingerprint="sha256:capability",
                    compatibility_fingerprint=None,
                    validation_status=ModelValidationStatus.VALID,
                    validation_error_code=None,
                    validation_snapshot=None,
                    validated_at=now,
                    created_at=now,
                ),
                provider=ModelProvider(
                    id=provider_id,
                    workspace_id=workspace_id,
                    name=provider_name,
                    enabled=True,
                    created_at=now,
                    updated_at=now,
                ),
                provider_revision=provider_revision,
            )

        async def scenario() -> None:
            for provider_name in ("Mimo-shaped name", "unrelated provider"):
                invalid = bundle(provider_name, timeout=60, retries=2)
                with patch(
                    "apps.worker.dependencies.execute_in_transaction",
                    new=AsyncMock(return_value=invalid),
                ):
                    loader = _chat_model_loader(
                        Mock(),
                        Mock(),
                        chat_deadline_seconds=301,
                    )
                    with self.assertRaises(ChatModelExecutionError) as raised:
                        await loader(invalid.current_revision.id)
                self.assertEqual(
                    raised.exception.diagnostic,
                    {"check": "chat_retry_budget"},
                )

            accepted = bundle("any provider", timeout=60, retries=1)
            secret_store = Mock()
            secret_store.read.return_value = "secret"
            with (
                patch(
                    "apps.worker.dependencies.execute_in_transaction",
                    new=AsyncMock(return_value=accepted),
                ),
                patch(
                    "apps.worker.dependencies.LangChainChatModelAdapter",
                    return_value="adapter",
                ),
            ):
                loader = _chat_model_loader(
                    Mock(),
                    secret_store,
                    chat_deadline_seconds=420,
                )
                self.assertEqual(
                    await loader(accepted.current_revision.id),
                    "adapter",
                )

        asyncio.run(scenario())

    def test_validation_snapshot_none_is_bound_as_sql_null(self) -> None:
        column_type = ModelProfileRevisionRow.__table__.c.validation_snapshot.type
        self.assertTrue(column_type.none_as_null)

    def test_secret_store_uses_opaque_mode_0600_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalModelSecretStore(Path(directory))
            reference = store.write("super-secret")
            path = Path(directory) / reference

            self.assertNotIn("super-secret", reference)
            self.assertEqual(path.read_text(encoding="utf-8"), "super-secret")
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            self.assertEqual(store.read(reference), "super-secret")

    def test_profile_schema_keeps_chat_and_embedding_parameters_distinct(self) -> None:
        provider_id = uuid4()
        value = ModelProfileCreate.model_validate({
            "provider_id": str(provider_id),
            "name": "Chat",
            "kind": "chat",
            "model": "model-a",
            "parameters": {
                "type": "chat",
                "temperature": 0.4,
                "sampling_top_k": 20,
                "reasoning_effort": "high",
            },
        })
        self.assertEqual(value.parameters.sampling_top_k, 20)
        defaults = ModelProfileCreate.model_validate({
            "provider_id": str(provider_id),
            "name": "Knowledge Chat",
            "kind": "chat",
            "model": "model-b",
            "parameters": {"type": "chat"},
        })
        self.assertEqual(defaults.parameters.temperature, 0.2)
        self.assertEqual(defaults.parameters.top_p, 0.9)
        self.assertEqual(defaults.parameters.sampling_top_k, 40)
        self.assertEqual(defaults.parameters.max_output_tokens, 4096)
        maximum = ModelProfileCreate.model_validate({
            "provider_id": str(provider_id),
            "name": "Long Knowledge Chat",
            "kind": "chat",
            "model": "model-c",
            "parameters": {"type": "chat", "max_output_tokens": 8192},
        })
        self.assertEqual(maximum.parameters.max_output_tokens, 8192)
        with self.assertRaises(ValidationError):
            ModelProfileCreate.model_validate({
                "provider_id": str(provider_id),
                "name": "Too Long",
                "kind": "chat",
                "model": "model-d",
                "parameters": {"type": "chat", "max_output_tokens": 8193},
            })
        request = ChatModelRequest(
            (ChatModelMessage("user", "question"),),
            max_output_tokens=8192,
        )
        self.assertEqual(request.max_output_tokens, 8192)
        with self.assertRaises(ValueError):
            ChatModelRequest(
                (ChatModelMessage("user", "question"),),
                max_output_tokens=8193,
            )
        embedding = ModelProfileCreate.model_validate({
                "provider_id": str(provider_id),
                "name": "Embedding",
                "kind": "text_embedding",
                "model": "embedding-a",
                "parameters": {"type": "embedding", "dimension": 724},
            })
        self.assertEqual(embedding.parameters.dimension, 724)
        automatic = ModelProfileCreate.model_validate({
            "provider_id": str(provider_id),
            "name": "Automatic embedding",
            "kind": "text_embedding",
            "model": "embedding-b",
            "parameters": {"type": "embedding"},
        })
        self.assertEqual(automatic.parameters.dimension, "auto")
        for dimension in (63, 4097):
            with self.assertRaises(ValidationError):
                ModelProfileCreate.model_validate({
                    "provider_id": str(provider_id),
                    "name": "Invalid embedding",
                    "kind": "text_embedding",
                    "model": "embedding-invalid",
                    "parameters": {"type": "embedding", "dimension": dimension},
                })
        with self.assertRaises(ValidationError):
            ModelProfileCreate.model_validate({
                "provider_id": str(provider_id),
                "name": "Invalid shared embedding",
                "kind": "text_embedding",
                "model": "embedding-invalid",
                "parameters": {
                    "type": "embedding",
                    "shared_text_image_space_confirmed": True,
                },
            })

    def test_embedding_dimension_selection_snapshot_and_execution_mode(self) -> None:
        cases = (
            ((724, 1024, 2048), None, None, 1024, "automatic_1024"),
            ((724, 2048), None, None, 2048, "automatic_above_1024"),
            ((256, 724), None, None, 724, "automatic_below_1024"),
            ((724, 1024, 2048), 2048, None, 2048, "provider_recommended"),
            ((724, 1024, 2048), None, 724, 724, "provider_default"),
        )
        for candidates, recommended, default, expected, source in cases:
            selected = select_automatic_embedding_dimension(
                candidates,
                provider_recommended_dimension=recommended,
                provider_default_dimension=default,
            )
            self.assertEqual(selected, (expected, source))
        with self.assertRaisesRegex(ValueError, "embedding_dimension_required"):
            select_automatic_embedding_dimension(())
        for invalid in ((63,), (4097,), (True,), (724.0,)):
            with self.assertRaises(ValueError):
                select_automatic_embedding_dimension(invalid)

        snapshot = EmbeddingValidationSnapshot(
            provider_supported_dimensions=(724, 1024),
            verified_dimensions=(724,),
            provider_default_dimension=1024,
            recommended_dimension=1024,
            selected_dimension=724,
            selection_source=EmbeddingDimensionSelectionSource.USER_PROBE,
            dimension_request_mode=EmbeddingDimensionRequestMode.EXPLICIT,
            input_capabilities=(
                EmbeddingInputCapability.TEXT_DOCUMENT,
                EmbeddingInputCapability.TEXT_QUERY,
            ),
            shared_text_image_space_confirmed=False,
        )
        self.assertEqual(
            EmbeddingValidationSnapshot.from_mapping(snapshot.as_dict()), snapshot
        )

        text_space = uuid4()
        self.assertIs(
            derive_embedding_execution_mode(text_space, None),
            EmbeddingExecutionMode.TEXT_ONLY,
        )
        self.assertIs(
            derive_embedding_execution_mode(text_space, text_space),
            EmbeddingExecutionMode.UNIFIED_MULTIMODAL,
        )
        self.assertIs(
            derive_embedding_execution_mode(text_space, uuid4()),
            EmbeddingExecutionMode.DUAL_SPACE_MULTIMODAL,
        )

    def test_service_parameter_guard_covers_profile_update_boundaries(self) -> None:
        _require_parameters(
            ModelKind.TEXT_EMBEDDING,
            {"type": "embedding", "dimension": 64},
        )
        _require_parameters(
            ModelKind.MULTIMODAL_EMBEDDING,
            {
                "type": "embedding",
                "dimension": 4096,
                "shared_text_image_space_confirmed": True,
            },
        )
        for invalid_dimension in (True, 63, 4097, 724.0):
            with self.assertRaises(ResourceStateConflictError):
                _require_parameters(
                    ModelKind.TEXT_EMBEDDING,
                    {"type": "embedding", "dimension": invalid_dimension},
                )
        with self.assertRaises(ResourceStateConflictError):
            _require_parameters(
                ModelKind.TEXT_EMBEDDING,
                {
                    "type": "embedding",
                    "dimension": "auto",
                    "shared_text_image_space_confirmed": True,
                },
            )

    def test_openai_compatible_catalog_loads_and_normalizes_models(self) -> None:
        self.assertEqual(
            parse_model_catalog({"models": [{"name": "model-b"}, "model-a"]}),
            ("model-a", "model-b"),
        )

        async def scenario() -> None:
            async def respond(request: httpx.Request) -> httpx.Response:
                self.assertEqual(str(request.url), "https://provider.invalid/v1/models")
                self.assertEqual(request.headers["Authorization"], "Bearer secret")
                return httpx.Response(200, json={
                    "data": [
                        {"id": "model-z"},
                        {"id": "model-a"},
                        {"id": "model-a"},
                    ],
                })

            now = datetime.now(UTC)
            workspace_id = uuid4()
            provider_id = uuid4()
            bundle = ModelProviderBundle(
                ModelProvider(
                    id=provider_id,
                    workspace_id=workspace_id,
                    name="Provider",
                    enabled=True,
                    created_at=now,
                    updated_at=now,
                ),
                ModelProviderRevision(
                    id=uuid4(),
                    workspace_id=workspace_id,
                    provider_id=provider_id,
                    revision=1,
                    protocol=ModelProviderProtocol.OPENAI_COMPATIBLE,
                    base_url="https://provider.invalid/v1/",
                    secret_reference=str(uuid4()),
                    timeout_seconds=30,
                    max_retries=0,
                    max_concurrency=2,
                    configuration_fingerprint="sha256:test",
                    created_at=now,
                ),
            )
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(respond)
            ) as client:
                adapter = OpenAICompatibleModelCatalogAdapter(client)
                self.assertEqual(
                    await adapter.list_models(bundle, "secret"),
                    ("model-a", "model-z"),
                )

        asyncio.run(scenario())

    def test_fingerprints_are_stable_and_embedding_compatibility_is_separate(self) -> None:
        first = provider_fingerprint(
            protocol=ModelProviderProtocol.OPENAI_COMPATIBLE,
            base_url="http://localhost:8000/v1/",
            timeout_seconds=30,
            max_retries=2,
            max_concurrency=2,
        )
        second = provider_fingerprint(
            protocol=ModelProviderProtocol.OPENAI_COMPATIBLE,
            base_url="http://localhost:8000/v1",
            timeout_seconds=30,
            max_retries=2,
            max_concurrency=2,
        )
        self.assertEqual(first, second)
        fingerprints = model_fingerprints(ModelKind.TEXT_EMBEDDING, "embed-a", {
            "type": "embedding",
            "dimension": 1024,
            "max_batch_size": 10,
            "distance_metric": "cosine",
            "vector_data_type": "float32",
            "normalization": "l2",
        })
        self.assertIsNone(fingerprints[2])

        now = datetime.now(UTC)
        workspace_id = uuid4()
        provider_id = uuid4()
        provider_revision_id = uuid4()
        profile_id = uuid4()
        profile_revision_id = uuid4()
        provider = ModelProvider(
            id=provider_id,
            workspace_id=workspace_id,
            name="Provider",
            enabled=True,
            created_at=now,
            updated_at=now,
        )
        provider_revision = ModelProviderRevision(
            id=provider_revision_id,
            workspace_id=workspace_id,
            provider_id=provider_id,
            revision=1,
            protocol=ModelProviderProtocol.OPENAI_COMPATIBLE,
            base_url="https://provider.invalid/v1",
            secret_reference="secret-reference",
            timeout_seconds=30,
            max_retries=2,
            max_concurrency=2,
            configuration_fingerprint="sha256:provider",
            created_at=now,
        )
        profile = ModelProfile(
            id=profile_id,
            workspace_id=workspace_id,
            provider_id=provider_id,
            name="Embedding",
            kind=ModelKind.TEXT_EMBEDDING,
            enabled=True,
            created_at=now,
            updated_at=now,
        )
        revision = ModelProfileRevision(
            id=profile_revision_id,
            workspace_id=workspace_id,
            profile_id=profile_id,
            provider_revision_id=provider_revision_id,
            revision=1,
            model="embed-a",
            configuration={"type": "embedding", "dimension": 724},
            configuration_fingerprint="sha256:profile",
            capability_fingerprint="sha256:unverified",
            compatibility_fingerprint=None,
            validation_status=ModelValidationStatus.UNVERIFIED,
            validation_error_code=None,
            validation_snapshot=None,
            validated_at=None,
            created_at=now,
        )
        bundle = ModelProfileBundle(
            profile=profile,
            current_revision=revision,
            provider=provider,
            provider_revision=provider_revision,
        )
        snapshot_724 = EmbeddingValidationSnapshot(
            provider_supported_dimensions=None,
            verified_dimensions=(724,),
            provider_default_dimension=None,
            recommended_dimension=None,
            selected_dimension=724,
            selection_source=EmbeddingDimensionSelectionSource.USER_PROBE,
            dimension_request_mode=EmbeddingDimensionRequestMode.EXPLICIT,
            input_capabilities=(
                EmbeddingInputCapability.TEXT_DOCUMENT,
                EmbeddingInputCapability.TEXT_QUERY,
            ),
            shared_text_image_space_confirmed=False,
        )
        capability = embedding_capability_fingerprint(snapshot_724)
        compatibility = embedding_compatibility_fingerprint(bundle, snapshot_724)
        self.assertEqual(
            capability,
            embedding_capability_fingerprint(snapshot_724),
        )
        self.assertEqual(
            compatibility,
            embedding_compatibility_fingerprint(bundle, snapshot_724),
        )
        snapshot_2048 = EmbeddingValidationSnapshot(
            provider_supported_dimensions=None,
            verified_dimensions=(2048,),
            provider_default_dimension=None,
            recommended_dimension=None,
            selected_dimension=2048,
            selection_source=EmbeddingDimensionSelectionSource.USER_PROBE,
            dimension_request_mode=EmbeddingDimensionRequestMode.EXPLICIT,
            input_capabilities=snapshot_724.input_capabilities,
            shared_text_image_space_confirmed=False,
        )
        self.assertNotEqual(
            compatibility,
            embedding_compatibility_fingerprint(bundle, snapshot_2048),
        )

    def test_routing_adapter_caches_revision_models_and_preserves_legacy(self) -> None:
        async def scenario() -> None:
            loaded: list[object] = []

            async def loader(revision_id):
                loaded.append(revision_id)
                return _ChatModel("configured")

            adapter = RoutingChatModelAdapter(
                loader,
                legacy_fallback=_ChatModel("legacy"),
            )
            legacy = await adapter.complete(ChatModelRequest((ChatModelMessage("user", "x"),)))
            revision_id = uuid4()
            request = ChatModelRequest(
                (ChatModelMessage("user", "x"),),
                model_profile_revision_id=revision_id,
            )
            configured = await adapter.complete(request)
            await adapter.complete(request)
            self.assertEqual(legacy.model, "legacy")
            self.assertEqual(configured.model, "configured")
            self.assertEqual(loaded, [revision_id])

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
