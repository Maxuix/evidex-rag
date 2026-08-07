from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import os
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import httpx
from pydantic import ValidationError

from rag_kb.adapters.model_api.model_catalog import (
    OpenAICompatibleModelCatalogAdapter,
    parse_model_catalog,
)
from rag_kb.adapters.model_api.routing_chat import RoutingChatModelAdapter
from rag_kb.adapters.model_secrets.local import LocalModelSecretStore
from rag_kb.domain import (
    ChatModelMessage,
    ChatModelRequest,
    ChatModelResponse,
    ModelKind,
    ModelProvider,
    ModelProviderBundle,
    ModelProviderProtocol,
    ModelProviderRevision,
)
from rag_kb.schemas.model_settings import ModelProfileCreate
from rag_kb.services.model_settings import model_fingerprints, provider_fingerprint


class _ChatModel:
    def __init__(self, model: str) -> None:
        self.model = model

    async def complete(self, request: ChatModelRequest) -> ChatModelResponse:
        return ChatModelResponse(self.model, self.model, None, None, {})

    async def complete_streaming(self, request, *, on_content_delta):
        await on_content_delta(self.model)
        return await self.complete(request)


class ModelSettingsTests(unittest.TestCase):
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
        with self.assertRaises(ValidationError):
            ModelProfileCreate.model_validate({
                "provider_id": str(provider_id),
                "name": "Embedding",
                "kind": "text_embedding",
                "model": "embedding-a",
                "parameters": {"type": "embedding", "dimension": 768},
            })

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
        self.assertIsNotNone(fingerprints[2])
        self.assertNotEqual(fingerprints[0], fingerprints[2])

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
