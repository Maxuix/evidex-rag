from __future__ import annotations

import unittest
from types import SimpleNamespace
from uuid import UUID, uuid4

from apps.model_asset_runtime import build_dynamic_embedding_loaders
from rag_kb.adapters.model_api.langchain_embeddings import (
    LangChainEmbeddingModelAdapter,
)
from rag_kb.adapters.model_api.multimodal_embeddings import (
    TongyiVisionEmbeddingAdapter,
)
from rag_kb.domain import (
    EmbeddingSpaceDefinition,
    ModelKind,
    ModelValidationStatus,
)


class DynamicEmbeddingLoaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_database_revisions_build_and_cache_both_adapter_kinds(
        self,
    ) -> None:
        text_space = _space(uuid4(), "sha256:text", dimension=1024)
        multimodal_space = _space(
            uuid4(),
            "sha256:multimodal",
            dimension=768,
        )
        repository = _ModelSettingsRepository({
            text_space.model_profile_revision_id: _bundle(
                text_space,
                ModelKind.TEXT_EMBEDDING,
            ),
            multimodal_space.model_profile_revision_id: _bundle(
                multimodal_space,
                ModelKind.MULTIMODAL_EMBEDDING,
            ),
        })
        factory = _UnitOfWorkFactory(repository)
        secrets = _SecretStore()
        loaders = build_dynamic_embedding_loaders(factory, secrets)

        text = await loaders.embedding(text_space)
        multimodal = await loaders.multimodal(multimodal_space)

        self.assertIsInstance(text, LangChainEmbeddingModelAdapter)
        self.assertIsInstance(multimodal, TongyiVisionEmbeddingAdapter)
        self.assertEqual(text.embedding_space, text_space)
        self.assertEqual(multimodal.embedding_space, multimodal_space)
        self.assertIs(await loaders.embedding(text_space), text)
        self.assertIs(await loaders.multimodal(multimodal_space), multimodal)
        self.assertEqual(
            repository.revision_ids,
            [
                text_space.model_profile_revision_id,
                multimodal_space.model_profile_revision_id,
            ],
        )
        self.assertEqual(secrets.references, ["secret-ref", "secret-ref"])
        self.assertEqual(factory.commits, 2)

    async def test_compatibility_mismatch_fails_before_secret_access(self) -> None:
        space = _space(uuid4(), "sha256:expected", dimension=768)
        incompatible = _bundle(
            space,
            ModelKind.MULTIMODAL_EMBEDDING,
            compatibility_fingerprint="sha256:other",
        )
        repository = _ModelSettingsRepository({
            space.model_profile_revision_id: incompatible,
        })
        factory = _UnitOfWorkFactory(repository)
        secrets = _SecretStore()
        loaders = build_dynamic_embedding_loaders(factory, secrets)

        with self.assertRaisesRegex(
            ValueError,
            "embedding model profile is unavailable",
        ):
            await loaders.multimodal(space)

        self.assertEqual(secrets.references, [])
        self.assertEqual(factory.commits, 0)


class _ModelSettingsRepository:
    def __init__(self, bundles: dict[UUID | None, object]) -> None:
        self._bundles = bundles
        self.revision_ids: list[UUID] = []

    async def get_profile_revision(self, revision_id: UUID):
        self.revision_ids.append(revision_id)
        return self._bundles.get(revision_id)


class _UnitOfWork:
    def __init__(
        self,
        factory: _UnitOfWorkFactory,
        repository: _ModelSettingsRepository,
    ) -> None:
        self._factory = factory
        self.model_settings = repository

    async def __aenter__(self):
        return self

    async def __aexit__(self, exception_type, exception, traceback):
        return None

    async def commit(self) -> None:
        self._factory.commits += 1


class _UnitOfWorkFactory:
    def __init__(self, repository: _ModelSettingsRepository) -> None:
        self._repository = repository
        self.commits = 0

    def __call__(self, *, purpose, mode):
        del purpose, mode
        return _UnitOfWork(self, self._repository)


class _SecretStore:
    def __init__(self) -> None:
        self.references: list[str] = []

    def read(self, reference: str) -> str:
        self.references.append(reference)
        return "test-secret"


def _space(
    revision_id: UUID,
    compatibility_fingerprint: str,
    *,
    dimension: int,
) -> EmbeddingSpaceDefinition:
    return EmbeddingSpaceDefinition(
        provider_identity="database-provider",
        endpoint_identity="database-endpoint",
        requested_model="database-model",
        resolved_model="database-model",
        model_version="revision-1",
        deployment_revision=None,
        dimension=dimension,
        distance_metric="cosine",
        vector_data_type="float32",
        normalization="l2",
        configuration_fingerprint="sha256:configuration",
        tokenizer_fingerprint=None,
        compatibility_fingerprint=compatibility_fingerprint,
        model_profile_revision_id=revision_id,
    )


def _bundle(
    space: EmbeddingSpaceDefinition,
    kind: ModelKind,
    *,
    compatibility_fingerprint: str | None = None,
):
    return SimpleNamespace(
        profile=SimpleNamespace(kind=kind, enabled=True),
        provider=SimpleNamespace(enabled=True),
        provider_revision=SimpleNamespace(
            base_url="https://provider.invalid/v1",
            secret_reference="secret-ref",
            timeout_seconds=30,
            max_retries=2,
            max_concurrency=2,
        ),
        current_revision=SimpleNamespace(
            validation_status=ModelValidationStatus.VALID,
            compatibility_fingerprint=(
                compatibility_fingerprint
                if compatibility_fingerprint is not None
                else space.compatibility_fingerprint
            ),
            configuration={"max_batch_size": 8},
        ),
    )
