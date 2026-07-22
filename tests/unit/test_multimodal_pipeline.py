from __future__ import annotations

import hashlib
import math
import tempfile
import unittest
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
from docx import Document
from PIL import Image

from rag_kb.adapters import LocalIndexAssetStore, TongyiVisionEmbeddingAdapter
from apps.api.routers.assets import read_index_asset
from rag_kb.auth import AuthContext, SingleWorkspaceAccessPolicy
from rag_kb.adapters.parser.docx_pictures import partition_docx_multimodal
from rag_kb.adapters.parser.multimodal_elements import bounded_image_asset
from rag_kb.document_processing import (
    assemble_multimodal_units,
    asset_manifest_hash,
    element_sequence_hash,
)
from rag_kb.domain import (
    ContentModality,
    EmbeddingSpaceDefinition,
    ErrorCode,
    ImageEmbeddingInput,
    IndexAssetIdentity,
    IndexAssetSnapshot,
    IndexingExecutionError,
    ParsedDocument,
    ParsedElement,
    ParserExecutionError,
    ParserLimits,
    ParserSource,
    VectorSearchHit,
)
from rag_kb.retrieval.fusion import reciprocal_rank_fusion
from rag_kb.services import IndexAssetService


WORKSPACE = UUID("01900000-0000-7000-8000-000000000901")
INDEXED_VERSION = UUID("01900000-0000-7000-8000-000000000902")


def _png(width: int = 120, height: int = 80) -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), (32, 96, 160)).save(output, format="PNG")
    return output.getvalue()


def _docx() -> bytes:
    document = Document()
    document.add_heading("Architecture", level=1)
    document.add_paragraph("The body text remains an independent text unit.")
    document.add_picture(BytesIO(_png()))
    document.add_paragraph("Figure 1: service topology", style="Caption")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Service"
    table.cell(0, 1).text = "Owner"
    table.cell(1, 0).text = "API"
    table.cell(1, 1).text = "Platform"
    output = BytesIO()
    document.save(output)
    return output.getvalue()


def _space() -> EmbeddingSpaceDefinition:
    return EmbeddingSpaceDefinition(
        provider_identity="alibaba-cloud-model-studio-qwen",
        requested_model="tongyi-embedding-vision-flash-2026-03-06",
        resolved_model="tongyi-embedding-vision-flash-2026-03-06",
        model_version="tongyi-embedding-vision-flash-2026-03-06",
        dimension=768,
        distance_metric="cosine",
        vector_data_type="float32",
        normalization="l2",
        endpoint_identity="alibaba-model-studio-beijing-multimodal-embedding",
        deployment_revision=None,
        configuration_fingerprint="sha256:test-configuration",
        tokenizer_fingerprint=None,
        compatibility_fingerprint="sha256:test-multimodal",
    )


class MultimodalParserTests(unittest.TestCase):
    def test_docx_picture_table_order_and_hashes_are_repeatable(self) -> None:
        source = ParserSource(
            "mixed.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            _docx(),
        )

        first = partition_docx_multimodal(source, ParserLimits(), "profile-v1")
        second = partition_docx_multimodal(source, ParserLimits(), "profile-v1")

        self.assertEqual(
            [item.category for item in first.elements],
            ["Title", "NarrativeText", "Image", "FigureCaption", "Table"],
        )
        self.assertEqual(len(first.assets), 1)
        self.assertEqual(first.assets[0].media_type, "image/png")
        self.assertEqual(element_sequence_hash(first), element_sequence_hash(second))
        self.assertEqual(asset_manifest_hash(first.assets), asset_manifest_hash(second.assets))

        units = assemble_multimodal_units(first)
        self.assertEqual(
            [item.modality for item in units],
            [ContentModality.TEXT, ContentModality.IMAGE, ContentModality.TABLE],
        )
        image = units[1]
        self.assertEqual(image.content, "Figure 1: service topology")
        self.assertEqual(image.required_representations, ("native_image",))
        self.assertNotIn("Figure 1", units[0].content)
        self.assertIn("Service\tOwner", units[2].content)

    def test_image_sniffing_and_limits_fail_closed(self) -> None:
        asset = bounded_image_asset(
            _png(), kind="fixture", source_location={"page_number": 1}, limits=ParserLimits()
        )
        self.assertEqual((asset.width, asset.height), (120, 80))
        self.assertEqual(asset.content_sha256, hashlib.sha256(asset.content).hexdigest())

        with self.assertRaises(ParserExecutionError) as malformed:
            bounded_image_asset(
                b"not-an-image", kind="fixture", source_location={}, limits=ParserLimits()
            )
        self.assertEqual(malformed.exception.code, ErrorCode.PARSER_OUTPUT_INVALID)

        with self.assertRaises(ParserExecutionError) as oversized:
            bounded_image_asset(
                _png(20, 20),
                kind="fixture",
                source_location={},
                limits=replace(ParserLimits(), max_image_pixels=100),
            )
        self.assertEqual(oversized.exception.code, ErrorCode.PARSER_RESOURCE_LIMIT)


class IndexAssetStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_atomic_idempotent_asset_round_trip_and_checksum_guard(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            staging = Path(root) / "staging"
            final = Path(root) / "final"
            staging.mkdir()
            final.mkdir()
            store = LocalIndexAssetStore(staging, final)
            content = _png()
            checksum = hashlib.sha256(content).hexdigest()
            identity = IndexAssetIdentity(WORKSPACE, INDEXED_VERSION, checksum)

            await store.put(identity, content, checksum)
            await store.put(identity, content, checksum)
            self.assertEqual(await store.read(identity), content)
            self.assertEqual(store.parse_uri(identity.storage_uri), identity)
            with self.assertRaises(Exception):
                await store.put(identity, b"different", checksum)
            await store.delete(identity)
            with self.assertRaises(Exception):
                await store.read(identity)

    async def test_authorized_service_and_http_response_keep_binary_out_of_json(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            staging = Path(root) / "staging"
            final = Path(root) / "final"
            staging.mkdir()
            final.mkdir()
            store = LocalIndexAssetStore(staging, final)
            content = _png()
            checksum = hashlib.sha256(content).hexdigest()
            identity = IndexAssetIdentity(WORKSPACE, INDEXED_VERSION, checksum)
            await store.put(identity, content, checksum)
            snapshot = IndexAssetSnapshot(
                id=uuid4(),
                workspace_id=WORKSPACE,
                kb_id=uuid4(),
                document_id=uuid4(),
                document_version_id=uuid4(),
                indexed_document_version_id=INDEXED_VERSION,
                storage_uri=identity.storage_uri,
                media_type="image/png",
                checksum_sha256=checksum,
            )
            service = IndexAssetService(
                _AssetUowFactory(snapshot),
                SingleWorkspaceAccessPolicy(WORKSPACE),
                store,
            )
            context = AuthContext("principal", "client", WORKSPACE)

            loaded = await service.read(context, snapshot.id)
            response = await read_index_asset(
                SimpleNamespace(
                    app=SimpleNamespace(
                        state=SimpleNamespace(
                            dependencies=SimpleNamespace(
                                index_asset_service=service
                            )
                        )
                    )
                ),
                snapshot.id,
                context,
            )

            self.assertEqual(loaded.content, content)
            self.assertEqual(response.body, content)
            self.assertEqual(response.headers["content-type"], "image/png")
            self.assertEqual(response.headers["etag"], f'"sha256:{checksum}"')
            self.assertEqual(response.headers["x-content-type-options"], "nosniff")
            self.assertEqual(response.headers["cache-control"], "private, max-age=300")


class MultimodalEmbeddingAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_and_image_inputs_are_separate_and_response_is_validated(self) -> None:
        requests: list[dict] = []
        vector = [0.0] * 768
        vector[0] = 1.0

        async def handler(request: httpx.Request) -> httpx.Response:
            payload = __import__("json").loads(request.content)
            requests.append(payload)
            count = len(payload["input"]["contents"])
            return httpx.Response(
                200,
                json={
                    "output": {
                        "embeddings": [
                            {"index": index, "embedding": vector}
                            for index in range(count)
                        ]
                    }
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = TongyiVisionEmbeddingAdapter(
                endpoint="https://provider.invalid/embeddings",
                api_key="test-only",
                embedding_space=_space(),
                max_batch_size=20,
                timeout_seconds=2,
                max_retries=0,
                max_concurrency=1,
                client=client,
            )
            text = await adapter.embed_texts(("diagram",))
            image = await adapter.embed_images(
                (ImageEmbeddingInput(_png(), "image/png", hashlib.sha256(_png()).hexdigest()),)
            )

        self.assertEqual(text.vectors[0][0], 1.0)
        self.assertEqual(image.vectors[0][0], 1.0)
        self.assertEqual(requests[0]["input"]["contents"], [{"text": "query: diagram"}])
        self.assertEqual(
            requests[0]["parameters"],
            {"dimension": 768, "output_type": "dense", "res_level": 1},
        )
        self.assertTrue(requests[1]["input"]["contents"][0]["image"].startswith("data:image/png;base64,"))

    async def test_non_normalized_provider_vector_is_rejected(self) -> None:
        async def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"output": {"embeddings": [{"index": 0, "embedding": [1.0] * 768}]}},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = TongyiVisionEmbeddingAdapter(
                endpoint="https://provider.invalid/embeddings",
                api_key="test-only",
                embedding_space=_space(),
                max_batch_size=20,
                timeout_seconds=2,
                max_retries=0,
                max_concurrency=1,
                client=client,
            )
            with self.assertRaises(IndexingExecutionError) as invalid:
                await adapter.embed_texts(("query",))
        self.assertEqual(invalid.exception.code, ErrorCode.EMBEDDING_RESPONSE_INVALID)


class RankFusionTests(unittest.TestCase):
    def test_rrf_deduplicates_evidence_group_and_preserves_lane_diagnostics(self) -> None:
        group = "asset-group"
        text = _hit(uuid4(), "caption_text", "image", group, 0.08, "caption")
        duplicate_text = _hit(uuid4(), "ocr_text", "image", group, 0.09, "ocr")
        native = _hit(uuid4(), "native_image", "image", group, 0.04, "")

        fused = reciprocal_rank_fusion(
            (text, duplicate_text), (native,), rrf_k=60, top_k=10
        )

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].text_rank, 1)
        self.assertEqual(fused[0].cross_modal_rank, 1)
        self.assertEqual(
            set(fused[0].matched_representations),
            {"caption_text", "ocr_text", "native_image"},
        )
        self.assertTrue(math.isclose(fused[0].score, 2 / 61, abs_tol=1e-12))


def _hit(
    chunk_id: UUID,
    representation: str,
    modality: str,
    group: str,
    distance: float,
    text: str,
) -> VectorSearchHit:
    return VectorSearchHit(
        workspace_id=WORKSPACE,
        knowledge_base_id=uuid4(),
        index_revision_id=uuid4(),
        index_chunk_id=chunk_id,
        indexed_document_version_id=INDEXED_VERSION,
        document_id=uuid4(),
        document_version_id=uuid4(),
        ordinal=0,
        text=text,
        source_location={},
        hierarchy={},
        source_metadata={},
        cosine_distance=distance,
        build_status="ready",
        serving_status="serving",
        is_current_serving_version=True,
        modality=modality,
        evidence_group_key=group,
        representation_kind=representation,
    )


class _AssetRepository:
    def __init__(self, snapshot: IndexAssetSnapshot) -> None:
        self.snapshot = snapshot

    async def get_asset(self, asset_id: UUID):
        return self.snapshot if asset_id == self.snapshot.id else None


class _AssetUow:
    def __init__(self, snapshot: IndexAssetSnapshot) -> None:
        self.workspace_id = WORKSPACE
        self.indexing = _AssetRepository(snapshot)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def commit(self):
        return None


class _AssetUowFactory:
    def __init__(self, snapshot: IndexAssetSnapshot) -> None:
        self.snapshot = snapshot

    def __call__(self, **kwargs):
        del kwargs
        return _AssetUow(self.snapshot)
