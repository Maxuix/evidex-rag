from __future__ import annotations

import asyncio
import io
import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from PIL import Image

from rag_kb.adapters.file_store.local import LocalFileStore
from rag_kb.auth import AuthContext
from rag_kb.domain import (
    Document,
    DocumentMutationResult,
    FileLocation,
    InvalidStorageIdentityError,
    SourceFileDigest,
    SourceFileIntegrityError,
)
from rag_kb.document_processing.markdown_bundle import MARKDOWN_BUNDLE_MEDIA_TYPE
from rag_kb.ports.files import SourceFileStore
from rag_kb.ports.markdown_media import FetchedImage
from rag_kb.services.files import SourceFileService
from rag_kb.services.markdown_media import MarkdownMediaNormalizer


WORKSPACE = UUID("01900000-0000-7000-8000-000000000001")
KB_ID = UUID("01900000-0000-7000-8000-000000000010")
DOCUMENT_ID = UUID("01900000-0000-7000-8000-000000000020")
VERSION_ID = UUID("01900000-0000-7000-8000-000000000021")
IDEMPOTENCY_KEY = UUID("01900000-0000-7000-8000-000000000030")


class LocalFileStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.staging = self.root / "staging"
        self.final = self.root / "final"
        self.staging.mkdir()
        self.final.mkdir()
        self.store = LocalFileStore(self.staging, self.final)
        self.assertIsInstance(self.store, SourceFileStore)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_stage_finalize_and_restart_are_idempotent(self) -> None:
        async def scenario() -> None:
            staged = await self.store.stage(WORKSPACE, "operation", io.BytesIO(b"hello"))
            self.assertEqual(staged.digest.size_bytes, 5)
            self.assertEqual(
                await self.store.inspect(staged.identity, FileLocation.STAGING),
                staged.digest,
            )
            stored = await self.store.list_files()
            self.assertEqual(len(stored), 1)
            path = self.staging / stored[0].opaque_name
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)

            await self.store.finalize(staged.identity, staged.digest)
            restarted = LocalFileStore(self.staging, self.final)
            await restarted.finalize(staged.identity, staged.digest)
            self.assertIsNone(
                await restarted.inspect(staged.identity, FileLocation.STAGING)
            )
            self.assertEqual(await restarted.read_final(staged.identity), b"hello")

        asyncio.run(scenario())

    def test_content_changes_cannot_overwrite_an_existing_staged_operation(self) -> None:
        async def scenario() -> None:
            first = await self.store.stage(WORKSPACE, "same-scope", io.BytesIO(b"first"))
            second = await self.store.stage(WORKSPACE, "same-scope", io.BytesIO(b"second"))
            self.assertNotEqual(first.identity, second.identity)
            self.assertEqual(
                await self.store.inspect(first.identity, FileLocation.STAGING),
                first.digest,
            )
            self.assertEqual(
                await self.store.inspect(second.identity, FileLocation.STAGING),
                second.digest,
            )

        asyncio.run(scenario())

    def test_integrity_and_storage_identity_fail_closed(self) -> None:
        async def scenario() -> None:
            staged = await self.store.stage(WORKSPACE, "operation", io.BytesIO(b"hello"))
            with self.assertRaises(SourceFileIntegrityError):
                await self.store.finalize(
                    staged.identity,
                    SourceFileDigest("0" * 64, staged.digest.size_bytes),
                )
            with self.assertRaises(InvalidStorageIdentityError):
                self.store.parse_uri("file:///tmp/not-allowed")
            with self.assertRaises(InvalidStorageIdentityError):
                self.store.parse_uri(f"local-source://{WORKSPACE}/../escape")

        asyncio.run(scenario())


class SourceFileServiceTests(unittest.TestCase):
    def test_orchestration_keeps_staging_and_database_order_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            final = root / "final"
            staging.mkdir()
            final.mkdir()
            store = LocalFileStore(staging, final)
            context = AuthContext("principal", "client", WORKSPACE)
            events: list[str] = []

            class Documents:
                async def reserve_version(self, *args, **kwargs):
                    source = kwargs["source"]
                    identity = store.parse_uri(source.storage_uri)
                    self_digest = SourceFileDigest(
                        source.checksum_sha256, source.size_bytes
                    )
                    self_test.assertEqual(
                        await store.inspect(identity, FileLocation.STAGING), self_digest
                    )
                    self_test.assertIsNone(
                        await store.inspect(identity, FileLocation.FINAL)
                    )
                    events.append("reserved")
                    return DocumentMutationResult(
                        document=_document(), document_version_id=VERSION_ID
                    )

                async def activate_reserved_version(self, *args, **kwargs):
                    files = await store.list_files()
                    self_test.assertEqual(len(files), 1)
                    self_test.assertEqual(files[0].location, FileLocation.FINAL)
                    events.append("activated")
                    return DocumentMutationResult(
                        document=_document(), document_version_id=VERSION_ID
                    )

            self_test = self

            async def scenario() -> None:
                result = await SourceFileService(Documents(), store).store_and_activate(  # type: ignore[arg-type]
                    context,
                    IDEMPOTENCY_KEY,
                    kb_id=KB_ID,
                    document_id=None,
                    display_name="Guide",
                    original_filename="guide.md",
                    media_type="text/markdown",
                    source=io.BytesIO(b"durable source"),
                )
                self.assertEqual(result.document_version_id, VERSION_ID)

            asyncio.run(scenario())
            self.assertEqual(events, ["reserved", "activated"])

    def test_markdown_snapshot_replay_does_not_fetch_remote_media_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            final = root / "final"
            staging.mkdir()
            final.mkdir()
            store = LocalFileStore(staging, final)
            context = AuthContext("principal", "client", WORKSPACE)
            image_target = io.BytesIO()
            Image.new("RGB", (4, 3), "blue").save(image_target, "PNG")

            class Fetcher:
                calls = 0

                def fetch(self, url: str, *, max_bytes: int) -> FetchedImage:
                    del max_bytes
                    self.calls += 1
                    return FetchedImage(image_target.getvalue(), url)

            class Documents:
                async def reserve_version(self, *args, **kwargs):
                    self_test.assertEqual(
                        kwargs["source"].media_type,
                        MARKDOWN_BUNDLE_MEDIA_TYPE,
                    )
                    return DocumentMutationResult(
                        document=_document(), document_version_id=VERSION_ID
                    )

                async def activate_reserved_version(self, *args, **kwargs):
                    return DocumentMutationResult(
                        document=_document(), document_version_id=VERSION_ID
                    )

            self_test = self
            fetcher = Fetcher()
            service = SourceFileService(
                Documents(),  # type: ignore[arg-type]
                store,
                MarkdownMediaNormalizer(fetcher),
            )
            markdown = b"![chart](https://example.com/chart.png)\n"

            async def scenario() -> None:
                for _ in range(2):
                    await service.store_and_activate(
                        context,
                        IDEMPOTENCY_KEY,
                        kb_id=KB_ID,
                        document_id=None,
                        display_name="Guide",
                        original_filename="guide.md",
                        media_type="text/markdown",
                        source=io.BytesIO(markdown),
                        normalize_markdown_media=True,
                    )

            asyncio.run(scenario())
            self.assertEqual(fetcher.calls, 1)


def _document() -> Document:
    now = datetime.now(UTC)
    return Document(
        id=DOCUMENT_ID,
        workspace_id=WORKSPACE,
        kb_id=KB_ID,
        display_name="Guide",
        current_version=None,
        deleted_at=None,
        created_at=now,
        updated_at=now,
    )
