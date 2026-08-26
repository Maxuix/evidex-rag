from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch
from uuid import UUID

import tiktoken

from rag_kb.domain import ChunkingPreset, ConversationTurn
from rag_kb.document_processing.profiles import (
    CHUNK_TOKENIZER,
    profile_fingerprint,
    profile_for_preset,
)
from rag_kb.document_processing.tokenization import (
    count_chunk_tokens,
    split_by_tokens,
)
from rag_kb.memory import (
    hydrate_conversation_context,
    select_conversation_context,
    serialize_conversation_context,
)
from rag_kb.tokenizer import (
    CL100K_BASE_ASSET_FILENAME,
    CL100K_BASE_ASSET_SHA256,
    CL100K_BASE_ASSET_SIZE,
    CL100K_BASE_ENCODING_NAME,
    CL100K_BASE_SOURCE_URL,
    CL100K_BASE_SPECIAL_TOKENS,
    CL100K_BASE_TOKENIZER_VERSION,
    DEFAULT_ASSET_PATH,
    DEFAULT_MANIFEST_PATH,
    TokenizerArtifactError,
    clear_tokenizer_cache,
    get_cl100k_base_encoding,
    load_cl100k_base_encoding,
    validate_tokenizer_asset,
)


ROOT = Path(__file__).resolve().parents[2]


class TokenizerAssetTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_tokenizer_cache()
        self.addCleanup(clear_tokenizer_cache)

    def test_packaged_manifest_and_asset_are_exact(self) -> None:
        manifest = validate_tokenizer_asset()

        self.assertEqual(manifest.encoding, CL100K_BASE_ENCODING_NAME)
        self.assertEqual(manifest.source_url, CL100K_BASE_SOURCE_URL)
        self.assertEqual(manifest.tiktoken_version, CL100K_BASE_TOKENIZER_VERSION)
        self.assertEqual(manifest.filename, CL100K_BASE_ASSET_FILENAME)
        self.assertEqual(manifest.byte_size, CL100K_BASE_ASSET_SIZE)
        self.assertEqual(manifest.sha256, CL100K_BASE_ASSET_SHA256)
        self.assertEqual(DEFAULT_ASSET_PATH.stat().st_size, CL100K_BASE_ASSET_SIZE)
        self.assertEqual(
            hashlib.sha256(DEFAULT_ASSET_PATH.read_bytes()).hexdigest(),
            CL100K_BASE_ASSET_SHA256,
        )
        self.assertEqual(
            json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))[
                "sha256"
            ],
            CL100K_BASE_ASSET_SHA256,
        )

    def test_frozen_token_ids_and_counts_cover_unicode_and_special_text(self) -> None:
        encoding = get_cl100k_base_encoding()
        samples = {
            "english": (
                "Hello, world! This is a deterministic tokenizer.",
                [9906, 11, 1917, 0, 1115, 374, 264, 73449, 47058, 13],
            ),
            "cjk": (
                "你好，世界。知识库检索。",
                [
                    57668,
                    53901,
                    3922,
                    3574,
                    244,
                    98220,
                    1811,
                    53283,
                    6744,
                    228,
                    46056,
                    98657,
                    52084,
                    1811,
                ],
            ),
            "mixed_unicode": (
                "naïve café — 你好 🌍 é",
                [
                    3458,
                    38672,
                    588,
                    53050,
                    2001,
                    220,
                    57668,
                    53901,
                    11410,
                    234,
                    235,
                    384,
                    54939,
                ],
            ),
            "special": (
                "[]{}<>\n\t#$%^&*()_+|~=`",
                [
                    1318,
                    6390,
                    63637,
                    197,
                    49177,
                    46999,
                    5,
                    9,
                    368,
                    62,
                    10,
                    91,
                    93,
                    23046,
                ],
            ),
        }

        for label, (text, expected_ids) in samples.items():
            with self.subTest(sample=label):
                self.assertEqual(encoding.encode(text), expected_ids)
                self.assertEqual(count_chunk_tokens(text), len(expected_ids))

        for token, expected_id in CL100K_BASE_SPECIAL_TOKENS.items():
            with self.subTest(special_token=token):
                self.assertEqual(
                    encoding.encode(token, allowed_special={token}),
                    [expected_id],
                )

    def test_loader_is_offline_and_ignores_the_tiktoken_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with patch.dict("os.environ", {"TIKTOKEN_CACHE_DIR": str(cache)}):
                with patch.object(
                    tiktoken,
                    "get_encoding",
                    side_effect=AssertionError("network-aware resolver used"),
                ):
                    clear_tokenizer_cache()
                    encoding = get_cl100k_base_encoding()
                    self.assertEqual(encoding.encode("offline"), [64629])
            self.assertEqual(tuple(cache.iterdir()), ())

    def test_missing_corrupt_truncated_and_wrong_manifest_fail_closed(self) -> None:
        original = DEFAULT_ASSET_PATH.read_bytes()
        base_manifest = json.loads(
            DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8")
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / DEFAULT_MANIFEST_PATH.name
            asset_path = root / CL100K_BASE_ASSET_FILENAME
            manifest_path.write_text(
                json.dumps(base_manifest),
                encoding="utf-8",
            )

            with self.subTest(kind="missing"):
                with self.assertRaises(TokenizerArtifactError) as raised:
                    validate_tokenizer_asset(
                        manifest_path=manifest_path,
                        asset_path=asset_path,
                    )
                self.assertEqual(raised.exception.code, "asset_missing")

            asset_path.write_bytes(original[:-1])
            with self.subTest(kind="truncated"):
                with self.assertRaises(TokenizerArtifactError) as raised:
                    validate_tokenizer_asset(
                        manifest_path=manifest_path,
                        asset_path=asset_path,
                    )
                self.assertEqual(raised.exception.code, "asset_size")

            asset_path.write_bytes(b"not a tokenizer table\n")
            with self.subTest(kind="corrupt"):
                with self.assertRaises(TokenizerArtifactError) as raised:
                    validate_tokenizer_asset(
                        manifest_path=manifest_path,
                        asset_path=asset_path,
                    )
                self.assertEqual(raised.exception.code, "asset_size")

            wrong_manifest = {**base_manifest, "sha256": "0" * 64}
            manifest_path.write_text(
                json.dumps(wrong_manifest),
                encoding="utf-8",
            )
            asset_path.write_bytes(original)
            with self.subTest(kind="wrong_manifest"):
                with self.assertRaises(TokenizerArtifactError) as raised:
                    validate_tokenizer_asset(
                        manifest_path=manifest_path,
                        asset_path=asset_path,
                    )
                self.assertEqual(raised.exception.code, "manifest_sha256")

    def test_cold_process_chat_and_document_paths_do_not_use_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            script = textwrap.dedent(
                """
                import json
                import os
                import socket
                import urllib.request
                from uuid import UUID

                import requests
                import tiktoken

                def blocked(*args, **kwargs):
                    raise AssertionError("network access attempted")

                socket.getaddrinfo = blocked
                socket.create_connection = blocked
                socket.socket.connect = blocked
                urllib.request.urlopen = blocked
                requests.sessions.Session.request = blocked
                tiktoken.get_encoding = blocked

                from rag_kb.document_processing.tokenization import (
                    count_chunk_tokens,
                    split_by_tokens,
                )
                from rag_kb.domain import ConversationTurn
                from rag_kb.memory import (
                    hydrate_conversation_context,
                    select_conversation_context,
                    serialize_conversation_context,
                )

                turn = ConversationTurn(
                    user_message_id=UUID(int=1),
                    user_content="用户 asks: offline?",
                    assistant_message_id=UUID(int=2),
                    assistant_content="是，离线可用。",
                )
                snapshot = select_conversation_context((turn,))
                serialized = serialize_conversation_context(snapshot)
                hydrated = hydrate_conversation_context(serialized)
                document = "离线 tokenizer: hello, 世界!"
                parts = split_by_tokens(document, max_tokens=8, overlap_tokens=2)
                print(json.dumps({
                    "cache_dir": os.environ["TIKTOKEN_CACHE_DIR"],
                    "token_count": snapshot.token_count,
                    "round_trip": hydrated == snapshot,
                    "document_count": count_chunk_tokens(document),
                    "parts": parts,
                }, ensure_ascii=False, sort_keys=True))
                """
            )
            environment = {
                **__import__("os").environ,
                "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}",
                "TIKTOKEN_CACHE_DIR": str(cache),
            }
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

            result = json.loads(completed.stdout)
            self.assertEqual(result["cache_dir"], str(cache))
            self.assertTrue(result["round_trip"])
            self.assertGreater(result["token_count"], 0)
            self.assertGreater(result["document_count"], 0)
            self.assertGreater(len(result["parts"]), 1)
            self.assertEqual(tuple(cache.iterdir()), ())

    def test_chat_and_chunk_profile_identities_are_unchanged(self) -> None:
        self.assertEqual(
            CHUNK_TOKENIZER,
            {
                "tokenizer": "cl100k_base",
                "tokenizer_library": "tiktoken",
                "tokenizer_version": "0.13.0",
            },
        )
        expected_fingerprints = {
            ChunkingPreset.STRUCTURAL_BALANCED_V2: (
                "648f0555417a8d6e741a54b3cfbdea30b025223715760de6ba7d7468f1721220"
            ),
            ChunkingPreset.SEMANTIC_BALANCED_V1: (
                "f0d9ed75f2039aad563625c42b9d495b42eec85dbaaf44cf2b862ee8c9f31a5d"
            ),
        }
        for preset, expected in expected_fingerprints.items():
            with self.subTest(preset=preset):
                profile = profile_for_preset(preset)
                self.assertEqual(
                    profile_fingerprint(
                        profile.parser_config,
                        profile.chunking_config,
                    ),
                    expected,
                )

        turn = ConversationTurn(
            user_message_id=UUID(int=1),
            user_content="user 0",
            assistant_message_id=UUID(int=100),
            assistant_content="assistant 0",
        )
        snapshot = select_conversation_context((turn,))
        self.assertEqual(snapshot.token_count, 67)
        self.assertEqual(
            snapshot.content_hash,
            "sha256:77101ec95e869034dc2eb9cc5c9636bd807915f6d2aa5b7878616db8448bfb21",
        )
        self.assertEqual(
            hydrate_conversation_context(serialize_conversation_context(snapshot)),
            snapshot,
        )
        self.assertEqual(
            split_by_tokens(
                "alpha beta gamma delta 你好 世界 epsilon zeta",
                max_tokens=4,
                overlap_tokens=1,
            ),
            ("alpha beta gamma delta", "delta 你好", "好 世", "界 epsilon zeta"),
        )

    def test_production_sources_have_no_direct_tiktoken_resolution(self) -> None:
        production_files = tuple((ROOT / "src").rglob("*.py")) + tuple(
            (ROOT / "apps").rglob("*.py")
        )
        offenders = [
            str(path.relative_to(ROOT))
            for path in production_files
            if "tiktoken.get_encoding" in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(offenders, [])


class TokenizerStartupPreflightTests(unittest.TestCase):
    def test_api_preflight_runs_before_dependency_construction(self) -> None:
        from apps.api.dependencies import build_api_dependencies

        failure = TokenizerArtifactError("asset_sha256")
        with (
            patch(
                "apps.api.dependencies.preflight_tokenizer",
                side_effect=failure,
            ) as preflight,
            patch("apps.api.dependencies.create_database_resources") as create,
        ):
            with self.assertRaises(TokenizerArtifactError) as raised:
                build_api_dependencies()

        self.assertIs(raised.exception, failure)
        preflight.assert_called_once_with()
        create.assert_not_called()

    def test_worker_preflight_runs_before_dependency_construction(self) -> None:
        from apps.worker.dependencies import build_worker_dependencies

        failure = TokenizerArtifactError("asset_sha256")
        with (
            patch(
                "apps.worker.dependencies.preflight_tokenizer",
                side_effect=failure,
            ) as preflight,
            patch("apps.worker.dependencies.create_database_resources") as create,
        ):
            with self.assertRaises(TokenizerArtifactError) as raised:
                build_worker_dependencies()

        self.assertIs(raised.exception, failure)
        preflight.assert_called_once_with()
        create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
