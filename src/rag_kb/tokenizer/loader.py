"""Load the fixed cl100k_base tokenizer without any external I/O."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import re

import tiktoken


TOKENIZER_LIBRARY = "tiktoken"
CL100K_BASE_ENCODING_NAME = "cl100k_base"
CL100K_BASE_TOKENIZER_VERSION = "0.13.0"
CL100K_BASE_ASSET_FILENAME = "cl100k_base.tiktoken"
CL100K_BASE_SOURCE_URL = (
    "https://openaipublic.blob.core.windows.net/encodings/"
    "cl100k_base.tiktoken"
)
CL100K_BASE_ASSET_SHA256 = (
    "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
)
CL100K_BASE_ASSET_SIZE = 1_681_126
TOKENIZER_MANIFEST_SCHEMA_VERSION = 1

# This is the exact constructor data used by tiktoken 0.13.0's
# tiktoken_ext.openai_public.cl100k_base().  Keep the values here instead of
# importing that plugin: its constructor resolves the BPE table through the
# network-aware tiktoken loader.
CL100K_BASE_PAT_STR = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|"
    r"\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|"
    r"\s*[\r\n]|\s+(?!\S)|\s"
)
CL100K_BASE_SPECIAL_TOKENS = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}

_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "encoding",
        "source_url",
        "tiktoken_version",
        "filename",
        "byte_size",
        "sha256",
    }
)
_EXPECTED_MERGEABLE_TOKEN_COUNT = 100_256
_HASH_BUFFER_BYTES = 1024 * 1024
_ASSET_DIRECTORY = Path(__file__).with_name("assets")
DEFAULT_MANIFEST_PATH = _ASSET_DIRECTORY / "cl100k_base.manifest.json"
DEFAULT_ASSET_PATH = _ASSET_DIRECTORY / CL100K_BASE_ASSET_FILENAME


class TokenizerArtifactError(RuntimeError):
    """The fixed tokenizer manifest or BPE table is missing or invalid."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class TokenizerAssetManifest:
    """The identity and content address of one frozen tokenizer asset."""

    schema_version: int
    encoding: str
    source_url: str
    tiktoken_version: str
    filename: str
    byte_size: int
    sha256: str

    @classmethod
    def load(cls, manifest_path: Path) -> TokenizerAssetManifest:
        try:
            payload = manifest_path.read_text(encoding="utf-8")
            value = json.loads(payload)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise TokenizerArtifactError("manifest_unreadable") from error
        if not isinstance(value, dict) or set(value) != _EXPECTED_MANIFEST_FIELDS:
            raise TokenizerArtifactError("manifest_shape")

        schema_version = value.get("schema_version")
        if schema_version != TOKENIZER_MANIFEST_SCHEMA_VERSION:
            raise TokenizerArtifactError("manifest_schema_version")
        if value.get("encoding") != CL100K_BASE_ENCODING_NAME:
            raise TokenizerArtifactError("manifest_encoding")
        if value.get("source_url") != CL100K_BASE_SOURCE_URL:
            raise TokenizerArtifactError("manifest_source_url")
        if value.get("tiktoken_version") != CL100K_BASE_TOKENIZER_VERSION:
            raise TokenizerArtifactError("manifest_tiktoken_version")
        if value.get("filename") != CL100K_BASE_ASSET_FILENAME:
            raise TokenizerArtifactError("manifest_filename")

        byte_size = value.get("byte_size")
        if (
            not isinstance(byte_size, int)
            or isinstance(byte_size, bool)
            or byte_size != CL100K_BASE_ASSET_SIZE
        ):
            raise TokenizerArtifactError("manifest_byte_size")
        sha256 = value.get("sha256")
        if not isinstance(sha256, str) or not _HEX_DIGEST.fullmatch(sha256):
            raise TokenizerArtifactError("manifest_sha256")
        if sha256 != CL100K_BASE_ASSET_SHA256:
            raise TokenizerArtifactError("manifest_sha256")

        return cls(
            schema_version=schema_version,
            encoding=CL100K_BASE_ENCODING_NAME,
            source_url=CL100K_BASE_SOURCE_URL,
            tiktoken_version=CL100K_BASE_TOKENIZER_VERSION,
            filename=CL100K_BASE_ASSET_FILENAME,
            byte_size=byte_size,
            sha256=sha256,
        )


def validate_tokenizer_asset(
    *,
    manifest_path: Path | None = None,
    asset_path: Path | None = None,
) -> TokenizerAssetManifest:
    """Verify the package-owned manifest, bytes, and BPE record structure."""

    manifest, _mergeable_ranks = _load_verified_asset(
        manifest_path=manifest_path,
        asset_path=asset_path,
    )
    return manifest


def load_cl100k_base_encoding(
    *,
    manifest_path: Path | None = None,
    asset_path: Path | None = None,
) -> tiktoken.Encoding:
    """Construct an exact cl100k_base encoding from local verified bytes."""

    manifest, mergeable_ranks = _load_verified_asset(
        manifest_path=manifest_path,
        asset_path=asset_path,
    )
    if getattr(tiktoken, "__version__", None) != manifest.tiktoken_version:
        raise TokenizerArtifactError("dependency_version")
    try:
        return tiktoken.Encoding(
            name=manifest.encoding,
            pat_str=CL100K_BASE_PAT_STR,
            mergeable_ranks=mergeable_ranks,
            special_tokens=dict(CL100K_BASE_SPECIAL_TOKENS),
        )
    except Exception as error:
        raise TokenizerArtifactError("encoding_construction") from error


@lru_cache(maxsize=1)
def get_cl100k_base_encoding() -> tiktoken.Encoding:
    """Return the validated encoding, cached only for this Python process."""

    return load_cl100k_base_encoding()


def clear_tokenizer_cache() -> None:
    """Clear the process-local encoding cache for cold-start tests."""

    get_cl100k_base_encoding.cache_clear()


def preflight_tokenizer() -> None:
    """Fail early if the application tokenizer asset cannot be loaded."""

    get_cl100k_base_encoding()


def _load_verified_asset(
    *,
    manifest_path: Path | None,
    asset_path: Path | None,
) -> tuple[TokenizerAssetManifest, dict[bytes, int]]:
    resolved_manifest_path = Path(manifest_path or DEFAULT_MANIFEST_PATH)
    manifest = TokenizerAssetManifest.load(resolved_manifest_path)
    resolved_asset_path = Path(asset_path or resolved_manifest_path.parent / manifest.filename)
    data = _read_asset(resolved_asset_path)
    if len(data) != manifest.byte_size:
        raise TokenizerArtifactError("asset_size")
    if _sha256(data) != manifest.sha256:
        raise TokenizerArtifactError("asset_sha256")
    mergeable_ranks = _parse_tiktoken_bpe(data)
    return manifest, mergeable_ranks


def _read_asset(path: Path) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise TokenizerArtifactError("asset_missing")
        return path.read_bytes()
    except TokenizerArtifactError:
        raise
    except (OSError, UnicodeError) as error:
        raise TokenizerArtifactError("asset_unreadable") from error


def _sha256(data: bytes) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(data), _HASH_BUFFER_BYTES):
        digest.update(data[start : start + _HASH_BUFFER_BYTES])
    return digest.hexdigest()


def _parse_tiktoken_bpe(data: bytes) -> dict[bytes, int]:
    """Parse and validate the fixed newline-delimited tiktoken BPE format."""

    if not data or not data.endswith(b"\n"):
        raise TokenizerArtifactError("asset_format")

    ranks: dict[bytes, int] = {}
    seen_ranks: set[int] = set()
    for expected_rank, line in enumerate(data.splitlines()):
        fields = line.split()
        if len(fields) != 2:
            raise TokenizerArtifactError("asset_format")
        encoded_token, encoded_rank = fields
        try:
            token = base64.b64decode(encoded_token, validate=True)
            rank = int(encoded_rank, 10)
        except (ValueError, TypeError):
            raise TokenizerArtifactError("asset_format") from None
        if not token or rank != expected_rank or rank in seen_ranks or token in ranks:
            raise TokenizerArtifactError("asset_format")
        ranks[token] = rank
        seen_ranks.add(rank)

    if len(ranks) != _EXPECTED_MERGEABLE_TOKEN_COUNT:
        raise TokenizerArtifactError("asset_format")
    return ranks


__all__ = [
    "CL100K_BASE_ASSET_FILENAME",
    "CL100K_BASE_ASSET_SHA256",
    "CL100K_BASE_ASSET_SIZE",
    "CL100K_BASE_ENCODING_NAME",
    "CL100K_BASE_PAT_STR",
    "CL100K_BASE_SOURCE_URL",
    "CL100K_BASE_SPECIAL_TOKENS",
    "CL100K_BASE_TOKENIZER_VERSION",
    "DEFAULT_ASSET_PATH",
    "DEFAULT_MANIFEST_PATH",
    "TOKENIZER_LIBRARY",
    "TOKENIZER_MANIFEST_SCHEMA_VERSION",
    "TokenizerArtifactError",
    "TokenizerAssetManifest",
    "clear_tokenizer_cache",
    "get_cl100k_base_encoding",
    "load_cl100k_base_encoding",
    "preflight_tokenizer",
    "validate_tokenizer_asset",
]
