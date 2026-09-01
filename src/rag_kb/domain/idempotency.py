"""Transport-independent idempotency scope and canonical request hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TypeAlias
from uuid import UUID


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class IdempotencyScope:
    """The local endpoint/key uniqueness boundary for a mutating request."""

    endpoint: str
    idempotency_key: UUID

    def __post_init__(self) -> None:
        method, separator, path = self.endpoint.partition(" ")
        if not separator or not method.isalpha() or method != method.upper():
            raise ValueError("endpoint must start with an uppercase HTTP method")
        if not path.startswith("/"):
            raise ValueError("endpoint must contain an absolute API path")


def canonical_request_hash(payload: JsonValue) -> str:
    """Hash one deterministic UTF-8 JSON representation of a request body."""

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"
