"""FastAPI header contracts for mutating operations."""

from typing import Annotated
from uuid import UUID

from fastapi import Header


RequiredIdempotencyKey = Annotated[
    UUID,
    Header(
        alias="Idempotency-Key",
        description="Required UUID scoped by principal, client, and endpoint.",
    ),
]
