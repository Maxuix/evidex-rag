"""Common public API DTOs for failures and cursor pagination."""

from __future__ import annotations

from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from rag_kb.domain import ErrorCode


ItemT = TypeVar("ItemT")
SortExpression = Annotated[
    str,
    Field(pattern=r"^-?[a-z][a-z0-9_]*$", min_length=1, max_length=64),
]
OpaqueCursor = Annotated[str, Field(min_length=1, max_length=2048)]


class PublicSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FieldViolation(PublicSchema):
    location: tuple[str | int, ...]
    message: str
    error_type: str


class ProblemDetails(PublicSchema):
    type: str
    title: str
    status: Annotated[int, Field(ge=400, le=599)]
    detail: str
    instance: str
    code: ErrorCode
    trace_id: str
    retryable: bool
    errors: tuple[FieldViolation, ...] | None = None


class PaginationQuery(PublicSchema):
    limit: Annotated[int, Field(ge=1, le=100)] = 50
    cursor: OpaqueCursor | None = None
    sort: SortExpression = "created_at"


class CursorPayload(PublicSchema):
    version: Literal[1] = 1
    sort: SortExpression
    values: Annotated[tuple[str, ...], Field(min_length=1, max_length=8)]


class CursorPage(PublicSchema, Generic[ItemT]):
    items: tuple[ItemT, ...]
    next_cursor: OpaqueCursor | None = None
