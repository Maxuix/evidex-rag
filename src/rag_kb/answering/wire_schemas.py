"""Project-owned provider wire schemas, independent of LangChain types."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rag_kb.domain import ChatOutputSchema
from rag_kb.memory.query import WireContextualQuery


class WireAnswerClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str = Field(max_length=4000)
    citation_ids: list[str] = Field(max_length=100)


class WireAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    outcome: Literal["answered", "partial", "refused"]
    claims: list[WireAnswerClaim] = Field(max_length=100)
    missing_aspects: list[str] = Field(max_length=100)


OUTPUT_SCHEMAS: dict[ChatOutputSchema, type[BaseModel]] = {
    ChatOutputSchema.ANSWER_V1: WireAnswer,
    ChatOutputSchema.CONTEXTUAL_QUERY_V1: WireContextualQuery,
}
