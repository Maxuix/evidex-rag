"""Project-owned provider wire schemas, independent of LangChain types."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_kb.domain import ChatOutputSchema
from rag_kb.memory.query import WireContextualQuery


class WireAnswerClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str = Field(max_length=4000)
    citation_ids: list[str] = Field(max_length=100)


class WireAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    outcome: Literal["answered", "partial", "acknowledged", "refused"]
    claims: list[WireAnswerClaim] = Field(max_length=100)
    missing_aspects: list[str] = Field(max_length=100)


class WireRetrievalAgentQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=2048)
    based_on_observation_ids: list[str] = Field(max_length=24)


class WireRetrievalAgentAction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal["retrieval_agent_action_v1"]
    action: Literal["search", "finish"]
    objective: str | None = Field(default=None, max_length=1024)
    queries: list[WireRetrievalAgentQuery] = Field(default_factory=list, max_length=3)
    proposed_reason: Literal[
        "sufficient",
        "partial",
        "no_evidence",
        "no_progress",
        "budget_exhausted",
        "conflict_unresolved",
        "premise_unsupported",
    ] | None = None
    selected_evidence_keys: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def validate_action_shape(self) -> "WireRetrievalAgentAction":
        if self.action == "search":
            if (
                self.objective is None
                or not self.objective.strip()
                or not self.queries
                or self.proposed_reason is not None
                or self.selected_evidence_keys
            ):
                raise ValueError("search action fields are inconsistent")
        elif (
            self.objective is not None
            or self.queries
            or self.proposed_reason is None
        ):
            raise ValueError("finish action fields are inconsistent")
        return self


class WireResearchAspect(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    aspect: str = Field(min_length=1, max_length=1024)
    status: Literal["supported", "partial", "missing", "conflict"]
    evidence_keys: list[str] = Field(max_length=100)


class WireResearchResultVerification(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal["research_result_verification_v1"]
    status: Literal[
        "sufficient", "partial", "no_evidence", "conflict", "premise_unsupported"
    ]
    aspects: list[WireResearchAspect] = Field(min_length=1, max_length=100)
    missing_aspects: list[str] = Field(max_length=100)
    conflicts: list[str] = Field(max_length=100)


class WireAutoRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal["auto_route_v1"]
    mode: Literal["simple", "agent"]
    reason_codes: list[
        Literal[
            "single_lookup",
            "direct_summary",
            "multi_view_required",
            "multi_hop_required",
            "evidence_uncertain",
        ]
    ] = Field(min_length=1, max_length=4)


OUTPUT_SCHEMAS: dict[ChatOutputSchema, type[BaseModel]] = {
    ChatOutputSchema.ANSWER_V1: WireAnswer,
    ChatOutputSchema.CONTEXTUAL_QUERY_V2: WireContextualQuery,
    ChatOutputSchema.RETRIEVAL_AGENT_ACTION_V1: WireRetrievalAgentAction,
    ChatOutputSchema.RESEARCH_RESULT_VERIFICATION_V1: (
        WireResearchResultVerification
    ),
    ChatOutputSchema.AUTO_ROUTE_V1: WireAutoRoute,
}
