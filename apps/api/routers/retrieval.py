"""Authorized retrieval capability and debug transport."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from apps.api.openapi import problem_responses
from apps.api.security import get_auth_context
from rag_kb.auth import AuthContext
from rag_kb.domain import GraphRetrievalRequest
from rag_kb.retrieval import RetrievalRequest
from rag_kb.schemas import (
    EvidencePackResponse,
    RetrievalCapabilitiesResponse,
    RetrievalCapabilityResponse,
    RetrievalQueryRequest,
)


router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.get(
    "/capabilities",
    response_model=RetrievalCapabilitiesResponse,
    responses=problem_responses(403, 500),
)
async def retrieval_capabilities(
    request: Request,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> RetrievalCapabilitiesResponse:
    del context
    snapshot = request.app.state.dependencies.retrieval_service.capabilities_snapshot()
    return RetrievalCapabilitiesResponse(
        default_mode=snapshot.default_mode,
        modes=tuple(
            RetrievalCapabilityResponse(
                mode=capability.mode,
                strategy=capability.strategy,
                profile_version=capability.profile_version,
                enabled=capability.enabled,
            )
            for capability in snapshot.modes
        ),
    )


@router.post(
    "/query",
    response_model=EvidencePackResponse,
    responses=problem_responses(404, 409, 422, 500, 502, 503),
)
async def query_retrieval(
    request: Request,
    payload: RetrievalQueryRequest,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> EvidencePackResponse:
    if payload.mode == "graph":
        evidence = await request.app.state.dependencies.retrieval_service.retrieve_graph(
            context,
            GraphRetrievalRequest(
                knowledge_base_id=payload.knowledge_base_id,
                query=payload.query,
                top_k=payload.top_k,
                include_debug=payload.include_debug,
            ),
        )
    else:
        evidence = await request.app.state.dependencies.retrieval_service.retrieve(
            context,
            RetrievalRequest(
                knowledge_base_id=payload.knowledge_base_id,
                query=payload.query,
                top_k=payload.top_k,
                strategy=payload.strategy,
                rerank_mode=payload.rerank_mode,
                include_debug=payload.include_debug,
            ),
        )
    return EvidencePackResponse.from_domain(evidence)
