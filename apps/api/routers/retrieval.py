"""Authorized retrieval debug transport."""

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
    RetrievalQueryRequest,
)


router = APIRouter(prefix="/retrieval", tags=["retrieval"])


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
