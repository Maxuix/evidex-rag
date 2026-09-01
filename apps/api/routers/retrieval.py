"""Authorized retrieval debug transport."""

from __future__ import annotations

from fastapi import APIRouter, Request

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
)
async def query_retrieval(
    request: Request,
    payload: RetrievalQueryRequest,
) -> EvidencePackResponse:
    if payload.mode == "graph":
        evidence = await request.app.state.dependencies.retrieval_service.retrieve_graph(
            GraphRetrievalRequest(
                knowledge_base_id=payload.knowledge_base_id,
                query=payload.query,
                top_k=payload.top_k,
                include_debug=payload.include_debug,
            ),
        )
    else:
        evidence = await request.app.state.dependencies.retrieval_service.retrieve(
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
