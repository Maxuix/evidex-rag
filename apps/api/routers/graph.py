"""Entity Graph configuration transport."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request

from apps.api.openapi import problem_responses
from apps.api.security import get_auth_context
from rag_kb.auth import AuthContext
from rag_kb.domain import GRAPH_EXTRACTOR_VERSION
from rag_kb.graph import GraphConfigView
from rag_kb.schemas import GraphConfigResponse, GraphConfigUpdate


router = APIRouter(prefix="/knowledge-bases", tags=["graph"])


@router.get(
    "/{kb_id}/graph-config",
    response_model=GraphConfigResponse,
    responses=problem_responses(404, 422),
)
async def get_graph_config(
    request: Request,
    kb_id: UUID,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> GraphConfigResponse:
    view = await request.app.state.dependencies.graph_configuration_service.get_view(
        context, kb_id
    )
    return _response(view)


@router.put(
    "/{kb_id}/graph-config",
    response_model=GraphConfigResponse,
    responses=problem_responses(404, 409, 422),
)
async def update_graph_config(
    request: Request,
    kb_id: UUID,
    payload: GraphConfigUpdate,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> GraphConfigResponse:
    service = request.app.state.dependencies.graph_configuration_service
    if payload.retry:
        await service.retry(context, kb_id, force_rebuild=payload.force_rebuild)
    else:
        await service.configure(
            context,
            kb_id,
            enabled=payload.enabled,
            chat_profile_revision_id=payload.chat_profile_revision_id,
            force_rebuild=payload.force_rebuild,
        )
    return _response(await service.get_view(context, kb_id))


def _response(view: GraphConfigView) -> GraphConfigResponse:
    snapshot = view.snapshot
    return GraphConfigResponse(
        knowledge_base_id=snapshot.knowledge_base_id,
        enabled=snapshot.status.value != "disabled",
        status=snapshot.status.value,
        build_id=snapshot.build_id,
        chat_profile_revision_id=snapshot.chat_profile_revision_id,
        profile_name=view.profile_name,
        provider_name=view.provider_name,
        model=view.model,
        extractor_version=snapshot.extractor_version,
        last_error_code=snapshot.last_error_code,
        eligible_chunk_count=snapshot.eligible_chunk_count,
        processed_chunk_count=snapshot.processed_chunk_count,
        extracted_chunk_count=snapshot.extracted_chunk_count,
        empty_chunk_count=snapshot.empty_chunk_count,
        protocol_skipped_count=snapshot.protocol_skipped_count,
        resource_skipped_count=snapshot.resource_skipped_count,
        allowed_skipped_count=0,
        requires_rebuild=(
            snapshot.status.value != "disabled"
            and snapshot.extractor_version != GRAPH_EXTRACTOR_VERSION
        ),
    )
