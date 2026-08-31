"""Authorized derived visual asset delivery."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response

from apps.api.security import get_auth_context
from rag_kb.auth import AuthContext
from rag_kb.domain import ResourceNotFoundError


router = APIRouter(prefix="/index-assets", tags=["index-assets"])


@router.get(
    "/{asset_id}/content",
)
async def read_index_asset(
    request: Request,
    asset_id: UUID,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> Response:
    service = getattr(request.app.state.dependencies, "index_asset_service", None)
    if service is None:
        raise ResourceNotFoundError("index asset was not found")
    value = await service.read(
        context, asset_id
    )
    return Response(
        content=value.content,
        media_type=value.snapshot.media_type,
        headers={
            "Content-Length": str(len(value.content)),
            "ETag": f'"sha256:{value.snapshot.checksum_sha256}"',
            "Cache-Control": "private, max-age=300",
            "X-Content-Type-Options": "nosniff",
        },
    )
