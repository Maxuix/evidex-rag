"""Fail-closed API identity boundary and request-safe metadata."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from fastapi import Depends, Request
from starlette.types import ASGIApp, Receive, Scope, Send

from apps.api.errors import problem_response
from rag_kb.auth import AuthContext, MetadataFilter
from rag_kb.schemas import ErrorCode


FORBIDDEN_IDENTITY_HEADERS = frozenset(
    {
        b"authorization",
        b"x-auth-request-user",
        b"x-client-id",
        b"x-forwarded-user",
        b"x-principal-id",
        b"x-workspace-id",
    }
)


class IdentityOverrideMiddleware:
    """Reject every client attempt to select a development identity."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if any(
            name.lower() in FORBIDDEN_IDENTITY_HEADERS
            for name, _ in scope["headers"]
        ):
            request = Request(scope, receive=receive)
            response = problem_response(
                request,
                code=ErrorCode.IDENTITY_OVERRIDE_NOT_ALLOWED,
                status=400,
                title="Identity override not allowed",
                detail="Development identity is controlled by server configuration.",
                retryable=False,
            )
            await response(scope, receive, send)
            return
        scope.setdefault("state", {})["auth_context"] = (
            scope["app"].state.dependencies.auth_provider.get_context()
        )
        await self.app(scope, receive, send)


def get_auth_context(request: Request) -> AuthContext:
    """Resolve the non-null server-owned identity for one request."""

    return request.state.auth_context


def get_metadata_filter(
    request: Request,
    context: AuthContext = Depends(get_auth_context),
) -> MetadataFilter:
    return request.app.state.dependencies.access_policy.metadata_filter(context)


@dataclass(frozen=True, slots=True)
class SafeRequestMetadata:
    """Correlation fields permitted in logs; headers and bodies are absent."""

    trace_id: str
    method: str
    path: str
    principal_id: str
    client_id: str
    workspace_id: UUID

    @classmethod
    def from_request(
        cls,
        request: Request,
        context: AuthContext,
    ) -> SafeRequestMetadata:
        return cls(
            trace_id=request.state.trace_id,
            method=request.method,
            path=request.url.path,
            principal_id=context.principal_id,
            client_id=context.client_id,
            workspace_id=context.workspace_id,
        )
