"""Content-safe request completion logging middleware."""

from __future__ import annotations

import logging
from time import perf_counter

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag_kb.auth import AuthContext
from rag_kb.observability import get_logger, log_event


LOGGER = get_logger("rag_kb.api.request")


class RequestLoggingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = perf_counter()
        status_code = 500

        async def capture_status(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, capture_status)
        finally:
            state = scope.get("state", {})
            context = state.get("auth_context")
            identity = context if isinstance(context, AuthContext) else None
            duration_ms = round((perf_counter() - started) * 1000, 3)
            path = _request_path(scope)
            level = _request_log_level(path, status_code)
            log_event(
                LOGGER,
                "http_request_completed",
                level=level,
                trace_id=state.get("trace_id"),
                method=scope["method"],
                path=path,
                status_code=status_code,
                duration_ms=duration_ms,
                principal_id=identity.principal_id if identity else None,
                client_id=identity.client_id if identity else None,
                workspace_id=identity.workspace_id if identity else None,
            )


def _request_log_level(path: str, status_code: int) -> int:
    if status_code >= 500:
        return logging.ERROR
    if status_code >= 400:
        return logging.WARNING
    if path in {"/health/live", "/health/ready"}:
        return logging.DEBUG
    return logging.INFO


def _request_path(scope: Scope) -> str:
    route = scope.get("route")
    for attribute in ("path_format", "path"):
        path = getattr(route, attribute, None)
        if isinstance(path, str) and path.startswith("/") and len(path) <= 512:
            return path
    return "/<unmatched>"
