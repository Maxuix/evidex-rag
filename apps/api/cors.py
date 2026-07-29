"""Lifespan-configured CORS wrapper with explicit local origins."""

from __future__ import annotations

from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send


class ConfiguredCorsMiddleware:
    """Apply Starlette CORS only from validated runtime settings."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._configured: dict[tuple[str, ...], ASGIApp] = {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or "origin" not in Headers(scope=scope):
            await self.app(scope, receive, send)
            return

        dependencies = scope["app"].state.dependencies
        origins = dependencies.settings.security.allowed_cors_origins
        configured = self._configured.get(origins)
        if configured is None:
            configured = CORSMiddleware(
                self.app,
                allow_origins=list(origins),
                allow_credentials=False,
                allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
                allow_headers=[
                    "Content-Type",
                    "Idempotency-Key",
                    "X-Document-Metadata",
                ],
                max_age=600,
            )
            self._configured[origins] = configured
        await configured(scope, receive, send)
