"""FastAPI application factory for the versioned public API."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from apps.api.cors import ConfiguredCorsMiddleware
from apps.api.dependencies import ApiDependencies, build_api_dependencies
from apps.api.errors import install_problem_handlers
from apps.api.middleware import TraceIdMiddleware
from apps.api.security import IdentityOverrideMiddleware


API_PREFIX = "/api/v1"


def create_app(
    *,
    dependencies: ApiDependencies | None = None,
    routers: Iterable[APIRouter] = (),
) -> FastAPI:
    """Create an app without opening connections or publishing placeholder routes."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved = dependencies or build_api_dependencies()
        app.state.dependencies = resolved
        try:
            yield
        finally:
            await resolved.close()
            app.state.dependencies = None

    app = FastAPI(
        title="Enterprise Knowledge Base API",
        version="0.1.0",
        openapi_version="3.1.0",
        openapi_url=f"{API_PREFIX}/openapi.json",
        docs_url=f"{API_PREFIX}/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(IdentityOverrideMiddleware)
    app.add_middleware(ConfiguredCorsMiddleware)
    app.add_middleware(TraceIdMiddleware)
    install_problem_handlers(app)
    for router in routers:
        app.include_router(router, prefix=API_PREFIX)
    return app


application = create_app()
