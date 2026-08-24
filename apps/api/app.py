"""FastAPI application factory for the versioned public API."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from apps.api.cors import ConfiguredCorsMiddleware
from apps.api.dependencies import ApiDependencies, build_api_dependencies
from apps.api.errors import install_problem_handlers
from apps.api.health import install_health_routes
from apps.api.middleware import TraceIdMiddleware
from apps.api.openapi import install_openapi_contract
from apps.api.request_logging import RequestLoggingMiddleware
from apps.api.routers import BUSINESS_ROUTERS
from apps.api.security import IdentityOverrideMiddleware
from rag_kb.observability import configure_logging, get_logger, log_event
from rag_kb.config import Settings


API_PREFIX = "/api/v1"
LOGGER = get_logger("rag_kb.api.runtime")


def create_app(
    *,
    dependencies: ApiDependencies | None = None,
    settings: Settings | None = None,
    routers: Iterable[APIRouter] | None = None,
) -> FastAPI:
    """Create an app without opening connections or publishing placeholder routes."""

    if dependencies is not None and settings is not None:
        raise ValueError("dependencies and settings are mutually exclusive")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved = dependencies or build_api_dependencies(settings=settings)
        app.state.dependencies = resolved
        try:
            configure_logging(
                level=resolved.settings.observability.log_level,
                process="api",
                log_directory=getattr(
                    resolved.settings.observability,
                    "log_directory",
                    None,
                ),
            )
            await resolved.start()
            log_event(LOGGER, "process_ready")
            yield
        finally:
            await resolved.close()
            app.state.dependencies = None
            log_event(LOGGER, "process_stopped")

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
    app.add_middleware(RequestLoggingMiddleware)
    install_problem_handlers(app)
    install_health_routes(app)
    for router in BUSINESS_ROUTERS if routers is None else routers:
        app.include_router(router, prefix=API_PREFIX)
    install_openapi_contract(app)
    return app


application = create_app()
