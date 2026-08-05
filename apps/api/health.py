"""Process liveness and read-only dependency readiness endpoints."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def install_health_routes(app: FastAPI) -> None:
    @app.get("/health/live", include_in_schema=False)
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready", include_in_schema=False)
    async def ready(request: Request) -> JSONResponse:
        try:
            await request.app.state.dependencies.check_readiness()
        except Exception:
            return JSONResponse(
                {
                    "status": "not_ready",
                    "components": {"database": "unavailable"},
                },
                status_code=503,
            )
        return JSONResponse(
            {
                "status": "ready",
                "components": {"database": "ready"},
            }
        )
