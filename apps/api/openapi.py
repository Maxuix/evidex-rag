"""OpenAPI helpers for the normalized Problem Details transport."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from rag_kb.schemas import ProblemDetails


DESCRIPTIONS = {
    400: "Invalid request or cursor",
    403: "Access denied",
    404: "Resource not found",
    409: "Idempotency or resource conflict",
    413: "Request body too large",
    415: "Unsupported document media type",
    422: "Request validation failed",
    500: "Internal server error",
}


def problem_responses(*statuses: int) -> dict[int, dict[str, Any]]:
    return {
        status: {
            "model": ProblemDetails,
            "description": DESCRIPTIONS[status],
        }
        for status in statuses
    }


def install_openapi_contract(app: FastAPI) -> None:
    """Make documented Problem Details media types match runtime responses."""

    generated_openapi = app.openapi

    def openapi() -> dict[str, Any]:
        document = generated_openapi()
        for path in document.get("paths", {}).values():
            for operation in path.values():
                if not isinstance(operation, dict):
                    continue
                for response in operation.get("responses", {}).values():
                    content = response.get("content", {})
                    application_json = content.get("application/json")
                    reference = (
                        application_json.get("schema", {}).get("$ref", "")
                        if application_json
                        else ""
                    )
                    if reference.endswith("/ProblemDetails"):
                        response["content"] = {
                            "application/problem+json": application_json
                        }
        return document

    app.openapi = openapi  # type: ignore[method-assign]
