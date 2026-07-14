"""Problem Details exceptions and FastAPI exception normalization."""

from __future__ import annotations

from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from rag_kb.auth import AccessDeniedError
from rag_kb.schemas import ErrorCode, FieldViolation, ProblemDetails


class ApiProblem(Exception):
    """A safe public error whose detail may be returned to the client."""

    def __init__(
        self,
        *,
        code: ErrorCode,
        status: int,
        title: str,
        detail: str,
        retryable: bool = False,
        errors: tuple[FieldViolation, ...] | None = None,
    ) -> None:
        super().__init__(detail)
        self.code = code
        self.status = status
        self.title = title
        self.detail = detail
        self.retryable = retryable
        self.errors = errors


def install_problem_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiProblem, _api_problem_handler)  # type: ignore[arg-type]
    app.add_exception_handler(
        AccessDeniedError,
        _access_denied_handler,  # type: ignore[arg-type]
    )
    app.add_exception_handler(
        RequestValidationError,
        _request_validation_handler,  # type: ignore[arg-type]
    )
    app.add_exception_handler(
        HTTPException,
        _http_exception_handler,  # type: ignore[arg-type]
    )
    app.add_exception_handler(Exception, _unexpected_exception_handler)


async def _api_problem_handler(request: Request, error: ApiProblem) -> JSONResponse:
    return problem_response(
        request,
        code=error.code,
        status=error.status,
        title=error.title,
        detail=error.detail,
        retryable=error.retryable,
        errors=error.errors,
    )


async def _access_denied_handler(
    request: Request,
    error: AccessDeniedError,
) -> JSONResponse:
    del error
    return problem_response(
        request,
        code=ErrorCode.ACCESS_DENIED,
        status=403,
        title="Access denied",
        detail="The requested resource is not authorized for this identity.",
        retryable=False,
    )


async def _request_validation_handler(
    request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    violations = tuple(
        FieldViolation(
            location=tuple(item["loc"]),
            message=item["msg"],
            error_type=item["type"],
        )
        for item in error.errors()
    )
    invalid_idempotency_key = any(
        len(violation.location) >= 2
        and violation.location[0] == "header"
        and str(violation.location[1]).lower() == "idempotency-key"
        for violation in violations
    )
    if invalid_idempotency_key:
        code = ErrorCode.INVALID_IDEMPOTENCY_KEY
        title = "Invalid Idempotency-Key"
        detail = "Idempotency-Key must be a valid UUID."
    else:
        code = ErrorCode.REQUEST_VALIDATION_FAILED
        title = "Request validation failed"
        detail = "One or more request fields are invalid."
    return problem_response(
        request,
        code=code,
        status=422,
        title=title,
        detail=detail,
        retryable=False,
        errors=violations,
    )


async def _http_exception_handler(
    request: Request,
    error: HTTPException,
) -> JSONResponse:
    if error.status_code == 404:
        code = ErrorCode.RESOURCE_NOT_FOUND
    elif error.status_code == 405:
        code = ErrorCode.METHOD_NOT_ALLOWED
    else:
        code = ErrorCode.HTTP_ERROR
    title = _http_title(error.status_code)
    detail = error.detail if isinstance(error.detail, str) else title
    return problem_response(
        request,
        code=code,
        status=error.status_code,
        title=title,
        detail=detail,
        retryable=False,
        headers=error.headers,
    )


async def _unexpected_exception_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del error
    return problem_response(
        request,
        code=ErrorCode.INTERNAL_SERVER_ERROR,
        status=500,
        title="Internal Server Error",
        detail="The request could not be completed.",
        retryable=False,
    )


def problem_response(
    request: Request,
    *,
    code: ErrorCode,
    status: int,
    title: str,
    detail: str,
    retryable: bool,
    errors: tuple[FieldViolation, ...] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    trace_id = getattr(request.state, "trace_id", "trace-unavailable")
    problem = ProblemDetails(
        type=f"urn:rag-kb:error:{code.value.lower().replace('_', '-')}",
        title=title,
        status=status,
        detail=detail,
        instance=request.url.path,
        code=code,
        trace_id=trace_id,
        retryable=retryable,
        errors=errors,
    )
    response_headers: dict[str, str] = dict(headers or {})
    response_headers["X-Trace-ID"] = trace_id
    return JSONResponse(
        problem.model_dump(mode="json", exclude_none=True),
        status_code=status,
        media_type="application/problem+json",
        headers=response_headers,
    )


def _http_title(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return "HTTP Error"
