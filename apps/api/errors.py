"""Problem Details exceptions and FastAPI exception normalization."""

from __future__ import annotations

from http import HTTPStatus
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from rag_kb.auth import AccessDeniedError
from rag_kb.observability import get_logger, log_event
from rag_kb.services import (
    AnswerPolicyNotSupportedError,
    ChatSessionBusyError,
    FileAdmissionError,
    IdempotencyKeyReusedError,
    ResourceNameConflictError,
    ResourceNotFoundError,
    ResourceStateConflictError,
)
from rag_kb.schemas import ErrorCode, FieldViolation, ProblemDetails
from rag_kb.retrieval import RetrievalExecutionError


LOGGER = get_logger("rag_kb.api.errors")


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
    app.add_exception_handler(ResourceNotFoundError, _resource_not_found_handler)  # type: ignore[arg-type]
    app.add_exception_handler(ResourceNameConflictError, _resource_name_conflict_handler)  # type: ignore[arg-type]
    app.add_exception_handler(ResourceStateConflictError, _resource_state_conflict_handler)  # type: ignore[arg-type]
    app.add_exception_handler(IdempotencyKeyReusedError, _idempotency_reused_handler)  # type: ignore[arg-type]
    app.add_exception_handler(FileAdmissionError, _file_admission_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RetrievalExecutionError, _retrieval_execution_handler)  # type: ignore[arg-type]
    app.add_exception_handler(
        AnswerPolicyNotSupportedError,
        _answer_policy_not_supported_handler,  # type: ignore[arg-type]
    )
    app.add_exception_handler(
        ChatSessionBusyError,
        _chat_session_busy_handler,  # type: ignore[arg-type]
    )


async def _chat_session_busy_handler(
    request: Request, error: ChatSessionBusyError
) -> JSONResponse:
    del error
    return problem_response(
        request,
        code=ErrorCode.CHAT_SESSION_BUSY,
        status=409,
        title="Chat session busy",
        detail="This chat session already has a queued or running ChatRun.",
        retryable=True,
    )


async def _answer_policy_not_supported_handler(
    request: Request, error: AnswerPolicyNotSupportedError
) -> JSONResponse:
    del error
    return problem_response(
        request,
        code=ErrorCode.ANSWER_POLICY_NOT_SUPPORTED,
        status=422,
        title="Answer policy not supported",
        detail="The requested answer policy is not supported.",
        retryable=False,
    )


async def _retrieval_execution_handler(
    request: Request,
    error: RetrievalExecutionError,
) -> JSONResponse:
    mapping = {
        ErrorCode.CAPABILITY_NOT_ENABLED: (
            409,
            "Capability not enabled",
            "The requested retrieval capability is not enabled.",
            False,
        ),
        ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE: (
            503,
            "Embedding provider unavailable",
            "The query embedding provider is temporarily unavailable.",
            True,
        ),
        ErrorCode.EMBEDDING_RESPONSE_INVALID: (
            502,
            "Embedding response invalid",
            "The query embedding provider returned an invalid response.",
            False,
        ),
        ErrorCode.EMBEDDING_SPACE_MISMATCH: (
            503,
            "Retrieval configuration unavailable",
            "The active index is incompatible with the configured retrieval space.",
            False,
        ),
        ErrorCode.RETRIEVAL_DEADLINE_EXCEEDED: (
            503,
            "Retrieval deadline exceeded",
            "The retrieval request exceeded its execution deadline.",
            True,
        ),
    }
    status, title, detail, retryable = mapping.get(
        error.code,
        (
            500,
            "Internal Server Error",
            "The retrieval request could not be completed.",
            False,
        ),
    )
    return problem_response(
        request,
        code=error.code,
        status=status,
        title=title,
        detail=detail,
        retryable=retryable,
    )


async def _file_admission_handler(
    request: Request, error: FileAdmissionError
) -> JSONResponse:
    status = 422
    if error.code in {
        ErrorCode.FILE_TOO_LARGE,
        ErrorCode.FILE_ARCHIVE_LIMIT_EXCEEDED,
        ErrorCode.FILE_STRUCTURE_LIMIT_EXCEEDED,
    }:
        status = 413
    elif error.code in {
        ErrorCode.FILE_MEDIA_TYPE_UNSUPPORTED,
        ErrorCode.FILE_MEDIA_TYPE_MISMATCH,
        ErrorCode.PARSER_NOT_CONFIGURED,
    }:
        status = 415
    details = {
        ErrorCode.FILE_NAME_INVALID: "The document filename is invalid.",
        ErrorCode.FILE_MEDIA_TYPE_UNSUPPORTED: "The document media type is unsupported.",
        ErrorCode.FILE_MEDIA_TYPE_MISMATCH: "The filename extension and media type do not match.",
        ErrorCode.FILE_TOO_LARGE: "The document exceeds the configured byte limit.",
        ErrorCode.FILE_INVALID_UTF8: "The document is not valid UTF-8 text.",
        ErrorCode.FILE_LINE_LIMIT_EXCEEDED: "The document exceeds the configured line limit.",
        ErrorCode.FILE_STRUCTURE_LIMIT_EXCEEDED: (
            "The document exceeds a configured structural resource limit."
        ),
        ErrorCode.FILE_CONTENT_INVALID: "The document content does not match the declared format.",
        ErrorCode.FILE_ARCHIVE_LIMIT_EXCEEDED: "The document archive exceeds a configured safety limit.",
        ErrorCode.MARKDOWN_BUNDLE_INVALID: "The Markdown bundle is invalid.",
        ErrorCode.MARKDOWN_MEDIA_UNRESOLVED: "A Markdown image could not be resolved.",
        ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED: "A Markdown image format is unsupported.",
        ErrorCode.MARKDOWN_MEDIA_FETCH_FAILED: "A remote Markdown image could not be fetched safely.",
        ErrorCode.PARSER_NOT_CONFIGURED: "No parser is configured for this document format.",
    }
    markdown_media_details = {
        "reference_scheme": (
            "Markdown images must use HTTP(S), a protocol-relative public URL, "
            "a bundle-relative path, or a supported image Data URI."
        ),
        "data_uri": "A Markdown image Data URI is invalid or unsupported.",
        "image_empty": "A Markdown image is empty.",
        "image_format": (
            "A Markdown image must be PNG, JPEG, WebP, or a supported static "
            "raster image."
        ),
        "image_animated": (
            "Animated Markdown images are unsupported; provide a static image."
        ),
        "image_dimensions": (
            "A Markdown image exceeds the configured pixel or dimension limit."
        ),
        "image_total_pixels": (
            "Markdown images exceed the configured total pixel-processing limit."
        ),
        "image_decode": "A Markdown image is corrupt or cannot be decoded.",
        "html_image": "A Markdown HTML image tag is invalid or has no src.",
        "html_image_missing_src": (
            "A Markdown HTML image tag has no src attribute."
        ),
        "html_image_attribute": (
            "A Markdown HTML image tag contains an invalid or oversized attribute."
        ),
        "html_image_structure": (
            "A Markdown HTML image uses an unsupported wrapper or structure."
        ),
    }
    detail = details[error.code]
    if (
        error.code is ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED
        and error.check in markdown_media_details
    ):
        detail = markdown_media_details[error.check]
    safe_reason = (
        error.check
        if error.check in markdown_media_details
        else error.code.value
    )
    log_event(
        LOGGER,
        "file_admission_rejected",
        process="api",
        trace_id=getattr(request.state, "trace_id", None),
        reason_code=safe_reason,
    )
    return problem_response(
        request,
        code=error.code,
        status=status,
        title="Document upload rejected",
        detail=detail,
        retryable=False,
    )


async def _resource_not_found_handler(request: Request, error: ResourceNotFoundError) -> JSONResponse:
    del error
    return problem_response(
        request,
        code=ErrorCode.RESOURCE_NOT_FOUND,
        status=404,
        title="Resource not found",
        detail="The requested resource was not found.",
        retryable=False,
    )


async def _resource_name_conflict_handler(request: Request, error: ResourceNameConflictError) -> JSONResponse:
    del error
    return problem_response(
        request,
        code=ErrorCode.RESOURCE_NAME_CONFLICT,
        status=409,
        title="Resource name conflict",
        detail="A resource with the requested name already exists.",
        retryable=False,
    )


async def _resource_state_conflict_handler(request: Request, error: ResourceStateConflictError) -> JSONResponse:
    del error
    return problem_response(
        request,
        code=ErrorCode.RESOURCE_STATE_CONFLICT,
        status=409,
        title="Resource state conflict",
        detail="The requested operation conflicts with the current resource state.",
        retryable=False,
    )


async def _idempotency_reused_handler(request: Request, error: IdempotencyKeyReusedError) -> JSONResponse:
    del error
    return problem_response(
        request,
        code=ErrorCode.IDEMPOTENCY_KEY_REUSED,
        status=409,
        title="Idempotency-Key reused",
        detail="The Idempotency-Key was already used with a different request.",
        retryable=False,
    )


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
    unsupported_answer_policy = any(
        violation.error_type == "answer_policy_not_supported"
        for violation in violations
    )
    invalid_idempotency_key = any(
        len(violation.location) >= 2
        and violation.location[0] == "header"
        and str(violation.location[1]).lower() == "idempotency-key"
        for violation in violations
    )
    if unsupported_answer_policy:
        code = ErrorCode.ANSWER_POLICY_NOT_SUPPORTED
        title = "Answer policy not supported"
        detail = "The requested answer policy is not supported."
    elif invalid_idempotency_key:
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
    log_event(
        LOGGER,
        "unexpected_api_error",
        level=logging.ERROR,
        trace_id=getattr(request.state, "trace_id", None),
        method=request.method,
        path=_normalized_route_path(request),
        error_type=type(error).__name__,
    )
    return problem_response(
        request,
        code=ErrorCode.INTERNAL_SERVER_ERROR,
        status=500,
        title="Internal Server Error",
        detail="The request could not be completed.",
        retryable=False,
    )


def _normalized_route_path(request: Request) -> str:
    route = request.scope.get("route")
    for attribute in ("path_format", "path"):
        path = getattr(route, attribute, None)
        if isinstance(path, str) and path.startswith("/"):
            return path
    return "/unresolved"


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
