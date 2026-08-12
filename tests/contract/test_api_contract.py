from __future__ import annotations

import asyncio
import base64
import json
import logging
import unittest
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, ValidationError

from apps.api.app import API_PREFIX, create_app
from apps.api.errors import ApiProblem
from apps.api.idempotency import RequiredIdempotencyKey
from apps.api.pagination import decode_cursor, encode_cursor
from apps.api.routers.retrieval import router as retrieval_router
from apps.api.security import (
    SafeRequestMetadata,
    get_auth_context,
    get_metadata_filter,
)
from rag_kb.auth import (
    AccessDeniedError,
    AuthContext,
    DevelopmentAuthProvider,
    MetadataFilter,
    SingleWorkspaceAccessPolicy,
)
from rag_kb.document_processing.profiles import DOCLING_TEXT_PARSER_CONFIG
from rag_kb.domain import (
    CONTEXTUAL_QUERY_VERSION,
    AdmissionLimits,
    AnswerStyle,
    ChatCitation,
    ChatMessage,
    ChatProgressActivity,
    ChatProgressFacts,
    ChatProgressSnapshot,
    ChatProgressStage,
    ChatProgressUpdate,
    ChatPreviewDelta,
    ChatPreviewReset,
    ChatPreviewResetReason,
    ChatRun,
    ChatSession,
    ChatSessionBusyError,
    ContextualizedQuery,
    Document,
    DocumentChunk,
    DocumentChunkInspection,
    DocumentDetail,
    DocumentIndexSummary,
    DocumentMutationResult,
    DocumentVersion,
    Evidence,
    EvidencePack,
    EmbeddingRoleSummary,
    FileAdmissionError,
    IdempotencyKeyReusedError,
    IdempotencyScope,
    InsufficiencyPolicy,
    IndexingJobSnapshot,
    KnowledgeBase,
    KnowledgeBaseEmbeddingSummary,
    Page,
    QueryContextStatus,
    QueryRewriteSource,
    ResourceNotFoundError,
    ResourceStateConflictError,
    RetrievalDebug,
    RetrievalExecutionError,
    RetrievalQueryPlan,
    RetrievalStrategy,
    canonical_request_hash,
)
from rag_kb.memory import (
    empty_conversation_context,
    serialize_contextualized_query,
    serialize_conversation_context,
)
from rag_kb.services.admission import (
    FileAdmissionService,
    SUPPORTED_UPLOAD_MEDIA_TYPES_BY_EXTENSION,
)
from rag_kb.services.chat_delivery import (
    ChatEventWatcher,
    ChatSseConnectionLimiter,
    ChatTerminalWatcher,
)
from rag_kb.schemas import (
    CursorPayload,
    ErrorCode,
    KnowledgeBaseCreate,
    ModelProfileCreate,
    PaginationQuery,
)
from rag_kb.retrieval.profile import exact_profile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = UUID("01900000-0000-7000-8000-000000000001")
ALLOWED_ORIGIN = "http://127.0.0.1:3000"


def _encode_upload_metadata(
    filename: str,
    display_name: str | None = None,
    *,
    version: int = 1,
) -> str:
    payload: dict[str, object] = {"v": version, "filename": filename}
    if display_name is not None:
        payload["display_name"] = display_name
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


class ExampleInput(BaseModel):
    count: Annotated[int, Field(ge=1)]


router = APIRouter()


@router.get("/problem")
async def problem() -> None:
    raise ApiProblem(
        code=ErrorCode.CAPABILITY_NOT_ENABLED,
        status=409,
        title="Capability not enabled",
        detail="The requested capability is disabled in P1A.",
    )


@router.get("/unexpected")
async def unexpected() -> None:
    raise RuntimeError("internal-secret-must-not-leak")


@router.get("/unexpected/{resource_id}")
async def unexpected_with_path_parameter(resource_id: str) -> None:
    del resource_id
    raise RuntimeError("internal-secret-must-not-leak")


@router.get("/denied")
async def denied() -> None:
    raise AccessDeniedError("policy internals must not leak")


@router.get("/markdown-media-error")
async def markdown_media_error() -> None:
    raise FileAdmissionError(
        ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
        check="image_animated",
    )


@router.get("/markdown-html-image-error")
async def markdown_html_image_error() -> None:
    raise FileAdmissionError(
        ErrorCode.MARKDOWN_MEDIA_UNSUPPORTED,
        check="html_image_structure",
    )


@router.post("/validate")
async def validate(payload: ExampleInput) -> dict[str, int]:
    return {"count": payload.count}


@router.post("/idempotency")
async def idempotency(key: RequiredIdempotencyKey) -> dict[str, str]:
    return {"key": str(key)}


@router.get("/cursor")
async def cursor(value: str) -> CursorPayload:
    return decode_cursor(value)


@router.get("/identity")
async def identity(
    request: Request,
    context: Annotated[AuthContext, Depends(get_auth_context)],
    metadata_filter: Annotated[MetadataFilter, Depends(get_metadata_filter)],
) -> dict[str, object]:
    safe = SafeRequestMetadata.from_request(request, context)
    return {
        "principal_id": context.principal_id,
        "client_id": context.client_id,
        "workspace_id": context.workspace_id,
        "filter_workspace_id": metadata_filter.workspace_id,
        "safe_request_metadata": {
            "trace_id": safe.trace_id,
            "method": safe.method,
            "path": safe.path,
            "principal_id": safe.principal_id,
            "client_id": safe.client_id,
            "workspace_id": safe.workspace_id,
        },
    }


class StubApiDependencies:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(
            security=SimpleNamespace(allowed_cors_origins=(ALLOWED_ORIGIN,)),
            observability=SimpleNamespace(log_level="INFO"),
        )
        self.auth_provider = DevelopmentAuthProvider(
            deployment_profile="development",
            principal_id="development-principal",
            client_id="development-web",
            workspace_id=WORKSPACE,
        )
        self.access_policy = SingleWorkspaceAccessPolicy(WORKSPACE)
        self.closed = False
        self.started = False

    async def start(self) -> None:
        self.started = True
        await self.check_readiness()

    async def check_readiness(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


class StubRetrievalService:
    def __init__(self) -> None:
        self.requests = []
        self.failure: RetrievalExecutionError | None = None

    def capabilities_snapshot(self):
        return SimpleNamespace(
            default_mode="vector",
            modes=(
                SimpleNamespace(
                    mode="vector",
                    strategy="exact_vector",
                    profile_version="exact_vector_v2",
                    enabled=True,
                ),
                SimpleNamespace(
                    mode="hybrid",
                    strategy="hybrid",
                    profile_version="hybrid_fts_rrf_v2",
                    enabled=False,
                ),
            ),
        )

    async def retrieve(self, context, retrieval_request):
        self.requests.append((context, retrieval_request))
        if self.failure is not None:
            raise self.failure
        revision_id = UUID("01900000-0000-7000-8000-000000000092")
        plan = RetrievalQueryPlan(
            workspace_id=context.workspace_id,
            knowledge_base_id=retrieval_request.knowledge_base_id,
            strategy=RetrievalStrategy.EXACT_VECTOR,
            top_k=retrieval_request.top_k,
        )
        evidence = Evidence(
            rank=1,
            index_chunk_id=UUID("01900000-0000-7000-8000-000000000093"),
            indexed_document_version_id=UUID(
                "01900000-0000-7000-8000-000000000094"
            ),
            document_id=UUID("01900000-0000-7000-8000-000000000095"),
            document_version_id=UUID(
                "01900000-0000-7000-8000-000000000096"
            ),
            index_revision_id=revision_id,
            ordinal=0,
            text="safe evidence",
            source_location={"line_start": 1, "line_end": 1},
            hierarchy={},
            source_metadata={"filename": "safe.txt"},
            score=1.0,
            document_display_name="Safe evidence",
            document_original_filename="safe.txt",
        )
        return EvidencePack(
            knowledge_base_id=retrieval_request.knowledge_base_id,
            index_revision_id=revision_id,
            strategy=RetrievalStrategy.EXACT_VECTOR,
            evidence=(evidence,),
            debug=(
                RetrievalDebug(plan, revision_id, 1)
                if retrieval_request.include_debug
                else None
            ),
        )


class _PreviewSubscription:
    def __init__(self, *events: object) -> None:
        self.events: asyncio.Queue[object] = asyncio.Queue()
        for event in events:
            self.events.put_nowait(event)
        self.closed = False
        self.discarded = False

    async def next_event(self):
        return await self.events.get()

    def discard_pending(self) -> None:
        self.discarded = True
        while not self.events.empty():
            self.events.get_nowait()

    async def close(self) -> None:
        self.closed = True


class _PreviewBroker:
    def __init__(self, subscription: _PreviewSubscription) -> None:
        self.subscription = subscription
        self.run_ids: list[UUID] = []

    async def subscribe(self, run_id: UUID) -> _PreviewSubscription:
        self.run_ids.append(run_id)
        return self.subscription


@dataclass
class AsgiResponse:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> dict[str, object]:
        return json.loads(self.body)


async def request(
    app,
    method: str,
    path: str,
    *,
    query: str = "",
    headers: dict[str, str] | None = None,
    json_body: object | None = None,
    raw_body: bytes | None = None,
    suppress_application_error: bool = False,
    disconnect_immediately: bool = True,
) -> AsgiResponse:
    if json_body is not None and raw_body is not None:
        raise ValueError("request cannot contain both JSON and raw body")
    body = raw_body if raw_body is not None else (
        b"" if json_body is None else json.dumps(json_body).encode("utf-8")
    )
    request_headers = {
        "host": "testserver",
        "content-type": "application/json",
        **(headers or {}),
    }
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": query.encode("ascii"),
        "root_path": "",
        "headers": [
            (name.lower().encode("ascii"), value.encode("ascii"))
            for name, value in request_headers.items()
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "state": {},
    }
    sent_request = False
    messages: list[dict[str, object]] = []
    response_complete = asyncio.Event()

    async def receive() -> dict[str, object]:
        nonlocal sent_request
        if not sent_request:
            sent_request = True
            return {"type": "http.request", "body": body, "more_body": False}
        if not disconnect_immediately:
            await response_complete.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)
        if (
            message["type"] == "http.response.body"
            and not message.get("more_body", False)
        ):
            response_complete.set()

    try:
        await app(scope, receive, send)
    except Exception:
        if not suppress_application_error:
            raise

    start = next(
        message for message in messages if message["type"] == "http.response.start"
    )
    response_body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    response_headers = {
        name.decode("latin-1"): value.decode("latin-1")
        for name, value in start["headers"]
    }
    return AsgiResponse(
        status=start["status"],
        headers=response_headers,
        body=response_body,
    )


class ApiContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.dependencies = StubApiDependencies()
        self.app = create_app(
            dependencies=self.dependencies,  # type: ignore[arg-type]
            routers=(router,),
        )
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()

    async def asyncTearDown(self) -> None:
        await self.lifespan.__aexit__(None, None, None)

    async def test_problem_details_and_server_trace_id(self) -> None:
        response = await request(
            self.app,
            "GET",
            f"{API_PREFIX}/problem",
            headers={"x-trace-id": "client-selected-trace"},
        )
        body = response.json()
        UUID(response.headers["x-trace-id"])
        self.assertEqual(response.status, 409)
        self.assertEqual(response.headers["content-type"], "application/problem+json")
        self.assertEqual(body["code"], "CAPABILITY_NOT_ENABLED")
        self.assertEqual(body["trace_id"], response.headers["x-trace-id"])
        self.assertNotEqual(body["trace_id"], "client-selected-trace")
        self.assertFalse(body["retryable"])

    async def test_markdown_media_error_exposes_safe_specific_detail(
        self,
    ) -> None:
        with self.assertLogs("rag_kb.api.errors", level="INFO") as captured:
            response = await request(
                self.app,
                "GET",
                f"{API_PREFIX}/markdown-media-error",
            )

        self.assertEqual(response.status, 422)
        self.assertEqual(
            response.json()["code"],
            "MARKDOWN_MEDIA_UNSUPPORTED",
        )
        self.assertEqual(
            response.json()["detail"],
            "Animated Markdown images are unsupported; provide a static image.",
        )
        admission_record = next(
            record
            for record in captured.records
            if getattr(record, "safe_event", None)
            == "file_admission_rejected"
        )
        self.assertEqual(
            admission_record.safe_fields["trace_id"],
            response.json()["trace_id"],
        )
        self.assertEqual(
            admission_record.safe_fields["reason_code"],
            "image_animated",
        )

        with self.assertLogs("rag_kb.api.errors", level="INFO") as html_logs:
            html_response = await request(
                self.app,
                "GET",
                f"{API_PREFIX}/markdown-html-image-error",
            )
        self.assertEqual(html_response.status, 422)
        self.assertEqual(
            html_response.json()["detail"],
            "A Markdown HTML image uses an unsupported wrapper or structure.",
        )
        html_record = next(
            record
            for record in html_logs.records
            if getattr(record, "safe_event", None)
            == "file_admission_rejected"
        )
        self.assertEqual(
            html_record.safe_fields["reason_code"],
            "html_image_structure",
        )

    async def test_health_routes_report_process_and_dependency_state(self) -> None:
        live = await request(self.app, "GET", "/health/live")
        ready = await request(self.app, "GET", "/health/ready")

        self.assertEqual(live.status, 200)
        self.assertEqual(live.json(), {"status": "alive"})
        self.assertEqual(ready.status, 200)
        self.assertEqual(
            ready.json(),
            {
                "status": "ready",
                "components": {"database": "ready"},
            },
        )

    async def test_readiness_failure_is_generic_and_does_not_leak_details(self) -> None:
        async def unavailable() -> None:
            raise RuntimeError("database-password-must-not-leak")

        self.dependencies.check_readiness = unavailable  # type: ignore[method-assign]
        response = await request(self.app, "GET", "/health/ready")
        self.assertEqual(response.status, 503)
        self.assertEqual(
            response.json(),
            {
                "status": "not_ready",
                "components": {"database": "unavailable"},
            },
        )
        self.assertNotIn("database-password", response.body.decode("utf-8"))

    async def test_validation_and_idempotency_header_errors_are_normalized(
        self,
    ) -> None:
        validation = await request(
            self.app,
            "POST",
            f"{API_PREFIX}/validate",
            json_body={"count": 0},
        )
        self.assertEqual(validation.status, 422)
        self.assertEqual(validation.json()["code"], "REQUEST_VALIDATION_FAILED")
        self.assertTrue(validation.json()["errors"])

        for headers in ({}, {"idempotency-key": "not-a-uuid"}):
            with self.subTest(headers=headers):
                invalid_key = await request(
                    self.app,
                    "POST",
                    f"{API_PREFIX}/idempotency",
                    headers=headers,
                )
                self.assertEqual(invalid_key.status, 422)
                self.assertEqual(
                    invalid_key.json()["code"],
                    "INVALID_IDEMPOTENCY_KEY",
                )

    async def test_internal_errors_do_not_expose_exception_details(self) -> None:
        with self.assertLogs("rag_kb.api.errors", level=logging.ERROR) as captured:
            response = await request(
                self.app,
                "GET",
                f"{API_PREFIX}/unexpected/runtime-resource-id",
                suppress_application_error=True,
            )
        body = response.json()
        self.assertEqual(response.status, 500)
        self.assertEqual(body["code"], "INTERNAL_SERVER_ERROR")
        self.assertNotIn("internal-secret", response.body.decode("utf-8"))
        error_record = next(
            record
            for record in captured.records
            if getattr(record, "safe_event", None) == "unexpected_api_error"
        )
        self.assertEqual(error_record.safe_fields["trace_id"], body["trace_id"])
        self.assertEqual(error_record.safe_fields["method"], "GET")
        self.assertEqual(
            error_record.safe_fields["path"],
            f"{API_PREFIX}/unexpected/{{resource_id}}",
        )
        self.assertEqual(error_record.safe_fields["error_type"], "RuntimeError")
        self.assertEqual(error_record.safe_exception["type"], "RuntimeError")
        self.assertTrue(error_record.safe_exception["fingerprint"])
        self.assertNotIn("internal-secret", repr(captured.records))
        self.assertNotIn("runtime-resource-id", repr(captured.records))

    async def test_access_denial_is_normalized_without_policy_details(self) -> None:
        response = await request(self.app, "GET", f"{API_PREFIX}/denied")
        self.assertEqual(response.status, 403)
        self.assertEqual(response.json()["code"], "ACCESS_DENIED")
        self.assertNotIn("policy internals", response.body.decode("utf-8"))

    async def test_unknown_route_uses_problem_details(self) -> None:
        response = await request(self.app, "GET", f"{API_PREFIX}/missing")
        self.assertEqual(response.status, 404)
        self.assertEqual(response.json()["code"], "RESOURCE_NOT_FOUND")

    async def test_cursor_round_trip_and_invalid_cursor_problem(self) -> None:
        payload = CursorPayload(sort="-created_at", values=("2026-07-14", "item-1"))
        encoded = encode_cursor(payload)
        self.assertNotIn("2026-07-14", encoded)
        self.assertEqual(decode_cursor(encoded), payload)

        response = await request(
            self.app,
            "GET",
            f"{API_PREFIX}/cursor",
            query="value=not!base64",
        )
        self.assertEqual(response.status, 400)
        self.assertEqual(response.json()["code"], "INVALID_CURSOR")

    async def test_valid_idempotency_key_is_a_uuid(self) -> None:
        key = uuid4()
        response = await request(
            self.app,
            "POST",
            f"{API_PREFIX}/idempotency",
            headers={"idempotency-key": str(key)},
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(response.json()["key"], str(key))

    async def test_identity_is_server_owned_and_safe_metadata_excludes_input(self) -> None:
        response = await request(
            self.app,
            "GET",
            f"{API_PREFIX}/identity",
            query="secret=must-not-appear",
            headers={"x-api-key": "must-not-appear"},
        )

        body = response.json()
        self.assertEqual(response.status, 200)
        self.assertEqual(body["principal_id"], "development-principal")
        self.assertEqual(body["client_id"], "development-web")
        self.assertEqual(body["workspace_id"], str(WORKSPACE))
        self.assertEqual(body["filter_workspace_id"], str(WORKSPACE))
        self.assertEqual(
            body["safe_request_metadata"]["path"],  # type: ignore[index]
            f"{API_PREFIX}/identity",
        )
        rendered = response.body.decode("utf-8")
        self.assertNotIn("must-not-appear", rendered)
        self.assertNotIn("x-api-key", rendered)

    async def test_identity_override_headers_are_rejected(self) -> None:
        header_names = (
            "authorization",
            "x-auth-request-user",
            "x-client-id",
            "x-forwarded-user",
            "x-principal-id",
            "x-workspace-id",
        )
        for header_name in header_names:
            with self.subTest(header_name=header_name):
                response = await request(
                    self.app,
                    "GET",
                    f"{API_PREFIX}/identity",
                    headers={header_name: "client-selected-value"},
                )
                self.assertEqual(response.status, 400)
                self.assertEqual(
                    response.json()["code"],
                    "IDENTITY_OVERRIDE_NOT_ALLOWED",
                )
                self.assertEqual(
                    response.json()["trace_id"],
                    response.headers["x-trace-id"],
                )

    async def test_cors_is_explicit_origin_only_and_never_credentialed(self) -> None:
        allowed = await request(
            self.app,
            "GET",
            f"{API_PREFIX}/identity",
            headers={"origin": ALLOWED_ORIGIN},
        )
        self.assertEqual(allowed.headers["access-control-allow-origin"], ALLOWED_ORIGIN)
        self.assertNotIn("access-control-allow-credentials", allowed.headers)

        disallowed = await request(
            self.app,
            "GET",
            f"{API_PREFIX}/identity",
            headers={"origin": "https://attacker.example"},
        )
        self.assertNotIn("access-control-allow-origin", disallowed.headers)
        self.assertNotIn("access-control-allow-credentials", disallowed.headers)

    async def test_cors_preflight_allows_only_declared_method_and_header(self) -> None:
        allowed = await request(
            self.app,
            "OPTIONS",
            f"{API_PREFIX}/identity",
            headers={
                "origin": ALLOWED_ORIGIN,
                "access-control-request-method": "GET",
                "access-control-request-headers": (
                    "content-type,idempotency-key,x-document-metadata"
                ),
            },
        )
        self.assertEqual(allowed.status, 200)
        self.assertEqual(allowed.headers["access-control-allow-origin"], ALLOWED_ORIGIN)
        self.assertNotIn("access-control-allow-credentials", allowed.headers)

        put_allowed = await request(
            self.app,
            "OPTIONS",
            f"{API_PREFIX}/identity",
            headers={
                "origin": ALLOWED_ORIGIN,
                "access-control-request-method": "PUT",
            },
        )
        self.assertEqual(put_allowed.status, 200)

        disallowed = await request(
            self.app,
            "OPTIONS",
            f"{API_PREFIX}/identity",
            headers={
                "origin": ALLOWED_ORIGIN,
                "access-control-request-method": "CONNECT",
            },
        )
        self.assertEqual(disallowed.status, 400)

    async def test_lifespan_owns_dependency_shutdown(self) -> None:
        class StubDependencies:
            closed = False
            started = False
            settings = SimpleNamespace(
                security=SimpleNamespace(allowed_cors_origins=()),
                observability=SimpleNamespace(log_level="INFO"),
            )

            async def start(self) -> None:
                self.started = True

            async def close(self) -> None:
                self.closed = True

        dependencies = StubDependencies()
        app = create_app(dependencies=dependencies)  # type: ignore[arg-type]
        async with app.router.lifespan_context(app):
            self.assertIs(app.state.dependencies, dependencies)
            self.assertTrue(dependencies.started)
            self.assertFalse(dependencies.closed)

        self.assertTrue(dependencies.closed)
        self.assertIsNone(app.state.dependencies)


class CommonContractTests(unittest.TestCase):
    def test_flexible_embedding_requests_are_discriminated_and_bounded(self) -> None:
        provider_id = uuid4()
        for dimension in ("auto", 64, 724, 4096):
            profile = ModelProfileCreate.model_validate(
                {
                    "provider_id": str(provider_id),
                    "name": "Embedding",
                    "kind": "text_embedding",
                    "model": "embedding-model",
                    "parameters": {
                        "type": "embedding",
                        "dimension": dimension,
                    },
                }
            )
            self.assertEqual(profile.parameters.dimension, dimension)
        for dimension in (63, 4097):
            with self.assertRaises(ValidationError):
                ModelProfileCreate.model_validate(
                    {
                        "provider_id": str(provider_id),
                        "name": "Embedding",
                        "kind": "text_embedding",
                        "model": "embedding-model",
                        "parameters": {
                            "type": "embedding",
                            "dimension": dimension,
                        },
                    }
                )

        unified = KnowledgeBaseCreate.model_validate(
            {
                "name": "Unified",
                "parsing": {"preset": "multimodal_local_v2"},
                "embedding": {
                    "strategy": "unified_multimodal",
                    "profile_revision_id": str(uuid4()),
                },
            }
        )
        self.assertEqual(unified.embedding.strategy, "unified_multimodal")
        with self.assertRaises(ValidationError):
            KnowledgeBaseCreate.model_validate(
                {
                    "name": "Invalid",
                    "embedding": {"strategy": "dual_space"},
                }
            )

    def test_pagination_bounds_and_sort_shape(self) -> None:
        self.assertEqual(PaginationQuery().limit, 50)
        self.assertEqual(PaginationQuery(limit=100, sort="-created_at").limit, 100)
        for payload in ({"limit": 0}, {"limit": 101}, {"sort": "created at"}):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                PaginationQuery.model_validate(payload)

    def test_idempotency_scope_and_request_hash_are_stable(self) -> None:
        first = canonical_request_hash({"question": "知识", "top_k": 8})
        second = canonical_request_hash({"top_k": 8, "question": "知识"})
        self.assertEqual(first, second)
        self.assertRegex(first, r"^sha256:[0-9a-f]{64}$")

        key = uuid4()
        endpoint = "POST /api/v1/chat/runs"
        scope = IdempotencyScope("principal", "client", endpoint, key)
        self.assertEqual(scope.idempotency_key, key)
        with self.assertRaises(ValueError):
            IdempotencyScope("", "client", endpoint, key)
        with self.assertRaises(ValueError):
            IdempotencyScope("principal", "client", "/api/v1/chat/runs", key)

    def test_production_openapi_publishes_only_eligible_business_routes(self) -> None:
        production = create_app().openapi()
        snapshot = json.loads(
            (PROJECT_ROOT / "tests/contract/snapshots/openapi-v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(production, snapshot)
        self.assertEqual(
            set(production["paths"]),
            {
                "/api/v1/documents/{document_id}",
                "/api/v1/documents/{document_id}/chunks",
                "/api/v1/documents/{document_id}/chunks/{chunk_id}",
                "/api/v1/documents/{document_id}/versions",
                "/api/v1/knowledge-bases",
                "/api/v1/knowledge-bases/{kb_id}",
                "/api/v1/knowledge-bases/{kb_id}/documents",
                "/api/v1/knowledge-bases/{kb_id}/indexing-jobs",
                "/api/v1/indexing-jobs/{job_id}",
                "/api/v1/indexing-jobs/{job_id}/retry",
                "/api/v1/index-assets/{asset_id}/content",
                "/api/v1/retrieval/query",
                "/api/v1/retrieval/capabilities",
                "/api/v1/chat/sessions",
                "/api/v1/chat/sessions/{session_id}/messages",
                "/api/v1/chat/runs",
                "/api/v1/chat/runs/{run_id}",
                "/api/v1/chat/runs/{run_id}/final-context",
                "/api/v1/chat/runs/{run_id}/events",
                "/api/v1/model-settings",
                "/api/v1/model-providers",
                "/api/v1/model-providers/{provider_id}",
                "/api/v1/model-providers/{provider_id}/models",
                "/api/v1/model-profiles",
                "/api/v1/model-profiles/{profile_id}",
                "/api/v1/model-profiles/{profile_id}/validate",
                "/api/v1/model-selection",
            },
        )
        upload = production["paths"]["/api/v1/knowledge-bases/{kb_id}/documents"]["post"]
        self.assertEqual(upload["responses"]["202"]["description"], "Successful Response")
        self.assertEqual(
            upload["requestBody"]["content"]["text/markdown"]["schema"]["format"],
            "binary",
        )
        supported_upload_media_types = {
            media_type
            for _, media_type in SUPPORTED_UPLOAD_MEDIA_TYPES_BY_EXTENSION
        }
        self.assertEqual(
            set(upload["requestBody"]["content"]),
            supported_upload_media_types,
        )
        version_upload = production["paths"][
            "/api/v1/documents/{document_id}/versions"
        ]["post"]
        self.assertEqual(
            set(version_upload["requestBody"]["content"]),
            supported_upload_media_types,
        )
        create_conflict = production["paths"]["/api/v1/knowledge-bases"][
            "post"
        ]["responses"]["409"]
        self.assertEqual(
            set(create_conflict["content"]), {"application/problem+json"}
        )
        self.assertEqual(
            create_conflict["content"]["application/problem+json"]["schema"][
                "$ref"
            ],
            "#/components/schemas/ProblemDetails",
        )
        event_limit = production["paths"][
            "/api/v1/chat/runs/{run_id}/events"
        ]["get"]["responses"]["429"]
        self.assertEqual(
            set(event_limit["content"]),
            {"application/problem+json"},
        )


class RetrievalApiContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.dependencies = StubApiDependencies()
        self.service = StubRetrievalService()
        self.dependencies.retrieval_service = self.service
        self.app = create_app(
            dependencies=self.dependencies,  # type: ignore[arg-type]
            routers=(retrieval_router,),
        )
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()

    async def asyncTearDown(self) -> None:
        await self.lifespan.__aexit__(None, None, None)

    async def test_exact_retrieval_returns_evidence_and_authorized_safe_debug(self) -> None:
        kb_id = UUID("01900000-0000-7000-8000-000000000091")
        response = await request(
            self.app,
            "POST",
            f"{API_PREFIX}/retrieval/query",
            json_body={
                "knowledge_base_id": str(kb_id),
                "query": "  查询 ABC-42  ",
                "top_k": 5,
                "strategy": "exact_vector",
                "include_debug": True,
            },
        )

        self.assertEqual(response.status, 200)
        body = response.json()
        self.assertEqual(body["knowledge_base_id"], str(kb_id))
        self.assertEqual(body["evidence"][0]["text"], "safe evidence")
        self.assertEqual(body["debug"]["query_plan"]["strategy"], "exact_vector")
        self.assertNotIn("query", body["debug"]["query_plan"])
        _, retrieval_request = self.service.requests[0]
        self.assertEqual(retrieval_request.query, "查询 ABC-42")
        self.assertEqual(retrieval_request.top_k, 5)

    async def test_capabilities_are_strict_process_snapshot_without_uow(self) -> None:
        response = await request(
            self.app,
            "GET",
            f"{API_PREFIX}/retrieval/capabilities",
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(
            response.json(),
            {
                "default_mode": "vector",
                "modes": [
                    {
                        "mode": "vector",
                        "strategy": "exact_vector",
                        "profile_version": "exact_vector_v2",
                        "enabled": True,
                    },
                    {
                        "mode": "hybrid",
                        "strategy": "hybrid",
                        "profile_version": "hybrid_fts_rrf_v2",
                        "enabled": False,
                    },
                ],
            },
        )
        self.assertEqual(self.service.requests, [])

    async def test_client_cannot_inject_mandatory_filters(self) -> None:
        forbidden_fields = (
            "workspace_id",
            "index_revision_id",
            "revision_selector",
            "current_document_version_only",
            "build_status",
            "serving_status",
            "distance_metric",
            "candidate_count",
            "ef_search",
            "iterative_scan",
            "filters",
            "debug",
        )
        for field_name in forbidden_fields:
            with self.subTest(field_name=field_name):
                response = await request(
                    self.app,
                    "POST",
                    f"{API_PREFIX}/retrieval/query",
                    json_body={
                        "knowledge_base_id": (
                            "01900000-0000-7000-8000-000000000091"
                        ),
                        "query": "query",
                        "include_debug": True,
                        field_name: "client-controlled",
                    },
                )
                self.assertEqual(response.status, 422)
                self.assertEqual(
                    response.json()["code"],
                    "REQUEST_VALIDATION_FAILED",
                )
        self.assertEqual(self.service.requests, [])

    async def test_retrieval_failures_are_content_safe_problem_details(self) -> None:
        cases = (
            (ErrorCode.CAPABILITY_NOT_ENABLED, 409, False),
            (ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE, 503, True),
            (ErrorCode.EMBEDDING_RESPONSE_INVALID, 502, False),
            (ErrorCode.LOCAL_RERANKER_UNAVAILABLE, 503, False),
            (ErrorCode.EMBEDDING_SPACE_MISMATCH, 503, False),
            (ErrorCode.RETRIEVAL_DEADLINE_EXCEEDED, 503, True),
            (ErrorCode.INTERNAL_SERVER_ERROR, 500, False),
        )
        for code, status, retryable in cases:
            with self.subTest(code=code):
                self.service.failure = RetrievalExecutionError(
                    code,
                    diagnostic={"secret": "must-not-leak"},
                )
                response = await request(
                    self.app,
                    "POST",
                    f"{API_PREFIX}/retrieval/query",
                    json_body={
                        "knowledge_base_id": (
                            "01900000-0000-7000-8000-000000000091"
                        ),
                        "query": "query",
                    },
                )
                body = response.json()
                self.assertEqual(response.status, status)
                self.assertEqual(body["code"], code.value)
                self.assertEqual(body["retryable"], retryable)
                self.assertNotIn("must-not-leak", response.body.decode())


class _FakeKnowledgeBaseService:
    def __init__(self) -> None:
        self.value = _knowledge_base_value()

    async def create(
        self,
        context,
        key,
        *,
        name,
        parsing_preset,
        chunking_preset,
        retrieval_defaults,
        answer_policy_defaults,
        embedding_selection=None,
    ):
        del context, key, embedding_selection
        from rag_kb.document_processing.profiles import profile_for_preset

        self.value = dataclass_replace(
            self.value,
            name=name,
            parser_config=profile_for_preset(
                chunking_preset, parsing_preset
            ).parser_config,
            chunking_config=profile_for_preset(
                chunking_preset, parsing_preset
            ).chunking_config,
            retrieval_defaults=retrieval_defaults,
            answer_policy_defaults=answer_policy_defaults,
        )
        return self.value

    async def get(self, context, kb_id):
        del context
        if kb_id != self.value.id:
            raise ResourceNotFoundError("internal detail")
        return self.value

    async def list(self, context, *, limit, sort, after):
        del context, limit, sort, after
        return Page(items=(self.value,))

    async def update(
        self,
        context,
        key,
        kb_id,
        *,
        name,
        retrieval_defaults,
        answer_policy_defaults,
    ):
        del context
        if key == UUID("00000000-0000-0000-0000-000000000099"):
            raise IdempotencyKeyReusedError("internal hash detail")
        if kb_id != self.value.id:
            raise ResourceNotFoundError("internal detail")
        self.value = dataclass_replace(
            self.value,
            name=name or self.value.name,
            retrieval_defaults=retrieval_defaults or self.value.retrieval_defaults,
            answer_policy_defaults=(
                answer_policy_defaults or self.value.answer_policy_defaults
            ),
        )
        return self.value

    async def delete(self, context, key, kb_id):
        del context, key
        if kb_id != self.value.id:
            raise ResourceNotFoundError("internal detail")
        self.value = dataclass_replace(
            self.value,
            deleted_at=datetime(2026, 8, 9, tzinfo=UTC),
        )
        return self.value


class _FakeDocumentService:
    def __init__(self) -> None:
        self.value = _document_value()

    async def get(self, context, document_id):
        del context
        if document_id != self.value.id:
            raise ResourceNotFoundError("internal detail")
        return self.value

    async def get_detail(self, context, document_id):
        document = await self.get(context, document_id)
        return DocumentDetail(
            document=document,
            index=DocumentIndexSummary(
                indexed_document_version_id=UUID(
                    "01900000-0000-7000-8000-000000000026"
                ),
                index_revision_id=UUID(
                    "01900000-0000-7000-8000-000000000012"
                ),
                build_status="ready",
                serving_status="serving",
                unit_count=7,
                asset_count=3,
                representation_count=11,
                composite_chunk_count=4,
                visual_unit_count=3,
                relation_count=5,
                text_representation_count=4,
                native_image_representation_count=2,
                table_representation_count=2,
            ),
        )

    async def list(self, context, *, kb_id, limit, sort, after):
        del context, limit, sort, after
        return Page(items=(self.value,)) if kb_id == self.value.kb_id else Page(items=())

    async def inspect_chunks(self, context, document_id, *, limit, after):
        del context, limit
        if document_id != self.value.id:
            raise ResourceNotFoundError("internal detail")
        return DocumentChunkInspection(
            document_id=self.value.id,
            document_version_id=self.value.current_version.id,
            indexed_document_version_id=UUID("01900000-0000-7000-8000-000000000026"),
            index_revision_id=UUID("01900000-0000-7000-8000-000000000012"),
            total_chunks=1,
            items=(
                DocumentChunk(
                    id=UUID("01900000-0000-7000-8000-000000000031"),
                    ordinal=0,
                    modality="text",
                    content="Stable chunk text",
                    token_count=3,
                    source_location={"page_number": 1},
                    hierarchy={"section": "Overview"},
                    source_metadata={},
                    evidence_group_key="group-1",
                    representations=("text",),
                ),
            ) if after is None else (),
        )

    async def delete(self, context, key, document_id):
        del context, key
        if document_id != self.value.id:
            raise ResourceNotFoundError("internal detail")
        return DocumentMutationResult(
            document=self.value,
            source_change_id=UUID("01900000-0000-7000-8000-000000000025"),
            source_change_seq=2,
            index_revision_id=UUID("01900000-0000-7000-8000-000000000012"),
        )

    async def exclude_chunk(self, context, *, document_id, chunk_id):
        del context
        if (
            document_id != self.value.id
            or chunk_id != UUID("01900000-0000-7000-8000-000000000031")
        ):
            raise ResourceNotFoundError("internal chunk detail")
        return datetime(2026, 8, 9, tzinfo=UTC)


class _FakeSourceFileService:
    def __init__(self, documents: _FakeDocumentService) -> None:
        self.documents = documents
        self.calls: list[dict[str, object]] = []

    async def store_and_activate(self, context, key, **kwargs):
        del context, key
        content = kwargs["source"].read()
        self.calls.append({**kwargs, "content": content, "source": None})
        prior = self.documents.value
        version = dataclass_replace(
            prior.current_version,
            original_filename=kwargs["original_filename"],
            media_type=kwargs["media_type"],
            size_bytes=len(content),
        )
        document = dataclass_replace(
            prior,
            display_name=kwargs["display_name"],
            current_version=version,
        )
        self.documents.value = document
        return DocumentMutationResult(
            document=document,
            document_version_id=version.id,
            source_change_id=UUID("01900000-0000-7000-8000-000000000025"),
            source_change_seq=1,
            indexed_document_version_id=UUID("01900000-0000-7000-8000-000000000026"),
            index_revision_id=UUID("01900000-0000-7000-8000-000000000012"),
            job_id=UUID("01900000-0000-7000-8000-000000000027"),
            job_status="queued",
        )


class _FakeIndexingJobService:
    def __init__(self) -> None:
        now = datetime(2026, 7, 14, tzinfo=UTC)
        self.value = IndexingJobSnapshot(
            job_id=UUID("01900000-0000-7000-8000-000000000027"),
            kb_id=_knowledge_base_value().id,
            document_id=_document_value().id,
            document_version_id=_document_value().current_version.id,
            indexed_document_version_id=UUID(
                "01900000-0000-7000-8000-000000000026"
            ),
            index_revision_id=_knowledge_base_value().active_index_revision_id,
            job_status="failed",
            phase="embedding",
            attempt=3,
            build_status="failed",
            serving_status="candidate",
            claimed_at=None,
            heartbeat_at=None,
            next_attempt_at=None,
            error_code="EMBEDDING_PROVIDER_UNAVAILABLE",
            error_detail={"attempt": 3},
            can_retry=True,
            created_at=now,
            updated_at=now,
        )

    async def get(self, context, job_id):
        del context
        if job_id != self.value.job_id:
            raise ResourceNotFoundError("internal indexing detail")
        return self.value

    async def list(self, context, *, kb_id, limit, after):
        del context, limit, after
        if kb_id != self.value.kb_id:
            raise ResourceNotFoundError("internal knowledge base detail")
        return Page(items=(self.value,))

    async def retry(self, context, key, job_id):
        del context, key
        if job_id != self.value.job_id:
            raise ResourceNotFoundError("internal indexing detail")
        if not self.value.can_retry:
            raise ResourceStateConflictError("internal indexing state")
        self.value = dataclass_replace(
            self.value,
            job_status="queued",
            phase="queued",
            attempt=0,
            build_status="queued",
            error_code=None,
            error_detail=None,
            can_retry=False,
        )
        return self.value


class _FakeChatService:
    def __init__(self) -> None:
        self.session = _chat_session_value()
        self.run = _chat_run_value(self.session)
        self.create_run_calls: list[dict[str, object]] = []

    async def create_session(self, context, *, kb_id, title):
        del context
        if kb_id != _knowledge_base_value().id:
            raise ResourceNotFoundError("internal chat knowledge-base detail")
        self.session = dataclass_replace(self.session, kb_id=kb_id, title=title)
        self.run = _chat_run_value(self.session)
        return self.session

    async def list_sessions(self, context, *, limit, sort, after, kb_id=None):
        del context, limit, sort, after
        if kb_id is not None and kb_id != self.session.kb_id:
            raise ResourceNotFoundError("internal chat knowledge-base detail")
        return Page(items=(self.session,))

    async def list_messages(
        self, context, session_id, *, limit, sort, after
    ):
        del context, limit, sort, after
        if session_id != self.session.id:
            raise ResourceNotFoundError("internal chat session detail")
        return Page(items=_chat_messages(self.run))

    async def create_run(self, context, key, **values):
        del context
        if values["retrieval_mode"] == "hybrid":
            raise RetrievalExecutionError(
                ErrorCode.CAPABILITY_NOT_ENABLED,
                diagnostic={"capability": "hybrid"},
            )
        self.create_run_calls.append({"key": key, **values})
        if key == UUID("00000000-0000-0000-0000-000000000099"):
            raise IdempotencyKeyReusedError("internal chat hash detail")
        if key == UUID("00000000-0000-0000-0000-000000000098"):
            raise ChatSessionBusyError("internal active run detail")
        if values["session_id"] != self.session.id:
            raise ResourceNotFoundError("internal chat session detail")
        policy = {
            "grounding_policy": "evidence_only",
            "answer_style": (
                values["answer_style"] or AnswerStyle.CONCISE
            ).value,
            "insufficiency_policy": (
                values["insufficiency_policy"]
                or InsufficiencyPolicy.PARTIAL_ANSWER
            ).value,
            "citation_required": True,
            "citation_granularity": "claim_level",
            "answer_task": "answer",
            "policy_version": "p1",
        }
        snapshot = empty_conversation_context()
        original = ContextualizedQuery(
            version=CONTEXTUAL_QUERY_VERSION,
            status=QueryContextStatus.ORIGINAL,
            original_query=values["message"],
            standalone_query=values["message"],
            context_hash=snapshot.content_hash,
            rewrite_source=QueryRewriteSource.ORIGINAL,
        )
        self.run = dataclass_replace(
            self.run,
            effective_policy=policy,
            retrieval_strategy=exact_profile(
                top_k=values["top_k"],
                rerank=False,
            ).as_dict(),
            conversation_context=serialize_conversation_context(snapshot),
            contextualized_query=serialize_contextualized_query(original),
        )
        return self.run

    async def get_run(self, context, run_id):
        del context
        if run_id != self.run.id:
            raise ResourceNotFoundError("internal chat run detail")
        return self.run


class ContentApiContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.dependencies = StubApiDependencies()
        self.dependencies.knowledge_base_service = _FakeKnowledgeBaseService()
        self.dependencies.document_service = _FakeDocumentService()
        self.dependencies.file_admission_service = FileAdmissionService(
            AdmissionLimits(max_bytes=32, max_lines=3, max_csv_cells=2)
        )
        self.dependencies.source_file_service = _FakeSourceFileService(
            self.dependencies.document_service
        )
        self.dependencies.indexing_job_service = _FakeIndexingJobService()
        self.dependencies.chat_service = _FakeChatService()
        self.dependencies.chat_terminal_watcher = ChatTerminalWatcher(
            self.dependencies.chat_service,  # type: ignore[arg-type]
            poll_interval_seconds=0.001,
            jitter_ratio=0,
            max_duration_seconds=0.01,
        )
        self.dependencies.chat_event_watcher = ChatEventWatcher(
            self.dependencies.chat_terminal_watcher
        )
        self.dependencies.chat_preview_broker = None
        self.dependencies.chat_sse_connection_limiter = (
            ChatSseConnectionLimiter(2)
        )
        self.app = create_app(dependencies=self.dependencies)  # type: ignore[arg-type]
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()

    async def asyncTearDown(self) -> None:
        await self.lifespan.__aexit__(None, None, None)

    async def test_knowledge_base_routes_require_idempotency_and_use_typed_dtos(self) -> None:
        missing = await request(
            self.app,
            "POST",
            f"{API_PREFIX}/knowledge-bases",
            json_body={"name": "Engineering"},
        )
        self.assertEqual(missing.status, 422)
        self.assertEqual(missing.json()["code"], "INVALID_IDEMPOTENCY_KEY")

        created = await request(
            self.app,
            "POST",
            f"{API_PREFIX}/knowledge-bases",
            headers={"idempotency-key": str(uuid4())},
            json_body={"name": "  Engineering  ", "retrieval_defaults": {"top_k": 8}},
        )
        self.assertEqual(created.status, 201)
        self.assertEqual(created.json()["name"], "Engineering")
        self.assertEqual(created.json()["retrieval_defaults"]["strategy"], "exact_vector")
        self.assertEqual(created.json()["retrieval_defaults"]["top_k"], 8)
        self.assertEqual(
            created.json()["chunking"],
            {
                "preset": "structural_balanced_v2",
                "profile": "structural_by_title_token_v4",
            },
        )
        self.assertEqual(
            created.json()["answer_policy_defaults"],
            {
                "answer_style": "concise",
                "insufficiency_policy": "partial_answer",
            },
        )

        updated = await request(
            self.app,
            "PATCH",
            f"{API_PREFIX}/knowledge-bases/{_knowledge_base_value().id}",
            headers={"idempotency-key": str(uuid4())},
            json_body={
                "answer_policy_defaults": {
                    "answer_style": "summary",
                    "insufficiency_policy": "partial_answer",
                }
            },
        )
        self.assertEqual(updated.status, 200)
        self.assertEqual(
            updated.json()["answer_policy_defaults"],
            {"answer_style": "summary", "insufficiency_policy": "partial_answer"},
        )

        semantic = await request(
            self.app,
            "POST",
            f"{API_PREFIX}/knowledge-bases",
            headers={"idempotency-key": str(uuid4())},
            json_body={
                "name": "Semantic",
                "chunking": {"preset": "semantic_balanced_v1"},
            },
        )
        self.assertEqual(semantic.status, 201)
        self.assertEqual(
            semantic.json()["chunking"],
            {
                "preset": "semantic_balanced_v1",
                "profile": "semantic_breakpoint_v3",
            },
        )

        for body in (
            {"name": "Unknown", "chunking": {"preset": "unknown"}},
            {
                "name": "Tuned",
                "chunking": {
                    "preset": "semantic_balanced_v1",
                    "min_tokens": 10,
                },
            },
        ):
            rejected = await request(
                self.app,
                "POST",
                f"{API_PREFIX}/knowledge-bases",
                headers={"idempotency-key": str(uuid4())},
                json_body=body,
            )
            self.assertEqual(rejected.status, 422)

        immutable = await request(
            self.app,
            "PATCH",
            f"{API_PREFIX}/knowledge-bases/{_knowledge_base_value().id}",
            headers={"idempotency-key": str(uuid4())},
            json_body={"chunking": {"preset": "semantic_balanced_v1"}},
        )
        self.assertEqual(immutable.status, 422)

        listed = await request(self.app, "GET", f"{API_PREFIX}/knowledge-bases")
        self.assertEqual(listed.status, 200)
        self.assertEqual(len(listed.json()["items"]), 1)

        invalid_position = encode_cursor(
            CursorPayload(sort="created_at", values=("not-a-time", "not-a-uuid"))
        )
        invalid = await request(
            self.app,
            "GET",
            f"{API_PREFIX}/knowledge-bases",
            query=f"cursor={invalid_position}",
        )
        self.assertEqual(invalid.status, 400)
        self.assertEqual(invalid.json()["code"], "INVALID_CURSOR")

    async def test_problem_mappings_redact_domain_details(self) -> None:
        missing = await request(
            self.app,
            "GET",
            f"{API_PREFIX}/knowledge-bases/{uuid4()}",
        )
        self.assertEqual(missing.status, 404)
        self.assertEqual(missing.json()["code"], "RESOURCE_NOT_FOUND")
        self.assertNotIn("internal detail", missing.body.decode())

        conflict = await request(
            self.app,
            "PATCH",
            f"{API_PREFIX}/knowledge-bases/{_knowledge_base_value().id}",
            headers={
                "idempotency-key": "00000000-0000-0000-0000-000000000099"
            },
            json_body={"name": "Changed"},
        )
        self.assertEqual(conflict.status, 409)
        self.assertEqual(conflict.json()["code"], "IDEMPOTENCY_KEY_REUSED")
        self.assertNotIn("internal hash", conflict.body.decode())

    async def test_knowledge_base_delete_requires_idempotency_and_returns_tombstone(
        self,
    ) -> None:
        value = self.dependencies.knowledge_base_service.value
        missing_key = await request(
            self.app,
            "DELETE",
            f"{API_PREFIX}/knowledge-bases/{value.id}",
        )
        self.assertEqual(missing_key.status, 422)
        deleted = await request(
            self.app,
            "DELETE",
            f"{API_PREFIX}/knowledge-bases/{value.id}",
            headers={"idempotency-key": str(uuid4())},
        )
        self.assertEqual(deleted.status, 200)
        self.assertEqual(deleted.json()["id"], str(value.id))
        self.assertEqual(deleted.json()["name"], value.name)
        self.assertIsNotNone(deleted.json()["deleted_at"])

    async def test_document_upload_version_read_and_delete_are_published(self) -> None:
        value = _document_value()
        loaded = await request(
            self.app, "GET", f"{API_PREFIX}/documents/{value.id}"
        )
        self.assertEqual(loaded.status, 200)
        self.assertNotIn("storage_uri", loaded.json()["current_version"])
        self.assertEqual(
            loaded.json()["index"],
            {
                "indexed_document_version_id": (
                    "01900000-0000-7000-8000-000000000026"
                ),
                "index_revision_id": "01900000-0000-7000-8000-000000000012",
                "build_status": "ready",
                "serving_status": "serving",
                "unit_count": 7,
                "asset_count": 3,
                "representation_count": 11,
                "composite_chunk_count": 4,
                "visual_unit_count": 3,
                "relation_count": 5,
                "text_representation_count": 4,
                "native_image_representation_count": 2,
                "table_representation_count": 2,
            },
        )

        chunks = await request(
            self.app, "GET", f"{API_PREFIX}/documents/{value.id}/chunks"
        )
        self.assertEqual(chunks.status, 200)
        self.assertEqual(chunks.json()["total_chunks"], 1)
        self.assertEqual(chunks.json()["items"][0]["ordinal"], 0)
        self.assertIsNone(chunks.json()["items"][0]["excluded_at"])
        self.assertNotIn("storage_uri", chunks.body.decode())

        excluded = await request(
            self.app,
            "DELETE",
            (
                f"{API_PREFIX}/documents/{value.id}/chunks/"
                "01900000-0000-7000-8000-000000000031"
            ),
        )
        self.assertEqual(excluded.status, 200)
        self.assertEqual(
            excluded.json()["chunk_id"],
            "01900000-0000-7000-8000-000000000031",
        )
        self.assertIsNotNone(excluded.json()["excluded_at"])

        deleted = await request(
            self.app,
            "DELETE",
            f"{API_PREFIX}/documents/{value.id}",
            headers={"idempotency-key": str(uuid4())},
        )
        self.assertEqual(deleted.status, 200)
        self.assertEqual(deleted.json()["source_change_seq"], 2)

        uploaded = await request(
            self.app,
            "POST",
            f"{API_PREFIX}/knowledge-bases/{value.kb_id}/documents",
            headers={
                "idempotency-key": str(uuid4()),
                "content-type": "text/markdown; charset=utf-8",
                "x-document-metadata": _encode_upload_metadata(
                    "项目说明（终版）.md",
                    "项目说明",
                ),
            },
            raw_body=b"# Guide\ncontent",
        )
        self.assertEqual(uploaded.status, 202)
        self.assertEqual(uploaded.json()["job_status"], "queued")
        self.assertEqual(uploaded.json()["document"]["display_name"], "项目说明")
        self.assertEqual(
            uploaded.json()["document"]["current_version"]["original_filename"],
            "项目说明（终版）.md",
        )

        versioned = await request(
            self.app,
            "POST",
            f"{API_PREFIX}/documents/{value.id}/versions",
            headers={
                "idempotency-key": str(uuid4()),
                "content-type": "text/plain",
                "x-document-metadata": _encode_upload_metadata("guide.txt"),
            },
            raw_body=b"replacement",
        )
        self.assertEqual(versioned.status, 202)
        self.assertEqual(len(self.dependencies.source_file_service.calls), 2)

    async def test_upload_metadata_rejects_invalid_or_ambiguous_values_before_handoff(
        self,
    ) -> None:
        value = _document_value()
        missing = await request(
            self.app,
            "POST",
            f"{API_PREFIX}/knowledge-bases/{value.kb_id}/documents",
            headers={
                "idempotency-key": str(uuid4()),
                "content-type": "text/markdown",
                "x-document-filename": "retired.md",
            },
            raw_body=b"safe",
        )
        self.assertEqual(missing.status, 422)
        self.assertEqual(missing.json()["code"], "REQUEST_VALIDATION_FAILED")
        self.assertEqual(missing.json()["errors"][0]["error_type"], "missing")

        invalid_utf8 = base64.urlsafe_b64encode(b"\xff").rstrip(b"=").decode()
        invalid_values = (
            ("%%%", {}, "upload_metadata_encoding"),
            (invalid_utf8, {}, "upload_metadata_utf8"),
            (
                _encode_upload_metadata("guide.md", version=2),
                {},
                "upload_metadata_version",
            ),
            (
                _encode_upload_metadata("bad\nname.md"),
                {},
                None,
            ),
            (
                _encode_upload_metadata("guide.md", "x" * 256),
                {},
                None,
            ),
        )
        for metadata, extra_headers, error_type in invalid_values:
            with self.subTest(error_type=error_type, metadata=metadata):
                response = await request(
                    self.app,
                    "POST",
                    f"{API_PREFIX}/knowledge-bases/{value.kb_id}/documents",
                    headers={
                        "idempotency-key": str(uuid4()),
                        "content-type": "text/markdown",
                        "x-document-metadata": metadata,
                        **extra_headers,
                    },
                    raw_body=b"safe",
                )
                self.assertEqual(response.status, 422)
                if error_type is not None:
                    self.assertEqual(response.json()["code"], "REQUEST_VALIDATION_FAILED")
                    self.assertEqual(
                        response.json()["errors"][0]["error_type"],
                        error_type,
                    )
        self.assertEqual(self.dependencies.source_file_service.calls, [])

    async def test_upload_admission_failures_are_problem_details_and_do_not_handoff(self) -> None:
        value = _document_value()
        cases = (
            ("guide.rtf", "application/rtf", b"binary", 415, "PARSER_NOT_CONFIGURED"),
            ("guide.md", "text/plain", b"text", 415, "FILE_MEDIA_TYPE_MISMATCH"),
            ("guide.txt", "text/plain", b"\xff", 422, "FILE_INVALID_UTF8"),
            ("guide.txt", "text/plain", b"x" * 33, 413, "FILE_TOO_LARGE"),
            (
                "guide.csv",
                "text/csv",
                b"a,b\n1,2\n",
                413,
                "FILE_STRUCTURE_LIMIT_EXCEEDED",
            ),
        )
        for filename, media_type, body, status, code in cases:
            with self.subTest(code=code):
                response = await request(
                    self.app,
                    "POST",
                    f"{API_PREFIX}/knowledge-bases/{value.kb_id}/documents",
                    headers={
                        "idempotency-key": str(uuid4()),
                        "content-type": media_type,
                        "x-document-metadata": _encode_upload_metadata(filename),
                    },
                    raw_body=body,
                )
                self.assertEqual(response.status, status)
                self.assertEqual(response.json()["code"], code)
        self.assertEqual(self.dependencies.source_file_service.calls, [])

    async def test_markdown_v2_upload_requests_admission_snapshot(self) -> None:
        created = await request(
            self.app,
            "POST",
            f"{API_PREFIX}/knowledge-bases",
            headers={"idempotency-key": str(uuid4())},
            json_body={
                "name": "Markdown media",
                "parsing": {"preset": "multimodal_local_v2"},
            },
        )
        self.assertEqual(created.status, 201)
        self.assertEqual(
            created.json()["parsing"],
            {
                "preset": "multimodal_local_v2",
                "profile": "docling_multimodal_local_v4",
            },
        )

        uploaded = await request(
            self.app,
            "POST",
            (
                f"{API_PREFIX}/knowledge-bases/"
                f"{self.dependencies.knowledge_base_service.value.id}/documents"
            ),
            headers={
                "idempotency-key": str(uuid4()),
                "content-type": "text/markdown",
                "x-document-metadata": _encode_upload_metadata("evidence.md"),
            },
            raw_body=b"![x](https://x.co/a)",
        )

        self.assertEqual(uploaded.status, 202)
        self.assertTrue(
            self.dependencies.source_file_service.calls[-1][
                "normalize_markdown_media"
            ]
        )

    async def test_indexing_status_and_explicit_retry_are_published(self) -> None:
        service = self.dependencies.indexing_job_service
        listed = await request(
            self.app,
            "GET",
            f"{API_PREFIX}/knowledge-bases/{service.value.kb_id}/indexing-jobs",
        )
        self.assertEqual(listed.status, 200)
        self.assertEqual(listed.json()["items"][0]["job_id"], str(service.value.job_id))

        status = await request(
            self.app,
            "GET",
            f"{API_PREFIX}/indexing-jobs/{service.value.job_id}",
        )
        self.assertEqual(status.status, 200)
        self.assertEqual(status.json()["status"], "failed")
        self.assertEqual(
            status.json()["error"]["code"],
            "EMBEDDING_PROVIDER_UNAVAILABLE",
        )
        self.assertTrue(status.json()["can_retry"])

        missing_key = await request(
            self.app,
            "POST",
            f"{API_PREFIX}/indexing-jobs/{service.value.job_id}/retry",
        )
        self.assertEqual(missing_key.status, 422)
        self.assertEqual(missing_key.json()["code"], "INVALID_IDEMPOTENCY_KEY")

        retried = await request(
            self.app,
            "POST",
            f"{API_PREFIX}/indexing-jobs/{service.value.job_id}/retry",
            headers={"idempotency-key": str(uuid4())},
        )
        self.assertEqual(retried.status, 202)
        self.assertEqual(
            retried.headers["location"],
            f"/api/v1/indexing-jobs/{service.value.job_id}",
        )
        self.assertEqual(retried.json()["status"], "queued")
        self.assertEqual(retried.json()["attempt"], 0)
        self.assertIsNone(retried.json()["error"])

    async def test_indexing_status_hides_cross_workspace_or_unknown_ids(self) -> None:
        response = await request(
            self.app,
            "GET",
            f"{API_PREFIX}/indexing-jobs/{uuid4()}",
        )
        self.assertEqual(response.status, 404)
        self.assertNotIn("internal indexing", response.body.decode())

    async def test_chat_session_run_history_and_status_contracts(self) -> None:
        chat = self.dependencies.chat_service
        session = await request(
            self.app,
            "POST",
            f"{API_PREFIX}/chat/sessions",
            json_body={
                "knowledge_base_id": str(_knowledge_base_value().id),
                "title": "  Incident response  ",
            },
        )
        self.assertEqual(session.status, 201)
        self.assertEqual(session.json()["title"], "Incident response")
        self.assertEqual(
            session.headers["location"],
            f"/api/v1/chat/sessions/{chat.session.id}/messages",
        )

        missing_key = await request(
            self.app,
            "POST",
            f"{API_PREFIX}/chat/runs",
            json_body=_chat_run_request(chat.session.id),
        )
        self.assertEqual(missing_key.status, 422)
        self.assertEqual(missing_key.json()["code"], "INVALID_IDEMPOTENCY_KEY")

        disabled_hybrid = await request(
            self.app,
            "POST",
            f"{API_PREFIX}/chat/runs",
            headers={"idempotency-key": str(uuid4())},
            json_body={
                **_chat_run_request(chat.session.id),
                "retrieval": {
                    "mode": "hybrid",
                    "top_k": 10,
                    "rerank_mode": "classic",
                },
            },
        )
        self.assertEqual(disabled_hybrid.status, 409)
        self.assertEqual(disabled_hybrid.json()["code"], "CAPABILITY_NOT_ENABLED")
        self.assertFalse(disabled_hybrid.json()["retryable"])
        self.assertNotIn("hybrid", disabled_hybrid.body.decode())

        key = uuid4()
        created = await request(
            self.app,
            "POST",
            f"{API_PREFIX}/chat/runs",
            headers={"idempotency-key": str(key)},
            json_body=_chat_run_request(chat.session.id),
        )
        body = created.json()
        self.assertEqual(created.status, 202)
        self.assertEqual(created.headers["location"], body["status_url"])
        self.assertEqual(body["status"], "queued")
        self.assertEqual(body["assistant_status"], "generating")
        self.assertEqual(body["agent"]["version"], "native_tool_calling_agent_v1")
        self.assertIsNone(body["agent"]["trace"])
        self.assertEqual(
            body["query_context"],
            {
                "strategy": "recent_completed_turns_v1",
                "status": "original",
                "history_turn_count": 0,
                "history_token_count": 0,
                "history_truncated": False,
                "standalone_query": "查询 RUN-ORD-14",
                "rewrite_source": "original",
            },
        )
        self.assertIsNone(body["answer"])
        self.assertEqual(
            body["events_url"], f"{body['status_url']}/events"
        )
        self.assertEqual(
            body["final_context_url"], f"{body['status_url']}/final-context"
        )
        self.assertEqual(
            body["effective_answer_policy"],
            {
                "grounding_policy": "evidence_only",
                "answer_style": "summary",
                "insufficiency_policy": "partial_answer",
                "citation_required": True,
                "citation_granularity": "claim_level",
                "answer_task": "answer",
                "policy_version": "p1",
            },
        )
        self.assertEqual(chat.create_run_calls[-1]["key"], key)
        self.assertEqual(chat.create_run_calls[-1]["message"], "查询 RUN-ORD-14")

        status_response = await request(
            self.app, "GET", f"{API_PREFIX}/chat/runs/{chat.run.id}"
        )
        history = await request(
            self.app,
            "GET",
            f"{API_PREFIX}/chat/sessions/{chat.session.id}/messages",
        )
        listed = await request(self.app, "GET", f"{API_PREFIX}/chat/sessions")
        filtered = await request(
            self.app,
            "GET",
            f"{API_PREFIX}/chat/sessions",
            query=f"knowledge_base_id={_knowledge_base_value().id}",
        )
        self.assertEqual(status_response.status, 200)
        self.assertEqual(status_response.json()["run_id"], str(chat.run.id))
        self.assertEqual(
            [item["role"] for item in history.json()["items"]],
            ["user", "assistant"],
        )
        self.assertEqual(listed.json()["items"][0]["id"], str(chat.session.id))
        self.assertEqual(filtered.json()["items"][0]["id"], str(chat.session.id))

    async def test_chat_final_context_returns_messages_and_authorized_media(self) -> None:
        chat = self.dependencies.chat_service
        chat.run = dataclass_replace(
            chat.run,
            status="completed",
            final_llm_context={
                "version": "final_llm_context_v1",
                "operation": "repair_answer",
                "output_schema": "answer_v1",
                "max_output_tokens": None,
                "messages": [
                    {"role": "system", "content": "System instructions"},
                    {"role": "user", "content": "Final repair input"},
                ],
                "media": [
                    {
                        "message_index": 1,
                        "citation_ids": ["cite_2"],
                        "asset": {
                            "id": "01900000-0000-7000-8000-000000000044",
                            "media_type": "image/png",
                            "checksum_sha256": "a" * 64,
                            "content_url": "/api/v1/index-assets/01900000-0000-7000-8000-000000000044/content",
                            "width": 320,
                            "height": 200,
                        },
                    }
                ],
            },
        )

        response = await request(
            self.app,
            "GET",
            f"{API_PREFIX}/chat/runs/{chat.run.id}/final-context",
        )

        self.assertEqual(response.status, 200)
        self.assertTrue(response.json()["available"])
        self.assertEqual(response.json()["operation"], "repair_answer")
        self.assertEqual(
            response.json()["messages"][1]["content"], "Final repair input"
        )
        self.assertEqual(
            response.json()["media"][0]["asset"]["media_type"], "image/png"
        )

    async def test_chat_rejects_policy_weakening_and_redacts_conflicts(self) -> None:
        chat = self.dependencies.chat_service
        unsupported_policies = (
            {
                "answer_style": "concise",
                "grounding_policy": "model_knowledge_allowed",
            },
            {"answer_style": "detailed"},
            {"insufficiency_policy": "ask_for_clarification"},
            {"citation_required": False},
            {"citation_granularity": "paragraph_level"},
            {"answer_task": "report"},
            {"policy_version": "client-version"},
        )
        for answer_policy in unsupported_policies:
            with self.subTest(answer_policy=answer_policy):
                forbidden = await request(
                    self.app,
                    "POST",
                    f"{API_PREFIX}/chat/runs",
                    headers={"idempotency-key": str(uuid4())},
                    json_body={
                        **_chat_run_request(chat.session.id),
                        "answer_policy": answer_policy,
                    },
                )
                self.assertEqual(forbidden.status, 422)
                self.assertEqual(
                    forbidden.json()["code"], "ANSWER_POLICY_NOT_SUPPORTED"
                )
                self.assertNotIn("client-version", forbidden.body.decode())
        self.assertEqual(chat.create_run_calls, [])

        conflict = await request(
            self.app,
            "POST",
            f"{API_PREFIX}/chat/runs",
            headers={
                "idempotency-key": "00000000-0000-0000-0000-000000000099"
            },
            json_body=_chat_run_request(chat.session.id),
        )
        self.assertEqual(conflict.status, 409)
        self.assertEqual(conflict.json()["code"], "IDEMPOTENCY_KEY_REUSED")
        self.assertNotIn("internal chat hash", conflict.body.decode())

        busy = await request(
            self.app,
            "POST",
            f"{API_PREFIX}/chat/runs",
            headers={
                "idempotency-key": "00000000-0000-0000-0000-000000000098"
            },
            json_body=_chat_run_request(chat.session.id),
        )
        self.assertEqual(busy.status, 409)
        self.assertEqual(busy.json()["code"], "CHAT_SESSION_BUSY")
        self.assertTrue(busy.json()["retryable"])
        self.assertNotIn("internal active run", busy.body.decode())

    async def test_terminal_sse_uses_committed_answer_and_native_headers(self) -> None:
        chat = self.dependencies.chat_service
        citation = ChatCitation(
            ordinal=0,
            index_chunk_id=UUID("01900000-0000-7000-8000-000000000041"),
            document_id=UUID("01900000-0000-7000-8000-000000000042"),
            document_version_id=UUID("01900000-0000-7000-8000-000000000043"),
            document_display_name="Committed source",
            document_original_filename="committed.png",
            quoted_text="Committed evidence",
            source_location={"line": 4},
            score=0.9,
            modality="image",
            asset_snapshot={
                "id": "01900000-0000-7000-8000-000000000044",
                "media_type": "image/png",
                "checksum_sha256": "a" * 64,
                "content_url": (
                    "/api/v1/index-assets/"
                    "01900000-0000-7000-8000-000000000044/content"
                ),
                "width": 320,
                "height": 200,
                "visual_unit_id": "01900000-0000-7000-8000-000000000041",
                "parent_citation_id": "cite_1",
                "relation_type": "explicit_figure_reference",
                "selection_reason": "selected_explicit_reference",
            },
            matched_representations=("native_image",),
        )
        chat.run = dataclass_replace(
            chat.run,
            status="completed",
            assistant_status="completed",
            assistant_content="Committed answer [1]",
            citations=(citation,),
            completed_at=datetime.now(UTC),
        )

        streamed = await request(
            self.app,
            "GET",
            f"{API_PREFIX}/chat/runs/{chat.run.id}/events",
            disconnect_immediately=False,
        )
        status_response = await request(
            self.app, "GET", f"{API_PREFIX}/chat/runs/{chat.run.id}"
        )

        self.assertEqual(streamed.status, 200)
        self.assertTrue(
            streamed.headers["content-type"].startswith("text/event-stream")
        )
        self.assertEqual(streamed.headers["cache-control"], "no-cache")
        self.assertEqual(streamed.headers["x-accel-buffering"], "no")
        self.assertIn(b"event: answer.completed", streamed.body)
        self.assertNotIn(b"id:", streamed.body)
        event = _sse_data(streamed.body)
        self.assertEqual(event["answer"], "Committed answer [1]")
        self.assertEqual(event["citations"][0]["quoted_text"], "Committed evidence")
        self.assertEqual(event["citations"][0]["modality"], "image")
        self.assertEqual(
            event["citations"][0]["asset"]["parent_citation_id"], "cite_1"
        )
        self.assertEqual(
            event["citations"][0]["asset"]["relation_type"],
            "explicit_figure_reference",
        )
        self.assertEqual(event["effective_answer_policy"], chat.run.effective_policy)
        self.assertEqual(status_response.json()["citations"], event["citations"])
        self.assertEqual(
            await self.dependencies.chat_sse_connection_limiter.active(
                "development-principal", chat.run.id
            ),
            0,
        )

    async def test_sse_delivers_progress_preview_and_reset_before_terminal(
        self,
    ) -> None:
        chat = self.dependencies.chat_service
        subscription = _PreviewSubscription(
            ChatProgressSnapshot(
                chat.run.id,
                1,
                1,
                ChatProgressUpdate(
                    ChatProgressStage.RETRIEVE_EVIDENCE,
                    ChatProgressActivity.SEARCH_KNOWLEDGE_BASE,
                    completed_stages=(
                        ChatProgressStage.UNDERSTAND_QUERY,
                    ),
                    facts=ChatProgressFacts(
                        objective="查找负责人",
                        queries=("负责人",),
                    ),
                ),
            ),
            ChatPreviewDelta(chat.run.id, 1, 1, "未验证片段"),
            ChatPreviewReset(
                chat.run.id,
                1,
                2,
                ChatPreviewResetReason.VALIDATION_REPAIR,
            ),
        )
        broker = _PreviewBroker(subscription)
        self.dependencies.chat_preview_broker = broker

        streamed = await request(
            self.app,
            "GET",
            f"{API_PREFIX}/chat/runs/{chat.run.id}/events",
            disconnect_immediately=False,
        )

        self.assertEqual(streamed.status, 200)
        self.assertIn(b"event: agent.progress", streamed.body)
        self.assertIn(b"event: answer.preview.delta", streamed.body)
        self.assertIn(b"event: answer.preview.reset", streamed.body)
        self.assertNotIn(b"event: answer.completed", streamed.body)
        data = [
            json.loads(line.removeprefix(b"data: "))
            for line in streamed.body.splitlines()
            if line.startswith(b"data: ")
        ]
        self.assertEqual(
            data,
            [
                {
                    "run_id": str(chat.run.id),
                    "attempt": 1,
                    "seq": 1,
                    "active_stage": "retrieve_evidence",
                    "activity": "search_knowledge_base",
                    "completed_stages": ["understand_query"],
                    "status": "active",
                    "facts": {
                        "objective": "查找负责人",
                        "queries": ["负责人"],
                        "evidence_count": None,
                        "new_evidence_count": None,
                        "retrieval_calls": None,
                        "covered_aspects": [],
                        "missing_aspects": [],
                        "conflict_count": None,
                    },
                },
                {
                    "run_id": str(chat.run.id),
                    "attempt": 1,
                    "seq": 1,
                    "delta": "未验证片段",
                },
                {
                    "run_id": str(chat.run.id),
                    "attempt": 1,
                    "seq": 2,
                    "reason": "validation_repair",
                },
            ],
        )
        self.assertEqual(broker.run_ids, [chat.run.id])
        self.assertTrue(subscription.closed)
        self.assertEqual(
            await self.dependencies.chat_sse_connection_limiter.active(
                "development-principal", chat.run.id
            ),
            0,
        )

    async def test_failed_sse_rejects_replay_and_enforces_connection_limit(self) -> None:
        chat = self.dependencies.chat_service
        chat.run = dataclass_replace(
            chat.run,
            status="failed",
            assistant_status="failed",
            error_code="CHAT_PROVIDER_UNAVAILABLE",
            error_detail={"http_status": 503},
            error_retryable=True,
            completed_at=datetime.now(UTC),
        )
        path = f"{API_PREFIX}/chat/runs/{chat.run.id}/events"

        failed = await request(
            self.app, "GET", path, disconnect_immediately=False
        )
        self.assertIn(b"event: run.failed", failed.body)
        self.assertEqual(
            _sse_data(failed.body)["error"],
            {
                "code": "CHAT_PROVIDER_UNAVAILABLE",
                "detail": {"http_status": 503},
                "retryable": True,
            },
        )

        replay = await request(
            self.app,
            "GET",
            path,
            headers={"last-event-id": "1"},
        )
        self.assertEqual(replay.status, 400)
        self.assertEqual(replay.json()["code"], "REQUEST_VALIDATION_FAILED")

        limiter = self.dependencies.chat_sse_connection_limiter
        self.assertTrue(await limiter.acquire("development-principal", chat.run.id))
        self.assertTrue(await limiter.acquire("development-principal", chat.run.id))
        limited = await request(self.app, "GET", path)
        self.assertEqual(limited.status, 429)
        self.assertEqual(
            limited.json()["code"], "CHAT_SSE_CONNECTION_LIMIT_EXCEEDED"
        )
        await limiter.release("development-principal", chat.run.id)
        await limiter.release("development-principal", chat.run.id)

        missing = await request(
            self.app,
            "GET",
            f"{API_PREFIX}/chat/runs/{uuid4()}/events",
        )
        self.assertEqual(missing.status, 404)

    async def test_sse_disconnect_releases_only_the_subscription(self) -> None:
        chat = self.dependencies.chat_service
        path = f"{API_PREFIX}/chat/runs/{chat.run.id}/events"

        disconnected = await request(self.app, "GET", path)

        self.assertEqual(disconnected.status, 200)
        self.assertEqual(disconnected.body, b"")
        self.assertEqual(chat.run.status, "queued")
        self.assertEqual(
            await self.dependencies.chat_sse_connection_limiter.active(
                "development-principal", chat.run.id
            ),
            0,
        )
        recovered = await request(
            self.app, "GET", f"{API_PREFIX}/chat/runs/{chat.run.id}"
        )
        self.assertEqual(recovered.json()["status"], "queued")


def dataclass_replace(value, **changes):
    from dataclasses import replace

    return replace(value, **changes)


def _sse_data(body: bytes) -> dict[str, object]:
    line = next(item for item in body.splitlines() if item.startswith(b"data: "))
    return json.loads(line.removeprefix(b"data: "))


def _knowledge_base_value() -> KnowledgeBase:
    now = datetime(2026, 7, 14, tzinfo=UTC)
    return KnowledgeBase(
        id=UUID("01900000-0000-7000-8000-000000000010"),
        workspace_id=WORKSPACE,
        name="knowledge-base",
        source_change_seq=0,
        active_index_revision_id=UUID("01900000-0000-7000-8000-000000000012"),
        embedding_space_id=UUID("01900000-0000-7000-8000-000000000013"),
        chunking_config={
            "profile": "unstructured_by_title_token_v2",
            "strategy": "by_title",
            "max_tokens": 800,
            "new_after_n_tokens": 600,
            "tokenizer": "cl100k_base",
            "tokenizer_library": "tiktoken",
            "tokenizer_version": "0.13.0",
            "overlap": 100,
            "overlap_unit": "tokens",
            "overlap_all": False,
            "combine_text_under_n_chars": 300,
            "combine_text_under_n_chars_unit": "characters",
            "multipage_sections": False,
            "include_orig_elements": True,
            "metadata_policy": "bounded_v2",
        },
        retrieval_defaults={"strategy": "exact_vector", "top_k": 10},
        answer_policy_defaults={
            "answer_style": "concise",
            "insufficiency_policy": "partial_answer",
        },
        provisioned_at=now,
        created_at=now,
        updated_at=now,
        parser_config=DOCLING_TEXT_PARSER_CONFIG,
        embedding=KnowledgeBaseEmbeddingSummary(
            strategy="text_only",
            text=EmbeddingRoleSummary(
                embedding_space_id=UUID(
                    "01900000-0000-7000-8000-000000000013"
                ),
                profile_revision_id=None,
                dimension=1024,
            ),
        ),
    )


def _document_value() -> Document:
    now = datetime(2026, 7, 14, tzinfo=UTC)
    document_id = UUID("01900000-0000-7000-8000-000000000020")
    return Document(
        id=document_id,
        workspace_id=WORKSPACE,
        kb_id=_knowledge_base_value().id,
        display_name="guide.md",
        current_version=DocumentVersion(
            id=UUID("01900000-0000-7000-8000-000000000021"),
            document_id=document_id,
            version_number=1,
            source_status="available",
            checksum_sha256="0" * 64,
            storage_uri="file:///must-not-be-public",
            original_filename="guide.md",
            media_type="text/markdown",
            size_bytes=42,
            created_at=now,
        ),
        deleted_at=None,
        created_at=now,
        updated_at=now,
    )


def _chat_session_value() -> ChatSession:
    now = datetime(2026, 7, 15, tzinfo=UTC)
    return ChatSession(
        id=UUID("01900000-0000-7000-8000-000000000030"),
        workspace_id=WORKSPACE,
        kb_id=_knowledge_base_value().id,
        principal_id="development-principal",
        title=None,
        created_at=now,
        updated_at=now,
    )


def _chat_run_value(session: ChatSession) -> ChatRun:
    now = datetime(2026, 7, 15, tzinfo=UTC)
    return ChatRun(
        id=UUID("01900000-0000-7000-8000-000000000031"),
        workspace_id=WORKSPACE,
        kb_id=session.kb_id,
        session_id=session.id,
        user_message_id=UUID("01900000-0000-7000-8000-000000000032"),
        assistant_message_id=UUID("01900000-0000-7000-8000-000000000033"),
        index_revision_id=_knowledge_base_value().active_index_revision_id,
        status="queued",
        principal_id="development-principal",
        client_id="development-web",
        endpoint="POST /api/v1/chat/runs",
        idempotency_key=UUID("01900000-0000-7000-8000-000000000034"),
        request_hash="sha256:" + "1" * 64,
        requested_policy={},
        effective_policy={
            "grounding_policy": "evidence_only",
            "answer_style": "concise",
            "insufficiency_policy": "partial_answer",
            "citation_required": True,
            "citation_granularity": "claim_level",
            "answer_task": "answer",
            "policy_version": "p1",
        },
        retrieval_strategy=exact_profile(top_k=10, rerank=False).as_dict(),
        model_configuration={"configuration_fingerprint": "sha256:safe"},
        assistant_status="generating",
        assistant_content="",
        citations=(),
        attempt=0,
        error_code=None,
        error_detail=None,
        error_retryable=None,
        usage=None,
        timing=None,
        created_at=now,
        updated_at=now,
        completed_at=None,
        conversation_context=serialize_conversation_context(
            empty_conversation_context()
        ),
    )


def _chat_messages(run: ChatRun) -> tuple[ChatMessage, ChatMessage]:
    return (
        ChatMessage(
            id=run.user_message_id,
            session_id=run.session_id,
            chat_run_id=None,
            role="user",
            assistant_status=None,
            client_request_id=run.idempotency_key,
            content="查询 RUN-ORD-14",
            created_at=run.created_at,
        ),
        ChatMessage(
            id=run.assistant_message_id,
            session_id=run.session_id,
            chat_run_id=run.id,
            role="assistant",
            assistant_status="generating",
            client_request_id=None,
            content="",
            created_at=run.created_at,
        ),
    )


def _chat_run_request(session_id: UUID) -> dict[str, object]:
    return {
        "session_id": str(session_id),
        "knowledge_base_id": str(_knowledge_base_value().id),
        "message": "  查询 RUN-ORD-14  ",
        "answer_policy": {
            "answer_style": "summary",
            "insufficiency_policy": "partial_answer",
        },
        "retrieval": {"mode": "vector", "top_k": 8},
    }


if __name__ == "__main__":
    unittest.main()
