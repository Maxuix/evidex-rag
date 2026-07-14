from __future__ import annotations

import json
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
from rag_kb.domain import (
    Document,
    DocumentMutationResult,
    DocumentVersion,
    IdempotencyKeyReusedError,
    IdempotencyScope,
    KnowledgeBase,
    Page,
    ResourceNotFoundError,
    canonical_request_hash,
)
from rag_kb.schemas import CursorPayload, ErrorCode, PaginationQuery


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = UUID("01900000-0000-7000-8000-000000000001")
ALLOWED_ORIGIN = "http://127.0.0.1:3000"


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


@router.get("/denied")
async def denied() -> None:
    raise AccessDeniedError("policy internals must not leak")


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

    async def start(self) -> SimpleNamespace:
        self.started = True
        return await self.check_readiness()

    async def check_readiness(self) -> SimpleNamespace:
        return SimpleNamespace(
            database="ready",
            queue="ready",
            queue_backend="postgresql",
        )

    async def close(self) -> None:
        self.closed = True


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
    suppress_application_error: bool = False,
) -> AsgiResponse:
    body = b"" if json_body is None else json.dumps(json_body).encode("utf-8")
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

    async def receive() -> dict[str, object]:
        nonlocal sent_request
        if not sent_request:
            sent_request = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

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
                "components": {
                    "database": "ready",
                    "queue": "ready",
                    "queue_backend": "postgresql",
                },
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
                "components": {
                    "database": "unavailable",
                    "queue": "unavailable",
                },
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
        response = await request(
            self.app,
            "GET",
            f"{API_PREFIX}/unexpected",
            suppress_application_error=True,
        )
        body = response.json()
        self.assertEqual(response.status, 500)
        self.assertEqual(body["code"], "INTERNAL_SERVER_ERROR")
        self.assertNotIn("internal-secret", response.body.decode("utf-8"))

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
                "access-control-request-headers": "content-type",
            },
        )
        self.assertEqual(allowed.status, 200)
        self.assertEqual(allowed.headers["access-control-allow-origin"], ALLOWED_ORIGIN)
        self.assertNotIn("access-control-allow-credentials", allowed.headers)

        disallowed = await request(
            self.app,
            "OPTIONS",
            f"{API_PREFIX}/identity",
            headers={
                "origin": ALLOWED_ORIGIN,
                "access-control-request-method": "PUT",
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
                "/api/v1/knowledge-bases",
                "/api/v1/knowledge-bases/{kb_id}",
                "/api/v1/knowledge-bases/{kb_id}/documents",
            },
        )
        self.assertNotIn(
            "post",
            production["paths"]["/api/v1/knowledge-bases/{kb_id}/documents"],
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


class _FakeKnowledgeBaseService:
    def __init__(self) -> None:
        self.value = _knowledge_base_value()

    async def create(self, context, key, *, name, retrieval_defaults):
        del context, key
        self.value = dataclass_replace(
            self.value, name=name, retrieval_defaults=retrieval_defaults
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
        self, context, key, kb_id, *, name, retrieval_defaults
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

    async def list(self, context, *, kb_id, limit, sort, after):
        del context, limit, sort, after
        return Page(items=(self.value,)) if kb_id == self.value.kb_id else Page(items=())

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


class ContentApiContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.dependencies = StubApiDependencies()
        self.dependencies.knowledge_base_service = _FakeKnowledgeBaseService()
        self.dependencies.document_service = _FakeDocumentService()
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

    async def test_document_reads_and_delete_are_published_but_upload_is_not(self) -> None:
        value = _document_value()
        loaded = await request(
            self.app, "GET", f"{API_PREFIX}/documents/{value.id}"
        )
        self.assertEqual(loaded.status, 200)
        self.assertNotIn("storage_uri", loaded.json()["current_version"])

        deleted = await request(
            self.app,
            "DELETE",
            f"{API_PREFIX}/documents/{value.id}",
            headers={"idempotency-key": str(uuid4())},
        )
        self.assertEqual(deleted.status, 200)
        self.assertEqual(deleted.json()["source_change_seq"], 2)

        unavailable_upload = await request(
            self.app,
            "POST",
            f"{API_PREFIX}/knowledge-bases/{value.kb_id}/documents",
            headers={"idempotency-key": str(uuid4())},
            json_body={},
        )
        self.assertEqual(unavailable_upload.status, 405)


def dataclass_replace(value, **changes):
    from dataclasses import replace

    return replace(value, **changes)


def _knowledge_base_value() -> KnowledgeBase:
    now = datetime(2026, 7, 14, tzinfo=UTC)
    return KnowledgeBase(
        id=UUID("01900000-0000-7000-8000-000000000010"),
        workspace_id=WORKSPACE,
        name="knowledge-base",
        source_change_seq=0,
        active_index_revision_id=UUID("01900000-0000-7000-8000-000000000012"),
        embedding_space_id=UUID("01900000-0000-7000-8000-000000000013"),
        retrieval_defaults={"strategy": "exact_vector", "top_k": 10},
        provisioned_at=now,
        created_at=now,
        updated_at=now,
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


if __name__ == "__main__":
    unittest.main()
