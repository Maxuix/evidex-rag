from __future__ import annotations

import json
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter
from pydantic import BaseModel, Field, ValidationError

from apps.api.app import API_PREFIX, create_app
from apps.api.errors import ApiProblem
from apps.api.idempotency import RequiredIdempotencyKey
from apps.api.pagination import decode_cursor, encode_cursor
from rag_kb.domain import IdempotencyScope, canonical_request_hash
from rag_kb.schemas import CursorPayload, ErrorCode, PaginationQuery


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


@router.post("/validate")
async def validate(payload: ExampleInput) -> dict[str, int]:
    return {"count": payload.count}


@router.post("/idempotency")
async def idempotency(key: RequiredIdempotencyKey) -> dict[str, str]:
    return {"key": str(key)}


@router.get("/cursor")
async def cursor(value: str) -> CursorPayload:
    return decode_cursor(value)


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
        self.app = create_app(routers=(router,))

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

    async def test_lifespan_owns_dependency_shutdown(self) -> None:
        class StubDependencies:
            closed = False

            async def close(self) -> None:
                self.closed = True

        dependencies = StubDependencies()
        app = create_app(dependencies=dependencies)  # type: ignore[arg-type]
        async with app.router.lifespan_context(app):
            self.assertIs(app.state.dependencies, dependencies)
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

    def test_production_openapi_has_no_placeholder_business_routes(self) -> None:
        production = create_app().openapi()
        snapshot = json.loads(
            (PROJECT_ROOT / "tests/contract/snapshots/openapi-v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(production, snapshot)
        self.assertEqual(production["paths"], {})


if __name__ == "__main__":
    unittest.main()
