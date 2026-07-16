#!/usr/bin/env python3
"""Deterministic OpenAI-compatible provider for isolated Stage 06 E2E tests."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


DIMENSION = 1024
MODEL_CHAT = "deepseek-v4-flash"
MODEL_EMBEDDING = "text-embedding-v4"
INDEX_FAILURE_MARKER = "E2E_INDEX_FAIL"
CHAT_FAILURE_MARKER = "E2E_CHAT_FAIL"
CHAT_DELAY_MARKER = "E2E_CHAT_DELAY"


def _unit_vector(index: int) -> list[float]:
    vector = [0.0] * DIMENSION
    vector[index] = 1.0
    return vector


def deterministic_embedding(value: str) -> list[float]:
    """Map fixture markers to stable orthogonal vectors without content logging."""

    upper = value.upper()
    if "E2E_ALPHA" in upper:
        return _unit_vector(0)
    if "E2E_BETA" in upper:
        return _unit_vector(1)
    if INDEX_FAILURE_MARKER in upper:
        return _unit_vector(2)
    # Keep all unmarked inputs valid and normalized while deterministic.
    bucket = 3 + (sum(value.encode("utf-8")) % (DIMENSION - 3))
    return _unit_vector(bucket)


@dataclass(frozen=True, slots=True)
class StubResponse:
    status: int
    body: dict[str, Any]
    delay_seconds: float = 0.0


class ProviderScenario:
    """Thread-safe finite behavior used by both the HTTP server and unit tests."""

    def __init__(self, *, indexing_failures: int = 3, chat_delay_seconds: float = 2.5):
        if indexing_failures < 0 or chat_delay_seconds < 0:
            raise ValueError("stub limits must be non-negative")
        self._indexing_failures = indexing_failures
        self._chat_delay_seconds = chat_delay_seconds
        self._failed_embedding_calls = 0
        self._lock = threading.Lock()

    def embeddings(self, payload: dict[str, Any]) -> StubResponse:
        values = payload.get("input")
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            return StubResponse(400, {"error": {"code": "invalid_input"}})
        marked = any(INDEX_FAILURE_MARKER in item.upper() for item in values)
        if marked:
            with self._lock:
                if self._failed_embedding_calls < self._indexing_failures:
                    self._failed_embedding_calls += 1
                    return StubResponse(503, {"error": {"code": "e2e_unavailable"}})
        return StubResponse(
            200,
            {
                "model": MODEL_EMBEDDING,
                "data": [
                    {"index": index, "embedding": deterministic_embedding(value)}
                    for index, value in enumerate(values)
                ],
                "usage": {"prompt_tokens": len(values), "total_tokens": len(values)},
            },
        )

    def chat(self, payload: dict[str, Any]) -> StubResponse:
        messages = payload.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            return StubResponse(400, {"error": {"code": "invalid_messages"}})
        try:
            system = messages[0]["content"]
            user = json.loads(messages[-1]["content"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return StubResponse(400, {"error": {"code": "invalid_messages"}})
        if not isinstance(system, str) or not isinstance(user, dict):
            return StubResponse(400, {"error": {"code": "invalid_messages"}})
        question = user.get("question", "")
        if not isinstance(question, str):
            return StubResponse(400, {"error": {"code": "invalid_question"}})
        if CHAT_FAILURE_MARKER in question.upper():
            return StubResponse(503, {"error": {"code": "e2e_unavailable"}})

        evidence = user.get("evidence", [])
        citations = [
            item.get("citation_id")
            for item in evidence
            if isinstance(item, dict) and isinstance(item.get("citation_id"), str)
        ]
        if "assess whether supplied evidence" in system:
            content = {
                "coverage": "sufficient" if citations else "none",
                "usable_citation_ids": citations[:1],
                "supported_aspects": ["e2e_fact"] if citations else [],
                "missing_aspects": [],
            }
        elif "internal answer draft" in system or "repair an untrusted" in system:
            required_outcome = user.get("required_outcome", "answered")
            content = {
                "outcome": required_outcome,
                "claims": (
                    [{"text": "The handbook confirms the E2E fact.", "citation_ids": citations[:1]}]
                    if citations
                    else []
                ),
                "missing_aspects": user.get(
                    "required_missing_aspects", user.get("missing_aspects", [])
                ),
            }
        else:
            return StubResponse(400, {"error": {"code": "unknown_operation"}})

        delay = self._chat_delay_seconds if CHAT_DELAY_MARKER in question.upper() else 0.0
        return StubResponse(
            200,
            {
                "model": MODEL_CHAT,
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(content, separators=(",", ":")),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
            delay_seconds=delay,
        )


class ProviderHandler(BaseHTTPRequestHandler):
    scenario = ProviderScenario()

    def do_GET(self) -> None:  # noqa: N802 - HTTP handler contract
        if self.path == "/health":
            self._send(StubResponse(200, {"status": "ready"}))
        else:
            self._send(StubResponse(404, {"error": {"code": "not_found"}}))

    def do_POST(self) -> None:  # noqa: N802 - HTTP handler contract
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_048_576:
                raise ValueError
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError
        except (ValueError, json.JSONDecodeError):
            self._send(StubResponse(400, {"error": {"code": "invalid_json"}}))
            return
        if self.path == "/v1/embeddings":
            response = self.scenario.embeddings(payload)
        elif self.path == "/v1/chat/completions":
            response = self.scenario.chat(payload)
        else:
            response = StubResponse(404, {"error": {"code": "not_found"}})
        self._send(response)

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _send(self, response: StubResponse) -> None:
        if response.delay_seconds:
            time.sleep(response.delay_seconds)
        body = json.dumps(response.body, separators=(",", ":")).encode("utf-8")
        self.send_response(response.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("x-request-id", f"e2e-{uuid.uuid4()}")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass


def main() -> None:
    # The service is reachable only inside the disposable E2E Compose network.
    ThreadingHTTPServer(("0.0.0.0", 8089), ProviderHandler).serve_forever()


if __name__ == "__main__":
    main()
