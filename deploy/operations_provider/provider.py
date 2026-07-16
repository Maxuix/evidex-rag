#!/usr/bin/env python3
"""Deterministic network-only provider for Stage 06 recovery exercises."""

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
INDEX_DELAY_MARKER = "OPS_INDEX_DELAY"
INDEX_FAILURE_MARKER = "OPS_INDEX_FAIL"
CHAT_DELAY_MARKER = "OPS_CHAT_DELAY"


def _unit_vector(index: int) -> list[float]:
    vector = [0.0] * DIMENSION
    vector[index] = 1.0
    return vector


def deterministic_embedding(value: str) -> list[float]:
    upper = value.upper()
    if "OPS_ALPHA" in upper:
        return _unit_vector(0)
    if "OPS_BETA" in upper:
        return _unit_vector(1)
    if INDEX_DELAY_MARKER in upper:
        return _unit_vector(2)
    if INDEX_FAILURE_MARKER in upper:
        return _unit_vector(3)
    bucket = 4 + (sum(value.encode("utf-8")) % (DIMENSION - 4))
    return _unit_vector(bucket)


@dataclass(frozen=True, slots=True)
class StubResponse:
    status: int
    body: dict[str, Any]
    delay_seconds: float = 0.0


class OperationsScenario:
    """Thread-safe, finite fault behavior with no content logging."""

    def __init__(
        self,
        *,
        indexing_failures: int = 3,
        indexing_delay_seconds: float = 3.0,
        chat_delay_seconds: float = 3.0,
    ) -> None:
        if min(indexing_failures, indexing_delay_seconds, chat_delay_seconds) < 0:
            raise ValueError("provider fault controls must be non-negative")
        self._indexing_failures = indexing_failures
        self._indexing_delay_seconds = indexing_delay_seconds
        self._chat_delay_seconds = chat_delay_seconds
        self._failed_embedding_calls = 0
        self._lock = threading.Lock()

    def embeddings(self, payload: dict[str, Any]) -> StubResponse:
        values = payload.get("input")
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            return StubResponse(400, {"error": {"code": "invalid_input"}})
        upper = tuple(item.upper() for item in values)
        if any(INDEX_FAILURE_MARKER in item for item in upper):
            with self._lock:
                if self._failed_embedding_calls < self._indexing_failures:
                    self._failed_embedding_calls += 1
                    return StubResponse(503, {"error": {"code": "ops_unavailable"}})
        delay = (
            self._indexing_delay_seconds
            if any(INDEX_DELAY_MARKER in item for item in upper)
            else 0.0
        )
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
            delay_seconds=delay,
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
        citations = [
            item.get("citation_id")
            for item in user.get("evidence", [])
            if isinstance(item, dict) and isinstance(item.get("citation_id"), str)
        ]
        if "assess whether supplied evidence" in system:
            content = {
                "coverage": "sufficient" if citations else "none",
                "usable_citation_ids": citations[:1],
                "supported_aspects": ["operations_fact"] if citations else [],
                "missing_aspects": [],
            }
        elif "internal answer draft" in system or "repair an untrusted" in system:
            content = {
                "outcome": user.get("required_outcome", "answered"),
                "claims": (
                    [
                        {
                            "text": "The operations fixture confirms the requested fact.",
                            "citation_ids": citations[:1],
                        }
                    ]
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
    scenario = OperationsScenario()

    def do_GET(self) -> None:  # noqa: N802
        response = (
            StubResponse(200, {"status": "ready"})
            if self.path == "/health"
            else StubResponse(404, {"error": {"code": "not_found"}})
        )
        self._send(response)

    def do_POST(self) -> None:  # noqa: N802
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
        self.send_header("x-request-id", f"ops-{uuid.uuid4()}")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass


def main() -> None:
    ThreadingHTTPServer(("0.0.0.0", 8089), ProviderHandler).serve_forever()


if __name__ == "__main__":
    main()
