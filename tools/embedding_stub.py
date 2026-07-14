#!/usr/bin/env python3
"""Deterministic OpenAI-compatible embedding stub for Compose smoke only."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


VECTOR = [1.0] + [0.0] * 1023


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - HTTP handler contract
        self._send({"status": "ready"})

    def do_POST(self) -> None:  # noqa: N802 - HTTP handler contract
        length = min(int(self.headers.get("Content-Length", "0")), 1_048_576)
        payload = json.loads(self.rfile.read(length))
        values = payload.get("input", [])
        self._send(
            {
                "model": "text-embedding-v4",
                "data": [
                    {"index": index, "embedding": VECTOR}
                    for index, _value in enumerate(values)
                ],
            }
        )

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _send(self, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8089), Handler).serve_forever()
