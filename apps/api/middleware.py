"""Content-safe request tracing middleware."""

from __future__ import annotations

from uuid import uuid4

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag_kb.observability import bind_log_context


class TraceIdMiddleware:
    """Assign a server-generated trace identifier to every HTTP request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        trace_id = str(uuid4())
        state = scope.setdefault("state", {})
        state["trace_id"] = trace_id

        async def send_with_trace(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Trace-ID"] = trace_id
            await send(message)

        with bind_log_context(trace_id=trace_id):
            await self.app(scope, receive, send_with_trace)
