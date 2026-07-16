"""Content-safe static runtime for the local observation frontend."""

from __future__ import annotations

import argparse
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import posixpath
from urllib.parse import unquote, urlsplit


API_PATH = "/api/v1"
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000/api/v1"


class ContentSafeHandler(SimpleHTTPRequestHandler):
    """Serve only compiled assets, health, and a non-secret runtime config."""

    server_version = "RagKbFrontend/1.0"

    def __init__(
        self,
        *args: object,
        api_base_url: str,
        api_origin: str,
        **kwargs: object,
    ) -> None:
        self.api_base_url = api_base_url
        self.api_origin = api_origin
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def list_directory(self, path: str):  # type: ignore[no-untyped-def]
        del path
        self.send_error(HTTPStatus.NOT_FOUND)
        return None

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = canonical_request_path(urlsplit(self.path).path)
        if path == "/health":
            self._send_json({"status": "ok"}, cache_control="no-store")
            return
        if path == "/runtime-config.json":
            self._send_json(
                {"api_base_url": self.api_base_url},
                cache_control="no-store",
            )
            return
        self._serve_static_or_index(path)

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
        path = canonical_request_path(urlsplit(self.path).path)
        if path in {"/health", "/runtime-config.json"}:
            payload = (
                {"status": "ok"}
                if path == "/health"
                else {"api_base_url": self.api_base_url}
            )
            body = _json_bytes(payload)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        self._serve_static_or_index(path, head_only=True)

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; "
            "base-uri 'none'; "
            f"connect-src 'self' {self.api_origin}; "
            "font-src 'self'; "
            "form-action 'self'; "
            "frame-ancestors 'none'; "
            "img-src 'self' data:; "
            "object-src 'none'; "
            "script-src 'self'; "
            "style-src 'self'",
        )
        super().end_headers()

    def _serve_static_or_index(self, request_path: str, *, head_only: bool = False) -> None:
        translated = Path(self.translate_path(request_path))
        if translated.is_file():
            self.path = request_path
            if request_path.startswith("/assets/"):
                self._asset_cache = True
            try:
                if head_only:
                    super().do_HEAD()
                else:
                    super().do_GET()
            finally:
                self._asset_cache = False
            return

        # Missing files and asset requests remain 404; only extensionless UI
        # routes receive the compiled application shell.
        if request_path.startswith("/assets/") or Path(request_path).suffix:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        index = Path(self.directory) / "index.html"
        if not index.is_file():
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
            return
        self.path = "/index.html"
        if head_only:
            super().do_HEAD()
        else:
            super().do_GET()

    def send_header(self, keyword: str, value: str) -> None:
        if keyword.lower() == "cache-control":
            return super().send_header(keyword, value)
        if keyword.lower() == "content-type" and self.path.endswith(".html"):
            value = "text/html; charset=utf-8"
        super().send_header(keyword, value)

    def guess_type(self, path: str) -> str:
        if path.endswith(".js"):
            return "text/javascript"
        return super().guess_type(path)

    def _send_json(self, value: dict[str, str], *, cache_control: str) -> None:
        body = _json_bytes(value)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(body)

    def send_response(self, code: int, message: str | None = None) -> None:
        super().send_response(code, message)
        if getattr(self, "_asset_cache", False) and code == HTTPStatus.OK:
            super().send_header(
                "Cache-Control", "public, max-age=31536000, immutable"
            )
        elif code == HTTPStatus.OK and self.path.endswith(".html"):
            super().send_header("Cache-Control", "no-cache")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent / "dist")
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    args = parser.parse_args(argv)

    api_base_url, api_origin = validate_api_base_url(args.api_base_url)
    directory = args.directory.resolve(strict=True)
    if not directory.is_dir() or not (directory / "index.html").is_file():
        parser.error("compiled frontend directory is missing index.html")

    handler = partial(
        ContentSafeHandler,
        directory=str(directory),
        api_base_url=api_base_url,
        api_origin=api_origin,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        json.dumps(
            {
                "event": "test_frontend_ready",
                "level": "INFO",
                "process": "frontend",
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def validate_api_base_url(value: str) -> tuple[str, str]:
    parsed = urlsplit(value)
    try:
        parsed.port
    except ValueError:
        raise ValueError(
            "API base URL must be an uncredentialed loopback /api/v1 URL"
        ) from None
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or parsed.netloc.endswith(":")
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != API_PATH
    ):
        raise ValueError("API base URL must be an uncredentialed loopback /api/v1 URL")
    try:
        loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = parsed.hostname == "localhost"
    if not loopback:
        raise ValueError("API base URL must use a loopback host")
    normalized = value.rstrip("/")
    return normalized, f"{parsed.scheme}://{parsed.netloc}"


def canonical_request_path(value: str) -> str:
    """Decode and normalize a URL path before applying privileged routes."""

    decoded = unquote(value)
    normalized = posixpath.normpath(f"/{decoded.lstrip('/')}")
    if decoded.endswith("/") and normalized != "/":
        normalized += "/"
    return normalized


def _json_bytes(value: dict[str, str]) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
