from __future__ import annotations

from functools import partial
import http.client
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
from types import ModuleType
import unittest
from unittest.mock import patch

from tools.smoke_local import check_frontend


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = PROJECT_ROOT / "apps" / "web-chat" / "server.py"
API_BASE_URL = "http://127.0.0.1:8000/api/v1"
API_ORIGIN = "http://127.0.0.1:8000"
INDEX_BODY = (
    b'<!doctype html><html><head><script type="module" src="/assets/app-deadbeef.js"></script>'
    b'<link rel="stylesheet" href="/assets/app-deadbeef.css"></head>'
    b'<body><div id="root">compiled-shell</div></body></html>'
)


def _load_server_module(
    path: Path = SERVER_PATH,
    module_name: str = "rag_kb_user_frontend_server",
) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load user frontend server from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SERVER_MODULE = _load_server_module()


class FrontendServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dist = self.root / "dist"
        (self.dist / "assets" / "chunks").mkdir(parents=True)
        (self.dist / "index.html").write_bytes(INDEX_BODY)
        (self.dist / "assets" / "app-deadbeef.js").write_text(
            "globalThis.frontendLoaded = true;",
            encoding="utf-8",
        )
        (self.dist / "assets" / "app-deadbeef.css").write_text("body { color: black; }")
        (self.dist / "assets" / "chunks" / "chunk-deadbeef.js").write_text(
            "export const compiled = true;",
            encoding="utf-8",
        )

        handler = partial(
            SERVER_MODULE.ContentSafeHandler,
            directory=str(self.dist),
            api_base_url=API_BASE_URL,
            api_origin=API_ORIGIN,
        )
        self.server = SERVER_MODULE.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.server_thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        self.server_thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=2)
        self.temporary_directory.cleanup()

    def request(
        self,
        path: str,
        *,
        method: str = "GET",
    ) -> tuple[int, dict[str, str], bytes]:
        host, port = self.server.server_address
        connection = http.client.HTTPConnection(host, port, timeout=2)
        try:
            connection.request(method, path)
            response = connection.getresponse()
            body = response.read()
            headers = {key.lower(): value for key, value in response.getheaders()}
            return response.status, headers, body
        finally:
            connection.close()

    def test_runtime_api_url_accepts_only_loopback_api_v1_urls(self) -> None:
        valid_urls = {
            "http://127.0.0.1:8000/api/v1": (
                "http://127.0.0.1:8000/api/v1",
                "http://127.0.0.1:8000",
            ),
            "http://localhost/api/v1/": (
                "http://localhost/api/v1",
                "http://localhost",
            ),
            "http://[::1]:8080/api/v1": (
                "http://[::1]:8080/api/v1",
                "http://[::1]:8080",
            ),
        }
        for value, expected in valid_urls.items():
            with self.subTest(value=value):
                self.assertEqual(SERVER_MODULE.validate_api_base_url(value), expected)

        invalid_urls = (
            "",
            "https://127.0.0.1:8000/api/v1",
            "http://api:8000/api/v1",
            "http://192.168.1.10:8000/api/v1",
            "http://user@localhost:8000/api/v1",
            "http://user:password@localhost:8000/api/v1",
            "http://localhost:8000/api/v2",
            "http://localhost:8000/api/v1?workspace=other",
            "http://localhost:8000/api/v1#fragment",
            "http://localhost:notaport/api/v1",
            "http://localhost:99999/api/v1",
            "http://localhost:/api/v1",
        )
        for value in invalid_urls:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    SERVER_MODULE.validate_api_base_url(value)

    def test_health_returns_minimal_uncached_json(self) -> None:
        status, headers, body = self.request("/health")

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"status": "ok"})
        self.assertEqual(headers["content-type"], "application/json; charset=utf-8")
        self.assertEqual(headers["cache-control"], "no-store")

    def test_health_rejects_missing_or_empty_shell(self) -> None:
        for remove in (False, True):
            with self.subTest(remove=remove):
                (self.dist / "index.html").write_bytes(b"")
                if remove:
                    (self.dist / "index.html").unlink()
                status, _, body = self.request("/health")
                self.assertEqual(status, 503)
                self.assertEqual(json.loads(body), {"status": "unavailable"})

    def test_health_rejects_unreadable_shell_for_get_and_head(self) -> None:
        with patch.object(Path, "open", side_effect=PermissionError):
            status, _, body = self.request("/health")
            self.assertEqual(status, 503)
            self.assertEqual(json.loads(body), {"status": "unavailable"})
            status, _, body = self.request("/health", method="HEAD")
            self.assertEqual(status, 503)
            self.assertEqual(body, b"")

    def test_smoke_reads_compiled_assets_and_rejects_a_missing_bundle(self) -> None:
        host, port = self.server.server_address
        origin = f"http://{host}:{port}"
        check_frontend(origin)
        (self.dist / "assets" / "app-deadbeef.js").unlink()
        with self.assertRaises(OSError):
            check_frontend(origin)

    def test_smoke_rejects_error_html_returned_as_a_successful_homepage(self) -> None:
        (self.dist / "index.html").write_text("<html><body>Error response</body></html>")
        host, port = self.server.server_address
        with self.assertRaisesRegex(RuntimeError, "missing the compiled application"):
            check_frontend(f"http://{host}:{port}")

    def test_runtime_config_is_uncached_and_has_security_headers(self) -> None:
        status, headers, body = self.request("/runtime-config.json")

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"api_base_url": API_BASE_URL})
        self.assertEqual(headers["cache-control"], "no-store")
        self.assertEqual(headers["x-content-type-options"], "nosniff")
        self.assertEqual(headers["referrer-policy"], "no-referrer")
        self.assertEqual(headers["x-frame-options"], "DENY")
        self.assertEqual(
            headers["permissions-policy"],
            "camera=(), microphone=(), geolocation=()",
        )
        self.assertEqual(
            headers["content-security-policy"],
            "default-src 'self'; "
            "base-uri 'none'; "
            f"connect-src 'self' {API_ORIGIN}; "
            "font-src 'self'; "
            "form-action 'self'; "
            "frame-ancestors 'none'; "
            f"img-src 'self' data: {API_ORIGIN}; "
            "object-src 'none'; "
            "script-src 'self'; "
            "style-src 'self'",
        )

    def test_runtime_config_aliases_are_canonicalized_before_dispatch(self) -> None:
        aliases = (
            "/runtime-config%2Ejson",
            "/%72untime-config.json",
            "/ui/../runtime-config.json",
            "/ui/%2e%2e/runtime-config.json",
        )

        for path in aliases:
            with self.subTest(path=path):
                status, headers, body = self.request(path)
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body), {"api_base_url": API_BASE_URL})
                self.assertEqual(headers["cache-control"], "no-store")

                head_status, head_headers, head_body = self.request(
                    path,
                    method="HEAD",
                )
                self.assertEqual(head_status, 200)
                self.assertEqual(head_headers["cache-control"], "no-store")
                self.assertEqual(head_body, b"")

    def test_spa_deep_link_falls_back_to_compiled_index(self) -> None:
        status, headers, body = self.request("/chat/runs/run-1?tab=evidence")

        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "text/html; charset=utf-8")
        self.assertEqual(headers["cache-control"], "no-cache")
        self.assertEqual(body, INDEX_BODY)

    def test_missing_asset_remains_not_found(self) -> None:
        status, _, body = self.request("/assets/missing-deadbeef.js")

        self.assertEqual(status, 404)
        self.assertNotEqual(body, INDEX_BODY)

    def test_asset_directory_listing_is_denied(self) -> None:
        status, _, body = self.request("/assets/chunks/")

        self.assertEqual(status, 404)
        self.assertNotIn(b"chunk-deadbeef.js", body)
        self.assertNotIn(b"Directory listing", body)

    def test_source_files_cannot_escape_the_compiled_directory(self) -> None:
        source_marker = b"SOURCE_MARKER_MUST_NOT_LEAK"
        (self.root / "frontend-source.ts").write_bytes(source_marker)
        attempted_paths = (
            "/server.py",
            "/package.json",
            "/src/main.tsx",
            "/assets/app-deadbeef.js.map",
            "/../frontend-source.ts",
            "/%2e%2e/frontend-source.ts",
            "/%2e%2e%2ffrontend-source.ts",
            "/assets/%2e%2e/%2e%2e/frontend-source.ts",
        )

        for path in attempted_paths:
            with self.subTest(path=path):
                status, _, body = self.request(path)
                self.assertEqual(status, 404)
                self.assertNotIn(source_marker, body)
                self.assertNotIn(b"ContentSafeHandler", body)


if __name__ == "__main__":
    unittest.main()
