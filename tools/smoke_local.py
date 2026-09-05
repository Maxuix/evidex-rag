#!/usr/bin/env python3
"""Run the small read-only smoke check for the local RAG application."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from urllib.parse import urljoin, urlsplit
from urllib.error import URLError
from urllib.request import urlopen

from tools.local_runtime import LocalRuntimeError, resolve_local_runtime


def read_json(url: str) -> dict:
    with urlopen(url, timeout=5) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise RuntimeError(f"{url} did not return a JSON object")
    return value


class FrontendShell(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.has_root = False
        self.assets: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        self.has_root |= values.get("id") == "root"
        source = values.get("src")
        href = values.get("href")
        if tag == "script" and source:
            self.assets[source] = "script"
        if tag == "link" and "stylesheet" in (values.get("rel") or "").split() and href:
            self.assets[href] = "style"


def check_frontend(origin: str) -> None:
    """Check the delivered shell and its actual JS/CSS, beyond health JSON."""
    homepage = origin.rstrip("/") + "/"
    with urlopen(homepage, timeout=5) as response:
        if response.status != 200 or response.headers.get_content_type() != "text/html":
            raise RuntimeError("frontend homepage is not HTML")
        shell = FrontendShell()
        shell.feed(response.read().decode("utf-8"))
    if not shell.has_root or not {"script", "style"}.issubset(shell.assets.values()):
        raise RuntimeError("frontend homepage is missing the compiled application")
    expected = urlsplit(homepage)
    for reference, kind in shell.assets.items():
        url = urljoin(homepage, reference)
        target = urlsplit(url)
        if (target.scheme, target.netloc) != (expected.scheme, expected.netloc):
            raise RuntimeError("frontend asset is outside the local application origin")
        allowed = {"text/css"} if kind == "style" else {"text/javascript", "application/javascript"}
        with urlopen(url, timeout=5) as response:
            if (
                response.status != 200
                or response.headers.get_content_type() not in allowed
                or not response.read(1)
            ):
                raise RuntimeError("frontend compiled asset is missing or has the wrong content type")


def main() -> int:
    try:
        runtime = resolve_local_runtime()
    except (LocalRuntimeError, OSError):
        print("Local smoke failed: local runtime manifest is invalid")
        return 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default=runtime.api_origin)
    parser.add_argument("--frontend", default=runtime.frontend_origin)
    args = parser.parse_args()

    try:
        live = read_json(f"{args.api}/health/live")
        ready = read_json(f"{args.api}/health/ready")
        frontend = read_json(f"{args.frontend}/health")
        openapi = read_json(f"{args.api}/api/v1/openapi.json")
        check_frontend(args.frontend)
    except (OSError, URLError, ValueError, RuntimeError) as error:
        print(f"Local smoke failed: {error}")
        return 1

    paths = openapi.get("paths")
    required = {
        "/api/v1/knowledge-bases",
        "/api/v1/retrieval/query",
        "/api/v1/chat/sessions",
    }
    if (
        live.get("status") != "alive"
        or ready.get("status") != "ready"
        or frontend.get("status") != "ok"
        or not isinstance(paths, dict)
        or not required.issubset(paths)
    ):
        print("Local smoke failed: health or basic API surface is incomplete")
        return 1

    print("Local smoke passed: API, database readiness, frontend HTML/JS/CSS, and core routes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
