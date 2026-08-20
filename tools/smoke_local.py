#!/usr/bin/env python3
"""Run the small read-only smoke check for the local RAG application."""

from __future__ import annotations

import argparse
import json
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

    print("Local smoke passed: API, database readiness, frontend, and core routes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
