"""Minimal content-safe static server for the Stage 02 frontend shell."""

from __future__ import annotations

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class ContentSafeHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        del format, args


def main() -> int:
    directory = Path(__file__).resolve().parent
    handler = partial(ContentSafeHandler, directory=str(directory))
    server = ThreadingHTTPServer(("0.0.0.0", 3000), handler)
    print(
        json.dumps(
            {
                "event": "frontend_shell_ready",
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


if __name__ == "__main__":
    raise SystemExit(main())
