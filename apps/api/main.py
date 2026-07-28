"""Uvicorn entrypoint for direct and Compose development execution."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from rag_kb.config import Settings, load_settings
from rag_kb.observability import configure_logging, get_logger, log_event


LOGGER = get_logger("rag_kb.api.runtime")


def server_address(
    settings: Settings,
    *,
    container_listen: bool,
) -> tuple[str, int]:
    host = "0.0.0.0" if container_listen else str(settings.app.bind_host)
    return host, settings.app.api_port


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--container-listen",
        action="store_true",
        help="listen on the container interface; Compose must publish loopback-only",
    )
    arguments = parser.parse_args()
    configure_logging(level="INFO")
    try:
        settings = load_settings()
        configure_logging(level=settings.observability.log_level)
        host, port = server_address(
            settings,
            container_listen=arguments.container_listen,
        )
        uvicorn.run(
            "apps.api.app:application",
            host=host,
            port=port,
            access_log=False,
            proxy_headers=False,
            server_header=False,
            log_config=None,
        )
    except (Exception, SystemExit) as error:
        log_event(
            LOGGER,
            "process_failed",
            level=logging.ERROR,
            process="api",
            error_type=type(error).__name__,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
