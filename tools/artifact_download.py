"""Bounded retry helpers for immutable build-time model downloads."""

from __future__ import annotations

from collections.abc import Callable
import sys
import time
from typing import TypeVar

import httpx
from huggingface_hub.errors import HfHubHTTPError
import requests


_T = TypeVar("_T")
_RETRY_DELAYS_SECONDS = (2, 4, 8, 16, 30, 30)
_TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


def retry_transient_download(
    operation: Callable[[], _T],
    *,
    description: str,
) -> _T:
    """Retry only transport failures and explicitly transient HTTP statuses."""

    for attempt, delay_seconds in enumerate(
        (*_RETRY_DELAYS_SECONDS, None),
        start=1,
    ):
        try:
            return operation()
        except Exception as error:
            if delay_seconds is None or not _is_transient_network_error(error):
                raise
            print(
                f"transient download failure for {description}; "
                f"retrying in {delay_seconds}s "
                f"(attempt {attempt + 1}/{len(_RETRY_DELAYS_SECONDS) + 1})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay_seconds)
    raise AssertionError("download retry loop exhausted without returning")


def _is_transient_network_error(error: BaseException) -> bool:
    for current in _exception_chain(error):
        if isinstance(
            current,
            (
                httpx.TransportError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ),
        ):
            return True
        if isinstance(current, (HfHubHTTPError, httpx.HTTPStatusError)):
            response = current.response
            if response is not None and response.status_code in _TRANSIENT_HTTP_STATUSES:
                return True
        if isinstance(current, requests.exceptions.HTTPError):
            response = current.response
            if response is not None and response.status_code in _TRANSIENT_HTTP_STATUSES:
                return True
    return False


def _exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return tuple(chain)
