"""Credential-free HTTP image fetcher with public-address pinning."""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
from urllib.parse import quote, urljoin, urlsplit

from rag_kb.adapters.markdown_media.contracts import FetchedImage
from rag_kb.domain import ErrorCode, FileAdmissionError


_MAX_REMOTE_URL_LENGTH = 2048
_CONNECT_TIMEOUT_SECONDS = 5.0
_READ_TIMEOUT_SECONDS = 10.0
_MAX_REDIRECTS = 3


class PublicHttpImageFetcher:
    """Fetch one image while pinning each hop to a validated public address."""

    def fetch(self, url: str, *, max_bytes: int) -> FetchedImage:
        current = url
        for redirect_count in range(_MAX_REDIRECTS + 1):
            response, body = self._request_once(current, max_bytes=max_bytes)
            try:
                if response.status in {301, 302, 303, 307, 308}:
                    if redirect_count >= _MAX_REDIRECTS:
                        raise _fetch_error()
                    location = response.getheader("Location")
                    if not location:
                        raise _fetch_error()
                    current = urljoin(current, location)
                    continue
                if response.status < 200 or response.status >= 300:
                    raise _fetch_error()
                return FetchedImage(content=body, final_url=current)
            finally:
                response.close()
        raise _fetch_error()

    @staticmethod
    def _request_once(
        url: str,
        *,
        max_bytes: int,
    ) -> tuple[http.client.HTTPResponse, bytes]:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or len(url) > _MAX_REMOTE_URL_LENGTH
        ):
            raise _fetch_error()
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            request_host = parsed.hostname.encode("idna").decode("ascii")
        except (UnicodeError, ValueError) as error:
            raise _fetch_error() from error
        addresses = _public_addresses(request_host, port)
        raw_socket: socket.socket | ssl.SSLSocket | None = None
        last_error: OSError | None = None
        for family, socktype, protocol, _canonical, address in addresses:
            candidate = socket.socket(family, socktype, protocol)
            candidate.settimeout(_CONNECT_TIMEOUT_SECONDS)
            try:
                candidate.connect(address)
                raw_socket = candidate
                break
            except OSError as error:
                last_error = error
                candidate.close()
        if raw_socket is None:
            raise _fetch_error() from last_error
        try:
            if parsed.scheme == "https":
                raw_socket = ssl.create_default_context().wrap_socket(
                    raw_socket,
                    server_hostname=request_host,
                )
            raw_socket.settimeout(_READ_TIMEOUT_SECONDS)
            target = quote(
                parsed.path or "/",
                safe="/%:@!$&'()*+,;=-._~",
            )
            if parsed.query:
                query = quote(
                    parsed.query,
                    safe="=&%:@!$()*+,;/?-._~",
                )
                target = f"{target}?{query}"
            host = f"[{request_host}]" if ":" in request_host else request_host
            if parsed.port is not None:
                host = f"{host}:{parsed.port}"
            request = (
                f"GET {target} HTTP/1.1\r\n"
                f"Host: {host}\r\n"
                "Accept: image/png,image/jpeg,image/webp\r\n"
                "User-Agent: rag-kb-markdown-media/1\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii", errors="strict")
            raw_socket.sendall(request)
            response = http.client.HTTPResponse(raw_socket)
            response.begin()
            length = response.getheader("Content-Length")
            if length is not None:
                try:
                    if int(length) > max_bytes:
                        raise _fetch_error()
                except ValueError as error:
                    raise _fetch_error() from error
            body = _read_bounded(response, max_bytes)
            return response, body
        except FileAdmissionError:
            raw_socket.close()
            raise
        except (
            UnicodeError,
            OSError,
            ssl.SSLError,
            http.client.HTTPException,
        ) as error:
            raw_socket.close()
            raise _fetch_error() from error


def _public_addresses(host: str, port: int):
    try:
        addresses = socket.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror as error:
        raise _fetch_error() from error
    if not addresses:
        raise _fetch_error()
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise _fetch_error()
    return addresses


def _read_bounded(
    response: http.client.HTTPResponse,
    max_bytes: int,
) -> bytes:
    chunks: list[bytes] = []
    observed = 0
    while True:
        block = response.read(min(64 * 1024, max_bytes + 1 - observed))
        if not block:
            break
        chunks.append(block)
        observed += len(block)
        if observed > max_bytes:
            raise _fetch_error()
    return b"".join(chunks)


def _fetch_error() -> FileAdmissionError:
    return FileAdmissionError(ErrorCode.MARKDOWN_MEDIA_FETCH_FAILED)
