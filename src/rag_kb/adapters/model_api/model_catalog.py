"""Model discovery for OpenAI-compatible provider endpoints."""

from __future__ import annotations

from typing import Any

import httpx

from rag_kb.domain import ModelProviderBundle, ModelProviderProtocol


_MAX_CATALOG_BYTES = 1024 * 1024
_MAX_MODELS = 1000
_RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})


class OpenAICompatibleModelCatalogAdapter:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def list_models(
        self,
        provider: ModelProviderBundle,
        api_key: str,
    ) -> tuple[str, ...]:
        revision = provider.current_revision
        if revision.protocol is not ModelProviderProtocol.OPENAI_COMPATIBLE:
            raise ValueError("provider does not support model discovery")
        url = f"{revision.base_url.rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        if self._client is not None:
            return await self._request(
                self._client,
                url,
                headers,
                max_retries=revision.max_retries,
            )
        async with httpx.AsyncClient(timeout=revision.timeout_seconds) as client:
            return await self._request(
                client,
                url,
                headers,
                max_retries=revision.max_retries,
            )

    async def _request(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        *,
        max_retries: int,
    ) -> tuple[str, ...]:
        response: httpx.Response | None = None
        for attempt in range(max_retries + 1):
            try:
                response = await client.get(url, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt >= max_retries:
                    raise
                continue
            if (
                response.status_code not in _RETRYABLE_STATUSES
                or attempt >= max_retries
            ):
                break
        assert response is not None
        response.raise_for_status()
        if len(response.content) > _MAX_CATALOG_BYTES:
            raise ValueError("provider model catalog is too large")
        return parse_model_catalog(response.json())


def parse_model_catalog(payload: Any) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        raise ValueError("provider model catalog must be an object")
    items = payload.get("data")
    if not isinstance(items, list):
        items = payload.get("models")
    if not isinstance(items, list):
        raise ValueError("provider model catalog has no model list")
    model_ids: set[str] = set()
    for item in items[:_MAX_MODELS]:
        candidate: Any = None
        if isinstance(item, str):
            candidate = item
        elif isinstance(item, dict):
            candidate = item.get("id") or item.get("name") or item.get("model")
        if isinstance(candidate, str):
            normalized = candidate.strip()
            if 0 < len(normalized) <= 255:
                model_ids.add(normalized)
    if not model_ids:
        raise ValueError("provider model catalog contains no usable models")
    return tuple(sorted(model_ids, key=str.casefold))
