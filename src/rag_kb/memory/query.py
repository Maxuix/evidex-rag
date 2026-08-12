"""Durable serialization for the legacy contextual-query snapshot column."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rag_kb.domain import (
    CONTEXTUAL_QUERY_VERSION,
    ChatModelCallRecord,
    ChatModelOperation,
    ContextualizedQuery,
    QueryContextStatus,
    QueryRewriteSource,
)


def serialize_contextualized_query(value: ContextualizedQuery) -> dict[str, Any]:
    if value.rewrite_source is None:
        raise ValueError("contextual query is missing rewrite source")
    return {
        "version": value.version,
        "status": value.status.value,
        "original_query": value.original_query,
        "standalone_query": value.standalone_query,
        "context_hash": value.context_hash,
        "model_calls": [
            {
                "operation": call.operation.value,
                "model": call.model,
                "provider_request_id": call.provider_request_id,
                "usage": dict(call.usage),
            }
            for call in value.model_calls
        ],
        "created_at": value.created_at.isoformat() if value.created_at else None,
        "origin_attempt": value.origin_attempt,
        "rewrite_source": value.rewrite_source.value,
    }


def hydrate_contextualized_query(value: object) -> ContextualizedQuery:
    if not isinstance(value, dict):
        raise ValueError("contextualized query must be an object")
    expected = {
        "version",
        "status",
        "original_query",
        "standalone_query",
        "context_hash",
        "model_calls",
        "created_at",
        "origin_attempt",
        "rewrite_source",
    }
    if value.get("version") != CONTEXTUAL_QUERY_VERSION or set(value) != expected:
        raise ValueError("contextualized query shape is invalid")
    calls = _hydrate_calls(value["model_calls"])
    created = value["created_at"]
    if created is not None and not isinstance(created, str):
        raise ValueError("contextualized query creation time is invalid")
    if (
        not isinstance(value["status"], str)
        or not isinstance(value["original_query"], str)
        or (
            value["standalone_query"] is not None
            and not isinstance(value["standalone_query"], str)
        )
        or not isinstance(value["context_hash"], str)
        or not isinstance(value["rewrite_source"], str)
    ):
        raise ValueError("contextualized query fields are invalid")
    try:
        result = ContextualizedQuery(
            version=CONTEXTUAL_QUERY_VERSION,
            status=QueryContextStatus(value["status"]),
            original_query=value["original_query"],
            standalone_query=value["standalone_query"],
            context_hash=value["context_hash"],
            model_calls=calls,
            created_at=datetime.fromisoformat(created) if created else None,
            origin_attempt=value["origin_attempt"],
            rewrite_source=QueryRewriteSource(value["rewrite_source"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("contextualized query fields are invalid") from error
    if serialize_contextualized_query(result) != value:
        raise ValueError("contextualized query is not canonical")
    return result


def _hydrate_calls(value: object) -> tuple[ChatModelCallRecord, ...]:
    if not isinstance(value, list):
        raise ValueError("contextualized model calls must be an array")
    calls: list[ChatModelCallRecord] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "operation",
            "model",
            "provider_request_id",
            "usage",
        }:
            raise ValueError("contextualized model call shape is invalid")
        if item["operation"] != ChatModelOperation.CONTEXTUALIZE_QUERY.value:
            raise ValueError("contextualized model call operation is invalid")
        if (
            not isinstance(item["model"], str)
            or not item["model"]
            or (
                item["provider_request_id"] is not None
                and not isinstance(item["provider_request_id"], str)
            )
            or not isinstance(item["usage"], dict)
        ):
            raise ValueError("contextualized model call fields are invalid")
        calls.append(
            ChatModelCallRecord(
                operation=ChatModelOperation(item["operation"]),
                model=item["model"],
                provider_request_id=item["provider_request_id"],
                usage=item["usage"],
            )
        )
    return tuple(calls)
