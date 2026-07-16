#!/usr/bin/env python3
"""Freeze the Stage 06 W01 frontend's reviewed public API subset.

The checked OpenAPI snapshot remains the transport authority.  This focused
check records only the operations and DTO details consumed by the observation
frontend, then verifies that the TypeScript client still uses the bounded raw
upload and retrieval-debug request shapes.  It performs no network or file
writes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENAPI = PROJECT_ROOT / "tests/contract/snapshots/openapi-v1.json"
DEFAULT_CLIENT = PROJECT_ROOT / "apps/web-test/src/api/client.ts"
PROBLEM_REF = "#/components/schemas/ProblemDetails"


@dataclass(frozen=True, slots=True)
class OperationSpec:
    method: str
    path: str
    statuses: frozenset[str]
    success_status: str
    response_schema: str | None


@dataclass(frozen=True, slots=True)
class SchemaSpec:
    fields: frozenset[str]
    optional: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class EnumSpec:
    schema: str
    field: str | None
    values: frozenset[object]


def _operation(
    method: str,
    path: str,
    statuses: str,
    success_status: str,
    response_schema: str | None,
) -> OperationSpec:
    return OperationSpec(
        method=method,
        path=path,
        statuses=frozenset(statuses.split()),
        success_status=success_status,
        response_schema=response_schema,
    )


W01_OPERATIONS = (
    _operation(
        "get",
        "/api/v1/knowledge-bases",
        "200 400 422",
        "200",
        "KnowledgeBasePage",
    ),
    _operation(
        "post",
        "/api/v1/knowledge-bases",
        "201 409 422",
        "201",
        "KnowledgeBaseResponse",
    ),
    _operation(
        "get",
        "/api/v1/knowledge-bases/{kb_id}/documents",
        "200 400 404 422",
        "200",
        "DocumentPage",
    ),
    _operation(
        "post",
        "/api/v1/knowledge-bases/{kb_id}/documents",
        "202 404 409 413 415 422",
        "202",
        "DocumentUploadResponse",
    ),
    _operation(
        "get",
        "/api/v1/documents/{document_id}",
        "200 404 422",
        "200",
        "DocumentResponse",
    ),
    _operation(
        "post",
        "/api/v1/documents/{document_id}/versions",
        "202 404 409 413 415 422",
        "202",
        "DocumentUploadResponse",
    ),
    _operation(
        "get",
        "/api/v1/indexing-jobs/{job_id}",
        "200 404 422",
        "200",
        "IndexingJobResponse",
    ),
    _operation(
        "post",
        "/api/v1/indexing-jobs/{job_id}/retry",
        "202 404 409 422",
        "202",
        "IndexingJobResponse",
    ),
    _operation(
        "get",
        "/api/v1/chat/sessions",
        "200 400 422",
        "200",
        "ChatSessionPage",
    ),
    _operation(
        "post",
        "/api/v1/chat/sessions",
        "201 404 422",
        "201",
        "ChatSessionResponse",
    ),
    _operation(
        "get",
        "/api/v1/chat/sessions/{session_id}/messages",
        "200 400 404 422",
        "200",
        "ChatMessagePage",
    ),
    _operation(
        "post",
        "/api/v1/chat/runs",
        "202 404 409 422",
        "202",
        "ChatRunResponse",
    ),
    _operation(
        "get",
        "/api/v1/chat/runs/{run_id}",
        "200 404 422",
        "200",
        "ChatRunResponse",
    ),
    _operation(
        "get",
        "/api/v1/chat/runs/{run_id}/events",
        "200 400 404 422 429",
        "200",
        None,
    ),
    _operation(
        "post",
        "/api/v1/retrieval/query",
        "200 404 409 422 500 502 503",
        "200",
        "EvidencePackResponse",
    ),
)


def _schema(fields: str, optional: str = "") -> SchemaSpec:
    return SchemaSpec(
        fields=frozenset(fields.split()),
        optional=frozenset(optional.split()),
    )


# Exact JSON properties and OpenAPI-required fields for response DTOs read by
# W01.  Optional/defaulted response properties remain recorded so their removal
# is detected even though OpenAPI does not put them in ``required``.
RESPONSE_SCHEMAS = {
    "ProblemDetails": _schema(
        "type title status detail instance code trace_id retryable errors", "errors"
    ),
    "KnowledgeBasePage": _schema("items next_cursor", "next_cursor"),
    "KnowledgeBaseResponse": _schema(
        "id name source_change_seq active_index_revision_id embedding_space_id "
        "retrieval_defaults answer_policy_defaults provisioned_at created_at updated_at",
        "answer_policy_defaults",
    ),
    "RetrievalDefaults": _schema("strategy top_k", "strategy top_k"),
    "KnowledgeBaseAnswerPolicyDefaults": _schema(
        "answer_style insufficiency_policy", "answer_style insufficiency_policy"
    ),
    "DocumentPage": _schema("items next_cursor", "next_cursor"),
    "DocumentResponse": _schema(
        "id kb_id display_name current_version deleted_at created_at updated_at"
    ),
    "DocumentVersionResponse": _schema(
        "id version_number source_status checksum_sha256 original_filename media_type "
        "size_bytes created_at"
    ),
    "DocumentUploadResponse": _schema(
        "document document_version_id source_change_id source_change_seq "
        "indexed_document_version_id index_revision_id job_id job_status"
    ),
    "IndexingErrorResponse": _schema("code detail"),
    "IndexingJobResponse": _schema(
        "job_id kb_id document_id document_version_id indexed_document_version_id "
        "index_revision_id status phase attempt build_status serving_status claimed_at "
        "heartbeat_at next_attempt_at error can_retry created_at updated_at"
    ),
    "ChatSessionPage": _schema("items next_cursor", "next_cursor"),
    "ChatSessionResponse": _schema(
        "id knowledge_base_id title created_at updated_at"
    ),
    "ChatMessagePage": _schema("items next_cursor", "next_cursor"),
    "ChatMessageResponse": _schema(
        "id session_id run_id role assistant_status content created_at"
    ),
    "ChatRunErrorResponse": _schema("code detail retryable"),
    "ChatCitationResponse": _schema(
        "ordinal index_chunk_id document_id document_version_id quoted_text "
        "source_location score"
    ),
    "EffectiveAnswerPolicyResponse": _schema(
        "grounding_policy answer_style insufficiency_policy citation_required "
        "citation_granularity answer_task policy_version"
    ),
    "ChatRunRetrievalResponse": _schema("strategy top_k rerank"),
    "ChatRunResponse": _schema(
        "run_id knowledge_base_id session_id user_message_id assistant_message_id "
        "index_revision_id status assistant_status answer citations status_url events_url "
        "effective_answer_policy retrieval attempt error usage timing created_at updated_at "
        "completed_at"
    ),
    "EvidenceResponse": _schema(
        "rank index_chunk_id indexed_document_version_id document_id document_version_id "
        "index_revision_id ordinal text source_location hierarchy source_metadata score "
        "score_kind"
    ),
    "RetrievalQueryPlanResponse": _schema(
        "workspace_id knowledge_base_id strategy top_k revision_selector "
        "current_document_version_only build_status serving_status distance_metric "
        "candidate_count ef_search iterative_scan rerank"
    ),
    "RetrievalDebugResponse": _schema(
        "query_plan resolved_active_revision_id result_count"
    ),
    "EvidencePackResponse": _schema(
        "knowledge_base_id index_revision_id strategy evidence debug", "debug"
    ),
}


ENUMS = (
    EnumSpec("AnswerStyle", None, frozenset({"concise", "summary"})),
    EnumSpec(
        "InsufficiencyPolicy", None, frozenset({"refuse", "partial_answer"})
    ),
    EnumSpec(
        "RetrievalStrategy",
        None,
        frozenset({"exact_vector", "ann_vector", "lexical", "hybrid"}),
    ),
    EnumSpec("EvidenceScoreKind", None, frozenset({"cosine_similarity"})),
    EnumSpec("RevisionSelector", None, frozenset({"active"})),
    EnumSpec("IterativeScanMode", None, frozenset({"disabled"})),
    EnumSpec(
        "DocumentVersionResponse",
        "source_status",
        frozenset({"available", "unavailable", "deleted"}),
    ),
    EnumSpec("DocumentUploadResponse", "job_status", frozenset({"queued"})),
    EnumSpec(
        "IndexingJobResponse",
        "status",
        frozenset({"queued", "running", "completed", "failed", "cancelled"}),
    ),
    EnumSpec(
        "IndexingJobResponse",
        "build_status",
        frozenset({"queued", "processing", "ready", "failed"}),
    ),
    EnumSpec(
        "IndexingJobResponse",
        "serving_status",
        frozenset({"candidate", "serving", "retired"}),
    ),
    EnumSpec("ChatMessageResponse", "role", frozenset({"user", "assistant"})),
    EnumSpec(
        "ChatMessageResponse",
        "assistant_status",
        frozenset({"generating", "completed", "failed"}),
    ),
    EnumSpec(
        "ChatRunResponse",
        "status",
        frozenset({"queued", "running", "completed", "failed", "cancelled"}),
    ),
    EnumSpec(
        "ChatRunResponse",
        "assistant_status",
        frozenset({"generating", "completed", "failed"}),
    ),
    EnumSpec(
        "EffectiveAnswerPolicyResponse",
        "grounding_policy",
        frozenset({"evidence_only"}),
    ),
    EnumSpec(
        "EffectiveAnswerPolicyResponse",
        "citation_required",
        frozenset({True}),
    ),
    EnumSpec(
        "EffectiveAnswerPolicyResponse",
        "citation_granularity",
        frozenset({"claim_level"}),
    ),
    EnumSpec(
        "EffectiveAnswerPolicyResponse", "answer_task", frozenset({"answer"})
    ),
    EnumSpec(
        "EffectiveAnswerPolicyResponse", "policy_version", frozenset({"p1"})
    ),
    EnumSpec(
        "ChatRunRetrievalResponse", "strategy", frozenset({"exact_vector"})
    ),
    EnumSpec("ChatRunRetrievalResponse", "rerank", frozenset({False})),
)


# Methods are checked separately so two operations sharing one route cannot be
# mistaken for each other.  The patterns intentionally describe transport use,
# not component or UI implementation details.
CLIENT_METHODS: dict[str, tuple[str, str]] = {
    "listKnowledgeBases": ("GET", r'this\.withQuery\(\s*"/knowledge-bases"'),
    "createKnowledgeBase": ("POST", r'this\.request\(\s*"/knowledge-bases"'),
    "listDocuments": (
        "GET",
        r'this\.withQuery\(\s*`/knowledge-bases/\$\{[^}]+\}/documents`',
    ),
    "getDocument": ("GET", r'this\.request\(\s*`/documents/\$\{[^}]+\}`'),
    "uploadDocument": (
        "UPLOAD",
        r'this\.upload\(\s*`/knowledge-bases/\$\{[^}]+\}/documents`',
    ),
    "uploadDocumentVersion": (
        "UPLOAD",
        r'this\.upload\(\s*`/documents/\$\{[^}]+\}/versions`',
    ),
    "getIndexingJob": (
        "GET",
        r'this\.request\(\s*`/indexing-jobs/\$\{[^}]+\}`',
    ),
    "retryIndexingJob": (
        "POST",
        r'this\.request\(\s*`/indexing-jobs/\$\{[^}]+\}/retry`',
    ),
    "listChatSessions": ("GET", r'this\.withQuery\(\s*"/chat/sessions"'),
    "createChatSession": ("POST", r'this\.request\(\s*"/chat/sessions"'),
    "listChatMessages": (
        "GET",
        r'this\.withQuery\(\s*`/chat/sessions/\$\{[^}]+\}/messages`',
    ),
    "createChatRun": ("POST", r'this\.request\(\s*"/chat/runs"'),
    "getChatRun": ("GET", r'`/chat/runs/\$\{[^}]+\}`'),
    "queryRetrievalDebug": ("POST", r'this\.request\(\s*"/retrieval/query"'),
    "subscribeChatRun": (
        "EVENTSOURCE",
        r'new\s+EventSource\(this\.resolvePublicApiUrl\(eventsUrl\)\)',
    ),
}


EXPECTED_CLIENT_ROUTE_LITERALS = frozenset(
    {
        "/knowledge-bases",
        "/knowledge-bases/{parameter}/documents",
        "/documents/{parameter}",
        "/documents/{parameter}/versions",
        "/indexing-jobs/{parameter}",
        "/indexing-jobs/{parameter}/retry",
        "/chat/sessions",
        "/chat/sessions/{parameter}/messages",
        "/chat/runs",
        "/chat/runs/{parameter}",
        "/retrieval/query",
    }
)


def check_frontend_contract(
    openapi: dict[str, Any], client_source: str
) -> list[str]:
    """Return all W01 OpenAPI and frontend-client contract deviations."""

    return [
        *_check_operations(openapi),
        *_check_upload_contract(openapi, client_source),
        *_check_response_schemas(openapi),
        *_check_enums(openapi),
        *_check_retrieval_request(openapi, client_source),
        *_check_client_methods(client_source),
    ]


def _check_operations(openapi: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    paths = openapi.get("paths", {})
    for spec in W01_OPERATIONS:
        operation = paths.get(spec.path, {}).get(spec.method)
        label = f"{spec.method.upper()} {spec.path}"
        if not isinstance(operation, dict):
            errors.append(f"missing frontend operation: {label}")
            continue
        responses = operation.get("responses", {})
        actual_statuses = frozenset(responses)
        if actual_statuses != spec.statuses:
            errors.append(
                f"frontend response statuses drifted: {label}: "
                f"expected {sorted(spec.statuses)}, got {sorted(actual_statuses)}"
            )
            continue
        success = responses[spec.success_status]
        content = success.get("content", {})
        if spec.response_schema is None:
            if "text/event-stream" not in content:
                errors.append(f"frontend SSE media type drifted: {label}")
        else:
            actual_ref = (
                content.get("application/json", {}).get("schema", {}).get("$ref")
            )
            expected_ref = f"#/components/schemas/{spec.response_schema}"
            if actual_ref != expected_ref:
                errors.append(
                    f"frontend success schema drifted: {label}: "
                    f"expected {expected_ref}, got {actual_ref}"
                )
        for status, response in responses.items():
            if status == spec.success_status:
                continue
            actual_ref = (
                response.get("content", {})
                .get("application/problem+json", {})
                .get("schema", {})
                .get("$ref")
            )
            if actual_ref != PROBLEM_REF:
                errors.append(
                    f"frontend Problem Details drifted: {label} {status}"
                )
    return errors


def _check_upload_contract(
    openapi: dict[str, Any], client_source: str
) -> list[str]:
    errors: list[str] = []
    expected_headers = {
        "Idempotency-Key": True,
        "X-Document-Filename": True,
        "X-Document-Display-Name": False,
    }
    upload_paths = (
        "/api/v1/knowledge-bases/{kb_id}/documents",
        "/api/v1/documents/{document_id}/versions",
    )
    for path in upload_paths:
        operation = openapi.get("paths", {}).get(path, {}).get("post", {})
        headers = {
            item.get("name"): item.get("required")
            for item in operation.get("parameters", ())
            if item.get("in") == "header"
        }
        if headers != expected_headers:
            errors.append(
                f"frontend upload headers drifted: POST {path}: "
                f"expected {expected_headers}, got {headers}"
            )
        request_body = operation.get("requestBody", {})
        media = request_body.get("content", {})
        if request_body.get("required") is not True or set(media) != {
            "text/plain",
            "text/markdown",
        }:
            errors.append(f"frontend upload media types drifted: POST {path}")
        for media_type, value in media.items():
            schema = value.get("schema", {})
            if schema.get("type") != "string" or schema.get("format") != "binary":
                errors.append(
                    f"frontend upload body drifted: POST {path} {media_type}"
                )

    try:
        upload = _method_body(client_source, "upload")
    except ValueError as error:
        errors.append(str(error))
        return errors
    header_object = re.search(
        r"headers\s*:\s*\{(?P<headers>.*?)\}\s*,\s*body\s*:\s*[A-Za-z_$][\w$]*",
        upload,
        re.DOTALL,
    )
    if header_object is None:
        errors.append("frontend raw upload header object is missing")
    else:
        header_names = set(
            re.findall(r'["\']([^"\']+)["\']\s*:', header_object.group("headers"))
        )
        if header_names != set(expected_headers) | {"Content-Type"}:
            errors.append(
                "frontend raw upload client headers drifted: "
                f"got {sorted(header_names)}"
            )
    if not re.search(r'method\s*:\s*"POST"', upload):
        errors.append("frontend raw upload is no longer POST")
    if not re.search(r"body\s*:\s*[A-Za-z_$][\w$]*", upload):
        errors.append("frontend raw upload no longer sends the File body")
    if "text/plain" not in upload or "text/markdown" not in upload:
        errors.append("frontend raw upload media mapping drifted")
    if "FormData" in client_source:
        errors.append("frontend upload must not use multipart FormData")
    return errors


def _check_response_schemas(openapi: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    schemas = openapi.get("components", {}).get("schemas", {})
    for name, expected in RESPONSE_SCHEMAS.items():
        schema = schemas.get(name)
        if not isinstance(schema, dict):
            errors.append(f"missing frontend response schema: {name}")
            continue
        actual_fields = frozenset(schema.get("properties", {}))
        if actual_fields != expected.fields:
            errors.append(
                f"frontend response fields drifted: {name}: "
                f"expected {sorted(expected.fields)}, got {sorted(actual_fields)}"
            )
        actual_required = frozenset(schema.get("required", ()))
        expected_required = expected.fields - expected.optional
        if actual_required != expected_required:
            errors.append(
                f"frontend required response fields drifted: {name}: "
                f"expected {sorted(expected_required)}, got {sorted(actual_required)}"
            )
        if schema.get("additionalProperties") is not False:
            errors.append(f"frontend response schema is no longer strict: {name}")
    return errors


def _check_enums(openapi: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    schemas = openapi.get("components", {}).get("schemas", {})
    for expected in ENUMS:
        schema = schemas.get(expected.schema)
        if not isinstance(schema, dict):
            errors.append(f"missing frontend enum schema: {expected.schema}")
            continue
        node = schema
        label = expected.schema
        if expected.field is not None:
            node = schema.get("properties", {}).get(expected.field, {})
            label = f"{label}.{expected.field}"
        actual = _schema_values(openapi, node)
        if actual != expected.values:
            errors.append(
                f"frontend enum drifted: {label}: "
                f"expected {sorted(expected.values, key=str)}, "
                f"got {sorted(actual, key=str)}"
            )
    return errors


def _schema_values(
    openapi: dict[str, Any], node: dict[str, Any], seen: frozenset[str] = frozenset()
) -> frozenset[object]:
    reference = node.get("$ref")
    if isinstance(reference, str):
        prefix = "#/components/schemas/"
        if not reference.startswith(prefix) or reference in seen:
            return frozenset()
        target = reference.removeprefix(prefix)
        schema = openapi.get("components", {}).get("schemas", {}).get(target, {})
        return _schema_values(openapi, schema, seen | {reference})
    values: set[object] = set(node.get("enum", ()))
    if "const" in node:
        values.add(node["const"])
    for keyword in ("anyOf", "oneOf", "allOf"):
        for child in node.get(keyword, ()):
            if isinstance(child, dict):
                values.update(_schema_values(openapi, child, seen))
    return frozenset(values)


def _check_retrieval_request(
    openapi: dict[str, Any], client_source: str
) -> list[str]:
    errors: list[str] = []
    schema = (
        openapi.get("components", {})
        .get("schemas", {})
        .get("RetrievalQueryRequest", {})
    )
    expected_fields = {
        "knowledge_base_id",
        "query",
        "top_k",
        "strategy",
        "rerank",
        "include_debug",
    }
    if set(schema.get("properties", {})) != expected_fields:
        errors.append("frontend retrieval request fields drifted in OpenAPI")
    if set(schema.get("required", ())) != {"knowledge_base_id", "query"}:
        errors.append("frontend retrieval required request fields drifted")
    if schema.get("additionalProperties") is not False:
        errors.append("frontend retrieval request no longer rejects extra filters")
    properties = schema.get("properties", {})
    defaults = {
        "top_k": 10,
        "strategy": "exact_vector",
        "rerank": False,
        "include_debug": False,
    }
    for field, expected in defaults.items():
        if properties.get(field, {}).get("default") != expected:
            errors.append(f"frontend retrieval default drifted: {field}")
    top_k = properties.get("top_k", {})
    if top_k.get("minimum") != 1 or top_k.get("maximum") != 100:
        errors.append("frontend retrieval top_k bounds drifted")
    operation_schema = (
        openapi.get("paths", {})
        .get("/api/v1/retrieval/query", {})
        .get("post", {})
        .get("requestBody", {})
        .get("content", {})
        .get("application/json", {})
        .get("schema", {})
        .get("$ref")
    )
    if operation_schema != "#/components/schemas/RetrievalQueryRequest":
        errors.append("frontend retrieval operation request schema drifted")

    try:
        method = _method_body(client_source, "queryRetrievalDebug")
    except ValueError as error:
        errors.append(str(error))
        return errors
    body_match = re.search(
        r"body\s*:\s*JSON\.stringify\(\s*\{(?P<body>.*?)\}\s*\)",
        method,
        re.DOTALL,
    )
    if body_match is None:
        errors.append("frontend retrieval debug JSON body is missing")
        return errors
    actual_fields: dict[str, str] = {}
    for item in body_match.group("body").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, value = item.split(":", 1)
            actual_fields[name.strip()] = re.sub(r"\s+", "", value)
        else:
            actual_fields[item] = item
    expected_client_keys = {
        "knowledge_base_id",
        "query",
        "top_k",
        "strategy",
        "rerank",
        "include_debug",
    }
    fixed_client_fields = {
        "strategy": '"exact_vector"',
        "rerank": "false",
        "include_debug": "true",
    }
    dynamic_fields_are_identifiers = all(
        re.fullmatch(r"[A-Za-z_$][\w$]*", actual_fields.get(field, ""))
        for field in ("knowledge_base_id", "query", "top_k")
    )
    if (
        set(actual_fields) != expected_client_keys
        or any(actual_fields.get(key) != value for key, value in fixed_client_fields.items())
        or not dynamic_fields_are_identifiers
    ):
        errors.append(
            "frontend retrieval debug body drifted: "
            f"expected fixed fields {fixed_client_fields} and only bounded inputs, "
            f"got {actual_fields}"
        )
    return errors


def _check_client_methods(client_source: str) -> list[str]:
    errors: list[str] = []
    lowered_source = client_source.lower()
    for header in (
        "authorization",
        "x-auth-request-user",
        "x-client-id",
        "x-forwarded-user",
        "x-principal-id",
        "x-workspace-id",
    ):
        if header in lowered_source:
            errors.append(f"frontend client declares forbidden identity header: {header}")
    if 'credentials: "omit"' not in client_source:
        errors.append("frontend client no longer omits browser credentials")

    for name, (transport, route_pattern) in CLIENT_METHODS.items():
        try:
            body = _method_body(client_source, name)
        except ValueError as error:
            errors.append(str(error))
            continue
        if re.search(route_pattern, body, re.DOTALL) is None:
            errors.append(f"frontend client route drifted: {name}")
        method = re.search(r'method\s*:\s*"([A-Z]+)"', body)
        if transport == "POST" and (method is None or method.group(1) != "POST"):
            errors.append(f"frontend client method drifted: {name} must use POST")
        if transport == "GET" and method is not None:
            errors.append(f"frontend client method drifted: {name} must use GET")
        if transport == "EVENTSOURCE":
            for event_name in ("answer.completed", "run.failed"):
                if f'addEventListener("{event_name}"' not in body:
                    errors.append(
                        f"frontend SSE event listener drifted: {name} {event_name}"
                    )

    literals = frozenset(
        re.sub(r"\$\{[^}]+\}", "{parameter}", match.group("value"))
        for match in re.finditer(
            r'(?P<quote>["`])(?P<value>/(?:knowledge-bases|documents|indexing-jobs|chat|retrieval)[^"`]*)'
            r'(?P=quote)',
            client_source,
        )
    )
    if literals != EXPECTED_CLIENT_ROUTE_LITERALS:
        errors.append(
            "frontend client public route literals drifted: "
            f"expected {sorted(EXPECTED_CLIENT_ROUTE_LITERALS)}, got {sorted(literals)}"
        )
    return errors


def _method_body(source: str, name: str) -> str:
    declaration = re.search(
        rf"(?m)^\s*(?:private\s+)?(?:async\s+)?{re.escape(name)}\s*\(", source
    )
    if declaration is None:
        raise ValueError(f"frontend client method is missing: {name}")
    open_parenthesis = source.find("(", declaration.start())
    close_parenthesis = _matching_delimiter(source, open_parenthesis, "(", ")")
    open_brace = source.find("{", close_parenthesis + 1)
    if open_brace < 0:
        raise ValueError(f"frontend client method body is missing: {name}")
    close_brace = _matching_delimiter(source, open_brace, "{", "}")
    return source[open_brace + 1 : close_brace]


def _matching_delimiter(
    source: str, start: int, opening: str, closing: str
) -> int:
    depth = 0
    quote: str | None = None
    escaped = False
    index = start
    while index < len(source):
        character = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            index += 1
            continue
        if character in {'"', "'", "`"}:
            quote = character
        elif character == "/" and following == "/":
            newline = source.find("\n", index + 2)
            index = len(source) if newline < 0 else newline
            continue
        elif character == "/" and following == "*":
            end = source.find("*/", index + 2)
            if end < 0:
                raise ValueError("unterminated comment in frontend client")
            index = end + 2
            continue
        elif character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise ValueError("unterminated delimiter in frontend client")


def main(
    openapi_path: Path = DEFAULT_OPENAPI,
    client_path: Path = DEFAULT_CLIENT,
) -> int:
    openapi = json.loads(openapi_path.read_text(encoding="utf-8"))
    client_source = client_path.read_text(encoding="utf-8")
    errors = check_frontend_contract(openapi, client_source)
    if errors:
        for error in errors:
            print(f"frontend API contract error: {error}", file=sys.stderr)
        return 1
    print(
        "Frontend API contract check passed: W01 public subset and client are current"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
