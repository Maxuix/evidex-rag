#!/usr/bin/env python3
"""Run bounded, secret-safe capability probes against P1A model providers."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import math
import os
import socket
import ssl
import stat
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHAT_MODEL = "deepseek-v4-flash"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIMENSION = 1024
TIMEOUT_SECONDS = 30
MAX_CONCURRENCY = 2
ENV_NAMES = (
    "RAG_CHAT_BASE_URL",
    "RAG_CHAT_API_KEY",
    "RAG_EMBEDDING_BASE_URL",
    "RAG_EMBEDDING_API_KEY",
)


def tls_context() -> tuple[ssl.SSLContext, dict[str, str]]:
    """Build a verifying context, preferring the installed certifi CA bundle."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context(), {"source": "system_default"}
    return ssl.create_default_context(cafile=certifi.where()), {
        "source": "certifi",
        "version": importlib.metadata.version("certifi"),
    }


TLS_CONTEXT, TLS_CA = tls_context()


class ProbeFailure(RuntimeError):
    """A sanitized capability-probe failure."""

    def __init__(self, stage: str, category: str, status: int | None = None):
        self.stage = stage
        self.category = category
        self.status = status
        suffix = f" status={status}" if status is not None else ""
        super().__init__(f"{stage}: {category}{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file", type=Path, default=Path("/tmp/rag-provider.env")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("verification/providers/verification-report-v1.0.json"),
    )
    return parser.parse_args()


def load_env(path: Path) -> dict[str, str]:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ProbeFailure("configuration", "env_file_permissions_too_broad")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ProbeFailure("configuration", "malformed_env_line")
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name in ENV_NAMES:
            values[name] = value
    missing = [name for name in ENV_NAMES if not values.get(name)]
    if missing:
        raise ProbeFailure("configuration", "required_value_missing")
    return values


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def endpoint_configuration_hash(base_url: str, model: str, settings: dict[str, Any]) -> str:
    return canonical_hash(
        {
            "normalized_base_url": base_url.rstrip("/"),
            "model": model,
            "settings": settings,
        }
    )


def request_json(
    *,
    stage: str,
    base_url: str,
    path: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int, float]:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    body = None
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "rag-provider-baseline/1.0",
    }
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(
            request, timeout=TIMEOUT_SECONDS, context=TLS_CONTEXT
        ) as response:
            status = response.status
            decoded = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ProbeFailure(stage, "http_error", exc.code) from None
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, ssl.SSLCertVerificationError):
            category = "tls_certificate_verification_error"
        elif isinstance(reason, socket.gaierror):
            category = "dns_resolution_error"
        elif isinstance(reason, (TimeoutError, socket.timeout)):
            category = "network_timeout_error"
        elif isinstance(reason, ConnectionRefusedError):
            category = "connection_refused"
        else:
            category = f"network_error_{type(reason).__name__}"
        raise ProbeFailure(stage, category) from None
    except (TimeoutError, json.JSONDecodeError):
        raise ProbeFailure(stage, "timeout_or_invalid_json") from None
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    if status < 200 or status >= 300 or not isinstance(decoded, dict):
        raise ProbeFailure(stage, "invalid_success_response", status)
    return decoded, status, latency_ms


def usage_summary(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {"present": False, "numeric_fields": []}
    numeric_fields = sorted(
        key for key, value in usage.items() if isinstance(value, (int, float))
    )
    return {"present": True, "numeric_fields": numeric_fields}


def chat_payload(*, concurrency_probe: bool = False) -> dict[str, Any]:
    marker = "concurrency" if concurrency_probe else "structured"
    return {
        "model": CHAT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return only one JSON object with string field status and "
                    "integer field value. Do not include markdown."
                ),
            },
            {
                "role": "user",
                "content": f"JSON capability check: {marker}. Use status ok and value 1.",
            },
        ],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "max_tokens": 64,
        "stream": False,
    }


def probe_chat(values: dict[str, str]) -> dict[str, Any]:
    base_url = values["RAG_CHAT_BASE_URL"]
    api_key = values["RAG_CHAT_API_KEY"]
    models, model_status, model_latency = request_json(
        stage="chat_model_list",
        base_url=base_url,
        path="models",
        api_key=api_key,
    )
    model_ids = {
        item.get("id")
        for item in models.get("data", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if CHAT_MODEL not in model_ids:
        raise ProbeFailure("chat_model_list", "requested_model_absent")

    response, status, latency = request_json(
        stage="chat_structured_output",
        base_url=base_url,
        path="chat/completions",
        api_key=api_key,
        payload=chat_payload(),
    )
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
        parsed = json.loads(content)
        valid_shape = (
            isinstance(parsed, dict)
            and parsed.get("status") == "ok"
            and parsed.get("value") == 1
        )
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        valid_shape = False
        choice = {}
    if not valid_shape:
        raise ProbeFailure("chat_structured_output", "schema_validation_failed")

    def concurrent_call() -> float:
        _, _, call_latency = request_json(
            stage="chat_concurrency",
            base_url=base_url,
            path="chat/completions",
            api_key=api_key,
            payload=chat_payload(concurrency_probe=True),
        )
        return call_latency

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        latencies = list(pool.map(lambda _: concurrent_call(), range(MAX_CONCURRENCY)))

    settings = {
        "response_format": "json_object",
        "thinking": "disabled",
        "stream": False,
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_retries": 2,
        "max_concurrency": MAX_CONCURRENCY,
    }
    return {
        "provider_id": "deepseek",
        "logical_endpoint_id": "deepseek-official-chat",
        "requested_model": CHAT_MODEL,
        "resolved_model": response.get("model") or CHAT_MODEL,
        "endpoint_configuration_fingerprint": endpoint_configuration_hash(
            base_url, CHAT_MODEL, settings
        ),
        "model_list": {
            "http_status": model_status,
            "latency_ms": model_latency,
            "requested_model_present": True,
        },
        "structured_output": {
            "http_status": status,
            "latency_ms": latency,
            "json_parsed": True,
            "schema_valid": True,
            "finish_reason": choice.get("finish_reason"),
        },
        "usage": usage_summary(response),
        "concurrency": {
            "configured_max": MAX_CONCURRENCY,
            "attempts": MAX_CONCURRENCY,
            "passed": True,
            "latency_ms": latencies,
        },
        "timeout_seconds": TIMEOUT_SECONDS,
        "timeout_enforced_by_client": True,
    }


def l2_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def probe_embedding(values: dict[str, str]) -> dict[str, Any]:
    base_url = values["RAG_EMBEDDING_BASE_URL"]
    api_key = values["RAG_EMBEDDING_API_KEY"]
    batch_size = 10
    payload = {
        "model": EMBEDDING_MODEL,
        "input": [f"RAG capability test item {index}" for index in range(batch_size)],
        "dimensions": EMBEDDING_DIMENSION,
        "encoding_format": "float",
    }
    response, status, latency = request_json(
        stage="embedding_batch",
        base_url=base_url,
        path="embeddings",
        api_key=api_key,
        payload=payload,
    )
    data = response.get("data")
    if not isinstance(data, list) or len(data) != batch_size:
        raise ProbeFailure("embedding_batch", "unexpected_batch_size")
    indices = [item.get("index") for item in data if isinstance(item, dict)]
    if indices != list(range(batch_size)):
        raise ProbeFailure("embedding_batch", "batch_order_mismatch")
    vectors = [item.get("embedding") for item in data]
    if any(not isinstance(vector, list) for vector in vectors):
        raise ProbeFailure("embedding_batch", "missing_vector")
    dimensions = {len(vector) for vector in vectors}
    if dimensions != {EMBEDDING_DIMENSION}:
        raise ProbeFailure("embedding_batch", "dimension_mismatch")
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for vector in vectors
        for value in vector
    ):
        raise ProbeFailure("embedding_batch", "non_finite_vector_value")
    norms = [l2_norm(vector) for vector in vectors]
    normalized = all(abs(norm - 1.0) <= 0.001 for norm in norms)
    settings = {
        "dimensions": EMBEDDING_DIMENSION,
        "encoding_format": "float",
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_retries": 2,
        "max_concurrency": MAX_CONCURRENCY,
    }
    return {
        "provider_id": "alibaba-cloud-model-studio-qwen",
        "logical_endpoint_id": "alibaba-model-studio-beijing-embedding",
        "requested_model": EMBEDDING_MODEL,
        "resolved_model": response.get("model") or EMBEDDING_MODEL,
        "endpoint_configuration_fingerprint": endpoint_configuration_hash(
            base_url, EMBEDDING_MODEL, settings
        ),
        "http_status": status,
        "latency_ms": latency,
        "batch_size_tested": batch_size,
        "batch_order_preserved": True,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "finite_float_values": True,
        "l2_norm": {
            "minimum": round(min(norms), 8),
            "maximum": round(max(norms), 8),
            "provider_output_normalized": normalized,
            "tolerance": 0.001,
        },
        "usage": usage_summary(response),
        "timeout_seconds": TIMEOUT_SECONDS,
        "timeout_enforced_by_client": True,
    }


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "report_id": "provider-smoke-deepseek-qwen-v1.0",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "secret_values_recorded": False,
        "raw_base_urls_recorded": False,
        "full_prompts_or_responses_recorded": False,
        "environment_file": str(args.env_file),
        "configured": {
            "chat_model": CHAT_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "embedding_metric": "cosine",
            "vector_data_type": "float32",
            "tls_ca": TLS_CA,
        },
    }
    try:
        values = load_env(args.env_file)
        report["chat"] = probe_chat(values)
        report["embedding"] = probe_embedding(values)
        report["status"] = "passed"
    except (OSError, ProbeFailure) as exc:
        if isinstance(exc, ProbeFailure):
            report["failure"] = {
                "stage": exc.stage,
                "category": exc.category,
                "http_status": exc.status,
            }
        else:
            report["failure"] = {"stage": "configuration", "category": "io_error"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(args.report),
                "failure": report.get("failure"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
