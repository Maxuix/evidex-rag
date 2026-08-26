#!/usr/bin/env python3
"""Fail closed when a large-evaluation plan lacks its supplied host runtime.

The evaluator does not start, stop, create, migrate, or otherwise manage any
runtime dependency.  It checks only the user-provided owner-only host runtime
and reports a content-safe blocking code when the runtime is unavailable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import socket
from typing import Any, Mapping

from apps.worker.dependencies import build_worker_dependencies
from tools.evaluation_campaign_state import digest
from tools.evaluation_runtime import (
    EvaluationRuntimeError,
    canonical_evaluation_runtime_manifest,
    load_evaluation_runtime,
)
from tools.provision_large_evaluation_host import BINDINGS_SCHEMA, SPECS
from tools.prepare_large_evaluation import SCHEMA_VERSION
from tools.run_adaptive_graph_r4 import R4RunnerError, _load_runtime_facts


PLAN_SCHEMA = SCHEMA_VERSION
MAX_PLAN_BYTES = 1_000_000


class LargeEvaluationRuntimeError(RuntimeError):
    """Raised for content-safe large-evaluation readiness failures."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--evaluation-runtime",
        type=Path,
        default=canonical_evaluation_runtime_manifest(),
    )
    return parser


def _load_private_plan(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise LargeEvaluationRuntimeError("large_evaluation_plan_permissions_invalid")
    if path.stat().st_size > MAX_PLAN_BYTES:
        raise LargeEvaluationRuntimeError("large_evaluation_plan_too_large")
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise LargeEvaluationRuntimeError("large_evaluation_plan_json_invalid") from error
    if not isinstance(plan, dict) or plan.get("schema_version") != PLAN_SCHEMA:
        raise LargeEvaluationRuntimeError("large_evaluation_plan_schema_invalid")
    binding = plan.get("plan_binding")
    if not isinstance(binding, dict) or binding.get("runtime_contract") != "isolated_host_native_only":
        raise LargeEvaluationRuntimeError("large_evaluation_plan_binding_invalid")
    provider = binding.get("provider_contract")
    expected_provider = {
        "provider": "OpenCode Go",
        "chat_model": "mimo-v2.5",
        "text_embedding_model": "qwen3.7-text-embedding",
        "multimodal_embedding_model": "tongyi-embedding-vision-flash-2026-03-06",
    }
    if provider != expected_provider:
        raise LargeEvaluationRuntimeError("large_evaluation_provider_contract_invalid")
    if plan.get("plan_binding_sha256") != digest(binding):
        raise LargeEvaluationRuntimeError("large_evaluation_plan_binding_digest_invalid")
    return plan


def _assert_loopback_port(*, name: str, port: int) -> None:
    """Fail with a named service cause before opening database clients."""

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            pass
    except OSError as error:
        raise LargeEvaluationRuntimeError(f"host_{name}_unreachable") from error


def _assert_resolved_chat_model(configuration: object) -> None:
    if (
        not isinstance(configuration, Mapping)
        or configuration.get("resolved_model") != "mimo-v2.5"
    ):
        raise LargeEvaluationRuntimeError("large_evaluation_chat_model_mismatch")


def _assert_all_suite_bindings(runtime_root: Path, plan: Mapping[str, Any]) -> None:
    path = runtime_root / "large-evaluation-bindings.json"
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise LargeEvaluationRuntimeError("large_evaluation_bindings_unavailable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise LargeEvaluationRuntimeError("large_evaluation_bindings_invalid") from error
    suites = value.get("suites") if isinstance(value, dict) else None
    expected = {spec.dataset_id for spec in SPECS.values()}
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != BINDINGS_SCHEMA
        or not isinstance(suites, dict)
        or set(suites) != expected
        or value.get("binding_sha256") != digest({"suites": suites})
    ):
        raise LargeEvaluationRuntimeError("large_evaluation_bindings_invalid")
    planned = {item.get("dataset_id") for item in plan["plan_binding"]["corpora"]}
    if planned != expected:
        raise LargeEvaluationRuntimeError("large_evaluation_plan_suite_set_invalid")
    for spec in SPECS.values():
        entry = suites[spec.dataset_id]
        if not isinstance(entry, dict) or not all(
            isinstance(entry.get(field), str)
            for field in ("corpus_sha256", "knowledge_base_id", "index_revision_id")
        ):
            raise LargeEvaluationRuntimeError("large_evaluation_bindings_invalid")
        if spec.graph_enabled != isinstance(entry.get("graph_build_id"), str):
            raise LargeEvaluationRuntimeError("large_evaluation_bindings_invalid")


async def check(plan_path: Path, runtime_path: Path) -> dict[str, Any]:
    """Read the plan and supplied host dependencies without mutating either."""

    plan = _load_private_plan(plan_path)
    runtime = load_evaluation_runtime(
        runtime_path,
        require_adaptive_graph=True,
        allow_canonical_checkout=True,
    )
    _assert_all_suite_bindings(runtime.runtime_root, plan)
    identity = runtime.adaptive_graph
    if identity is None:
        raise LargeEvaluationRuntimeError("large_evaluation_runtime_identity_missing")
    _assert_loopback_port(name="postgres", port=runtime.ports["postgres"])
    _assert_loopback_port(name="falkordb", port=runtime.ports["falkordb"])
    dependencies = build_worker_dependencies(
        env_file=runtime.env_file,
        worker_id="large-evaluation-readiness",
    )
    try:
        await dependencies.check_readiness()
        _, bundle, model_configuration = await _load_runtime_facts(
            dependencies,
            kb_id=identity.knowledge_base_id,
            model_revision_id=identity.answer_profile_revision_id,
        )
    finally:
        await dependencies.close()
    _assert_resolved_chat_model(model_configuration)
    # Model calls are intentionally excluded.  Provider availability is proven
    # by the one-case recoverable smoke immediately before a large run, so this
    # guard remains safe and inexpensive enough to run repeatedly.
    return {
        "status": "ready_for_smoke_revalidation",
        "plan_binding_sha256": plan.get("plan_binding_sha256"),
        "runtime": {
            "workspace_id": str(identity.workspace_id),
            "knowledge_base_id": str(identity.knowledge_base_id),
            "index_revision_id": str(identity.index_revision_id),
            "graph_build_id": str(identity.graph_build_id),
            "chat_model_profile_revision_id": str(identity.answer_profile_revision_id),
        },
        "checks": {
            "plan_binding": "ok",
            "host_dependencies": "ok",
            "chat_profile": "ok",
            "provider_call": "requires_recoverable_smoke_revalidation",
        },
        "runtime_bundle_present": bundle is not None,
    }


def main() -> int:
    arguments = _parser().parse_args()
    try:
        result = asyncio.run(check(arguments.plan, arguments.evaluation_runtime))
    except (EvaluationRuntimeError, LargeEvaluationRuntimeError, R4RunnerError) as error:
        result = {"status": "blocked", "failure_code": str(error)}
    except Exception as error:  # Content-safe dependency/readiness failure.
        result = {"status": "blocked", "failure_code": type(error).__name__}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ready_for_smoke_revalidation" else 2


if __name__ == "__main__":
    raise SystemExit(main())
