#!/usr/bin/env python3
"""Probe every required evaluation provider using fixed, non-corpus inputs.

The caller supplies an already-running isolated host runtime.  This tool never
starts or stops dependencies, mutates a model profile, or retains model output,
vectors, prompts, credentials, or corpus content.  Its owner-only checkpoint
records a completed provider call before moving to the next provider, so a
rerun resumes without repeating successful probes.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
from uuid import UUID

from apps.api.dependencies import _validate_model_profile
from apps.worker.dependencies import build_worker_dependencies
from rag_kb.adapters.model_secrets.local import LocalModelSecretStore
from rag_kb.answering.agent import _tools
from rag_kb.domain import (
    ChatModelExecutionError,
    ChatModelMessage,
    ChatModelRequest,
    ChatToolChoice,
    ModelKind,
    ModelValidationStatus,
)
from rag_kb.services.model_settings import ModelProfileValidationError
from rag_kb.uow import TransactionMode, execute_in_transaction
from tools.evaluation_campaign_state import write_private_json
from tools.evaluation_campaign_state import digest
from tools.evaluation_resilience import (
    HEARTBEAT_INTERVAL_SECONDS,
    PROVIDER_CASE_RETRY_POLICY,
    RESILIENCE_POLICY,
    RESILIENCE_POLICY_SHA256,
    is_retryable_provider_failure,
    seconds_until,
    timestamp_after,
)
from tools.evaluation_runtime import (
    EvaluationRuntimeError,
    canonical_evaluation_runtime_manifest,
    load_evaluation_runtime,
)


CONFIRM = "RUN_EVALUATION_PROVIDER_SMOKE"
SCHEMA = "evaluation_provider_smoke_v1"
EXPECTED_MODELS = {
    "chat": "mimo-v2.5",
    "text_embedding": "qwen3.7-text-embedding",
    "multimodal_embedding": "tongyi-embedding-vision-flash-2026-03-06",
}


class ProviderSmokeError(RuntimeError):
    """Stable, content-safe provider smoke failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-runtime",
        type=Path,
        default=canonical_evaluation_runtime_manifest(),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--chat-profile-revision-id", type=UUID, required=True)
    parser.add_argument(
        "--text-embedding-profile-revision-id", type=UUID, required=True
    )
    parser.add_argument(
        "--multimodal-embedding-profile-revision-id", type=UUID, required=True
    )
    parser.add_argument("--confirm")
    return parser


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": SCHEMA,
            "status": "started",
            "providers": {},
            "attempts": {},
            "steps": [],
        }
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise ProviderSmokeError("provider_smoke_checkpoint_permissions_invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ProviderSmokeError("provider_smoke_checkpoint_json_invalid") from error
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != SCHEMA
        or not isinstance(value.get("providers"), dict)
        or not isinstance(value.get("attempts", {}), dict)
        or not isinstance(value.get("steps"), list)
    ):
        raise ProviderSmokeError("provider_smoke_checkpoint_schema_invalid")
    return value


def _expected_binding(arguments: argparse.Namespace, runtime) -> dict[str, Any]:
    return {
        "runtime_build_revision": runtime.build_revision,
        "chat_profile_revision_id": str(arguments.chat_profile_revision_id),
        "text_embedding_profile_revision_id": str(
            arguments.text_embedding_profile_revision_id
        ),
        "multimodal_embedding_profile_revision_id": str(
            arguments.multimodal_embedding_profile_revision_id
        ),
        "models": dict(EXPECTED_MODELS),
        "resilience_policy_sha256": RESILIENCE_POLICY_SHA256,
    }


def _bind_checkpoint(
    checkpoint: dict[str, Any], binding: dict[str, Any]
) -> None:
    existing = checkpoint.get("binding")
    if existing is not None and (
        existing != binding or checkpoint.get("binding_sha256") != digest(binding)
    ):
        raise ProviderSmokeError("provider_smoke_checkpoint_binding_changed")
    if existing is None and checkpoint.get("providers"):
        raise ProviderSmokeError("provider_smoke_checkpoint_binding_missing")
    existing_policy = checkpoint.get("resilience_policy")
    if existing_policy is not None and existing_policy != RESILIENCE_POLICY:
        raise ProviderSmokeError("provider_smoke_resilience_policy_changed")
    checkpoint["binding"] = binding
    checkpoint["binding_sha256"] = digest(binding)
    checkpoint["resilience_policy"] = RESILIENCE_POLICY
    checkpoint.setdefault("attempts", {})


def _step(checkpoint: dict[str, Any], name: str, status: str) -> None:
    checkpoint["steps"].append(
        {"name": name, "status": status, "at": datetime.now(UTC).isoformat()}
    )


def _safe_failure_summary(error: BaseException) -> dict[str, Any]:
    result: dict[str, Any] = {"type": type(error).__name__}
    if isinstance(error, ProviderSmokeError):
        result["code"] = error.code
        return result
    if isinstance(error, ModelProfileValidationError):
        result["code"] = error.error_code
        return result
    if not isinstance(error, ChatModelExecutionError):
        return result
    result["code"] = error.code.value
    diagnostic = {
        key: error.diagnostic[key]
        for key in ("check", "http_status", "retryable")
        if isinstance(error.diagnostic.get(key), (str, int, bool))
    }
    if diagnostic:
        result["diagnostic"] = diagnostic
    return result


async def _profile_bundle(dependencies, *, revision_id: UUID, kind: ModelKind):
    async def load(unit_of_work):
        return await unit_of_work.model_settings.get_profile_revision(revision_id)

    bundle = await execute_in_transaction(
        dependencies.unit_of_work,
        load,
        mode=TransactionMode.REPEATABLE_READ_ONLY,
    )
    if (
        bundle is None
        or bundle.profile.kind is not kind
        or not bundle.profile.enabled
        or not bundle.provider.enabled
        or bundle.current_revision.validation_status
        is not ModelValidationStatus.VALID
    ):
        raise ProviderSmokeError("provider_smoke_profile_not_ready")
    return bundle


async def _probe_chat(dependencies, *, revision_id: UUID) -> dict[str, Any]:
    """Use the same required-tool protocol as the production Agent."""

    started = datetime.now(UTC)
    response = await dependencies.chat_model_adapter.complete(
        ChatModelRequest(
            messages=(
                ChatModelMessage(
                    "system",
                    "You are an evaluation provider health probe. "
                    "Call exactly one supplied tool.",
                ),
                ChatModelMessage("user", "evaluation provider smoke"),
            ),
            # A valid search tool payload needs more than a tiny completion
            # allowance; keep this well below normal ChatRun limits.
            max_output_tokens=512,
            model_profile_revision_id=revision_id,
            tools=_tools(adaptive=False),
            tool_choice=ChatToolChoice.REQUIRED,
        )
    )
    if response.model != EXPECTED_MODELS["chat"]:
        raise ProviderSmokeError("provider_smoke_chat_model_mismatch")
    if not response.tool_calls:
        raise ProviderSmokeError("provider_smoke_chat_no_tool_call")
    if len(response.tool_calls) != 1:
        raise ProviderSmokeError("provider_smoke_chat_protocol_invalid")
    return {
        "model": EXPECTED_MODELS["chat"],
        "duration_ms": int((datetime.now(UTC) - started).total_seconds() * 1000),
        "validation_call_count": 1,
        "usage": {
            key: value
            for key, value in response.usage.items()
            if isinstance(key, str)
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        },
    }


async def _probe_profile(
    dependencies,
    *,
    provider: str,
    revision_id: UUID,
    kind: ModelKind,
) -> dict[str, Any]:
    bundle = await _profile_bundle(
        dependencies, revision_id=revision_id, kind=kind
    )
    expected = EXPECTED_MODELS[provider]
    if bundle.current_revision.model != expected:
        raise ProviderSmokeError("provider_smoke_model_mismatch")
    if kind is ModelKind.CHAT:
        return await _probe_chat(dependencies, revision_id=revision_id)
    secret_store = LocalModelSecretStore(dependencies.settings.model_secrets.root_path)
    try:
        api_key = await asyncio.to_thread(
            secret_store.read, bundle.provider_revision.secret_reference
        )
    except (OSError, ValueError) as error:
        raise ProviderSmokeError("provider_smoke_secret_unavailable") from error
    started = datetime.now(UTC)
    snapshot = await _validate_model_profile(bundle, api_key)
    observation: dict[str, Any] = {
        "model": expected,
        "duration_ms": int((datetime.now(UTC) - started).total_seconds() * 1000),
    }
    if snapshot is None:
        raise ProviderSmokeError("provider_smoke_embedding_snapshot_missing")
    observation["dimension"] = snapshot.selected_dimension
    observation["validation_call_count"] = 1
    return observation


async def _checkpointed_probe(
    arguments: argparse.Namespace,
    checkpoint: dict[str, Any],
    *,
    provider: str,
    operation,
) -> dict[str, Any]:
    task = asyncio.create_task(operation())
    try:
        while True:
            done, _pending = await asyncio.wait(
                {task}, timeout=HEARTBEAT_INTERVAL_SECONDS
            )
            if done:
                return await task
            attempt = checkpoint["attempts"][provider]
            attempt["heartbeat_at"] = datetime.now(UTC).isoformat()
            checkpoint["heartbeat_at"] = attempt["heartbeat_at"]
            write_private_json(arguments.checkpoint, checkpoint)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def _wait_for_retry(
    arguments: argparse.Namespace,
    checkpoint: dict[str, Any],
    *,
    provider: str,
) -> None:
    while True:
        attempt = checkpoint["attempts"][provider]
        remaining = seconds_until(attempt.get("next_retry_at"))
        if remaining <= 0:
            return
        await asyncio.sleep(min(HEARTBEAT_INTERVAL_SECONDS, remaining))
        now = datetime.now(UTC).isoformat()
        attempt["heartbeat_at"] = now
        checkpoint["heartbeat_at"] = now
        write_private_json(arguments.checkpoint, checkpoint)


async def _run(arguments: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    checkpoint = _load_checkpoint(arguments.checkpoint)
    runtime = load_evaluation_runtime(
        arguments.evaluation_runtime,
        require_adaptive_graph=False,
        allow_canonical_checkout=True,
    )
    _bind_checkpoint(checkpoint, _expected_binding(arguments, runtime))
    if checkpoint.get("status") == "completed":
        return 0, {"status": "completed", "resumed": True}
    checkpoint.pop("failure_code", None)
    checkpoint.pop("last_failure", None)
    dependencies = None
    try:
        checkpoint["status"] = "running"
        _step(checkpoint, "runtime_loaded", "ok")
        write_private_json(arguments.checkpoint, checkpoint)
        dependencies = build_worker_dependencies(
            env_file=runtime.env_file,
            worker_id="evaluation-provider-smoke",
        )
        await dependencies.check_readiness()
        _step(checkpoint, "dependencies_ready", "ok")
        write_private_json(arguments.checkpoint, checkpoint)
        probes = (
            (
                "chat",
                arguments.chat_profile_revision_id,
                ModelKind.CHAT,
            ),
            (
                "text_embedding",
                arguments.text_embedding_profile_revision_id,
                ModelKind.TEXT_EMBEDDING,
            ),
            (
                "multimodal_embedding",
                arguments.multimodal_embedding_profile_revision_id,
                ModelKind.MULTIMODAL_EMBEDDING,
            ),
        )
        for provider, revision_id, kind in probes:
            if (
                checkpoint["providers"].get(provider, {}).get("status")
                == "completed"
            ):
                continue
            attempt = checkpoint["attempts"].setdefault(
                provider,
                {"status": "pending", "attempt_count": 0},
            )
            if attempt.get("status") == "retry_wait":
                await _wait_for_retry(arguments, checkpoint, provider=provider)
            while True:
                attempt_count = int(attempt.get("attempt_count", 0)) + 1
                attempt.update(
                    {
                        "status": "running",
                        "attempt_count": attempt_count,
                        "started_at": datetime.now(UTC).isoformat(),
                    }
                )
                attempt.pop("next_retry_at", None)
                _step(checkpoint, provider, "started")
                checkpoint["status"] = "running"
                write_private_json(arguments.checkpoint, checkpoint)
                try:
                    observation = await _checkpointed_probe(
                        arguments,
                        checkpoint,
                        provider=provider,
                        operation=lambda: _probe_profile(
                            dependencies,
                            provider=provider,
                            revision_id=revision_id,
                            kind=kind,
                        ),
                    )
                except Exception as error:
                    if (
                        not is_retryable_provider_failure(error)
                        or attempt_count >= PROVIDER_CASE_RETRY_POLICY.max_attempts
                    ):
                        raise
                    failure = _safe_failure_summary(error)
                    delay = PROVIDER_CASE_RETRY_POLICY.delay_after(attempt_count)
                    next_retry_at = timestamp_after(delay)
                    attempt.update(
                        {
                            "status": "retry_wait",
                            "last_failure": failure,
                            "last_retry_at": datetime.now(UTC).isoformat(),
                            "next_retry_at": next_retry_at,
                        }
                    )
                    checkpoint["status"] = "retry_wait"
                    _step(checkpoint, provider, "retry_wait")
                    write_private_json(arguments.checkpoint, checkpoint)
                    print(
                        json.dumps(
                            {
                                "event": "provider_smoke_retry_scheduled",
                                "provider": provider,
                                "attempt": attempt_count,
                                "failure_code": failure.get("code"),
                                "retry_delay_seconds": delay,
                                "next_retry_at": next_retry_at,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    await _wait_for_retry(
                        arguments, checkpoint, provider=provider
                    )
                    continue
                break
            attempt["status"] = "completed"
            attempt["completed_at"] = datetime.now(UTC).isoformat()
            attempt.pop("next_retry_at", None)
            checkpoint["providers"][provider] = {
                "status": "completed",
                **observation,
            }
            _step(checkpoint, provider, "ok")
            write_private_json(arguments.checkpoint, checkpoint)
        checkpoint["status"] = "completed"
        _step(checkpoint, "runner", "ok")
        checkpoint_sha256 = write_private_json(arguments.checkpoint, checkpoint)
        return 0, {"status": "completed", "checkpoint_sha256": checkpoint_sha256}
    except BaseException as error:
        failure = _safe_failure_summary(error)
        checkpoint["status"] = "failed"
        checkpoint["failure_code"] = failure["type"]
        checkpoint["last_failure"] = failure
        _step(checkpoint, "runner", "failed")
        checkpoint_sha256 = write_private_json(arguments.checkpoint, checkpoint)
        return 2, {
            "status": "failed",
            "failure_code": failure["type"],
            "error_code": failure.get("code"),
            "checkpoint_sha256": checkpoint_sha256,
        }
    finally:
        if dependencies is not None:
            await dependencies.close()


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.confirm != CONFIRM:
        raise SystemExit(f"--confirm must equal {CONFIRM}")
    try:
        exit_code, result = asyncio.run(_run(arguments))
    except (EvaluationRuntimeError, ProviderSmokeError) as error:
        print(json.dumps({"status": "failed", "failure_code": type(error).__name__}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
