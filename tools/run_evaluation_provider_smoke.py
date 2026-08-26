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
from rag_kb.uow import TransactionMode, UnitOfWorkPurpose, execute_in_transaction
from tools.evaluation_campaign_state import write_private_json
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
        or not isinstance(value.get("steps"), list)
    ):
        raise ProviderSmokeError("provider_smoke_checkpoint_schema_invalid")
    return value


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
        purpose=UnitOfWorkPurpose.READ_SNAPSHOT,
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


async def _run(arguments: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    checkpoint = _load_checkpoint(arguments.checkpoint)
    if checkpoint.get("status") == "completed":
        return 0, {"status": "completed", "resumed": True}
    checkpoint.pop("failure_code", None)
    checkpoint.pop("last_failure", None)
    runtime = load_evaluation_runtime(
        arguments.evaluation_runtime,
        require_adaptive_graph=False,
        allow_canonical_checkout=True,
    )
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
            _step(checkpoint, provider, "started")
            write_private_json(arguments.checkpoint, checkpoint)
            observation = await _probe_profile(
                dependencies,
                provider=provider,
                revision_id=revision_id,
                kind=kind,
            )
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
