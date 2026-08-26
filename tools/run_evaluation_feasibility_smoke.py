#!/usr/bin/env python3
"""Run one recoverable, host-only RAG evaluation feasibility observation.

This deliberately small evaluator proves the mechanics needed before a long
locked evaluation: the bound host runtime is readable, a real Agent can run,
and every transition is persisted atomically.  It never creates data, starts
services, or manages Docker.  The caller supplies the already-running isolated
host runtime and may request one intentional interruption after the first
persisted observation to verify resumability.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from apps.worker.dependencies import build_worker_dependencies
from rag_kb.answering.agent import AGENT_TRACE_ARTIFACT, NativeToolCallingAgent
from rag_kb.domain import RerankMode
from rag_kb.retrieval.profile import exact_profile
from rag_kb.services.chat_execution import ChatEvidenceRetriever
from tools.evaluation_runtime import (
    EvaluationRuntimeError,
    canonical_evaluation_runtime_manifest,
    load_evaluation_runtime,
)
from tools.run_adaptive_graph_r4 import (
    _evaluator_chat_model,
    _execution_context,
    _load_runtime_facts,
)


CONFIRM = "RUN_EVALUATION_FEASIBILITY_SMOKE"
SCHEMA = "evaluation_feasibility_smoke_v1"
PROBE_CASE_ID = "simple_agent_resume_probe"


class FeasibilitySmokeError(RuntimeError):
    """Content-safe feasibility failure."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-runtime",
        type=Path,
        default=canonical_evaluation_runtime_manifest(),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--locked-output", type=Path, required=True)
    parser.add_argument(
        "--agent-timeout-seconds",
        type=float,
        default=20.0,
        help="bounded feasibility timeout; the production ChatRun deadline is unchanged",
    )
    parser.add_argument("--interrupt-after-case", action="store_true")
    parser.add_argument("--confirm")
    return parser


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_atomic(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    payload = _canonical_bytes(value)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": SCHEMA,
            "status": "started",
            "case_observations": [],
            "steps": [],
        }
    if path.stat().st_mode & 0o077:
        raise FeasibilitySmokeError("checkpoint_permissions_invalid")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
        raise FeasibilitySmokeError("checkpoint_schema_invalid")
    if not isinstance(value.get("case_observations"), list):
        raise FeasibilitySmokeError("checkpoint_observations_invalid")
    if not isinstance(value.get("steps"), list):
        raise FeasibilitySmokeError("checkpoint_steps_invalid")
    return value


def _step(state: dict[str, Any], name: str, status: str) -> None:
    state["steps"].append(
        {
            "name": name,
            "status": status,
            "at": datetime.now(UTC).isoformat(),
        }
    )


async def _run(arguments: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    checkpoint = _load_checkpoint(arguments.checkpoint)
    _step(checkpoint, "runner_started", "ok")
    _write_atomic(arguments.checkpoint, checkpoint)
    if checkpoint["case_observations"]:
        checkpoint["status"] = "completed_after_resume"
        checkpoint["metrics"] = {
            "observation_count": len(checkpoint["case_observations"]),
            "required_intermediate_fields_complete": all(
                {
                    "case_id",
                    "outcome",
                    "usage",
                    "model_call_usage",
                    "total_tokens",
                    "trace_event_count",
                    "duration_ms",
                }
                <= set(item)
                for item in checkpoint["case_observations"]
                if isinstance(item, dict)
            ),
            "model_calls_repeated_on_resume": 0,
        }
        _step(checkpoint, "resume", "ok")
        locked_sha256 = _write_atomic(arguments.locked_output, checkpoint)
        checkpoint_sha256 = _write_atomic(arguments.checkpoint, checkpoint)
        return 0, {
            "status": checkpoint["status"],
            "checkpoint_sha256": checkpoint_sha256,
            "locked_sha256": locked_sha256,
        }

    dependencies = None
    try:
        runtime = load_evaluation_runtime(
            arguments.evaluation_runtime,
            require_adaptive_graph=True,
            allow_canonical_checkout=True,
        )
        identity = runtime.adaptive_graph
        if identity is None:
            raise FeasibilitySmokeError("runtime_identity_missing")
        checkpoint["runtime"] = {
            "workspace_id": str(identity.workspace_id),
            "knowledge_base_id": str(identity.knowledge_base_id),
            "index_revision_id": str(identity.index_revision_id),
            "chat_model_profile_revision_id": str(identity.answer_profile_revision_id),
        }
        _step(checkpoint, "runtime_loaded", "ok")
        _write_atomic(arguments.checkpoint, checkpoint)

        dependencies = build_worker_dependencies(
            env_file=runtime.env_file,
            worker_id=f"evaluation-feasibility-{uuid4().hex}",
        )
        await dependencies.check_readiness()
        _step(checkpoint, "dependencies_ready", "ok")
        _write_atomic(arguments.checkpoint, checkpoint)

        _, bundle, model_configuration = await _load_runtime_facts(
            dependencies,
            kb_id=identity.knowledge_base_id,
            model_revision_id=identity.answer_profile_revision_id,
        )
        _step(checkpoint, "model_profile_resolved", "ok")
        _write_atomic(arguments.checkpoint, checkpoint)
        model, selected_model = await _evaluator_chat_model(
            dependencies, bundle=bundle, model_override=None
        )
        if selected_model.get("chat_model") != "mimo-v2.5":
            raise FeasibilitySmokeError("unexpected_chat_model")
        _step(checkpoint, "chat_provider_resolved", "ok")
        _write_atomic(arguments.checkpoint, checkpoint)

        class ProgressModel:
            def __init__(self, delegate: Any) -> None:
                self._delegate = delegate
                self._call_count = 0

            async def complete(self, *args: Any, **kwargs: Any) -> Any:
                self._call_count += 1
                step_name = f"model_call_{self._call_count}"
                _step(checkpoint, step_name, "started")
                _write_atomic(arguments.checkpoint, checkpoint)
                response = await self._delegate.complete(*args, **kwargs)
                _step(checkpoint, step_name, "ok")
                _write_atomic(arguments.checkpoint, checkpoint)
                return response

        class ProgressRetriever:
            def __init__(self, delegate: Any) -> None:
                self._delegate = delegate

            async def retrieve_query(self, *args: Any, **kwargs: Any) -> Any:
                _step(checkpoint, "simple_retrieval", "started")
                _write_atomic(arguments.checkpoint, checkpoint)
                result = await self._delegate.retrieve_query(*args, **kwargs)
                _step(checkpoint, "simple_retrieval", "ok")
                _write_atomic(arguments.checkpoint, checkpoint)
                return result

            async def search_graph_relations(self, *args: Any, **kwargs: Any) -> Any:
                _step(checkpoint, "graph_retrieval", "started")
                _write_atomic(arguments.checkpoint, checkpoint)
                result = await self._delegate.search_graph_relations(*args, **kwargs)
                _step(checkpoint, "graph_retrieval", "ok")
                _write_atomic(arguments.checkpoint, checkpoint)
                return result

            async def graph_relations_capable(self, *args: Any, **kwargs: Any) -> Any:
                return await self._delegate.graph_relations_capable(*args, **kwargs)

        class ProgressVisualPreparer:
            def __init__(self, delegate: Any) -> None:
                self._delegate = delegate

            async def run(self, *args: Any, **kwargs: Any) -> Any:
                _step(checkpoint, "visual_preparation", "started")
                _write_atomic(arguments.checkpoint, checkpoint)
                result = await self._delegate.run(*args, **kwargs)
                _step(checkpoint, "visual_preparation", "ok")
                _write_atomic(arguments.checkpoint, checkpoint)
                return result

        agent = NativeToolCallingAgent(
            ProgressModel(model),
            ProgressRetriever(ChatEvidenceRetriever(dependencies.retrieval_service)),
            ProgressVisualPreparer(dependencies.visual_evidence_preparer),
            min_cosine_similarity=dependencies.settings.retrieval.min_cosine_similarity,
            min_rerank_score=dependencies.settings.retrieval.min_rerank_score,
            cross_modal_min_cosine_similarity=(
                dependencies.settings.retrieval.cross_modal_min_cosine_similarity
            ),
        )
        context = _execution_context(
            settings=dependencies.settings,
            kb_id=identity.knowledge_base_id,
            index_revision_id=identity.index_revision_id,
            question="What information is available in this knowledge base?",
            model_configuration=model_configuration,
            rerank_mode=RerankMode.CLASSIC,
        )
        context = replace(
            context,
            retrieval_strategy=exact_profile(
                top_k=3, rerank_mode=RerankMode.CLASSIC
            ).as_dict(),
            agent_configuration={
                "version": "native_tool_calling_agent_v3",
                "budget": {"max_model_rounds": 3, "max_graph_calls": 1},
            },
        )
        _step(checkpoint, "agent_case", "started")
        _write_atomic(arguments.checkpoint, checkpoint)
        started = datetime.now(UTC)
        state = await asyncio.wait_for(
            agent.run(context), timeout=arguments.agent_timeout_seconds
        )
        trace = state.artifacts.get(AGENT_TRACE_ARTIFACT)
        answering = state.answering
        if trace is None or answering is None or answering.validated is None:
            raise FeasibilitySmokeError("agent_observation_incomplete")
        model_call_usage = [
            dict(call.usage)
            for call in answering.model_calls
            if getattr(call, "usage", None)
        ]
        checkpoint["case_observations"].append(
            {
                "case_id": PROBE_CASE_ID,
                "outcome": answering.validated.outcome.value,
                "usage": dict(trace.as_dict().get("usage", {})),
                "model_call_usage": model_call_usage,
                "total_tokens": sum(
                    int(item.get("total_tokens", 0) or 0)
                    for item in model_call_usage
                ),
                "trace_event_count": len(trace.events),
                "duration_ms": int((datetime.now(UTC) - started).total_seconds() * 1000),
            }
        )
        checkpoint["status"] = "case_persisted"
        _step(checkpoint, "agent_case", "ok")
        checkpoint_sha256 = _write_atomic(arguments.checkpoint, checkpoint)
        if arguments.interrupt_after_case:
            checkpoint["status"] = "intentionally_interrupted_after_persisted_case"
            _step(checkpoint, "intentional_interrupt", "requested")
            checkpoint_sha256 = _write_atomic(arguments.checkpoint, checkpoint)
            return 75, {
                "status": checkpoint["status"],
                "checkpoint_sha256": checkpoint_sha256,
            }
        return await _run(arguments)
    except BaseException as error:
        checkpoint["status"] = "failed"
        checkpoint["failure_code"] = type(error).__name__
        _step(checkpoint, "runner", "failed")
        checkpoint_sha256 = _write_atomic(arguments.checkpoint, checkpoint)
        return 2, {
            "status": "failed",
            "failure_code": type(error).__name__,
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
    except (EvaluationRuntimeError, FeasibilitySmokeError) as error:
        print(json.dumps({"status": "failed", "failure_code": type(error).__name__}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
