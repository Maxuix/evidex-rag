#!/usr/bin/env python3
"""Provision and score one isolated Enterprise extraction pilot arm."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any
from uuid import UUID

from apps.worker.dependencies import build_worker_dependencies
from rag_kb.adapters.graph_store.postgres import PgGraphStore
from rag_kb.domain import (
    ENTERPRISE_GRAPH_SCHEMA_PROFILE_DIGEST,
    ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY,
)
from tools.evaluation_campaign_state import write_private_json
from tools.evaluation_runtime import (
    canonical_evaluation_runtime_manifest,
    load_evaluation_runtime,
)
from tools.prepare_large_evaluation import ROOT
from tools.provision_large_evaluation_host import (
    DATASET_SPECS,
    ProvisioningSpec,
    provision,
)
from tools.run_large_evaluation import _enterprise_extraction_observation, _jsonl


CONFIRM = "RUN_ENTERPRISE_EXTRACTION_PILOT"
QUALIFICATION_CORPUS_ROOT = ROOT / "evaluation/enterprise-profile-qualification-v1"
STRESS_CORPUS_ROOT = ROOT / "evaluation/enterprise-profile-qualification-v2"
CORPUS_ROOT = QUALIFICATION_CORPUS_ROOT
_CORPORA = {
    "qualification": (QUALIFICATION_CORPUS_ROOT, 138),
    "stress": (STRESS_CORPUS_ROOT, 12),
}
_ARM_RE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument(
        "--corpus",
        choices=tuple(_CORPORA),
        default="qualification",
    )
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--chat-profile-revision-id", type=UUID, required=True)
    parser.add_argument(
        "--evaluation-runtime",
        type=Path,
        default=canonical_evaluation_runtime_manifest(),
    )
    parser.add_argument("--timeout-seconds", type=float, default=14_400.0)
    return parser


def _spec(arm: str, corpus: str = "qualification") -> tuple[str, ProvisioningSpec]:
    if _ARM_RE.fullmatch(arm) is None:
        raise ValueError("enterprise_pilot_arm_invalid")
    try:
        corpus_root, expected_document_count = _CORPORA[corpus]
    except KeyError as error:
        raise ValueError("enterprise_pilot_corpus_invalid") from error
    corpus_key = "quality" if corpus == "qualification" else corpus
    dataset_key = f"enterprise_{corpus_key}_pilot_{arm.replace('-', '_')}"
    if corpus == "qualification":
        dataset_key = f"enterprise_pilot_{arm.replace('-', '_')}"
    spec = ProvisioningSpec(
        dataset_id=f"enterprise-profile-{corpus_key}-pilot-{arm}-v1",
        corpus_root=corpus_root / "documents",
        expected_document_count=expected_document_count,
        knowledge_base_name=f"enterprise-profile-{corpus_key}-pilot-{arm}-graphiti-v4",
        confirmation=CONFIRM,
        graph_schema_key=ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY,
        graph_schema_digest=ENTERPRISE_GRAPH_SCHEMA_PROFILE_DIGEST,
    )
    return dataset_key, spec


def _gold(corpus_root: Path = CORPUS_ROOT) -> dict[str, list[dict[str, Any]]]:
    identity_controls_path = corpus_root / "identity_controls.jsonl"
    return {
        "entities": _jsonl(corpus_root / "entities.jsonl"),
        "relations": _jsonl(corpus_root / "relations.jsonl"),
        "controls": _jsonl(corpus_root / "negative_controls.jsonl"),
        "identity_controls": (
            _jsonl(identity_controls_path) if identity_controls_path.exists() else []
        ),
    }


async def _score(
    *,
    runtime_path: Path,
    binding: dict[str, Any],
    arm: str,
    corpus: str = "qualification",
) -> dict[str, Any]:
    runtime = load_evaluation_runtime(
        runtime_path,
        require_adaptive_graph=False,
        allow_canonical_checkout=True,
    )
    dependencies = build_worker_dependencies(
        env_file=runtime.env_file,
        worker_id=f"enterprise-extraction-pilot-{arm}",
    )
    try:
        await dependencies.check_readiness()
        workspace_id = dependencies.settings.identity.workspace_id
        knowledge_base_id = UUID(str(binding["knowledge_base_id"]))
        graph_build_id = UUID(str(binding["graph_build_id"]))
        index_revision_id = UUID(str(binding["index_revision_id"]))
        graph_store = PgGraphStore(dependencies.database.sessions)
        build = await graph_store.get_active_graphiti_build(
            workspace_id, knowledge_base_id
        )
        if (
            build is None
            or build.build_id != graph_build_id
            or build.index_revision_id != index_revision_id
            or build.schema_profile_key != ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY
            or build.schema_profile_digest != ENTERPRISE_GRAPH_SCHEMA_PROFILE_DIGEST
        ):
            raise RuntimeError("enterprise_pilot_graph_identity_changed")
        edges = await dependencies.graphiti_runtime.diagnostic_edges(build)
        corpus_root, _ = _CORPORA[corpus]
        gold = _gold(corpus_root)
        observation = _enterprise_extraction_observation(
            edges,
            gold["entities"],
            gold["relations"],
            gold["controls"],
            gold["identity_controls"],
        )
        return {
            "schema_version": "enterprise_extraction_pilot_v1",
            "status": "completed",
            "arm": arm,
            "corpus": corpus,
            "created_at": datetime.now(UTC).isoformat(),
            "knowledge_base_id": str(knowledge_base_id),
            "index_revision_id": str(index_revision_id),
            "graph_build_id": str(graph_build_id),
            "schema_profile_key": ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY,
            "schema_profile_digest": ENTERPRISE_GRAPH_SCHEMA_PROFILE_DIGEST,
            "observation": observation,
        }
    finally:
        await dependencies.close()


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.confirm != CONFIRM:
        raise SystemExit("enterprise_pilot_confirmation_invalid")
    dataset_key, spec = _spec(arguments.arm, arguments.corpus)
    DATASET_SPECS[dataset_key] = spec
    bindings_name = f"enterprise-profile-quality-pilot-{arguments.arm}-bindings.json"
    runtime = load_evaluation_runtime(
        arguments.evaluation_runtime,
        require_adaptive_graph=False,
        allow_canonical_checkout=True,
    )
    runtime_manifest = json.loads(runtime.manifest.read_text(encoding="utf-8"))
    original_adaptive_graph = runtime_manifest["adaptive_graph"]
    try:
        provision(
            dataset=dataset_key,
            confirmation=CONFIRM,
            timeout_seconds=arguments.timeout_seconds,
            runtime_path=arguments.evaluation_runtime,
            chat_profile_revision_id=arguments.chat_profile_revision_id,
            judge_profile_revision_id=arguments.chat_profile_revision_id,
            bindings_path=Path(bindings_name),
        )
        bindings_path = runtime.runtime_root / bindings_name
        bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
        binding = bindings["suites"][spec.dataset_id]
        report = asyncio.run(
            _score(
                runtime_path=arguments.evaluation_runtime,
                binding=binding,
                arm=arguments.arm,
                corpus=arguments.corpus,
            )
        )
        output = (
            runtime.runtime_root
            / "enterprise-profile-quality-pilot"
            / (
                f"{arguments.arm}.json"
                if arguments.corpus == "qualification"
                else f"{arguments.corpus}-{arguments.arm}.json"
            )
        )
        write_private_json(output, report)
    finally:
        restored_manifest = json.loads(runtime.manifest.read_text(encoding="utf-8"))
        restored_manifest["adaptive_graph"] = original_adaptive_graph
        write_private_json(runtime.manifest, restored_manifest)
    print(
        json.dumps(
            {
                "status": "completed",
                "arm": arguments.arm,
                "corpus": arguments.corpus,
                "report": str(output),
                "micro_precision": report["observation"]["micro_precision"]["value"],
                "micro_recall": report["observation"]["micro_recall"]["value"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
