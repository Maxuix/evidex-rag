#!/usr/bin/env python3
"""Qualify MuSiQue mini Graph candidates against one frozen host runtime."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import UUID

from apps.worker.dependencies import build_worker_dependencies
from rag_kb.adapters.graph_store.postgres import PgGraphStore
from rag_kb.auth import AuthContext
from rag_kb.domain import (
    GENERIC_GRAPH_SCHEMA_PROFILE_DIGEST,
    GENERIC_GRAPH_SCHEMA_PROFILE_KEY,
    GRAPH_EXTRACTOR_VERSION,
    RerankMode,
    RetrievalRequest,
    RetrievalStrategy,
)
from rag_kb.retrieval.service import _pack_graph_search_evidence  # noqa: SLF001
from rag_kb.uow import UnitOfWorkPurpose, execute_in_transaction
from tools.evaluation_runtime import DEFAULT_RUNTIME_MANIFEST, load_evaluation_runtime
from tools.run_adaptive_graph_r4 import _serving_chunk_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = PROJECT_ROOT / "evaluation/routing-rag-musique-mini"
MANIFEST_PATH = CORPUS_ROOT / "manifest.json"
CASES_PATH = CORPUS_ROOT / "cases.jsonl"
DOCUMENTS_PATH = CORPUS_ROOT / "documents.jsonl"
DEFAULT_OUTPUT = CORPUS_ROOT / "qualification.json"
EXPECTED_DATASET_ID = "routing-rag-musique-full-mini-v1"
EXPECTED_KB_NAME = "routing-rag-musique-full-mini-semantic-v4-graphiti-v3"
EXPECTED_SCHEMA_PROFILE_KEY = GENERIC_GRAPH_SCHEMA_PROFILE_KEY
EXPECTED_SCHEMA_PROFILE_DIGEST = GENERIC_GRAPH_SCHEMA_PROFILE_DIGEST
EXPECTED_EXTRACTOR_VERSION = GRAPH_EXTRACTOR_VERSION
CONFIRM = "QUALIFY_ROUTING_RAG_MUSIQUE_MINI"
SIMPLE_TOP_K = 10
GRAPH_EDGE_LIMIT = 16
GRAPH_SOURCE_TARGET = 12
GRAPH_SOURCE_LIMIT = 16


class QualificationError(ValueError):
    """The frozen corpus or runtime is not the expected qualification target."""


def _jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_complete(
    paths: Sequence[Sequence[str]], retrieved_document_ids: set[str]
) -> bool:
    return any(set(path).issubset(retrieved_document_ids) for path in paths)


def _chunk_document_map(
    serving_rows: Sequence[Mapping[str, Any]],
    document_rows: Sequence[Mapping[str, Any]],
) -> dict[UUID, str]:
    filename_to_document_id = {
        str(row["filename"]): str(row["document_id"]) for row in document_rows
    }
    actual_filenames = {str(row["original_filename"]) for row in serving_rows}
    if actual_filenames != set(filename_to_document_id):
        raise QualificationError("musique_serving_document_set_changed")
    return {
        UUID(str(row["index_chunk_id"])): filename_to_document_id[
            str(row["original_filename"])
        ]
        for row in serving_rows
    }


def _document_ids(
    chunk_ids: Sequence[UUID], chunk_to_document: Mapping[UUID, str]
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(chunk_to_document[item] for item in chunk_ids if item in chunk_to_document)
    )


async def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != EXPECTED_DATASET_ID:
        raise QualificationError("musique_dataset_identity_changed")
    cases = _jsonl(CASES_PATH)
    documents = _jsonl(DOCUMENTS_PATH)
    graph_cases = tuple(
        case for case in cases if case.get("route_label") == "graph_needed_candidate"
    )
    if len(graph_cases) != 10 or len(documents) != manifest.get("document_count"):
        raise QualificationError("musique_corpus_cardinality_changed")

    runtime = load_evaluation_runtime(
        arguments.evaluation_runtime,
        require_adaptive_graph=True,
        allow_canonical_checkout=True,
    )
    identity = runtime.adaptive_graph
    if identity is None:
        raise QualificationError("musique_runtime_identity_missing")
    if (
        identity.schema_profile_key != EXPECTED_SCHEMA_PROFILE_KEY
        or identity.schema_profile_digest != EXPECTED_SCHEMA_PROFILE_DIGEST
        or identity.extractor_version != EXPECTED_EXTRACTOR_VERSION
    ):
        raise QualificationError("musique_runtime_profile_identity_changed")
    dependencies = build_worker_dependencies(env_file=runtime.env_file)
    try:
        await dependencies.check_readiness()
        workspace_id = dependencies.settings.identity.workspace_id
        if workspace_id != identity.workspace_id:
            raise QualificationError("musique_workspace_identity_changed")
        async def load_knowledge_base(unit_of_work):
            return await unit_of_work.knowledge_bases.get(identity.knowledge_base_id)

        knowledge_base = await execute_in_transaction(
            dependencies.unit_of_work,
            load_knowledge_base,
            purpose=UnitOfWorkPurpose.REQUEST,
        )
        if (
            knowledge_base is None
            or knowledge_base.name != EXPECTED_KB_NAME
            or knowledge_base.active_index_revision_id != identity.index_revision_id
        ):
            raise QualificationError("musique_knowledge_base_identity_changed")
        graph_store = PgGraphStore(dependencies.database.sessions)
        build = await graph_store.get_active_graphiti_build(
            workspace_id, identity.knowledge_base_id
        )
        if (
            build is None
            or build.build_id != identity.graph_build_id
            or build.index_revision_id != identity.index_revision_id
            or build.schema_profile_key != EXPECTED_SCHEMA_PROFILE_KEY
            or build.schema_profile_digest != EXPECTED_SCHEMA_PROFILE_DIGEST
            or build.extractor_version != EXPECTED_EXTRACTOR_VERSION
        ):
            raise QualificationError("musique_graph_build_identity_changed")
        episode_uuid = await graph_store.first_graphiti_episode_uuid(
            workspace_id, identity.knowledge_base_id, build.build_id
        )
        if not await dependencies.graphiti_runtime.probe(
            build, episode_uuid=episode_uuid, require_complete=True
        ):
            raise QualificationError("musique_graph_runtime_not_ready")

        serving_rows = await _serving_chunk_rows(
            dependencies,
            workspace_id=workspace_id,
            kb_id=identity.knowledge_base_id,
            index_revision_id=identity.index_revision_id,
        )
        chunk_to_document = _chunk_document_map(serving_rows, documents)
        auth = AuthContext(
            principal_id=dependencies.settings.identity.principal_id,
            client_id=dependencies.settings.identity.client_id,
            workspace_id=workspace_id,
        )
        records: list[dict[str, Any]] = []
        for case in graph_cases:
            simple = await dependencies.retrieval_service.retrieve(
                auth,
                RetrievalRequest(
                    identity.knowledge_base_id,
                    str(case["question"]),
                    top_k=SIMPLE_TOP_K,
                    strategy=RetrievalStrategy.EXACT_VECTOR,
                    rerank_mode=RerankMode.CLASSIC,
                    include_debug=True,
                ),
            )
            simple_chunk_ids = tuple(item.index_chunk_id for item in simple.evidence)
            search_graphiti = (
                dependencies.retrieval_service._search_graphiti_candidates  # noqa: SLF001
            )
            candidates = await search_graphiti(
                workspace_id,
                identity.knowledge_base_id,
                build=build,
                index_revision_id=identity.index_revision_id,
                query=str(case["question"]),
                edge_limit=GRAPH_EDGE_LIMIT,
                rerank_mode=RerankMode.CLASSIC,
            )
            graph_evidence, new_chunk_ids = _pack_graph_search_evidence(
                candidates,
                excluded_index_chunk_ids=frozenset(simple_chunk_ids),
                source_chunk_target=GRAPH_SOURCE_TARGET,
                source_chunk_limit=GRAPH_SOURCE_LIMIT,
            )
            graph_chunk_ids = tuple(item.index_chunk_id for item in graph_evidence)
            simple_documents = _document_ids(simple_chunk_ids, chunk_to_document)
            graph_documents = _document_ids(graph_chunk_ids, chunk_to_document)
            combined = set(simple_documents) | set(graph_documents)
            paths = tuple(tuple(path) for path in case["required_paths"])
            simple_complete = _path_complete(paths, set(simple_documents))
            combined_complete = _path_complete(paths, combined)
            qualified = not simple_complete and combined_complete and bool(new_chunk_ids)
            records.append(
                {
                    "case_id": str(case["case_id"]),
                    "hop_count": int(case["hop_count"]),
                    "qualified_graph_needed": qualified,
                    "simple_path_complete": simple_complete,
                    "simple_document_ids": simple_documents,
                    "simple_plus_graph_path_complete": combined_complete,
                    "graph_document_ids": graph_documents,
                    "graph_new_chunk_count": len(new_chunk_ids),
                    "graph_hydrated_chunk_count": len(candidates.hydrated_chunk_ids),
                    "graph_path_hop_counts": {
                        "hop1": sum(path.hop_count == 1 for path in candidates.traversal.paths),
                        "hop2": sum(path.hop_count == 2 for path in candidates.traversal.paths),
                        "hop3": sum(path.hop_count == 3 for path in candidates.traversal.paths),
                    },
                }
            )
            print(
                json.dumps(
                    {
                        "event": "musique_case_qualified",
                        "case_id": case["case_id"],
                        "qualified_graph_needed": qualified,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        payload = {
            "schema": "musique_mini_host_qualification_v1",
            "dataset_id": EXPECTED_DATASET_ID,
            "created_at": datetime.now(UTC).isoformat(),
            "corpus": {
                "manifest_sha256": _sha256(MANIFEST_PATH),
                "cases_sha256": _sha256(CASES_PATH),
                "documents_sha256": _sha256(DOCUMENTS_PATH),
            },
            "runtime": {
                "workspace_id": str(workspace_id),
                "knowledge_base_id": str(identity.knowledge_base_id),
                "index_revision_id": str(identity.index_revision_id),
                "graph_build_id": str(identity.graph_build_id),
                "schema_profile_key": build.schema_profile_key,
                "schema_profile_digest": build.schema_profile_digest,
                "extractor_version": build.extractor_version,
            },
            "budgets": {
                "simple_top_k": SIMPLE_TOP_K,
                "graph_edge_limit": GRAPH_EDGE_LIMIT,
                "graph_source_chunk_target": GRAPH_SOURCE_TARGET,
                "graph_source_chunk_limit": GRAPH_SOURCE_LIMIT,
            },
            "summary": {
                "candidate_count": len(records),
                "qualified_graph_needed_count": sum(
                    bool(record["qualified_graph_needed"]) for record in records
                ),
            },
            "records": records,
        }
        arguments.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return payload["summary"]
    finally:
        await dependencies.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-runtime", type=Path, default=DEFAULT_RUNTIME_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confirm")
    return parser


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.confirm != CONFIRM:
        parser.error(f"--confirm must equal {CONFIRM}")
    try:
        result = asyncio.run(_run(arguments))
    except QualificationError as error:
        parser.exit(2, f"qualification failed: {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
