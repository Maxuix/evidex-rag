#!/usr/bin/env python3
"""Run the bounded host observation pass for routing-rag-v3.

The runner never provisions a knowledge base or manages Docker.  It only uses
an already-bound, owner-only host test runtime. Execution performs real Simple,
Graph, Auto, and answer calls, so it requires an explicit Provider confirmation.
Its checkpoint is content-safe: no question, answer, filename, chunk body,
provider payload, exception message, or secret is persisted.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence
from uuid import UUID, uuid4

from apps.worker.dependencies import build_worker_dependencies
from rag_kb.adapters.graph_store.postgres import PgGraphStore
from rag_kb.answering.agent import AGENT_TRACE_ARTIFACT, NativeToolCallingAgent
from rag_kb.domain import (
    GRAPH_EXTRACTOR_VERSION,
    RerankMode,
    SOFTWARE_GRAPH_SCHEMA_PROFILE_DIGEST,
    SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY,
)
from rag_kb.services.chat_execution import ChatEvidenceRetriever
from tools.evaluate_adaptive_graph_route import (
    CapturingGraphSearchRetriever,
    ForcedGraphSearchChatModelPort,
    evaluate_graph_extraction,
    normalize_term,
    summarize_graph_route_trace,
)
from tools.evaluate_open_source_rag_v3 import (
    CASES_PATH,
    CORPUS_ROOT,
    GRAPH_EDGE_LIMIT,
    LAYERS,
    RELATIONS_PATH,
    V3EvaluationError,
    _canonical_bytes,
    _jsonl,
    _write_locked,
    corpus_identity,
    evaluate,
    frozen_configuration,
    observation_template,
)
from tools.evaluation_runtime import (
    EvaluationRuntimeError,
    canonical_evaluation_runtime_manifest,
    load_evaluation_runtime,
)
from tools.run_adaptive_graph_r4 import (
    ACTUAL_AUTO_CONTROLLER_MODE,
    R4RunnerError,
    _column_layers,
    _evaluator_chat_model,
    _execution_context,
    _load_runtime_facts,
    _serving_chunk_rows,
)


CONFIRM_EXTERNAL_CALLS = "RUN_ROUTING_RAG_V3_EXTERNAL_CALLS"
EXPECTED_KB_NAME = "routing-rag-v3-open-source-semantic-v4-graphiti-v3"
MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024


class V3RunnerError(RuntimeError):
    """The host observation run failed a content-safe evaluator invariant."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-runtime",
        type=Path,
        default=canonical_evaluation_runtime_manifest(),
    )
    parser.add_argument("--checkpoint-output", type=Path)
    parser.add_argument("--locked-output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm")
    return parser


def _normalized_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", "", text).replace("`", "").rstrip("。.!！")


def _nested_strings(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _nested_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _nested_strings(item)
    elif value is not None:
        yield str(value)


def relation_chunk_map(
    *,
    manifest: Mapping[str, Any],
    relations: Sequence[Mapping[str, Any]],
    serving_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Map public evidence locators to exactly one semantic-v4 serving chunk."""

    documents = manifest.get("documents")
    if not isinstance(documents, list):
        raise V3RunnerError("v3_document_manifest_invalid")
    filename_by_id = {
        str(row.get("document_id")): str(row.get("filename"))
        for row in documents
        if isinstance(row, Mapping)
    }
    if len(filename_by_id) != len(documents):
        raise V3RunnerError("v3_document_manifest_invalid")
    result: dict[str, str] = {}
    for relation in relations:
        relation_id = str(relation.get("relation_id", ""))
        filename = filename_by_id.get(str(relation.get("document_id", "")))
        evidence = _normalized_text(relation.get("evidence_text"))
        locator = relation.get("source_locator")
        section = (
            _normalized_text(locator.get("section_title"))
            if isinstance(locator, Mapping)
            else ""
        )
        if not relation_id or not filename or not evidence:
            raise V3RunnerError("v3_relation_locator_invalid")
        candidates: list[Mapping[str, Any]] = []
        for row in serving_rows:
            if str(row.get("original_filename")) != filename:
                continue
            content = _normalized_text(row.get("content"))
            if evidence in content or content in evidence:
                candidates.append(row)
        if len(candidates) > 1 and section:
            section_matches = [
                row
                for row in candidates
                if any(
                    section in _normalized_text(item)
                    for value in (
                        row.get("source_location"),
                        row.get("hierarchy"),
                        row.get("source_metadata"),
                    )
                    for item in _nested_strings(value)
                )
            ]
            if section_matches:
                candidates = section_matches
        unique = tuple(
            dict.fromkeys(str(row.get("index_chunk_id")) for row in candidates)
        )
        if len(unique) != 1:
            raise V3RunnerError("v3_relation_locator_not_unique")
        result[relation_id] = str(UUID(unique[0]))
    return result


def _relation_ids_for_chunks(
    chunk_ids: Iterable[str],
    *,
    relation_to_chunk: Mapping[str, str],
) -> list[str]:
    observed = {str(UUID(str(item))) for item in chunk_ids}
    return [
        relation_id
        for relation_id, chunk_id in relation_to_chunk.items()
        if chunk_id in observed
    ]


def _index_configuration(knowledge_base: Any, *, index_revision_id: UUID) -> dict[str, Any]:
    embedding = knowledge_base.embedding
    if (
        embedding is None
        or embedding.strategy != "text_only"
        or embedding.cross_modal is not None
        or embedding.text.profile_revision_id is None
    ):
        raise V3RunnerError("v3_embedding_configuration_invalid")
    return {
        "index_revision_id": str(index_revision_id),
        "parser_config": dict(knowledge_base.parser_config),
        "chunking_config": dict(knowledge_base.chunking_config),
        "retrieval_defaults": dict(knowledge_base.retrieval_defaults),
        "embedding": {
            "strategy": embedding.strategy,
            "profile_revision_id": str(embedding.text.profile_revision_id),
            "dimension": embedding.text.dimension,
        },
    }


def _runtime_identity(
    *,
    workspace_id: UUID,
    knowledge_base: Any,
    build: Any,
    index_revision_id: UUID,
) -> dict[str, str]:
    configuration = _index_configuration(
        knowledge_base, index_revision_id=index_revision_id
    )
    return {
        "workspace_id": str(workspace_id),
        "knowledge_base_id": str(knowledge_base.id),
        "index_revision_id": str(index_revision_id),
        "graph_build_id": str(build.build_id),
        "embedding_profile_revision_id": str(build.embedding_profile_revision_id),
        "graph_chat_profile_revision_id": str(build.chat_profile_revision_id),
        "schema_profile_key": str(build.schema_profile_key),
        "schema_profile_digest": str(build.schema_profile_digest),
        "extractor_version": str(build.extractor_version),
        "index_configuration_sha256": hashlib.sha256(
            _canonical_bytes(configuration)
        ).hexdigest(),
        "serving_document_set_sha256": str(build.serving_chunk_digest),
    }


def _graph_extraction_observation(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    all_relations = value.get("all_relations")
    precision = value.get("matched_observed_edge_precision")
    if not isinstance(all_relations, Mapping) or not isinstance(precision, Mapping):
        raise V3RunnerError("v3_graph_extraction_invalid")
    return {
        "extracted_relation_count": int(value["observed_edge_count"]),
        "exact_matched_extracted_relation_count": int(precision["numerator"]),
        "exact_gold_relation_ids": list(all_relations["complete_relation_ids"]),
        "topology_gold_relation_ids": list(all_relations["directed_relation_ids"]),
        "self_loop_count": int(value["self_loop_edge_count"]),
    }


def _forbidden_claim_hit(content: str, claims: object) -> bool:
    if not isinstance(claims, list) or any(not isinstance(item, str) for item in claims):
        raise V3RunnerError("v3_forbidden_claims_invalid")
    normalized_content = normalize_term(content)
    return any(
        normalized and normalized in normalized_content
        for claim in claims
        if (normalized := normalize_term(claim))
    )


def _checkpoint_payload(
    *,
    runtime_identity: Mapping[str, str],
    extraction: Mapping[str, Any],
) -> dict[str, Any]:
    value = observation_template()
    value["runtime_identity"] = dict(runtime_identity)
    value["graph_extraction"] = dict(extraction)
    return value


def _write_checkpoint(path: Path, value: Mapping[str, Any]) -> str:
    payload = _canonical_bytes(value)
    if len(payload) > MAX_CHECKPOINT_BYTES:
        raise V3RunnerError("v3_checkpoint_too_large")
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            os.chmod(temporary, 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(payload).hexdigest()


def _load_checkpoint(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        value = dict(expected)
        _write_checkpoint(path, value)
        return value
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_mode & 0o077
        or path.stat().st_size > MAX_CHECKPOINT_BYTES
    ):
        raise V3RunnerError("v3_checkpoint_invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V3RunnerError("v3_checkpoint_invalid") from error
    if not isinstance(value, dict):
        raise V3RunnerError("v3_checkpoint_invalid")
    for key in (
        "schema_version",
        "corpus",
        "configuration",
        "configuration_sha256",
        "runtime_identity",
        "graph_extraction",
    ):
        if value.get(key) != expected.get(key):
            raise V3RunnerError("v3_checkpoint_identity_changed")
    expected_rows = expected["case_observations"]
    rows = value.get("case_observations")
    if (
        not isinstance(rows, list)
        or len(rows) != len(expected_rows)
        or [row.get("case_id") for row in rows if isinstance(row, Mapping)]
        != [row["case_id"] for row in expected_rows]
    ):
        raise V3RunnerError("v3_checkpoint_case_set_changed")
    saw_incomplete = False
    validation_value = copy.deepcopy(value)
    for index, row in enumerate(rows):
        current = isinstance(row.get("simple_relation_ids"), list)
        if current and saw_incomplete:
            raise V3RunnerError("v3_checkpoint_completion_order_invalid")
        saw_incomplete = saw_incomplete or not current
        if not current:
            if row != expected_rows[index]:
                raise V3RunnerError("v3_checkpoint_observation_invalid")
            validation_value["case_observations"][index] = {
                "case_id": expected_rows[index]["case_id"],
                "simple_relation_ids": [],
                "graph_full_relation_ids_by_layer": {
                    layer: [] for layer in LAYERS
                },
                "graph_incremental_packed_relation_ids": [],
                "auto": {
                    "attempted": False,
                    "admitted": False,
                    "new_source_backed_evidence_count": 0,
                },
                "actual_outcome": "refused",
                "forbidden_claim_hit": False,
            }
    try:
        evaluate(validation_value)
    except V3EvaluationError as error:
        raise V3RunnerError("v3_checkpoint_observation_invalid") from error
    return value


async def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    runtime = load_evaluation_runtime(
        arguments.evaluation_runtime,
        require_adaptive_graph=True,
        allow_canonical_checkout=True,
    )
    identity = runtime.adaptive_graph
    if identity is None:
        raise V3RunnerError("v3_runtime_identity_missing")
    dependencies = build_worker_dependencies(env_file=runtime.env_file)
    try:
        await dependencies.check_readiness()
        workspace_id = dependencies.settings.identity.workspace_id
        if workspace_id != identity.workspace_id:
            raise V3RunnerError("v3_workspace_identity_changed")
        knowledge_base, bundle, model_configuration = await _load_runtime_facts(
            dependencies,
            kb_id=identity.knowledge_base_id,
            model_revision_id=identity.answer_profile_revision_id,
        )
        if (
            knowledge_base.name != EXPECTED_KB_NAME
            or knowledge_base.active_index_revision_id != identity.index_revision_id
        ):
            raise V3RunnerError("v3_knowledge_base_identity_changed")
        graph_store = PgGraphStore(dependencies.database.sessions)
        build = await graph_store.get_active_graphiti_build(
            workspace_id, identity.knowledge_base_id
        )
        if (
            build is None
            or build.build_id != identity.graph_build_id
            or build.index_revision_id != identity.index_revision_id
            or build.extractor_version != GRAPH_EXTRACTOR_VERSION
            or build.schema_profile_key != SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY
            or build.schema_profile_digest != SOFTWARE_GRAPH_SCHEMA_PROFILE_DIGEST
            or identity.schema_profile_key != SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY
            or identity.schema_profile_digest != SOFTWARE_GRAPH_SCHEMA_PROFILE_DIGEST
            or identity.extractor_version != GRAPH_EXTRACTOR_VERSION
        ):
            raise V3RunnerError("v3_graph_build_identity_changed")
        episode_uuid = await graph_store.first_graphiti_episode_uuid(
            workspace_id, identity.knowledge_base_id, build.build_id
        )
        if not await dependencies.graphiti_runtime.probe(
            build, episode_uuid=episode_uuid, require_complete=True
        ):
            raise V3RunnerError("v3_graph_runtime_not_ready")

        manifest = json.loads(
            (CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8")
        )
        cases = _jsonl(CASES_PATH)
        relations = _jsonl(RELATIONS_PATH)
        entities = _jsonl(CORPUS_ROOT / "entities.jsonl")
        serving_rows = await _serving_chunk_rows(
            dependencies,
            workspace_id=workspace_id,
            kb_id=identity.knowledge_base_id,
            index_revision_id=identity.index_revision_id,
        )
        expected_filenames = {
            str(row["filename"]) for row in manifest["documents"]
        }
        actual_filenames = {
            str(row.get("original_filename")) for row in serving_rows
        }
        if actual_filenames != expected_filenames:
            raise V3RunnerError("v3_serving_document_set_changed")
        relation_to_chunk = relation_chunk_map(
            manifest=manifest,
            relations=relations,
            serving_rows=serving_rows,
        )
        focus = tuple(
            dict.fromkeys(
                relation_id
                for case in cases
                if case["semantic_intent"] == "graph"
                for path in case["valid_paths"]
                for relation_id in path
            )
        )
        extraction = evaluate_graph_extraction(
            await dependencies.graphiti_runtime.diagnostic_edges(build),
            entity_rows=entities,
            relation_rows=relations,
            focus_relation_ids=focus,
        )
        runtime_identity = _runtime_identity(
            workspace_id=workspace_id,
            knowledge_base=knowledge_base,
            build=build,
            index_revision_id=identity.index_revision_id,
        )
        extraction_observation = _graph_extraction_observation(extraction)
        preflight = {
            "status": "preflight_ok",
            "dataset_id": corpus_identity()["dataset_id"],
            "case_count": len(cases),
            "relation_count": len(relation_to_chunk),
            "runtime_identity": runtime_identity,
            "graph_extraction": extraction_observation,
        }
        if arguments.preflight_only:
            return preflight

        expected_checkpoint = _checkpoint_payload(
            runtime_identity=runtime_identity,
            extraction=extraction_observation,
        )
        checkpoint = _load_checkpoint(
            arguments.checkpoint_output, expected_checkpoint
        )
        completed_count = sum(
            isinstance(row["simple_relation_ids"], list)
            for row in checkpoint["case_observations"]
        )
        chat_model, _ = await _evaluator_chat_model(
            dependencies, bundle=bundle, model_override=None
        )
        observing_model = ForcedGraphSearchChatModelPort(
            chat_model, controller_mode=ACTUAL_AUTO_CONTROLLER_MODE
        )
        observing_retriever = CapturingGraphSearchRetriever(
            ChatEvidenceRetriever(dependencies.retrieval_service),
            stop_after_capture=False,
        )
        retrieval_settings = dependencies.settings.retrieval
        agent = NativeToolCallingAgent(
            observing_model,
            observing_retriever,
            dependencies.visual_evidence_preparer,
            min_cosine_similarity=retrieval_settings.min_cosine_similarity,
            min_rerank_score=retrieval_settings.min_rerank_score,
            cross_modal_min_cosine_similarity=(
                retrieval_settings.cross_modal_min_cosine_similarity
            ),
        )
        for index, case in enumerate(cases[completed_count:], start=completed_count):
            case_id = str(case["case_id"])
            observing_model.reset_case()
            observing_retriever.begin_case(case_id)
            try:
                state = await agent.run(
                    _execution_context(
                        settings=dependencies.settings,
                        kb_id=identity.knowledge_base_id,
                        index_revision_id=identity.index_revision_id,
                        question=str(case["question"]),
                        model_configuration=model_configuration,
                        rerank_mode=RerankMode.CLASSIC,
                    )
                )
                capture, simple_chunk_ids = (
                    observing_retriever.finish_observed_case()
                )
            except BaseException:
                observing_retriever.abandon_case()
                raise
            trace = state.artifacts.get(AGENT_TRACE_ARTIFACT)
            if trace is None or not hasattr(trace, "as_dict"):
                raise V3RunnerError("v3_agent_trace_missing")
            route = summarize_graph_route_trace(trace.as_dict())
            if (capture is not None) != bool(route["graph_route_attempted"]):
                raise V3RunnerError("v3_auto_capture_trace_mismatch")
            answering = state.answering
            if (
                answering is None
                or answering.validated is None
                or answering.rendered is None
            ):
                raise V3RunnerError("v3_answer_observation_missing")
            if case["semantic_intent"] == "graph":
                layers, graph_metrics = await _column_layers(
                    dependencies,
                    build=build,
                    query=str(case["question"]),
                    excluded_chunk_ids=tuple(simple_chunk_ids),
                    edge_limit=GRAPH_EDGE_LIMIT,
                    rerank_mode=RerankMode.CLASSIC,
                )
                relation_layers = {
                    layer: _relation_ids_for_chunks(
                        layers[layer], relation_to_chunk=relation_to_chunk
                    )
                    for layer in LAYERS
                }
                incremental_packed_relation_ids = _relation_ids_for_chunks(
                    graph_metrics["incremental_packed_chunk_ids"],
                    relation_to_chunk=relation_to_chunk,
                )
            else:
                relation_layers = {layer: [] for layer in LAYERS}
                incremental_packed_relation_ids = []
            checkpoint["case_observations"][index] = {
                "case_id": case_id,
                "simple_relation_ids": _relation_ids_for_chunks(
                    simple_chunk_ids, relation_to_chunk=relation_to_chunk
                ),
                "graph_full_relation_ids_by_layer": relation_layers,
                "graph_incremental_packed_relation_ids": incremental_packed_relation_ids,
                "auto": {
                    "attempted": bool(route["graph_route_attempted"]),
                    "admitted": bool(route["graph_route_admitted"]),
                    "new_source_backed_evidence_count": int(
                        route["graph_new_evidence_count"]
                    ),
                },
                "actual_outcome": answering.validated.outcome.value,
                "forbidden_claim_hit": _forbidden_claim_hit(
                    answering.rendered.content, case["forbidden_claims"]
                ),
            }
            _write_checkpoint(arguments.checkpoint_output, checkpoint)
            print(
                json.dumps(
                    {
                        "event": "v3_case_completed",
                        "case_id": case_id,
                        "completed_case_count": index + 1,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        locked = evaluate(checkpoint)
        locked_sha256 = _write_locked(arguments.locked_output, locked)
        return {
            "status": "completed",
            "case_count": len(cases),
            "checkpoint_sha256": hashlib.sha256(
                _canonical_bytes(checkpoint)
            ).hexdigest(),
            "locked_sha256": locked_sha256,
            "graph_needed_case_count": locked["metrics"]["graph_benefit"][
                "graph_needed_case_count"
            ],
        }
    finally:
        await dependencies.close()


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.dry_run:
        if any(
            value is not None
            for value in (
                arguments.confirm,
                arguments.checkpoint_output,
                arguments.locked_output,
            )
        ) or arguments.preflight_only:
            parser.error("--dry-run cannot be combined with execution options")
        print(
            json.dumps(
                {
                    "status": "offline_dry_run_ok",
                    "corpus": corpus_identity(),
                    "configuration": frozen_configuration(),
                },
                sort_keys=True,
            )
        )
        return 0
    # Preflight only inspects the already-isolated runtime and never executes a
    # case or calls a Provider.  Keep the external-traffic confirmation scoped
    # to the path that can actually generate that traffic.
    if not arguments.preflight_only and arguments.confirm != CONFIRM_EXTERNAL_CALLS:
        parser.error(f"--confirm must equal {CONFIRM_EXTERNAL_CALLS}")
    if not arguments.preflight_only and (
        arguments.checkpoint_output is None or arguments.locked_output is None
    ):
        parser.error("checkpoint and locked outputs are required")
    try:
        result = asyncio.run(_run(arguments))
    except (
        EvaluationRuntimeError,
        R4RunnerError,
        V3EvaluationError,
        V3RunnerError,
    ) as error:
        parser.exit(2, f"v3 runner failed: {type(error).__name__}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
