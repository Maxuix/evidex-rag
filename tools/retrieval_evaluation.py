#!/usr/bin/env python3
"""Run and report the versioned Stage 04 exact-vector/lexical evaluation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import asyncpg

from rag_kb.adapters.model_api import OpenAICompatibleEmbeddingProvider
from rag_kb.adapters.parser.plain_text import process_plain_text
from rag_kb.adapters.vector_store import FixedPgVectorSpace, PgVectorStore
from rag_kb.auth import AuthContext, SingleWorkspaceAccessPolicy
from rag_kb.db import DatabaseProcess, create_database_resources
from rag_kb.domain import (
    EmbeddingBatch,
    EmbeddingSpaceDefinition,
    EvaluationCaseDefinition,
    EvaluationCaseResult,
    EvaluationDatasetDefinition,
    EvaluationRunDefinition,
    ParserSource,
    RetrievalQueryPlan,
    RetrievalRequest,
    RetrievalStrategy,
    validate_embedding_vector,
)
from rag_kb.retrieval import RetrievalService
from rag_kb.services import EvaluationPersistenceService
from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWorkFactory


TOOL_VERSION = "1.0"
DEFAULT_CONFIG = Path("evaluation/configs/retrieval-evaluation-v1.0.json")
BASE_URL_ENV = "RAG_KB__MODEL_PROVIDER__EMBEDDING__BASE_URL"
API_KEY_ENV = "RAG_KB__MODEL_PROVIDER__EMBEDDING__API_KEY"
RUNTIME_DSN_ENV = "RAG_KB_TEST_RUNTIME_DSN"
RUNTIME_SQLALCHEMY_DSN_ENV = "RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN"


@dataclass(frozen=True, slots=True)
class ParsedFixture:
    entry: dict[str, Any]
    filename: str
    media_type: str
    chunks: tuple[Any, ...]
    vectors: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class CaseExecution:
    case_id: str
    sample_ids: tuple[str, ...]
    ranked_chunks: tuple[dict[str, Any], ...]
    query_embedding_ms: float
    retrieval_database_ms: float
    query_plan: dict[str, Any]
    result_count: int
    error_code: str | None = None


class TimedEmbeddingProvider:
    def __init__(self, provider: OpenAICompatibleEmbeddingProvider) -> None:
        self._provider = provider
        self.last_ms = 0.0

    @property
    def embedding_space(self) -> EmbeddingSpaceDefinition:
        return self._provider.embedding_space

    @property
    def max_batch_size(self) -> int:
        return self._provider.max_batch_size

    async def embed(self, texts: tuple[str, ...]) -> EmbeddingBatch:
        started = time.perf_counter_ns()
        try:
            return await self._provider.embed(texts)
        finally:
            self.last_ms = (time.perf_counter_ns() - started) / 1_000_000


class TimedVectorStore:
    def __init__(self, store: PgVectorStore) -> None:
        self._store = store
        self.last_ms = 0.0

    async def search(self, plan, query_embedding):
        started = time.perf_counter_ns()
        try:
            return await self._store.search(plan, query_embedding)
        finally:
            self.last_ms = (time.perf_counter_ns() - started) / 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the checked-in report and its recorded input checksums",
    )
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"evaluation input escapes repository root: {value}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_cases(path: Path) -> tuple[dict[str, Any], ...]:
    cases = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    ids = [case.get("case_id") for case in cases]
    if not cases or len(ids) != len(set(ids)):
        raise ValueError("golden cases must have unique IDs")
    return cases


def load_inputs(config_path: Path, root: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    if config_path != root and root not in config_path.parents:
        raise ValueError("evaluation config escapes repository root")
    config = _load_json(config_path)
    if config.get("schema_version") != "1.0":
        raise ValueError("unsupported retrieval evaluation config")
    paths = {
        key: _resolve(root, config[key])
        for key in (
            "corpus_manifest",
            "corpus_profile",
            "golden_dataset",
            "golden_manifest",
            "quality_baseline",
            "provider_declaration",
            "lexical_report",
            "report_schema",
        )
    }
    manifest = _load_json(paths["corpus_manifest"])
    profile = _load_json(paths["corpus_profile"])
    golden_manifest = _load_json(paths["golden_manifest"])
    quality = _load_json(paths["quality_baseline"])
    provider = _load_json(paths["provider_declaration"])
    lexical = _load_json(paths["lexical_report"])
    schema = _load_json(paths["report_schema"])
    cases = _load_cases(paths["golden_dataset"])
    if (
        profile.get("manifest_sha256") != sha256(paths["corpus_manifest"])
        or golden_manifest.get("corpus_manifest_sha256")
        != sha256(paths["corpus_manifest"])
        or golden_manifest.get("dataset_sha256") != sha256(paths["golden_dataset"])
        or golden_manifest.get("evaluation_config_sha256")
        != sha256(paths["quality_baseline"])
        or golden_manifest.get("case_count") != len(cases)
        or quality.get("embedding_space_fingerprint")
        != provider.get("embedding", {}).get("embedding_space", {}).get(
            "compatibility_fingerprint"
        )
        or lexical.get("selection", {}).get("frozen_selected_strategy_id")
        != config.get("lexical_strategy_id")
        or lexical.get("selection", {}).get("status") != "confirmed"
        or schema.get("properties", {}).get("schema_version", {}).get("const")
        != "1.0"
    ):
        raise ValueError("retrieval evaluation frozen input linkage failed")
    top_k = tuple(config["top_k_values"])
    if top_k != (1, 3, 5, 10):
        raise ValueError("retrieval evaluation top_k values are not frozen")
    return {
        "config": config,
        "paths": {**paths, "evaluation_config": config_path},
        "manifest": manifest,
        "profile": profile,
        "golden_manifest": golden_manifest,
        "quality": quality,
        "provider": provider,
        "lexical": lexical,
        "schema": schema,
        "cases": cases,
    }


def _metric_values(
    cases: tuple[dict[str, Any], ...],
    ranked: dict[str, tuple[str, ...]],
    top_k_values: tuple[int, ...],
) -> dict[str, Any]:
    answerable = tuple(
        case for case in cases if case["retrieval"]["expected_relevant_sample_ids"]
    )
    expected_empty = tuple(
        case for case in cases if case["retrieval"]["expected_empty"]
    )

    def recall(case: dict[str, Any], top_k: int) -> float:
        expected = set(case["retrieval"]["expected_relevant_sample_ids"])
        observed = set(ranked[case["case_id"]][:top_k])
        return len(expected & observed) / len(expected)

    reciprocal_ranks = []
    for case in answerable:
        expected = set(case["retrieval"]["expected_relevant_sample_ids"])
        rank = next(
            (
                index
                for index, sample_id in enumerate(ranked[case["case_id"]], start=1)
                if sample_id in expected
            ),
            None,
        )
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    forbidden_count = sum(
        sample_id in set(case["retrieval"]["forbidden_sample_ids"])
        for case in cases
        for sample_id in ranked[case["case_id"]]
    )
    expected_empty_safe = sum(
        not (
            set(ranked[case["case_id"]])
            & (
                set(case["retrieval"]["expected_relevant_sample_ids"])
                | set(case["retrieval"]["forbidden_sample_ids"])
            )
        )
        for case in expected_empty
    )
    return {
        "case_count": len(cases),
        "recall_at_k": {
            str(top_k): round(
                sum(recall(case, top_k) for case in answerable) / len(answerable),
                6,
            )
            if answerable
            else 0.0
            for top_k in top_k_values
        },
        "mrr": round(statistics.fmean(reciprocal_ranks), 6)
        if reciprocal_ranks
        else 0.0,
        "empty_result_rate": round(
            sum(not ranked[case["case_id"]] for case in cases) / len(cases), 6
        )
        if cases
        else 0.0,
        "false_empty_rate": round(
            sum(not ranked[case["case_id"]] for case in answerable)
            / len(answerable),
            6,
        )
        if answerable
        else 0.0,
        "expected_empty_accuracy": round(
            expected_empty_safe / len(expected_empty), 6
        )
        if expected_empty
        else 1.0,
        "forbidden_result_count": forbidden_count,
    }


def _latency(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)

    def percentile(value: float) -> float:
        if not ordered:
            return 0.0
        index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * value + 0.999999)))
        return round(ordered[index], 3)

    return {
        "count": len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "maximum": round(max(ordered), 3) if ordered else 0.0,
    }


def _embedding_definition(declaration: dict[str, Any]) -> EmbeddingSpaceDefinition:
    embedding = declaration["embedding"]
    space = embedding["embedding_space"]
    return EmbeddingSpaceDefinition(
        provider_identity=embedding["provider_id"],
        endpoint_identity=embedding["logical_endpoint_id"],
        requested_model=embedding["requested_model"],
        resolved_model=embedding["resolved_model"],
        model_version=embedding["model_version"],
        deployment_revision=embedding["deployment_revision"],
        dimension=embedding["capabilities"]["embedding_dimension"],
        distance_metric=space["metric"],
        vector_data_type=space["vector_data_type"],
        normalization="l2",
        configuration_fingerprint=embedding["configuration_fingerprint"],
        tokenizer_fingerprint=None,
        compatibility_fingerprint=space["compatibility_fingerprint"],
    )


def _provider(inputs: dict[str, Any]) -> OpenAICompatibleEmbeddingProvider:
    base_url = os.environ.get(BASE_URL_ENV)
    api_key = os.environ.get(API_KEY_ENV)
    if not base_url or not api_key:
        raise ValueError(
            f"real embedding evaluation requires {BASE_URL_ENV} and {API_KEY_ENV}"
        )
    declaration = inputs["provider"]["embedding"]
    return OpenAICompatibleEmbeddingProvider(
        base_url=base_url,
        api_key=api_key,
        embedding_space=_embedding_definition(inputs["provider"]),
        max_batch_size=declaration["capabilities"]["max_batch_size"],
        timeout_seconds=declaration["timeout_seconds"],
        max_retries=declaration["max_retries"],
        max_concurrency=declaration["max_concurrency"],
    )


async def _parse_and_embed(
    inputs: dict[str, Any],
    provider: OpenAICompatibleEmbeddingProvider,
) -> tuple[ParsedFixture, ...]:
    config = inputs["config"]
    manifest_path = inputs["paths"]["corpus_manifest"]
    manifest_root = manifest_path.parent.resolve()
    fixtures: list[ParsedFixture] = []
    for entry in inputs["manifest"]["documents"]:
        path = (manifest_root / entry["path"]).resolve()
        if path != manifest_root and manifest_root not in path.parents:
            raise ValueError("corpus document path escapes its manifest root")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"corpus checksum changed: {entry['sample_id']}")
        media_type = "text/markdown" if path.suffix.lower() == ".md" else "text/plain"
        processed = process_plain_text(
            ParserSource(path.name, media_type, content),
            max_characters=config["parser"]["max_characters"],
            overlap_characters=config["parser"]["overlap_characters"],
            max_chunks=config["parser"]["max_chunks_per_document"],
        )
        vectors: list[tuple[float, ...]] = []
        for offset in range(0, len(processed.chunks), provider.max_batch_size):
            drafts = processed.chunks[offset : offset + provider.max_batch_size]
            batch = await provider.embed(tuple(draft.text for draft in drafts))
            if batch.model != provider.embedding_space.resolved_model:
                raise ValueError("embedding provider resolved model changed")
            for vector in batch.vectors:
                validate_embedding_vector(vector, provider.embedding_space)
                vectors.append(tuple(float(value) for value in vector))
        if len(vectors) != len(processed.chunks):
            raise ValueError("embedding provider returned incomplete corpus vectors")
        fixtures.append(
            ParsedFixture(entry, path.name, media_type, processed.chunks, tuple(vectors))
        )
    return tuple(fixtures)


def _stable_uuid(evaluation_id: str, name: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"rag-kb:{evaluation_id}:{name}")


async def _seed_database(
    connection: asyncpg.Connection,
    inputs: dict[str, Any],
    fixtures: tuple[ParsedFixture, ...],
) -> tuple[UUID, UUID, UUID]:
    evaluation_id = inputs["config"]["evaluation_id"]
    workspace_id = _stable_uuid(evaluation_id, "workspace")
    knowledge_base_id = _stable_uuid(evaluation_id, "knowledge-base")
    revision_id = _stable_uuid(evaluation_id, "index-revision")
    embedding_space_id = _stable_uuid(evaluation_id, "embedding-space")
    definition = _embedding_definition(inputs["provider"])
    by_document: dict[str, list[ParsedFixture]] = {}
    for fixture in fixtures:
        by_document.setdefault(fixture.entry["logical_document_id"], []).append(fixture)

    async with connection.transaction():
        await connection.execute(
            "INSERT INTO workspace (id, name) VALUES ($1, $2)",
            workspace_id,
            f"evaluation-{evaluation_id}",
        )
        await connection.execute(
            """
            INSERT INTO embedding_space (
                id, workspace_id, provider_identity, endpoint_identity,
                requested_model, resolved_model, model_version,
                deployment_revision, dimension, distance_metric,
                vector_data_type, normalization, configuration_fingerprint,
                tokenizer_fingerprint, compatibility_fingerprint
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15
            )
            """,
            embedding_space_id,
            workspace_id,
            definition.provider_identity,
            definition.endpoint_identity,
            definition.requested_model,
            definition.resolved_model,
            definition.model_version,
            definition.deployment_revision,
            definition.dimension,
            definition.distance_metric,
            definition.vector_data_type,
            definition.normalization,
            definition.configuration_fingerprint,
            definition.tokenizer_fingerprint,
            definition.compatibility_fingerprint,
        )
        await connection.execute(
            """
            INSERT INTO knowledge_base (
                id, workspace_id, name, source_change_seq, retrieval_defaults
            ) VALUES ($1, $2, $3, $4, $5::jsonb)
            """,
            knowledge_base_id,
            workspace_id,
            f"evaluation-{evaluation_id}",
            len(fixtures),
            json.dumps({"strategy": "exact_vector", "top_k": 10}),
        )
        await connection.execute(
            """
            INSERT INTO index_revision (
                id, workspace_id, kb_id, embedding_space_id, status,
                source_snapshot_seq, parser_config, chunking_config
            ) VALUES ($1, $2, $3, $4, 'active', $5, $6::jsonb, $7::jsonb)
            """,
            revision_id,
            workspace_id,
            knowledge_base_id,
            embedding_space_id,
            len(fixtures),
            json.dumps({"profile": inputs["config"]["parser"]["profile"]}),
            json.dumps(
                {
                    "profile": inputs["config"]["parser"]["chunking_profile"],
                    "max_characters": inputs["config"]["parser"]["max_characters"],
                    "overlap_characters": inputs["config"]["parser"][
                        "overlap_characters"
                    ],
                }
            ),
        )
        source_sequence = 0
        for logical_document_id, versions in sorted(by_document.items()):
            document_id = _stable_uuid(evaluation_id, f"document:{logical_document_id}")
            current = next(
                (item for item in versions if item.entry["is_current_version"]),
                max(versions, key=lambda item: item.entry["version"]),
            )
            deleted = current.entry["lifecycle_state"] == "deleted"
            await connection.execute(
                """
                INSERT INTO document (
                    id, workspace_id, kb_id, display_name, deleted_at
                ) VALUES ($1, $2, $3, $4, CASE WHEN $5 THEN now() ELSE NULL END)
                """,
                document_id,
                workspace_id,
                knowledge_base_id,
                logical_document_id,
                deleted,
            )
            current_version_id: UUID | None = None
            for version_number, fixture in enumerate(
                sorted(versions, key=lambda item: item.entry["version"]), start=1
            ):
                source_sequence += 1
                sample_id = fixture.entry["sample_id"]
                version_id = _stable_uuid(evaluation_id, f"version:{sample_id}")
                indexed_id = _stable_uuid(evaluation_id, f"indexed:{sample_id}")
                if fixture.entry["is_current_version"]:
                    current_version_id = version_id
                source_status = "deleted" if deleted else "available"
                await connection.execute(
                    """
                    INSERT INTO document_version (
                        id, workspace_id, kb_id, document_id, version_number,
                        source_status, checksum_sha256, storage_uri,
                        original_filename, media_type, size_bytes
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
                    version_id,
                    workspace_id,
                    knowledge_base_id,
                    document_id,
                    version_number,
                    source_status,
                    fixture.entry["sha256"],
                    f"evaluation://{sample_id}",
                    fixture.filename,
                    fixture.media_type,
                    sum(len(chunk.text.encode("utf-8")) for chunk in fixture.chunks),
                )
                await connection.execute(
                    """
                    INSERT INTO indexed_document_version (
                        id, workspace_id, kb_id, document_id, document_version_id,
                        index_revision_id, source_change_seq, build_status,
                        serving_status
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    indexed_id,
                    workspace_id,
                    knowledge_base_id,
                    document_id,
                    version_id,
                    revision_id,
                    source_sequence,
                    fixture.entry["build_status"],
                    fixture.entry["serving_status"],
                )
                for chunk, vector in zip(fixture.chunks, fixture.vectors, strict=True):
                    chunk_id = _stable_uuid(
                        evaluation_id, f"chunk:{sample_id}:{chunk.ordinal}"
                    )
                    await connection.execute(
                        """
                        INSERT INTO index_chunk (
                            id, workspace_id, kb_id, indexed_document_version_id,
                            ordinal, content, content_hash, token_count,
                            source_location, hierarchy, source_metadata
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8,
                            $9::jsonb, $10::jsonb, $11::jsonb
                        )
                        """,
                        chunk_id,
                        workspace_id,
                        knowledge_base_id,
                        indexed_id,
                        chunk.ordinal,
                        chunk.text,
                        chunk.content_sha256,
                        len(chunk.text),
                        json.dumps(
                            {
                                "start_character": chunk.start_character,
                                "end_character": chunk.end_character,
                            }
                        ),
                        json.dumps({"headings": list(chunk.heading_hierarchy)}),
                        json.dumps(
                            {
                                "evaluation_sample_id": sample_id,
                                "original_filename": fixture.filename,
                            }
                        ),
                    )
                    await connection.execute(
                        """
                        INSERT INTO vector_record_1024 (
                            workspace_id, kb_id, index_chunk_id,
                            embedding_space_id, embedding
                        ) VALUES ($1, $2, $3, $4, $5::vector)
                        """,
                        workspace_id,
                        knowledge_base_id,
                        chunk_id,
                        embedding_space_id,
                        _vector_literal(vector),
                    )
            if current_version_id is None:
                raise ValueError(f"document has no current version: {logical_document_id}")
            await connection.execute(
                "UPDATE document SET current_version_id = $1 WHERE id = $2",
                current_version_id,
                document_id,
            )
        await connection.execute(
            """
            UPDATE knowledge_base
            SET active_index_revision_id = $1, provisioned_at = now()
            WHERE id = $2
            """,
            revision_id,
            knowledge_base_id,
        )
    return workspace_id, knowledge_base_id, revision_id


def _vector_literal(vector: tuple[float, ...]) -> str:
    return "[" + ",".join(format(value, ".9g") for value in vector) + "]"


async def _execute_cases(
    inputs: dict[str, Any],
    provider: OpenAICompatibleEmbeddingProvider,
    sessions,
    workspace_id: UUID,
    knowledge_base_id: UUID,
) -> tuple[CaseExecution, ...]:
    timed_provider = TimedEmbeddingProvider(provider)
    timed_store = TimedVectorStore(
        PgVectorStore(sessions, FixedPgVectorSpace(provider.embedding_space))
    )
    service = RetrievalService(
        SingleWorkspaceAccessPolicy(workspace_id),
        timed_provider,
        timed_store,
    )
    context = AuthContext("evaluation-runner", "offline-evaluation", workspace_id)
    executions: list[CaseExecution] = []
    for case in inputs["cases"]:
        pack = await service.retrieve(
            context,
            RetrievalRequest(
                knowledge_base_id,
                case["question"],
                top_k=max(inputs["config"]["top_k_values"]),
                include_debug=True,
            ),
        )
        sample_ids = tuple(
            str(item.source_metadata["evaluation_sample_id"])
            for item in pack.evidence
        )
        assert pack.debug is not None
        plan = pack.debug.query_plan
        executions.append(
            CaseExecution(
                case_id=case["case_id"],
                sample_ids=sample_ids,
                ranked_chunks=tuple(
                    {
                        "rank": item.rank,
                        "sample_id": sample_id,
                        "index_chunk_id": str(item.index_chunk_id),
                        "chunk_ordinal": item.ordinal,
                        "score": round(item.score, 8),
                    }
                    for item, sample_id in zip(pack.evidence, sample_ids, strict=True)
                ),
                query_embedding_ms=round(timed_provider.last_ms, 3),
                retrieval_database_ms=round(timed_store.last_ms, 3),
                query_plan={
                    "strategy": plan.strategy.value,
                    "top_k": plan.top_k,
                    "revision_selector": plan.revision_selector.value,
                    "current_document_version_only": plan.current_document_version_only,
                    "build_status": plan.build_status,
                    "serving_status": plan.serving_status,
                    "distance_metric": plan.distance_metric,
                    "candidate_count": plan.candidate_count,
                    "ef_search": plan.ef_search,
                    "iterative_scan": plan.iterative_scan.value,
                    "rerank": plan.rerank,
                },
                result_count=len(pack.evidence),
            )
        )
    return tuple(executions)


def _case_subset(
    cases: tuple[dict[str, Any], ...],
    dimension: str,
    value: str,
) -> tuple[dict[str, Any], ...]:
    if dimension == "language":
        return tuple(case for case in cases if case["language"] == value)
    if dimension == "tag":
        return tuple(case for case in cases if value in case["tags"])
    if dimension == "expected_empty_reason":
        return tuple(
            case
            for case in cases
            if case["retrieval"].get("empty_reason") == value
        )
    if dimension == "filter_selectivity":
        return cases
    raise ValueError(f"unsupported evaluation segment: {dimension}")


def _segments(
    cases: tuple[dict[str, Any], ...],
    ranked_by_strategy: dict[str, dict[str, tuple[str, ...]]],
    top_k_values: tuple[int, ...],
    filter_band: str,
) -> list[dict[str, Any]]:
    dimensions = {
        "language": sorted({case["language"] for case in cases}),
        "tag": sorted({tag for case in cases for tag in case["tags"]}),
        "expected_empty_reason": sorted(
            {
                case["retrieval"]["empty_reason"]
                for case in cases
                if case["retrieval"].get("empty_reason")
            }
        ),
        "filter_selectivity": [filter_band],
    }
    output: list[dict[str, Any]] = []
    for strategy_id, ranked in ranked_by_strategy.items():
        for dimension, values in dimensions.items():
            for value in values:
                subset = _case_subset(cases, dimension, value)
                output.append(
                    {
                        "dimension": dimension,
                        "value": value,
                        "strategy_id": strategy_id,
                        "metrics": _metric_values(subset, ranked, top_k_values),
                    }
                )
    return output


def _lexical_rankings(
    lexical_report: dict[str, Any],
    strategy_id: str,
) -> dict[str, tuple[str, ...]]:
    candidate = next(
        item for item in lexical_report["candidates"] if item["strategy_id"] == strategy_id
    )
    return {
        case["case_id"]: tuple(result["sample_id"] for result in case["results"])
        for case in candidate["cases"]
    }


def _gate(
    exact_metrics: dict[str, Any],
    segments: list[dict[str, Any]],
    embedding_latency: dict[str, Any],
    database_latency: dict[str, Any],
    thresholds: dict[str, Any],
    exact_strategy_id: str,
) -> dict[str, Any]:
    exact_identifier = next(
        segment["metrics"]
        for segment in segments
        if segment["strategy_id"] == exact_strategy_id
        and segment["dimension"] == "tag"
        and segment["value"] == "exact_identifier"
    )
    language_metrics = [
        segment["metrics"]
        for segment in segments
        if segment["strategy_id"] == exact_strategy_id
        and segment["dimension"] == "language"
    ]
    checks = {
        "overall_recall_at_5": exact_metrics["recall_at_k"]["5"]
        >= thresholds["exact_vector_recall_at_5_overall_minimum"],
        "overall_mrr": exact_metrics["mrr"]
        >= thresholds["exact_vector_mrr_minimum"],
        "exact_identifier_recall_at_5": exact_identifier["recall_at_k"]["5"]
        >= thresholds["exact_identifier_recall_at_5_minimum"],
        "per_language_recall_at_5": all(
            metrics["recall_at_k"]["5"]
            >= thresholds["per_language_recall_at_5_minimum"]
            for metrics in language_metrics
        ),
        "false_empty_rate": exact_metrics["false_empty_rate"]
        <= thresholds["false_empty_rate_maximum"],
        "expected_empty_accuracy": exact_metrics["expected_empty_accuracy"]
        >= thresholds["expected_empty_accuracy_minimum"],
        "forbidden_result_count": exact_metrics["forbidden_result_count"]
        <= thresholds["forbidden_result_count_maximum"],
        "query_embedding_p95_ms": embedding_latency["p95"]
        <= thresholds["query_embedding_p95_ms_maximum"],
        "exact_database_retrieval_p95_ms": database_latency["p95"]
        <= thresholds["exact_database_retrieval_p95_ms_maximum"],
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "thresholds": {
            key: value
            for key, value in thresholds.items()
            if key
            in {
                "exact_vector_recall_at_5_overall_minimum",
                "exact_vector_mrr_minimum",
                "exact_identifier_recall_at_5_minimum",
                "per_language_recall_at_5_minimum",
                "false_empty_rate_maximum",
                "expected_empty_accuracy_minimum",
                "forbidden_result_count_maximum",
                "query_embedding_p95_ms_maximum",
                "exact_database_retrieval_p95_ms_maximum",
            }
        },
    }


def _commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


async def _database_metadata(connection: asyncpg.Connection) -> dict[str, Any]:
    server_version = await connection.fetchval("SHOW server_version")
    vector_version = await connection.fetchval(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    )
    chunk_count = await connection.fetchval("SELECT count(*) FROM index_chunk")
    vector_count = await connection.fetchval("SELECT count(*) FROM vector_record_1024")
    eligible_chunk_count = await connection.fetchval(
        """
        SELECT count(*)
        FROM index_chunk AS chunk
        JOIN indexed_document_version AS indexed
          ON indexed.id = chunk.indexed_document_version_id
        JOIN document AS document
          ON document.id = indexed.document_id
        JOIN document_version AS version
          ON version.id = indexed.document_version_id
        WHERE indexed.build_status = 'ready'
          AND indexed.serving_status = 'serving'
          AND document.deleted_at IS NULL
          AND version.source_status = 'available'
        """
    )
    return {
        "postgresql_version": server_version,
        "pgvector_version": vector_version,
        "chunk_count": chunk_count,
        "vector_count": vector_count,
        "eligible_chunk_count_after_mandatory_filters": eligible_chunk_count,
        "concurrency": 1,
        "warm_up_queries": 0,
        "provider_network_included_in_query_embedding_ms": True,
    }


def _evaluation_cases(inputs: dict[str, Any]) -> EvaluationDatasetDefinition:
    return EvaluationDatasetDefinition(
        name="synthetic-v1-golden",
        version="1.0",
        manifest_hash=sha256(inputs["paths"]["golden_manifest"]),
        metadata={
            "dataset_id": inputs["golden_manifest"]["dataset_id"],
            "dataset_sha256": sha256(inputs["paths"]["golden_dataset"]),
            "corpus_id": inputs["manifest"]["corpus_id"],
            "corpus_profile_id": inputs["golden_manifest"]["corpus_profile_id"],
        },
        cases=tuple(
            EvaluationCaseDefinition(
                case["case_id"],
                case["question"],
                {
                    "language": case["language"],
                    "retrieval": case["retrieval"],
                },
                tuple(case["tags"]),
            )
            for case in inputs["cases"]
        ),
    )


def _persisted_results(
    cases: tuple[dict[str, Any], ...],
    executions: tuple[CaseExecution, ...],
    top_k_values: tuple[int, ...],
) -> tuple[EvaluationCaseResult, ...]:
    by_id = {execution.case_id: execution for execution in executions}
    results: list[EvaluationCaseResult] = []
    for case in cases:
        execution = by_id[case["case_id"]]
        single_ranked = {case["case_id"]: execution.sample_ids}
        metrics = _metric_values((case,), single_ranked, top_k_values)
        metrics.update(
            {
                "query_embedding_ms": execution.query_embedding_ms,
                "retrieval_database_ms": execution.retrieval_database_ms,
            }
        )
        results.append(
            EvaluationCaseResult(
                case["case_id"],
                {
                    "query_plan": execution.query_plan,
                    "result_count": execution.result_count,
                    "ranked_chunks": list(execution.ranked_chunks),
                },
                metrics,
            )
        )
    return tuple(results)


def build_report(
    inputs: dict[str, Any],
    executions: tuple[CaseExecution, ...],
    *,
    run_id: UUID,
    workspace_id: UUID,
    knowledge_base_id: UUID,
    revision_id: UUID,
    started_at: datetime,
    finished_at: datetime,
    commit: str,
    database_metadata: dict[str, Any],
) -> dict[str, Any]:
    config = inputs["config"]
    cases = inputs["cases"]
    embedding_definition = _embedding_definition(inputs["provider"])
    top_k_values = tuple(config["top_k_values"])
    exact_ranked = {
        execution.case_id: execution.sample_ids for execution in executions
    }
    lexical_ranked = _lexical_rankings(inputs["lexical"], config["lexical_strategy_id"])
    expected_ids = {case["case_id"] for case in cases}
    if set(exact_ranked) != expected_ids or set(lexical_ranked) != expected_ids:
        raise ValueError("exact and lexical strategies must cover the same golden cases")
    ranked_by_strategy = {
        config["exact_strategy_id"]: exact_ranked,
        config["lexical_strategy_id"]: lexical_ranked,
    }
    overall = {
        strategy_id: _metric_values(cases, ranked, top_k_values)
        for strategy_id, ranked in ranked_by_strategy.items()
    }
    segments = _segments(
        cases,
        ranked_by_strategy,
        top_k_values,
        config["filter_selectivity"]["band"],
    )
    embedding_latency = _latency(
        [execution.query_embedding_ms for execution in executions]
    )
    database_latency = _latency(
        [execution.retrieval_database_ms for execution in executions]
    )
    thresholds = inputs["quality"]["acceptance_thresholds"]
    exact_gate = _gate(
        overall[config["exact_strategy_id"]],
        segments,
        embedding_latency,
        database_latency,
        thresholds,
        config["exact_strategy_id"],
    )
    failures = [execution for execution in executions if execution.error_code]
    input_records = {
        key: {
            "path": config[key],
            "sha256": sha256(inputs["paths"][key]),
        }
        for key in (
            "corpus_manifest",
            "corpus_profile",
            "golden_dataset",
            "golden_manifest",
            "quality_baseline",
            "provider_declaration",
            "lexical_report",
            "report_schema",
        )
    }
    input_records["evaluation_config"] = {
        "path": str(inputs["paths"]["evaluation_config"].relative_to(root_path())),
        "sha256": sha256(inputs["paths"]["evaluation_config"]),
    }
    return {
        "schema_version": "1.0",
        "eval_run": {
            "eval_run_id": str(run_id),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "environment": "local_disposable_postgresql_real_qwen_embedding",
            "commit": commit,
            "tool": {
                "name": "tools/retrieval_evaluation.py",
                "version": TOOL_VERSION,
                "sha256": sha256(root_path() / "tools/retrieval_evaluation.py"),
            },
            "host": {
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
        },
        "inputs": {
            "corpus_profile_id": inputs["golden_manifest"]["corpus_profile_id"],
            "corpus_manifest_sha256": sha256(inputs["paths"]["corpus_manifest"]),
            "dataset_id": inputs["golden_manifest"]["dataset_id"],
            "dataset_sha256": sha256(inputs["paths"]["golden_dataset"]),
            "index_revision": {
                "id": str(revision_id),
                "parser": config["parser"],
                "database": database_metadata,
            },
            "embedding_space_fingerprint": inputs["quality"][
                "embedding_space_fingerprint"
            ],
            "embedding_model": {
                "provider_identity": embedding_definition.provider_identity,
                "endpoint_identity": embedding_definition.endpoint_identity,
                "requested_model": embedding_definition.requested_model,
                "resolved_model": embedding_definition.resolved_model,
                "model_version": embedding_definition.model_version,
                "deployment_revision": embedding_definition.deployment_revision,
                "dimension": embedding_definition.dimension,
                "configuration_fingerprint": (
                    embedding_definition.configuration_fingerprint
                ),
            },
            "provider_declaration": config["provider_declaration"],
            "answer_policy_version": config["answer_policy_version"],
            "workspace_id": str(workspace_id),
            "knowledge_base_id": str(knowledge_base_id),
            "recorded_inputs": input_records,
            "filter_selectivity": config["filter_selectivity"],
        },
        "strategies": [
            {
                "strategy_id": config["exact_strategy_id"],
                "configuration": {
                    "distance": "cosine",
                    "ann": False,
                    "top_k_values": list(top_k_values),
                    "query_shape": "single_statement_snapshot_with_mandatory_filters",
                    "mandatory_filters": config["mandatory_filters"],
                },
                "overall": overall[config["exact_strategy_id"]],
            },
            {
                "strategy_id": config["lexical_strategy_id"],
                "configuration": {
                    "serving_enabled": False,
                    "source_report": config["lexical_report"],
                    "top_k_values": list(top_k_values),
                },
                "overall": overall[config["lexical_strategy_id"]],
            },
        ],
        "segments": segments,
        "case_results": [
            {
                "case_id": execution.case_id,
                "strategy_id": config["exact_strategy_id"],
                "query_plan": execution.query_plan,
                "result_count": execution.result_count,
                "ranked_chunks": list(execution.ranked_chunks),
                "query_embedding_ms": execution.query_embedding_ms,
                "retrieval_database_ms": execution.retrieval_database_ms,
                "error_code": execution.error_code,
            }
            for execution in executions
        ],
        "answer_metrics": {
            "citation_identifier_validity": None,
            "structural_claim_coverage": None,
            "semantic_support_rate": None,
            "unsupported_claim_rate": None,
            "refusal_accuracy": None,
            "partial_answer_accuracy": None,
            "malicious_instruction_bypass_count": None,
            "label_source": "not_evaluated_stage04_retrieval_only",
        },
        "latency": {
            "query_embedding_ms": embedding_latency,
            "retrieval_database_ms": database_latency,
            "answer_pipeline_ms": _latency([]),
        },
        "usage": {
            "query_embedding_tokens": None,
            "chat_prompt_tokens": None,
            "chat_completion_tokens": None,
            "chat_total_tokens": None,
            "provider_usage_note": "embedding adapter does not expose token usage",
        },
        "failures": {
            "attempted_cases": len(cases),
            "failed_cases": len(failures),
            "failure_rate": round(len(failures) / len(cases), 6),
            "by_stable_code": {
                code: sum(execution.error_code == code for execution in failures)
                for code in sorted(
                    {execution.error_code for execution in failures if execution.error_code}
                )
            },
        },
        "gate_decisions": {
            "exact_vector": exact_gate,
            "lexical_comparison": {
                "status": "confirmed_evaluation_only",
                "serving_enabled": False,
                "source_selection": inputs["lexical"]["selection"],
            },
            "hnsw": {
                "status": "disabled_not_eligible",
                "reason": config["hnsw"]["w05_status"],
                "decision_owner": config["hnsw"]["decision_owner"],
                "enabled": False,
            },
        },
        "coverage_limitations": [
            "synthetic-v1 is not representative production scale",
            "all current cases share one high-selectivity mandatory-filter fixture",
            "answer, citation, semantic-support, token-usage, and chat latency are not evaluated in Stage 04",
            "provider aliases do not expose immutable weights revisions; the configuration fingerprint is the reproducibility boundary",
        ],
    }


def validate_report(report: dict[str, Any], inputs: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "eval_run",
        "inputs",
        "strategies",
        "segments",
        "answer_metrics",
        "latency",
        "usage",
        "failures",
        "gate_decisions",
    }
    if report.get("schema_version") != "1.0" or not required <= set(report):
        raise ValueError("evaluation report does not satisfy the v1.0 top-level contract")
    if report["failures"]["failure_rate"] != 0:
        raise ValueError("evaluation report contains failed cases")
    if report["gate_decisions"]["exact_vector"]["status"] != "passed":
        raise ValueError("exact-vector regression gate did not pass")
    if report["gate_decisions"]["hnsw"].get("enabled") is not False:
        raise ValueError("W05 must not enable HNSW")
    if report["eval_run"].get("tool", {}).get("sha256") != sha256(
        root_path() / "tools/retrieval_evaluation.py"
    ):
        raise ValueError("recorded evaluation tool checksum changed")
    for key, record in report["inputs"]["recorded_inputs"].items():
        expected_path = (
            str(inputs["paths"][key].relative_to(root_path()))
            if key == "evaluation_config"
            else inputs["config"][key]
        )
        if record["path"] != expected_path:
            raise ValueError(f"recorded evaluation input path changed: {key}")
        if record["sha256"] != sha256(inputs["paths"][key]):
            raise ValueError(f"recorded evaluation input checksum changed: {key}")


def root_path() -> Path:
    return Path(__file__).resolve().parents[1]


async def run_evaluation(inputs: dict[str, Any], root: Path) -> dict[str, Any]:
    runtime_dsn = os.environ.get(RUNTIME_DSN_ENV)
    sqlalchemy_dsn = os.environ.get(RUNTIME_SQLALCHEMY_DSN_ENV)
    if not runtime_dsn or not sqlalchemy_dsn:
        raise ValueError(
            f"evaluation requires {RUNTIME_DSN_ENV} and {RUNTIME_SQLALCHEMY_DSN_ENV}"
        )
    provider = _provider(inputs)
    started_at = datetime.now(UTC)
    fixtures = await _parse_and_embed(inputs, provider)
    connection = await asyncpg.connect(runtime_dsn)
    database = create_database_resources(
        sqlalchemy_dsn,
        pool_size=2,
        max_overflow=0,
        process=DatabaseProcess.WORKER,
    )
    try:
        workspace_id, knowledge_base_id, revision_id = await _seed_database(
            connection, inputs, fixtures
        )
        metadata = await _database_metadata(connection)
        executions = await _execute_cases(
            inputs,
            provider,
            database.sessions,
            workspace_id,
            knowledge_base_id,
        )
        finished_at = datetime.now(UTC)
        commit = _commit(root)
        run_id = uuid5(
            NAMESPACE_URL,
            f"{inputs['config']['evaluation_id']}:{commit}:{started_at.isoformat()}",
        )
        report = build_report(
            inputs,
            executions,
            run_id=run_id,
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
            revision_id=revision_id,
            started_at=started_at,
            finished_at=finished_at,
            commit=commit,
            database_metadata=metadata,
        )
        validate_report(report, inputs)
        definition = EvaluationRunDefinition(
            run_id=run_id,
            knowledge_base_id=knowledge_base_id,
            index_revision_id=revision_id,
            dataset=_evaluation_cases(inputs),
            run_config={
                "workspace_id": str(workspace_id),
                "evaluation_id": inputs["config"]["evaluation_id"],
                "commit": commit,
                "inputs": report["inputs"],
                "strategies": report["strategies"],
                "answer_policy_version": inputs["config"]["answer_policy_version"],
            },
            started_at=started_at,
        )
        persistence = EvaluationPersistenceService(
            SqlAlchemyUnitOfWorkFactory(database.sessions, workspace_id)
        )
        await persistence.start(definition)
        completed = await persistence.complete(
            definition,
            _persisted_results(
                inputs["cases"],
                executions,
                tuple(inputs["config"]["top_k_values"]),
            ),
            completed_at=finished_at,
        )
        if completed.result_count != len(inputs["cases"]):
            raise ValueError("persisted evaluation result count is incomplete")
        return report
    finally:
        await database.close()
        await connection.close()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    try:
        inputs = load_inputs(config_path.resolve(), root.resolve())
        output_value = args.output or Path(inputs["config"]["output"])
        output_path = output_value if output_value.is_absolute() else root / output_value
        if args.check:
            report = _load_json(output_path)
            validate_report(report, inputs)
            print("retrieval evaluation report is valid against recorded inputs")
            return 0
        report = asyncio.run(run_evaluation(inputs, root))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(canonical_json(report), encoding="utf-8")
        print(f"wrote {output_path.relative_to(root)}")
        return 0
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"retrieval evaluation error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
