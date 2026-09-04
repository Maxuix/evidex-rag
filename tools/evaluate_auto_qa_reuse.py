#!/usr/bin/env python3
"""Read-only paired retrieval replay using existing questions; never indexes or generates QA."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlsplit
from uuid import UUID

from sqlalchemy import text

from apps.api.dependencies import build_api_dependencies
from rag_kb.adapters.local_reranker import LocalMiniLmReranker
from rag_kb.adapters.local_reranker_artifacts import verify_local_reranker_artifacts
from rag_kb.config import load_settings
from rag_kb.domain import EvidencePack, ModelRerankScore, RetrievalDebug, RetrievalQueryPlan, RetrievalStrategy, RerankMode, RerankDocument
from rag_kb.retrieval.service import _matched_questions
from tools.build_document_qa_corpus import validate_corpus
from tools.evaluate_auto_qa_retrieval import _case_result, _evaluation_cases, evidence_matches, summarize
from tools.evaluation_runtime import load_evaluation_runtime

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / ".runtime/evaluations/auto-qa-source-20260905"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def paired_changes(baseline, enhanced):
    by_id = {item["case_id"]: item for item in enhanced}
    result = {}
    for group in ("direct", "paraphrase"):
        pairs = [(item, by_id[item["case_id"]]) for item in baseline if item["group"] == group]
        result[group] = {
            name: [left["case_id"] for left, right in pairs if predicate(left, right)]
            for name, predicate in {
                "lost_top1": lambda a, b: a["hit_1"] and not b["hit_1"],
                "lost_top10": lambda a, b: a["recall_10"] and not b["recall_10"],
                "new_top10": lambda a, b: not a["recall_10"] and b["recall_10"],
                "rank_worsened": lambda a, b: (b["relevant_rank"] or 11) > (a["relevant_rank"] or 11),
            }.items()
        }
    return result


def retrieval_gate_passed(result):
    if result.get("completed_cases") != result.get("total_cases"):
        return False
    enhanced = result["arms"]["minilm_augmented"]["summary"]["retrieval_by_group"]
    for baseline in ("classic_source", "minilm_source"):
        original = result["arms"][baseline]["summary"]["retrieval_by_group"]
        for group, expected_count in (("direct", 61), ("paraphrase", 24)):
            changes = result["paired"][baseline][group]
            if changes["lost_top1"] or changes["lost_top10"]:
                return False
            if enhanced.get(group, {}).get("count") != expected_count:
                return False
            for metric in ("hit_at_1", "mrr_at_10", "recall_at_5", "recall_at_10"):
                if enhanced[group][metric] < original[group][metric]:
                    return False
    return True


def trace_legacy_losses(legacy, replay):
    old_on = {row["case_id"]: row for row in legacy["arms"]["on"]["semantic"]["cases"]}
    candidates = {row["case_id"]: row for row in replay["candidate_checks"]}
    arms = {name: {row["case_id"]: row for row in data["cases"]}
            for name, data in replay["arms"].items()}
    losses = []
    for row in legacy["arms"]["off"]["semantic"]["cases"]:
        case_id = row["case_id"]
        if row["group"] not in {"direct", "paraphrase"} or not row["recall_10"] or old_on[case_id]["recall_10"]:
            continue
        matched = [hit for hit in candidates.get(case_id, {}).get("candidates", [])
                   if hit["historical_label_match"]]
        losses.append({
            "case_id": case_id, "group": row["group"], "legacy_off_rank": row["relevant_rank"],
            "replayed": case_id in candidates,
            "source_candidate_match": any(hit["source"] for hit in matched),
            "any_candidate_match": bool(matched),
            "scored_candidate_match": any(hit.get("model_score") is not None for hit in matched),
            "ranks": {name: rows.get(case_id, {}).get("relevant_rank") for name, rows in arms.items()},
        })
    return losses


class CachedReranker:
    """Cache identical query/source pairs across arms, not their final rankings."""

    profile = RerankMode.LOCAL_MINILM_V1

    def __init__(self, adapter, path, identity, *, cache_only=False):
        self.adapter, self.path, self.identity = adapter, path, identity
        self.cache_only = cache_only
        self.values = json.loads(path.read_text()) if path.exists() else {}

    @property
    def max_documents(self):
        return self.adapter.max_documents

    def key(self, query, document):
        payload = [self.identity, query, str(document.index_chunk_id), document.text, document.hierarchy, document.modality]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    async def score(self, query, documents):
        missing = tuple(item for item in documents if self.key(query, item) not in self.values)
        if missing:
            if self.cache_only:
                raise RuntimeError("model score cache missing; local inference is disabled")
            scores = await self.adapter.score(query, missing)
            for document, score in zip(missing, scores, strict=True):
                assert document.index_chunk_id == score.index_chunk_id
                self.values[self.key(query, document)] = {**asdict(score), "index_chunk_id": str(score.index_chunk_id)}
            _write(self.path, self.values)
        return tuple(ModelRerankScore(**{**self.values[self.key(query, item)], "index_chunk_id": item.index_chunk_id}) for item in documents)


async def run(arguments):
    runtime = load_evaluation_runtime()
    settings = load_settings(env_file=runtime.env_file)
    url = urlsplit(settings.database.runtime_dsn.get_secret_value())
    if url.hostname not in {"127.0.0.1", "localhost"} or url.port != runtime.ports["postgres"]:
        raise RuntimeError("evaluation database is not the isolated host runtime")
    state = json.loads(arguments.state.read_text())
    corpus = validate_corpus(arguments.corpus_root)
    if state["status"] != "indexed" or state["corpus_sha256"] != corpus["dataset_sha256"]:
        raise RuntimeError("existing indexed QA corpus does not match")
    requirements = {"chat": "mimo-v2.5", "text_embedding": "qwen3.7-text-embedding", "multimodal_embedding": "tongyi-embedding-vision-flash-2026-03-06"}
    if any(state["profiles"][key]["model"] != model for key, model in requirements.items()):
        raise RuntimeError("frozen evaluation models do not match requirements")
    manifest_path = ROOT / "config/local-reranker-artifacts-v1.json"
    manifest = verify_local_reranker_artifacts(arguments.reranker_assets, manifest_path)
    dependencies = build_api_dependencies(settings=settings)
    # This is adapter replay, not an API/worker startup. It only reads old index columns;
    # it does not migrate the preserved evaluation DB or start brokers/workers.
    service = dependencies.retrieval_service
    cache = CachedReranker(LocalMiniLmReranker(artifacts_path=arguments.reranker_assets, manifest_path=manifest_path),
                           arguments.output.parent / "model-scores.json", manifest.revision,
                           cache_only=arguments.cached_model_scores_only)
    service._text_reranker = cache
    query_file = arguments.output.parent / "query-vectors.json"
    query_cache = json.loads(query_file.read_text()) if query_file.exists() else {}
    kb_id = UUID(state["arms"]["on"]["knowledge_base_id"])
    plan = RetrievalQueryPlan(settings.identity.workspace_id, kb_id, RetrievalStrategy.EXACT_VECTOR,
                              top_k=10, candidate_count=40, rerank_mode=RerankMode.LOCAL_MINILM_V1,
                              allow_unverified_auto_qa=True)
    try:
        async with dependencies.database.sessions() as session:
            async with session.begin():
                await session.execute(text("SET TRANSACTION READ ONLY"))
                counts = (await session.execute(text(
                    "SELECT count(DISTINCT c.id) chunks, count(q.ordinal) questions "
                    "FROM index_chunk c LEFT JOIN index_chunk_question q ON q.index_chunk_id=c.id "
                    "WHERE c.kb_id=:kb AND c.workspace_id=:workspace"
                ), {"kb": kb_id, "workspace": settings.identity.workspace_id})).mappings().one()
        if (counts["chunks"], counts["questions"]) != (2480, 12400):
            raise RuntimeError("existing QA inventory differs; will not regenerate")
        provider = await service._text_embedding_provider(plan)
        if provider.embedding_space.requested_model != requirements["text_embedding"]:
            raise RuntimeError("resolved query embedding model differs")
        print(json.dumps({"event": "preflight", **dict(counts), "qa_generation_calls": 0}), flush=True)
        if arguments.preflight_only:
            return
        arms = {name: [] for name in ("classic_source", "minilm_source", "minilm_augmented", "minilm_source_60")}
        result = {"schema_version": "auto_qa_reuse_v1", "status": "running", "top_k": 10, "qa_generation_calls": 0,
                  "corpus_sha256": state["corpus_sha256"], "question_count": counts["questions"],
                  "reranker_revision": manifest.revision, "arms": {}, "candidate_checks": [],
                  "limitations": ["Existing unverified questions reused; no new grounded generation assessed.",
                                  "Historical page/overlap labels retained; this is not independent holdout validation.",
                                  "Reranker caches shared across arms; elapsed times are not comparable cold latencies."]}
        cases = _evaluation_cases(arguments.corpus_root)
        result["total_cases"] = len(cases)
        result["completed_cases"] = 0
        for position, case in enumerate(cases, start=1):
            query = str(case["question"])
            query_key = hashlib.sha256((provider.embedding_space.compatibility_fingerprint + "\n" + query).encode()).hexdigest()
            if query_key not in query_cache:
                if arguments.cached_queries_only:
                    raise RuntimeError("query cache missing; no embedding call allowed")
                query_cache[query_key] = list(await service._embed_query(query, provider))
                _write(query_file, query_cache)
            vector = tuple(query_cache[query_key])
            baseline = await service._search_text(replace(plan, auto_qa_candidate_count=0), vector, provider)
            augmented = await service._search_text(plan, vector, provider)
            expanded = await service._search_text(replace(plan, candidate_count=60, auto_qa_candidate_count=0), vector, provider)
            if baseline is None or augmented is None or expanded is None:
                raise RuntimeError("frozen serving index unavailable")
            if any(str(value.resolved_active_revision_id) != state["arms"]["on"]["index_revision_id"] for value in (baseline, augmented, expanded)):
                raise RuntimeError("serving revision changed during evaluation")
            original = {hit.index_chunk_id: hit.cosine_distance for hit in baseline.hits}
            augmented_sources = {hit.index_chunk_id: hit.cosine_distance for hit in augmented.hits if hit.source_candidate}
            if original != augmented_sources:
                raise RuntimeError("source candidate preservation invariant failed")
            result["candidate_checks"].append({
                "case_id": case["evaluation_case_id"], "source_preserved": True,
                "source_ids": [str(item.index_chunk_id) for item in baseline.hits],
                "candidates": [{"id": str(hit.index_chunk_id), "source": hit.source_candidate,
                                "source_distance": hit.cosine_distance, "question_distance": hit.question_cosine_distance,
                                "matched_question": hit.matched_question,
                                "historical_label_match": evidence_matches(case, hit)} for hit in augmented.hits],
            })
            for name, found, mode, budget in (
                ("classic_source", baseline, RerankMode.CLASSIC, 40),
                ("minilm_source", baseline, RerankMode.LOCAL_MINILM_V1, 40),
                ("minilm_augmented", augmented, RerankMode.LOCAL_MINILM_V1, 40),
                ("minilm_source_60", expanded, RerankMode.LOCAL_MINILM_V1, 60),
            ):
                current = replace(plan, rerank_mode=mode, candidate_count=budget,
                                  auto_qa_candidate_count=20 if name == "minilm_augmented" else 0)
                profile = service.execution_profile(strategy=current.strategy, top_k=10, rerank_mode=mode)
                started = time.perf_counter()
                evidence, model_count, windows = await service._normalize(current, found, query=query, profile=profile)
                pack = EvidencePack(kb_id, found.resolved_active_revision_id, current.strategy, evidence,
                    debug=RetrievalDebug(current, found.resolved_active_revision_id, len(evidence), matched_questions=_matched_questions(found)))
                item = _case_result(case, pack, (time.perf_counter() - started) * 1000)
                item.update(model_candidate_count=model_count, model_window_count=windows,
                            final_ids=[str(value.index_chunk_id) for value in evidence])
                arms[name].append(item)
            for hit, diagnostic in zip(augmented.hits, result["candidate_checks"][-1]["candidates"], strict=True):
                if hit.modality not in {"text", "table"} or not hit.text.strip():
                    diagnostic["model_score"] = None
                    continue
                document = RerankDocument(index_chunk_id=hit.index_chunk_id, text=hit.text,
                                          hierarchy=hit.hierarchy, modality=hit.modality)
                score = cache.values.get(cache.key(query, document))
                diagnostic["model_score"] = score["score"] if score else None
            result["arms"] = {name: {"cases": values, "summary": summarize(values)} for name, values in arms.items()}
            result["paired"] = {name: paired_changes(arms[name], arms["minilm_augmented"]) for name in ("classic_source", "minilm_source", "minilm_source_60")}
            result["completed_cases"] = position
            _write(arguments.output, result)
            print(json.dumps({"event": "case_complete", "position": position, "total": len(cases), "case_id": case["evaluation_case_id"]}), flush=True)
        legacy_path = arguments.state.parent / "retrieval.json"
        if legacy_path.exists():
            legacy = json.loads(legacy_path.read_text())
            if legacy.get("corpus_sha256") != result["corpus_sha256"]:
                raise RuntimeError("historical retrieval corpus differs")
            result["legacy_top10_losses"] = trace_legacy_losses(legacy, result)
        result["status"] = "complete"
        result["retrieval_gate_passed"] = retrieval_gate_passed(result)
        result["release_gate_passed"] = False  # Holdout and answer-quality gates remain separate.
        _write(arguments.output, result)
    finally:
        await dependencies.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=ROOT / ".runtime/evaluations/auto-qa-ab-20260904b/state.json")
    parser.add_argument("--corpus-root", type=Path, default=ROOT / "evaluation/document-qa-v1")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "retrieval.json")
    parser.add_argument("--reranker-assets", type=Path, default=ROOT / ".runtime/model-assets/local-reranker")
    parser.add_argument("--cached-queries-only", action="store_true")
    parser.add_argument("--cached-model-scores-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
