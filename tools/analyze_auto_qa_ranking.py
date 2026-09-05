#!/usr/bin/env python3
"""Diagnose frozen Auto-QA rankings without embeddings, inference, or QA generation.

An explicit --export-sources reads the preserved host evaluation database once.
Normal runs use only local snapshots and require exact reproduction of all three
historical source-40 arms before reporting counterfactual ranking experiments.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import UUID

from rag_kb.domain import RerankDocument
from rag_kb.retrieval.reranker import _text_similarity, score_hits
from tools.build_document_qa_corpus import validate_corpus
from tools.evaluate_auto_qa_retrieval import _evaluation_cases, evidence_matches
from tools.evaluate_auto_qa_reuse import CachedReranker, _write, paired_changes

ROOT = Path(__file__).resolve().parents[1]
REPLAY_ROOT = ROOT / ".runtime/evaluations/auto-qa-source-20260905"
OUTPUT_ROOT = ROOT / ".runtime/evaluations/auto-qa-diagnosis-20260905"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


async def export_sources(arguments):
    # Deliberately no application dependencies, model providers, worker or API startup.
    from sqlalchemy import text
    from rag_kb.config import load_settings
    from rag_kb.db import DatabaseProcess, create_database_resources
    from tools.evaluation_runtime import load_evaluation_runtime

    runtime = load_evaluation_runtime()
    settings = load_settings(env_file=runtime.env_file)
    url = urlsplit(settings.database.runtime_dsn.get_secret_value())
    require(url.hostname in {"127.0.0.1", "localhost"} and url.port == runtime.ports["postgres"],
            "Only the existing isolated host evaluation database may be read")
    state = json.loads(arguments.state.read_text())
    corpus = validate_corpus(arguments.corpus_root)
    require(state["status"] == "indexed" and state["corpus_sha256"] == corpus["dataset_sha256"],
            "Frozen corpus/state mismatch; regeneration is not allowed")
    scope = {"workspace": settings.identity.workspace_id,
             "kb": UUID(state["arms"]["on"]["knowledge_base_id"]),
             "revision": UUID(state["arms"]["on"]["index_revision_id"])}
    async with create_database_resources(settings.database.runtime_dsn.get_secret_value(),
            pool_size=1, max_overflow=0, process=DatabaseProcess.MAINTENANCE) as db:
        async with db.sessions() as session, session.begin():
            await session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
            rows = (await session.execute(text("""
                SELECT c.id, c.content, c.hierarchy, c.source_location, c.source_metadata,
                       c.modality, c.ordinal, c.content_hash
                FROM index_chunk c JOIN indexed_document_version t ON t.id=c.indexed_document_version_id
                WHERE c.workspace_id=:workspace AND c.kb_id=:kb AND t.index_revision_id=:revision
                ORDER BY c.id
            """), scope)).mappings().all()
            questions = (await session.execute(text("""
                SELECT q.index_chunk_id, q.ordinal, q.question
                FROM index_chunk_question q JOIN index_chunk c ON c.id=q.index_chunk_id
                JOIN indexed_document_version t ON t.id=c.indexed_document_version_id
                WHERE c.workspace_id=:workspace AND c.kb_id=:kb AND t.index_revision_id=:revision
                ORDER BY q.index_chunk_id, q.ordinal
            """), scope)).mappings().all()
    require(len(rows) == 2480 and len(questions) == 12400, "Preserved QA inventory changed")
    chunks = {str(row["id"]): {key: value for key, value in row.items() if key != "id"}
              for row in rows}
    for chunk in chunks.values():
        chunk["questions"] = []
    for row in questions:
        chunks[str(row["index_chunk_id"])]["questions"].append(row["question"])
    _write(arguments.sources, {"corpus_sha256": state["corpus_sha256"], "chunks": chunks,
                              "database_writes": 0, "model_calls": 0})
    print(json.dumps({"exported_chunks": len(chunks), "existing_questions": len(questions),
                      "database_writes": 0, "model_calls": 0}), flush=True)


def row_result(case, ordered):
    ids = [str(hit.index_chunk_id) for hit in ordered[:10]]
    ranks = [i for i, hit in enumerate(ordered[:10], 1) if hit.label_match]
    rank = min(ranks, default=None)
    return {"case_id": case["evaluation_case_id"], "group": case["group"], "final_ids": ids,
            "relevant_rank": rank, "hit_1": rank == 1, "recall_5": rank is not None and rank <= 5,
            "recall_10": rank is not None, "reciprocal_rank_10": 1 / rank if rank else 0}


def summarize(rows):
    groups = {}
    for group in ("direct", "paraphrase"):
        selected = [row for row in rows if row["group"] == group]
        groups[group] = {"count": len(selected), "hit_1": sum(row["hit_1"] for row in selected),
                         "hit_5": sum(row["recall_5"] for row in selected),
                         "hit_10": sum(row["recall_10"] for row in selected),
                         "mrr_10": round(sum(row["reciprocal_rank_10"] for row in selected) / len(selected), 6)}
    return groups


def classic_order(query, hits):
    return sorted(score_hits(query, hits), key=lambda item: (
        -item.score, item.hit.cosine_distance, item.hit.index_chunk_id.int))


@lru_cache(maxsize=8192)
def similarity(left, right):
    return _text_similarity(left, right)


def rank_with_scores(hits, scores, *, mmr):
    # Same production MMR formula/ties, incrementally maintaining max redundancy.
    # Avoid recomputing token sets for every prefix in every diagnostic arm.
    positions = {hit.index_chunk_id: i for i, hit in enumerate(hits)}
    remaining = list(hits)
    redundancy = {hit.index_chunk_id: 0.0 for hit in hits}
    selected = []
    while remaining:
        best = max(remaining, key=lambda hit: (
            (0.75*scores[hit.index_chunk_id] - 0.25*redundancy[hit.index_chunk_id])
            if mmr and selected else scores[hit.index_chunk_id],
            scores[hit.index_chunk_id], -positions[hit.index_chunk_id], -hit.index_chunk_id.int))
        selected.append(best)
        remaining.remove(best)
        if mmr:
            for hit in remaining:
                redundancy[hit.index_chunk_id] = max(redundancy[hit.index_chunk_id], similarity(hit.text, best.text))
    return selected


def inspect_inputs(arguments, cases, checks, snapshot, cache):
    from rag_kb.adapters.local_reranker import (
        LocalMiniLmTokenizer, build_local_rerank_windows, _build_xlm_roberta_pair,
        _encode_without_special_tokens,
    )
    tokenizer = LocalMiniLmTokenizer.load(arguments.reranker_assets)
    inspected = 0
    queries = []
    details = []
    selected_cases = {"financebench_id_00563", "financebench_id_01091", "financebench_id_00222", "para-cfqa-89"}
    for case in cases:
        query = case["question"]
        tokens = _encode_without_special_tokens(tokenizer, query)
        canonical = tokenizer.backend_tokenizer.encode(query, "A short source.").ids
        manual = _build_xlm_roberta_pair(tokenizer, tokens, _encode_without_special_tokens(tokenizer, "A short source."))
        require(list(manual) == canonical, "Manual pair format differs from the pinned tokenizer")
        queries.append({"case_id": case["evaluation_case_id"], "group": case["group"],
                        "query_tokens": len(tokens), "truncated": len(tokens) > 96})
        for row in checks[case["evaluation_case_id"]]["candidates"]:
            source = snapshot["chunks"][row["id"]]
            doc = RerankDocument(index_chunk_id=UUID(row["id"]), text=source["content"],
                                hierarchy=source["hierarchy"], modality=source["modality"])
            cached = cache.values.get(cache.key(query, doc))
            if cached is None:
                require(row["model_score"] is None, "Expected score cache is missing")
                continue
            windows = build_local_rerank_windows(tokenizer, query, (doc,))
            require(len(windows) == cached["window_count"], "Window count differs from frozen inference")
            winning = cached["winning_window_index"]
            require(0 <= winning < len(windows), "Invalid winning window index")
            inspected += 1
            if case["evaluation_case_id"] in selected_cases and (row["historical_label_match"] or row["model_score"] > 0.1):
                details.append({"case_id": case["evaluation_case_id"], "id": row["id"],
                    "historical_label_match": row["historical_label_match"], "model_score": row["model_score"],
                    "window_count": len(windows), "winning_window_index": winning,
                    "winning_input": tokenizer.backend_tokenizer.decode(list(windows[winning].input_ids), skip_special_tokens=False)})
    _write(arguments.output.with_name("input-audit.json"), {"model_calls": 0,
        "pair_format_checks": len(queries), "window_count_checks": inspected,
        "queries": queries, "selected_winning_inputs": details})


def analyze(arguments):
    replay_bytes = arguments.replay.read_bytes()
    replay = json.loads(replay_bytes)
    source_bytes = arguments.sources.read_bytes()
    snapshot = json.loads(source_bytes)
    cases = _evaluation_cases(arguments.corpus_root)
    corpus = validate_corpus(arguments.corpus_root)
    require(replay["status"] == "complete" and replay["completed_cases"] == len(cases) == 103,
            "A complete frozen replay is required")
    require(replay["corpus_sha256"] == snapshot["corpus_sha256"] == corpus["dataset_sha256"],
            "Corpus/source/replay identities differ")
    checks = {row["case_id"]: row for row in replay["candidate_checks"]}
    historical = {name: {row["case_id"]: row for row in arm["cases"]}
                  for name, arm in replay["arms"].items()}
    cache = CachedReranker(None, arguments.model_scores, replay["reranker_revision"], cache_only=True)
    arms = defaultdict(list)
    audit = []
    for case in cases:
        case_id, query = case["evaluation_case_id"], case["question"]
        check = checks[case_id]
        hits = []
        for row in check["candidates"]:
            source = snapshot["chunks"][row["id"]]
            hit = SimpleNamespace(index_chunk_id=UUID(row["id"]), text=source["content"],
                hierarchy=source["hierarchy"], modality=source["modality"],
                source_location=source["source_location"], source_metadata=source["source_metadata"],
                cosine_distance=row["source_distance"], label_match=row["historical_label_match"],
                source=row["source"], matched_question=row["matched_question"])
            require(evidence_matches(case, hit) == hit.label_match, f"Label replay mismatch: {case_id}")
            require(hit.modality in {"text", "table"}, "This text-lane analysis cannot omit an image candidate")
            document = RerankDocument(index_chunk_id=hit.index_chunk_id, text=hit.text,
                                     hierarchy=hit.hierarchy, modality=hit.modality)
            # Exact key covers query, source text, hierarchy, modality, ID and model revision.
            key = cache.key(query, document)
            hit.model = cache.values.get(key)
            hits.append(hit)
        sources = [hit for hit in hits if hit.source and 1 - hit.cosine_distance >= 0.35]
        augmented = [hit for hit in hits if hit.matched_question or (hit.source and 1 - hit.cosine_distance >= 0.35)]
        require([str(hit.index_chunk_id) for hit in hits if hit.source] == check["source_ids"],
                "Source candidate identity/order mismatch")
        pools = {"source": sources, "augmented": augmented}
        case_arms = {}
        diagnostics = {}
        for pool_name, pool in pools.items():
            require(all(hit.model is not None for hit in pool), f"Missing frozen model score: {case_id}; no inference allowed")
            for hit in pool:
                stored = next(row for row in check["candidates"] if row["id"] == str(hit.index_chunk_id))
                require(stored["model_score"] == hit.model["score"], "Model score cache differs from replay")
            classic = classic_order(query, pool)
            initial = [item.hit for item in classic]
            model_scores = {hit.index_chunk_id: hit.model["score"] for hit in pool}
            classic_scores = {item.hit.index_chunk_id: item.score for item in classic}
            orders = {
                f"classic_{pool_name}": initial,
                f"model_raw_{pool_name}": rank_with_scores(initial, model_scores, mmr=False),
                f"model_mmr_{pool_name}": rank_with_scores(initial, model_scores, mmr=True),
                f"classic_mmr_{pool_name}": rank_with_scores(initial, classic_scores, mmr=True),
                f"model_logit_mmr_{pool_name}": rank_with_scores(initial,
                    {hit.index_chunk_id: hit.model["raw_logit"] for hit in initial}, mmr=True),
            }
            # Prespecified diagnostic weights, not a parameter search or release configuration.
            for weight in (0.1, 0.2):
                fused = {hit.index_chunk_id: (1-weight)*classic_scores[hit.index_chunk_id]
                         + weight*model_scores[hit.index_chunk_id] for hit in initial}
                orders[f"fusion_{int(weight*100)}_{pool_name}"] = rank_with_scores(initial, fused, mmr=False)
            if pool_name == "source":
                orders["vector_source"] = sorted(pool, key=lambda hit: (hit.cosine_distance, hit.index_chunk_id.int))
            case_arms.update(orders)
            diagnostics[pool_name] = {str(item.hit.index_chunk_id): {
                "classic_score": item.score, "lexical_score": item.lexical_score,
                "lexical_coverage": item.lexical_coverage,
                "model_score": item.hit.model["score"], "raw_logit": item.hit.model["raw_logit"],
                "window_count": item.hit.model["window_count"],
                "winning_window_index": item.hit.model["winning_window_index"],
                "model_raw_rank": next(i for i,h in enumerate(orders[f"model_raw_{pool_name}"],1) if h is item.hit),
                "model_mmr_rank": next(i for i,h in enumerate(orders[f"model_mmr_{pool_name}"],1) if h is item.hit),
            } for item in classic}
        for name, prior in (("classic_source", "classic_source"), ("model_mmr_source", "minilm_source"),
                            ("model_mmr_augmented", "minilm_augmented")):
            require(row_result(case, case_arms[name])["final_ids"] == historical[prior][case_id]["final_ids"],
                    f"Production ranking reproduction failed: {case_id}/{name}")
        # Membership protection counterfactual: append only, never edit the baseline prefix.
        prefix = case_arms["classic_source"][:10]
        extras = [hit for hit in case_arms["classic_augmented"] if hit not in prefix and not hit.source][:2]
        case_arms["protected_append"] = prefix + extras
        require(row_result(case, case_arms["protected_append"])["final_ids"] ==
                row_result(case, prefix)["final_ids"], "Protected Top-10 differs from the frozen baseline")
        for name, ordered in case_arms.items():
            arms[name].append(row_result(case, ordered))
        audit.append({"case_id": case_id, "query": query, "group": case["group"],
            "source_candidate_match": any(hit.label_match for hit in sources),
            "qa_only_candidate_match": any(hit.label_match and not hit.source for hit in augmented),
            "augmented_candidate_match": any(hit.label_match for hit in augmented),
            "source_count": len(sources), "qa_only_count": sum(not hit.source for hit in augmented),
            "appended_ids": [str(hit.index_chunk_id) for hit in extras],
            "candidates": [{**row, "filename": snapshot["chunks"][row["id"]]["source_metadata"].get("original_filename"),
                "modality": snapshot["chunks"][row["id"]]["modality"],
                "source_location": snapshot["chunks"][row["id"]]["source_location"],
                "scores": {pool: values.get(row["id"]) for pool, values in diagnostics.items()}}
                for row in check["candidates"]]})
    result = {"schema_version": "auto_qa_ranking_diagnosis_v1", "completed_cases": len(cases),
        "model_calls": 0, "qa_generation_calls": 0, "ranking_reproductions": 3*len(cases),
        "replay_sha256": hashlib.sha256(replay_bytes).hexdigest(),
        "source_snapshot_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "model_cache_sha256": hashlib.sha256(arguments.model_scores.read_bytes()).hexdigest(),
        "corpus_sha256": corpus["dataset_sha256"],
        "arms": {name: {"cases": rows, "summary": summarize(rows),
                        "vs_classic_source": paired_changes(arms["classic_source"], rows)} for name, rows in arms.items()},
        "mmr_effect": {pool: paired_changes(arms[f"model_raw_{pool}"], arms[f"model_mmr_{pool}"])
                       for pool in ("source", "augmented")},
        "candidate_coverage": {group: {
            "source_matched_cases": sum(a["source_candidate_match"] for a in audit if a["group"] == group),
            "union_matched_cases": sum(a["augmented_candidate_match"] for a in audit if a["group"] == group),
            "new_coverage_cases": [a["case_id"] for a in audit if a["group"] == group
                                   and a["augmented_candidate_match"] and not a["source_candidate_match"]],
            "qa_only_candidate_pairs": sum(a["qa_only_count"] for a in audit if a["group"] == group),
            "qa_only_label_matched_pairs": sum(not h["source"] and h["historical_label_match"]
                for a in audit if a["group"] == group for h in a["candidates"]),
            "classic_pool_score_drift_cases": sum(any(h["scores"]["source"] and
                h["scores"]["source"]["classic_score"] != h["scores"]["augmented"]["classic_score"]
                for h in a["candidates"]) for a in audit if a["group"] == group),
        } for group in ("direct", "paraphrase")},
        "audit": audit,
        "limitations": ["Frozen historical labels; development data only.",
                        "Fusion arms are diagnostics, not validated release parameters.",
                        "Protected append is evaluated at the unchanged Top-10; no answer-quality claim.",
                        "No per-window logits are cached; window aggregation cannot be causally ablated."]}
    _write(arguments.output, result)
    if arguments.audit_inputs:
        inspect_inputs(arguments, cases, checks, snapshot, cache)
    print(json.dumps({"cases": len(cases), "ranking_reproductions": result["ranking_reproductions"],
        "model_calls": 0, "summaries": {name: arm["summary"] for name, arm in result["arms"].items()}},
        ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=ROOT / ".runtime/evaluations/auto-qa-ab-20260904b/state.json")
    parser.add_argument("--corpus-root", type=Path, default=ROOT / "evaluation/document-qa-v1")
    parser.add_argument("--replay", type=Path, default=REPLAY_ROOT / "retrieval.json")
    parser.add_argument("--model-scores", type=Path, default=REPLAY_ROOT / "model-scores.json")
    parser.add_argument("--sources", type=Path, default=OUTPUT_ROOT / "sources.json")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "analysis.json")
    parser.add_argument("--export-sources", action="store_true")
    parser.add_argument("--audit-inputs", action="store_true", help="Rebuild token windows only; never run the model")
    parser.add_argument("--reranker-assets", type=Path, default=ROOT / ".runtime/model-assets/local-reranker")
    args = parser.parse_args()
    if args.export_sources:
        asyncio.run(export_sources(args))
    analyze(args)


if __name__ == "__main__":
    main()
