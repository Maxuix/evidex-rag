#!/usr/bin/env python3
"""Audit frozen source recall. No provider/model calls, QA generation or DB writes.

--export-ranks explicitly reads the preserved isolated host index using cached
query vectors. Normal runs use the resulting exact ranks without database access.
Diagnostic depth slices do not change retrieval configuration or final rankings.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import UUID

from tools.analyze_auto_qa_ranking import classic_order, require, row_result
from tools.build_document_qa_corpus import validate_corpus
from tools.evaluate_auto_qa_retrieval import _evaluation_cases, evidence_matches

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSIS = ROOT / ".runtime/evaluations/auto-qa-diagnosis-20260905"
REPLAY = ROOT / ".runtime/evaluations/auto-qa-source-20260905"
REPAIR = ROOT / ".runtime/evaluations/minilm-repair-20260905"
OUTPUT = ROOT / ".runtime/evaluations/source-recall-20260905"
CORPUS = ROOT / "evaluation/document-qa-v1"
STATE = ROOT / ".runtime/evaluations/auto-qa-ab-20260904b/state.json"
DEPTHS = (10, 20, 40, 60, 80, 100, 200, 2480)
MIN_SIMILARITY = 0.35


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    path.chmod(0o600)


def inputs():
    corpus = validate_corpus(CORPUS)
    snapshot = read(DIAGNOSIS / "sources.json")
    baseline = read(DIAGNOSIS / "analysis.json")
    repair = read(REPAIR / "repaired-int8.json")
    require(corpus["dataset_sha256"] == snapshot["corpus_sha256"]
            == baseline["corpus_sha256"] == repair["corpus_sha256"], "Corpus identity changed")
    cases = [c for c in _evaluation_cases(CORPUS) if c["group"] in {"direct", "paraphrase"}]
    require(len(cases) == 85 and len(snapshot["chunks"]) == 2480, "Frozen inventory changed")
    return corpus, snapshot, baseline, repair, cases


async def export_ranks(path):
    from sqlalchemy import text
    from rag_kb.adapters.vector_store.pgvector import PgVectorStore
    from rag_kb.config import load_settings
    from rag_kb.db import DatabaseProcess, create_database_resources
    from tools.evaluation_runtime import load_evaluation_runtime

    corpus, snapshot, _, _, cases = inputs()
    state = read(STATE)
    require(state["status"] == "indexed" and state["corpus_sha256"] == corpus["dataset_sha256"],
            "Preserved evaluation state mismatch; regeneration forbidden")
    runtime = load_evaluation_runtime()
    settings = load_settings(env_file=runtime.env_file)
    url = urlsplit(settings.database.runtime_dsn.get_secret_value())
    require(url.hostname in {"127.0.0.1", "localhost"} and url.port == runtime.ports["postgres"],
            "Only the preserved isolated host database may be read")
    query_cache = read(REPLAY / "query-vectors.json")
    params = {"workspace_id": settings.identity.workspace_id,
              "knowledge_base_id": UUID(state["arms"]["on"]["knowledge_base_id"]),
              "revision_id": UUID(state["arms"]["on"]["index_revision_id"])}
    result = {"corpus_sha256": corpus["dataset_sha256"], "source_sha256": digest(DIAGNOSIS / "sources.json"),
              "query_cache_sha256": digest(REPLAY / "query-vectors.json"),
              "database_writes": 0, "model_calls": 0, "qa_generation_calls": 0,
              "source_only": True, "cases": []}
    async with create_database_resources(settings.database.runtime_dsn.get_secret_value(),
            pool_size=1, max_overflow=0, process=DatabaseProcess.MAINTENANCE) as db:
        async with db.sessions() as session, session.begin():
            await session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
            await session.execute(text("SET LOCAL statement_timeout = '30s'"))
            space = (await session.execute(text("""
                SELECT es.id, es.dimension, es.requested_model, es.compatibility_fingerprint
                FROM embedding_space es
                JOIN index_revision_embedding_space link ON link.embedding_space_id=es.id
                JOIN index_revision r ON r.id=link.index_revision_id
                JOIN knowledge_base kb ON kb.id=r.kb_id AND kb.workspace_id=r.workspace_id
                WHERE r.id=:revision_id AND r.workspace_id=:workspace_id
                  AND kb.id=:knowledge_base_id AND kb.active_index_revision_id=r.id
                  AND kb.deleted_at IS NULL AND r.status='active' AND link.role='text_retrieval'
            """), params)).mappings().one()
            require(space["requested_model"] == "qwen3.7-text-embedding", "Frozen embedding model differs")
            source_rows = (await session.execute(text("""
                SELECT c.id,c.content,c.content_hash,c.hierarchy,c.source_location,c.source_metadata,c.modality,c.ordinal
                FROM index_chunk c JOIN indexed_document_version t ON t.id=c.indexed_document_version_id
                WHERE c.workspace_id=:workspace_id AND c.kb_id=:knowledge_base_id
                  AND t.index_revision_id=:revision_id
            """), params)).mappings().all()
            require({str(r["id"]) for r in source_rows} == set(snapshot["chunks"]), "Source membership changed")
            for row in source_rows:
                require(all(value == snapshot["chunks"][str(row["id"])][key]
                            for key, value in row.items() if key != "id"), "Source contents changed")
            statement = PgVectorStore._text_statement(space["dimension"])
            names = ("active_revision_id", "compatibility_fingerprint", "index_chunk_id",
                     "cosine_distance", "source_candidate")
            statement = statement.with_only_columns(*(statement.selected_columns[n] for n in names),
                                                     maintain_column_froms=True)
            result["embedding_model"] = space["requested_model"]
            result["compatibility_fingerprint"] = space["compatibility_fingerprint"]
            result["dimension"] = space["dimension"]
            result["revision_id"] = str(params["revision_id"])
            for position, case in enumerate(cases, 1):
                key = hashlib.sha256((space["compatibility_fingerprint"] + "\n" + case["question"]).encode()).hexdigest()
                require(key in query_cache, "Query cache missing; no model fallback allowed")
                vector = query_cache[key]
                require(len(vector) == space["dimension"] and all(math.isfinite(x) for x in vector),
                        "Invalid cached query vector")
                rows = (await session.execute(statement, {**params, "query_embedding": vector,
                    "top_k": len(source_rows) + 1, "auto_qa_candidate_count": 0,
                    "allow_unverified_auto_qa": False, "space_role": "text_retrieval",
                    "representation_kinds": ["text", "caption_text", "ocr_text", "table_text"],
                    "expected_dimension": space["dimension"]})).mappings().all()
                require(len(rows) == len(source_rows) and
                        {str(r["index_chunk_id"]) for r in rows} == set(snapshot["chunks"]),
                        "Some frozen chunks are ineligible or have no matching source vector")
                require(all(r["source_candidate"] and r["active_revision_id"] == params["revision_id"]
                            and r["compatibility_fingerprint"] == space["compatibility_fingerprint"] for r in rows),
                        "Wrong retrieval scope or non-source candidate")
                result["cases"].append({"case_id": case["evaluation_case_id"],
                    "ranks": [[str(r["index_chunk_id"]), r["cosine_distance"]] for r in rows]})
                if position % 10 == 0 or position == len(cases):
                    print(json.dumps({"exact_rank_queries": position, "total": len(cases)}), flush=True)
    result["status"] = "complete"
    write(path, result)


def hit(chunk_id, distance, chunks, case):
    source = chunks[chunk_id]
    value = SimpleNamespace(index_chunk_id=UUID(chunk_id), text=source["content"],
        hierarchy=source["hierarchy"], modality=source["modality"],
        source_location=source["source_location"], source_metadata=source["source_metadata"],
        cosine_distance=distance)
    value.label_match = evidence_matches(case, value)
    return value


def summarize(rows):
    return {"n": len(rows),
        "vector_top1": sum(r["best_label_rank"] == 1 for r in rows),
        "candidate_coverage": {str(k): sum(r["coverage"][str(k)] for r in rows) for k in DEPTHS},
        "missing_before_ranking": sum(not r["coverage"]["40"] for r in rows),
        "classic_top1": sum(r["classic_rank"] == 1 for r in rows),
        "classic_top10": sum(r["classic_rank"] is not None for r in rows),
        "minilm_top1": sum(r["minilm_rank"] == 1 for r in rows),
        "minilm_top10": sum(r["minilm_rank"] is not None for r in rows),
        "classic_lost_after_recall": sum(r["coverage"]["40"] and r["classic_rank"] is None for r in rows),
        "minilm_lost_after_recall": sum(r["coverage"]["40"] and r["minilm_rank"] is None for r in rows),
        "threshold_dropped_pairs_at_40": sum(r["threshold_dropped_at_40"] for r in rows)}


def analyze(ranks_path, output):
    corpus, snapshot, baseline, repair, cases = inputs()
    full = read(ranks_path)
    require(full["status"] == "complete" and full["source_only"]
            and full["corpus_sha256"] == corpus["dataset_sha256"]
            and full["source_sha256"] == digest(DIAGNOSIS / "sources.json")
            and full["query_cache_sha256"] == digest(REPLAY / "query-vectors.json"), "Exact rank snapshot mismatch")
    ranks = {r["case_id"]: r["ranks"] for r in full["cases"]}
    require(set(ranks) == {c["evaluation_case_id"] for c in cases}, "Exact rank cases differ")
    audits = {r["case_id"]: r for r in baseline["audit"]}
    classic = {r["case_id"]: r for r in baseline["arms"]["classic_source"]["cases"]}
    mini = {r["case_id"]: r for r in repair["arms"]["max_mmr"]["cases"]}
    mini_inputs = {r["case_id"]: r for r in repair["diagnostics"]}
    rows = []
    chunks = snapshot["chunks"]
    for case in cases:
        cid = case["evaluation_case_id"]
        ordered = ranks[cid]
        require(len(ordered) == len(chunks) and {i for i, _ in ordered} == set(chunks), "Incomplete full ranks")
        require(all(math.isfinite(d) and -1e-6 <= d <= 2.000001 for _, d in ordered), "Invalid cosine distance")
        require(ordered == sorted(ordered, key=lambda p: (p[1], UUID(p[0]).int)), "Exact rank order is invalid")
        frozen = [r for r in audits[cid]["candidates"] if r["source"]]
        require([r["id"] for r in frozen] == [i for i, _ in ordered[:40]], "Frozen source-40 membership/order differs")
        require(all(abs(old["source_distance"] - distance) < 1e-9
                    for old, (_, distance) in zip(frozen, ordered[:40], strict=True)), "Frozen distances differ")
        hits = [hit(i, d, chunks, case) for i, d in ordered]
        admitted = [h for h in hits[:40] if 1-h.cosine_distance >= MIN_SIMILARITY]
        require(all(h.label_match == old["historical_label_match"] for h, old in zip(hits[:40], frozen, strict=True)),
                "Frozen evidence labels differ")
        require(len(admitted) == audits[cid]["source_count"], "Admitted source count differs")
        require(row_result(case, [s.hit for s in classic_order(case["question"], admitted)])["final_ids"]
                == classic[cid]["final_ids"], "Classic ranking reproduction failed")
        require({str(h.index_chunk_id) for h in admitted} == {c["id"] for c in mini_inputs[cid]["candidates"]},
                "MiniLM was scored on a different candidate pool")
        require(all(i in {str(h.index_chunk_id) for h in admitted} for i in mini[cid]["final_ids"]),
                "MiniLM final result contains an unscored source")
        by_id = {str(h.index_chunk_id): h for h in admitted}
        require(row_result(case, [by_id[i] for i in mini[cid]["final_ids"]]) == mini[cid],
                "MiniLM frozen rank metrics differ from source labels")
        labelled = [{"id": str(h.index_chunk_id), "rank": n, "similarity": 1-h.cosine_distance,
                     "location": h.source_location} for n, h in enumerate(hits, 1) if h.label_match]
        filename = Path(case["document_path"]).name
        doc_hits = [h for h in hits if h.source_metadata.get("original_filename") == filename]
        doc_rank = next((n for n, h in enumerate(doc_hits, 1) if h.label_match), None)
        row = {"case_id": cid, "base_case_id": case.get("base_case_id"), "group": case["group"],
            "dataset": case["source_dataset"], "question_type": case["question_type"],
            "query": case["question"], "target_filename": filename,
            "labelled_chunks": labelled, "best_label_rank": labelled[0]["rank"] if labelled else None,
            "oracle_within_document_rank": doc_rank,
            "target_document_chunks_at_40": sum(h in doc_hits for h in hits[:40]),
            "coverage": {str(k): any(h.label_match and 1-h.cosine_distance >= MIN_SIMILARITY
                                     for h in hits[:k]) for k in DEPTHS},
            "threshold_dropped_at_40": len(hits[:40])-len(admitted),
            "classic_rank": classic[cid]["relevant_rank"], "minilm_rank": mini[cid]["relevant_rank"]}
        rows.append(row)
    originals = {r["case_id"]: r for r in rows if r["group"] == "direct"}
    paraphrases = [r for r in rows if r["group"] == "paraphrase"]
    pairs = [{"original": p["base_case_id"], "paraphrase": p["case_id"],
              "original_best_rank": originals[p["base_case_id"]]["best_label_rank"],
              "paraphrase_best_rank": p["best_label_rank"],
              "original_in_40": originals[p["base_case_id"]]["coverage"]["40"],
              "paraphrase_in_40": p["coverage"]["40"]} for p in paraphrases]
    result = {"status": "complete", "source_only": True, "model_calls": 0, "database_writes": 0,
        "qa_generation_calls": 0, "corpus_sha256": corpus["dataset_sha256"],
        "inputs": {str(p.relative_to(ROOT)): digest(p) for p in [ranks_path, DIAGNOSIS / "sources.json",
            DIAGNOSIS / "analysis.json", REPAIR / "repaired-int8.json"]},
        "verified_source40_queries": len(rows), "verified_classic_rankings": len(rows),
        "summary": {g: summarize([r for r in rows if r["group"] == g]) for g in ["direct", "paraphrase"]},
        "paired_originals": summarize([originals[p["base_case_id"]] for p in paraphrases]),
        "dataset_summary": {g: {d: summarize([r for r in rows if r["group"] == g and r["dataset"] == d])
            for d in sorted({r["dataset"] for r in rows if r["group"] == g})} for g in ["direct", "paraphrase"]},
        "paired_coverage_transitions": dict(Counter(f'{p["original_in_40"]}->{p["paraphrase_in_40"]}' for p in pairs)),
        "pairs": pairs, "cases": rows,
        "limitations": ["Historical page/overlap labels measure any labelled evidence, not complete answer coverage.",
            "Depth slices are recall diagnostics, not new final-ranking/answer evaluations.",
            "Within-document ranks use gold scope only as an oracle; no deployable routing claim."]}
    write(output, result)
    print(json.dumps({k: result[k] for k in ["verified_source40_queries", "summary", "paired_coverage_transitions"]},
                     ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-ranks", action="store_true")
    parser.add_argument("--ranks", type=Path, default=OUTPUT / "exact-ranks.json")
    parser.add_argument("--output", type=Path, default=OUTPUT / "analysis.json")
    args = parser.parse_args()
    if args.export_ranks:
        asyncio.run(export_ranks(args.ranks))
    analyze(args.ranks, args.output)


if __name__ == "__main__":
    main()
