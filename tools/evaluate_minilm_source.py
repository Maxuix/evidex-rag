#!/usr/bin/env python3
"""Source-only MiniLM diagnosis using fixed candidates and a pinned float reference.

No database, embedding provider, LLM or QA generator is initialized. Runtime/input
variants are diagnostic; original source evidence and historical labels stay frozen.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from uuid import UUID

import onnxruntime as ort

from rag_kb.adapters.local_reranker import LocalMiniLmTokenizer, build_local_rerank_windows, _infer_windows, _sigmoid
from rag_kb.adapters.local_reranker_artifacts import verify_local_reranker_artifacts
from rag_kb.domain import Evidence, ModelRerankScore, RerankDocument, RerankMode, RetrievalQueryPlan, RetrievalStrategy
from rag_kb.ports.model_api import RerankerAdapterError
from rag_kb.retrieval.service import RetrievalService, rerank_document_from_evidence
from tools.analyze_auto_qa_ranking import classic_order, rank_with_scores, require, row_result, summarize
from tools.build_document_qa_corpus import validate_corpus
from tools.evaluate_auto_qa_retrieval import _evaluation_cases, evidence_matches
from tools.evaluate_auto_qa_reuse import _write, paired_changes

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSIS = ROOT / ".runtime/evaluations/auto-qa-diagnosis-20260905"
OUTPUT = ROOT / ".runtime/evaluations/minilm-source-20260905"
REFERENCE_SHA256 = "3e9a03ed1e966f7c5288dd4230e3d6a9bf5e3a170a06f1f4241c5bca12c6487c"
REVISION = "1427fd652930e4ba29e8149678df786c240d8825"


def document_contexts(chunks):
    contexts = {}
    for chunk in sorted(chunks.values(), key=lambda value: value["ordinal"]):
        metadata = chunk["source_metadata"]
        contexts.setdefault(metadata["document_id"], metadata["original_filename"] + "\n" + chunk["content"].split("\n\n")[0])
    return contexts


def project_document(hit, contexts, variant):
    if variant == "repaired":
        return rerank_document_from_evidence(hit, contexts[hit.source_metadata["document_id"]])
    document = RerankDocument(index_chunk_id=hit.index_chunk_id, text=hit.text,
                             hierarchy=hit.hierarchy, modality=hit.modality)
    if variant == "context":
        document = replace(document, hierarchy={"titles": [
            {"text": contexts[hit.source_metadata["document_id"]]}, *hit.hierarchy.get("titles", [])]})
    elif variant != "production":
        raise ValueError("unknown input variant")
    return document


class ReferenceScorer:
    def __init__(self, arguments):
        self.backend = arguments.backend
        self.path = (arguments.reference / "onnx/model.onnx" if self.backend == "fp32"
                     else arguments.tokenizer / "onnx/model_qint8_arm64.onnx")
        self.model_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        require(self.backend == "int8" or self.model_sha256 == REFERENCE_SHA256, "Pinned float reference hash mismatch")
        manifest = json.loads((arguments.reference / "reference-manifest.json").read_text())
        require(manifest["revision"] == REVISION, "Wrong reference model revision")
        verify_local_reranker_artifacts(arguments.tokenizer, ROOT / "config/local-reranker-artifacts-v1.json")
        self.tokenizer = LocalMiniLmTokenizer.load(arguments.tokenizer)
        self.cache_path = arguments.output_root / f"scores-{arguments.variant}-{self.backend}.json"
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.cache_only = arguments.cache_only
        self.window_code = hashlib.sha256((ROOT / "src/rag_kb/adapters/local_reranker.py").read_bytes()).hexdigest()
        self.session = None
        self.new_pairs = 0
        self.new_windows = 0
        self.seconds = 0.0
        self.window_batch_size = 8 if self.backend == "int8" else arguments.window_batch_size

    def key(self, query, document):
        payload = [getattr(self, "model_sha256", REFERENCE_SHA256), self.window_code, ort.__version__,
                   query, str(document.index_chunk_id), document.text, document.hierarchy, document.modality,
                   document.document_context, getattr(self, "batch_signature", ""),
                   getattr(self, "window_batch_size", 1)]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def score(self, query, documents):
        if getattr(self, "backend", "fp32") == "int8":
            # Dynamic quantization is batch-sensitive: cache the complete ordered request.
            self.batch_signature = ""
            self.batch_signature = hashlib.sha256("".join(self.key(query, doc) for doc in documents).encode()).hexdigest()
        missing = [doc for doc in documents if self.key(query, doc) not in self.cache]
        if missing:
            require(not self.cache_only, "Missing reference score; cache-only run cannot infer")
            if getattr(self, "backend", "fp32") == "int8":
                missing = list(documents)
            if self.session is None:
                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                options.inter_op_num_threads = 1
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(str(self.path), sess_options=options, providers=["CPUExecutionProvider"])
            pending = [missing[start:start+20] for start in range(0, len(missing), 20)]
            while pending:
                batch = pending.pop(0)
                windows = build_local_rerank_windows(self.tokenizer, query, tuple(batch))
                if len(windows) > 80:
                    require(len(batch) >= 2, "Single document exceeds runtime window budget")
                    midpoint = len(batch) // 2
                    pending[:0] = [batch[:midpoint], batch[midpoint:]]
                    continue
                started = time.perf_counter()
                logits = tuple(value for offset in range(0, len(windows), self.window_batch_size)
                    for value in _infer_windows(self.session, windows[offset:offset+self.window_batch_size], self.tokenizer.pad_token_id))
                self.seconds += time.perf_counter() - started
                require(len(logits) == len(windows), "Reference output cardinality differs")
                require(all(math.isfinite(value) for value in logits), "Non-finite inference output")
                grouped = defaultdict(list)
                for window, value in zip(windows, logits, strict=True):
                    grouped[window.index_chunk_id].append(value)
                for doc in batch:
                    values = grouped[doc.index_chunk_id]
                    require(bool(values), "Missing reference document windows")
                    self.cache[self.key(query, doc)] = {"logits": values, "window_count": len(values),
                                                       "window_batch_size": self.window_batch_size}
                self.new_pairs += len(batch)
                self.new_windows += len(windows)
            _write(self.cache_path, self.cache)
        result = {doc.index_chunk_id: self.cache[self.key(query, doc)] for doc in documents}
        for value in result.values():
            logits = value.get("logits")
            require(isinstance(logits, list) and bool(logits) and len(logits) == value.get("window_count")
                    and all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in logits),
                    "Invalid cached reference window scores")
        return result


def oracle_scope_diagnostic(cases, diagnostics):
    """Gold-document filter is an explanatory upper-bound control, never runtime routing."""
    definitions = {case["evaluation_case_id"]: case for case in cases}
    arms = {"classic": [], "model": []}
    for row in diagnostics:
        case = definitions[row["case_id"]]
        if case["group"] not in {"direct", "paraphrase"}:
            continue
        filename = Path(case["document_path"]).name
        candidates = [item for item in row["candidates"] if item["filename"] == filename]
        for name, selected in (("classic", candidates),
                ("model", sorted(candidates, key=lambda item: -max(item["logits"])))):
            hits = [SimpleNamespace(index_chunk_id=UUID(item["id"]), label_match=item["label"]) for item in selected]
            arms[name].append(row_result(case, hits))
    return {"oracle": True, "deployable": False,
            "warning": "Gold document scope only; no new recall. This is not a measured runtime improvement.",
            "arms": {name: {"summary": summarize(rows), "cases": rows} for name, rows in arms.items()}}


async def verify_service_case(query, initial, documents, scores, contexts, expected_ids):
    """Exercise real context projection/batching/ordering using cached local logits."""
    workspace, kb, revision = UUID(int=1), UUID(int=2), UUID(int=3)
    expected = {doc.index_chunk_id: doc for doc in documents}
    introductions = {UUID(hit.source_metadata["document_version_id"]):
                     contexts[hit.source_metadata["document_id"]] for hit in initial}
    received = []

    class Store:
        async def rerank_document_contexts(self, **scope):
            require(scope["workspace_id"] == workspace and scope["knowledge_base_id"] == kb
                    and scope["index_revision_id"] == revision, "Service context scope differs")
            return {id: introductions[id] for id in scope["indexed_document_version_ids"]}

    class Adapter:
        profile = RerankMode.LOCAL_MINILM_V1
        max_documents = 20

        async def score(self, actual_query, batch):
            require(actual_query == query, "Service query changed")
            if sum(scores[doc.index_chunk_id]["window_count"] for doc in batch) > 80:
                raise RerankerAdapterError("local_reranker_window_limit")
            result = []
            for doc in batch:
                require(doc == expected[doc.index_chunk_id], "Service model input differs from evaluated input")
                received.append(doc.index_chunk_id)
                values = scores[doc.index_chunk_id]["logits"]
                raw = max(values)
                result.append(ModelRerankScore(doc.index_chunk_id, _sigmoid(raw), raw,
                                               len(values), values.index(raw)))
            return tuple(result)

    candidates = tuple(Evidence(rank=rank, index_chunk_id=hit.index_chunk_id,
        indexed_document_version_id=UUID(hit.source_metadata["document_version_id"]),
        document_id=UUID(hit.source_metadata["document_id"]),
        document_version_id=UUID(hit.source_metadata["document_version_id"]),
        index_revision_id=revision, ordinal=0, text=hit.text, hierarchy=hit.hierarchy,
        source_metadata=hit.source_metadata, source_location=hit.source_location,
        score=1-hit.cosine_distance, modality=hit.modality) for rank, hit in enumerate(initial, 1))
    service = RetrievalService(workspace, None, Store(), text_reranker=Adapter())
    plan = RetrievalQueryPlan(workspace, kb, RetrievalStrategy.EXACT_VECTOR,
                             top_k=10, candidate_count=40, rerank_mode=RerankMode.LOCAL_MINILM_V1)
    ordered, count, _ = await service._finish_reranking(query, candidates, plan)
    require(count == len(documents) and received == [doc.index_chunk_id for doc in documents],
            "Runtime batch membership/order differs")
    require([str(item.index_chunk_id) for item in ordered] == expected_ids, "Runtime final ranking differs")


def run(arguments):
    snapshot = json.loads((DIAGNOSIS / "sources.json").read_text())
    baseline = json.loads((DIAGNOSIS / "analysis.json").read_text())
    corpus = validate_corpus(arguments.corpus_root)
    require(snapshot["corpus_sha256"] == baseline["corpus_sha256"] == corpus["dataset_sha256"], "Frozen corpus identity mismatch")
    cases = _evaluation_cases(arguments.corpus_root)
    require(len(cases) == baseline["completed_cases"] == 103, "A full 103-case replay is required")
    chunks = snapshot["chunks"]
    contexts = document_contexts(chunks)
    if arguments.variant == "repaired":
        contexts = {}
        for chunk in sorted(chunks.values(), key=lambda value: value["ordinal"]):
            if chunk["modality"] in {"text", "table"} and chunk["content"].strip():
                contexts.setdefault(chunk["source_metadata"]["document_id"], chunk["content"][:2048])
    audit_by_id = {row["case_id"]: row for row in baseline["audit"]}
    originals = {row["case_id"]: row for row in baseline["arms"]["classic_source"]["cases"]}
    scorer = ReferenceScorer(arguments)
    arms = defaultdict(list)
    diagnostics = []
    result = {"schema_version": "minilm_source_reference_v1", "variant": arguments.variant,
        "reference_revision": REVISION, "reference_sha256": scorer.model_sha256, "backend": arguments.backend,
        "source_only": True, "qa_generation_calls": 0, "remote_model_calls": 0,
        "corpus_sha256": corpus["dataset_sha256"], "window_code_sha256": scorer.window_code,
        "onnxruntime_version": ort.__version__, "completed_cases": 0, "arms": {}, "diagnostics": diagnostics,
        "service_replay_verified_case_ids": []}
    for position, case in enumerate(cases, 1):
        cid, query = case["evaluation_case_id"], case["question"]
        hits = []
        for old in audit_by_id[cid]["candidates"]:
            if not old["source"]:
                continue
            if 1 - old["source_distance"] < .35:
                continue
            source = chunks[old["id"]]
            hit = SimpleNamespace(index_chunk_id=UUID(old["id"]), text=source["content"],
                hierarchy=source["hierarchy"], modality=source["modality"], source_metadata=source["source_metadata"],
                source_location=source["source_location"], cosine_distance=old["source_distance"],
                label_match=old["historical_label_match"])
            require(hit.modality in {"text", "table"}, "Unexpected image in frozen text-only candidate lane")
            require(hit.label_match == evidence_matches(case, hit), "Frozen source label mismatch")
            hits.append(hit)
        require(len(hits) == audit_by_id[cid]["source_count"], "Source candidate membership differs")
        classic_scored = classic_order(query, hits)
        initial = [item.hit for item in classic_scored]
        require(row_result(case, initial)["final_ids"] == originals[cid]["final_ids"], "Classic reproduction differs")
        documents = tuple(project_document(hit, contexts, arguments.variant) for hit in initial)
        scores = scorer.score(query, documents)
        require(set(scores) == {hit.index_chunk_id for hit in hits}, "Reference score identities differ")
        for pooling in ("max", "mean", "first"):
            logits = {id: (max(value["logits"]) if pooling == "max" else
                         sum(value["logits"])/len(value["logits"]) if pooling == "mean" else value["logits"][0])
                      for id, value in scores.items()}
            probabilities = {id: _sigmoid(value) for id, value in logits.items()}
            for mmr in (False, True):
                name = f"{pooling}_{'mmr' if mmr else 'raw'}"
                arms[name].append(row_result(case, rank_with_scores(initial, probabilities, mmr=mmr)))
        for weight in (.1, .2):
            # Two fixed diagnostic weights, shared with the preceding investigation.
            # No per-query/case/group tuning and no changes to production defaults.
            fused = {item.hit.index_chunk_id: (1-weight)*item.score + weight*_sigmoid(max(scores[item.hit.index_chunk_id]["logits"]))
                     for item in classic_scored}
            arms[f"fusion_{int(weight*100)}"].append(row_result(case, rank_with_scores(initial, fused, mmr=False)))
        if arguments.verify_service and cid in {"cfqa-101", "para-cfqa-89", "financebench_id_00678"}:
            require(arguments.variant == "repaired", "Service verification requires repaired inputs")
            asyncio.run(verify_service_case(query, initial, documents, scores, contexts,
                                            arms["max_mmr"][-1]["final_ids"]))
            result["service_replay_verified_case_ids"].append(cid)
        diagnostics.append({"case_id": cid, "group": case["group"], "query": query,
            "candidates": [{"id": str(hit.index_chunk_id), "label": hit.label_match,
                "filename": hit.source_metadata["original_filename"], "modality": hit.modality,
                "logits": scores[hit.index_chunk_id]["logits"]} for hit in initial]})
        result["completed_cases"] = position
        result["arms"] = {name: {"cases": rows} for name, rows in arms.items()}
        _write(arguments.output_root / f"{arguments.variant}-{arguments.backend}.json", result)
        print(json.dumps({"case": position, "total": len(cases), "id": cid,
                          "new_pairs": scorer.new_pairs, "inference_seconds": round(scorer.seconds, 2)}), flush=True)
    for name, rows in arms.items():
        result["arms"][name].update(summary=summarize(rows),
            vs_classic=paired_changes(list(originals.values()), rows),
            vs_cached_int8=paired_changes(baseline["arms"]["model_raw_source"]["cases"], rows))
    result.update(status="complete", new_inference_pairs=scorer.new_pairs,
                  new_inference_windows=scorer.new_windows, inference_seconds=scorer.seconds,
                  cached_inference_pairs=len(scorer.cache),
                  batch_sizes_present=sorted({row.get("window_batch_size", 8) for row in scorer.cache.values()}))
    result["oracle_document_scope"] = oracle_scope_diagnostic(cases, diagnostics)
    _write(arguments.output_root / f"{arguments.variant}-{arguments.backend}.json", result)
    print(json.dumps({name: values["summary"] for name, values in result["arms"].items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("production", "context", "repaired"), default="repaired")
    parser.add_argument("--backend", choices=("fp32", "int8"), default="fp32")
    parser.add_argument("--reference", type=Path, default=ROOT / ".runtime/model-assets/minilm-reference-1427fd6")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / ".runtime/model-assets/local-reranker")
    parser.add_argument("--corpus-root", type=Path, default=ROOT / "evaluation/document-qa-v1")
    parser.add_argument("--output-root", type=Path, default=ROOT / ".runtime/evaluations/minilm-repair-20260905")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--verify-service", action="store_true",
                        help="Verify three representative runtime context/batching/ranking cases from cached logits")
    parser.add_argument("--window-batch-size", type=int, choices=(1, 8), default=1,
                        help="Float-only batching; the probe checks batch/padding agreement against native PyTorch")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
