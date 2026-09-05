"""Bounded HotpotQA token pilot through the current API and native worker.

Uses the existing owner-only host runtime, without starting background consumers.
Dataset preparation is offline; run performs paid calls and stops on any failure.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time
from uuid import NAMESPACE_URL, uuid5

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / ".runtime/evaluations/hotpot-token-pilot-20260905"
SOURCE_URL = "https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
DOWNLOAD_URL = "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def documents(cases):
    result = {}
    for case in cases:
        for title, sentences in case["context"]:
            text = title + "\n\n" + "".join(sentences) + "\n"
            key = digest(text.encode())
            result[key] = {"id": key, "title": title, "text": text}
    return [result[key] for key in sorted(result)]


def prepare(root, source):
    from rag_kb.tokenizer import get_cl100k_base_encoding

    require(not (root / "manifest.json").exists(), "Manifest already exists; reuse it")
    raw = source.read_bytes()
    data = json.loads(raw)
    require(len(data) == 7405, "Unexpected HotpotQA dev size")
    require(len({r["_id"] for r in data}) == len(data), "Duplicate case IDs")
    invalid_evidence_ids = []
    for row in data:
        require(row["type"] in {"bridge", "comparison"}, "Unknown question type")
        by_title = {title: sentences for title, sentences in row["context"]}
        require(isinstance(row["question"], str) and isinstance(row["answer"], str), "Invalid QA")
        if any(title not in by_title or not 0 <= index < len(by_title[title])
               for title, index in row["supporting_facts"]):
            invalid_evidence_ids.append(row["_id"])
    eligible = [row for row in data if row["_id"] not in invalid_evidence_ids]
    # Proportional strata, random order independent of answers and system output.
    rng = random.Random(20260905)
    groups = {kind: sorted([r for r in eligible if r["type"] == kind], key=lambda r: r["_id"])
              for kind in ("bridge", "comparison")}
    for group in groups.values():
        rng.shuffle(group)
    comparison_1000 = round(len(groups["comparison"]) / len(eligible) * 1000)
    full = groups["bridge"][:1000-comparison_1000] + groups["comparison"][:comparison_1000]
    comparison_20 = round(comparison_1000 / 50)
    pilot = groups["bridge"][:20-comparison_20] + groups["comparison"][:comparison_20]
    rng.shuffle(pilot)
    encoding = get_cl100k_base_encoding()
    manifest = {"source_url": SOURCE_URL, "download_url": DOWNLOAD_URL, "source_sha256": digest(raw), "seed": 20260905,
                "license": "CC BY-SA 4.0", "source_count": len(data),
                "excluded_invalid_evidence_ids": invalid_evidence_ids,
                "protocol": "shared corpus from all selected context paragraphs; original questions",
                "local_token_estimator": "cl100k_base (not provider billing tokenizer)"}
    for name, cases in [("pilot", pilot), ("target", full)]:
        docs = documents(cases)
        write(root / f"{name}-cases.json", cases)
        write(root / f"{name}-documents.json", docs)
        manifest[name] = {"questions": len(cases), "types": dict(Counter(c["type"] for c in cases)),
                          "documents": len(docs), "utf8_bytes": sum(len(d["text"].encode()) for d in docs),
                          "estimated_source_tokens": sum(len(encoding.encode(d["text"], disallowed_special=())) for d in docs),
                          "case_ids": [c["_id"] for c in cases],
                          "cases_sha256": digest((root / f"{name}-cases.json").read_bytes()),
                          "documents_sha256": digest((root / f"{name}-documents.json").read_bytes())}
    write(root / "manifest.json", manifest)
    print(json.dumps({k: {x:y for x,y in manifest[k].items() if x != 'case_ids'} for k in ['pilot','target']}), flush=True)


class EmbeddingMeter:
    """Observe validated numeric usage only; never retain HTTP bodies or vectors."""
    def __init__(self, root):
        self.root = root
        self.phase = "preflight"
        self.case_id = None
        self.calls = read(root / "embedding-usage.json") if (root / "embedding-usage.json").exists() else []

    @contextmanager
    def installed(self):
        import httpx
        original = httpx.AsyncClient.send
        meter = self

        async def measured(client, request, *args, **kwargs):
            is_embedding = request.url.path.endswith("/embeddings")
            response = await original(client, request, *args, **kwargs)
            if is_embedding:
                await response.aread()
                payload = json.loads(request.content)
                require(payload.get("model") == "qwen3.7-text-embedding", "Unexpected embedding model")
                usage = response.json().get("usage") if response.is_success else None
                safe = {k: v for k, v in (usage or {}).items()
                        if k in {"prompt_tokens", "total_tokens"} and type(v) is int and v >= 0}
                row = {"phase": meter.phase, "case_id": meter.case_id,
                       "status": response.status_code, "usage": safe or None,
                       "input_count": len(payload["input"]) if isinstance(payload.get("input"), list) else 1}
                meter.calls.append(row)
                write(meter.root / "embedding-usage.json", meter.calls)
            return response

        httpx.AsyncClient.send = measured
        try:
            yield self
        finally:
            httpx.AsyncClient.send = original


async def assert_idle(worker):
    from sqlalchemy import text
    async with worker.database.sessions() as session:
        await session.execute(text("SET TRANSACTION READ ONLY"))
        for table in ("chat_run", "indexing_job"):
            count = await session.scalar(text(f"SELECT count(*) FROM {table} WHERE status::text IN ('queued','running','processing')"))
            require(count == 0, f"Unrelated or unfinished {table} work exists; inspect before continuing")


async def run(root, limit):
    import httpx
    from apps.api.app import create_app
    from apps.worker.dependencies import build_worker_dependencies
    from rag_kb.config import load_settings
    from tools.evaluation_runtime import load_evaluation_runtime

    manifest = read(root / "manifest.json")
    require(1 <= limit <= manifest["pilot"]["questions"], "Pilot limited to 20 cases")
    for name in ("cases", "documents"):
        require(digest((root / f"pilot-{name}.json").read_bytes()) == manifest["pilot"][f"{name}_sha256"], "Frozen input changed")
    runtime = load_evaluation_runtime()
    settings = load_settings(env_file=runtime.env_file)
    state_path = root / "state.json"
    state = read(state_path) if state_path.exists() else {"documents": {}, "runs": {}, "results": {}}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    patch = subprocess.check_output(["git", "diff", "--", "src", "apps"], cwd=ROOT)
    if state.get("knowledge_base"):
        require(state["code_commit"] == commit and state["working_code_diff_sha256"] == digest(patch),
                "Application code changed during the pilot")
    state["code_commit"] = commit
    state["working_code_diff_sha256"] = digest(patch)
    write(state_path, state)
    app = create_app(settings=settings)
    worker = build_worker_dependencies(settings)
    meter = EmbeddingMeter(root)
    stopped = asyncio.Event()
    with meter.installed():
        try:
            async with app.router.lifespan_context(app):
                await worker.start()
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://pilot/api/v1/", timeout=120) as client:
                    async def api(method, path, **kwargs):
                        headers = kwargs.get("headers", {})
                        if "Idempotency-Key" in headers:
                            headers["Idempotency-Key"] = str(uuid5(NAMESPACE_URL, headers["Idempotency-Key"]))
                        response = await client.request(method, path, **kwargs)
                        require(response.is_success, f"API {method} {path}: {response.status_code}")
                        return response.json()

                    profiles = await api("GET", "model-settings")
                    required = {"chat": "mimo-v2.5", "text_embedding": "qwen3.7-text-embedding",
                                "multimodal_embedding": "tongyi-embedding-vision-flash-2026-03-06"}
                    selected = {}
                    providers = {p["id"]:p for p in profiles["providers"]}
                    for kind, model in required.items():
                        matches = [p for p in profiles["profiles"] if p["kind"] == kind and p["model"] == model and p["enabled"] and p["validation_status"] == "valid" and p["provider_secret_available"]]
                        require(len(matches) == 1, f"Required model unavailable: {model}")
                        selected[kind] = matches[0]
                    chat_provider = providers[selected["chat"]["provider_id"]]
                    require(chat_provider["base_url"].rstrip('/') == "https://opencode.ai/zen/go/v1", "OpenCode Go required")
                    models = {k: {f:v[f] for f in ["model", "revision_id", "parameters"]} for k,v in selected.items()}
                    require(not state.get("knowledge_base") or state["models"] == models, "Model parameters changed during the pilot")
                    state["models"] = models
                    write(state_path, state)
                    await assert_idle(worker)
                    if "knowledge_base" not in state:
                        state["knowledge_base"] = await api("POST", "knowledge-bases", headers={"Idempotency-Key": "hotpot-token-pilot-20260905"}, json={
                            "name": "HotpotQA token pilot 20260905", "embedding": {"strategy": "text_only", "text_profile_revision_id": selected["text_embedding"]["revision_id"]}})
                        write(state_path, state)
                    kb = state["knowledge_base"]["id"]
                    meter.phase = "indexing"
                    docs = read(root / "pilot-documents.json")
                    for index, doc in enumerate(docs, 1):
                        key = doc["id"]
                        if key in state["documents"] and state["documents"][key].get("ready"):
                            continue
                        require(key not in state["documents"], "Unfinished document requires inspection")
                        filename = "hotpot-" + key[:20] + ".txt"
                        metadata = base64.urlsafe_b64encode(json.dumps({"v":1,"filename":filename,"display_name":doc["title"]}).encode()).decode().rstrip('=')
                        uploaded = await api("POST", f"knowledge-bases/{kb}/documents", content=doc["text"].encode(), headers={"Content-Type":"text/plain","X-Document-Metadata":metadata,"Idempotency-Key":"hotpot-doc-"+key[:40]})
                        state["documents"][key] = uploaded
                        write(state_path, state)
                        lease = await worker.indexing_scheduler.claim_once()
                        require(lease is not None and str(lease.job_id) == uploaded["job_id"], "Unexpected indexing lease")
                        await worker.indexing_scheduler.execute(lease, stopped)
                        detail = await api("GET", "documents/"+uploaded["document"]["id"])
                        require(detail.get("index", {}).get("build_status") == "ready", "Indexing failed; stop for investigation")
                        state["documents"][key]["ready"] = True
                        state["documents"][key]["index"] = detail["index"]
                        write(state_path, state)
                        if index % 10 == 0 or index == len(docs):
                            print(json.dumps({"event":"indexed","done":index,"total":len(docs)}), flush=True)
                    meter.phase = "answering"
                    for index, case in enumerate(read(root / "pilot-cases.json")[:limit], 1):
                        key = case["_id"]
                        if key in state["results"]:
                            require(state["results"][key]["status"] == "completed", "Earlier failed case requires investigation")
                            continue
                        require(key not in state["runs"], "Unfinished run requires inspection")
                        meter.case_id = key
                        session = await api("POST", "chat/sessions", json={"knowledge_base_id":kb,"title":"Hotpot pilot "+key})
                        created = await api("POST", "chat/runs", headers={"Idempotency-Key":"hotpot-run-"+key}, json={"session_id":session["id"],"knowledge_base_id":kb,"message":case["question"],"model_profile_revision_id":selected["chat"]["revision_id"],"retrieval":{"mode":"auto","top_k":10,"rerank_mode":"classic"}})
                        run_id = created["run_id"]
                        state["runs"][key] = run_id
                        write(state_path, state)
                        started = time.perf_counter()
                        lease = await worker.chat_scheduler.claim_once()
                        require(lease is not None and str(lease.run_id) == run_id, "Unexpected chat lease")
                        await worker.chat_scheduler.execute(lease, stopped)
                        terminal = await api("GET", "chat/runs/"+run_id)
                        # API output is already validated; no raw provider response is persisted.
                        row = {"case_id":key,"type":case["type"],"question":case["question"],"gold_answer":case["answer"],
                               "elapsed_seconds":round(time.perf_counter()-started,3),
                               **{k:terminal.get(k) for k in ["status","answer","citations","agent","usage","timing","error","model","retrieval"]}}
                        state["results"][key] = row
                        write(state_path, state)
                        print(json.dumps({"event":"answered","done":index,"case_id":key,"status":row["status"],"usage":row["usage"],"seconds":row["elapsed_seconds"]}), flush=True)
                        require(row["status"] == "completed", "Model run failed; stop, do not substitute provider or parameters")
                        require(isinstance(row["usage"], dict), "Model token usage unavailable")
                    print(json.dumps({"event":"pilot_complete","cases":len(state["results"])}), flush=True)
        finally:
            await worker.close()


def summarize(root, *, allow_partial=False):
    manifest, state = read(root / "manifest.json"), read(root / "state.json")
    embeddings = read(root / "embedding-usage.json")
    attempted = list(state["results"].values())
    rows = [r for r in attempted if r["status"] == "completed"]
    incomplete = [r for r in attempted if r["status"] != "completed"]
    complete = len(rows) == manifest["pilot"]["questions"] and not incomplete
    require(complete or allow_partial, "Incomplete pilot; use --allow-partial only for a labelled preliminary report")
    require(all(c["status"] == 200 and c["usage"] is not None for c in embeddings), "Embedding accounting incomplete")
    totals = []
    for row in rows:
        usage = row["usage"]["totals"]
        require(all(type(usage.get(k)) is int for k in ["prompt_tokens","completion_tokens","total_tokens"]), "Chat accounting incomplete")
        require(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "Chat token total mismatch")
        totals.append(usage["total_tokens"])
    index_tokens = sum(c["usage"]["total_tokens"] for c in embeddings if c["phase"] == "indexing")
    query_tokens = sum(c["usage"]["total_tokens"] for c in embeddings if c["phase"] == "answering")
    by_type = {}
    for kind, weight in manifest["target"]["types"].items():
        subset = [r for r in rows if r["type"] == kind]
        require(subset, f"No pilot samples for {kind}")
        by_type[kind] = {"n":len(subset), "target_n":weight,
                         "prompt_mean":statistics.mean(r["usage"]["totals"]["prompt_tokens"] for r in subset),
                         "completion_mean":statistics.mean(r["usage"]["totals"]["completion_tokens"] for r in subset),
                         "total_mean":statistics.mean(r["usage"]["totals"]["total_tokens"] for r in subset)}
    source_ratio = manifest["target"]["estimated_source_tokens"] / manifest["pilot"]["estimated_source_tokens"]
    projection = {"chat_prompt_tokens":round(sum(v["target_n"]*v["prompt_mean"] for v in by_type.values())),
                  "chat_completion_tokens":round(sum(v["target_n"]*v["completion_mean"] for v in by_type.values())),
                  "index_embedding_tokens":round(index_tokens*source_ratio),
                  "query_embedding_tokens":round(query_tokens/len(rows)*manifest["target"]["questions"])}
    projection["chat_total_tokens"] = projection["chat_prompt_tokens"] + projection["chat_completion_tokens"]
    projection["first_run_total_tokens"] = projection["chat_total_tokens"] + projection["index_embedding_tokens"] + projection["query_embedding_tokens"]
    # Stratified bootstrap estimates sampling uncertainty only, not corpus-size effects.
    rng = random.Random(20260905)
    replicates = []
    for _ in range(5000):
        value = 0
        for kind, weight in manifest["target"]["types"].items():
            samples = [r["usage"]["totals"]["total_tokens"] for r in rows if r["type"] == kind]
            value += weight * statistics.mean(rng.choices(samples,k=len(samples)))
        replicates.append(value)
    replicates.sort()
    projection["chat_bootstrap_95_percent_interval"] = [round(replicates[125]),round(replicates[4874])]
    if not complete:
        projection.pop("chat_bootstrap_95_percent_interval")
    report = {"status":"complete" if complete else "partial_provider_unavailable",
              "planned_questions":manifest["pilot"]["questions"], "attempted_questions":len(attempted),
              "incomplete_runs":[{"case_id":r["case_id"], "status":r["status"],"error":r["error"],
                                  "known_usage":r.get("usage"),"unreported_request_tokens":None} for r in incomplete],
              "pilot_questions":len(rows), "pilot_ready_documents":sum(d.get("ready",False) for d in state["documents"].values()),
              "pilot_chat_prompt_tokens":sum(r["usage"]["totals"]["prompt_tokens"] for r in rows),
              "pilot_chat_completion_tokens":sum(r["usage"]["totals"]["completion_tokens"] for r in rows),
              "pilot_chat_total_tokens":sum(totals), "pilot_index_embedding_tokens":index_tokens,
              "pilot_query_embedding_tokens":query_tokens,
              "pilot_total_tokens":sum(totals)+index_tokens+query_tokens,
              "pilot_model_calls":sum(len(r["usage"]["calls"]) for r in rows),
              "chat_tokens_per_question":{"mean":statistics.mean(totals),"median":statistics.median(totals),"min":min(totals),"max":max(totals)},
              "by_type":by_type,"target_projection":projection,
              "limitations":["Pilot uses 200 documents; target has 9793, so harder retrieval may change Agent rounds.",
                             "Index projection scales provider-reported pilot usage by local cl100k source-token ratio; it is an estimate.",
                             "Bootstrap interval covers sample variation only; token totals are not monetary costs.",
                             "No Graph build, Auto-QA generation, or LLM Judge was requested for this text-only pilot."]}
    if not complete:
        report["limitations"].append("Incomplete pilot: projection uses successful cases only and omits unknown failed-request usage; not a reliable final budget.")
    write(root / "summary.json",report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare","run","summarize"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    require(args.root.is_dir() and args.root.stat().st_mode & 0o777 == 0o700, "Owner-only pilot directory required")
    if args.mode == "prepare":
        prepare(args.root, args.source or args.root / "hotpot_dev_distractor_v1.json")
    elif args.mode == "run":
        asyncio.run(run(args.root, args.limit))
    else:
        summarize(args.root, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
