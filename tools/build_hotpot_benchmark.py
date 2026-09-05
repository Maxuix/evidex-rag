"""Build and validate the frozen 1,000-question HotpotQA slice, offline."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "evaluation/hotpotqa-1000-v1"
DEFAULT_SOURCE = ROOT / ".runtime/evaluations/hotpot-token-pilot-20260905/hotpot_dev_distractor_v1.json"
ARTIFACTS = ("questions.jsonl", "cases.jsonl", "documents.jsonl", "hotpot_dev_1000.json")
NOTICES = ("README.md", "SOURCE_NOTICES.md", "LICENSE-data.txt", "selection.json",
           "scoring/hotpot_evaluate_v1.py", "scoring/LICENSE.txt")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(value, message):
    if not value:
        raise ValueError(message)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def derive(rows):
    """Only title and original context text enter ingestible documents."""
    docs, cases, questions = {}, [], []
    for row in rows:
        refs, contexts = [], {}
        for title, sentences in row["context"]:
            require(title not in contexts, "Ambiguous duplicate title in one question")
            body = title + "\n\n" + "".join(sentences) + "\n"
            digest = sha(body.encode())
            doc_id = "hotpot-" + digest
            docs[doc_id] = {"document_id": doc_id, "path": f"documents/{digest}.txt",
                            "title": title, "sha256": digest, "text": body}
            refs.append(doc_id)
            contexts[title] = (doc_id, sentences)
        evidence = []
        for title, index in row["supporting_facts"]:
            require(title in contexts, "Evidence document missing")
            doc_id, sentences = contexts[title]
            require(type(index) is int and 0 <= index < len(sentences), "Evidence sentence out of bounds")
            quote = sentences[index]
            start = len(title) + 2 + sum(len(s) for s in sentences[:index])
            evidence.append({"document_id": doc_id, "title": title, "sentence_index": index,
                             "char_start": start, "char_end": start + len(quote), "quote": quote})
        questions.append({"case_id": row["_id"], "question": row["question"]})
        cases.append({"case_id": row["_id"], "question": row["question"], "answer": row["answer"],
                      "type": row["type"], "level": row["level"], "answerable": True,
                      "context_document_ids": refs,
                      "required_document_ids": sorted({e["document_id"] for e in evidence}),
                      "supporting_facts": evidence})
    return questions, cases, [docs[k] for k in sorted(docs)]


def build(root, source):
    if (root / "manifest.json").exists():
        return validate(root)
    require(not (root / "documents").exists(), "Partial output exists; inspect before rebuilding")
    selection = read(root / "selection.json")
    raw = source.read_bytes()
    require(sha(raw) == selection["source_sha256"], "Source SHA-256 mismatch")
    all_rows = json.loads(raw)
    by_id = {r["_id"]: r for r in all_rows}
    ids = selection["case_ids"]
    require(len(all_rows) == len(by_id) == 7405, "Invalid source inventory")
    require(len(ids) == len(set(ids)) == 1000, "Exactly 1,000 unique IDs required")
    require(not set(ids) & set(selection["excluded_invalid_evidence_ids"]), "Invalid excluded case selected")
    original = [by_id[i] for i in ids]
    questions, cases, docs = derive(original)
    for notice in NOTICES:
        require((root / notice).is_file(), f"Required notice missing: {notice}")
    (root / "documents").mkdir()
    for doc in docs:
        (root / doc["path"]).write_text(doc["text"], encoding="utf-8")
    jsonl(root / "questions.jsonl", questions)
    jsonl(root / "cases.jsonl", cases)
    jsonl(root / "documents.jsonl", [{k:v for k,v in d.items() if k != "text"} for d in docs])
    dump(root / "hotpot_dev_1000.json", original)
    manifest = {"dataset": selection["dataset"], "version": 1, "license": "CC BY-SA 4.0",
                "source_sha256": selection["source_sha256"], "seed": selection["seed"],
                "questions": len(cases), "types": dict(Counter(c["type"] for c in cases)),
                "documents": len(docs), "document_bytes": sum(len(d["text"].encode()) for d in docs),
                "supporting_fact_annotations": sum(len(c["supporting_facts"]) for c in cases),
                "coordinate_system": "zero-based Python Unicode character offsets, end-exclusive; sentence indices are upstream zero-based",
                "protocol": "shared KB: all selected contexts including distractors; original distractor contexts also retained",
                "artifacts": {p:sha((root / p).read_bytes()) for p in (*ARTIFACTS, *NOTICES)}}
    dump(root / "manifest.json", manifest)
    return validate(root)


def validate(root):
    manifest = read(root / "manifest.json")
    selection = read(root / "selection.json")
    for path, digest in manifest["artifacts"].items():
        require(sha((root / path).read_bytes()) == digest, f"Artifact changed: {path}")
    original = read(root / "hotpot_dev_1000.json")
    require([r["_id"] for r in original] == selection["case_ids"], "Question identity/order changed")
    require(len(original) == len(set(selection["case_ids"])) == 1000, "Question count changed")
    questions, cases, docs = derive(original)
    require(load_rows(root / "questions.jsonl") == questions, "Questions changed")
    require(load_rows(root / "cases.jsonl") == cases, "Gold/evidence mapping changed")
    expected_meta = [{k:v for k,v in d.items() if k != "text"} for d in docs]
    require(load_rows(root / "documents.jsonl") == expected_meta, "Document catalog changed")
    expected = {d["path"] for d in docs}
    actual = {str(p.relative_to(root)) for p in (root / "documents").rglob("*") if p.is_file()}
    require(actual == expected, "Missing or extra ingestible documents")
    by_id = {d["document_id"]: d for d in docs}
    for doc in docs:
        data = (root / doc["path"]).read_bytes()
        require(data == doc["text"].encode() and sha(data) == doc["sha256"], "Document content changed")
    for case in cases:
        for e in case["supporting_facts"]:
            require(by_id[e["document_id"]]["text"][e["char_start"]:e["char_end"]] == e["quote"], "Evidence span mismatch")
    require(manifest["documents"] == len(docs), "Document count changed")
    require(manifest["types"] == dict(Counter(c["type"] for c in cases)), "Type counts changed")
    return {k:manifest[k] for k in ("dataset","questions","types","documents","document_bytes","supporting_fact_annotations")}


def pack(root):
    validate(root)
    files = [*ARTIFACTS, *NOTICES, "manifest.json"]
    files += [d["path"] for d in load_rows(root / "documents.jsonl")]
    path = root / "benchmark.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(root.name + "/" + name, date_time=(2026, 9, 5, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (root / name).read_bytes())
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, "ZIP integrity failed")
    (root / "benchmark.zip.sha256").write_text(sha(path.read_bytes()) + "  benchmark.zip\n")
    return {"path":str(path),"bytes":path.stat().st_size,"sha256":sha(path.read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "validate", "pack"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    result = build(args.root, args.source) if args.mode == "build" else (
        validate(args.root) if args.mode == "validate" else pack(args.root))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
