"""Offline safeguards for the paid pilot's corpus and usage boundaries."""
import asyncio
import json

import httpx
import pytest

from tools.evaluate_hotpot_tokens import EmbeddingMeter, documents, summarize, write


def test_shared_corpus_keeps_distractors_without_qa_labels():
    rows = [{"question": "private-question", "answer": "private-answer",
             "context": [["Evidence", ["Public fact."]], ["Distractor", ["Other fact."]]]},
            {"context": [["Evidence", ["Public fact."]], ["Evidence", ["Different version."]]]}]
    result = documents(rows)
    assert len(result) == 3
    assert {r["title"] for r in result} == {"Evidence", "Distractor"}
    assert "private-question" not in json.dumps(result)
    assert "private-answer" not in json.dumps(result)


def test_embedding_meter_records_usage_preserves_response_and_restores_client(tmp_path):
    original = httpx.AsyncClient.send
    body = {"data": [{"embedding": [0.1, 0.2]}],
            "usage": {"prompt_tokens": 23, "total_tokens": 23, "secret": "omit"}}

    async def exercise():
        meter = EmbeddingMeter(tmp_path)
        with meter.installed():
            async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=body))) as client:
                response = await client.post("https://example.test/v1/embeddings", json={
                    "model": "qwen3.7-text-embedding", "input": ["private-document"]})
                assert response.json() == body
        assert meter.calls[0]["usage"] == {"prompt_tokens": 23, "total_tokens": 23}

    asyncio.run(exercise())
    assert httpx.AsyncClient.send is original
    saved = (tmp_path / "embedding-usage.json").read_text()
    assert "private-document" not in saved
    assert "embedding\"" not in saved
    assert "secret" not in saved


def test_missing_embedding_usage_is_not_reported_as_zero(tmp_path):
    async def exercise():
        meter = EmbeddingMeter(tmp_path)
        with meter.installed():
            async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"data": []}))) as client:
                await client.post("https://example.test/v1/embeddings", json={
                    "model": "qwen3.7-text-embedding", "input": ["sample"]})
        assert meter.calls[0]["usage"] is None

    asyncio.run(exercise())


def test_projection_uses_target_strata_and_counts_embedding_once(tmp_path, capsys):
    write(tmp_path / "manifest.json", {
        "pilot": {"questions":2,"estimated_source_tokens": 100},
        "target": {"questions":1000,"types":{"bridge":799,"comparison":201},
                   "estimated_source_tokens":10000}})
    rows = {}
    for kind, prompt, completion in [("bridge",9,1),("comparison",27,3)]:
        rows[kind] = {"status":"completed","type":kind,
                      "usage":{"calls":{"1":{}},"totals":{
                          "prompt_tokens":prompt,"completion_tokens":completion,
                          "total_tokens":prompt+completion}}}
    write(tmp_path / "state.json", {"results":rows,"documents":{"a":{"ready":True}}})
    write(tmp_path / "embedding-usage.json", [
        {"phase":"indexing","status":200,"usage":{"total_tokens":200}},
        {"phase":"answering","status":200,"usage":{"total_tokens":10}}])
    summarize(tmp_path)
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["pilot_total_tokens"] == 250
    projection = report["target_projection"]
    assert projection["chat_total_tokens"] == 14020
    assert projection["index_embedding_tokens"] == 20000
    assert projection["query_embedding_tokens"] == 5000
    assert projection["first_run_total_tokens"] == 39020
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["pilot"]["questions"] = 3
    write(tmp_path / "manifest.json", manifest)
    with pytest.raises(RuntimeError, match="Incomplete pilot"):
        summarize(tmp_path)
    summarize(tmp_path, allow_partial=True)
    partial = json.loads((tmp_path / "summary.json").read_text())
    assert partial["status"] != "complete"
    assert "chat_bootstrap_95_percent_interval" not in partial["target_projection"]
