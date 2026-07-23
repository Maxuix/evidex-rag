#!/usr/bin/env python3
"""Run a generated composite-v2 corpus against an isolated local RAG stack."""

from __future__ import annotations

import argparse
import base64
import hashlib
from http.client import HTTPResponse
import io
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches
from PIL import Image, ImageDraw, ImageFont


MEDIA_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api",
        default="http://127.0.0.1:8000/api/v1",
        help="loopback API base URL ending in /api/v1",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="run retrieval evaluation without provider-backed ChatRun cases",
    )
    arguments = parser.parse_args()
    if not 1 <= arguments.top_k <= 20:
        parser.error("--top-k must be between 1 and 20")
    if arguments.timeout_seconds <= 0 or arguments.poll_seconds <= 0:
        parser.error("timeouts must be positive")

    try:
        api = _validated_api_base(arguments.api)
    except ValueError as error:
        parser.error(str(error))

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="rag-kb-real-evaluation-") as directory:
        root = Path(directory)
        corpus = _generate_corpus(root)
        report = _evaluate(
            api,
            corpus,
            top_k=arguments.top_k,
            timeout_seconds=arguments.timeout_seconds,
            poll_seconds=arguments.poll_seconds,
            evaluate_chat=not arguments.skip_chat,
        )
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    succeeded = metrics.get("all_required_cases_recalled") is True
    if not arguments.skip_chat:
        succeeded = succeeded and metrics.get("all_required_chat_completed") is True
        succeeded = succeeded and metrics.get("chat_attachment_expectations_met") is True
    return 0 if succeeded else 1


def _evaluate(
    api: str,
    corpus: dict[str, Path],
    *,
    top_k: int,
    timeout_seconds: float,
    poll_seconds: float,
    evaluate_chat: bool,
) -> dict[str, object]:
    kb = _json_request(
        f"{api}/knowledge-bases",
        method="POST",
        headers={"Idempotency-Key": str(uuid4())},
        payload={
            "name": f"multimodal-real-eval-{uuid4().hex[:10]}",
            "parsing": {"preset": "multimodal_local_v1"},
            "chunking": {"preset": "structural_balanced_v2"},
        },
    )
    kb_id = _required_string(kb, "id")
    documents: dict[str, str] = {}
    indexing: dict[str, dict[str, object]] = {}

    for label, path in corpus.items():
        upload = _upload(api, kb_id, label, path)
        document = upload.get("document")
        if not isinstance(document, dict):
            raise RuntimeError("upload response omitted document")
        documents[label] = _required_string(document, "id")
        job_id = _required_string(upload, "job_id")
        job, elapsed = _wait_for_job(
            api,
            job_id,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        indexing[label] = {
            "status": job.get("status"),
            "phase": job.get("phase"),
            "build_status": job.get("build_status"),
            "serving_status": job.get("serving_status"),
            "error_code": (
                job.get("error", {}).get("code")
                if isinstance(job.get("error"), dict)
                else None
            ),
            "elapsed_seconds": round(elapsed, 3),
            "source_bytes": path.stat().st_size,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if label != "resource_stress" and job.get("status") != "completed":
            raise RuntimeError(f"required corpus item failed indexing: {label}")

    cases = (
        {
            "case_key": "architecture_text",
            "query": "Which component receives approved requests from the Aster control plane, as shown in Fig. 7?",
            "relevant_label": "architecture_pdf",
            "lane": "text",
            "expects_visual": True,
            "expected_relation": "explicit_figure_reference",
            "evaluate_chat": True,
        },
        {
            "case_key": "architecture_visual",
            "query": "Which document contains a violet hexagon linked to three green circles?",
            "relevant_label": "architecture_pdf",
            "lane": "image",
            "expects_visual": True,
            "expected_relation": None,
            "evaluate_chat": True,
        },
        {
            "case_key": "scan_ocr",
            "query": "Find the document containing NEBULA-OCR-4821.",
            "relevant_label": "scanned_pdf",
            "lane": "text",
            "expects_visual": True,
            "expected_relation": None,
            "evaluate_chat": False,
        },
        {
            "case_key": "docx_table",
            "query": "What latency is listed for the Quartz region?",
            "relevant_label": "rich_docx",
            "lane": "text",
            "expects_visual": False,
            "expected_relation": None,
            "evaluate_chat": True,
        },
        {
            "case_key": "docx_layout",
            "query": "What color is the needle in the Quartz regional compass?",
            "relevant_label": "rich_docx",
            "lane": "image",
            "expects_visual": True,
            "expected_relation": None,
            "evaluate_chat": True,
        },
        {
            "case_key": "long_text",
            "query": "What checkpoint code is assigned to the Lantern archive?",
            "relevant_label": "long_text",
            "lane": "text",
            "expects_visual": False,
            "expected_relation": None,
            "evaluate_chat": True,
        },
        {
            "case_key": "watermark_text",
            "query": "What is the Zephyr retention period?",
            "relevant_label": "repeated_watermark",
            "lane": "text",
            "expects_visual": False,
            "expected_relation": None,
            "evaluate_chat": False,
        },
    )
    if indexing["resource_stress"]["status"] == "completed":
        cases += (
            {
                "case_key": "stress_table",
                "query": "What value belongs to STRESS-ROW-079?",
                "relevant_label": "resource_stress",
                "lane": "text",
                "expects_visual": False,
                "expected_relation": None,
                "evaluate_chat": False,
            },
        )

    results: list[dict[str, object]] = []
    for case in cases:
        case_key = str(case["case_key"])
        query = str(case["query"])
        relevant_label = str(case["relevant_label"])
        lane = str(case["lane"])
        query_started = time.perf_counter()
        response = _json_request(
            f"{api}/retrieval/query",
            method="POST",
            payload={
                "knowledge_base_id": kb_id,
                "query": query,
                "top_k": top_k,
                "strategy": "exact_vector",
                "rerank": True,
                "include_debug": True,
            },
        )
        elapsed = time.perf_counter() - query_started
        evidence = response.get("evidence")
        if not isinstance(evidence, list):
            raise RuntimeError("retrieval response omitted evidence")
        expected_document_id = documents[relevant_label]
        rank = next(
            (
                index
                for index, item in enumerate(evidence, start=1)
                if isinstance(item, dict)
                and item.get("document_id") == expected_document_id
            ),
            None,
        )
        relevant_evidence = [
            item
            for item in evidence
            if isinstance(item, dict)
            and item.get("document_id") == expected_document_id
        ]
        related_visuals = [
            visual
            for item in relevant_evidence
            for visual in item.get("related_visuals", [])
            if isinstance(item.get("related_visuals"), list)
            and isinstance(visual, dict)
        ]
        direct_visual_count = sum(
            item.get("modality") in {"image", "table"}
            for item in relevant_evidence
        )
        predicted_visual = bool(related_visuals or direct_visual_count)
        expected_relation = case["expected_relation"]
        relation_matched = expected_relation is None or any(
            item.get("relation_type") == expected_relation
            for item in related_visuals
        )
        results.append(
            {
                "case_key": case_key,
                "lane": lane,
                "relevant_document": relevant_label,
                "rank": rank,
                "recalled": rank is not None,
                "group_recalled": rank is not None and relation_matched,
                "expects_visual": bool(case["expects_visual"]),
                "predicted_visual": predicted_visual,
                "relation_matched": relation_matched,
                "related_visual_count": len(related_visuals),
                "direct_visual_count": direct_visual_count,
                "elapsed_seconds": round(elapsed, 3),
                "result_count": len(evidence),
                "top_modalities": [
                    item.get("modality")
                    for item in evidence
                    if isinstance(item, dict)
                ],
                "top_matched_representations": [
                    item.get("matched_representations")
                    for item in evidence
                    if isinstance(item, dict)
                ],
            }
        )

    required = [item for item in results if item["case_key"] != "stress_table"]
    text_cases = [item for item in required if item["lane"] == "text"]
    image_cases = [item for item in required if item["lane"] == "image"]
    chat_results = (
        _evaluate_chat_cases(
            api,
            kb_id,
            documents,
            cases,
            top_k=top_k,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        if evaluate_chat
        else []
    )
    retrieval_attachment = _attachment_metrics(required)
    chat_attachment = _attachment_metrics(chat_results)
    metrics = {
        "top_k": top_k,
        "text_recall_at_k": _recall(text_cases),
        "text_mrr": _mrr(text_cases),
        "image_recall_at_k": _recall(image_cases),
        "image_mrr": _mrr(image_cases),
        "group_recall_at_k": _group_recall(required),
        "retrieval_visual_attachment_precision": retrieval_attachment["precision"],
        "retrieval_visual_attachment_accuracy": retrieval_attachment["accuracy"],
        "retrieval_visual_count": sum(
            int(item["related_visual_count"]) + int(item["direct_visual_count"])
            for item in required
        ),
        "all_required_cases_recalled": all(item["recalled"] for item in required),
        "retrieval_call_count": len(results),
        "retrieval_elapsed_seconds": round(
            sum(float(item["elapsed_seconds"]) for item in results), 3
        ),
        "chat_case_count": len(chat_results),
        "chat_visual_attachment_precision": chat_attachment["precision"],
        "chat_visual_attachment_accuracy": chat_attachment["accuracy"],
        "chat_attached_image_count": sum(
            int(item["attached_image_count"]) for item in chat_results
        ),
        "chat_incorrect_visual_count": sum(
            int(item["incorrect_visual_count"]) for item in chat_results
        ),
        "chat_provider_call_count": sum(
            int(item["provider_call_count"]) for item in chat_results
        ),
        "all_required_chat_completed": (
            all(item["status"] == "completed" for item in chat_results)
            if evaluate_chat
            else None
        ),
        "chat_attachment_expectations_met": (
            all(
                bool(item["predicted_visual"]) == bool(item["expects_visual"])
                and int(item["incorrect_visual_count"]) == 0
                for item in chat_results
            )
            if evaluate_chat
            else None
        ),
    }
    return {
        "knowledge_base_id": kb_id,
        "corpus": {
            label: {
                "document_id": documents[label],
                **indexing[label],
            }
            for label in corpus
        },
        "cases": results,
        "chat_cases": chat_results,
        "metrics": metrics,
        "limitations": {
            "provider_token_usage_available": True,
            "chat_evaluated": evaluate_chat,
            "note": (
                "Retrieval elapsed time is measured at the public API boundary; "
                "the API does not expose per-lane provider timing or embedding "
                "token accounting. Asset-read failure is covered by deterministic "
                "unit tests rather than destructive corpus mutation."
            ),
        },
    }


def _evaluate_chat_cases(
    api: str,
    kb_id: str,
    documents: dict[str, str],
    cases: tuple[dict[str, object], ...],
    *,
    top_k: int,
    timeout_seconds: float,
    poll_seconds: float,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for case in cases:
        if not case["evaluate_chat"]:
            continue
        session = _json_request(
            f"{api}/chat/sessions",
            method="POST",
            payload={
                "knowledge_base_id": kb_id,
                "title": f"composite-eval-{case['case_key']}",
            },
        )
        session_id = _required_string(session, "id")
        created = _json_request(
            f"{api}/chat/runs",
            method="POST",
            headers={"Idempotency-Key": str(uuid4())},
            payload={
                "session_id": session_id,
                "knowledge_base_id": kb_id,
                "message": case["query"],
                "retrieval": {
                    "mode": "vector",
                    "top_k": top_k,
                    "rerank": True,
                },
            },
        )
        run_id = _required_string(created, "run_id")
        terminal, elapsed = _wait_for_chat_run(
            api,
            run_id,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        citations = terminal.get("citations")
        if not isinstance(citations, list):
            citations = []
        visual_citations = [
            item
            for item in citations
            if isinstance(item, dict)
            and item.get("modality") in {"image", "table"}
            and isinstance(item.get("asset"), dict)
        ]
        expected_document_id = documents[str(case["relevant_label"])]
        incorrect_visual_count = sum(
            item.get("document_id") != expected_document_id
            for item in visual_citations
        )
        attempt = _latest_attempt(terminal.get("timing"))
        visual_facts = attempt.get("visual_evidence", {})
        if not isinstance(visual_facts, dict):
            visual_facts = {}
        attached_count = int(visual_facts.get("attached_image_count", 0))
        calls = terminal.get("usage", {}).get("calls", {}) if isinstance(terminal.get("usage"), dict) else {}
        if not isinstance(calls, dict):
            calls = {}
        operations: dict[str, int] = {}
        for item in calls.values():
            if not isinstance(item, dict):
                continue
            operation = str(item.get("operation", "unknown"))
            operations[operation] = operations.get(operation, 0) + 1
        results.append(
            {
                "case_key": case["case_key"],
                "status": terminal.get("status"),
                "error_code": (
                    terminal.get("error", {}).get("code")
                    if isinstance(terminal.get("error"), dict)
                    else None
                ),
                "expects_visual": bool(case["expects_visual"]),
                "predicted_visual": attached_count > 0,
                "attached_image_count": attached_count,
                "visual_citation_count": len(visual_citations),
                "incorrect_visual_count": incorrect_visual_count,
                "provider_call_count": len(calls),
                "provider_call_operations": operations,
                "repair_attempted": bool(
                    attempt.get("validation", {}).get("repair_attempted", False)
                    if isinstance(attempt.get("validation"), dict)
                    else False
                ),
                "elapsed_seconds": round(elapsed, 3),
                "pipeline_duration_ms": attempt.get("duration_ms"),
                "retrieval_diagnostics": attempt.get("retrieval", {}),
                "visual_rejection_counts": visual_facts.get(
                    "rejection_counts", {}
                ),
            }
        )
    return results


def _wait_for_chat_run(
    api: str,
    run_id: str,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    while True:
        run = _json_request(f"{api}/chat/runs/{run_id}")
        if run.get("status") in {"completed", "failed", "cancelled"}:
            return run, time.perf_counter() - started
        if time.perf_counter() - started >= timeout_seconds:
            raise RuntimeError("ChatRun exceeded the evaluation timeout")
        time.sleep(poll_seconds)


def _latest_attempt(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("attempts"), dict):
        return {}
    attempts = value["attempts"]
    keys = [key for key in attempts if str(key).isdigit()]
    if not keys:
        return {}
    result = attempts[max(keys, key=lambda item: int(item))]
    return result if isinstance(result, dict) else {}


def _upload(api: str, kb_id: str, label: str, path: Path) -> dict[str, Any]:
    media_type = MEDIA_TYPES[path.suffix.lower()]
    metadata = base64.urlsafe_b64encode(
        json.dumps(
            {"v": 1, "filename": path.name, "display_name": label},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii").rstrip("=")
    return _json_request(
        f"{api}/knowledge-bases/{kb_id}/documents",
        method="POST",
        headers={
            "Content-Type": media_type,
            "Idempotency-Key": str(uuid4()),
            "X-Document-Metadata": metadata,
        },
        body=path.read_bytes(),
    )


def _wait_for_job(
    api: str,
    job_id: str,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    while True:
        job = _json_request(f"{api}/indexing-jobs/{job_id}")
        if job.get("status") in {"completed", "failed", "cancelled"}:
            return job, time.perf_counter() - started
        if time.perf_counter() - started >= timeout_seconds:
            raise RuntimeError("indexing job exceeded the evaluation timeout")
        time.sleep(poll_seconds)


def _json_request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: dict[str, object] | None = None,
    body: bytes | None = None,
) -> dict[str, Any]:
    request_headers = {"Accept": "application/json", **(headers or {})}
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, headers=request_headers, method=method)
    try:
        with urlopen(request, timeout=30) as response:
            return _read_json(response)
    except HTTPError as error:
        try:
            detail = _read_json(error)
            code = detail.get("code", "HTTP_ERROR")
        except (ValueError, TypeError):
            code = "HTTP_ERROR"
        raise RuntimeError(f"local API request failed with HTTP {error.code}: {code}") from error
    except URLError as error:
        raise RuntimeError("local API is unavailable") from error


def _read_json(response: HTTPResponse | HTTPError) -> dict[str, Any]:
    value = json.loads(response.read())
    if not isinstance(value, dict):
        raise ValueError("API response is not a JSON object")
    return value


def _required_string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise RuntimeError(f"API response omitted {key}")
    return item


def _validated_api_base(value: str) -> str:
    parsed = urlparse(value.rstrip("/"))
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/api/v1"
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("--api must be a loopback /api/v1 URL")
    try:
        parsed.port
    except ValueError as error:
        raise ValueError("--api contains an invalid port") from error
    return parsed.geturl()


def _recall(cases: list[dict[str, object]]) -> float:
    return round(sum(bool(item["recalled"]) for item in cases) / len(cases), 6)


def _mrr(cases: list[dict[str, object]]) -> float:
    return round(
        sum(1.0 / int(item["rank"]) for item in cases if item["rank"] is not None)
        / len(cases),
        6,
    )


def _group_recall(cases: list[dict[str, object]]) -> float:
    return round(
        sum(bool(item["group_recalled"]) for item in cases) / len(cases), 6
    )


def _attachment_metrics(cases: list[dict[str, object]]) -> dict[str, float]:
    if not cases:
        return {"precision": 0.0, "accuracy": 0.0}
    predicted = [item for item in cases if item["predicted_visual"]]
    true_positive = sum(bool(item["expects_visual"]) for item in predicted)
    correct = sum(
        bool(item["predicted_visual"]) == bool(item["expects_visual"])
        for item in cases
    )
    return {
        "precision": round(true_positive / len(predicted), 6) if predicted else 1.0,
        "accuracy": round(correct / len(cases), 6),
    }


def _generate_corpus(root: Path) -> dict[str, Path]:
    architecture_image = _architecture_image()
    architecture_pdf = root / "architecture-with-figure.pdf"
    _write_pdf(
        architecture_pdf,
        (
            "ASTER CONTROL PLANE ARCHITECTURE",
            "The Aster control plane sends approved requests to the Cobalt gateway.",
            "The gateway owns tenant routing and forwards work to isolated executors.",
            "As shown in Fig. 7, the gateway connects the control plane to three executor pools.",
            "Figure 7. A violet gateway hexagon linked to three green executor circles.",
        ),
        architecture_image,
        image_draws=((72, 330, 468, 292),),
    )

    scanned_pdf = root / "scanned-nebula.pdf"
    scan = Image.new("RGB", (1600, 1100), "white")
    draw = ImageDraw.Draw(scan)
    font = _font(72)
    draw.text((120, 220), "NEBULA-OCR-4821", fill="black", font=font)
    draw.text((120, 360), "Scanned maintenance certificate", fill="black", font=_font(46))
    draw.rectangle((100, 170, 1450, 500), outline="navy", width=8)
    scan.save(scanned_pdf, "PDF", resolution=150.0)

    rich_docx = root / "rich-quartz.docx"
    document = Document()
    document.add_heading("Quartz Regional Operations", level=1)
    document.add_paragraph("The following table is the approved latency register.")
    picture = root / "quartz-figure.png"
    _compass_image().save(picture, "PNG")
    document.add_picture(str(picture), width=Inches(3.5))
    caption = document.add_paragraph("Figure 1. Quartz regional compass")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Region"
    table.rows[0].cells[1].text = "Latency"
    for region, latency in (("Amber", "22 ms"), ("Quartz", "37 ms"), ("Indigo", "51 ms")):
        cells = table.add_row().cells
        cells[0].text = region
        cells[1].text = latency
    document.save(rich_docx)

    long_text = root / "long-lantern.txt"
    paragraphs = [
        f"Archive segment {index:03d} follows the standard checksum and retention procedure."
        for index in range(220)
    ]
    paragraphs.insert(
        137,
        "The Lantern archive uses checkpoint code LANTERN-CHECKPOINT-927 for recovery.",
    )
    long_text.write_text("\n\n".join(paragraphs), encoding="utf-8")

    repeated_watermark = root / "repeated-watermark.pdf"
    watermark = _watermark_image()
    _write_pdf(
        repeated_watermark,
        (
            "ZEPHYR RETENTION POLICY",
            "The Zephyr retention period is 45 days after final approval.",
            "Repeated logos are decorative and do not change the policy text.",
        ),
        watermark,
        image_draws=tuple(
            (72 + column * 150, 220 + row * 120, 96, 64)
            for row in range(3)
            for column in range(3)
        ),
    )

    resource_stress = root / "resource-stress.docx"
    stress = Document()
    stress.add_heading("Bounded Resource Stress", level=1)
    oversized = root / "oversized-resource.png"
    _oversized_image().save(oversized, "PNG", optimize=True)
    stress.add_picture(str(oversized), width=Inches(6.0))
    stress.add_paragraph("Figure 1. Deliberately oversized pixel dimensions")
    stress_table = stress.add_table(rows=1, cols=2)
    stress_table.rows[0].cells[0].text = "Key"
    stress_table.rows[0].cells[1].text = "Value"
    for index in range(180):
        cells = stress_table.add_row().cells
        cells[0].text = f"STRESS-ROW-{index:03d}"
        cells[1].text = f"bounded-value-{index:03d}"
    stress.save(resource_stress)

    return {
        "architecture_pdf": architecture_pdf,
        "scanned_pdf": scanned_pdf,
        "rich_docx": rich_docx,
        "long_text": long_text,
        "repeated_watermark": repeated_watermark,
        "resource_stress": resource_stress,
    }


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _architecture_image() -> Image.Image:
    image = Image.new("RGB", (900, 560), "white")
    draw = ImageDraw.Draw(image)
    center = (300, 280)
    radius = 120
    points = [
        (
            center[0] + radius * math.cos(math.pi / 3 * index),
            center[1] + radius * math.sin(math.pi / 3 * index),
        )
        for index in range(6)
    ]
    draw.polygon(points, fill=(125, 55, 190), outline="black")
    for node in ((650, 120), (720, 280), (650, 440)):
        draw.line((center, node), fill="black", width=12)
        draw.ellipse((node[0] - 55, node[1] - 55, node[0] + 55, node[1] + 55), fill=(35, 170, 70), outline="black", width=6)
    return image


def _compass_image() -> Image.Image:
    image = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((150, 40, 490, 380), outline="navy", width=12)
    draw.polygon(((320, 65), (370, 230), (320, 200), (270, 230)), fill="orange")
    draw.text((260, 170), "Q", fill="navy", font=_font(64))
    return image


def _watermark_image() -> Image.Image:
    image = Image.new("RGB", (300, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((15, 15, 285, 185), radius=35, outline=(160, 160, 160), width=8)
    draw.text((70, 65), "ZEPHYR", fill=(170, 170, 170), font=_font(38))
    return image


def _oversized_image() -> Image.Image:
    image = Image.new("RGB", (4500, 3800), (242, 246, 250))
    draw = ImageDraw.Draw(image)
    for offset in range(0, 4500, 300):
        draw.line((offset, 0, 4500 - offset, 3800), fill=(20, 90, 160), width=18)
    return image


def _write_pdf(
    path: Path,
    lines: tuple[str, ...],
    image: Image.Image,
    *,
    image_draws: tuple[tuple[int, int, int, int], ...],
) -> None:
    jpeg = io.BytesIO()
    image.save(jpeg, "JPEG", quality=90)
    image_bytes = jpeg.getvalue()
    text_commands = ["BT", "/F1 16 Tf", "72 735 Td"]
    for index, line in enumerate(lines):
        if index:
            text_commands.append("0 -28 Td")
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        text_commands.append(f"({escaped}) Tj")
    text_commands.append("ET")
    for x, y, width, height in image_draws:
        text_commands.append(f"q {width} 0 0 {height} {x} {y} cm /Im1 Do Q")
    content = "\n".join(text_commands).encode("ascii")
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> /XObject << /Im1 5 0 R >> >> /Contents 6 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            f"<< /Type /XObject /Subtype /Image /Width {image.width} "
            f"/Height {image.height} /ColorSpace /DeviceRGB /BitsPerComponent 8 "
            f"/Filter /DCTDecode /Length {len(image_bytes)} >>\nstream\n"
        ).encode("ascii")
        + image_bytes
        + b"\nendstream",
        f"<< /Length {len(content)} >>\nstream\n".encode("ascii")
        + content
        + b"\nendstream",
    )
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, value in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(value)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(bytes(output))


if __name__ == "__main__":
    raise SystemExit(main())
