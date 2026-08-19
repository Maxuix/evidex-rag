#!/usr/bin/env python3
"""Build and validate a routing-oriented RAG evaluation corpus.

The corpus is intentionally self-contained.  It snapshots the generated
Graph RAG hard cases and adds one-hop text, Markdown-table, and inline-chart
materials.  Each case declares the expected retrieval route without implying
that the application currently performs automatic query routing.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

import build_graph_rag_corpus as graph_corpus


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evaluation" / "routing-rag-v1"
CORPUS_SCHEMA = "routing_rag_corpus_v1"
CASE_SCHEMA = "routing_rag_case_v1"
GRAPH_SOURCE_ID = "graph-rag-v1"
EXPECTED_OUTCOMES = frozenset({"answered", "refused"})
NEGATIVE_CONTROL_KINDS = frozenset(
    {"contradicted", "closed_world_absence", "open_world_unanswerable"}
)


@dataclass(frozen=True)
class StandardDocument:
    document_id: str
    filename: str
    title: str
    format: str
    material_type: str
    content: str


def _route(
    route: str,
    *,
    mode: str | None,
    modality: str | None,
    reason: str,
    top_k: int | None = None,
    rerank_mode: str | None = None,
    parsing_preset: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "route": route,
        "reason": reason,
    }
    if mode is not None:
        runtime: dict[str, object] = {"mode": mode}
        if top_k is not None:
            runtime["top_k"] = top_k
        if rerank_mode is not None:
            runtime["rerank_mode"] = rerank_mode
        value["runtime_request"] = runtime
    if modality is not None:
        value["expected_evidence_modality"] = modality
    if parsing_preset is not None:
        value["required_parsing_preset"] = parsing_preset
    return value


def _standard_documents(chart_uris: dict[str, str]) -> tuple[StandardDocument, ...]:
    return (
        StandardDocument(
            "single-01",
            "single-01_service_catalog.md",
            "服务目录与支持承诺",
            "md",
            "table",
            """# 服务目录与支持承诺

本目录只描述单项服务的责任团队与支持承诺，下面四项是本目录的完整服务集合。各行均为独立事实，不能把产品名称相似的条目合并。

| 服务代号 | 责任团队 | 首次响应目标 | 升级联系人 |
| --- | --- | --- | --- |
| Atlas Sync | 数据交付组 | 30 分钟 | 唐铭 |
| Harbor Ledger | 财务平台组 | 2 小时 | 林见鹿 |
| Quartz Alert | 运行保障组 | 15 分钟 | 顾南乔 |
| Lumen Archive | 信息治理组 | 4 小时 | 苏晚舟 |

表中“首次响应目标”只指收到有效工单后的首次确认，不等同于问题修复时长。
""",
        ),
        StandardDocument(
            "single-02",
            "single-02_incident_sla.md",
            "事件响应分级表",
            "md",
            "table",
            """# 事件响应分级表

以下表格用于值班路由。事件级别由影响范围判定，责任团队以表中登记为准。

| 事件级别 | 首次响应 | 业务负责人 | 复盘时限 |
| --- | --- | --- | --- |
| P1 | 10 分钟 | 平台值班组 | 2 个工作日 |
| P2 | 30 分钟 | 应用保障组 | 5 个工作日 |
| P3 | 4 小时 | 服务运营组 | 10 个工作日 |
| P4 | 1 个工作日 | 需求协调组 | 不要求 |

本文件没有股权、任职或跨文件桥接关系，应命中单一表格证据。
""",
        ),
        StandardDocument(
            "single-03",
            "single-03_facility_capacity.md",
            "区域机房容量快照",
            "md",
            "table",
            """# 区域机房容量快照

容量数值为本季度末可用机柜，不包含已锁定但未交付的扩容资源。

| 区域 | 可用机柜 | 当前利用率 | 备用电源检查日 |
| --- | ---: | ---: | --- |
| 榆川 | 84 | 72% | 2026-06-18 |
| 嘉澜 | 46 | 81% | 2026-06-21 |
| 容城 | 112 | 63% | 2026-06-16 |
| 东港 | 39 | 88% | 2026-06-23 |

“利用率”是静态指标，不代表任何项目的所有权或服务关系。
""",
        ),
        StandardDocument(
            "single-04",
            "single-04_contract_calendar.md",
            "供应合同续约日历",
            "md",
            "table",
            """# 供应合同续约日历

本日历只记录合同管理动作。供应商名称与其他语料中的企业名称相似时，仍应以本表合同编号定位。

| 合同编号 | 供应商 | 续约通知截止日 | 合同管理员 |
| --- | --- | --- | --- |
| CT-2041 | 明舟物流有限公司 | 2026-09-05 | 沈砚 |
| CT-2042 | 澄海云网服务有限公司 | 2026-08-28 | 梁青禾 |
| CT-2043 | 北辰能源（东港）有限公司 | 2026-10-12 | 周既白 |
| CT-2044 | 西木商业管理有限公司 | 2026-11-03 | 顾南乔 |

合同编号是本文件中最可靠的检索锚点。
""",
        ),
        StandardDocument(
            "single-05",
            "single-05_policy_notes.txt",
            "归档策略说明",
            "txt",
            "text",
            """归档策略说明

“湖镜归档”策略适用于已关闭的审计导出。信息治理组规定：审计导出完成后保留 180 天，到期后进入人工复核队列。

若请求单上出现 ORB-61，值班人员应把它归入湖镜归档策略，而不是产品发布或基础设施容量流程。该代码的处理负责人是信息治理组。

本说明是单一政策事实，不需要检索企业关系链。
""",
        ),
        StandardDocument(
            "single-06",
            "single-06_release_notice.txt",
            "产品发布通知",
            "txt",
            "text",
            """产品发布通知

2026 年 7 月，Aster Console 4.2 由运行保障组发布到内部服务门户。该版本的变更编号为 REL-42A，主要增加告警确认记录导出功能。

发布通知同时指出，所有需要回滚的请求应提交给运行保障组；文中没有涉及图谱路径、合同主体或股权关系。
""",
        ),
        StandardDocument(
            "single-07",
            "single-07_regional_fulfillment_chart.md",
            "区域按时交付率图表",
            "md",
            "chart",
            f"""# 区域按时交付率图表

图 1 是 2025 年第二季度的区域按时交付率。具体百分比仅在图像中呈现，正文不会重复各区域数值。

![图 1：区域按时交付率柱状图]({chart_uris['regional_fulfillment']})

图表用于视觉证据路由，不应把它误判为跨文档实体路径问题。
""",
        ),
        StandardDocument(
            "single-08",
            "single-08_channel_mix_chart.md",
            "渠道结构图表",
            "md",
            "chart",
            f"""# 渠道结构图表

图 2 展示本季度渠道收入占比。各渠道的精确比例仅写在图像标签中，正文不提供同义改写或表格副本。

![图 2：渠道收入占比饼图]({chart_uris['channel_mix']})

该文档测试图片证据和视觉理解，不测试企业关系追踪。
""",
        ),
    )


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (position[0] - (right - left) / 2, position[1] - (bottom - top) / 2),
        text,
        font=font,
        fill=fill,
    )


def _render_bar_chart(values: tuple[tuple[str, int], ...]) -> bytes:
    image = Image.new("RGB", (1000, 620), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(34)
    label_font = _font(24)
    value_font = _font(22)
    _draw_centered(draw, (500, 42), "Q2 On-time Fulfillment (%)", title_font, "#172033")
    left, top, right, bottom = 120, 120, 920, 510
    draw.line((left, bottom, right, bottom), fill="#374151", width=3)
    draw.line((left, top, left, bottom), fill="#374151", width=3)
    for grid in (60, 70, 80, 90, 100):
        y = bottom - (grid - 50) / 50 * (bottom - top)
        draw.line((left, y, right, y), fill="#e5e7eb", width=1)
        _draw_centered(draw, (80, int(y)), str(grid), value_font, "#4b5563")
    colors = ("#2563eb", "#7c3aed", "#059669", "#ea580c")
    width = 105
    gap = 82
    for index, (label, value) in enumerate(values):
        x = left + 55 + index * (width + gap)
        y = bottom - (value - 50) / 50 * (bottom - top)
        draw.rounded_rectangle((x, y, x + width, bottom), radius=10, fill=colors[index % len(colors)])
        _draw_centered(draw, (x + width // 2, int(y) - 24), f"{value}%", value_font, "#111827")
        _draw_centered(draw, (x + width // 2, bottom + 34), label, label_font, "#111827")
    _draw_centered(draw, (500, 575), "Source: internal operations dashboard", value_font, "#6b7280")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _render_pie_chart(values: tuple[tuple[str, int], ...]) -> bytes:
    image = Image.new("RGB", (1000, 620), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(34)
    label_font = _font(25)
    _draw_centered(draw, (500, 42), "Quarterly Revenue Channel Mix", title_font, "#172033")
    box = (130, 130, 600, 600)
    colors = ("#0ea5e9", "#22c55e", "#f59e0b")
    start = -90.0
    for index, (label, value) in enumerate(values):
        end = start + 360.0 * value / 100.0
        draw.pieslice(box, start=start, end=end, fill=colors[index])
        start = end
    legend_x = 680
    for index, (label, value) in enumerate(values):
        y = 190 + index * 105
        draw.rounded_rectangle((legend_x, y, legend_x + 36, y + 36), radius=5, fill=colors[index])
        draw.text((legend_x + 54, y + 2), f"{label}: {value}%", font=label_font, fill="#111827")
    draw.text((680, 510), "Source: finance planning", font=_font(20), fill="#6b7280")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _data_uri(payload: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def _copy_graph_documents(output: Path) -> tuple[dict[str, str], dict[str, object]]:
    """Build the source corpus in a temp directory and snapshot its documents/gold."""

    documents_dir = output / "documents"
    gold_dir = output / "gold" / GRAPH_SOURCE_ID
    gold_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rag-kb-routing-graph-") as temporary:
        staged_root = Path(temporary) / GRAPH_SOURCE_ID
        graph_corpus.build(staged_root)
        source_manifest = json.loads((staged_root / "manifest.json").read_text(encoding="utf-8"))
        filename_by_document_id = {
            str(row["document_id"]): str(row["filename"])
            for row in source_manifest["documents"]
        }
        copied_filenames: dict[str, str] = {}
        for document_id, source_filename in filename_by_document_id.items():
            copied_filename = f"graph-{source_filename}"
            shutil.copy2(staged_root / "documents" / source_filename, documents_dir / copied_filename)
            copied_filenames[document_id] = copied_filename
        for filename in ("manifest.json", "entities.jsonl", "relations.jsonl", "cases.jsonl"):
            shutil.copy2(staged_root / filename, gold_dir / filename)
    return copied_filenames, source_manifest


def _graph_cases(copied_filenames: dict[str, str]) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for source in graph_corpus.CASES:
        if source["current_graph_support"] != "1_2_hop":
            continue
        answer_document_id = str(source["answer_document_id"])
        answer_relation_ids = tuple(str(item) for item in source["answer_relation_ids"])
        gold_path = tuple(str(item) for item in source["gold_path"])
        answer_locators = [
            {
                "kind": "graph_relation",
                "relation_id": relation_id,
                "document_filename": copied_filenames[answer_document_id],
                "source_case_id": source["case_id"],
            }
            for relation_id in answer_relation_ids
        ]
        path_context_locators = [
            {
                "kind": "graph_relation",
                "relation_id": relation_id,
                "document_filename": copied_filenames[answer_document_id],
                "source_case_id": source["case_id"],
            }
            for relation_id in gold_path
            if relation_id not in answer_relation_ids
        ]
        cases.append(
            {
                "schema": CASE_SCHEMA,
                "case_id": f"route-{source['case_id']}",
                "category": "graph_multi_hop",
                "question": source["question"],
                "expected_answer": source["expected_answer"],
                "answerable": True,
                "expected_outcome": "answered",
                "negative_control_kind": None,
                "expected_answer_aspects": [
                    {
                        "aspect_id": "answer",
                        "answer_variants": [source["expected_answer"]],
                    }
                ],
                "answer_gold_source_locators": answer_locators,
                "path_context_locators": path_context_locators,
                "forbidden_claims": [],
                "expected_route": _route(
                    "graph",
                    mode="graph",
                    modality="text",
                    top_k=10,
                    rerank_mode="classic",
                    reason="答案依赖跨文档显式实体路径，而不是单一 chunk。",
                ),
                "source": {
                    "corpus": GRAPH_SOURCE_ID,
                    "source_case_id": source["case_id"],
                    "document_filename": copied_filenames[answer_document_id],
                    "gold_path": list(gold_path),
                    "answer_relation_ids": list(answer_relation_ids),
                },
                "expected_answer_terms": [source["expected_answer"]],
            }
        )
    return cases


def _standard_cases(chart_hashes: dict[str, str]) -> list[dict[str, object]]:
    text_route = lambda reason: _route(
        "simple", mode="vector", modality="text", top_k=5, reason=reason
    )
    table_route = lambda reason: _route(
        "simple", mode="vector", modality="table", top_k=5, reason=reason
    )
    chart_route = lambda reason: _route(
        "simple",
        mode="vector",
        modality="image",
        top_k=5,
        parsing_preset="multimodal_local_v2",
        reason=reason,
    )
    def row(
        case_id: str,
        category: str,
        question: str,
        answer: str,
        route: dict[str, object],
        document_id: str,
        filename: str,
        locator: dict[str, object],
        *,
        answerable: bool = True,
        expected_outcome: str | None = None,
        negative_control_kind: str | None = None,
        expected_answer_aspects: list[dict[str, object]] | None = None,
        forbidden_claims: list[str] | None = None,
    ) -> dict[str, object]:
        resolved_outcome = expected_outcome or ("answered" if answerable else "refused")
        answer_locators = [
            {
                "kind": "document_locator",
                "document_id": document_id,
                "document_filename": filename,
                "locator": locator,
            }
        ] if resolved_outcome == "answered" else []
        return {
            "schema": CASE_SCHEMA,
            "case_id": case_id,
            "category": category,
            "question": question,
            "expected_answer": answer,
            "answerable": answerable,
            "expected_outcome": resolved_outcome,
            "negative_control_kind": negative_control_kind,
            "expected_answer_aspects": expected_answer_aspects or (
                [{"aspect_id": "answer", "answer_variants": [answer]}]
                if resolved_outcome == "answered"
                else []
            ),
            "answer_gold_source_locators": answer_locators,
            "path_context_locators": [],
            "forbidden_claims": forbidden_claims or [],
            "expected_route": route,
            "source": {
                "document_id": document_id,
                "document_filename": filename,
                "locator": locator,
            },
            "expected_answer_terms": [] if resolved_outcome == "refused" else [answer],
        }

    cases = [
        row(
            "table-001", "single_hop_table", "Atlas Sync 的首次响应目标是多少？", "30 分钟",
            table_route("答案是同一 Markdown 表格的一格。"), "single-01", "single-01_service_catalog.md",
            {"kind": "markdown_table", "row_key": "Atlas Sync", "column": "首次响应目标"},
        ),
        row(
            "table-002", "single_hop_table", "哪个团队负责 Quartz Alert？", "运行保障组",
            table_route("产品和负责人同处一行。"), "single-01", "single-01_service_catalog.md",
            {"kind": "markdown_table", "row_key": "Quartz Alert", "column": "责任团队"},
        ),
        row(
            "table-003", "single_hop_table", "P2 事件的复盘时限是什么？", "5 个工作日",
            table_route("事件级别、复盘时限均在分级表中。"), "single-02", "single-02_incident_sla.md",
            {"kind": "markdown_table", "row_key": "P2", "column": "复盘时限"},
        ),
        row(
            "table-004", "single_hop_table", "P1 事件由哪个业务负责人处理？", "平台值班组",
            table_route("单行表格查询，不需要关系扩展。"), "single-02", "single-02_incident_sla.md",
            {"kind": "markdown_table", "row_key": "P1", "column": "业务负责人"},
        ),
        row(
            "table-005", "single_hop_table", "容城机房当前有多少可用机柜？", "112",
            table_route("容量值在区域机房表的一行。"), "single-03", "single-03_facility_capacity.md",
            {"kind": "markdown_table", "row_key": "容城", "column": "可用机柜"},
        ),
        row(
            "table-006", "single_hop_table", "哪一地区的当前利用率为 88%？", "东港",
            table_route("按表格数值反查区域。"), "single-03", "single-03_facility_capacity.md",
            {"kind": "markdown_table", "row_key": "88%", "column": "区域"},
        ),
        row(
            "table-007", "single_hop_table", "合同 CT-2043 的续约通知截止日是哪天？", "2026-10-12",
            table_route("合同编号是唯一锚点。"), "single-04", "single-04_contract_calendar.md",
            {"kind": "markdown_table", "row_key": "CT-2043", "column": "续约通知截止日"},
        ),
        row(
            "table-008", "single_hop_table", "哪位合同管理员负责 CT-2041？", "沈砚",
            table_route("合同编号和管理员在同一表格行。"), "single-04", "single-04_contract_calendar.md",
            {"kind": "markdown_table", "row_key": "CT-2041", "column": "合同管理员"},
        ),
        row(
            "text-001", "single_hop_text", "湖镜归档策略的保留时长是多少？", "180 天",
            text_route("答案在一个 TXT 段落内。"), "single-05", "single-05_policy_notes.txt",
            {"kind": "paragraph", "anchor": "湖镜归档"},
        ),
        row(
            "text-002", "single_hop_text", "ORB-61 的处理负责人是哪个团队？", "信息治理组",
            text_route("请求代码和团队在同一段说明中。"), "single-05", "single-05_policy_notes.txt",
            {"kind": "paragraph", "anchor": "ORB-61"},
        ),
        row(
            "text-003", "single_hop_text", "Aster Console 4.2 的变更编号是什么？", "REL-42A",
            text_route("版本和变更编号是单文件事实。"), "single-06", "single-06_release_notice.txt",
            {"kind": "paragraph", "anchor": "Aster Console 4.2"},
        ),
        row(
            "text-004", "single_hop_text", "需要回滚的请求应提交给哪个团队？", "运行保障组",
            text_route("回滚规则直接写在发布通知中。"), "single-06", "single-06_release_notice.txt",
            {"kind": "paragraph", "anchor": "回滚"},
        ),
        row(
            "chart-001", "single_hop_chart", "图 1 中哪个区域的按时交付率最高？", "East（94%）",
            chart_route("数值只在柱状图中可见。"), "single-07", "single-07_regional_fulfillment_chart.md",
            {"kind": "inline_png", "chart_id": "regional_fulfillment", "sha256": chart_hashes["regional_fulfillment"]},
        ),
        row(
            "chart-002", "single_hop_chart", "图 1 中 West 与 North 的按时交付率相差多少个百分点？", "4 个百分点",
            chart_route("需要读取同一柱状图的两个标签后做简单差值。"), "single-07", "single-07_regional_fulfillment_chart.md",
            {"kind": "inline_png", "chart_id": "regional_fulfillment", "sha256": chart_hashes["regional_fulfillment"]},
        ),
        row(
            "chart-003", "single_hop_chart", "图 2 中 Direct 渠道的收入占比是多少？", "35%",
            chart_route("比例只出现在饼图图例中。"), "single-08", "single-08_channel_mix_chart.md",
            {"kind": "inline_png", "chart_id": "channel_mix", "sha256": chart_hashes["channel_mix"]},
        ),
        row(
            "chart-004", "single_hop_chart", "图 2 中占比最大的渠道是什么？", "Retail（40%）",
            chart_route("比较图例中的视觉数值。"), "single-08", "single-08_channel_mix_chart.md",
            {"kind": "inline_png", "chart_id": "channel_mix", "sha256": chart_hashes["channel_mix"]},
        ),
        row(
            "abstain-001", "negative_control", "服务目录中是否有名为 Nova Gate 的服务？", "不存在",
            _route("simple", mode="vector", modality=None, top_k=5, reason="语料没有该服务。"),
            "single-01", "single-01_service_catalog.md", {"kind": "closed_world_absence", "anchor": "Nova Gate"},
            answerable=True, expected_outcome="answered", negative_control_kind="closed_world_absence",
        ),
        row(
            "abstain-002", "negative_control", "北辰能源有限公司是否控股东港储能科技有限公司？", "无证据，应该拒答",
            _route("simple", mode="vector", modality=None, top_k=5, reason="近似名称不构成同一实体或控股关系。"),
            "doc16", "graph-16_noise_consumer_brand.txt", {"kind": "near_name_disambiguation"}, answerable=False,
            negative_control_kind="open_world_unanswerable",
        ),
        row(
            "abstain-003", "negative_control", "图 2 是否说明 Retail 渠道的占比低于 20%？", "否，Retail 渠道占比为 40%",
            _route("simple", mode="vector", modality="image", top_k=5, parsing_preset="multimodal_local_v2", reason="图像证据与断言相反。"),
            "single-08", "single-08_channel_mix_chart.md", {"kind": "contradicted_inline_png", "chart_id": "channel_mix"},
            answerable=True, expected_outcome="answered", negative_control_kind="contradicted",
            forbidden_claims=["Retail 渠道占比低于 20%"],
        ),
    ]
    return cases


def _assert_standard_documents(documents: tuple[StandardDocument, ...]) -> None:
    ids = [document.document_id for document in documents]
    filenames = [document.filename for document in documents]
    if len(ids) != len(set(ids)) or len(filenames) != len(set(filenames)):
        raise ValueError("standard document identifiers must be unique")
    if any(document.format not in {"md", "txt"} for document in documents):
        raise ValueError("routing corpus supports only Markdown and text documents")


def build(output: Path, *, force: bool = False) -> None:
    if output.exists() and force:
        for child in output.iterdir():
            if child.name in {"README.md", ".gitkeep"}:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output.mkdir(parents=True, exist_ok=True)
    documents_dir = output / "documents"
    assets_dir = output / "assets"
    documents_dir.mkdir(exist_ok=True)
    assets_dir.mkdir(exist_ok=True)

    chart_payloads = {
        "regional_fulfillment": _render_bar_chart(
            (("North", 87), ("East", 94), ("South", 89), ("West", 91))
        ),
        "channel_mix": _render_pie_chart(
            (("Direct", 35), ("Partner", 25), ("Retail", 40))
        ),
    }
    chart_hashes: dict[str, str] = {}
    chart_uris: dict[str, str] = {}
    chart_files = {
        "regional_fulfillment": "regional-fulfillment.png",
        "channel_mix": "channel-mix.png",
    }
    for chart_id, payload in chart_payloads.items():
        (assets_dir / chart_files[chart_id]).write_bytes(payload)
        chart_hashes[chart_id] = hashlib.sha256(payload).hexdigest()
        chart_uris[chart_id] = _data_uri(payload)

    standard_documents = _standard_documents(chart_uris)
    _assert_standard_documents(standard_documents)
    for document in standard_documents:
        (documents_dir / document.filename).write_text(document.content, encoding="utf-8")

    copied_graph_filenames, graph_manifest = _copy_graph_documents(output)
    graph_cases = _graph_cases(copied_graph_filenames)
    standard_cases = _standard_cases(chart_hashes)
    all_cases = [*graph_cases, *standard_cases]
    _write_jsonl(output / "cases.jsonl", all_cases)
    _write_jsonl(
        output / "charts.jsonl",
        (
            {
                "chart_id": chart_id,
                "asset_filename": chart_files[chart_id],
                "sha256": chart_hashes[chart_id],
                "data": dict(
                    (("North", 87), ("East", 94), ("South", 89), ("West", 91))
                    if chart_id == "regional_fulfillment"
                    else (("Direct", 35), ("Partner", 25), ("Retail", 40))
                ),
            }
            for chart_id in ("regional_fulfillment", "channel_mix")
        ),
    )
    manifest = {
        "schema": CORPUS_SCHEMA,
        "dataset_id": "routing-rag-v1",
        "language": "zh-CN",
        "synthetic": True,
        "source_graph_corpus": {
            "dataset_id": GRAPH_SOURCE_ID,
            "document_count": graph_manifest["document_count"],
            "logical_section_count": graph_manifest["logical_section_count"],
        },
        "document_count": len(copied_graph_filenames) + len(standard_documents),
        "graph_document_count": len(copied_graph_filenames),
        "standard_document_count": len(standard_documents),
        "format_counts": {
            "md": sum(document.format == "md" for document in standard_documents)
            + sum(filename.endswith(".md") for filename in copied_graph_filenames.values()),
            "txt": sum(document.format == "txt" for document in standard_documents)
            + sum(filename.endswith(".txt") for filename in copied_graph_filenames.values()),
        },
        "chart_count": len(chart_payloads),
        "case_count": len(all_cases),
        "route_case_counts": {
            route: sum(
                case["expected_route"]["route"] == route for case in all_cases
            )
            for route in ("graph", "simple")
        },
        "documents": [
            *(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "format": Path(filename).suffix.lstrip("."),
                    "source": GRAPH_SOURCE_ID,
                    "material_type": "graph_relation_narrative",
                }
                for document_id, filename in copied_graph_filenames.items()
            ),
            *(
                {
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "format": document.format,
                    "source": "routing-rag-v1",
                    "material_type": document.material_type,
                }
                for document in standard_documents
            ),
        ],
        "route_contract": {
            "graph": {
                "runtime_mode": "graph",
                "top_k": 10,
                "rerank_mode": "classic",
                "expected_evidence_modality": "text",
            },
            "simple": {
                "runtime_mode": "vector",
                "description": "Normal RAG chain. The expected evidence modality is defined per case, not as a route.",
                "expected_evidence_modalities": ["text", "table", "image"],
                "negative_control_expected_outcome": {
                    "contradicted": "answered_with_citation",
                    "closed_world_absence": "answered_with_citation",
                    "open_world_unanswerable": "refused",
                },
            },
        },
        "case_contract": {
            "schema": CASE_SCHEMA,
            "expected_outcomes": sorted(EXPECTED_OUTCOMES),
            "negative_control_kinds": sorted(NEGATIVE_CONTROL_KINDS),
            "benefit_locator": "answer_gold_source_locators",
            "bridge_locator": "path_context_locators",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"built routing-rag-v1: {manifest['document_count']} documents, "
        f"{len(all_cases)} cases at {output}"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate(output: Path) -> int:
    errors: list[str] = []
    manifest_path = output / "manifest.json"
    cases_path = output / "cases.jsonl"
    chart_path = output / "charts.jsonl"
    if not all(path.exists() for path in (manifest_path, cases_path, chart_path)):
        print(f"missing generated corpus files under {output}")
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CORPUS_SCHEMA:
        errors.append("manifest schema mismatch")
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        errors.append("manifest documents must be a list")
        documents = []
    names = [str(document.get("filename", "")) for document in documents if isinstance(document, dict)]
    document_ids = [str(document.get("document_id", "")) for document in documents if isinstance(document, dict)]
    if len(names) != len(set(names)):
        errors.append("document filenames must be unique")
    if len(document_ids) != len(set(document_ids)) or any(not document_id for document_id in document_ids):
        errors.append("document ids must be unique and non-empty")
    if manifest.get("document_count") != len(names):
        errors.append("document count mismatch")
    if manifest.get("graph_document_count") != len(graph_corpus.DOCUMENTS):
        errors.append("graph document count mismatch")
    if manifest.get("standard_document_count") != 8:
        errors.append("standard document count mismatch")
    documents_dir = output / "documents"
    for filename in names:
        path = documents_dir / filename
        if not path.exists():
            errors.append(f"missing document {filename}")
            continue
        if path.suffix not in {".md", ".txt"}:
            errors.append(f"unsupported document suffix for {filename}")
        if not path.read_text(encoding="utf-8").strip():
            errors.append(f"empty document {filename}")

    charts = _read_jsonl(chart_path)
    chart_by_id = {str(chart["chart_id"]): chart for chart in charts}
    if len(chart_by_id) != 2:
        errors.append("expected two unique chart rows")
    for chart_id, chart in chart_by_id.items():
        asset = output / "assets" / str(chart["asset_filename"])
        if not asset.exists():
            errors.append(f"missing chart asset {chart_id}")
            continue
        payload = asset.read_bytes()
        if hashlib.sha256(payload).hexdigest() != chart.get("sha256"):
            errors.append(f"chart hash mismatch for {chart_id}")
        try:
            with Image.open(io.BytesIO(payload)) as image:
                if image.format != "PNG" or image.width < 600 or image.height < 400:
                    errors.append(f"chart image is not a usable PNG for {chart_id}")
        except OSError:
            errors.append(f"chart image is invalid for {chart_id}")

    graph_gold_dir = output / "gold" / GRAPH_SOURCE_ID
    for filename in ("manifest.json", "entities.jsonl", "relations.jsonl", "cases.jsonl"):
        if not (graph_gold_dir / filename).exists():
            errors.append(f"missing graph gold file {filename}")
    graph_relations = {
        str(row["relation_id"]): row
        for row in _read_jsonl(graph_gold_dir / "relations.jsonl")
    } if (graph_gold_dir / "relations.jsonl").exists() else {}

    cases = _read_jsonl(cases_path)
    case_ids = [str(case.get("case_id", "")) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        errors.append("case ids must be unique")
    if manifest.get("case_count") != len(cases):
        errors.append("case count mismatch")
    allowed_routes = {"graph", "simple"}
    observed_negative_kinds: set[str] = set()
    expected_route_counts = {route: 0 for route in allowed_routes}
    for case in cases:
        route = case.get("expected_route")
        if not isinstance(route, dict) or route.get("route") not in allowed_routes:
            errors.append(f"{case.get('case_id')}: unsupported expected route")
            continue
        route_name = str(route["route"])
        expected_route_counts[route_name] += 1
        case_id = str(case.get("case_id", ""))
        expected_outcome = case.get("expected_outcome")
        if expected_outcome not in EXPECTED_OUTCOMES:
            errors.append(f"{case_id}: invalid expected_outcome")
        answerable = case.get("answerable")
        if not isinstance(answerable, bool) or answerable != (expected_outcome == "answered"):
            errors.append(f"{case_id}: answerable and expected_outcome disagree")
        negative_kind = case.get("negative_control_kind")
        if case.get("category") == "negative_control":
            if negative_kind not in NEGATIVE_CONTROL_KINDS:
                errors.append(f"{case_id}: negative control kind is missing or invalid")
            else:
                observed_negative_kinds.add(str(negative_kind))
        elif negative_kind is not None:
            errors.append(f"{case_id}: non-negative case has a negative control kind")
        aspects = case.get("expected_answer_aspects")
        if not isinstance(aspects, list) or (expected_outcome == "answered" and not aspects):
            errors.append(f"{case_id}: answered case needs non-empty answer aspects")
        answer_locators = case.get("answer_gold_source_locators")
        if not isinstance(answer_locators, list):
            errors.append(f"{case_id}: answer gold locators must be a list")
        elif expected_outcome == "answered" and not answer_locators:
            errors.append(f"{case_id}: answered case needs answer gold locators")
        path_locators = case.get("path_context_locators")
        if not isinstance(path_locators, list):
            errors.append(f"{case_id}: path context locators must be a list")
        if not isinstance(case.get("forbidden_claims"), list):
            errors.append(f"{case_id}: forbidden_claims must be a list")
        source = case.get("source")
        if not isinstance(source, dict):
            errors.append(f"{case.get('case_id')}: source must be an object")
            continue
        filename = source.get("document_filename")
        if isinstance(filename, str) and not (documents_dir / filename).exists():
            errors.append(f"{case.get('case_id')}: source document is missing")
        document_id = source.get("document_id")
        if isinstance(document_id, str) and document_id not in document_ids:
            errors.append(f"{case.get('case_id')}: source document id is unknown")
        if route_name == "graph":
            runtime = route.get("runtime_request")
            path = source.get("gold_path")
            if not isinstance(runtime, dict) or runtime.get("mode") != "graph":
                errors.append(f"{case.get('case_id')}: graph route must request graph mode")
            if not isinstance(path, list) or not path:
                errors.append(f"{case.get('case_id')}: graph route needs a gold path")
            elif any(str(relation_id) not in graph_relations for relation_id in path):
                errors.append(f"{case_id}: graph path references unknown relation")
            relation_ids = set(str(item) for item in source.get("answer_relation_ids", ()))
            locator_ids = {
                str(item.get("relation_id"))
                for item in answer_locators or ()
                if isinstance(item, dict) and item.get("kind") == "graph_relation"
            }
            if locator_ids != relation_ids:
                errors.append(f"{case_id}: answer locators must match answer_relation_ids")
            context_ids = {
                str(item.get("relation_id"))
                for item in path_locators or ()
                if isinstance(item, dict) and item.get("kind") == "graph_relation"
            }
            if context_ids & locator_ids:
                errors.append(f"{case_id}: path context overlaps answer gold")
        elif route_name == "simple":
            runtime = route.get("runtime_request")
            if not isinstance(runtime, dict) or runtime.get("mode") != "vector":
                    errors.append(f"{case_id}: simple route must request vector mode")
            modality = route.get("expected_evidence_modality")
            if modality == "table":
                text = (documents_dir / str(filename)).read_text(encoding="utf-8") if isinstance(filename, str) and (documents_dir / str(filename)).exists() else ""
                if "| ---" not in text:
                        errors.append(f"{case_id}: table case source lacks Markdown table")
            if modality == "image":
                locator = source.get("locator")
                chart_id = locator.get("chart_id") if isinstance(locator, dict) else None
                if not isinstance(chart_id, str) or chart_id not in chart_by_id:
                    if case.get("answerable"):
                        errors.append(f"{case_id}: chart case references unknown chart")
                elif isinstance(filename, str):
                    text = (documents_dir / filename).read_text(encoding="utf-8")
                    payload = (output / "assets" / str(chart_by_id[chart_id]["asset_filename"])).read_bytes()
                    if _data_uri(payload) not in text:
                        errors.append(f"{case_id}: chart Markdown does not embed its chart")
        if expected_outcome == "answered":
            answer = str(case.get("expected_answer", ""))
            if not answer:
                errors.append(f"{case_id}: answered case has no expected answer")
        if negative_kind == "contradicted" and expected_outcome != "answered":
            errors.append(f"{case_id}: contradicted control must be answered with citation")
        if negative_kind == "open_world_unanswerable" and expected_outcome != "refused":
            errors.append(f"{case_id}: open-world control must be refused")
    if manifest.get("route_case_counts") != expected_route_counts:
        errors.append("route case counts mismatch")
    if observed_negative_kinds != set(NEGATIVE_CONTROL_KINDS):
        errors.append("negative controls must cover contradicted, closed_world_absence, and open_world_unanswerable")
    if errors:
        print("validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        f"validated routing-rag-v1: {manifest['document_count']} documents, "
        f"{len(cases)} cases, routes={manifest['route_case_counts']}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="replace generated files under output")
    arguments = parser.parse_args()
    if arguments.command == "build":
        build(arguments.output, force=arguments.force)
        return validate(arguments.output)
    return validate(arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
