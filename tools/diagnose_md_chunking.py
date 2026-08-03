"""验证 Markdown 走 Semantic 切分时的碎片化根因。

用项目自身的 Docling + semantic 模块解析一个典型 Markdown 样本，
逐 item 打印 boundary 判定，逐 unit 打印 hard_boundary_before，并调用真实
semantic planner/assembler 展示 SECTION-only 后处理后的最终 Chunk。
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
import tempfile
from uuid import UUID

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from docling.document_converter import DocumentConverter

from rag_kb.document_processing.docling.provenance import (
    item_surfaces,
    surface_kind,
)
from rag_kb.document_processing.docling.semantic import (
    assemble_semantic_chunks,
    docling_semantic_units,
    docling_unit_sequence_hash,
)
from rag_kb.document_processing.docling.traversal import (
    ItemKind,
    classify_item,
    item_text,
    iterate_body_items,
    parent_ref,
)
from rag_kb.document_processing.profiles import (
    profile_fingerprint,
    profile_for_preset,
)
from rag_kb.document_processing.semantic_boundaries import build_chunk_plan
from rag_kb.document_processing.tokenization import count_chunk_tokens
from rag_kb.domain import ChunkingPreset

SAMPLE = """# 项目概述
这是一个企业知识库系统，用于管理内部文档。

# 核心功能
支持多种文档格式的解析。

# 架构设计
系统采用微服务架构。前端使用 React 开发。后端使用 Python 编写。数据库选用 PostgreSQL。缓存使用 Redis。消息队列使用 RabbitMQ。检索引擎使用 Elasticsearch。这些组件协同工作，提供高性能的知识检索能力。

# 部署方式
支持 Docker 容器化部署。
"""


def main() -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as handle:
        handle.write(SAMPLE)
        tmp_path = handle.name

    try:
        converter = DocumentConverter()
        doc = converter.convert(source=tmp_path).document
    finally:
        os.unlink(tmp_path)

    sk = surface_kind(doc)
    mimetype = getattr(doc.origin, "mimetype", None)
    print("=" * 72)
    print(f"surface_kind = {sk!r}    mimetype = {mimetype!r}")
    print("(Markdown 不在 _SURFACE_BY_MIMETYPE 中 -> logical -> 无 page 边界)")
    print("=" * 72)

    print("\n[1] Docling 解析后的 item 序列与 boundary 判定")
    print("-" * 72)
    prev_surface = None
    prev_parent = None
    prev_kind = None
    pending = 0
    kind = sk
    for item, _level in iterate_body_items(doc):
        ik = classify_item(item)
        if ik in {ItemKind.PICTURE, ItemKind.CAPTION}:
            continue
        if ik in {ItemKind.TITLE, ItemKind.SECTION_HEADER}:
            pending += 1
            print(f"  [{ik:14}] (title)   -> 累积到 pending_titles: {item_text(item, doc)[:36]!r}")
            continue
        text = item_text(item, doc)
        if not text:
            continue
        surfaces = item_surfaces(item, kind=kind)
        surface = min(s.ordinal for s in surfaces) if surfaces else None
        parent = parent_ref(item)
        boundary = None
        if prev_surface is not None and surface is not None and surface != prev_surface:
            boundary = "page"
        elif ik is ItemKind.TABLE or prev_kind is ItemKind.TABLE:
            boundary = "table"
        elif ik in {ItemKind.CODE, ItemKind.FORMULA} or prev_kind in {ItemKind.CODE, ItemKind.FORMULA}:
            boundary = "block"
        elif pending:
            boundary = "section"
        elif prev_parent is not None and parent is not None and parent != prev_parent:
            boundary = "section"
        preview = text.replace("\n", "|")[:42]
        print(f"  [{ik:14}] parent={parent!r:28} boundary={boundary!r:9} text={preview!r}")
        pending = 0
        if surface is not None:
            prev_surface = surface
        prev_parent = parent
        prev_kind = ik

    print("\n[2] 生成的 SemanticUnit 序列（_merge_short_fragments 之后）")
    print("-" * 72)
    units = docling_semantic_units(doc)
    for u in units:
        preview = u.text.replace("\n", "|")[:48]
        print(f"  unit#{u.ordinal:2} tokens={u.token_count:4} "
              f"hard_boundary_before={u.hard_boundary_before!r:9} text={preview!r}")

    total = count_chunk_tokens("\n\n".join(u.text for u in units))
    print(f"\n  总 units={len(units)}  总 tokens={total}")

    print("\n[3] 真实 semantic planner：生成不可变 plan（合成 vectors，不调用 provider）")
    print("-" * 72)
    profile = profile_for_preset(ChunkingPreset.SEMANTIC_BALANCED_V1)
    vectors = _synthetic_normalized_vectors(units)
    plan = build_chunk_plan(
        indexed_document_version_id=UUID("00000000-0000-0000-0000-000000000004"),
        source_checksum_sha256=hashlib.sha256(SAMPLE.encode("utf-8")).hexdigest(),
        profile_fingerprint=profile_fingerprint(
            profile.parser_config, profile.chunking_config
        ),
        units=units,
        vectors=vectors,
        sequence_hash=docling_unit_sequence_hash(units),
    )
    for boundary in plan.boundaries:
        print(
            f"  boundary after unit#{boundary.after_unit_ordinal}: "
            f"reason={boundary.reason.value!r} score={boundary.score_micros!r}"
        )
    print(f"  plan profile={profile.chunking_config['profile']!r} hash={plan.plan_hash}")

    chunks = assemble_semantic_chunks(doc, units, plan)
    print("\n[4] 最终 chunk 输出（真实 assemble_semantic_chunks）")
    print("-" * 72)
    for i, chunk in enumerate(chunks):
        rt = chunk.token_count
        text = chunk.text
        preview = text.replace("\n", "|")[:58]
        warn = "  <<< 合法 residual（见相邻 boundary/800 上限）" if rt < 220 else ""
        print(f"  chunk#{i} tokens={rt:4} text={preview!r}{warn}")


def _synthetic_normalized_vectors(units) -> tuple[tuple[float, ...], ...]:
    """Derive stable unit vectors without provider/network access."""

    vectors: list[tuple[float, ...]] = []
    for unit in units:
        digest = hashlib.sha256(
            f"{unit.ordinal}:{unit.text}".encode("utf-8")
        ).digest()
        raw = tuple((byte / 127.5) - 1.0 for byte in digest[:8])
        norm = math.sqrt(sum(value * value for value in raw))
        vectors.append(tuple(value / norm for value in raw))
    return tuple(vectors)


if __name__ == "__main__":
    main()
