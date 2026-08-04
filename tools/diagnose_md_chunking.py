"""只读诊断 Markdown 的真实 Docling projection 与切分结果。

用项目自身的 Docling consumer、semantic unit、planner/assembler 诊断内置样本，
或通过 ``--source`` 读取指定 Markdown。工具不复制 boundary 算法，也不改写源文件。
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import sys
import tempfile
from uuid import UUID

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from docling.document_converter import DocumentConverter

from rag_kb.document_processing.docling.provenance import surface_kind
from rag_kb.document_processing.docling.semantic import (
    assemble_semantic_chunks,
    docling_semantic_units,
    docling_unit_sequence_hash,
)
from rag_kb.document_processing.docling.traversal import (
    iterate_chunking_items,
)
from rag_kb.document_processing.docling.structural import assemble_structural
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        help="可选 Markdown 路径；仅只读转换，不改写源文件",
    )
    args = parser.parse_args()
    source_path, source_bytes, temporary = _source(args.source)
    try:
        doc = DocumentConverter().convert(source=str(source_path)).document
    finally:
        if temporary:
            os.unlink(source_path)

    sk = surface_kind(doc)
    mimetype = getattr(doc.origin, "mimetype", None)
    print("=" * 72)
    print(f"surface_kind = {sk!r}    mimetype = {mimetype!r}")
    print("(Markdown 不在 _SURFACE_BY_MIMETYPE 中 -> logical -> 无 page 边界)")
    print("=" * 72)

    print("\n[1] 真实 consumer-side chunking projection")
    print("-" * 72)
    projection = tuple(iterate_chunking_items(doc))
    for item in projection:
        preview = item.text.replace("\n", "|")[:42]
        print(
            f"  [{item.kind.value:14}] inline={item.inline!s:5} "
            f"container={item.semantic_container!r:24} refs={len(item.refs):3} "
            f"text={preview!r}"
        )

    structural = assemble_structural(doc)
    print("\n[2] Structural v4 输出")
    print("-" * 72)
    for index, chunk in enumerate(structural):
        preview = chunk.text.replace("\n", "|")[:58]
        print(
            f"  chunk#{index} tokens={chunk.token_count:4} "
            f"refs={len(chunk.item_refs):3} text={preview!r}"
        )

    print("\n[3] 真实 SemanticUnit 序列（_merge_short_fragments 之后）")
    print("-" * 72)
    units = docling_semantic_units(doc)
    for u in units:
        preview = u.text.replace("\n", "|")[:48]
        print(f"  unit#{u.ordinal:2} tokens={u.token_count:4} "
              f"hard_boundary_before={u.hard_boundary_before!r:9} text={preview!r}")

    total = count_chunk_tokens("\n\n".join(u.text for u in units))
    print(f"\n  总 units={len(units)}  总 tokens={total}")

    print("\n[4] 真实 semantic planner：生成不可变 plan（合成 vectors，不调用 provider）")
    print("-" * 72)
    profile = profile_for_preset(ChunkingPreset.SEMANTIC_BALANCED_V1)
    vectors = _synthetic_normalized_vectors(units)
    plan = build_chunk_plan(
        indexed_document_version_id=UUID("00000000-0000-0000-0000-000000000004"),
        source_checksum_sha256=hashlib.sha256(source_bytes).hexdigest(),
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
    print("\n[5] 最终 chunk 输出（真实 assemble_semantic_chunks）")
    print("-" * 72)
    for i, chunk in enumerate(chunks):
        rt = chunk.token_count
        text = chunk.text
        preview = text.replace("\n", "|")[:58]
        warn = "  <<< 合法 residual（见相邻 boundary/800 上限）" if rt < 220 else ""
        print(f"  chunk#{i} tokens={rt:4} text={preview!r}{warn}")


def _source(source: Path | None) -> tuple[Path, bytes, bool]:
    if source is not None:
        resolved = source.expanduser().resolve(strict=True)
        if not resolved.is_file():
            raise ValueError("--source must name a regular file")
        return resolved, resolved.read_bytes(), False
    handle = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
    try:
        handle.write(SAMPLE)
    finally:
        handle.close()
    return Path(handle.name), SAMPLE.encode("utf-8"), True


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
