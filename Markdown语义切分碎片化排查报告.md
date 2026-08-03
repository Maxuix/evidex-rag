# Markdown 语义切分碎片化问题排查报告

> 排查时间：2026-07-31
> 影响范围：Semantic（语义切分）策略 + Markdown（.md）输入
> 现象：Markdown 走 Semantic 切分时，大段落被拆成单句碎片，一个 chunk 内常常只有一句话；PDF 走同一条链路则正常。

---

## 一、问题现象

文件切分（Chunking）过程中，Markdown 格式文件走 Semantic 这条链路时，结果非常零碎：

- 一个大段落被拆解成零碎的几句话，一个 chunk 内可能只存在一句话；
- 完整语义被截断，chunk 不具备用作检索证据的价值；
- 没有实现 Semantic 所应实现的"语义切分"要求。

而同样走 Semantic 链路的 PDF 文件，切分结果正常。

## 二、排查方法

1. **代码定位**：从切分策略分发入口 `src/rag_kb/indexing/pipeline.py` 顺藤摸瓜，定位到 Semantic 切分的三个核心模块：
   - `src/rag_kb/document_processing/docling/semantic.py`（语义单元提取 + 边界判定）
   - `src/rag_kb/document_processing/semantic_boundaries.py`（区域切分 + 动态规划选边界）
   - `src/rag_kb/document_processing/docling/provenance.py`（surface 类型判定）

2. **链路对比**：确认 Markdown 与 PDF 走**完全相同**的 Semantic 代码路径，差异不在策略分发，而在 Docling 解析后的 item 结构不同。

3. **实证验证**：编写诊断脚本 `tools/diagnose_md_chunking.py`，用项目自身的 Docling + semantic 模块解析一个典型 Markdown 样本，逐 item 打印边界判定、逐 unit 打印 `hard_boundary_before`，并模拟区域切分，直接复现碎片化（见第五节）。

## 三、根因分析

### 3.0 结论先行

碎片化是**两个机制叠加**的结果：

1. **Markdown 标题密集 → 每个标题触发一个 SECTION 硬边界**（`has_pending_titles`）；
2. **每个硬边界强制切割独立区域，而小区域（≤ `max_chunk_tokens=800`）直接整体成 chunk，完全绕过 `min_chunk_tokens=220` 约束**。

PDF 不受影响，是因为它的硬边界以 PAGE（物理页）为主，频率低、每页内容多，区域天然较大。

### 3.1 切分策略不按文件类型分发

`src/rag_kb/document_processing/profiles.py` 第 214-232 行的 `resolve()` 仅依据知识库配置的 chunking profile 选择 STRUCTURAL 或 SEMANTIC，**与文件类型无关**。`src/rag_kb/indexing/pipeline.py` 第 358-381 行 `_chunks()` 对所有文件类型走同一段 Semantic 代码。因此 Markdown 与 PDF 的差异只能来自上游 Docling 解析后的 item 结构。

### 3.2 根因一：Markdown 标题密集触发大量 SECTION 硬边界（核心）

**边界判定函数** `semantic.py` 第 257-281 行 `_boundary()`：

```python
def _boundary(...) -> str | None:
    if (previous_surface is not None and surface is not None
            and surface != previous_surface):
        return _BOUNDARY_SURFACE          # page
    if item_kind is ItemKind.TABLE or previous_kind is ItemKind.TABLE:
        return _BOUNDARY_TABLE
    if item_kind in _BLOCK_KINDS or previous_kind in _BLOCK_KINDS:
        return _BOUNDARY_BLOCK
    if has_pending_titles:                # ← Markdown 频繁触发
        return _BOUNDARY_SECTION
    if (previous_parent is not None and parent is not None
            and parent != previous_parent):
        return _BOUNDARY_SECTION
    return None
```

在 `docling_semantic_units()`（第 86-152 行）中，每个 `#`/`##` 标题被累积到 `pending_titles`，**直到遇到下一个非标题内容 item 时**，`has_pending_titles=True` 触发一次 SECTION 边界，随后 `pending_titles.clear()`。

对 Markdown 而言：
- Docling 的 `MarkdownBackend` 用 `#` 语法精确识别标题，技术文档/README 中标题往往很密集；
- **每个标题后的第一个内容 item 都会获得一个 SECTION 硬边界**；
- 这些 SECTION 边界被写入 `SemanticUnit.hard_boundary_before`（值为 `"section"`）。

对 PDF 而言：
- 标题识别依赖字体大小启发式，频率远低于 Markdown；
- 主要硬边界是 PAGE（物理页），每页内容多，区域天然较大。

> 注：`surface_kind`（`provenance.py` 第 59-66 行）对 Markdown 返回 `"logical"`，因为 `"text/markdown"` 不在 `_SURFACE_BY_MIMETYPE` 映射中。因此 Markdown **没有 PAGE 边界**，硬边界几乎全部来自 SECTION。这是符合预期的（Markdown 无页概念），问题不在 surface 判定本身。

### 3.3 根因二：小区域完全绕过 min_chunk_tokens（核心）

**区域切分** `semantic_boundaries.py` 第 134-148 行 `_select_all_regions()`：

```python
def _select_all_regions(units, scores):
    boundaries: list[ChunkBoundary] = []
    start = 0
    for index in range(1, len(units) + 1):
        if index < len(units) and units[index].hard_boundary_before is None:
            continue                       # 同一区域，继续
        boundaries.extend(_select_region(units, scores, start, index))
        if index < len(units):
            reason = ChunkBoundaryReason(units[index].hard_boundary_before)
            boundaries.append(ChunkBoundary(index - 1, reason))  # 硬边界处强制切割
        start = index
    return tuple(boundaries)
```

**区域内边界选择** `semantic_boundaries.py` 第 151-164 行 `_select_region()`：

```python
def _select_region(units, scores, start, end):
    ...
    region_tokens = count_chunk_tokens(_joined(units, start, end))
    if region_tokens <= maximum:           # maximum = max_chunk_tokens = 800
        return []                          # ← 小区域直接整体成 chunk，不加内部边界
    ...
```

**关键缺陷**：当一个区域（两个硬边界之间的 units）总 tokens ≤ 800 时，`_select_region()` 直接返回空列表，整个区域成为一个 chunk——**即使该区域只有一句话、远低于 `min_chunk_tokens=220`**。

`min_chunk_tokens` 仅在 `region_tokens > maximum`（即 > 800）时，作为内部边界选择的约束（第 191 行 `if tokens < minimum and not region_is_small: continue`）。对于小区域，这个约束根本不参与计算。

**测试也确认了这一行为**：`tests/unit/test_semantic_chunking.py` 第 245-273 行 `test_hard_boundaries_allow_small_regions_and_block_smoothing` 显式断言"有硬边界时 chunk 可以小于 min_chunk_tokens(220)"。这说明当前行为是"被设计成这样"，但对 Markdown 而言产生了反效果。

### 3.4 根因三：_SENTENCE_BREAK 的换行切分（次要/加剧）

`semantic.py` 第 48 行：

```python
_SENTENCE_BREAK = re.compile(r"(?<=[。！？；.!?;])(?:[ \t]+|\n*)|\n+")
```

`\n+` 分支会在任何换行处切分。当一个 Docling item 的文本含换行（Markdown 多行构造常见）时，每行被拆成独立 fragment。不过 `_merge_short_fragments()`（第 353-377 行）能将**同一 item 内**的短 fragment 合并回来（上限 `analysis_unit_target_tokens=80`），且同 item 的 fragment 之间没有硬边界互相隔离。因此这条仅在 item 文本含换行时**加剧**碎片化，不是主因。

> 实证（第五节）显示：样本中"架构设计"section 的多句话被成功合并为 unit#2 + unit#3（同区域，无硬边界），最终合并成一个 chunk。证明**同一 section 内部不会碎片化，碎片化发生在不同 section 之间**。

## 四、Markdown 与 PDF 的差异

| 对比维度 | Markdown（碎片化） | PDF（正常） |
|---------|-------------------|------------|
| surface_kind | `logical`（mimetype 不在映射表） | `page`（application/pdf 在表中） |
| 标题识别 | `#` 语法精确，文档中常很密集 | 字体大小启发式，识别稀疏 |
| 主要硬边界来源 | SECTION（每个标题后触发） | PAGE（物理页边界） |
| 硬边界频率 | 高：每个标题切一次 | 低：每页切一次，每页内容多 |
| 单个区域内容量 | 单个 section，常常很短 | 多 section 聚合，区域较大 |
| 切分结果 | 碎片化：单句 chunk，远低于 220 | 正常：接近 target 550 tokens |

**核心结论**：两者走完全相同的 `semantic.py` / `semantic_boundaries.py` 代码，差异不在策略分发，而在 Docling 解析后 item 结构不同——Markdown 标题密集导致 SECTION 硬边界过密，叠加小区域绕过 `min_chunk_tokens` 的缺陷，产生碎片。

## 五、实证：验证脚本输出

诊断脚本 `tools/diagnose_md_chunking.py` 用一个总 tokens 仅 147 的 Markdown 样本（4 个标题 + 内容），实际运行项目代码得到：

```
surface_kind = 'logical'    mimetype = 'text/markdown'

[1] Docling 解析后的 item 序列与 boundary 判定
  [title] (title)   -> 累积到 pending_titles: '项目概述'
  [text ] parent='#/body'  boundary='section' text='这是一个企业知识库系统…'
  [title] (title)   -> 累积到 pending_titles: '核心功能'
  [text ] parent='#/body'  boundary='section' text='支持多种文档格式的解析。'
  [title] (title)   -> 累积到 pending_titles: '架构设计'
  [text ] parent='#/body'  boundary='section' text='系统采用微服务架构…'
  [title] (title)   -> 累积到 pending_titles: '部署方式'
  [text ] parent='#/body'  boundary='section' text='支持 Docker 容器化部署。'

[2] SemanticUnit 序列
  unit#0 tokens= 25 hard_boundary_before='section' text='项目概述 这是一个企业知识库系统…'
  unit#1 tokens= 16 hard_boundary_before='section' text='核心功能 支持多种文档格式的解析。'
  unit#2 tokens= 66 hard_boundary_before='section' text='架构设计 系统采用微服务架构。…'
  unit#3 tokens= 24 hard_boundary_before=None      text='这些组件协同工作…'
  unit#4 tokens= 16 hard_boundary_before='section' text='部署方式 支持 Docker 容器化部署。'
  总 units=5  总 tokens=147

[3] 区域切分
  region#0 units[0:1) tokens= 25  <<< 小于 min_chunk_tokens(220)，仍独立成 chunk!
  region#1 units[1:2) tokens= 16  <<< 小于 min_chunk_tokens(220)，仍独立成 chunk!
  region#2 units[2:4) tokens= 90  <<< 小于 min_chunk_tokens(220)，仍独立成 chunk!
  region#3 units[4:5) tokens= 16  <<< 小于 min_chunk_tokens(220)，仍独立成 chunk!

[4] 最终 chunk 输出
  chunk#0 tokens= 25 text='项目概述 这是一个企业知识库系统…'  <<< 碎片化！
  chunk#1 tokens= 16 text='核心功能 支持多种文档格式的解析。'  <<< 碎片化！
  chunk#2 tokens= 90 text='架构设计 系统采用微服务架构…'      <<< 碎片化！
  chunk#3 tokens= 16 text='部署方式 支持 Docker 容器化部署。'  <<< 碎片化！
```

**关键证据**：
- 所有 parent 都是 `#/body`（未变化），SECTION 边界**完全来自 `has_pending_titles`**，不是 parent 变化；
- 总 tokens 仅 147（远低于 `max_chunk_tokens=800`），本应合并成 1 个 chunk，却被 3 个 SECTION 硬边界切成 4 个碎片；
- 4 个 chunk 全部低于 `min_chunk_tokens=220`，最小仅 16 tokens；
- "架构设计"section 内的多句话（unit#2 + unit#3）因无硬边界而成功合并成一个 90-token chunk，证明**碎片化发生在 section 之间，而非 section 内部**。

## 六、解决方案

### 6.1 方案 B（推荐，最小改动）：小区域合并

**思路**：保留 SECTION 作为硬边界，但在 `_select_all_regions()` 切分出区域后，将 `< min_chunk_tokens` 的相邻小区域合并到相邻区域，只要合并后不超过 `max_chunk_tokens`。**仅跨越 SECTION 边界合并，不跨越 PAGE/TABLE/BLOCK**（后者会破坏 provenance 一致性，见 `provenance.py` 第 141-147 行的 surface kind 一致性校验）。

**改动范围**：仅 `src/rag_kb/document_processing/semantic_boundaries.py`，不修改 `SemanticUnit` 数据结构、不动 domain 层、不动持久化、不需要数据库迁移。

**核心实现**（替换现有 `_select_all_regions`，新增 `_merge_small_regions`）：

```python
def _select_all_regions(
    units: tuple[SemanticUnit, ...],
    scores: tuple[int | None, ...],
) -> tuple[ChunkBoundary, ...]:
    # 1. 按 hard boundary 切分原始区域
    raw_regions: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(units) + 1):
        if index < len(units) and units[index].hard_boundary_before is None:
            continue
        raw_regions.append((start, index))
        start = index

    # 2. 合并小于 min_chunk_tokens 的相邻区域（仅跨越 SECTION）
    regions = _merge_small_regions(units, raw_regions)

    # 3. 对每个最终区域选择内部边界，并在区域间保留 hard boundary
    boundaries: list[ChunkBoundary] = []
    for r_start, r_end in regions:
        boundaries.extend(_select_region(units, scores, r_start, r_end))
        if r_end < len(units):
            reason = ChunkBoundaryReason(units[r_end].hard_boundary_before)
            boundaries.append(ChunkBoundary(r_end - 1, reason))
    return tuple(boundaries)


def _merge_small_regions(
    units: tuple[SemanticUnit, ...],
    regions: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge regions below min_chunk_tokens into a neighbour.

    Only SECTION boundaries may be crossed: PAGE/TABLE/BLOCK stay, because a
    chunk spanning two surface kinds cannot be cited (provenance invariant).
    """
    minimum = _config_int("min_chunk_tokens")
    maximum = _config_int("max_chunk_tokens")
    if not regions:
        return regions

    result = list(regions)
    i = 0
    while i < len(result):
        r_start, r_end = result[i]
        region_tokens = count_chunk_tokens(_joined(units, r_start, r_end))
        if region_tokens >= minimum or i == len(result) - 1:
            i += 1
            continue
        # 优先向前合并（保持阅读顺序）
        boundary_reason = units[r_start].hard_boundary_before
        crossable = (
            boundary_reason is None
            or boundary_reason == ChunkBoundaryReason.SECTION.value
        )
        if i > 0 and crossable:
            p_start, _ = result[i - 1]
            if count_chunk_tokens(_joined(units, p_start, r_end)) <= maximum:
                result[i - 1] = (p_start, r_end)
                result.pop(i)
                continue
        # 否则向后合并
        if i < len(result) - 1:
            next_start = result[i + 1][0]
            next_boundary = units[next_start].hard_boundary_before
            next_crossable = (
                next_boundary is None
                or next_boundary == ChunkBoundaryReason.SECTION.value
            )
            if next_crossable:
                if count_chunk_tokens(_joined(units, r_start, result[i + 1][1])) <= maximum:
                    result[i] = (r_start, result[i + 1][1])
                    result.pop(i + 1)
                    continue
        i += 1
    return result
```

**对样本的预期效果**：4 个区域 (25/16/90/16 tokens) 逐步合并为 1 个区域 (147 tokens) → 输出 1 个 147-token chunk，语义完整。

**优点**：
- 改动集中在一个文件，风险低，易回滚；
- 保留 PAGE/TABLE/BLOCK 硬边界，provenance 安全；
- 合并后若区域仍 > `max_chunk_tokens`，`_select_region()` 会用动态规划按语义距离选最优切分点，恰好可能在原 SECTION 附近切分（基于语义而非结构）；
- 不影响 PDF 现有行为（PDF 区域本就大，不触发合并）。

**权衡**：合并会跨越 section 边界，chunk 的 `common_hierarchy`（公共标题前缀）可能变短。但相比单句碎片，跨 section 合并更能保留完整语义，可接受。

### 6.2 方案 A（长期更优）：将 SECTION 降级为软边界

**思路**：让标题边界参与语义距离加权（在标题处语义距离给高分），但**不强制切割区域**。只有 PAGE/TABLE/BLOCK 保持硬边界。这样动态规划在整个文档范围内根据语义距离 + token 约束选择切分点，标题仅作为"优先切分"的提示，更符合 Semantic chunking 的本质。

**改动范围**（较大）：
- `domain/parsing.py` 的 `SemanticUnit` 增加 `soft_boundary_before` 字段，或重构 `hard_boundary_before` 语义；
- `semantic.py` 的 `_boundary()` 区分硬/软边界，SECTION 标记为软；
- `semantic_boundaries.py` 的 `_select_all_regions` 只在硬边界切割，`smoothed_distances`/`_select_region` 在软边界位置给 score 加成；
- 持久化层、`docling_unit_sequence_hash`、`validate_plan`、相关测试同步修改；
- 由于 `IndexChunkPlan` 持久化且含 `unit_sequence_hash`，可能涉及存量数据兼容。

**优点**：从根本上让 Semantic 切分基于语义而非结构，标题密集不再导致硬切。

**缺点**：改动面广，涉及 domain/持久化/迁移，风险较高，建议作为后续重构。

### 6.3 不推荐的方案

- **调整 Docling Markdown 解析选项减少标题**：治标不治本，且依赖 Docling 配置能力，标题是文档真实结构不应抹除。
- **放宽 `_SENTENCE_BREAK` 的 `\n+` 分支**：仅缓解 item 内换行切分，不解决 section 间碎片化（主因）。
- **全局调大 `min_chunk_tokens`**：不解决"小区域绕过最小约束"的本质问题，且影响 PDF。

## 七、测试与验证建议

1. **新增回归测试**：在 `tests/unit/test_semantic_chunking.py` 增加用例，构造多标题短 section 的 unit 序列，断言修复后 chunk 数量减少、每个 chunk ≥ `min_chunk_tokens`（或文档总量本身就小）。

2. **保留现有测试意图**：现有 `test_hard_boundaries_allow_small_regions_and_block_smoothing` 断言"PAGE 硬边界允许小区域"。方案 B 只合并 SECTION，PAGE 边界仍允许小区域，需确认该测试仍通过（它用的是 page 边界，不会被合并）。

3. **运行诊断脚本**：用 `tools/diagnose_md_chunking.py` 对比修复前后输出，确认 4 碎片 → 1 完整 chunk。

4. **PDF 回归**：用 PDF 样本走 Semantic 切分，确认结果不变（PDF 区域大，不触发合并）。

5. **基础测试**：`PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/basic -v` 与 `PYTHONPATH=src:. .venv/bin/python -m unittest tests.unit.test_semantic_chunking -v`。

## 附录：关键代码位置索引

| 关注点 | 文件（相对路径） | 行号 | 函数/类名 |
|--------|-----------------|------|-----------|
| 语义单元提取 | `src/rag_kb/document_processing/docling/semantic.py` | 69-181 | `docling_semantic_units()` |
| 句子分割正则 | `src/rag_kb/document_processing/docling/semantic.py` | 48 | `_SENTENCE_BREAK` |
| 文本切片 | `src/rag_kb/document_processing/docling/semantic.py` | 304-319 | `_text_pieces()` |
| Fragment 合并 | `src/rag_kb/document_processing/docling/semantic.py` | 353-377 | `_merge_short_fragments()` |
| **边界判定（根因一）** | `src/rag_kb/document_processing/docling/semantic.py` | 257-281 | `_boundary()` |
| **Chunk 计划构建** | `src/rag_kb/document_processing/semantic_boundaries.py` | 22-63 | `build_chunk_plan()` |
| **区域分割（根因二）** | `src/rag_kb/document_processing/semantic_boundaries.py` | 134-148 | `_select_all_regions()` |
| 区域内边界选择 | `src/rag_kb/document_processing/semantic_boundaries.py` | 151-227 | `_select_region()` |
| 策略分发 | `src/rag_kb/indexing/pipeline.py` | 358-381 | `_chunks()` |
| Surface kind 判定 | `src/rag_kb/document_processing/docling/provenance.py` | 59-66 | `surface_kind()` |
| 语义切分配置 | `src/rag_kb/document_processing/profiles.py` | 151-177 | `SEMANTIC_CHUNKING_CONFIG` |
| 边界原因枚举 | `src/rag_kb/domain/chunking.py` | 25-32 | `ChunkBoundaryReason` |
| 语义单元类型 | `src/rag_kb/domain/parsing.py` | 112-121 | `SemanticUnit` |
| 碎片化行为测试 | `tests/unit/test_semantic_chunking.py` | 245-273 | `test_hard_boundaries_allow_small_regions_and_block_smoothing` |
| 诊断脚本 | `tools/diagnose_md_chunking.py` | — | — |

---

**结论**：根因是"Markdown 标题密集 → SECTION 硬边界过密"叠加"小区域绕过 `min_chunk_tokens`"两个机制。推荐采用方案 B（小区域合并），改动集中在 `semantic_boundaries.py` 一个文件，风险低、见效快，且不影响 PDF 现有行为。
