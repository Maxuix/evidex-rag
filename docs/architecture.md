# Local Knowledge Base & RAG — 当前架构与轻量化边界

| 字段 | 内容 |
| --- | --- |
| 文档状态 | 全项目唯一当前架构文档（描述事实，不是目标蓝图） |
| 最后核对 | 2026-08-19 |
| 核对基线 | `73b0302`、当前代码、配置、Alembic 迁移、Compose 与公开路由 |
| 适用对象 | 个人维护者、CODE AGENTS |
| 部署边界 | 单机、单用户、本地使用；不是共享或生产服务 |
| 设计优先级 | 功能可用与个人可维护性优先于平台化、通用化和生产完备性 |

## 1. 文档职责与事实优先级

本文集中维护系统现在能做什么、主要代码在哪里、运行时如何协作、核心数据与安全契约、
本地运行方式，以及哪些边界不能误解。它是 `docs/` 下唯一的架构事实入口；不再用多个专题
架构文件拆分同一事实。设置常量、内部 DTO、完整表字段和算法实现仍由代码、迁移与测试负责。

发生冲突时按以下顺序判断：

1. 可执行代码、Alembic 迁移、锁文件和实际运行配置是最终事实。
2. 本文记录稳定的产品、进程、数据和主要调用链，应与第一项保持一致。
3. [`.agent/PLAN.md`](../.agent/PLAN.md) 与 [`.agent/subplans/`](../.agent/subplans/)
   记录当前打算怎么做；[`.agent/TODO.md`](../.agent/TODO.md) 记录当前具体动作。
4. [`.agent/TRACKER.md`](../.agent/TRACKER.md) 记录当前做到哪里；
   [`.agent/LOG.md`](../.agent/LOG.md) 只记录已经实际发生的历史。
5. [`roadmap/`](roadmap/) 只保存候选方向；[`../archive/`](../archive/) 只保存历史快照。
   两者都不能作为自动恢复工作的指令。

不得从已完成、暂停或未激活的计划推断当前仍需实现某项能力。只有稳定架构事实发生变化时
才更新本文；局部修复、测试调整、内部重命名和不改变边界的重构不要求机械同步总览。

## 2. 产品定位与复杂度预算

这是一个本地运行的个人知识库与证据约束 RAG 应用。当前只服务一名维护者在一台机器上
开发和使用；目标是尽快得到可靠、可理解的功能闭环，而不是建设可复用 RAG 平台。

### 2.1 后续开发默认规则

- 优先在现有调用链上做最短的垂直修改。能用一个函数、一个现有 Service 或一个现有字段
  完成时，不新增子系统。
- 不为假设中的多租户、HA、多 Worker、外部插件、通用工作流、未来 provider、审计平台或
  未确定的 API 预建抽象。
- 单一实现不自动需要 Port、registry、capability、profile、factory、wire schema、持久
  ledger 和 UI 投影。真实外部 I/O 边界，或已经出现至少两个实际实现/消费者时，才考虑相应抽象。
- 不把“可恢复、可观测、可版本化、fail closed”机械套到每个中间步骤。只对源文件、数据库
  一致性、最终回答、引用、密钥和破坏性操作保留与实际风险相称的保护。
- 当前只维护 current-only schema。没有需要保留的个人数据时，经用户明确授权可重置本地
  数据，不为历史版本建立兼容层、回填框架或在线迁移平台。
- 新功能先用最小实现证明本地收益，再决定是否增加持久化、开关、诊断页面或通用协议。
  不能说明当前收益的抽象默认不引入。

现有代码中的分层、profile、manifest、调度和诊断机制是已实现事实，不是新功能的最低交付
模板。本阶段不为了“变轻”而进行高风险目录合并；先冻结抽象增长，后续只按真实维护痛点
逐项删除重复层。

### 2.2 当前已实现能力

- Chat 用户界面可创建和删除知识库，批量上传、更新、查看和软删除常见文本、PDF 与 Office
  文档，并预览/排除解析后的 chunk、查看索引进度和直接测试检索结果。
- 单个 Worker 异步解析文档、切分、生成文本/可选多模态向量并建立索引。
- 默认精确向量检索；可选 PostgreSQL FTS 混合召回，并支持文本、双空间多模态和显式统一
  图文空间三种索引/检索模式。
- 可选 Graphiti Graph 检索：由用户选择的 Chat Profile 在空闲 indexing lane 将每个 serving
  Chunk 串行摄入为一个 Episode，FalkorDB 保存图事实，PostgreSQL 保存不可变 build 代际、
  active 指针及 Episode→Chunk 映射。新 build staging 时旧 READY build 继续服务；覆盖门与
  运行时探测通过后才原子切换。在线用 Graphiti edge RRF 扩展候选，但证据正文始终回映射到
  当前 serving 原始 Chunk，fact 只进入调试投影，protected hybrid seed 不被图候选挤出。
- Chat 可选择 Chat-only 的 auto 模式：先执行冻结 revision 上的普通精确向量检索，只有原生
  Agent 在取得 Simple 结果后判断存在关系、别名、关系链或跨文档证据缺口时，才最多请求一次
  Graphiti supplement。supplement 复用 Graphiti candidate search 和原始 Chunk hydration，
  不再发起第二次 vector/FTS seed；未配置、未就绪、运行时不可用和无新证据都以安全结果码返回，
  edge fact 不进入 prompt、Citation 或回答正文。
- 持久 ChatSession / ChatRun、Session 短期上下文，以及动态提供当前可用工具的原生
  Tool-Calling Agent。常规工具为检索、计算和提交回答；满足门控时临时出现一次 Graphiti
  supplement。所有运行共享证据约束回答、拒答与引用边界。
- 本地用户 Chat 前端，以及文件协调、数据清理和本地评测工具。

### 2.3 当前明确不具备

- 生产认证、OIDC/JWT、ACL、多租户隔离、审计或合规保留。
- HA、多 Worker 安全接管、Outbox、第二队列、事件重放或正式备份恢复。
- 在线全量重建、历史 schema/profile 兼容、多嵌入空间在线迁移。
- 通用插件/连接器平台、公开管理控制面、长期记忆或拥有开放工具集的自主 Agent。
- 持久 step ledger、跨问题 planner、Claim–Evidence 图及其 UI。


由于没有生产认证、隔离和运维保障，不得把当前应用暴露为共享或互联网服务。这是安全边界，
不是要求提前补齐生产能力；除非用户明确改变产品定位，否则这些能力保持不做。

## 3. 技术与运行基线

精确版本由 `pyproject.toml`、`requirements.lock`、前端锁文件和 Compose 镜像摘要负责。
本文只记录影响理解架构的技术选择：

| 范围 | 当前选择 |
| --- | --- |
| 后端 | Python 3.12、FastAPI、Pydantic、异步 SQLAlchemy、asyncpg、Alembic |
| 数据库 | PostgreSQL 18 + pgvector；当前 migration head 为 `0015_remove_retired_chat_state` |
| 文档解析 | 原生 Docling；当前 PDF profile 在 Worker 管理的可终止子进程内按确定性页段解析 |
| Chat 执行 | 普通异步原生 Tool-Calling loop；无 Agent 框架或图运行时 |
| 模型接入 | OpenAI-compatible Chat/文本 Embedding；Tongyi 多模态 Embedding；固定离线 MiniLM reranker；已验证 Embedding 维度 64..4096 |
| 前端 | React、TypeScript、Vite；唯一前端为 `apps/web-chat` |

用户可在本地设置中维护 OpenAI-compatible Chat/文本 Embedding 与 Tongyi 多模态 Embedding。
Embedding Profile 可选择自动维度或显式验证 `64..4096` 内的一个候选维度；验证快照冻结实际
维度、请求参数模式和输入能力。干净安装不自动提供默认模型，模型提供商和模型均由用户在
设置中添加。文本和多模态向量统一存入按 space 与维度严格隔离的 `vector_record` 表。
本地资产存储和相关目录校验独立于可选的 legacy `MODEL_PROVIDER`。该环境组只为尚未绑定
模型修订的历史 ChatRun/Embedding Space 提供 adapter；当前原生 Agent 没有 Agent-level Provider
fallback。
纯 UI 配置的动态多模态模型无需旧环境 provider 即可索引和清理资产。

## 4. 总体架构

系统当前是模块化单体，通过几个本地进程运行：

```text
Browser
  └─ user Chat UI ───────┐
                         v
                    FastAPI API
                      │    │
       source files <─┘    └─> PostgreSQL + pgvector
                                  ^
                                  │
                              single Worker
                         ┌────────┴────────┐
                         │ indexing lane   │ chat lane
                         │ Docling child   │ model calls
                         └─────────────────┘

Maintenance CLI ──> local cleanup and reconciliation
Storage Init ─────> prepare source/asset/model-cache/model-secret/log directories and ownership
```

`compose.yaml` 运行 PostgreSQL、API、单 Worker、一次性 Storage Init/Migration/Maintenance，
以及用户前端。标准入口是 `./start-local.sh`。这些进程已经承载现有功能，因此继续保留；这不表示
后续功能必须增加新进程、队列、控制面或分布式协议。

主要运行原则：

- API 为摄取和 Chat 写入持久事实并快速返回；索引和回答由单 Worker 执行。
- Worker 为 Chat 和索引各运行一个串行 consumer；两个 lane 互不占用容量，不使用权重、aging
  或队列年龄探测。当前 PDF 每完成一个页段便释放 indexing lease，让已经到期的其他任务有机会
  先运行；stale recovery 由独立的低频 reconciler 执行。Graph 回填复用 indexing lane，只有
  文档任务为空时才领取一个内存 work item，不新增 Graph 任务表或进程。
- 显式 `POST /api/v1/retrieval/query` 是同步模型 I/O 的例外。
- PostgreSQL 是业务和任务状态的权威来源；本地卷保存源文件与派生资产。
- SSE 只交付完整但易失的安全 Agent 进度快照和已提交终态，不是持久事件系统；
  Web Chat 断线后不推测当前节点，完成后从持久 Agent Trace/result 重建摘要。
- 一个 Unit of Work 对应一个短数据库事务；外部模型和文件 I/O 不占用事务。
- 后续功能优先复用 API → PostgreSQL → 单 Worker 路径，只有确认无法满足需求时才增加组件。

## 5. 当前代码组织与轻量化边界

```text
apps/
  model_asset_runtime.py 最小共享模型/资产装配
  api/                 HTTP、错误映射、路由和依赖装配
  worker/              单 Worker 调度与依赖装配
  maintenance/         本地清理命令
  web-chat/            本地用户 Chat UI

src/rag_kb/
  domain/              业务类型、状态和错误
  schemas/             公开 API DTO
  services/            用例和跨资源协调
  answering/           原生 Agent loop、证据投影、校验、渲染与 runner
  document_processing/ Docling 消费、切分和纯处理逻辑
  indexing/            索引管线
  graph/               Graphiti build、Episode 回填与在线候选协调
  retrieval/           检索与融合
  scheduling/          PostgreSQL 任务调度
  memory/              Session 短期上下文
  ports/               主要外部 I/O 协议
  adapters/            文件、Docling、模型和检索存储实现
  repositories/        持久化访问
  uow/                 事务边界
  config/ db/ auth/ observability/
```

根目录 `architecture.toml` 描述当前 Python 顶层 import 方向。它用于防止环依赖和基础设施
反向渗透，不要求为每个新概念建立新目录或新层；basic suite 会实际解析该文件并逐个核对本地
import edge。`answering` 只依赖 context loader、retriever、visual preparer、result persister 与
progress reporter 的窄 Protocol，具体 `services` 实现由 Worker composition root 注入，因此
`answering` 与 `services` 不形成 runtime import 环。

### 5.1 保留的基础边界

- `domain` 不依赖 FastAPI、SQLAlchemy、LangChain、Agent/graph framework 或具体 adapter。
- SQL 留在 repository/db 一侧；外部 SDK 留在 adapter 或 composition root 一侧。
- API/Worker/Maintenance 继续保留各自 `dependencies.py` composition root；仅重复的模型/资产
  分支由 `apps/model_asset_runtime.py` 共享，不提供 registry、插件或通用 DI。
- 文件、数据库和 provider I/O 的 async/transaction 边界保持明确。
- 安全相关的 workspace 过滤、asset 授权、引用校验和 secret 隔离不得因简化而绕过。

### 5.2 不再强制复制的分层

- 小功能放入职责最接近的现有模块，优先直接、可读的函数或类；不要为了名称对称创建空
  package、转发 Service 或一对一 wrapper。
- Port 主要用于真实外部 I/O。纯内部协作只有出现第二个真实实现、确需替代难测依赖，或
  已经产生明显耦合时才抽协议。
- API DTO、domain dataclass 与 ORM model 的分离是当前公共/持久边界事实；内部临时数据
  不自动复制成三套模型，也不为单一调用新增 versioned wire。
- registry、profile、capability endpoint、feature flag、fingerprint、manifest、durable
  step ledger 和通用事件系统都需要当前、可验证的使用理由。
- 可以在独立的小型重构中合并已经证明重复的层，但不以追求理想目录图为由进行全仓搬迁。

## 6. 配置与外部依赖

配置由 `src/rag_kb/config/settings.py` 定义，以 `.env.example` 为支持面。当前设置对象严格、
冻结并拒绝未知字段。环境配置只保留本机/容器差异、secret、真实行为调节和有价值的回退；
单一实现身份、固定安全限制与兼容事实直接留在代码的单一来源，不再通过 Settings 重复转发。

当前重要选择：

- 身份固定为本地 development principal/workspace；调用者不能选择身份。
- ChatRun 冻结所选 Chat profile revision；索引与检索按 EmbeddingSpace 绑定的 profile revision
  解析 adapter。Simple 与手动 Graph snapshot 保存 profile version、strategy、`top_k` 和
  `rerank_mode`；Graph 外层模式额外冻结 `graphiti_edge_augmented_v1` 与 `graphiti_edge_v1`
  augmentation，内部 seed 仍使用 hybrid。Chat-only auto 使用独立的
  `adaptive_graphiti_v1` snapshot，冻结 exact-vector Simple、Agent evidence-aware router
  和 `graphiti_edge_v1` supplement，最多一次 supplement，不改变默认 vector 或直连 Graph API。
  retry 按当前进程配置解析候选数、阈值和融合权重。Chat 只有一个固定原生 Agent 路径。
- 检索默认 exact vector；hybrid FTS 由简单设置开关控制。
- live Agent progress transport 默认关闭，只能通过进程级 `CHAT_DELIVERY` 配置显式开启；它不属于
  Chat/Model/Retrieval profile，也不随 ChatRun 冻结，终态始终来自 ChatRun。
- 知识库创建时选择 text-only/multimodal parsing、structural/semantic chunking，以及兼容的
  text-only、dual-space 或显式 unified embedding 绑定。
- source、asset staging/final 与 parser-temp 目录始终按同一 root/文件系统边界校验，不依赖
  legacy 模型配置是否存在；legacy provider 仅为未绑定模型修订的历史运行事实提供可选 adapter，
  不参与当前 Agent 的失败回退。

新增配置只用于用户确实需要切换的行为、secret/环境差异，或一个值得保留的安全回退。不为
单一固定实现添加开关，也不保留已经不再支持的兼容设置。

## 7. 数据架构

Alembic 是 schema 来源。当前迁移链可从空数据库建立 head；`0004` 事务内原样迁移
`0003` 的 768/1024 维向量，`0005` 增加知识库软删除与 chunk 检索排除，`0006` 为索引任务
增加有界 PDF 解析进度和分段 continuation 状态，`0007` 调整回答策略默认值，`0008` 将
知识库检索默认中的布尔 `rerank` 迁移为显式 `rerank_mode`；`0009` 增加冻结的原生 Agent
budget/trace，`0010` 删除旧 workflow configuration/state 及其中的 ResearchResult/SearchTrace
诊断，`0011` 将原生 Agent 收敛为仅保留模型循环轮次上限并原样迁移历史 usage/trace 观测，
`0012` 曾增加旧自研实体图投影；`0013` 增加不可变 Graphiti build、active build 指针和
Episode→Chunk 映射；`0014` 删除旧 `index_graph_chunk`、`graph_entity_mention` 与
`graph_relation_assertion`，并把旧 Graph 配置安全降为 disabled；`0015` 删除 native Agent 从未
写入的 ChatRun `final_llm_context` 列。Graphiti 派生事实不阻塞
普通索引发布，只有 build ready、覆盖完整且运行时探测通过时才可用于在线检索。
除此之外不承诺任意历史版本兼容。主要持久事实为：

| 范围 | 主要实体 |
| --- | --- |
| 知识库与文档 | `Workspace`、`KnowledgeBase`、`Document`、`DocumentVersion` |
| 索引 | `EmbeddingSpace`、`IndexRevision`、`IndexedDocumentVersion`、`IndexingJob` |
| 检索数据 | `IndexChunk`、词法派生、资产/关系、可变维度 `VectorRecord`、Graph 配置、Graphiti build 与 Episode→Chunk 映射 |
| Chat | `ChatSession`、`ChatMessage`、`ChatRun`、`Citation` |
| 本地协调 | 幂等记录、文件清理记录、必要的索引计划/manifest |

数据保护重点是：源文件与数据库事实一致、同一文档只服务 ready 的索引、Chat 终态和引用
原子提交、失败任务能够在单 Worker 场景下重试。manifest 只验证一次 candidate 构建的完整性；
失败重试清空非资产派生事实时暂时保留资产记录，在事务外删除该 candidate 的本地资产，成功
后再删除资产记录并从源文件完整重建。当前 PDF 解析只在同一 source/profile/target 身份下
复用页段 checkpoint；下游 Chunk/Vector 仍完整重建，不提供跨 profile 恢复或历史兼容。不得由
现有索引链路推导出“任何新流程都必须拥有 ledger/manifest/replay”。

本地评测由独立的 `tools/evaluate_multimodal_real.py` 按需执行并输出结果，不进入 API、Worker、
Unit of Work 或业务 schema。没有当前消费者时不预建评测数据集、run/result 表和 repository。

没有要保留的数据时，本地 schema 变化可选择经用户确认后 reset；只有用户明确需要保留数据
时才设计回填或兼容迁移。任何删除本地数据的命令仍需明确授权。

## 8. 文档摄取与索引

```text
upload
  -> validate type/size/basic structure
  -> atomically store source file
  -> commit DocumentVersion + queued IndexingJob
  -> Worker claims job
  -> current PDF profiles: deterministic page segments in a killable child process
  -> persist content-safe progress/checkpoint and yield between segments
  -> globally reassemble and validate one DoclingDocument
  -> legacy/non-PDF profiles: one isolated Docling conversion
  -> structural or semantic chunks + optional visual assets
  -> role-bound text and optional multimodal embeddings
  -> persist derived rows and mark target ready/serving
```

当前支持 TXT、Markdown、HTML、CSV、PDF、DOCX、PPTX 和 XLSX。Markdown 可使用 `.mdz` Bundle
携带本地图片，普通公网图片会在准入阶段快照。本地 Docling 工件、格式/归档/像素/页数等
资源限制和可终止的解析子进程用于保护本地数据与 Worker。文档索引不再按墙钟耗时失败；
当前生产装配直接构造默认 `ParserLimits()`：PDF 使用 CPU 单线程、OCR/Layout/Table
batch 1 和每段初始 20 页，这些值目前不是用户运行时配置项。系统在段间持久化进度、
让出索引 lane。保持 Docling 区域 OCR 和 TableFormer Accurate，不用“存在文本层”关闭整份
OCR，也不提高 conversion/indexing 并发，以守住 6 GiB Worker 上限和混合页面、多模态资产质量。

结构切分和语义切分都直接消费一次 Docling conversion 结果。多模态路径保存受限的 page、
picture 或 table image，并把文本与视觉表示投影到现有 Evidence/asset 关系。dual 模式分别
使用文本和跨模态 space；unified 模式让文本、查询和图片复用同一已确认的多模态 profile 与
space。精确 profile
名称、token/图片预算、hash 和持久字段由 registry、settings、迁移和测试负责，不在总览重复。

删除先让数据库事实不可服务，再重试物理文件清理。Maintenance 只清理退休派生数据；不会
自动删除 active/candidate 数据。

## 9. 检索

默认路径是当前 serving revision 上的精确 pgvector cosine 检索。显式开启 hybrid 后，系统
并行执行 dense 与 PostgreSQL FTS，并用确定性 RRF 融合。text-only 生成一个文本 query
vector，dual 模式分别生成文本与跨模态 query vector，unified 模式只生成一个 query vector
并复用于文本/视觉 lane；每条 SQL 仍强制限定 role 绑定的 space 与维度。检索结果统一投影为
`EvidencePack`，再由回答链路进行阈值、关系和视觉准入。

检索始终限定当前 workspace、knowledge base、active revision、ready/serving target 和可用
源版本。现有 hybrid manifest 检查用于避免返回半成品索引；不支持时明确失败，不把不一致的
结果伪装成成功。

ChatRun 保存的 index revision 是创建时一致性 guard，而不是历史索引读取参数。当前公开 KB
生命周期没有 revision switch/reconfigure 入口，检索始终读取当前 active serving revision，并在
结果与 ChatRun guard 不一致时返回 `CHAT_REVISION_MISMATCH`；删除、失活或已替换的 revision 不会
为历史 ChatRun 继续服务。若未来新增 revision 切换 API，必须另行定义冻结检索参数和旧 revision
保留期，不能沿用当前 guard 假称已支持历史 serving。

当前精排选择为 `none | classic | local_minilm_v1`。候选先按 lane 准入一次：exact/dense 使用
文本 cosine 门，lexical 使用 FTS rank 与完整 manifest，cross-modal 使用自身 cosine 门；随后
才在已准入集合排序并最终截取 `top_k`。`classic` 保留本地确定性排序并作为知识库默认，不能
以词面覆盖或排序分改写 cosine/FTS 准入事实；hybrid 的 lexical lane 不复用 dense cosine 门。
`local_minilm_v1` 使用构建时固定、运行时离线的多语言 MiniLM ARM64 INT8 ONNX
工件，只对现有准入后的最多 20 个 text/table 候选重排。模型 tokenizer 将 query 截至 96
tokens、层级截至 32 tokens，并把超过剩余 512-token pair 预算的正文按段落或表格行临时窗口化
（64-token overlap、总窗口最多 80），以窗口最大 logit 聚合回原 Chunk。模型分数不覆盖原
Evidence score/准入事实，窗口也不持久化；纯视觉候选不送入模型。Native Agent 与
Retrieval Debug 都可使用该冻结模式；模型不可用时明确失败且不静默回退。

Native Agent 可通过 `search_knowledge_base` 每轮提交一至三条 Query，服务端在冻结的
workspace/knowledge-base/index revision、检索策略与 top-k 内执行并在证据池中按 chunk 去重。
adaptive ChatRun 先只允许 Simple lane；一次合法 Simple 调用完成后，即使其准入结果为空，Agent
也可按三个固定关系/证据缺口原因请求一次 `graphiti_supplement`，且该工具在最后一个普通模型
轮次仍可用。补充从 active READY build 解析 serving 身份；配置进入 building 只表示 staging，
不会遮蔽仍 active 的旧 READY build。补充复用 Graphiti candidate search、probe 和当前 serving
Chunk hydration。`none`/`classic` 按 edge/path/chunk 身份稳定排序且不依赖 MiniLM；只有冻结模式
为 `local_minilm_v1` 时才用 MiniLM 重排，低分或未打分 Chunk 不会因此被删除，模型分也不写入
`GRAPH_PATH` Evidence。最终最多加入 4 条新 Chunk、同一 edge 最多 2 条，
并排除已经在证据池中的 Chunk；它不执行第二次 vector/FTS seed。supplement 的 route result
只允许 `admitted`、`no_new_evidence`、`not_configured`、`not_ready`、`runtime_unavailable`
等安全码，公开 trace 保留真实 `tool=graphiti_supplement` 与 lane，edge fact 永不进入 Agent tool
result。
manual Graph 固定为最终 `top_k` 预留 2 个 Graph 槽：hybrid 候选查询宽度仍按
`min(40, max(12, top_k * 2))` 计算，但 seed 输出最多为 `top_k - 2`；packing 先完整保留这些 seed，
再按 path-whole 规则使用剩余至多 2 个槽，不会静默挤出已返回 seed。
Agent 只保留最多 8 个普通模型轮次的有限循环护栏，不限制 Query、计算或 EvidenceRef 的累计数，
也不比较或拒绝重复 Query。
题面中的文件名不触发分类、硬 document scope 或全文预读，因此同一知识库中被引用的其他文档
仍可被检索。每个结果获得运行内稳定 EvidenceRef；首次命中向模型发送索引保存的完整 chunk，
之后同一 EvidenceRef 只返回已发送标记，不截断、摘要
或重复发送正文，也不阻止重复 Query。普通检索未命中不能证明内容未出现，Agent
不支持以未命中构造缺失性结论。系统没有开放工具集、跨问题 planner、持久 step ledger 或
跨 run provenance。固定 Agent 另有一个受限 Decimal
`calculate` action：使用 512 字符表达式、只允许来源 Evidence 中的十进制操作数和
`+ - * /`，结果继续引用原始 Evidence，不把计算器伪装成来源。

## 10. Chat 与回答

API 创建 ChatRun 时在短事务内冻结知识库/revision、检索 preset 的 version/strategy/`top_k`/
`rerank_mode`（auto 另含 router/augmentation）、原生 Agent 模型循环轮次上限、回答策略、不可变模型修订和最近已完成 Session turns，然后
返回 `202`；模型调用由 Worker 执行。公开请求没有 workflow 模式。

回答策略中的 `answer_style` 与 `insufficiency_policy` 当前会被校验、冻结并通过 API 返回，但原生
Agent 尚未读取它们来改变 prompt、工具循环或确定性渲染；当前实际运行行为是单一的 evidence-only
逐 claim salvage/拒答路径，不能把这两个持久字段描述成已生效的 Agent 分支。

当前 Chat 执行是普通异步 Tool-Calling loop：

```text
load_context
  -> model chooses search_knowledge_base / calculate
  -> server executes bounded tool and returns stable refs
  -> adaptive mode may execute one Graphiti supplement after a completed Simple call
  -> model calls submit_answer when ready
  -> if the ordinary loop reaches its limit, one extra submit-only call finalizes
  -> claim-level deterministic validation and salvage
  -> persist_result
```

ChatRun 是唯一持久执行状态；没有 graph checkpoint、Controller、Verifier、独立 Generator/
Repair 或逐步骤 ledger。成功终态原子保存有界 Agent Trace。迁移
`0010_drop_legacy_workflow` 已删除旧 workflow configuration/state 及其中的
ResearchResult/SearchTrace 诊断；核心 ChatRun、消息、答案、Citation、usage 和 timing 事实保留。
ChatRun 内部 trace 保存 claim salvage 的 rejected count、内部 reason 与 submit-only repair，供本地
诊断使用；公开 API/SSE 会剥离这些内部字段，只保留隐私审查过的工具、lane、安全枚举与计数。
共享 trace artifact key 属于 domain 契约，不由 `services` 反向导入 Agent 实现。
`ChatAnsweringState` 只保存真实 Evidence 可用引用、原始 submit、确定性校验结果与渲染结果；
不再伪造旧 assessment/structure-validation 状态。成功终态 timing 记录实际 outcome、引用、检索、
视觉和 query-rewrite 事实，不写空 validation 占位。

回答边界保持：

- 零准入证据时确定性拒答；有证据时模型仍可判断问题无法充分回答。
- 文档、历史与图片都是 prompt 中的不可信数据，不能扩大权限或引用范围。
- `submit_answer` 必须通过严格参数和逐 claim 校验；非法 claim 被局部删除，仍有合法 claim 时
  降级为 `partial`，零合法 claim 才确定性拒答。
- 事实 claim 只能引用本次已授权 Evidence；实际未加载的图片不能产生视觉引用。
- Graphiti supplement 水合后的 text/table Chunk 同时保留 `graph_path` provenance 与对应的
  `text`/`table_text` 表示；路径 provenance 不是视觉形态，只有缺少可引用文本表示的纯视觉
  Evidence 才必须先实际加载资产。
- 视觉开关和图片数、单图 bytes、运行总 bytes、pixels 限额取自 ChatRun 创建时冻结的模型配置，
  并在整个 Agent run 内累计消费；进程 adapter 只执行不可变硬上限。caption/OCR/table text 是
  可独立引用的文本表示，即使原 Chunk modality 为 image 也不要求发送图片。citation 的 asset
  snapshot 只在对应资产通过授权与完整性检查并实际发送后附加，终态 visual content、decision
  与 snapshot 均按 asset identity 对齐。
- Agent 自行决定需要检索和引用哪些文档；服务端不从题面文件名生成 required Document 清单，
  也不以文档覆盖率改变回答结果。
- 最终答案、assistant message、Citation 和 ChatRun 终态在同一所有权边界提交。

Provider 单次调用的 SDK timeout 与有限 retry 由一个逻辑预算统一计算：
`timeout * (max_retries + 1) + 60 * max_retries + 1` 秒；Adapter 的外层总预算覆盖整个
semaphore/retry 窗口。Worker 启动时拒绝不大于该预算的 Chat Agent deadline；本地默认 deadline
为 420 秒，ChatRun 的既有 attempt 上限不因 Agent 而增加。
Agent progress 是 content-safe、易失且不可重放的快照；断线后读取权威 ChatRun。

这些约束直接保护回答可信度和个人数据，继续保留；模型调用的细粒度 timing、wire 版本和每个
中间 hash 不应自动成为未来功能的强制模板。

## 11. 公开接口与界面

公开接口位于 `/api/v1`，主要分为：

- knowledge bases、documents、document-version creation、chunk inspection/exclusion 和 indexing jobs；
- model settings snapshot、provider/profile mutation、provider model catalog、validation 和
  workspace model selection；
- retrieval capabilities/query、Graph 配置 `GET/PUT /knowledge-bases/{kb_id}/graph-config` 和授权
  `GET /index-assets/{asset_id}/content`；Graph 查询仍是明确的外层 `graph` 模式，路径仍只返回
  当前 serving 的原始 Chunk Evidence；Chat creation 另支持不出现在 retrieval capability
  列表中的 Chat-only `auto` 模式；
- chat sessions、messages、runs 和 events；SSE 进度事件为 `agent.progress`。

上传、状态、分页、幂等和错误的精确契约以 OpenAPI、schema 和路由测试为准。未实现能力不
添加占位成功路由。索引任务状态对当前 PDF 额外公开 content-safe 的阶段、页段、OCR、表格、
累计耗时和子进程峰值 RSS；前端沿用现有卡片、颜色、间距和进度条展示这些字段，不把
会重叠的阶段耗时相加成虚假的精确百分比。

`apps/web-chat` 是本地唯一前端，提供知识库创建/删除、文档批量导入/更新/删除、索引进度、
chunk 预览/排除、检索 debug，以及知识库/Session 选择、统一 Native Agent 提问、
终态回答、有界工具轨迹和证据抽屉。管理页复用 Chat 既有视觉 token 与侧栏，不形成第二套 UI。
它只调用公开 API；诊断功能不等于生产管理控制面。

## 12. 安全、失败与恢复边界

即使是本地个人项目，也继续保留以下低成本、高价值约束：

- 默认绑定 loopback；provider key、DSN 和本地口令只进入未提交环境文件。
- 服务端生成文件路径和身份；公开请求不能选择主机路径或 workspace。
- 用户内容、provider body、图片 bytes/Data URL 和 secret 默认不写日志。
- 数据库事务短小，外部 I/O 在事务外；过期 Worker attempt 不能覆盖新终态。
- 文件删除和本地数据 reset 属于破坏性操作，必须明确目标并得到用户授权。
- 引用和视觉资产在返回前重新限定 workspace/KB/version，并验证稳定身份。
- `vision_enabled=false` 时不读取或发送视觉资产；多轮检索不会重置视觉预算，也不会把未发送的
  table/image asset 复制进 citation snapshot。

索引 job、ChatRun lease、heartbeat、有限重试和文件协调是当前单 Worker 闭环的一部分。普通
轮询直接尝试原子 claim，不先执行只读队列探测；stale reconciliation 启动时执行，之后每
30 秒执行一次。PDF continuation 只是正常时间片让出，不增加 retry attempt；indexing lane
先按 `coalesce(next_attempt_at, created_at)` 排序，因此已到期任务可在页段之间运行，
已经到期的 continuation 也不会被后来任务永久饿死。Chat lane 独立按
`next_attempt_at NULLS FIRST`、`created_at`、`id` 排序已可 claim 的 run。失败的索引 candidate
重试时清空部分
plan/manifest/chunk/vector 等非资产
派生事实，同时保留资产记录直至本地资产删除成功，再删除记录并按当前 profile 从源文件重新
执行；serving target 不参与该清理。它们只保证当前运行模式，不宣称 HA 或多 Worker
takeover。新增本地功能若一次失败后重跑即可，默认复用现有 job/run 状态，不新建独立恢复
子系统。

## 13. 可观测性、运行与测试

API、Worker 和 Maintenance 输出版本化的安全结构化 JSON，并自动关联 Trace/Run/Job、
attempt、进程 runtime ID 与代码位置。异常只保留类型、稳定 fingerprint 和有限的
module/function/line 栈，不保存消息、源码行、locals、正文或密钥。Compose 将应用 JSONL
持久化到被 Git 忽略的 `.runtime/logs`，按 10 MiB/五份备份轮转；Docker 原始输出另有
20 MiB/五份的短期上限。一条本地命令可导出只含安全事件、健康状态、容器状态和 Git 摘要的
issue 诊断包。Worker 另有本地 heartbeat。PDF 子进程按节流频率报告 page parse、OCR、
Layout、Table、页面装配和文档装配计数；数据库只保存固定 allowlist，不保存正文或路径。
数据库 readiness 只确认连接和当前 Alembic head，不重复巡检 migration 已定义的全部表、
类型、索引和约束。当前没有 metrics、告警、分布式 tracing 或生产值班平台，本地范围也不
需要这些外部控制面。

`./start-local.sh` 通过 Git common directory 定位主 worktree，使同一仓库的 linked worktree
共享主 worktree 中被忽略的 `.env.local`、可选 `.env` 与稳定 Compose project；不会再按
worktree 各自生成数据库身份。每次启动 PostgreSQL 后，它通过容器内本地 socket 幂等校准
admin、migration、runtime 三个角色的密码、database/schema owner 和基础连接权限，再执行
Alembic。角色校准不会重置业务卷或改写业务数据；随后仍按原流程升级到 Alembic head。没有
`.env` 时直接使用 `.env.example`；需要本地应用覆盖时才在主 worktree 创建 `.env`。模型凭据
通过 Web Chat 右下角的模型设置维护，不提交到仓库。常用命令：

```bash
./start-local.sh
docker compose --env-file .env.local ps
docker compose --env-file .env.local logs --no-color api worker
PYTHONPATH=src:. .venv/bin/python tools/collect_diagnostics.py
PYTHONPATH=src:. .venv/bin/python tools/smoke_local.py
docker compose --env-file .env.local down
python3 tools/run_database_tests.py
```

测试使用 `unittest`，现有目录包括 `tests/basic`、`tests/unit`、`tests/contract` 和
`tests/integration`。验证按风险选择：

- Python 行为变化至少运行 basic suite 和最接近变更的聚焦测试。
- 数据库迁移、并发或 repository 行为变化才运行相关数据库 integration。
- 数据库 integration 只通过 `tools/run_database_tests.py` 运行：每次创建唯一命名、Docker 动态
  loopback 端口、tmpfs 数据目录和 `trust` 无密码认证的一次性 PostgreSQL，从空库迁移后注入
  测试 DSN，结束时校验 owner label 并清理容器。它不读取持久库 `.env.local`，也不连接或
  TRUNCATE 个人数据库，不自动拉取缺失镜像；linked worktree 缺少 `.venv` 时复用主 worktree
  的 Python 3.12 环境。
- 改哪个前端就构建哪个前端；不要求无关前端同时构建。
- 纯文档变更只需检查链接、路径和 Markdown/diff，不运行应用测试。
- 每个不变量只在最低且最有证明力的层级保留测试：basic 负责导入和依赖边界，unit/contract
  负责可执行行为与公开 wire contract，database integration 负责 migration、catalog 和
  repository 不变量。不并行保留只检查 AST 形状、ORM metadata 清单、已退役配置键或示例文件
  字面值的重复测试。
- 不默认增加全量矩阵、性能/安全/发布 gate 或版本化报告；只有当前问题需要时才增加。

需要长期保留某次实际验证证据时，将结果写入 [`test/`](test/)，使用
`NN-MMDD-short-test.md`。报告记录被测基线、范围、环境、命令或用户场景、实际结果、失败与
未验证项；它不定义另一套测试流程，也不替代 `AGENTS.md` 的验证规则或 `.agent/` 的当前状态。

默认轻量 Python 检查：

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/basic -v
```

`tools/reset_local.py` 会永久删除本地业务数据。执行前只能先用以下命令检查精确卷目标；
CODE AGENTS 未获用户明确授权不得去掉 `--inspect-only`：

```bash
PYTHONPATH=src:. .venv/bin/python tools/reset_local.py \
  --env-file .env.local \
  --project-name rag \
  --inspect-only \
  --confirm DESTROY_RAG_KB_LOCAL_DATA
```

## 14. 轻量维护协议

项目任务只通过 [`.agent/`](../.agent/) 维护；旧的 `docs/implementation-plans/` 与
`EXECUTION-TRACKER.md` 已归档并停用。四个根文件各自只有一种职责：

- `PLAN.md`：当前打算怎么做，包括目标、边界、策略、子计划和完成条件。
- `TODO.md`：当前具体要做什么，只保留可执行且可验收的动作。
- `TRACKER.md`：当前做到哪里，只写有证据的状态、阻塞、下一步与验证。
- `LOG.md`：历史实际发生了什么，按日期追加结果、偏差和验证，不回写未来意图。

只有一个当前 PLAN。小型插入工作可作为 TODO 记录而不改项目方向；多阶段工作才在
`subplans/` 使用 `NN-MMDD-short-plan.md` 拆分。替换计划前必须先把已发生结果写入 LOG，
不得从旧计划、归档或 roadmap 自动恢复工作。

大型工作以 Git 作为代码历史与恢复事实：每个进入 `in_progress` 的 subplan 使用独立的
`feat/`、`fix/`、`refactor/`、`docs/` 或 `chore/` 短名分支；提交保持小而完整，风险操作或
切换上下文前建立可恢复 checkpoint。临时 `wip:` 只允许留在工作分支，不能进入最终合并历史。
子计划只有在相关变更已提交、提交态验证通过、执行状态已同步后才能合并；远程 push/PR 仍需
用户明确授权。

整份 PLAN 完成、取消或被替换时，先确认所有应合并提交已从目标分支可达，再收口
TODO/TRACKER/LOG，把最终 `PLAN.md` 和完整 `subplans/` 一起移入
`archive/plans/NN-MMDD-short/`。随后创建下一份当前 PLAN，或明确写“无当前计划”，并清空
只属于旧计划的 TODO 与 subplans。`.agent/` 不保留已经结束的 PLAN 或子计划副本；LOG 继续作为
实际历史入口。若取消或替换时仍有未合并工作，最终记录必须保存准确 branch/commit 与处置；
不得为了收口而强行合并，也不得静默删除恢复分支。

| 变更 | 默认做法 |
| --- | --- |
| 小型 bug、内部重构、测试或文案 | 用短 TODO 跟踪，完成后更新 TRACKER/LOG；无需子计划或总架构更新 |
| 单一用户功能，范围清楚且可回退 | 用最短垂直实现；只调整受影响的 TODO、TRACKER、LOG 与契约 |
| schema、数据删除、跨进程/模块边界或多阶段高风险工作 | 更新 PLAN，并按需要建立少量子计划；执行期间持续更新 TRACKER |
| 稳定产品边界、进程、主要数据模型或公开 API 发生变化 | 同步本文相关章节，不要求复制实现细节 |
| 生产化、共享部署、多租户或 HA | 只有用户明确改变产品定位后才单独评估，不预建 |

其他规则：

- 没有活动工作时 PLAN 与 TRACKER 明确写“无”；不得把候选 roadmap 保持为默认下一步。
- PLAN 只说明意图，实际结果只进入 TRACKER/LOG；不得在四个文件之间复制完整正文。
- 配置、migration、profile、capability、ledger、监控和 UI 都不是新功能的自动配套项。
- 当复杂方案与简单方案能达到相同本地效果时，选择模块更少、持久状态更少、运行分支更少的方案。
- 若实现收益尚未验证，先做可删除的最小实验；不得让实验性路线反向扩大基础架构。

## 15. 文档治理

活动文档只允许以下结构：

```text
docs/
├── architecture.md
├── reviews/
├── roadmap/
└── test/
```

- `architecture.md` 是全项目唯一架构文档。稳定产品、进程、主要数据、公开 API、模块或运行
  边界变化时，必须在同一任务中主动更新相关章节。
- `reviews/` 只保存基于实际证据的 review 报告，命名为 `NN-MMDD-short-review.md`。报告本身
  不会自动成为任务；需要执行的结论必须进入 TODO。
- `test/` 只保存已经实际执行的测试报告，命名为 `NN-MMDD-short-test.md`。报告必须写明被测
  基线、范围与环境、命令或用户场景、结果与失败、未验证项；不得在此另写测试流程。失败或
  延后项只有进入 TODO 后才成为当前任务。
- `roadmap/` 只保存开放方向和概念边界，命名为 `NN-MMDD-short-roadmap.md`。不得写实现步骤、
  当前状态、验收清单或承诺日期；立项后才转入 PLAN/TODO。
- [`archive/docs-20260819/`](../archive/docs-20260819/) 保存旧架构专题、旧任务系统、旧 review、
  release/baseline 与其他历史资料；`archive/plans/NN-MMDD-short/` 保存以后结束的 PLAN 与其
  完整子计划。每次只新增一个最终归档包，已有归档只读且不能覆盖 Git、本文或 `.agent/`。

除固定的 `architecture.md` 外，文件名以短英文或简短拼音为主，避免重复项目名、阶段长句和
状态堆叠。新类型沿用 `NN-MMDD-short-type.md` 规则。
