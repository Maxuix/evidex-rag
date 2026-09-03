# Local Knowledge Base & RAG — 当前架构与轻量化边界

| 字段 | 内容 |
| --- | --- |
| 文档状态 | 全项目唯一当前架构文档（描述事实，不是目标蓝图） |
| 最后核对 | 2026-09-02 |
| 核对基线 | 当前 `main`、实际代码、配置、Alembic 迁移、Compose 与公开路由 |
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
3. [`.agent/PLAN.md`](../.agent/PLAN.md) 记录当前打算怎么做；需要时可在 `.agent/subplans/`
   放置独立子计划；[`.agent/TODO.md`](../.agent/TODO.md) 记录当前具体动作。
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
- 可选 Graphiti Graph 检索：由用户选择的 Chat Profile 和内置 Graph Schema Profile 在空闲
  indexing lane 将每个 serving Chunk 串行摄入为一个 Episode，FalkorDB 保存图事实，PostgreSQL
  保存不可变 build 代际、active 指针、冻结的 profile key/digest 及 Episode→Chunk 映射。
  新 build staging 时旧 READY build 继续服务；覆盖门与运行时探测通过后才原子切换。新 build
  使用 `graphiti_v4`，历史 `graphiti_v3` Software build 只按兼容路径读取；迁移 `0014` 留下的
  `graphiti_v1` 仅作为历史配置/build identity 读取，不参与 Graph serving 或 build。默认的
  `generic_open_domain_v1` 不传自定义 Graphiti ontology/instructions；`software_knowledge_v1`
  保留 Organization、Project、Repository、Service、
  License、LicenseExpression 与 AliasSurface 类型化抽取，并保存稳定的有向关系类型；
  `enterprise_knowledge_v1` 提供组织、职责、政策、流程、系统、项目、产品、设施、文档、术语和地点
  的类型化选择。检索先用
  Graphiti node hybrid search 解析题面实体，再以实体 UUID 为中心执行 node-distance、BM25、向量与
  原生 BFS 的组合搜索，最后只形成真实连通、无环的一至三跳路径。每一跳都必须回映射到当前 serving
  原始 Chunk。实体搜索窗口独立于最终路径 K；显式“最终/经由”问题优先完整长链，仅由多个关系词
  推断出的复合问题先为每个题面实体保留最短直接事实，再用剩余 K 补长链，避免单个稠密邻域或三跳
  路径耗尽证据预算。
  fact 只进入内部水合，不进入 Agent tool result；手动 Graph 先完整打包图路径，再用
  hybrid Evidence 回填余额。每个成功 Episode 的映射独立
  短事务提交；Episode UUID 由 build、Chunk 与 content hash 确定。失败 build 保留其 group 与映射，
  冻结输入未变化时显式 retry 复用同一 build 并只处理缺失 Chunk；输入变化或 force rebuild 才换代。
- Chat 可选择 Chat-only 的 auto 模式：在冻结 revision 上首轮同时暴露语义/关键词/邻域/文档清单
  检索与一等 `search_graph_relations` Graph Tool，由原生 Agent 自主选择，语义检索不是 Graph
  的前置条件。
  Graph Tool 只在存在 active READY build 时可见；每个 ChatRun 最多两次 Graph 调用，Graph 单次
  90 秒（ChatRun 绝对 deadline 为 600 秒形成 `min(90, remaining)`），候选 K 冻结为 16，完整
  一至三跳路径按 soft 12 / hard 16 去重 source chunk 原子打包，两次调用累计最多新增 32。
  不存在服务端提交 completeness guard。Agent 以冻结预算限制累计 token、证据和检索执行数；
  达到预算后切换到一次 submit-only 收尾，不对单条证据正文做按 token 截断。Graph Tool
  只返回 source chunk；edge fact 不进入 prompt、Citation 或回答正文；未配置、未就绪、
  运行时不可用、超时、被拒绝和无新增证据都以安全结果码返回，取消与超时可区分。
- 持久 ChatSession / ChatRun、Session 短期上下文，以及动态提供当前可用工具的原生
  Tool-Calling Agent。常规工具为 `semantic_search`、`keyword_search`（hybrid
  进程且 lexical manifest 覆盖完整时暴露）、`read_chunk_context`、`list_documents`、
  计算和提交回答；Graph READY 时首轮同时暴露 `search_graph_relations`，Graph 用完后从工具集移除。
  所有运行共享证据约束回答、拒答与引用边界。
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
| 数据库 | PostgreSQL 18 + pgvector；当前 migration head 为 `0028_agent_v5_default` |
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
- Chat 与索引任务的持久所有权令牌是递增 `attempt`，不保存 Worker 名称；heartbeat、终态、失败、
  reschedule 及索引 promotion 都以 repository 已绑定 workspace 加资源 ID、attempt 和活动状态做
  compare-and-set。`claimed_at`/`heartbeat_at` 继续支持超时回收。Graph build 仍使用独立的
  owner + lease token + expiry 协议，Worker liveness heartbeat 文件也保持不变。
- 显式 `POST /api/v1/retrieval/query` 是同步模型 I/O 的例外。
- PostgreSQL 是业务和任务状态的权威来源；本地卷保存源文件与派生资产。
- SSE 只交付完整但易失的安全 Agent 进度快照和已提交终态，不是持久事件系统；
  Web Chat 断线后不推测当前节点，完成后从持久 Agent Trace/result 重建摘要。
- 每次数据库操作使用独立的 SQLAlchemy `AsyncSession` 和一个显式短事务；成功显式 commit，
  失败或退出 rollback 并 close，且关闭 autobegin，避免边界外意外开启新事务。外部模型和文件
  I/O 不占用事务。
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
  config/ db/ observability/
```

根目录 `architecture.toml` 描述当前 Python 顶层 import 方向。它用于防止环依赖和基础设施
反向渗透，不要求为每个新概念建立新目录或新层；basic suite 会实际解析该文件并逐个核对本地
import edge。`answering`、`retrieval` 与 `scheduling` 直接依赖各自的具体应用服务类型；这些
依赖只在类型检查分支出现，runtime 仍由 Worker composition root 装配。外部模型、存储、解析器
和 Graphiti 边界继续集中在 `ports/`。

Graph schema profiles 是 `graph/` 下的代码所有、不可变 registry；adapter、repository 和
retrieval 只读取 registry 以解析冻结的 Profile identity 或编译后的 Graphiti 类型，具体依赖边
由根目录 `architecture.toml` 明确登记。当前内置 Profile 为默认的 `generic_open_domain_v1`、
兼容历史 Graphiti v3/v4 软件语料的 `software_knowledge_v1`，以及仅允许当前
`graphiti_v4` 的 `enterprise_knowledge_v1`。Enterprise contract 覆盖组织/组织单元、人员与
角色、RACI 职责、股权/设立/并购/投资、制度/流程、系统/产品、项目/设施/地点、文档/术语，
以及合作、合同、供应、研发、认证和许可关系；别名使用 Graphiti native dedupe，不建立独立
alias node。新增代码内置 Profile 不需要 catalog migration：Profile API 直接列举 registry，
PostgreSQL 继续只冻结每个 KB 配置和 build 的 key/digest；切换 Profile 必须创建新 build，
READY 旧 build 在切换完成前继续服务。

### 5.1 保留的基础边界

- `domain` 不依赖 FastAPI、SQLAlchemy、LangChain、Agent/graph framework 或具体 adapter。
- SQL 留在 repository/db 一侧；外部 SDK 留在 adapter 或 composition root 一侧。
- API/Worker/Maintenance 继续保留各自 `dependencies.py` composition root；仅重复的模型/资产
  分支由 `apps/model_asset_runtime.py` 共享，不提供 registry、插件或通用 DI。
- 文件、数据库和 provider I/O 的 async/transaction 边界保持明确。
- 安全相关的 workspace-bound repository、asset workspace 校验、引用校验和 secret 隔离不得因
  简化而绕过。

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

配置由 `src/rag_kb/config/settings.py` 定义，以 `.env.example` 为支持面，本地默认文件为
`.env.local`。同一 manifest 可以包含 Compose 身份/端口/数据库启动凭据与 `RAG_KB__...`
应用设置；应用加载器只接收后一个命名空间，仍严格拒绝其中的未知字段，不会把 Compose-only
键误当成应用配置。`tools/local_runtime.py doctor` 只输出 project、端口、枚举和计数，用于发现
双 env、旧 override、linked worktree 或非 canonical Compose project，不回显任何配置值或路径。
当前设置对象严格、冻结。环境配置只保留本机/容器差异、secret、真实行为调节和有价值的回退；
单一实现身份、固定安全限制与兼容事实直接留在代码的单一来源，不再通过 Settings 重复转发。

当前重要选择：

- 应用只保留一个配置确定的内部 workspace namespace。请求不携带、解析或传播 principal/client
  身份，旧身份 header 也不能改变 workspace；repository、文件和资产服务在 composition root
  绑定该 workspace。API 与 Worker 在 schema readiness 通过后幂等创建该 namespace，因此全新空库
  不依赖外部 provisioning。ChatRun 与 ContentMutation 幂等范围为 `endpoint + idempotency_key`。
- ChatRun 冻结所选 Chat profile revision；索引与检索按 EmbeddingSpace 绑定的 profile revision
  解析 adapter。Simple 与手动 Graph snapshot 保存 profile version、strategy、`top_k` 和
  `rerank_mode`；Graph 外层模式额外冻结 `graphiti_path_augmented_v3` 与 `graphiti_path_v3`
  augmentation，内部 seed 仍使用 hybrid。Chat-only auto 使用独立的
  `adaptive_graphiti_v3` snapshot，冻结 exact-vector Simple、一等 Graph Tool 参数
  （`graph_edge_limit=16`、`graph_source_chunk_target=12`、`graph_source_chunk_limit=16`、
  `graph_call_timeout_seconds=90`）与 router `native_agent_graph_tool_v1`，最多两次 Graph 调用，
  不改变默认 vector 或直连 Graph API。
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
 写入的 ChatRun `final_llm_context` 列；`0016` 把 ChatRun 的 agent/retrieval/trace snapshot
 确定性升级为 v3 一等 Graph Relations Tool（budget 增加 `max_graph_calls`，trace 增加
`call_index`、`invocation_source`、`duration_ms` 与各计数/跳数，旧 guard 事件标记
 `invocation_source=legacy_guard`，不算 duration 或计数）；`0017` 为 Graph config/build 增加
 Schema Profile key/digest 并把既有记录准确回填为 Software；`0018` 增加 per-build work lease，
 让同一 Graphiti build 的 Episode 写入在 Worker 之间串行。Graphiti 派生事实不阻塞
普通索引发布，只有 build ready、覆盖完整且运行时探测通过时才可用于在线检索。`0019` 为 source
file content mutation 增加 `pending/completed/failed` 终态、稳定 failure facts 和 reservation
时间，并由 bounded reconciler 按条目原子收敛；不可恢复的文件清理保留 durable cleanup 记录。
`0020` 显式授予 runtime role 对 Graphiti work lease 表的读写权限；`0021` 允许 Agent trace
记录 `clarify` outcome；`0022` 把 ChatRun agent budget 从两键
（模型轮次/Graph 调用）扩展为六键累计资源预算（新增 token 总量、证据条数、检索调用数与软
截止预留），既有行原地补齐默认值；`0023` 增加公开 trace diagnostics；`0024` 移除
`agent_configuration`/`agent_trace` 对旧截止预留字段和完整 JSON key 集合的 CHECK 约束，并将
新默认值收敛为五个当前预算字段。既有 ChatRun 不回填、不删除；读取路径会忽略历史快照中残留
的旧字段。`0025` 从单用户本地模型删除 principal/client 列并把幂等范围收敛为 endpoint + key；
`0026` 在零活动 Chat、索引和 Graph work 前置条件下删除 ChatRun/IndexingJob 的 `claimed_by`，
保留 attempt、claim/heartbeat、backoff、状态和错误事实；Graph work lease 不变。
`0027` 把新 ChatRun 的 `agent_configuration` 列默认值改为 `native_tool_calling_agent_v4`
与同值五键预算，不回填历史行。`0028` 把默认值改为 `native_tool_calling_agent_v5` 与
单键 token 预算（`max_total_tokens`），同样不回填历史行；v5 执行器读取历史 v3/v4
配置时只取其中的 token 上限。
P2 的实际数据核查确认 active/retired revision 指针仍承担当前与软删除恢复，两个 READY Graph build
均为 active，PDF 分段任务真实使用 continuation；因此 revision/build identity、完整性 manifest、Graph
lease 与 PDF checkpoint 都保留。未使用的 Enterprise Graph profile 只作为需单独授权的完整产品删除
候选，不在配置收敛中隐式移除。详见
[index/Graph/PDF retention review](reviews/16-0902-index-graph-retention-review.md)。
除此之外不承诺任意历史版本兼容。主要持久事实为：

| 范围 | 主要实体 |
| --- | --- |
| 知识库与文档 | `Workspace`、`KnowledgeBase`、`Document`、`DocumentVersion` |
| 索引 | `EmbeddingSpace`、`IndexRevision`、`IndexedDocumentVersion`、`IndexingJob` |
| 检索数据 | `IndexChunk`、词法派生、资产/关系、可变维度 `VectorRecord`、Graph 配置、Graphiti build 与 Episode→Chunk 映射 |
| Chat | `ChatSession`、`ChatMessage`、`ChatRun`、`Citation` |
| 本地协调 | `ContentMutation` 幂等/终态记录、文件清理记录、必要的索引计划/manifest、本地 model-secret 引用 |

数据保护重点是：源文件与数据库事实一致、同一文档只服务 ready 的索引、Chat 终态和引用
原子提交、失败任务能够在单 Worker 场景下重试。manifest 只验证一次 candidate 构建的完整性；
失败重试清空非资产派生事实时暂时保留资产记录，在事务外删除该 candidate 的本地资产，成功
后再删除资产记录并从源文件完整重建。当前 PDF 解析只在同一 source/profile/target 身份下
复用页段 checkpoint；下游 Chunk/Vector 仍完整重建，不提供跨 profile 恢复或历史兼容。不得由
现有索引链路推导出“任何新流程都必须拥有 ledger/manifest/replay”。

本地 evaluator 不属于个人正式 `rag` 的业务流程。所有测试和 evaluator 入口都由宿主机 checkout 的
`.venv` 执行，不在应用容器内运行。纯 corpus 校验和 `--dry-run` 不访问数据库、Graph 或 Provider；
任何会访问 API、数据库、Graph 或 Provider 的模式，都只能连接用户明确提供且已经运行的 Python-side
test services、可丢弃的 test database，以及通过身份校验的 workspace/profile/source/model-secret
副本。不存在或伪造 test runtime 时，入口在发起外部 I/O 前失败；个人正式 API、`.env.local`、正式
`rag` 数据和源码 UUID 都不是 evaluator fallback。

测试请求不包含 Docker lifecycle 权限：不得为了测试 build/pull/tag 镜像，也不得 create、recreate、
restart、stop 或 remove 容器。已经运行的 Python-side test services 只作为宿主机 Python 的外部依赖，
测试不能改变其 lifecycle；依赖不存在或版本不兼容时记录未验证并停止，不以构建镜像或刷新容器补齐
环境。正式 `rag` Compose 环境只供用户的正式/端到端运行，不能由测试请求隐式启动。

`tools/evaluation_runtime.py` 只验证 owner-only host-Python runtime，不提供环境创建、销毁或 Docker
lifecycle。当前修复与评测只从主 checkout 的 `.venv` 运行；需要外部状态时由用户提供已运行的
Python-side test services。任何正式 `rag` Docker lifecycle 都必须另有明确的 runtime 请求，不能从
“运行测试/评测”推导。

当前只维护 `tools/evaluate_agent_complex_qa.py` 一个通用回归入口及
`evaluation/document-qa-v1/` 核心语料。`--dry-run` 离线验证冻结 corpus、case 顺序、证据 span 与摘要；
经单独授权的真实运行可选择 exact-vector 或 hybrid 检索，验证回答、引用、算术、多文档与缺失证据，
并使用独立 Judge profile 进行语义评分。入口只连接 workspace-bound、owner-only 的 host test runtime，
不 provision 服务、不管理 Docker/Compose lifecycle，也不把问题、回答、正文、Provider payload 或 secret
写入 Git。Provider 或 runtime 身份缺失时在外部调用前停止；真实模型仍必须使用仓库规定的 OpenCode Go
及固定模型，不允许 fallback。

已完成的 Adaptive Graph、routing、open-source、enterprise 与 large-evaluation campaign runner、语料、
launchd/supervisor/host-provisioning 控制代码和专属测试位于
`archive/evaluations/01-0901-completed-campaigns/`，不再是当前命令或架构依赖。忽略目录下的历史 runtime
checkpoint/报告保持原位，归档不删除用户结果。

`text/plain` 是公开上传合同的一部分，不经过 Docling 不支持的 TXT converter，而是以严格 UTF-8 直接构造
受同一 item/character 预算约束的 `DoclingDocument`。Markdown/CSV/Office simple pipeline 不消费 PDF
OCR/layout/table 模型，因此只要求一个有效的本地 artifacts 目录；PDF 转换仍必须在 converter cache 前通过
完整 frozen artifact manifest 校验，先处理文本不能绕过 PDF artifact 门。

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
携带本地图片，也可使用受限 data URI；HTTP(S) 与 protocol-relative 图片在准入阶段确定性拒绝，
摄取过程不解析 DNS 或连接外部网址。本地 Docling 工件、格式/归档/像素/页数等
资源限制和可终止的解析子进程用于保护本地数据与 Worker。文档索引不再按墙钟耗时失败；
当前生产装配直接构造默认 `ParserLimits()`：PDF 使用 CPU 单线程、OCR/Layout/Table
batch 1 和每段初始 20 页，这些值目前不是用户运行时配置项。系统在段间持久化进度、
让出索引 lane。保持 Docling 区域 OCR 和 TableFormer Accurate，不用“存在文本层”关闭整份
OCR，也不提高 conversion/indexing 并发，以守住 6 GiB Worker 上限和混合页面、多模态资产质量。

结构切分和语义切分都直接消费一次 Docling conversion 结果。当前 semantic v4 除 section、page、
table 和非正文 block 外，还把空行分隔且带短标题的内部记录投影为 `record` 硬边界；这保留 TXT、
Markdown 和 Docling inline group 中原本存在的独立记录，不按业务实体或评测 relation 识别内容。
legacy semantic v3 可读取但不用于新索引；Graph v2 不允许直接建立在 v3 semantic revision 上。
多模态路径保存受限的 page、
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

Native Agent 通过按召回通道划分的工具选择检索方式，每轮恰好一个工具调用。
`semantic_search` 在冻结 workspace/knowledge-base/index revision、检索策略与 top-k 内做
exact dense（含跨模态 lane）加冻结 rerank；manual graph ChatRun 仍经该工具分派到
`retrieve_graph`。`keyword_search` 只在进程 hybrid 开启且 lexical manifest 覆盖完整时暴露，
执行 FTS 后按 `lexical_rank` 截取，证据 `score_kind=LEXICAL`、`score=1.0/lexical_rank`，
不伪装 cosine 分；manifest/版本类 `INDEX_REVISION_INCOMPATIBLE` 软失败并摘除该工具，
`CHAT_REVISION_MISMATCH` 仍 fatal。`read_chunk_context` 锚定已签发的 text/table EvidenceRef，
固定 ±1 邻域，邻域证据 `matched_representations` 按 modality 派生为 `text`/`table_text`，
准入只做池去重与证据预算，不走搜索语义下恒为 False 的 `eligibility.usable()`。
`list_documents` 返回 serving 文档元数据（可选大纲），不进证据池、不可引用，每次计 1 次
retrieval。结果在证据池中按 chunk 去重。`search_closed` 时四个证据获取工具与 Graph 一并移除，
只留 calculate 与 submit。
adaptive ChatRun 在存在 active READY build 时首轮同时暴露一等 `search_graph_relations` Tool；
Graph 调用数达到冻结上限（默认 2）后该 Tool 从后续轮次移除，超限调用被拒绝且不发起外部查询。
Graph 每次沿冻结的 build/extractor 身份执行，配置进入 building 只表示 staging，不会遮蔽仍
active 的旧 READY build。Graph 复用 Graphiti node-distance/BFS candidate search、probe 和当前
serving Chunk hydration；缺任意一跳来源时整条路径拒绝。`classic` 保留 Graphiti 原生
搜索顺序；显式 `local_minilm_v1` 在水合后按路径最弱来源分重排。低分或未打分 Chunk 不会因此被
删除，模型分也不写入 `GRAPH_PATH` Evidence。候选 K 与 soft/hard source chunk 上限来自冻结的
`adaptive_graphiti_v3` snapshot；完整一至三跳路径是唯一打包原子，不拆断路径，path 可复用
普通检索/上一次 Graph 已发送 Chunk（重复结果合并 provenance），`new_evidence_count` 只计真正新增
Chunk；达到 soft 12 后下一条完整路径加入后不超过 hard 16 则整条接收，超过才停止。它不执行
第二次 vector/FTS seed，也没有服务端提交 guard：submit 永远不再隐式触发 Graph 调用。
Graph 的 route result 只允许 `admitted`、`no_evidence`、`not_ready`、`timeout`、
`unavailable`、`rejected` 等安全码，公开 trace 保留 `tool=search_graph_relations` 与
`graph_relations` lane，edge fact 永不进入 Agent tool result。Graph 单次内层 90 秒 timeout 与该
结果码可区分外层 ChatRun 取消。
manual Graph 的 hybrid 候选查询宽度按 `min(40, max(12, top_k * 2))` 计算；packing 按 path-whole
规则优先保留完整图路径，再用未重复的 hybrid Evidence 回填到 `top_k`。
Agent（v5）不再设模型轮次、检索次数、Graph 次数或证据条数上限；冻结 budget 只保留
`max_total_tokens`（默认 150k）一个基础设施保险丝。模型在同一轮可以发起多个互不依赖的
工具调用，服务端并发执行、按 `index_chunk_id` 去重合并进统一证据池，并给每个调用各自
返回 tool 结果。Agent 不以时间决定控制流；每轮按 response usage
累计 token，token 保险丝触发后进入软 wrap-up——只留 `submit_answer` 工具，由模型自行选择
answered/partial/refused；整体超时仍是硬资源错误。检索收敛按完整模型轮统计：一轮内任一
检索产生新 chunk 即归零，成功检索但零新增记一次，连续两轮无新增后关闭检索工具，保留
至多一次 `calculate` 机会后只允许提交；`calculate`、`list_documents` 不参与统计。
检索执行失败只作为该调用的错误结果返回，不取消同轮其他调用；连续两轮没有任何成功工具
执行或有效提交则按资源错误终止（`protocol_error`），不合成回答。系统 Prompt 只做身份、
不可信数据与引用纪律约束，不做问题分类或通道路由；工具各自描述自身能力。
`submit_answer` 与其他调用同轮出现时，提交有效即终止（同轮其余调用不执行），无效则其余
调用照常执行并给一次修复反馈。Agent 不比较或拒绝重复 Query 本身。
历史 `retrieval_calls` 与已删除的 `max_retrieval_calls` 实际按 Query 执行数计量并继续保留兼容；Trace
同时发布语义明确的 `retrieval_queries`、`retrieval_tool_calls`、`semantic_tool_calls`、
`keyword_tool_calls`、`chunk_context_calls`、`document_list_calls` 与 `graph_tool_calls`。
v4 起不再发布 `simple_tool_calls`。成功 Trace 另汇总 prompt/completion/total token、停止原因、
连续无新增次数、累计耗时、hard deadline 与 deadline remaining。Runner 为每次 attempt 维护纯
内存 checkpoint；外层 deadline 取消 Agent 时，把已经完成的安全计数、事件与 model calls 写入
该 attempt 的 timing ledger，不在主循环增加 I/O，也不覆盖后来重试成功的最终 Trace。
题面中的文件名不触发分类、硬 document scope 或全文预读，因此同一知识库中被引用的其他文档
仍可被检索。每个结果获得运行内稳定 EvidenceRef；首次命中向模型发送索引保存的完整 chunk；
旧轮次的工具结果在后续请求中压缩为「ref + 来源 + 截断摘录」占位，被压缩的 ref 经再次命中
或 `read_chunk_context` 可重新获得完整正文。普通检索未命中不能证明内容未出现，Agent
不支持以未命中构造缺失性结论。系统没有开放工具集、跨问题 planner、持久 step ledger 或
跨 run provenance。固定 Agent 另有一个纯函数 Decimal
`calculate` action：只接收 512 字符内的 `+ - * /` 表达式并返回结果，不绑定证据；
引用哪些知识库内容由最终回答的 claim 自行决定。

## 10. Chat 与回答

API 创建 ChatRun 时在短事务内冻结知识库/revision、检索 preset 的 version/strategy/`top_k`/
`rerank_mode`（auto 另含 router/augmentation 与 Graph Tool 参数）、原生 Agent 五键预算
（模型轮次、Graph 调用、token 总量、证据条数与检索调用数）、
不可变模型修订和最近已完成 Session turns，然后
返回 `202`；模型调用由 Worker 执行。公开请求没有 workflow 模式。

回答行为是单一的 evidence-only 逐 claim salvage/拒答路径。无效的 `answer_style`、
`insufficiency_policy` 及单值策略标签已从创建/更新请求、服务层和 Worker context 移除；
不再提供默认值解析、覆盖优先级或组合校验。新 KB/ChatRun 在保留的策略 JSON 列写入空对象，
不迁移、不改写历史值。历史 `answer_policy_defaults`/`effective_answer_policy` 只读返回原始
JSON，不作为当前执行配置，也不做旧枚举解析。旧客户端提交策略字段会得到通用 422 参数错误。

旧查询改写器不再参与执行：新 ChatRun 的 `contextualized_query` 为 NULL，不再复制原始问题、
构造改写状态、记录空的改写诊断或向 Worker 传递该快照。当前问题和有界 Session 历史仍照常
输入原生 Agent。历史改写 JSON 保留，API 只投影已存的展示字段，不要求旧版本、hash、调用
次数或规范序列化一致；缺少快照时 `query_context.status=original`，改写文本/来源为 null，
即使存在会话历史也不会再显示待改写。有效的历史模型调用账目保留原 attempt/sequence，
不因移除改写器丢失或重复累计；无效历史计数不进入新汇总。历史字段不参与当前 prompt。

当前 Chat 执行是普通异步 Tool-Calling loop：

```text
load_context
  -> active READY build capability read without any model call
  -> model chooses any combination of semantic_search / keyword_search /
       read_chunk_context / list_documents / search_graph_relations / calculate,
       with independent calls executed concurrently in the same round
  -> server executes tools and returns stable refs from one merged evidence pool
  -> adaptive mode: Graph visible from the first round, uncapped per run
  -> model calls submit_answer when ready (answered / partial / refused)
  -> token fuse switches to a soft submit-only wrap-up round
  -> two consecutive rounds without new evidence close retrieval
  -> two consecutive stalled rounds end the run with a resource error
  -> submission shape validation only; the model's outcome stands
  -> persist_result
```

ChatRun 是唯一持久执行状态；没有 graph checkpoint、Controller、Verifier、独立 Generator/
Repair 或逐步骤 ledger。成功终态原子保存有界 Agent Trace。迁移
`0010_drop_legacy_workflow` 已删除旧 workflow configuration/state 及其中的
ResearchResult/SearchTrace 诊断；核心 ChatRun、消息、答案、Citation、usage 和 timing 事实保留。
ChatRun 内部 trace 保存 claim salvage 的 rejected count 与内部 reason，供本地
诊断使用；公开 API/SSE 会剥离这些内部字段，只保留隐私审查过的工具、lane、安全枚举与计数。
共享 trace artifact key 属于 domain 契约，不由 `services` 反向导入 Agent 实现。
`ChatAnsweringState` 只保存真实 Evidence 可用引用、模型调用、视觉附件、确定性校验结果与渲染结果；
不保存重复的原始 submit 草稿，也不伪造旧 assessment/structure-validation 状态。
`RenderedCitation` 只保存显示顺序与已准入 `PromptEvidence` 的引用，不再重复复制、校验整套证据元数据。
原始检索 Evidence 与准入后的 PromptEvidence 仍分开：后者承载模型实际可用的视觉快照。
数据库与公开 Citation 快照字段不变。成功终态 timing 记录实际 outcome、引用、检索、
视觉和 query-rewrite 事实，不写空 validation 占位。

回答边界保持：

- 拒答只由模型通过 `submit_answer` 自主给出（v5 起没有任何确定性拒答兜底）；
  基础设施终止（超时、连续停滞）一律记录为资源错误，不合成回答或拒答。
- 提交通过形状校验后直接渲染、持久化，不再追加独立
  LLM Verifier 或 JSON verdict 重试。语义支持度及题面预设命题由生成模型结合证据判断，
  不把引用合法性检查等同于事实正确性保证；原有 evidence-only 与错误前提拒答提示保留。
  历史 Trace 中的 `verifier` 事件仍可只读展示，新运行不产生此类事件；历史 token 统计不改写。
- 文档、历史与图片都是 prompt 中的不可信数据，不能扩大权限或引用范围。
- `submit_answer` 只做形状校验（outcome/claims/unanswered 结构）；claim 引用的非法 ref
  从引用集中静默丢弃，不删除 claim、不改写 outcome、不阻塞回答；形状非法的提交获得
  修复反馈并计入停滞保险丝。新 ChatRun 不再有 `clarify` outcome；历史 clarify 数据保持可读。
- 事实 claim 只能引用本次已授权 Evidence；实际未加载的图片不能产生视觉引用。
- 同一主题上互不兼容的证据直接写成普通 claim 文本，并在 `evidence_refs` 中引用冲突双方；
  没有 `kind/conflict/type/adjudication` 分支、冲突领域对象或额外持久化字段。
- 公开冲突评测不再把模型自报的结构标签当作正确性证明。`surface_evidence_conflict` 要求
  回答命中 gold 文本且至少引用两个不同文档，并单独报告多文档覆盖；该确定性指标不声称
  已完成语义级冲突判定。`answer_without_false_conflict` 按普通回答和 gold 命中评分。
  两类用例缺少可用 gold 时标记为未评估（`policy_correct=null`），从总计、动作和题型的
  正确率分母中排除并单列计数；历史观测不重算、不改写。
- Graph 关系检索水合后的 text/table Chunk 同时保留 `graph_path` provenance 与对应的
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

答案内容/形状和引用的校验集中在 `submit_answer` 输入边界。内部 `AnswerClaim`、
`ValidatedAnswer` 不重复验证已经归一化的结果；渲染器负责引用去重、编号和结果展示，
不再由结果 DTO 重查其编号与 outcome 组合。最终内容非空/大小限制、视觉准入与实际 bytes
核对、持久化前结果完整性检查仍保留。

Provider 单次调用的 SDK timeout 与有限 retry 由一个逻辑预算统一计算：
`timeout * (max_retries + 1) + 60 * max_retries + 1` 秒；Adapter 的外层总预算覆盖整个
semaphore/retry 窗口。Worker 启动时拒绝不大于该预算的 Chat Agent deadline；本地默认 deadline
为 600 秒，ChatRun 的既有 attempt 上限不因 Agent 而增加。
Agent progress 是 content-safe、易失且不可重放的快照；断线后读取权威 ChatRun。

这些约束直接保护回答可信度和个人数据，继续保留；模型调用的细粒度 timing、wire 版本和每个
中间 hash 不应自动成为未来功能的强制模板。

## 11. 公开接口与界面

公开接口位于 `/api/v1`，主要分为：

- knowledge bases、documents、document-version creation、chunk inspection/exclusion 和 indexing jobs；
- model settings snapshot、provider/profile mutation、provider model catalog、validation 和
  workspace model selection；
- retrieval query、Graph 配置 `GET/PUT /knowledge-bases/{kb_id}/graph-config` 和授权
  `GET /index-assets/{asset_id}/content`；Graph 查询仍是明确的外层 `graph` 模式，路径仍只返回
  当前 serving 的原始 Chunk Evidence；Chat creation 另支持 Chat-only `auto` 模式；
- chat sessions、messages、runs 和 events；SSE 进度事件为 `agent.progress`。

上传、状态、分页、幂等和错误的精确契约以 OpenAPI、schema 和路由测试为准。未实现能力不
添加占位成功路由。索引任务状态对当前 PDF 额外公开 content-safe 的阶段、页段、OCR、表格、
累计耗时和子进程峰值 RSS；前端沿用现有卡片、颜色、间距和进度条展示这些字段，不把
会重叠的阶段耗时相加成虚假的精确百分比。

`apps/web-chat` 是本地唯一前端，提供知识库创建/删除、文档批量导入/更新/删除、索引进度、
chunk 预览/排除、检索 debug，以及知识库/Session 选择、统一 Native Agent 提问、
终态回答、有界工具轨迹和证据抽屉。管理页复用 Chat 既有视觉 token 与侧栏，不形成第二套 UI。
它只调用公开 API；诊断功能不等于生产管理控制面。Chat 与管理页的异步读写使用 generation、
作用域身份和请求序号守卫，旧 KB/session/document 的 success、error、finally 回调不得覆盖当前视图。

## 12. 安全、失败与恢复边界

即使是本地个人项目，也继续保留以下低成本、高价值约束：

- 默认绑定 loopback；provider key、DSN 和本地口令只进入未提交环境文件。
- 服务端生成文件路径和身份；公开请求不能选择主机路径或 workspace。
- 用户内容、provider body、图片 bytes/Data URL 和 secret 默认不写日志。
- 数据库事务短小，外部 I/O 在事务外；过期 Worker attempt 不能覆盖新终态。
- 文件删除和本地数据 reset 属于破坏性操作，必须明确目标并得到用户授权。
- source-file reservation 在外部文件操作前持久化为 pending；宽限期内等待，完整性/身份不可恢复时
  进入 failed 并排队 durable cleanup，数据库/未知故障不伪装成业务终态。model-secret orphan
  reconciliation 先在只读事务取得所有 provider revision 引用，再按宽限期和 root 内 lstat 结果清理。
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

Graphiti build 继续使用 immutable build 代际作为 retry 事实。每个 build 冻结 Schema Profile key/digest
和 extractor generation；未知 profile、digest mismatch 或 extractor 不兼容都 fail closed。Episode 在
Graphiti 写入和边界清理完成后写一个 Graph 侧完成标记；若进程在 PostgreSQL 映射提交前退出，重试直接
复用该确定性 Episode UUID 并补交映射，不重复调用 Provider。只有缺少完成标记的未提交 Episode 才先
删除并以同一身份重新摄入；已提交映射不会重复调用 Provider。显式 retry 只有在 revision、serving digest、Chat/Embedding profile、model、dimension
与 extractor 均未变化时原地恢复；输入变化或 force rebuild 才 supersede 旧 build 并建立新代际。
同一 build 的 work item 先取得带 token 的 lease，Worker heartbeat 续租，stale lease 才可恢复；不同
build 仍可并行。
Graphiti 的 bulk gather 不作为 Provider 并发边界；本地 LLM/Embedding adapter 在每次实际请求上共享
build-scoped semaphore。bulk 失败直接结束当前 work，由已有显式 retry 或评测退避恢复，不在仍可能有
上游请求收尾时立即把整批切换为串行重放。PostgreSQL Episode mapping 批次若有任一项失去 lease 或
输入资格，整个 mapping 事务回滚。
Episode 写入后、mapping 提交前，Graphiti runtime 在当前 build-scoped Falkor graph 中删除
`source.uuid = target.uuid` 的非法 `RELATES_TO` 自环；这是对 structured extraction prompt 的持久化边界
保护，不依赖 Provider 永远服从提示。ready probe 同时要求自环为零、每条边具有关系类型、每个
AliasSurface 至少连接一个 canonical 节点；清理失败或 probe 不完整会保留 failed build 而不发布为 READY。
失败持久码由 `preflight`、`episode_extraction` 或 `finalize` phase 加只基于异常类型链的短 fingerprint
组成，因此可在没有旧日志时聚合故障位置，同时不保存异常消息、Provider payload、Chunk 正文或实体名称。
OpenCode-compatible structured output 还可能返回完整 JSON Schema、字段级 schema fragment，或把合法
字段与 `title/type/properties/description` 元数据混在同一对象。Graphiti adapter 在 optional typed
attribute model 边界执行字段白名单和 Pydantic 验证，识别上述 schema echo 后有界重试，连续失败则
fail closed，绝不把 schema 对象写成 FalkorDB property。

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

个人正式 runtime 只有一个身份：primary checkout、owner-only `.env.local`、Compose project
`rag` 以及 manifest 中的四个 loopback 端口。linked worktree 只用于代码和测试，不能重建、
migrate 或切换个人 stack。`./start-local.sh` 先执行 content-safe doctor；它不猜测 project/端口、
不从容器提取 credential、也不生成或覆盖配置。通过 preflight 后，starter 才按顺序启动
PostgreSQL、幂等校准现有 admin/migration/runtime 角色、构建带 Git revision label 的应用镜像、
准备 source storage、执行 Alembic 并等待 API/Worker/frontend 健康。现有业务卷不会因启动被重置。
固定 Docling/reranker 资产通过可覆盖的 HTTPS Hugging Face 镜像下载，瞬时网络错误做有界重试并
复用跨构建 cache；已有本地应用镜像通过只读 build context 引导后续构建。两条路径的最终镜像都
按固定 revision、文件大小和 SHA-256 manifest fail closed，旧镜像不会绕过当前 verifier。
`cl100k_base` 也是应用镜像内的 content-addressed 资产：原始
`src/rag_kb/tokenizer/assets/cl100k_base.tiktoken` 与版本化 manifest 一起随源码进入镜像，固定
SHA-256 为 `223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`。共享 loader 只读
这两个包内文件，按 `tiktoken==0.13.0` 的固定 regex、特殊 token 映射和 BPE bytes 构造编码，不调用
network-aware resolver，也不使用 `TIKTOKEN_CACHE_DIR`。API 与 Worker composition root 在创建数据库
资源、进入 readiness 前执行 tokenizer preflight；`source-data` volume 不承载 tokenizer 资产或其缓存。
前端 `npm ci` 同样使用可覆盖的 HTTPS registry、原生有限 fetch retry/timeout 与独立 BuildKit cache，
依赖闭集仍由 `package-lock.json` 决定。若宿主已安装的前端依赖通过 `npm ls --all`，starter 可在
宿主执行 Vite build，并只把生成的 `dist` 作为只读 build context 交给最终 Python frontend 镜像；
这条路径不安装或更新依赖，Docker `npm ci` 仍是 clean checkout fallback。

旧双-env、worktree override 和迁移备份已经退役；它们不是 runtime input，重新出现时 doctor 只报告
content-safe stale warning。唯一有效配置是 primary checkout 的 0600 `.env.local`，模型 provider/profile
继续通过 Web Chat 和数据库维护，不存在环境 fallback。常用命令：

```bash
./start-local.sh
PYTHONPATH=src:. .venv/bin/python tools/local_runtime.py doctor
docker compose --env-file .env.local --project-name rag ps
docker compose --env-file .env.local --project-name rag logs --no-color api worker
PYTHONPATH=src:. .venv/bin/python tools/smoke_local.py
docker compose --env-file .env.local --project-name rag down
```

以上 Compose/start 命令属于个人 runtime 运维，不属于测试流程，不能从“测试”请求中推导。

测试使用 `unittest`，现有目录包括 `tests/basic`、`tests/unit`、`tests/contract` 和
`tests/integration`。验证按风险选择：

- 所有 Python、后端和 evaluator 测试都从 checkout 使用 `.venv/bin/python` 与 `PYTHONPATH=src:.`
  执行，不进入应用容器；前端只用宿主机 Node/npm 验证。
- 普通 unit、contract、backend 和 evaluator 测试不得 build/pull/tag 镜像，不得 create/recreate/restart/stop/remove
  容器，也不得下载镜像依赖；用户明确请求数据库 integration 时，仅允许通过下方的 disposable
  PostgreSQL runner 创建其自有临时容器。
- Python 行为变化至少运行 basic suite 和最接近变更的聚焦测试。
- 数据库迁移、并发或 repository 行为变化才运行相关数据库 integration。
- 数据库 integration 只允许使用与正式 `rag` 完全隔离、可丢弃的测试库；只接受
  `RAG_KB_TEST_MIGRATION_DSN` 与 `RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN`。前者用于 Alembic、
  migration 和 `TRUNCATE ... CASCADE`，后者必须是 `postgresql+asyncpg://` 的 SQLAlchemy
  runtime DSN。不得从 `.env.local` 推断，不得连接或清理个人数据库。
- 用户明确请求数据库 integration 时，`tools/run_database_tests.py` 可以启动一个唯一、只绑定
  `127.0.0.1`、使用临时存储的 PostgreSQL 容器，数据库名固定为 `rag_kb_test`，在子进程环境中
  注入上述两个 DSN，执行迁移和测试后通过 owner label 清理容器。它不加入正式 `rag` Compose
  project，也不复用正式 `rag_kb` 数据库。普通 unit/contract 测试不自动触发 Docker 生命周期。
- 直接运行 integration 测试时，调用方也可以提供已经运行且可丢弃的测试库；缺少 DSN 时测试模块
  必须立即报错并指向上述 runner，不能用 `skip` 静默跳过。runner 发现任何 skipped 测试也返回失败，
  因此“未执行”不能被记为通过。
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

显式提供隔离测试 DSN 后，数据库检查同样直接从宿主机运行：

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/integration/db -v
```

`tools/reset_local.py` 会永久删除本地业务数据。执行前只能先用以下命令检查精确卷目标；
未获用户对该精确 reset 的明确授权不得去掉 `--inspect-only`：

```bash
PYTHONPATH=src:. .venv/bin/python tools/reset_local.py \
  --project-name rag \
  --inspect-only \
  --confirm DESTROY_RAG_KB_LOCAL_DATA
```

## 14. 轻量维护协议

项目任务只通过 [`.agent/`](../.agent/) 维护；旧的 `docs/implementation-plans/` 与
`EXECUTION-TRACKER.md` 已归档并停用。`.agent/` 是交接便笺，不是审批系统：

- 多阶段、高风险或改变方向时才写一份 current `PLAN.md`；小型明确工作可直接实现。
- `TODO.md` 只保留仍需执行的动作，`TRACKER.md` 只保留短状态/阻塞/下一步，`LOG.md` 追加实际结果。
- 只有需要独立范围或验证的阶段才建短 subplan；不要把同一事实复制到所有状态文件和报告。
- routine work 可在完成时一次更新状态，不要求为开始工作制造计划或 authorization gate。

Git 是恢复事实。小型可回退变更可以留在当前分支；大型、风险、并行或需独立 review 的工作才使用
`codex/` 前缀短分支。只提交任务拥有的文件并做最小充分验证。
local branch/commit/fast-forward merge 是普通实现动作；remote mutation 和 history rewrite 仍需确认。

只有依赖安装、不可恢复的本地数据操作、数据保留选择不清楚的 schema 变化、产品/架构扩张和远程
Git 操作需要先问。只读检查、宿主机 `.venv` 测试、可逆编辑和 local Git 不重复请求授权；测试授权
不包含 build/restart、镜像下载或容器生命周期操作，一次 destructive 授权也不能扩展到未列出的目标。

PLAN 结束时记录实际结果，把 PLAN 和其 subplans 一起移入 `archive/plans/NN-MMDD-short/`，然后把
PLAN/TODO/TRACKER 重置为“无当前工作”。roadmap、review 和 archive 都不会自动变成任务。实现默认选择
模块更少、持久状态更少、运行分支更少的方案；稳定产品/进程/数据/API 边界变化时才同步本文。

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
