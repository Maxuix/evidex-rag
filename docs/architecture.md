# Local Knowledge Base & RAG — 当前架构与轻量化边界

| 字段 | 内容 |
| --- | --- |
| 文档状态 | 全项目唯一当前架构文档（描述事实，不是目标蓝图） |
| 最后核对 | 2026-09-06 |
| 核对基线 | 当前 `main`、实际代码、配置、Alembic 迁移 `0032`、实际配置与公开路由 |
| 适用对象 | 个人维护者、项目贡献者 |
| 部署边界 | 单机、单用户、本地使用；不是共享或生产服务 |
| 设计优先级 | 功能可用与个人可维护性优先于平台化、通用化和生产完备性 |

## 1. 文档职责与事实优先级

本文集中维护系统现在能做什么、主要代码在哪里、运行时如何协作、核心数据与安全契约、
本地运行方式，以及哪些边界不能误解。它是 `docs/` 下唯一的架构事实入口；不再用多个专题
架构文件拆分同一事实。设置常量、内部 DTO、完整表字段和算法实现仍由代码、迁移与测试负责。

发生冲突时按以下顺序判断：

1. 可执行代码、Alembic 迁移、锁文件和实际运行配置是最终事实。
2. 本文记录稳定的产品、进程、数据和主要调用链，应与第一项保持一致。
3. [`../archive/`](../archive/) 只保存历史快照，不能作为当前事实或自动恢复工作的指令。

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
- 单个 Worker 异步解析文档、切分、生成文本/可选多模态向量并建立索引。新建知识库可选择冻结的 Auto-QA 问句索引：每个有原文的 Chunk 最多生成 5 个问题，经独立原文核验后保存支持跨度；问句向量只补充候选，主词法行保持原文；问题不是证据、引用或 Graph Episode。
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
- Chat 的公开选择收敛为 `text | auto | graph` 三种模式，默认推荐 `auto`。Text 只开放文档
  分块的语义、关键词和邻域检索；Auto 在此基础上按 READY capability 开放一等
  `search_graph_relations` Graph Tool，由原生 Agent 自主选择；Graph 强制经 `retrieve_graph`
  先打包完整图路径，再用未重复的 hybrid 文档证据回填。
  Graph Tool 只在存在 active READY build 时可见且不设每个 ChatRun 的调用次数上限；Graph 单次
  90 秒（ChatRun 绝对 deadline 为 600 秒形成 `min(90, remaining)`），候选 K 冻结为 16，完整
  一至三跳路径按 soft 12 / hard 16 去重 source chunk 原子打包。
  不存在服务端 completeness guard。Agent 的冻结预算只限制累计 token；达到预算后切换到一次
  无工具纯文本收尾，不对单条证据正文做按 token 截断。Graph Tool
  只返回 source chunk；edge fact 不进入 prompt、Citation 或回答正文；未配置、未就绪、
  运行时不可用、超时、被拒绝和无新增证据都以安全结果码返回，取消与超时可区分。
- 用户可多选知识库，Session 保存选择，ChatRun 冻结各库索引、检索配置与图 build；同一个 Agent
  自主选择指定库或全部所选库，回答引用和原文预览显示各自库名。
- 持久 ChatSession / ChatRun、Session 短期上下文，以及动态提供当前可用工具的原生
  Tool-Calling Agent。常规工具为 `semantic_search`、`keyword_search`（hybrid
  进程且 lexical manifest 覆盖完整时暴露）、`read_chunk_context`、`list_documents`、
  `calculate`；Graph READY 时首轮同时暴露 `search_graph_relations`。模型停止调用工具后直接写
  带行内 EvidenceRef 的最终正文。所有运行共享证据约束回答、拒答与引用边界。
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
| 数据库 | PostgreSQL 18 + pgvector；当前 migration head 为 `0029_agent_v6_default` |
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
  解析 adapter。Text 与手动 Graph snapshot 保存 profile version、strategy、`top_k` 和
  `rerank_mode`；Graph 外层模式额外冻结 `graphiti_path_augmented_v3` 与 `graphiti_path_v3`
  augmentation，内部 seed 仍使用 hybrid。默认 Chat-only Auto 使用独立的
  `adaptive_graphiti_v3` snapshot，冻结 exact-vector Simple、一等 Graph Tool 参数
  （`graph_edge_limit=16`、`graph_source_chunk_target=12`、`graph_source_chunk_limit=16`、
  `graph_call_timeout_seconds=90`）与 router `native_agent_graph_tool_v1`。Text 可选择
  `exact_vector` 或有界的 `iterative_balanced`；这三个 Chat 模式不改变直连 Retrieval Debug API
  的 `exact_vector | hybrid | iterative_balanced` strategy 合同。
  retry 按当前进程配置解析候选数、阈值和融合权重。Chat 只有一个固定原生 Agent 路径。
- Text 的 `exact_vector` 使用单轮 exact vector，`iterative_balanced` 复用同一 exact vector
  底座并由原生 Agent 执行最多两轮原文锚定扩展；Auto 的语义通道仍使用 exact vector。keyword
  tool 与 Graph 回填所需的 hybrid FTS 由简单设置开关控制。
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
单键 token 预算（`max_total_tokens`）；`0029` 只把新行默认版本推进到
`native_tool_calling_agent_v6`。`0030` 为 `IndexRevision` 增加冻结的 Auto-QA 配置
（默认关闭）和最小 `index_chunk_question` 表；旧知识库无需回填。`0027`–`0030` 都不回填历史行；
v6 执行器读取历史 v3/v4/v5 配置时只取其中的 token 上限。
`0031` 增加 Auto-QA 原文支持跨度；`0032_multi_kb_scope` 增加库描述、`chat_session_kb`
选择关联、`chat_run_kb` 不可变范围和 Citation 来源快照。迁移把旧会话、Run 与引用回填为
单库来源，并在切换约束前验证完整性；即使原 chunk 已删除，也保留引用原文。删除某库只移除
会话选择关联，不级联删除整个会话或 Run 范围历史。旧单库列允许 NULL，退出多库来源判定。
`0032` 不提供会丢失多库历史的 downgrade；部署前按本地保留数据流程应用迁移，readiness 检查该 head。
P2 的实际数据核查确认 active/retired revision 指针仍承担当前与软删除恢复，两个 READY Graph build
均为 active，PDF 分段任务真实使用 continuation；因此 revision/build identity、完整性 manifest、Graph
lease 与 PDF checkpoint 都保留。未使用的 Enterprise Graph profile 只作为需单独授权的完整产品删除
候选，不在配置收敛中隐式移除。相关保留边界已经固化在代码、迁移和当前运行配置中。
除此之外不承诺任意历史版本兼容。主要持久事实为：

| 范围 | 主要实体 |
| --- | --- |
| 知识库与文档 | `Workspace`、`KnowledgeBase`、`Document`、`DocumentVersion` |
| 索引 | `EmbeddingSpace`、`IndexRevision`、`IndexedDocumentVersion`、`IndexingJob` |
| 检索数据 | `IndexChunk`、词法派生、可选 `index_chunk_question`、资产/关系、可变维度 `VectorRecord`、Graph 配置、Graphiti build 与 Episode→Chunk 映射 |
| Chat | `ChatSession`、`chat_session_kb`、`ChatMessage`、`ChatRun`、`chat_run_kb`、`Citation` |
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
  -> bounded PDF page/evidence probe in the owned killable child (cached for resume)
  -> current PDF profiles: deterministic page segments in a killable child process
  -> persist content-safe progress/checkpoint and yield between segments
  -> globally reassemble and validate one DoclingDocument
  -> legacy/non-PDF profiles: one isolated Docling conversion
  -> structural or semantic chunks + optional visual assets
  -> role-bound text and optional multimodal embeddings
  -> optional Auto-QA: 0–5 questions, independent source verification, then accepted question embeddings
  -> persist derived rows and source-only lexical rows, then mark target ready/serving
```

当前支持 TXT、Markdown、HTML、CSV、PDF、DOCX、PPTX 和 XLSX。Markdown 可使用 `.mdz` Bundle
携带本地图片，也可使用受限 data URI；HTTP(S) 与 protocol-relative 图片在准入阶段确定性拒绝，
摄取过程不解析 DNS 或连接外部网址。本地 Docling 工件、格式/归档/像素/页数等
资源限制和可终止的解析子进程用于保护本地数据与 Worker。文档索引不再按墙钟耗时失败；
当前生产装配直接构造默认 `ParserLimits()`：PDF 使用 CPU 单线程、OCR/Layout/Table
batch 1 和每段初始 20 页，这些值目前不是用户运行时配置项。系统在段间持久化进度、
让出索引 lane。保持 Docling 区域 OCR 和 TableFormer Accurate，不用“存在文本层”关闭整份
OCR，也不提高 conversion/indexing 并发，以守住 6 GiB Worker 上限和混合页面、多模态资产质量。

当前新索引使用文本 `docling_text_local_v4` / 多模态 `docling_multimodal_local_v5`；旧 profile
仍可读取，升级不自动重建既有索引。XLSX 在加载 openpyxl/Docling 前按 worksheet relationships
检查真实单元格跨度、合并区域和累计 XML/单元格预算。PDF 页数与视觉页探测也在解析子进程
执行，不在 Worker 线程中提取 PDF 文本；图像覆盖率判定可保留带页码或嵌套 Form 的扫描页，
不会因少量文字层而直接丢弃整页证据。段间复用已绑定 source/profile 的探测结果。
操作符预算按整份文档累计（包含重复 Form 调用），默认 500 万；原 20 万上限会拒绝
普通长篇年报。保留 20 MiB 累计解压内容、500 页和深度限制；相关预算以代码中的
`ParserLimits` / `AdmissionLimits` 和可执行测试为准。

checkpoint 在下一段落盘及最终合并前检查累计预算，合并逐段释放输入；图像在解码/裁切前
校验总像素、尺寸和图像头。PDF 表格可通过已有页图与 bbox 裁切为 table image。
精确资源上限见 `ParserLimits` / `AdmissionLimits`；改变这些上限不能替代内容质量验证。
修复与保真/内存约束以代码和可执行测试为准。

结构切分和语义切分都直接消费一次 Docling conversion 结果。新建索引使用 structural v5 或
semantic v5；旧 structural v4、semantic v3/v4 的完整 profile 仍可执行，旧索引不会原地改写。
semantic v5 用不重叠的原文片段保留标点、数字、URL、代码缩进及分隔符，补齐末尾标题；
分析视图和计划 hash 包含来源连接规则。仅对超过上限的硬边界区域生成分析 embedding，
最终检索 embedding 仍生成。
语义 V5 的表格 embedding 使用 `compact_markdown_cells_v1`：仅去掉 Markdown 列对齐空格、
收敛分隔横线，保留单元格值、列顺序、对齐标记和证据原文，减少触发 2,048 UTF-8 字节二次窗口的机会。

两套 v5 共用按行且重复表头的表格切分；结构切分将超长正文的
标题附在有预算的正文片段上。文本解析模式保留作者图注文字，多模态模式沿用资产关系。
semantic 继续保留 section、page、table、非正文 block 和空行短标题 `record` 边界；只按来源
结构识别内部记录，不按业务实体或评测关系识别。Graph v2 可建立在 semantic v4/v5 上，
仍不允许直接使用 semantic v3。切分上限及检索默认值未改变，已有知识库升级需另行创建
使用新 profile 的索引版本并完成重建/切换。
多模态路径保存受限的 page、
picture 或 table image，并把文本与视觉表示投影到现有 Evidence/asset 关系。dual 模式分别
使用文本和跨模态 space；unified 模式让文本、查询和图片复用同一已确认的多模态 profile 与
space。精确 profile
名称、token/图片预算、hash 和持久字段由 registry、settings、迁移和测试负责，不在总览重复。
Auto-QA 属于 IndexRevision 的不可变索引表示，创建知识库时选择并冻结 Chat Profile Revision；
默认关闭。新配置冻结 `generation_policy=grounded_v2`：每个 Chunk 保存明确的已处理数量
（0–5），只有经独立核验、具备原文 SHA-256 与精确支持跨度的问题才能持久化及向量化。
未处理、支持失效或向量不完整时 candidate 不能 READY。迁移 `0031` 仅增加可空字段；
旧问句保持未验证状态，不因升级自动重生成，也不默认进入补充召回；兼容复评可显式使用
内部 `allow_unverified_auto_qa` 开关，公开检索请求不开放该开关。生成问题永不进入 Prompt Evidence、Citation 或 Graph Episode。
已有知识库不能原地开启；后续如需启用，要另做整库重建与 revision 切换。

已有 Auto-QA 词法行可用 `tools/rebuild_auto_qa_lexical.py` 修复：提供明确的 `--env-file`、
`--kb-id` 和 `--revision-id`，默认只读预览，确认目标后加 `--apply`。工具仅重算当前分析器的
已有词法行，先核对旧 manifest，再在单个事务中更新行及 manifest；不生成 QA、不调用
embedding、不改问句或源文档。缺失 manifest 或无原文表示时明确失败，不制造完整性记录。
Auto-QA 原文支持字段由 `0031_auto_qa_grounding` 提供；当前部署 head 与 readiness 见 §7。


删除先让数据库事实不可服务，再重试物理文件清理。Maintenance 只清理退休派生数据；不会
自动删除 active/candidate 数据。

## 9. 检索

默认路径是当前 serving revision 上的精确 pgvector cosine 检索。显式开启 hybrid 后，系统
并行执行 dense 与 PostgreSQL FTS，并用确定性 RRF 融合。text-only 生成一个文本 query
vector，dual 模式分别生成文本与跨模态 query vector，unified 模式只生成一个 query vector
并复用于文本/视觉 lane；每条 SQL 仍强制限定 role 绑定的 space 与维度。`semantic_search`
在同一 SQL 快照中独立保留原文候选（通常 40）和最多 20 个问句命中的不同 Chunk，再合并去重。
原文候选与原文 cosine 分数不被问句覆盖；问题距离只作召回及 debug。`classic`/`none`
忽略问句独有项，`local_minilm_v1` 对合并候选的原文评分。开启 Auto-QA 不改变既有重排默认值；
本地模型精排需显式选择。`keyword_search` 仍读取每 Chunk 一行的 FTS，新索引不再追加问句词项；
已有词法行需要单独按原文重建并更新 manifest，代码升级不会自动清除历史词项。检索结果统一投影为
`EvidencePack`，再由回答链路进行阈值、关系和视觉准入。问句命中仍水合原始 Chunk 正文；
`matched_question` 只出现在检索 debug。Agent 工具集合和 `tool_choice=auto` 不变，不新增
`auto_qa_search`。

Text Chat 的 `iterative_balanced` 是单一 Agent 路径上的检索策略：首轮使用原始问题，后续
最多两轮只允许引用已发出证据中的原文锚点，并在合并时保留首轮前三个结果、交错加入后续视图。
每轮仍复用 exact vector、同一精排配置和现有 EvidencePack/Citation 约束；锚点校验失败的查询
会被拒绝，轮数达到上限后直接进入回答阶段。Classic 仍是默认策略。

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

纯文本空间的 exact + `classic` 原文检索增加有界表格补全：保留原核心候选（Top-10 请求为
40 个），用同一查询向量再读前 100 个原文，仅把达到原 cosine 门的 text/table 项作为结构
锚点。只补同一 indexed document version、相邻 ordinal、相同来源表面、页差不超过 1 的
table；两侧都有章节标题且不同则拒绝，不递归扩展。补充项最多 200 个，使用自身原文向量
距离和独立引用；结构关系是额外准入依据，不借用锚点的 cosine 分。查询保留 workspace/KB、
active revision、ready/serving、源可用、未删除及非 excluded 约束，跨读前缀或 revision 漂移
明确失败。Classic 词法统计固定在原核心候选；原排序前 `ceil(top_k/2)` 个位置保持，余下位置
再按原文分数竞争。此保护保证领先位置，不能保证任意新数据的整个 Top-10 都无退步。
补充数量写入 `source_context_candidate_count` 调试字段。含问句命中的候选、Hybrid、MiniLM、
多模态和 Graph 路径不使用这条策略。无需迁移或重建旧索引；报告 80 记录冻结回归集的
收益与局限，尚未联合评测新 V5 切分重建后的索引。

`local_minilm_v1` 使用构建时固定、运行时离线的多语言 MiniLM ARM64 INT8 ONNX
工件，对全部已准入的 text/table 候选分批评分（每批最多 20）；不再先按 classic 截至 20。
合并候选上限 320，问句命中的原文候选允许送模型核验而不受原文 cosine 门预先排除。模型 tokenizer 将 query 截至 96
tokens；文件名与同一 serving 文档开头提供独立的 32-token 上下文，章节层级另保留 32 tokens。
上下文在评分前按候选 indexed document version 做一次有界读取，绑定 workspace/KB/active revision、
ready/serving、可用且未删除的源版本，忽略 excluded Chunk；每文档只读首个可用 text/table Chunk
前 2,048 字符，再由投影取第一段。它仅用于模型输入，不改原文引用、词法行或向量，无需迁移/重建索引。
超过剩余 512-token pair 预算的正文按段落或表格行临时窗口化
（64-token overlap、每批总窗口最多 80，超限时拆分文档批次），以窗口最大 logit 聚合回原 Chunk。模型分数不覆盖原
Evidence score/准入事实，窗口也不持久化。表前标题不再妨碍表头识别，表名及可识别的多行年份/
单位表头随窗口重复；表头过宽或一个 Chunk 含多个独立表格时保留全文普通窗口覆盖，不截掉
表头后继续假装列语义完整。纯视觉候选不送入模型。Native Agent 与
Retrieval Debug 都可使用该冻结模式；模型不可用时明确失败且不静默回退。
Text 与 Auto 允许三种精排；Graph 只允许 `classic | local_minilm_v1`，不开放 `none`。所有
`local_minilm_v1` Chat 请求的 `top_k` 都不得超过 20。

Native Agent 通过按召回通道划分的工具选择检索方式，每轮恰好一个工具调用。
`semantic_search` 在冻结 workspace/knowledge-base/index revision、检索策略与 top-k 内做
exact dense（含跨模态 lane）加冻结 rerank；Graph ChatRun 仍经该工具强制分派到
`retrieve_graph`，且不暴露可绕过图路径的 `keyword_search`。Text/Auto 的 `keyword_search`
只在进程 hybrid 开启且 lexical manifest 覆盖完整时暴露，
执行 FTS 后按 `lexical_rank` 截取，证据 `score_kind=LEXICAL`、`score=1.0/lexical_rank`，
不伪装 cosine 分；manifest/版本类 `INDEX_REVISION_INCOMPATIBLE` 软失败并摘除该工具，
`CHAT_REVISION_MISMATCH` 仍 fatal。`read_chunk_context` 锚定已签发的 text/table EvidenceRef，
固定 ±1 邻域，邻域证据 `matched_representations` 按 modality 派生为 `text`/`table_text`，
准入只做池去重与证据预算，不走搜索语义下恒为 False 的 `eligibility.usable()`。
`list_documents` 返回 serving 文档元数据（可选大纲），不进证据池、不可引用，每次计 1 次
retrieval。结果在证据池中按 chunk 去重。`search_closed` 时四个证据获取工具与 Graph 一并移除，
只保留至多一次 `calculate`，随后进入无工具文本终态轮。
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
第二次 vector/FTS seed，也没有服务端终态 guard：最终文本永远不再隐式触发 Graph 调用。
Graph 的 route result 只允许 `admitted`、`no_evidence`、`not_ready`、`timeout`、
`unavailable`、`rejected` 等安全码，公开 trace 保留 `tool=search_graph_relations` 与
`graph_relations` lane，edge fact 永不进入 Agent tool result。Graph 单次内层 90 秒 timeout 与该
结果码可区分外层 ChatRun 取消。
manual Graph 的 hybrid 候选查询宽度按 `min(40, max(12, top_k * 2))` 计算；packing 按 path-whole
规则优先保留完整图路径，再用未重复的 hybrid Evidence 回填到 `top_k`。
Agent（v6）不再设模型轮次、检索次数、Graph 次数或证据条数上限；冻结 budget 只保留
`max_total_tokens`（默认 400k）一个基础设施保险丝。模型在同一轮可以发起多个互不依赖的
工具调用，服务端并发执行、按 `index_chunk_id` 去重合并进统一证据池，并给每个调用各自
返回 tool 结果。Agent 不以时间决定控制流；每轮按 response usage
累计 token，token 保险丝触发后进入软 wrap-up——不再提供工具并要求模型输出纯文本终态；
整体超时仍是硬资源错误。检索收敛按完整模型轮统计：一轮内任一
检索产生新 chunk 即归零，成功检索但零新增记一次，连续两轮无新增后关闭检索工具，保留
至多一次 `calculate` 机会，之后同样不再提供工具；`calculate`、`list_documents` 不参与统计。
检索执行失败只作为该调用的错误结果返回，不取消同轮其他调用；连续两轮没有任何成功工具
执行时继续累计停滞，连续三轮则按资源错误终止（`protocol_error`），不合成回答。系统 Prompt 只做身份、
不可信数据与引用纪律约束，不做问题分类或通道路由；工具各自描述自身能力。
正常轮次使用自动工具选择；模型某轮没有工具调用时，该轮正文就是终态。Agent 不比较或拒绝
重复 Query 本身。
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
引用哪些知识库内容由最终正文中的行内 EvidenceRef 自行决定。

## 10. Chat 与回答

API 创建 ChatRun 时在短事务内冻结具体知识库集合及每库名称、描述、index revision、可用状态、
Graph build，以及三值检索 mode 对应 preset 的
version/strategy/`top_k`/`rerank_mode`（Auto 另含 router/augmentation 与 Graph Tool 参数）、
原生 Agent token budget、
不可变模型修订和最近已完成 Session turns，然后
返回 `202`；模型调用由 Worker 执行。公开请求没有 workflow 模式。
改变 Session 选择只影响下一轮；不可用库仍留在快照内。多库 Auto/Text 分别采用各库冻结的
Top-K、rerank 与索引空间，手动 Graph 采用经校验的本次 Graph 参数；不直接比较跨库原始分数。

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
  -> bounded untrusted KB directory and per-KB capability read without a model call
  -> model chooses any combination of semantic_search / keyword_search /
       read_chunk_context / list_documents / search_graph_relations / calculate,
       with independent calls executed concurrently in the same round
  -> resolve explicit KB or all_selected; bounded fan-out with per-KB evidence packs
  -> return source-labelled groups with globally stable refs
  -> adaptive mode: Graph visible from the first round, uncapped per run
  -> model stops calling tools and writes a plain-text final with inline [ev_N]
  -> token fuse switches to a tool-free soft wrap-up round
  -> close retrieval only after every selected KB has stalled for two rounds
  -> three consecutive stalled rounds end the run with a resource error
  -> resolve inline refs, renumber display citations, infer outcome
  -> persist_result
```

ChatRun 是唯一持久执行状态；没有 graph checkpoint、Controller、Verifier、独立 Generator/
Repair 或逐步骤 ledger。成功终态原子保存有界 Agent Trace。迁移
`0010_drop_legacy_workflow` 已删除旧 workflow configuration/state 及其中的
ResearchResult/SearchTrace 诊断；核心 ChatRun、消息、答案、Citation、usage 和 timing 事实保留。
ChatRun 内部旧 trace 保留有界工具事件和终态引用解析计数；新增的 `chat_activity_v1`
观测快照记录实际执行边界，公开经过校验的工具输入、来源元数据和结果计数，不保存模型推理
或原始 provider 输出。它不参与执行控制，也不是逐步骤事务日志。
共享 trace artifact key 属于 domain 契约，不由 `services` 反向导入 Agent 实现。
`ChatAnsweringState` 只保存真实 Evidence 可用引用、模型调用、视觉附件、确定性校验结果与渲染结果；
不保存重复的原始模型草稿，也不伪造旧 assessment/structure-validation 状态。
`RenderedCitation` 只保存显示顺序与已准入 `PromptEvidence` 的引用，不再重复复制、校验整套证据元数据。
原始检索 Evidence 与准入后的 PromptEvidence 仍分开：后者承载模型实际可用的视觉快照。
数据库与公开 Citation 新增 knowledge_base_id、名称及 index_revision_id 来源快照。
持久化前逐条验证来源属于冻结集合，并核对 chunk 的真实 workspace/KB/revision/document/version。
成功终态 timing 记录实际 outcome、引用、检索、
视觉和 query-rewrite 事实，不写空 validation 占位。

所有检索工具（`calculate` 除外）必填 `knowledge_base_id`：具体 UUID 或 `all_selected`。
全范围严格展开本轮所选集合，每个 query 在每个目标库独立执行，整个 Run 共用并发上限 4。
指定未选库、缺失或无效范围均拒绝；部分库失败保留其余结果，取消会取消并等待子任务结束。
底层只支持 active revision，执行前后核对冻结 index/build；变更返回明确版本错误，不读取替代版本。
Graph 全范围只组合各库局部图原文，不产生跨库图路径或实体映射。

目录包含不可信名称、描述、能力及每库最多 5 个标题；初始目录条目预算 12,000 字符，超出时
明确给出 `catalog_offset` 后续页。`list_documents` 的单库文档游标独立分页，目录不是引用证据。
独立子问题可同轮并发，依赖新实体/版本的补查留到后续轮次；完整列举不能把 Top-K 当作全集。

每库保留独立 `EvidencePack`，全局 ref 按真实 KB/revision/chunk 绑定来源。每轮各查询/库的完整
chunk 或完整图路径组轮转准入，共享 96,000 字符展示预算；装不下的组明确报告省略数，
不能把残缺路径显示成完整路径。图片与邻域读取按该 ref 的真实库授权，视觉预算仍在整个 Run 累计。
局部无新增证据按库逐轮计数，不因一个库重复空查而剥夺其他库的首次查询机会。
Trace 的有界 `scope_calls` 记录每个工具/query/库的状态及检索、合格、准入、展示、新内容和省略数；
它是执行观测，不证明语义完整性，也没有增加另一个生成或核验流水线。

回答边界保持：

- 模型用不带 EvidenceRef 的正文解释知识库为何无法作答；基础设施以“至少一个成功解析的引用”
  推断 `answered`，零成功引用推断 `refused`，v6 不产生新的 `partial`。基础设施终止（超时、
  连续停滞）一律记录为资源错误，不合成回答或拒答。
- 纯文本终态完成引用解析后直接渲染、持久化，不再追加独立
  LLM Verifier 或 JSON verdict 重试。语义支持度及题面预设命题由生成模型结合证据判断，
  不把引用合法性检查等同于事实正确性保证；原有 evidence-only 与错误前提拒答提示保留。
  历史 Trace 中的 `verifier` 事件仍可只读展示，新运行不产生此类事件；历史 token 统计不改写。
- 文档、历史与图片都是 prompt 中的不可信数据，不能扩大权限或引用范围。
- 正文中的 `[ev_N]`（兼容全角/半角括号及组内常用分隔符）按首次出现解析并重编号为展示
  `[1][2]`；未知 ref 从正文引用标记和引用集中静默丢弃，不阻塞终态。新 ChatRun 不再有
  `partial` 或 `clarify` outcome；历史值保持可读。
- 事实 claim 只能引用本次已授权 Evidence；实际未加载的图片不能产生视觉引用。
- 同一主题上互不兼容的证据直接写成普通正文，并在相应陈述后行内引用冲突双方；
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

答案内容和引用的归一化集中在纯文本终态边界。内部 `ValidatedAnswer` 在 v6 中保存一个全文
claim；渲染器负责引用合法性、去重、编号、outcome 推断和结果展示。最终内容非空/大小限制、
视觉准入与实际 bytes 核对、持久化前结果完整性检查仍保留。

Provider 单次调用的 SDK timeout 与有限 retry 由一个逻辑预算统一计算：
`timeout * (max_retries + 1) + 60 * max_retries + 1` 秒；Adapter 的外层总预算覆盖整个
semaphore/retry 窗口。Worker 启动时拒绝不大于该预算的 Chat Agent deadline；本地默认 deadline
为 600 秒，ChatRun 的既有 attempt 上限不因 Agent 而增加。
每次 attempt 使用一个内存 `ChatActivityRecorder` 观察模型轮次、工具调用与实际系统操作。
工具在并发派发前获得服务端 ordinal/step_id；各自进入执行、返回/合并和结束状态，合并后的
EvidenceRef 与新证据数沿用原 Agent 的证据顺序。输入只在对应参数校验通过后记录；计算
表达式还必须通过确定性求值。来源只包含库/索引/文档/版本/chunk 身份、标题、位置及内部 ref，
不含检索正文、模型正文、凭据或任意诊断对象。每步可带分库查询、状态和计数；
图谱按库记录路径与跳数，基础设施不可用显示失败。

同一 recorder 产生易失 `agent.activity` 事件与终态快照。`chat_activity_v1` 使用独立版本，
`run_id + attempt + step_id + seq` 标识更新；序号在进入有界队列前分配，前端可识别丢失。
PG NOTIFY 预览最多 4000 UTF-8 字节，超限缩短输入/来源/分库详情并标记 `details_truncated`，不阻塞
回答。每个终态快照最多 1024 步、1 MiB：先裁剪较早详情，再移除较早步骤，显式记录省略数。
成功快照进入既有 `agent_trace.activity`，失败、超时与协作式 Worker 停止的部分快照进入
原 attempt timing ledger；重新尝试不覆盖此前记录，无新表或迁移。强杀进程不能保留内存
中尚未提交的观察。`persist_result` 只有在提交后的 ChatRun 读取投影中才确认为成功，不伪造
落库耗时。

公开 ChatRun 的 `activities` 按 attempt 投影并校验快照，旧 trace 与 timing 不重复暴露内部
activity。无效/未知记录通过 `activity_unavailable` 提示，不影响回答读取；
`live_progress_available` 区分服务端未开启进度。旧 `agent.progress` 保留兼容。
实时轨迹不支持持久重放；刷新运行中的页面可能缺少之前的调用，断线保留已收记录、有限
重连三次并独立轮询 ChatRun，完成后以保存快照校准。终态优先于迟到事件或轮询。

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
- chat sessions、messages、runs 和 events；创建 Session/Run 接受 `knowledge_base_ids`，
  旧单库输入在入口归一化，`PATCH /chat/sessions/{id}/scope` 更新下一轮选择。Run 返回独立
  `knowledge_bases` 冻结范围；SSE 逐调用事件为 `agent.activity`，兼容阶段快照 `agent.progress`。

上传、状态、分页、幂等和错误的精确契约以 OpenAPI、schema 和路由测试为准。未实现能力不
添加占位成功路由。索引任务状态对当前 PDF 额外公开 content-safe 的阶段、页段、OCR、表格、
累计耗时和子进程峰值 RSS；前端沿用现有卡片、颜色、间距和进度条展示这些字段，不把
会重叠的阶段耗时相加成虚假的精确百分比。

`apps/web-chat` 是本地唯一前端，提供知识库创建/删除、文档批量导入/更新/删除、索引进度、
chunk 预览/排除、检索 debug，以及库描述编辑、知识库/Session 选择、统一 Native Agent 提问、
终态回答、逐调用时间线和证据抽屉。搜索范围入口位于输入框底部工具栏，默认只展示库名和
数量，点击打开可按名称/描述过滤的浮层；长列表在浮层内滚动，不撑高输入框。支持范围
多选、跨列表分页的全选和清空；全选保存
当时具体 ID，空选择不提交请求。切换会话恢复选择，当前 Run 保留自己的冻结范围。
时间线按真实模型轮次展示并行工具、输入、来源、
结果与耗时；运行中默认展开并跟随最新步骤，历史默认折叠，用户向上阅读时暂停跟随。
重复调用与不同 attempt 保留独立记录。旧 trace 只投影实际已存事件，缺少输入/轮次/耗时
直接说明，不能编造固定阶段或独立核验过程。最终引用仍使用 Citation 抽屉，普通检索片段
通过既有文档/chunk 接口读取，按各来源所属库的 Run 快照检查知识库、文档版本和 index revision，不用新版本替代
历史来源；文本以纯文本渲染。管理页复用 Chat 既有视觉 token 与侧栏，不形成第二套 UI。
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
两条前端镜像构建路径都会校准静态文件的读取和目录遍历权限，再切换到非 root 用户，避免
宿主 `umask` 造成部署后 404。启动和 `/health` 检查首页文件可读且非空；本地 smoke 另外请求
实际首页及其 JS/CSS 资源，确认应用入口、内容类型和资源可用性。

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

本文只保留最终架构事实。一次性评审、路线图、测试报告和文档站源码不进入当前公开 `docs/`；
测试入口以代码、锁文件和实际运行配置为准。

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

项目变更以 Git 历史、可执行测试和本文为准；仓库不保留临时任务状态、交接便笺或本地助手配置。
小型明确工作可以直接实现，多阶段或高风险工作应在提交前形成简短的可审阅记录。
只提交任务拥有的文件并执行最小充分验证；稳定产品、进程、数据或 API 边界变化时同步本文。

实现默认选择模块更少、持久状态更少、运行分支更少的方案。历史计划、评审和测试证据只作为背景
材料，不会自动转化为当前任务。

## 15. 文档治理

当前公开 `docs/` 只允许以下结构：

```text
docs/
└── architecture.md
```

- `architecture.md` 是全项目唯一架构文档。稳定产品、进程、主要数据、公开 API、模块或运行
  边界变化时，必须在同一任务中主动更新相关章节。
- 过程性评审、路线图、测试报告和文档站源码不进入当前公开 `docs/`。
- [`archive/docs-20260819/`](../archive/docs-20260819/) 保存旧架构专题、旧任务系统、旧 review、
  release/baseline 与其他历史资料；`archive/plans/NN-MMDD-short/` 保存历史计划与其完整子计划。
  `archive/` 是独立的历史归档目录，本规则不自动改写其中内容；已有归档只读且不能覆盖 Git 或本文。

除固定的 `architecture.md` 外，当前公开 `docs/` 不新增其他文档类型。
