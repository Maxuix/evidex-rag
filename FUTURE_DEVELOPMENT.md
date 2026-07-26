# FUTURE_DEVELOPMENT — 未来发展方向分析

> 生成时间:2026-07-26
> 生成方式:基于当前工作树(分支 `feat/docling-migration`,HEAD `29f7d4e`)的全量代码、
> 配置与文档审查。配套问题清单见 [PROJECT_REVIEW.md](PROJECT_REVIEW.md)。
> 本文是发散性分析与建议,不是已登记的实施计划;任何条目落地前仍须按
> [docs/implementation-plans/README.md](docs/implementation-plans/README.md) 建立正式计划。

## 0. 现状一句话定位

当前系统是一个**工程纪律极强的本地单租户证据约束多模态 RAG 原型**:
八种文档格式经原生 Docling 单次解析,双嵌入空间(1024 维文本 + 768 维图文)精确
pgvector 召回,确定性证据准入与视觉预算,固定 LangGraph 链路,PostgreSQL 单一事实源,
全链路 fail-closed。它刻意不做认证、多租户、HA、流式输出与近似索引
(见架构文档第 2 节)。这个"刻意收窄"的边界既是它最大的资产(几乎没有半成品),
也是下一阶段所有演进的起点。

## 1. 产品方向

### 1.1 从"本地原型"到"团队级知识库"(最自然的下一步)

现在的每一条 repository 查询都显式携带 `workspace_id`(`src/rag_kb/repositories/`),
身份是可替换的 `AuthProvider` 协议(`src/rag_kb/auth/provider.py`),这意味着
多租户的**数据边界已经预铺**,缺的是身份与授权层:

- **阶段 A(单团队共享)**:OIDC/JWT 登录 + 单 workspace,先解决"谁在用、谁上传了什么"
  的审计问题;`ChatRun`/`SourceChange` 已有完整事实账本,补 principal 归属即可。
- **阶段 B(多 workspace)**:workspace 生命周期 API、按 workspace 的配额与计费统计
  (provider usage 已持久化在 ChatRun 快照中,聚合即可得到成本报表)。
- **阶段 C(文档级 ACL)**:检索层已强制 revision/version/删除状态过滤,
  ACL 可以作为额外的检索过滤谓词进入 `retrieval/service.py` 的查询计划,
  证据准入不变——这是此架构做 ACL 比一般 RAG 容易的地方。

### 1.2 知识运营闭环(利用已埋下的评测底座)

数据库中已经存在 `EvalDataset`/`EvalCase`/`EvalRun`/`EvalResult` 四张表
(架构文档第 8 节)和 `tools/evaluate_multimodal_real.py` 真实评测工具,
但没有公开评测 API。这是**已付成本、未收回报**的资产:

- 把评测从一次性工具升级为常驻能力:上传标准问答集 → 定期跑
  Recall@K/MRR/附图精度 → 质量趋势仪表盘(诊断前端 `apps/web-test` 已有承载 UI 的位置)。
- 用户反馈回路:Chat 前端加"有用/无用 + 原因"轻量反馈,写入 EvalResult,
  低分问题自动进入评测集——形成"真实使用 → 评测集 → 回归门禁"的飞轮。
- 拒答分析:系统确定性拒答的每一次都有诊断快照,聚合"拒答原因分布"
  就是知识库覆盖缺口地图,直接指导补文档。

### 1.3 内容接入生态

目前只支持浏览器逐个上传(10 MiB 上限)。企业知识库的真实瓶颈是"内容进得来":

- 批量导入 CLI/目录监听(本地优先,符合当前部署边界);
- 连接器框架:Confluence/SharePoint/S3/邮件归档,每个连接器产出与现有上传相同的
  `DocumentVersion + SourceChange` 事实,索引链路完全复用;
- 增量同步语义:`SourceChange` 单调账本天然支持"源变更 → 新版本 → 候选提升",
  连接器只需要做变更检测。

### 1.4 场景化与差异化

这套系统的差异化不是"又一个 RAG",而是**可审计的证据链**:
成员级视觉 citation、最终 LLM 上下文快照(`0011` 迁移)、确定性准入、
冻结的 policy。适合往合规敏感场景走:制度问答、审计支持、医药/法务文档助手——
这些场景恰恰要求"答案的每个 claim 都能指回原文/原图,且能回放当时模型看到了什么"。

## 2. 技术创新机会

### 2.1 检索质量(收益最直接)

| 方向 | 说明 | 前置条件 |
| --- | --- | --- |
| HNSW/量化索引 | 当前 exact cosine 全表扫,语料到 10⁵ chunk 级会成为延迟主源;pgvector 0.8.2 已支持 HNSW,`RAG_KB__VECTOR_STORE__HNSW_ENABLED` 开关已预留(现固定 false) | 需要 recall 回归评测护航(1.2 的评测底座) |
| 混合检索 | BM25/tsvector 词法路 + 向量路做 RRF——现有 Evidence Group RRF 融合器(`src/rag_kb/retrieval/fusion.py`)可直接多加一路 | PostgreSQL 原生 FTS 即可,不必引入 ES |
| 离线 cross-encoder rerank | 当前 rerank 是确定性 lexical/vector 加权;一个本地小型 cross-encoder(如 bge-reranker)可显著提升 top-K 排序,且保持"不调用在线模型判证据"的原则 | 模型烘焙进镜像,复用 Docling 工件的离线 manifest 模式 |
| 层次化/摘要检索 | RAPTOR 式文档级摘要层,长文档先命中摘要再展开成员 chunk;现有 Evidence Group 机制就是"父文本携带成员"的原型 | 摘要生成引入新的 provider 成本,需预算控制 |
| GraphRAG | Docling 已产出结构关系(caption_of、同页、显式引用七层关系),把它们与跨文档实体链接结合,可回答"跨文档综合"类问题 | 大工程,建议先用评测证明单跳检索的天花板 |

### 2.2 Chat 与生成

- **Token streaming**:当前刻意不做(终态 SSE)。做的话不必破坏"PostgreSQL 唯一事实源"
  ——流式只作为 best-effort 预览通道,终态仍由持久 ChatRun 提交,断流回退到现有轮询。
- **跨 Session 长期记忆**:现在 memory/ 只做 6-turn 短期上下文。可加用户级偏好记忆
  (语言、详略偏好),但要延续"历史是 reference_only_untrusted"的纪律,
  记忆永远不能成为事实来源——这个不变量是本系统的灵魂,不能为功能让步。
- **Agentic 多步检索**:当前固定线性 LangGraph。可控的演进不是开放 agent,
  而是**有界迭代检索**:assess_evidence 判定覆盖不足时允许一次改写再检索
  (n 固定、确定性准入不变),LangGraph 图结构支持条件边,改动集中在
  `workflows/chat_graph.py` 一处。
- **多知识库联合问答**:检索计划已按 KB 冻结,扩展为"一次 run 绑定多个 KB revision"
  主要是查询计划与 citation 归属的工作。

### 2.3 多模态深化

- 当前明确没有生成式 Caption(架构文档第 7 节)。引入离线 VLM caption 作为
  **可选表示**(不替换作者 caption,只增加 `embedding_text` 的一个受限 section)
  可提升图表可检索性,且 Composite Chunk 的表示矩阵天然支持新增 representation。
- 图表语义理解:table structure 已有(TableFormer),下一步是 chart→数据点抽取,
  让"2023 年营收多少"能命中柱状图。
- 音视频:转写(ASR)后走既有文本链路,关键帧走既有图片链路;
  `IndexAsset`/relation 模型不需要大改。

### 2.4 成本与性能工程

- **嵌入批次**:当前固定 batch=10(`FixedBatchSize`,provider 上限),索引大文档时
  API 往返是主要延迟;可做并发批次(保持每批 10)与断点续传(`IndexChunkPlan` 已支持)。
- **解析进程隔离的再评估**:2026-07-20 移除了子进程隔离,Docling 常驻 ~2.4 GB RSS 且与
  Worker 同进程;当 chat lane 与 indexing lane 并发时,一次 OOM 会同时杀死聊天服务。
  本次审查确认这不只是理论风险(PROJECT_REVIEW H3:挂起转换可永久占用单线程 executor;
  H4:图片炸弹/密集 CSV 在应用限制前全量解码且无内存上限)。未来多 Worker 时建议把
  parsing 拆回独立进程/容器(indexing-worker 与 chat-worker 分离),这也是 2.5 节
  多 Worker 演进的第一步。
- **Provider 用量护栏**:usage 已逐 run 持久化,但没有全局预算/限流;
  加 workspace 级日预算与熔断,是共享部署前的必要护栏。

### 2.5 架构演进(通往共享部署)

架构文档第 2 节列出的"当前明确不具备"就是演进清单,建议顺序:

1. **对象存储适配**:`FileStore` 已是协议(`adapters/file_store/`),加 S3 实现,
   消除"共享本地卷"这个横向扩展的第一个硬约束;
2. **索引/聊天 Worker 分离**:同一 job 表、不同 lane 的进程分工,
   先解决重解析与低延迟聊天的资源隔离(无需 execution epoch);
3. **多 Worker 安全接管**:execution epoch + fencing token(架构文档已点名 Outbox/epoch
   为缺失项),`FOR UPDATE SKIP LOCKED` 的领取协议本身已是多消费者安全的,
   主要工作在"过期 lease 的写栅栏";
4. **K8s/Helm 部署形态**:镜像已 digest 固定、非 root、健康检查齐备,
   主要缺 secrets 管理(External Secrets/Vault)与迁移 Job 编排;
5. **可观测性平台化**:结构化 JSON 日志与稳定事件名已就绪,接 OpenTelemetry
   trace + Prometheus 指标(job 排队深度、provider 延迟、准入率、拒答率)。

## 3. 长期路线建议(12 个月视角)

| 阶段 | 主题 | 关键交付 |
| --- | --- | --- |
| 近期(1-2 月) | 还债与护栏 | 修复 PROJECT_REVIEW P0/P1(解析护栏 H3/H4、语义断点性能 H2、组键冲突 H1、可观测性 M6-M8);评测 API 化 + 质量基线;批量导入 CLI |
| 中期(3-6 月) | 检索质量与团队化 | HNSW + 混合检索(评测护航);OIDC 单团队登录与审计;用户反馈回路;流式预览 |
| 远期(6-12 月) | 共享部署与生态 | 对象存储 + Worker 分离 + 多 Worker;连接器框架;文档级 ACL;成本预算与计费报表 |

## 4. 需要守住的架构不变量(演进红线)

无论走哪条路线,以下现有纪律建议永不放弃,它们是这个代码库最值钱的部分:

1. **PostgreSQL 是唯一执行事实源**;任何流式/缓存/队列都只能是交付优化,不是状态。
2. **证据准入永远确定性**;模型可以生成答案,但不能决定"证据够不够格"。
3. **零准入证据必然拒答**;partial policy 不是放水通道。
4. **检索内容、历史、图片是不可信输入**;历史不成为 citation,图片内指令不改变 policy。
5. **未验证的 provider 输出不落库**;一次有界 repair,然后安全回退。
6. **fail-closed 启动与配置**;新能力宁可拒绝启动,不做静默降级。
7. **文档协议**:架构文档、tracker、计划与代码同一工作项内同步——这是本仓库
   可被多代 agent 连续维护的原因,规模化后只会更重要。

## 5. 不建议做的事

- **不要为"演示效果"引入开放式 agent 循环**(不受限的工具调用/自主检索):
  与可审计证据链定位冲突,且破坏现有确定性回放能力。
- **不要在评测底座建立前调检索参数**:cosine/rerank floor、RRF 权重都已冻结且有快照,
  没有回归评测就调参会摧毁现有的可比性。
- **不要提前抽象多 provider 路由**:当前单 provider 冻结身份 + 快照记录的模式很干净,
  等第二个真实 provider 需求出现再抽象。
- **不要把两个前端合并**:用户端(极简、CSP 严格)与诊断端(全量事实展示)的
  职责分离是有意设计,合并会让诊断信息渗入用户面。
