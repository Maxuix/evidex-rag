# 企业知识库前期架构方案评审

评审对象：`2026-07-10-enterprise-knowledge-base-design.md`  
评审日期：2026-07-13  
评审重点：前期技术路线、会造成后期返工的设计隐患，以及值得现在采用的改进方案

已确认项目定位：**仅用于本地开发验证，当前不面向企业服务器、共享测试环境或真实生产落地。** 因此，本报告不把企业级认证、高可用、正式备份、合规审计和多租户隔离列为 P0/P1A 必做项。

## 1. 结论

该方案**不存在需要推倒重来的技术路线错误**，总体方向合理，尤其是以下决策值得保留：

- 采用模块化单体，API 与 Worker 分进程部署；
- P1A 使用 PostgreSQL 持久化任务表，不为了“异步”过早引入 Redis/Celery；
- PostgreSQL + pgvector 作为初期关系数据和向量检索底座；
- 文档版本、索引版本、Embedding Space 分离，避免模型变更后静默复用不兼容向量；
- 以业务数据库作为最终事实源，SSE、Redis 和 LangGraph checkpoint 都不承担产品事实；
- LangGraph 延后到真正需要分支、恢复和人工介入时使用；
- 对幂等、失败恢复、引用可追溯和评测均预留了明确边界。

在本地开发验证范围内，当前版本仍有 **2 项 P0 阻断问题和 3 项高优先级问题**。这些问题如果不在实施计划前明确，容易造成数据库约束无法按文档落地、读取到不一致索引，或者无法正确验证 RAG 检索效果。其余企业化问题可以明确延后。

建议结论为：**有条件通过评审；完成第 3 节 R2–R4 的修订后再进入详细开发计划。**

## 2. 风险总览

| 编号 | 级别 | 问题 | 建议处理阶段 |
| --- | --- | --- | --- |
| R1 | 已解决 | 已确认 P1A 仅用于本地开发验证，开发身份方案符合当前范围 | 保留开发环境限制即可 |
| R2 | 阻断 | “一个只读事务即同一快照”不成立，默认 `READ COMMITTED` 每条语句使用新快照 | P0 固化查询和隔离策略 |
| R3 | 阻断 | 部分唯一约束的 SQL 表述不可直接实现；同 KB 外键和序列分配细节缺失 | P0 产出可执行迁移并做并发测试 |
| R4 | 高 | Embedding Space 动态建表增加实现复杂度；基础组件也未固定兼容版本 | P0 固定版本和建表方式 |
| R5 | 中 | 本地文件仍需明确共享卷和原子重命名条件，但正式备份恢复可以延后 | P1A |
| R6 | 高 | 单 Worker 同时处理索引与聊天，但没有公平调度、并发槽位和总超时定义 | P1A |
| R7 | 中 | P1A SSE 不应长期占数据库会话，且依赖的 FastAPI 最低版本未写明 | P1A |
| R8 | 高 | 只做向量检索但没有语料语言画像和关键词检索基线，企业编号/术语检索风险较高 | P0/P1A |
| R9 | 低 | 本地验证阶段只需避免测试数据无限堆积，正式保留与合规策略可延后 | 开发中处理 |
| R10 | 中 | “OpenAI-compatible”接口可能掩盖不同模型的结构化输出、限流和重试能力差异 | P0/P1A |
| R11 | 中 | Prompt Injection 描述过于绝对，容易形成错误安全承诺 | P1A 测试与文案修正 |
| R12 | 中 | P1B 同时引入 Redis、Outbox、多 Worker、LangGraph 和全量重建，范围耦合过大 | 排期时拆分，不阻塞 P1A |

## 3. 范围确认与实施计划前修正项

### R1. 认证问题已由项目定位解决

原方案在“P1A 不实现认证”与“所有非开发环境必须配置认证”之间存在表面冲突。现在已经确认整个项目仅运行在本地开发验证环境，因此该问题不再构成阻断。

当前只需保留以下约束：

- `DEPLOYMENT_PROFILE=development` 或等价显式开关；
- 默认绑定 `127.0.0.1`，不默认监听所有网卡；
- 使用服务器配置的固定开发 principal，禁止客户端通过请求头任意切换身份/workspace；
- README 标明“仅用于本地技术验证，不具备企业部署所需的身份、权限、审计和恢复保证”；
- 测试文档尽量使用脱敏或可公开数据。

`AuthContext -> AccessPolicy -> MetadataFilter` 接口仍值得保留，便于未来扩展，但 P0/P1A 不需要实现 OIDC、JWT、组、ACL 或可信代理认证。

### R2. 检索读取的事务快照保证不成立

**现状**

第 7 节写明：先读取 `active_index_revision_id`，再执行元数据/向量查询，并在“一个只读事务快照”内完成。但 PostgreSQL 默认隔离级别是 `READ COMMITTED`，同一事务内的后续语句会获取新的快照；仅仅使用同一个 `AsyncSession` 或事务并不能满足文档承诺。[PostgreSQL 官方事务隔离说明](https://www.postgresql.org/docs/18/transaction-iso.html)

在索引切换并发发生时，可能读取旧 selector，却看到已经切换后的 serving 状态，从而少返回内容或得到与所选 revision 不一致的结果。

**解决方案**

优先采用以下方案：

- 将 selector 解析、权限/serving 过滤和向量排序写成**一条 SQL 语句**，通过关联 `knowledge_base.active_index_revision_id` 完成；单条语句在 `READ COMMITTED` 下已有一致快照。
- 如果受向量适配器限制必须执行多条 SQL，则该只读用例显式使用 `REPEATABLE READ READ ONLY`，不要把它设置为全局默认隔离级别。
- 集成测试增加“检索与 revision activation 并发”的用例，结果只能完整属于旧 revision 或新 revision，不能混合。

### R3. 数据库约束和 SourceChange 序列需要改成可执行设计

**现状**

文档中的以下写法表达了正确意图，但不是可直接创建的 PostgreSQL `UNIQUE` 约束：

```sql
UNIQUE (kb_id) WHERE status = 'active'
UNIQUE (document_id, index_revision_id)
  WHERE build_status = 'ready' AND serving_status = 'serving'
```

PostgreSQL 对“只约束部分行”的唯一性要求使用**部分唯一索引**，不能写成普通表级唯一约束。[PostgreSQL 约束文档](https://www.postgresql.org/docs/18/ddl-constraints.html)

此外，`SourceChange` 要求 KB 内严格单调，但未定义并发上传时如何分配序号；若实现成 `MAX(seq) + 1` 会发生竞争。`active_index_revision_id` 的“同 KB 外键”也需要组合唯一键/组合外键或触发器的精确定义。

**解决方案**

- 在 Alembic 中使用 `CREATE UNIQUE INDEX ... WHERE ...`，SQLAlchemy 模型使用 `Index(..., unique=True, postgresql_where=...)`，不要声明成 `UniqueConstraint`。
- `SourceChange` 使用原子语句分配序号：

  ```sql
  UPDATE knowledge_base
     SET source_change_seq = source_change_seq + 1
   WHERE id = :kb_id
  RETURNING source_change_seq;
  ```

  同一事务插入 `SourceChange`，并增加 `UNIQUE (kb_id, source_change_seq)`。
- 在 `index_revision` 上增加 `UNIQUE (kb_id, id)`，再让 `knowledge_base` 的 `FOREIGN KEY (id, active_index_revision_id)` 引用 `index_revision(kb_id, id)`；仅在组合外键无法表达的状态一致性上使用 deferred constraint trigger。
- P0 必须产出真实 Alembic migration，并用真实 PostgreSQL 做“两个并发激活、两个并发上传、版本倒序完成”的集成测试。

### R4. 禁止运行时 DDL，并补充版本兼容矩阵

**现状**

第 6、8 节描述 VectorStore 在启动或 provisioning 时创建/验证每个 Embedding Space 的物理表和 HNSW 索引。若“创建”发生在应用运行期，API/Worker 数据库账号就需要 DDL 权限，这与最小权限和“迁移由显式部署命令执行”冲突。动态表名还会增加 SQL 注入防护、迁移追踪和清理难度。

同时，文档只写了大版本范围，没有确定 Python、FastAPI、SQLAlchemy、asyncpg、PostgreSQL 和 pgvector 的已验证组合。方案依赖 pgvector 的 iterative scan（0.8.0 才引入），也依赖 FastAPI 原生 SSE（0.135.0 才加入）。[pgvector 官方说明](https://github.com/pgvector/pgvector)；[FastAPI SSE 官方说明](https://fastapi.tiangolo.com/tutorial/server-sent-events/)

**解决方案**

- P1A 只有一个固定 Embedding Space，向量表和索引应由 Alembic/部署作业创建；运行时 VectorStore 只做能力验证，不执行 DDL。
- 使用两个数据库角色：migration role 拥有 DDL；runtime role 仅拥有所需 DML/sequence 权限。
- P0 增加兼容矩阵和锁定策略：Python 小版本、FastAPI `>=0.135.0`（若使用原生 `fastapi.sse`）、SQLAlchemy 2.0 已验证补丁版、PostgreSQL 主版本、pgvector `>=0.8.0` 的已验证补丁版、Docker 镜像 digest。
- 不使用浮动 `latest` 镜像；升级通过依赖更新 PR、迁移测试和检索回归评测完成。

## 4. P1A 开始前应补齐的问题

### R5. 明确本地文件存储的运行条件

**风险**

“原子 rename”只在 staging 与 final 位于同一文件系统时成立。Docker Compose 中 API 写文件、Worker 读文件，还要求二者挂载同一个持久卷和一致的容器内路径。该问题会直接影响本地闭环能否稳定运行，仍需处理。

Docker volume 可以满足本地持久化，但它本身不是正式备份机制。[Docker Volume 官方文档](https://docs.docker.com/engine/storage/volumes/)

**建议**

- P1A Compose 明确一个 source volume，同时挂载到 API 和 Worker；staging/final 必须在同一卷。
- 保留 staging/orphan janitor，并测试 API/Worker 重启后文件仍可读取。
- README 明确本地数据默认不提供主机故障恢复保证，重要测试数据应能从样例或脚本重新导入。
- 不需要在 P1A 实现正式备份、RPO/RTO 或 S3；这些内容等到部署目标变化后再处理。

### R6. 为单 Worker 增加调度和资源隔离规则

**风险**

同一 Worker 同时轮询 `IndexingJob` 和 `ChatRun`。如果按单队列串行或“先扫到谁处理谁”，大文档解析、批量 embedding 或长模型调用会让聊天任务长期饥饿。方案已有 heartbeat，却没有规定：

- 两类任务的优先级和公平性；
- 最大并发、每类并发槽位；
- 单次外部调用和整条任务的硬超时；
- heartbeat 使用的独立会话；
- SSE/API、poller、heartbeat 并发时的连接池预算。

SQLAlchemy 明确要求并发任务各自使用独立 `AsyncSession`，不能共享一个会话。[SQLAlchemy Session 并发规则](https://docs.sqlalchemy.org/en/20/orm/session_basics.html#is-the-session-thread-safe-is-asyncsession-safe-to-share-in-concurrent-tasks)

**建议**

- 仍保留一个 Worker 进程，但设置两个执行 lane：chat 和 indexing 各有独立 semaphore；chat 保留至少一个槽位。
- poller 使用加权轮询或明确优先级，并加入 aging，避免 indexing 永久饥饿。
- 每个 claim、heartbeat、状态更新使用自己的短生命周期 `AsyncSession`；heartbeat 不复用正在执行任务的 session。
- 定义 provider connect/read/total timeout、单任务总 deadline、取消后的状态转换和连接池上限。
- P1A 压测至少覆盖“一批文档正在索引时，新聊天请求仍在目标排队时延内开始”。

### R7. 补充 SSE 的资源管理约束

**风险**

P1A SSE 等待终态时，如果持有数据库事务/连接、每个客户端高频查库，或者没有最长连接时间，很容易耗尽 API 连接池。当前方案只说明 keepalive 和权威状态源，未说明如何等待终态。

**建议**

- SSE 生成器不得持有跨等待周期的数据库事务或 `AsyncSession`；每次状态检查使用新的短会话并立即释放。
- 设定有抖动的轮询间隔、最大 SSE 连接时长、单用户/单 run 连接数限制和断开检测。
- 继续以 `GET /chat/runs/{id}` 为权威恢复路径；P1A 没必要为了终态通知提前引入 Redis。
- 使用 FastAPI 原生 SSE 时固定 `>=0.135.0`，利用其 ping、`Cache-Control: no-cache` 和 `X-Accel-Buffering: no` 默认处理；若选择其他 SSE 库，则必须显式记录依赖和代理配置。[FastAPI SSE 官方文档](https://fastapi.tiangolo.com/tutorial/server-sent-events/)

### R8. 先确定语料语言画像，并给向量检索增加词法基线

**风险**

向量检索适合完成 P1A 闭环，但企业知识库大量问题依赖错误码、产品型号、法规条款、缩写和人名，纯向量检索可能稳定漏召回。方案把 hybrid retrieval 延后到 P2，但 P0 甚至没有说明目标语料是中文、英文还是混合语言；这会影响 embedding 模型、切分 token 计算、Unicode 规范化、全文检索分词和评测集。

pgvector 官方也说明 HNSW 在过滤条件下先扫描近邻再过滤，可能少于 `top_k`；iterative scan、部分索引和分区只能缓解，需要用真实过滤分布评测。[pgvector Filtering/Iterative Scan](https://github.com/pgvector/pgvector#filtering)

**建议**

- P0 增加 `CorpusProfile` 决策：主要语言、混合语言比例、典型文档长度、标识符密度、更新频率、敏感级别。
- P1A 保留 vector 为产品默认，但评测中增加一个词法检索基线和“精确编号/术语”用例。
- 英文或 PostgreSQL 支持良好的语言，可用 `tsvector/tsquery` 建立低成本 lexical baseline；PostgreSQL 原生支持全文匹配和排序。[PostgreSQL Full Text Search](https://www.postgresql.org/docs/18/textsearch.html)
- 中文/混合语言不要未经评测就假定默认 PostgreSQL 分词足够；先用真实语料比较候选 tokenizer/BM25 实现，再决定是否在 P1B/P2 做 PostgreSQL 内 hybrid、外部检索引擎或专用向量库。
- HNSW 只在 exact search 达不到延迟目标时开启；将 filtered Recall@k 作为启用 ANN 的门槛，而不是默认认为 HNSW 更先进。

### R9. 本地阶段只做最小数据清理

**风险**

当前 `DELETE` 是软删除，历史 Citation 又保存 quoted text 和来源快照。对本地开发验证而言，这不构成合规阻断，但反复测试后可能积累 orphan 文件、retired vectors、旧 chunks 和聊天数据，干扰调试并占用磁盘。

**建议**

P1A 只需提供一个可重复执行的开发清理命令或脚本，用于清理 orphan/staging 文件、retired vectors/chunks 和过期任务记录；也可以提供完整清空本地数据库与文件卷的 reset 流程。正式的保留矩阵、物理清除 API、legal hold 和合规审计全部延后。测试时避免导入未经脱敏的真实敏感文档。

## 5. 可在开发中处理、但应避免误导的事项

### R10. 将 Model Provider 设计为能力契约，而不只是相同字段

`base_url/api_key/model/timeout/max_retries/max_concurrency` 不足以统一真实模型差异。不同 HTTP 服务对 JSON Schema、流式、批量 embedding、最大输入 token、限流响应、usage、seed 和幂等标识支持不同。

建议增加启动期 capability discovery/静态声明：`supports_structured_output`、`max_input_tokens`、`embedding_dimension`、`max_batch_size`、`supports_usage`、`retryable_statuses`。P1A 对结构化答案可以采用“原生 schema 输出优先，普通 JSON + Pydantic 校验回退”，并对重试使用指数退避与随机抖动。无需为了统一接口引入重量级模型网关。

### R11. 不要把 Prompt Injection 防护写成绝对保证

“文档指令不能改变系统策略”应改成“系统不向模型提供工具、凭证和授权决策能力，并将检索内容作为不可信数据隔离；提示词降低但不能消除间接注入风险”。OWASP 明确指出 RAG 不能完全消除 Prompt Injection，恶意内容可来自被检索文档。[OWASP LLM01:2025](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)

P1A 不需要复杂检测模型，但应增加几条恶意文档测试：要求泄露系统提示词、要求忽略引用规则、伪造 citation ID、诱导输出其他 workspace 内容。现有“无工具、无凭证、服务端权限过滤、结构化 citation 校验”已经是正确的主要控制措施。

### R12. 将 P1B 拆成两个可独立验收的里程碑

当前 P1B 同时包含 Outbox、Redis Streams、多执行器、lease/epoch、LangGraph checkpoint、SSE replay 和全量索引重建。它们并不是一个不可分割能力，集中实施会显著扩大状态组合和测试矩阵。

建议排期时拆分：

- **P1B Delivery Reliability**：Outbox、多 Worker fencing、可靠队列、终态事件重放；
- **P1C Index Lifecycle**：SourceChange snapshot/catch-up、全量重建、切换和清理；
- LangGraph 只有在 P1B/P1C 出现真正的中断恢复或复杂分支需求时启用，不作为“可靠性升级”的强制依赖。

该拆分不需要改变当前公共 API 和核心数据模型。

## 6. 对“更先进方案”的判断

| 领域 | 当前方案 | 是否应替换 | 评审建议 |
| --- | --- | --- | --- |
| 架构形态 | 模块化单体 + 独立 Worker | 否 | 对 P1 最合适，微服务只会增加一致性和运维成本 |
| 异步任务 | PostgreSQL job table | 否 | P1A 足够；保留 claim、heartbeat、重试和公平调度即可 |
| 向量库 | pgvector | 否 | 先用真实指标判断；不要因“更专业”提前切 Qdrant/Milvus/Vespa |
| 检索 | vector-only | 暂不替换，但需基线 | 增加 lexical baseline 和语言画像，按评测决定 hybrid/rerank |
| 工作流 | 直接 pipeline，LangGraph 延后 | 否 | 这是比一开始上复杂图更稳妥的方案 |
| 流式交付 | 权威状态 + 终态 SSE | 否 | P1A 足够；重点补资源限制和版本固定 |
| 文件存储 | 本地持久卷 | 仅限开发/小试点 | 真正共享或多机部署前迁移到 S3 兼容存储 |
| 多租户隔离 | workspace 逻辑过滤 | P1 可保留 | 多租户/ACL 上线前再评估 PostgreSQL RLS、分区或物理集合隔离 |

当前不建议为了“先进”引入 Agentic RAG、知识图谱、专用向量数据库、多模型路由或复杂 reranker。这些能力都应由评测暴露的具体缺陷驱动，否则会增加成本和故障面，而不会自动提高答案质量。

## 7. 建议后的前期交付边界

### P0 必须完成

- 在 README 和配置中固定“仅限本地开发验证”的边界，默认绑定 loopback；
- 固定依赖/镜像兼容矩阵和数据库 runtime/migration 角色；
- 产出可执行 Alembic schema，落实部分唯一索引、组合外键和原子 SourceChange 序列；
- 固化检索的一条 SQL 快照策略或显式 `REPEATABLE READ READ ONLY`；
- 定义 CorpusProfile 和模型能力配置；
- 明确本地文件卷和原子 rename 条件。

### P1A 必须完成

- 一条完整的 txt/md 上传、索引、检索、回答和引用闭环；
- chat/indexing 两类任务公平调度、独立并发槽位、硬超时和独立 AsyncSession；
- SSE 不长期持有数据库连接，并有最大时长/连接数限制；
- exact vector 基线和 filtered Recall@k；加入标识符、无答案、更新、删除和恶意文档用例；
- 提供本地测试数据 reset/cleanup 流程，不要求正式认证和备份恢复。

### 可以继续延后

- 生产 PDF/Office/OCR 解析；
- 专用向量数据库；
- 完整混合检索和 reranker；
- LangGraph checkpoint；
- 长期用户记忆、外部连接器、复杂 ACL 管理界面；
- 高可用、自动扩缩容和完整可观测平台。

## 8. 最终建议

方案主干可以保留，不建议重写。进入开发计划前，优先修改 R2–R4；P1A 重点处理 R6 和 R8，并完成 R5、R7、R9 的轻量实现。认证、正式备份、合规删除、高可用和多租户隔离可以继续延后。

最重要的判断是：**当前缺少的不是更复杂的 RAG 框架，而是把数据库快照、可执行约束、任务资源边界、语料画像和检索评测定义清楚。** 这些问题越早解决，后续更换解析器、模型或检索策略的成本越低。
