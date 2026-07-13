# 企业知识库架构方案复审

评审对象：`2026-07-10-enterprise-knowledge-base-design.md`（修订版）  
复审日期：2026-07-13  
范围：核验上一轮评审的整改是否落实，并检查修订后仍存在或新引入的前期实现风险。

## 1. 复审结论

本次修订质量很高。上一轮列出的核心问题均已被正面处理，方案已经从“P1 过重、部分实现细节未闭合”收敛为**可以进入 P0/P1A 实施计划**的架构设计。

不过，在开始拆任务前仍建议修正两个高优先级状态机问题：

1. 构建中的 revision 在激活前应是 `candidate`，但文档又要求它在激活前存在 `serving` 关联，逻辑互相矛盾。
2. 同一文档连续上传多个版本时，旧索引任务晚完成，可能覆盖较新的版本成为 serving；当前文档尚未给出最后一道条件更新保护。

另外有数项 API、任务调度和配置可复现性问题需要在 P1A 计划中明确。它们不需要改变整体技术路线。

## 2. 上轮整改核验

| 上轮关注点 | 本次落实情况 | 复审判断 |
| --- | --- | --- |
| P1 范围过大 | 已拆分为 P1A 可用闭环与 P1B 可靠性硬化；P1A 不再强制 Outbox、多 worker 接管、checkpoint 恢复和 SSE 回放。 | 已落实 |
| 同步 ORM 与异步服务混用 | 第 3.1 节明确统一使用 `AsyncSession + asyncpg`，并禁止 async 路径直调同步 session。 | 已落实 |
| pgvector 多模型/维度的物理隔离 | 增加不可变 `EmbeddingSpace`；定义每个 space 独立的物理表、维度、类型和 HNSW 运算符类。 | 已落实 |
| ANN 过滤导致召回不足 | 增加 `RetrievalQueryPlan`、over-fetch、`ef_search`、bounded iterative scan、分区策略和带过滤的 Recall@k。 | 已落实 |
| 索引重建的读写竞态 | 增加 `SourceChange` 单调账本、snapshot/catch-up、selector、唯一约束及切换事务。 | 基本落实，见第 3.1 项需修正的状态矛盾 |
| 引用校验被误解为语义证明 | 改为 schema 约束的 claim-citation 关联与覆盖校验，并将语义支撑移至离线/人工评估。 | 已落实 |
| “流式”语义不准确 | 明确 P1A 是终态 SSE 或已验证答案的渐进展示，不是 provider token streaming。 | 已落实 |
| Redis 投递语义不完整 | 将可靠投递后移 P1B，并补充 consumer group、ack、reclaim、退避及 DLQ 要求。 | 已落实 |
| LangGraph 在固定线性 P1 中过重 | P1A 改为直接 `ChatPipelineService`，LangGraph/恢复只在 P1B 需要时启用。 | 已落实 |
| 无登录环境的授权边界 | 增加仅 development 可用的固定主体、loopback/受控网络限制，以及非开发环境 fail-closed。 | 已落实 |
| 文档解析和提示注入基线 | 增加文本限制、资源隔离子进程、无网络/无凭证、最小证据片段和未来二进制扫描边界。 | 已落实 |

## 3. 仍需修正的问题

### 3.1 高：revision 激活前后的 `candidate` / `serving` 状态相互矛盾

第 6 节将非 active revision 的内容描述为 `candidate`，第 7 节又要求 activation 在切换 selector **之前**确认新 revision 中“每个文档有一个 ready + serving association”。这是不可能同时满足的：如果它已经是 `serving`，则不再是等待激活的 candidate；如果仍是 candidate，就不符合激活前验证条件。

风险是实现人员会选择其一，导致：构建 revision 在尚未成为 active 前被检索到，或 activation 的完整性检查无法通过。

建议将激活事务明确为以下顺序（同一 PostgreSQL 事务内）：

```text
1. 锁定 KnowledgeBase、source-change 序列和新 revision
2. 验证：每个 snapshot 中的未删除文档恰有一个
   build_status=ready AND serving_status=candidate 的关联
3. 将新 revision 的这些 candidate 批量提升为 serving
4. 将旧 active revision 标记 retired（其关联一并 retired，或由 read selector 排除）
5. 更新 KnowledgeBase.active_index_revision_id，并将新 revision 标记 active
6. 提交
```

相应地，第 6 节的 activation 不变量应写为“激活前有且仅有一个 ready candidate；提交后有且仅有一个 ready serving”，而不是两者都要求为 serving。

### 3.2 高：连续版本上传可能让旧任务反向覆盖新版本

场景如下：上传 V1 后索引任务慢；随后上传 V2，V2 先完成并成为 serving；V1 的旧任务随后完成。当前规则只说“新版本准备好后切换”，没有要求 worker 在 promotion 时验证它仍是 `Document.current_version_id` 以及仍是该文档最新的 `SourceChange`。

风险是 V1 被迟到的 worker 提升为 serving，检索到过期内容；这类问题在网络波动、批量 embedding 或重试中很常见。

建议：

- `IndexedDocumentVersion` 保存创建它的 `source_change_seq`；
- promotion 使用单条条件更新或在同一事务中加锁，前提必须同时满足：目标为 `ready + candidate`、`Document.current_version_id = target.document_version_id`、且该 document 没有更高序号的 upsert/delete `SourceChange`；
- 条件不满足时将目标标记 `retired/superseded` 并排入清理，绝不提升；
- 为 “V1 上传 → V2 上传 → V2 完成 → V1 完成” 增加端到端并发测试。

### 3.3 中：P1A 的任务领取与 stale-work 规则还需选定一个可实现的基线

P1A 合理地不再依赖 Outbox 和多 worker，但“单 worker claim durable queued job”“启动时处理 stale running work”仍需要确定原子领取与过期判定方式。否则 worker 重启、任务卡在 provider 调用、人工误启动第二个 worker 时，可能重复执行或永久卡住。

建议在 P1A 计划中固定以下任一方案；推荐第一种，最小且不额外引入基础设施：

- **PostgreSQL job-table poller（推荐）**：使用 `FOR UPDATE SKIP LOCKED`/条件更新领取 `queued` job；记录 `claimed_by`、`claimed_at`、`attempt`、`next_attempt_at`、`heartbeat_at`；只有 heartbeat 超时后才允许 reconciliation 重新排队。Docker Compose 以单 worker 副本运行，但数据库条件仍防误启动重复执行。
- 或选定一个成熟队列框架，并在 P1A 文档中写清消息确认点、失败重试与 PostgreSQL job 状态谁是权威。

无论选哪种，P1A 的“stale”超时应大于模型调用、解析和 embedding 批次的最大合法时长，且每次重试要增加 `attempt` 并保留错误原因。

### 3.4 中：API 的 reindex 返回语义存在自相矛盾

第 11.1 节说“uploads, reindex requests, and ChatRun creation return `202` in every deployment profile”，但同一节又标注 `POST /documents/{id}/reindex` 是 P1B 能力，P1A 应返回 capability error。

建议改为：

- P1A 的该端点不暴露，或明确返回 `409/501` 加稳定码 `CAPABILITY_NOT_ENABLED`；
- 只有 P1B 启用时才返回 `202` 和 job 信息；
- OpenAPI 按部署能力生成/标注，前端不应把 P1A 的“不可用”显示为失败任务。

### 3.5 中：异步 ChatRun 的 provider 失败不应表述为直接 `503`

第 18 节仍写“Chat provider failure produces a retryable `503`”，但 `POST /chat/runs` 已经在调用模型前返回 `202`。模型调用失败发生在 worker 中，无法再改变原 POST 的 HTTP 响应。

建议修改为：worker 将 run 终态写为 `failed`，错误码为例如 `MODEL_PROVIDER_UNAVAILABLE`，其中包含 `retryable=true`；状态查询返回该业务错误，SSE 发出 `run.failed`。只有在创建 run 前就能发现的配置/连接健康问题，或同步管理接口，才适合直接返回 HTTP `503`。

### 3.6 中：EmbeddingSpace 需要记录“实际部署版本”，不能只记录模型别名

设计已保存 provider/model/version，但许多兼容 API 的 `model` 是可变别名（例如服务端将别名指向新权重），不能保证相同字符串产生相同向量。

P1A 至少应在创建固定 space 时记录：provider base URL 的逻辑标识、请求 model 名、服务端返回的 resolved model/deployment revision（若可得）、向量维度、归一化约定、tokenizer/chunker 版本和配置指纹。若服务端无法给出 revision，则将部署配置整体 fingerprint 作为人工变更受控项，并禁止无记录替换。

这不是要求实现模型注册中心，而是保证索引、评估和故障定位可复现。

### 3.7 中：文件写入与数据库提交之间的补偿责任需要落到 P1A

上传涉及本地文件存储和 PostgreSQL，二者没有共同事务。文档已在通用一致性模型中提到 compensation jobs，但 P1A 的单 worker 闭环没有明确谁、何时清理以下孤儿：文件已保存而数据库事务失败；数据库已记录而移动/读取文件失败；删除已提交但物理文件清理失败。

建议 P1A 采用简单的 staged-file 协议：先写入不可服务的临时路径，数据库提交 source metadata 后原子 rename 到最终路径；失败路径由定期 janitor 根据年龄和 DB 引用清扫。删除先使数据库不可服务，再异步删除文件；失败保留可重试清理记录。这样无需提前引入 Outbox。

### 3.8 低：EmbeddingSpace 的 pgvector 能力校验需在创建时 fail fast

文档已规定专用表和相符的 HNSW operator class，但 implementation plan 还应验证所选类型能容纳模型输出维度，并验证扩展/运算符类在目标 PostgreSQL 镜像存在。否则首次建表或建索引时才发现模型维度、`vector`/`halfvec` 类型或镜像版本不兼容。

建议将“创建/启动时校验 `EmbeddingSpace` 与 pgvector capability”加入 P0 集成测试及启动前检查；P1A 固定 space 只需验证一次。

## 4. 可保持不变的关键决定

以下选择在本次复审中依然正确，建议不要为了“更先进”而替换：

- P1A 继续使用 PostgreSQL + pgvector，不预设专用向量数据库迁移。
- P1A 使用直接应用服务流水线，不强制 LangGraph。
- P1A 只发布已提交、已验证的答案，不发布原始 token。
- 业务数据库保持 ChatRun、消息、引用和终态的唯一事实来源。
- `SourceChange` + active selector + 版本化构建作为 P1B 在线重建的基础。
- 通过结构化 claim/citation 约束降低无引文回答风险，同时把语义正确性留给评估体系。

## 5. 建议的实施前收口清单

在生成 P0/P1A 详细任务前，完成以下文档级调整即可：

1. 修正第 6/7 节中 building revision 的 `candidate` 与 `serving` 激活顺序。
2. 增加“promotion 仅允许最新 DocumentVersion”的条件更新规则和并发测试场景。
3. 选定 P1A 的 PostgreSQL poller 或具体成熟队列，并定义 claim、heartbeat、超时、退避和 attempt 字段。
4. 修正 reindex 的 P1A/P1B HTTP 响应约定，以及异步 provider 失败的业务错误表达。
5. 为 EmbeddingSpace 增加实际部署配置 fingerprint，并补充 pgvector 能力启动校验。
6. 为 local file store 补充 staged-file 与孤儿清理责任。

## 6. 最终判断

建议状态更新为：**附少量条件后批准进入 P0/P1A 实施计划**。

上述两项高优先级问题属于规则定义冲突和并发时序保护，修复成本低、但越晚处理越容易造成数据版本错误。其余问题可直接作为 P0/P1A 实施任务的验收条件，不需要重新调整总体架构。
