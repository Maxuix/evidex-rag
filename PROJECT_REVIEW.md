# PROJECT_REVIEW — 全面项目审查报告

> 审查日期:2026-07-26
> 审查基线:分支 `feat/docling-migration`,HEAD `29f7d4e`(工作树干净)
> 审查方式:16 个并行审查代理对 10 个子系统做全量代码阅读(共 1,025 次工具调用),
> 30 个重要发现全部经独立对抗性验证代理逐条复核确认(0 个被驳回),另有完整性批判轮
> 补充 4 个跨切面发现;关键发现由主审查者在源码中二次抽查。
> 配套的发散性分析见 [FUTURE_DEVELOPMENT.md](FUTURE_DEVELOPMENT.md)。
> 本报告只做分析,未修改任何现有代码。

---

## 1. 总体结论

这是一个**工程质量显著高于平均水平的代码库**。分层与框架封闭边界基本真实存在而非仅停留在
文档;数据库层防御纵深(部分唯一索引、复合同域外键、不可变触发器、角色分离)、幂等与
CAS 写入、fail-closed 配置、内容安全日志、无一处 `dangerouslySetInnerHTML`、无任何
TODO/FIXME、无已提交秘密——这些在审查中全部得到逐条验证。

审查未发现 **critical** 级问题(无可利用的安全漏洞或主路径数据损坏)。发现集中在四类:

1. **4 个 high 级问题**,全部与"移除 Unstructured 子进程隔离后未补等价护栏"以及
   多模态检索的一个键冲突有关(§4);
2. **12 个 medium 级问题**,主要是边缘路径正确性、可观测性盲区和运维韧性(§5);
3. **约 40 个 low/info 级**,以文档漂移、死配置、死代码和前端细节为主(§6);
4. 若干**结构性技术债**:分层规则靠约定而非机制守护、评测垂直是死代码、无 LICENSE(§6-§7)。

最大的系统性风险不是任何单个 bug,而是:**Worker 是单进程双 lane 共享一个事件循环和一个
单线程解析 executor,而解析路径存在三个可以独占 CPU/内存/线程的缺陷(H2/H3/H4),同时调度层
零日志、healthcheck 只检测新进程(M6/M7)——一旦卡死,聊天与索引同时静默瘫痪且外部无任何信号。**

---

## 2. 架构现状

### 2.1 运行时拓扑

模块化单体,按进程拆分部署(`compose.yaml`,全部镜像 digest 固定、非 root、仅绑定
127.0.0.1):

| 进程 | 入口 | 职责 |
| --- | --- | --- |
| API | `apps/api/main.py` | 只写持久事实(DocumentVersion/IndexingJob/ChatRun)并返回 202;终态 SSE;例外:`POST /retrieval/query` 在 API 进程内调用 embedding provider(架构文档第 10 节确认为有意设计,但与文档第 4 节"API 不执行模型调用"自相矛盾,见 L-drift) |
| Worker | `apps/worker/main.py` | 单进程双 lane(chat:indexing 默认 3:1 加权 + 30s aging),`FOR UPDATE SKIP LOCKED` 领取、lease/attempt 栅栏、心跳、有限重试;另含文件协调 janitor 循环 |
| Maintenance | `apps/maintenance/main.py` | 一次性清理:仅删除 RETIRED 且过 300s 宽限期的派生数据与过期任务事实 |
| PostgreSQL 18 + pgvector 0.8.2 | — | 唯一业务/执行事实源;1024 维文本与 768 维图文两张独立向量表,精确余弦 |
| web-chat / web-test | `apps/*/server.py` | 两个加固的纯静态服务器(CSP、无代理),React 19 SPA 直连公开 API |

### 2.2 分层(`src/rag_kb/`,147 个 Python 文件)

允许方向 `apps → services/workflows → domain + 协议`、`repositories/adapters/uow → domain`,
经 grep 逐层验证**基本属实**:`domain/` 零第三方依赖;LangChain 仅存在于
`adapters/model_api/`(3 个文件);LangGraph 仅存在于 `workflows/`(2 个文件);FastAPI 从未
进入 `src/rag_kb/`;SQLAlchemy 从未进入 domain/services/schemas/answering/memory/retrieval/
scheduling;`DoclingDocument` 只以不透明句柄形式穿过 `indexing/pipeline.py`。三个组合根
(`apps/*/dependencies.py`)为显式构造注入 + frozen dataclass + lifespan 管理,无全局状态。

已验证的例外(见 §5/§6):`langchain_chat.py:19` 向上导入 `answering.wire_schemas` 形成
顺序脆弱的包环(M-arch);`sqlalchemy_chat.py:42`/`sqlalchemy_indexing.py:67` 向上导入
`memory`/`document_processing`(L);且**全部分层规则仅有社会性约束**——唯一的机械守护是
覆盖 6 个文件的同步 SQLAlchemy AST 检查(`tests/unit/test_async_data_access.py`)。

### 2.3 各模块职责与质量印象(全部经代理通读)

| 模块 | 职责 | 审查印象 |
| --- | --- | --- |
| `apps/api/` | 薄传输层:DTO 转换、RFC-9457 problem+json、幂等(advisory lock + canonical hash,竞争安全)、上传准入(流式 10 MiB 上限、OOXML 归档校验、Base64URL 元数据) | 优秀;弱点在意外 500 的可观测性(M8)与个别输入上限缺失(L) |
| `scheduling/` + `adapters/job_queue` | 双 lane 领取/心跳/重试/对账;chat 终态写全程 lease 栅栏 | 协议正确;**零日志、异常全吞**(M6);indexing 管线体写入未带 lease 栅栏(L) |
| `indexing/` + `document_processing/` | parse-once、结构/语义断点切分、确定性 UUIDv5 身份、CAS 计划/manifest、fail-closed complete 校验 | 幂等设计教科书级;语义断点 DP 有 O(N·W) 重复 tokenize(H2),重试全量重购 embedding(M3) |
| `adapters/parser/docling/` | 内存单次转换、单线程 executor、离线内容寻址模型工件、转换后资源校验 | 工件校验与确定性极佳;**超时与内存护栏存在真实缺口**(H3/H4) |
| `retrieval/` + `adapters/vector_store/` | 单语句 lateral-join 精确余弦、双路准入、Decimal 精度 RRF、一次有界关系水合(无 N+1)、参数全绑定(无注入面) | 高质量;evidence-group 键跨文档冲突(H1)、水合与检索的版本谓词不对称(M2) |
| `workflows/` + `answering/` + `memory/` | 固定 8 节点 LangGraph、确定性证据准入、JSON 单对象提示词(摘录/历史/图片显式标注 untrusted)、严格 wire + 一次 repair + 安全回退、上下文 canonical hash 与 lease-CAS 复用 | 安全边界设计出色且大多属实;image-only citation 有一个准入-附加不一致(M1)、一个守卫路径 NameError(L) |
| `repositories/` + `uow/` + `db/` | 12 个线性迁移(head `0012` 与文档一致)、ORM/迁移逐表比对基本无漂移、单事务 UoW、全查询 workspace 限定 | 防御纵深;`updated_at` 实际从不维护(M10)、清理路径可孤儿化资产文件(M11) |
| `config/` + `observability/` | 冻结 Literal 化设置图、连接预算/时序不变量交叉校验、allowlist JSON 日志 | fail-closed 典范;代价是第三方日志被完全抹空(M8) |
| 前端 ×2 | 无 markdown-to-HTML、所有 URL 经同源 + `/api/v1` 前缀校验、SSE→轮询降级、幂等重试封装 | 干净;仅外围缺陷(诊断页"Rerank: disabled"标签错误等,L) |
| 测试 | basic 6 / unit ~296 / contract 49 / db-integration 63,手写 fake、AST 架构测试、OpenAPI 全等快照 | 质量高;盲区:maintenance 服务、评测垂直、web-chat 在 AGENTS.md 中缺席 |

---

## 3. 审查中确认的突出优点

值得点名(全部经验证,供后续演进时保持):

- **上传链路**:流式 10 MiB 硬限 → 严格 Base64URL/NFC 元数据 → 扩展名/媒体类型一致性 →
  OOXML 条目数/展开体积/加密/路径穿越拒绝 → staging(0600)→ 原子 rename → 单事务提交,
  失败态由 janitor 补偿回收。
- **文件与资产存储无调用者路径**:身份=UUID+64-hex 校验和键,`is_relative_to` 防护,下载端
  点无路径穿越;资产读取二次校验 workspace/版本绑定/SHA-256。
- **幂等索引**:确定性 UUIDv5 + `ON CONFLICT ... WHERE 内容相等` 的 insert-or-verify CAS,
  语义计划与多模态 manifest 一次持久化、重试复用不重购分析 embedding。
- **数据库先于代码执行不变量**:部分唯一索引(单活跃 revision、单 serving 版本、单非终态
  ChatRun)、复合同域外键阻断跨库拼接、document_version/source_change 不可变触发器。
- **秘密卫生**:git 全历史无密钥/env 文件;`bootstrap-roles.sql` 用 psql `%L` 引用,凭据不落
  任何被跟踪文件;`start-local.sh` umask/trap/0600 全套。
- **提示词信任标注**:整个用户回合序列化为单 JSON 对象,摘录键名即 `untrusted_excerpt`、
  历史 trust=`reference_only_untrusted`,结构性 breakout 在框架层即不可能;validator 将
  citation 限制在 envelope∩usable 交集内。

---

## 4. High 级发现(4 项,全部经对抗验证确认)

### H1 — 跨文档 evidence-group 键冲突,RRF 融合静默合并/丢弃无关证据

- 位置:`src/rag_kb/document_processing/docling/evidence.py:148`(table 组键)、`:335-339`(视觉组键);消费点 `src/rag_kb/retrieval/fusion.py:44-56`
- 事实:table 组键仅由裸 Docling ref(如 `#/tables/0`)派生,视觉组键仅由内容寻址
  `asset_key` 派生,均**不含文档命名空间**——不同于 chunk 组键(含
  `source_checksum_sha256`,`provenance.py:209-224`)。向量检索是知识库全域的
  (`pgvector.py:173-230`),因此任何知识库中≥2 个含表文档,其"第一张表"共享同一组键;
  字节相同的重复图(logo)同理。
- 影响:RRF 去重(`fusion.py` `seen_groups`)把低排名文档的命中静默折叠进另一文档的组,
  证据消失、融合分与 representation 混入无关文档——多模态 Chat 主路径,正常使用即触发。
- 建议:组键并入文档身份(与 `chunk_assembly_key` 同式),或在融合时以
  `(indexed_document_version_id, group_key)` 为去重键;补两文档同 ref 表的回归测试。

### H2 — 语义断点选择在最内层循环重复整段 tokenize,可阻塞 Worker 事件循环数分钟至数小时

- 位置:`src/rag_kb/document_processing/semantic_boundaries.py:186`;调用点 `src/rag_kb/indexing/pipeline.py:474`(同步、无 to_thread)
- 事实:`region_is_small` 只依赖 `(start, end)` 却在 DP 的每个 `(position, prior)` 对上重算,
  每次对全区间做未缓存的 tiktoken 编码(`tokenization.py:17-23` 只缓存 Encoding 对象)。
  无标题纯文本文档不产生硬边界(`semantic.py:257-281`),单区间可达 5,000 单元/50 万 token
  上限 → 数量级 5 万次全区间编码。
- 影响:`build_chunk_plan` 同步跑在 Worker 唯一事件循环上,心跳、900s deadline 定时器、
  对账和**整个 chat lane 全部冻结**;deadline 到期后 3 次重试各自重新购买最多 50 万 token
  的分析 embedding 再次冻结。选择 semantic preset 的普通几百 KB 纯文本上传就会阻塞数分钟。
- 建议:把 `region_is_small` 与区间 token 总量提出循环、维护每单元前缀 token 和;将整个
  chunk-plan 构建移入 `asyncio.to_thread`。

### H3 — 挂起的 Docling 转换永久占用单线程解析 executor;600s 转换超时对非 PDF 格式完全无效

- 位置:`src/rag_kb/adapters/parser/docling/parser.py:108`(`asyncio.shield` + `max_workers=1`,`:71-74`)
- 事实(已在安装的 docling 2.114.0 源码中核实):`document_timeout` 仅在分页 PDF 管线的
  页批次间检查(`.venv/.../docling/pipeline/base_pipeline.py:294-299`);
  MD/HTML/CSV/DOCX/PPTX/XLSX 走的 `SimplePipeline` **从不读取该配置**——
  `factory.py:56` 设置的 600s 对六种格式是死配置。调度器 900s deadline
  (`scheduling/indexing.py:190,221-228`)只能取消等待方任务;线程不可杀,shield 后转换线程
  继续运行。
- 影响:一个病态文档可无限期占用唯一解析线程;之后每个重试与所有其他文档的解析在
  executor 队列中排队、逐个 deadline 超时、耗尽 3 次重试后**整条索引 lane 静默终态失败**,
  直到进程重启——期间心跳与 healthcheck 全绿。架构文档第 9 节"Docling conversion timeout
  固定为 600 秒"与代码不符(doc-drift)。
- 建议:应用自持墙钟上限(`asyncio.wait_for` 包住 executor future,超时即标记 parser 中毒
  并重建 executor),或将转换放入可杀子进程(恢复移除 Unstructured 时失去的隔离);同步修订
  架构文档。

### H4 — 嵌入图片炸弹与密集 CSV 在应用限制生效前全量解码,进程无内存上限

- 位置:`src/rag_kb/adapters/parser/docling/parser.py:316`(40MP 校验在转换**完成后**);无任何代码设置 `PIL.Image.MAX_IMAGE_PIXELS`(全仓 grep 为零);`compose.yaml` 无任何 `mem_limit`/`pids_limit`
- 事实:docling 的 OOXML 后端无条件 PIL 解码并 PNG 重编码每张嵌入图
  (`.venv/.../msword_backend.py:2659-2667`),Pillow 默认解压炸弹阈值约 1.78 亿像素;CSV
  后端为每个单元格建一个 pydantic `TableCell`(`csv_backend.py:104-127`),准入的 10 MiB/20
  万行 CSV 可在应用的 item/字符上限运行前膨胀出数 GB 对象。
- 影响:一个通过准入的 10 MiB 上传可把单进程 Worker 推向 OOM——**chat lane 一同阵亡**,
  容器 `restart: unless-stopped` 复活后在同一文档上重复(有限 3 次)。无容器内存限制时还会
  拖垮同主机的 PostgreSQL/API。
- 建议:在解析线程内把 `PIL.Image.MAX_IMAGE_PIXELS` 设为自有 40MP 上限(炸弹→早期
  `PARSER_RESOURCE_LIMIT`);对 OOXML media 部件做转换前廉价尺寸预检;为 worker 服务加
  `mem_limit`(实测解析 RSS 峰值 2.6 GB + 余量)与 `pids_limit`,让炸弹只杀死作业而非主机。

---

## 5. Medium 级发现(12 项)

| # | 发现 | 位置 | 要点 |
| --- | --- | --- | --- |
| M1 | image-only citation 被跳过(max_images 截断/top-4 截断/去重)时**未从 usable 集合移除**,模型被要求引用从未收到的证据,正文只剩 `[image visual evidence]` 占位符 | `services/chat_visuals.py:102`、`services/visual_admission.py:236-247` | 与代码在迭代路径上自我执行的不变量矛盾(`test_over_budget_native_only_image_is_removed_and_refused` 正是此约定);≥3 个准入纯图命中即可触发。修复:按"实际附加了什么"反推保留集 |
| M2 | 关系水合额外要求 `current_version_id` 匹配而向量检索不要求;新版本上传后的整个重索引窗口(或索引失败后永久),旧 serving 版本仍被检索但**水合零关系**,视觉证据静默消失 | `repositories/sqlalchemy_indexing.py:228` vs `adapters/vector_store/pgvector.py:254-256,359` | `is_current_serving_version` 被硬编码 True,使 `_validate_scope` 自我认证。两条读路径谓词需一致 |
| M3 | 重试无断点续传:任意中途失败后全量重解析+重购全部 embedding,且向量 upsert 的 CAS 要求 provider 复现**位相同浮点**,漂移一位即 `INDEX_PERSISTENCE_FAILED`,3 次重试全灭 → 文档永久 FAILED | `repositories/sqlalchemy_indexing.py:1603-1617`、`indexing/pipeline.py:188-216` | 失败关闭无损坏,但把瞬态故障放大为确定性失败 + 3 倍 embedding 费用 |
| M4 | 文本 embedding 适配器外层 `asyncio.timeout` 与单次请求超时同为 30s,SDK 的 `max_retries=2` 对慢响应/429 退避场景形同虚设;失败逐级放大为整个索引作业重跑 | `adapters/model_api/langchain_embeddings.py:55-56,111` | 外层预算应为 `timeout×(retries+1)+退避`,或收回重试自管 |
| M5 | 扫描页探测对整本 PDF 同步跑 pypdf `extract_text`(至多 500 页),在事件循环上、无超时、绕过解析 executor | `adapters/parser/scanned_pages.py:26-31`、`pipeline.py:693` | 与 H2 同类的事件循环冻结,但输入是外来字节。移入 executor + 每页首个非空白字符即止 |
| M6 | 整个 `scheduling/` **没有一行日志**:reconcile 异常被 `return_exceptions=True` 丢弃、claim/心跳异常静默吞掉、execute 逃逸异常被 `_reap` 吸收 | `scheduling/worker.py:68-71,131-134,161`、`chat.py:169-170`、`indexing.py:269-270` | 运行角色权限回退、schema 漂移、连接池耗尽 → Worker 永远空转零输出。janitor 有日志,证明这是遗漏而非内容安全策略 |
| M7 | Worker healthcheck 起**新进程**校验 DB 可达,不观测服务循环;调度任务死亡后 `serve()` 继续 `stopped.wait()`,容器保持绿色 | `compose.yaml:116-124`、`apps/worker/main.py:59-68` | 与 H3/M6 叠加 = 全静默瘫痪。方案:调度循环触碰心跳文件供 `--check` 校验新鲜度;`asyncio.wait(FIRST_COMPLETED)` + 任务死亡即退出非零 |
| M8 | 意外 500 与 API 启动失败完全不可诊断:catch-all handler `del error` 不记日志;`ContentSafeJsonFormatter` 丢弃所有非 log_event 记录的 message 与 traceback,uvicorn lifespan 失败只剩 `{"event":"external_log"}` | `apps/api/errors.py:335-347`、`observability/logging.py:48-57`、`apps/api/main.py:43` | `error_type` 本就在 SAFE_FIELDS 白名单内却未使用;容器只会无解地重启循环 |
| M9 | 全栈无 `statement_timeout`/`lock_timeout`/`idle_in_transaction` 且 API 检索路径无 deadline;配置只预算了连接**数量**,没预算持有**时长** | `db/session.py:56-64`(全仓 grep 零命中) | 大库精确扫描 + 与 promotion 锁碰撞 → 少量慢查询悄悄耗尽 API 池拖垮全部端点,叠加 M8 无任何征兆 |
| M10 | `updated_at` 实际从不由服务端维护(`server_onupdate` 不生成 DDL、无触发器),chat 仓储的 claim/complete/settle 均不手动赋值;sessions 端点默认 `-updated_at` 排序**退化为创建序**,web-chat 会话列表不随活动上浮 | `db/models.py:134-140`、`repositories/sqlalchemy_chat.py`、`apps/api/routers/chat.py:87` | 加 BEFORE UPDATE 触发器或在全部状态迁移显式赋值 |
| M11 | Maintenance 清理以**资产数**为一批列文件、以**目标数**为一批删行,两种排序/上限不一致;单版本资产数超过 batch_size(默认 100,单文档上限 1000)时,行被删而文件未删,且资产店无 list/扫描能力 → **派生文件永久孤儿** | `services/maintenance.py:48-52`、`sqlalchemy_indexing.py:430-484,554-563` | 行删除应以文件删除成功为前提,或给 IndexAssetStore 补 orphan sweep |
| M12 | adapter 向上导入 `answering` 形成顺序脆弱包环:`answering/__init__ → pipeline_steps → adapters/model_api/__init__ → langchain_chat → answering.wire_schemas`,当前仅因 `__init__` 内 import 顺序侥幸成立 | `adapters/model_api/langchain_chat.py:19`、`adapters/model_api/__init__.py:3-8` | 构造时注入 schema registry,或把 wire-schema 身份下沉至 domain |

---

## 6. Low / Info 级发现(按主题归并,~40 项)

### 6.1 文档漂移(最大集群,均已核实两侧文件)

> 后续状态（2026-07-28）：当前代码复核见
> [实施计划](docs/implementation-plans/2026-07-28-resolve-documentation-drift.md)。
> 下表八项中七项在复核时仍存在并已完成清理；“600 秒转换超时仅对 PDF 有效”已先由
> 可终止 Docling 子进程和父进程全格式 wall-time 修复。下表继续保留 2026-07-26
> 审查基线下的原始发现，不作为当前架构来源。

| 文档侧 | 代码侧事实 |
| --- | --- |
| `docs/release/known-limitations.md:4`、`capability-matrix.md:6`:"仅支持 UTF-8 .txt/.md" | `services/admission.py:22-30` 实际准入 8 种格式 |
| `README.md:28`:"两个 configuration-only 回退开关" | 开关已移除且被 `extra="forbid"` 主动拒绝(`test_settings.py:273` 证明);指向的指南自己说的就是相反内容 |
| CLAUDE.md:111 与架构文档 §4:"API 从不调用模型 provider" | `POST /retrieval/query` 在 API 进程内调用 embedding(`retrieval/service.py:410-412`);架构文档 §10 又将其记为有意设计——三处自相矛盾 |
| OpenAPI 上传契约(`routers/documents.py:41` + `tests/contract/snapshots/openapi-v1.json`)只声明 4 种媒体类型 | 准入 8 种;契约快照把漂移锁死 |
| `.env.example:80` 仍称 Docling 为 "shadow parser" | Docling 已是唯一 parser |
| `AGENTS.md:5` 只提 `apps/web-test` | `apps/web-chat` 是 compose 接线的正式前端,截图规则也漏掉它 |
| CLAUDE.md:57 集成测试命令只导出 2 个 DSN 变量 | `test_schema.py:19` 需要第 3 个 `RAG_KB_TEST_RUNTIME_DSN`,按文档执行会静默跳过含"运行角色禁 DDL"在内的全部 6 个 schema 测试 |
| 架构文档 §9 "600 秒转换超时" | 对非 PDF 格式是死配置(见 H3) |

### 6.2 死配置 / 死代码

- `max_ocr_characters`/`max_ocr_tokens`:配置、校验、接线俱全但**无任何代码读取**(`domain/parsing.py:34`,全仓 grep 仅定义与接线)。
- `tracing_enabled`:接受 `true` 但零消费(`settings.py:572`);违反本仓"未实现开关应 fail closed"的自有惯例(同文件 `hnsw_enabled` 等均为 Literal[False])。
- 评测垂直(`services/evaluation.py`、283 行 `sqlalchemy_evaluation.py`、4 张 `eval_*` 表):零调用者、零测试;`tools/evaluate_multimodal_real.py` 并不使用它。
- `scheduling/indexing.py:104` 的 `IndexingJobScheduler.run()`:与 FairWorkerScheduler 并行的第二套派发循环,仅测试使用,生产死路径。
- 空占位包 `adapters/event_stream/`、`adapters/job_queue/`,前者 docstring 还引用已退休的 "P1B" 阶段标签(违反仓库自己的文档协议)。
- `answering/pipeline_steps.py:207` 的 `EvidenceCoverage.PARTIAL`/`AMBIGUOUS` 路由不可达(评估器只产出 NONE/SUFFICIENT)——partial 目前只能由模型自行选择,结构上暗示的"确定性部分覆盖评估"并不存在(info,建议文档化)。

### 6.3 正确性/安全小项

- `answering/pipeline_steps.py:263`:`ErrorCode` **未导入**,三条守卫路径(policy 校验、输入校验、assessment 状态)触发时抛 `NameError` 而非预期的 `CHAT_CONTEXT_INVALID`,被 runner 重分类为 `CHAT_PIPELINE_STEP_FAILED` 并丢失 check 名——守卫路径无测试所以存活至今(本审查已在源码二次确认)。
- 显示名/文件名校验只拒 Cc/Cs,放行 Cf(U+202E RLO 等 bidi 控制符)→ 列表/引用中可视觉伪装文件名(`routers/documents.py:429`、`admission.py:59-63`)。
- 检索 query 字符串无 max_length、JSON 端点无请求体上限(`schemas/retrieval.py:26`),与全表面其他输入(消息 32768、游标 2048、上传 10 MiB)不一致。
- indexing 管线体 DB 写入仅按状态栅栏,不带 lease(与 chat lane 的 `_owns_running_lease` 不对称);默认 `indexing_concurrency=1` 下窗口极窄,但 fail() 可在重排队窗口内把作业错误终态化(`sqlalchemy_indexing.py:1874-1955`)。
- 迁移 0008/0010 建表继承 0001 的默认特权却未像 0006 那样 `REVOKE UPDATE`,运行角色对三张"insert-only"表(manifest/relation/space)静默保有 UPDATE;`database-schema.md:190-195` 的"只授予所需操作"表述不实。
- `.dockerignore` 只排除字面 `.env`:`.env.local` 与含真实 provider key 的 `.env.bak-20260726-220238` 会进入构建上下文(当前 Dockerfile 只 COPY 白名单路径,故为潜伏而非现实泄露);另外 `COPY apps` 把 `apps/web-test/node_modules`(103 MB)烘进镜像。
- `start-local.sh:137-139` 默认路径给超级用户/迁移/运行三角色同一随机口令,削弱 bootstrap-roles.sql 建立的分离(loopback-only 下影响边际)。
- `compose.yaml:29,42`:不带 `--env-file` 的裸 `docker compose up` 可用公开已知口令 `replace-locally` 启动(建议 `${VAR:?}` 失败关闭)。
- Janitor 每 30s 对全部可用源文件做**全量 SHA-256 重哈希**(O(语料字节) 每 30 秒);孤儿扫描的保护集只含最老 100 条 pending mutation,积压超过时可能删除仍被引用的 staged 文件(概率低但为静默永久丢失)。
- 多模态 embedding 适配器对 429/5xx **零退避立即重试**且把 409 也当可重试(`multimodal_embeddings.py:133`);语义分析 embedding 在"存在硬边界"时就全量购买,即使所有区间都无需软边界决策(`pipeline.py:469`)。

### 6.4 前端与其他

- 诊断页锁定设置面板硬编码 "Rerank: disabled",而请求发送 `rerank: true` 且服务端确实执行 hybrid rerank——同页下方 query-plan 显示的又是真值,自相矛盾(`RetrievalView.tsx:92`)。
- web-chat 的 SSE settle 缺少 web-test 已有的非终态回退守卫(`App.tsx:285` vs `ChatView.tsx:288`);两个同协议实现不一致。
- 静态服务器无 per-connection 超时(slowloris;loopback 发布下有界)且默认绑定 `0.0.0.0`(容器内需要,裸跑时暴露 LAN)。
- 证据抽屉卡片仅鼠标可选、抽屉缺 dialog/aria-modal 语义(键盘/读屏可达性)。
- 6 个 `.DS_Store` 被跟踪且 `.gitignore` 无规则;`tests/unit/test_maintenance.py` 实际测试的是 `tools.reset_local`(命名误导);本地残留孤儿 `deploy/operations_provider/__pycache__/*.pyc`(未跟踪,建议删除)。
- **无任何 LICENSE 声明**(仓库、pyproject、两个 package.json 均无)——默认全权保留;若"企业知识库"要分发,这是合规缺口。顶层依赖无 copyleft(docling MIT、torch BSD-3、rapidocr Apache-2.0),传递树未离线核验。
- `tools/evaluate_multimodal_real.py` 成本控制总体良好(强制 loopback、6 文件有界语料、top_k≤20),但 5 个 provider 计费 Chat 用例**默认执行**,需记得 `--skip-chat`(建议反转默认)。

---

## 7. 风险汇总(哪些最可能咬人)

1. **单 Worker 静默瘫痪**(H2+H3+H4 × M6+M7):三个独立的"独占资源"缺陷,叠加零日志
   调度层与不观测服务循环的 healthcheck。这是唯一能同时打掉聊天与索引且外部无信号的组合。
2. **多模态证据质量静默劣化**(H1+M1+M2):三个互不相关的机制都以"不报错地少给/错给
   证据"收场——对一个以可审计证据链为核心卖点的系统,这类静默劣化比崩溃更伤。
3. **重试即放大**(M3+M4+6.3 零退避):瞬态 provider 故障被架构性放大为全量重跑、
   全额重购,最坏永久 FAILED。付费 API 语境下既是可用性也是成本风险。
4. **运维盲区**(M8+M9):意外失败无诊断信息 + 无任何时长护栏,故障发生时定位全靠猜。
5. **文档漂移积累**(6.1):本仓的核心工作流是"文档即事实源供 CODE AGENTS 使用",
   release 目录与 README 的滞后会直接误导下一轮自动化开发。

---

## 8. 改进建议与实施顺序

按"先止血、再还债、后打磨"排序;同一批内条目相互独立可并行。每批建议按仓库规范建立
dated plan 并同步架构文档。

### P0 — 立即(护栏与静默故障,预计 2-3 个工作项)

1. **解析护栏包**(对应 H3/H4/M5):`asyncio.wait_for` 包住转换 + executor 中毒重建;
   解析线程内设 `PIL.Image.MAX_IMAGE_PIXELS`;`scanned_surfaces` 移入 executor 并加预算;
   compose 为 worker 加 `mem_limit`/`pids_limit`。这一包恢复移除 Unstructured 隔离时
   失去的等价保护,应作为单一计划实施。
2. **可观测性止血**(对应 M6/M7/M8):scheduling 层补结构化日志(helper 已存在);
   catch-all 500 记录 `error_type`+trace_id;formatter 对 `exc_info` 至少输出异常类名;
   healthcheck 改为观测服务循环心跳。全部是小改动,合为一个计划。
3. **H2 性能修复**:循环不变量提升 + 前缀 token 和 + `to_thread`——几十行改动消除
   "普通上传冻结 Worker 数分钟"。

### P1 — 短期(正确性,预计 3-4 个工作项)

4. **H1 组键命名空间化** + 回归测试(触碰索引身份,需新 revision 兼容性评估,单独计划)。
5. **M1 视觉 citation 一致性**:按实际附加集合反推 usable;补三图两预算回归测试。
6. **M2 读路径谓词统一** + `is_current_serving_version` 真实计算。
7. **重试经济性**(M3/M4/6.3 退避):断点续传(已存在的稳定 ID 使跳过已持久化批次很容易)、
   向量 CAS 放宽为同 ID 覆盖、外层超时预算修正、多模态适配器加抖动退避。
8. **小正确性集**:`ErrorCode` 导入 + 守卫路径测试;`updated_at` 维护;M11 清理顺序;
   Cf 字符拒绝;检索 query max_length。

### P2 — 中期(债务与卫生)

9. **文档漂移一次性清账**(6.1 全部)+ 把 `docs/release/` 纳入架构文档 §18 同步矩阵。
10. **机械化分层守护**:一个 AST 测试断言允许的跨包导入边(模式已在
    `test_async_data_access.py` 存在);同时解开 M12 包环。
11. **死代码/死配置清理**:评测垂直(补最小测试或降级为纯 schema)、空占位包、
    `IndexingJobScheduler.run`、两个 OCR 死配置、`tracing_enabled` 改 DisabledFlag。
12. **部署卫生**:`.dockerignore` 改 `.env*`+`node_modules`;compose 口令 `:?` 失败关闭;
    三角色独立口令;`.DS_Store` 清理;LICENSE 与第三方声明基线。
13. **M9 时长护栏**:按进程差异化 `statement_timeout`/`lock_timeout`;API 检索加 deadline。
    (放 P2 是因为本地单用户下风险有限,共享部署前必须完成。)

### 与未来规划的衔接

P0-1 的"可杀子进程"方案与 [FUTURE_DEVELOPMENT.md](FUTURE_DEVELOPMENT.md) §2.4/§2.5 的
"索引/聊天 Worker 分离"同向——若 3-6 个月内计划做 Worker 分离,P0-1 可先做最小
wait_for+中毒标记版本,把子进程隔离留给分离工作。H1 修复涉及索引身份,宜与任何
re-index 类功能(如 HNSW 引入前的重建)合并窗口。

---

## 9. 本次审查执行的验证与未执行项

### 已执行

| 验证 | 结果 |
| --- | --- |
| `PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/basic` | **6/6 通过**(1.24s) |
| 秘密卫生:`git ls-files` env 类文件 + 全历史文件名扫描(近 60 提交) | 仅 `.env.example` 入库;历史无密钥/env 材料 |
| 迁移 head 与文档/`compatibility.py` 三方一致性 | `0012_citation_document_names` 三方一致 |
| ORM ↔ 迁移逐表比对(代理执行) | 仅命名级漂移(§6;vector_1024 未命名唯一约束、`server_onupdate` 空标记) |
| 关键发现源码二次抽查(H1 组键、ErrorCode 未导入) | 属实 |
| 架构文档 18 节逐节通读 + 8-10 项硬声明抽查(节点顺序、阈值、表集、端点、配置键) | 除 §6.1 所列漂移外全部与代码一致 |

### 未能执行(如实披露)

- `tests/unit`(~296)与 `tests/contract`(49)套件:本会话的命令安全分类器在执行窗口内持续
  不可用,python 运行命令被反复拒绝(basic 套件是在一次短暂恢复窗口跑通的)。两套件
  最近一次全绿记录见 EXECUTION-TRACKER(2026-07-26,unit 285/contract 49 时点)。
- `tests/integration/db`(63):需要导出真实 PostgreSQL DSN,未提供则按设计自跳过。
- 前端构建:`apps/web-chat` 无 `node_modules`,按仓库"不擅自安装依赖"规则未执行
  `npm install`;`apps/web-test` 构建因上述分类器问题未能执行。
- 需要运行栈的验证:锁竞争、真实语料下精确扫描延迟、SSE 关停行为、FS↔DB 崩溃恢复竞态
  ——均为静态推理。
- 付费 provider 路径(Qwen chat、text/multimodal embedding)按用户预算要求**未调用**;
  `tools/smoke_local.py`、`tools/evaluate_multimodal_real.py` 未运行。
- Docker 构建未执行(`.dockerignore` 泄露路径为静态推断);git 对象级历史扫描(大二进制/
  秘密 blob)因分类器不可用仅完成文件名级扫描;传递依赖 license 清单未解析;Docling 模型
  工件 manifest 未联网复验。

---

## 10. 审查覆盖度说明

10 个子系统审查代理各自通读所辖目录(架构/分层、API、Worker/调度、索引/检索、
Chat/回答、解析/文档处理、持久化、配置/部署、前端、测试/工具/文档/债务),合计约 2.4M
token 的阅读与交叉验证;30 个 critical/high/medium 候选发现逐条交由独立验证代理以
"试图驳倒"的立场复核(结果:0 驳回,但 10 项被降级、若干项被修正细节——降级理由已并入
上文各条);最后由完整性批判代理专门搜寻十个视角之间的缝隙(产出 M7/M9、内存上限、
LICENSE 四项)。已知盲区见 §9"未能执行"。
