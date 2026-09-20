<div align="center">
  <img src="assets/readme/hero.svg" alt="Evidex —— 本地知识库：导入文档，获得带引用的回答" width="100%" />

  <p>
    <img src="https://img.shields.io/badge/Python-3.12-3776AB" alt="Python 3.12" />
    <img src="https://img.shields.io/badge/FastAPI-异步%20API-009688" alt="FastAPI 异步 API" />
    <img src="https://img.shields.io/badge/PostgreSQL%2018-%2B%20pgvector-336791" alt="PostgreSQL 18 + pgvector" />
    <img src="https://img.shields.io/badge/React%2019-TypeScript-149ECA" alt="React 19 + TypeScript" />
    <img src="https://img.shields.io/badge/Docker%20Compose-单机部署-2496ED" alt="Docker Compose 单机部署" />
  </p>
</div>

<p align="center">
  <a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a>
</p>

**Evidex** 是一个本地优先、单用户的个人知识库和证据约束 RAG 应用。导入文档后，Worker 会完成解析、切分和索引；你可以通过对话提问，并获得带有行内引用的回答，每个事实性结论都能回溯到准确的源文档片段。所有文件、向量、图谱和聊天记录都留在本机，通过一个 Docker Compose 项目运行。

> **产品范围：** 这是个人单机应用，不是多租户服务。当前没有生产级认证或隔离能力，必须保持在本机回环地址上运行。详见[安全边界](#安全边界)。

## 截图

<table>
  <tr>
    <td width="50%">
      <img src="assets/readme/chat-agent-trace.jpg" alt="带有模型轮次、并行工具调用、来源和耗时的回答时间线" />
    </td>
    <td width="50%">
      <img src="assets/readme/chat-source-drawer.jpg" alt="展示引用背后准确检索片段的来源抽屉" />
    </td>
  </tr>
  <tr>
    <td><sub>回答过程时间线：展示每次工具调用、输入、来源、结果数量和耗时。</sub></td>
    <td><sub>引用来源抽屉：打开引用即可查看其背后的准确片段。</sub></td>
  </tr>
  <tr>
    <td colspan="2">
      <img src="assets/readme/kb-scope-picker.png" alt="支持多知识库范围选择的聊天输入框" />
      <br/><sub>多知识库对话：每次会话可以组合选择知识库，运行期间范围保持冻结。</sub>
    </td>
  </tr>
</table>

## 核心能力

### 文档导入与索引

- 支持 TXT、Markdown（包括带本地图片的 `.mdz` bundle）、HTML、CSV、PDF、DOCX、PPTX 和 XLSX。
- PDF 在 Worker 管理的可终止子进程中按页段处理，支持 OCR、版面和表格模型，并带有有界进度检查点。
- 支持结构化或语义切分、表格行感知切分，以及页面/图片/表格等可选视觉资产。
- 每个知识库可以选择纯文本、双空间或统一多模态向量，并严格校验向量维度。
- 可选 Auto-QA 索引：生成的问题必须通过源片段校验，只用于增加召回候选，不会成为证据、引用或图谱 episode。
- 失败候选会从源文件重试并清理；对外提供的索引不会处于半发布状态。

### 检索与回答

- 默认使用精确 pgvector 余弦检索；可选 PostgreSQL FTS，并通过确定性的 RRF 合并密集和词法结果。
- `iterative_balanced` 策略可以基于已获得证据中的引用片段执行有限的补充检索轮次。
- 支持 `none`、`classic`、`local_minilm_v1` 三种重排模式；MiniLM 重排器完全离线运行。
- 表格补全只拉取同一文档版本中的相邻片段，并保持结构锚点与得分隔离。
- 原生异步工具调用循环执行检索和计算工具；最终回答只能引用已验证且位于本次运行范围内的证据。
- 没有有效引用时，运行会明确拒答，不会用猜测补齐答案。

### 可选知识图谱

- 使用 Graphiti + FalkorDB 构建可版本化的知识图谱；新构建完成并通过覆盖率和运行时探针后才会原子切换。
- 每个 serving chunk 对应一个确定性的 Graphiti episode，重试不会重复调用模型。
- 内置通用、软件知识和企业知识三类强类型 schema profile。
- 图谱检索返回完整、无环且能映射回当前 serving chunk 的路径；任何不完整路径都会被拒绝。

### 多知识库与可观测性

- 每次聊天可以组合选择多个知识库，并冻结知识库集合、索引版本、检索配置和图谱构建版本。
- 引用保留知识库名称和版本快照，即使原始片段后来被删除也能追溯。
- 提供本地 doctor 预检、Worker 心跳、结构化 JSONL 日志和安全诊断包。
- 测试入口支持离线检查；数据库集成测试使用临时回环 PostgreSQL。

## 为什么不同

| 常见做法 | Evidex |
| --- | --- |
| 把 top-k 片段直接塞进 Prompt | 每条结论都必须满足证据契约；无效引用会从正文和引用集合中同时剔除 |
| Agent 框架黑盒 | 一个普通的异步工具调用循环，预算、轮次和活动轨迹都可检查、可持久化 |
| 默认开启一个含义模糊的 Hybrid | 精确向量检索是默认值；FTS 是显式开关，词法结果不会冒充余弦分数 |
| 把向量库当成事实源 | PostgreSQL 保存事实、任务、引用和图谱状态；向量与图谱都是可重建索引 |
| 升级时全量重建并中断服务 | 不可变索引版本和图谱代际在后台构建，完成后原子切换 |
| 把 Provider 配置散落在环境变量里 | Provider 在 Web Chat 中配置和校验，密钥只保存在未提交的本地存储中 |

## 架构

<img src="assets/readme/architecture.svg" alt="Evidex 运行时架构" width="100%" />

系统由一个 Compose 项目 `rag` 管理：PostgreSQL 18 + pgvector、FalkorDB、FastAPI API、单 Worker、一次性初始化/迁移/维护任务，以及 React Web Chat 前端。核心原则如下：

- API 只负责快速写入持久事实；模型调用和文档解析由 Worker 完成。
- 任务和运行通过递增的 `attempt` token 与 compare-and-set 更新管理，过期 Worker 不能覆盖新的终态。
- 每次数据库操作都是短事务；外部模型和文件 I/O 不会持有数据库事务。
- PostgreSQL 是业务和任务状态的事实源；本地卷保存源文件和派生资产。

### 技术栈

| 领域 | 选型 |
| --- | --- |
| 后端 | Python 3.12、FastAPI、Pydantic、异步 SQLAlchemy、asyncpg、Alembic |
| 存储 | PostgreSQL 18 + pgvector、PostgreSQL FTS、FalkorDB（Graphiti） |
| 解析 | Docling、RapidOCR、确定性的分段 PDF 管线 |
| 对话执行 | 原生异步工具调用循环；通过适配器接入聊天模型 |
| 模型 | OpenAI-compatible 聊天/文本向量、Tongyi 多模态向量、离线 MiniLM 重排器 |
| 前端 | React 19、TypeScript、Vite（`apps/web-chat`） |
| 运维 | Docker Compose、结构化 JSONL 日志、安全诊断 |

### 仓库结构

```text
apps/
  api/                 FastAPI HTTP 层
  worker/              Worker 调度与装配
  maintenance/         本地清理命令
  web-chat/            React 聊天前端

src/rag_kb/
  domain/              业务类型、状态和错误
  schemas/             对外 API DTO
  services/            用例与跨资源协调
  answering/           原生工具调用循环、证据校验和渲染
  document_processing/ 文档解析与切分
  indexing/            索引管线
  graph/               Graphiti 构建和 schema profile
  retrieval/           检索与融合
  scheduling/          PostgreSQL 任务调度
  ports/               外部 I/O 协议
  adapters/            文件、模型和存储适配器
  repositories/ uow/ db/ config/ observability/

tools/                 doctor、smoke、reset 和本地测试工具
tests/                 basic / unit / contract / integration 测试
docs/                  唯一的最终架构文档
```

## 快速开始

### 环境要求

- Docker Compose
- Python 3.12.13（用于 doctor 和测试；应用本身运行在容器内）
- Node 24 / npm 11（仅在 Docker 外开发前端时需要）

### 1. 创建本地配置

```bash
cp .env.example .env.local
# 编辑 .env.local，为三个数据库密码设置一致且仅本机使用的值
```

`.env.local` 是被 Git 忽略的本地配置文件，包含 Compose 身份、回环端口、数据库初始化凭据和 `RAG_KB__...` 应用设置。模型 Provider 不从该文件读取，而是在 Web Chat 中配置。

### 2. 准备宿主机工具链并运行 doctor

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev-macos.lock
PYTHONPATH=src:. .venv/bin/python tools/local_runtime.py doctor
```

### 3. 构建并启动

```bash
make up
```

`make up` 先运行 doctor 预检，再依次校准数据库角色、构建带 Git revision 标签的镜像、执行 Alembic 迁移并等待各服务健康；已有数据卷不会被重置。其余生命周期命令见 `make help`（`down`、`ps`、`logs`、`migrate`、`smoke` 等）。

### 4. 打开应用

- Web Chat：<http://127.0.0.1:3000>
- API 文档：<http://127.0.0.1:8000/api/v1/docs>

### 5. 配置模型并开始提问

启动后打开 Web Chat 的模型设置，添加并验证 OpenAI-compatible Provider、聊天模型和文本向量模型；如有需要，再添加 Tongyi 多模态向量模型。随后创建知识库、上传文档，等待索引完成即可提问。

## 配置要点

| 配置 | 默认值 | 作用 |
| --- | --- | --- |
| `RAG_KB_API_PORT` / `RAG_KB_FRONTEND_PORT` / `RAG_KB_POSTGRES_PORT` / `RAG_KB_FALKORDB_PORT` | `8000` / `3000` / `5432` / `6379` | 回环端口 |
| `POSTGRES_ADMIN_PASSWORD` / `RAG_KB_MIGRATION_PASSWORD` / `RAG_KB_RUNTIME_PASSWORD` | `replace-locally` | 数据库角色初始化凭据，必须替换 |
| `RAG_KB__RETRIEVAL__HYBRID_ENABLED` | `false` | 启用词法检索和 Hybrid 回填 |
| `RAG_KB__RETRIEVAL__MIN_COSINE_SIMILARITY` / `MIN_RERANK_SCORE` | `0.35` / `0.45` | 候选准入门槛 |
| `RAG_KB__JOB_POLLER__CHAT_DEADLINE_SECONDS` | `600` | 单次 ChatRun 绝对截止时间 |

## 对话模式与检索策略

**对话模式**（每次运行选择，默认 `auto`）：

- `text`：仅使用文档片段工具。
- `auto`：当存在 READY 图谱构建时，同时提供图谱工具，由模型决定是否调用。
- `graph`：优先检索完整的 1–3 跳图谱路径，再用 Hybrid 文档证据补齐预算。

**检索策略**：

- `exact_vector`：当前 serving revision 上的单轮精确 pgvector 余弦检索，默认值。
- `hybrid`：密集向量和 PostgreSQL FTS 通过确定性 RRF 合并。
- `iterative_balanced`：在 Text 模式中基于已有证据片段进行最多两轮额外检索。

重排模式为 `none`、`classic` 和 `local_minilm_v1`。

## 测试

```bash
# 基础测试：导入和架构边界
PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/basic -v

# 单元测试与契约测试
PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/unit -v
PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/contract -v

# 数据库集成测试
PYTHONPATH=src:. .venv/bin/python tools/run_database_tests.py

# 前端
cd apps/web-chat && npm ci && npm test && npm run build

```

## 安全边界

Evidex 设计为**一台机器上的一个用户**：

- 所有服务绑定回环地址；没有生产认证、OIDC/OAuth、RBAC/ACL、多租户或审计能力，不要暴露到公网。
- Provider 密钥、DSN 和本地密码只应保存在未提交的本地文件或 Docker 私有卷中。
- 文档正文、聊天历史、检索片段和上传图片都被视为不可信内容，不能扩大权限、引用范围或系统指令。
- `.env.local`、`.runtime/`、数据库文件、模型密钥和构建产物均不应提交到 Git。

## 相关文档

- [架构记录](docs/architecture.md)

## License

当前仓库尚未添加 LICENSE，代码默认保留作者全部权利。如需公开分发或接受贡献，请先补充合适的许可证文件。
