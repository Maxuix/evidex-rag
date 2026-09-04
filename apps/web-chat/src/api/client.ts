import type {
  ApiProblem,
  ChatMessage,
  ChatProgressSnapshot,
  ChatRun,
  ChatRunCreate,
  ChatSession,
  ChatTerminalEvent,
  ChunkingPreset,
  DocumentChunkInspection,
  DocumentDetail,
  DocumentRecord,
  DocumentUpload,
  GraphConfig,
  GraphConfigUpdate,
  GraphSchemaProfile,
  IndexingJob,
  KnowledgeBase,
  KnowledgeBaseEmbeddingSelection,
  ModelKind,
  ModelCatalog,
  ModelProfile,
  ModelProvider,
  ModelProviderProtocol,
  ModelSelection,
  ModelSettings,
  Page,
  ParsingPreset,
  RetrievalEvidencePack,
  RerankMode,
  UUID,
} from "./types";
import { UI_POLICY } from "../uiPolicy";

const API_PATH = "/api/v1";
const OOXML = "application/vnd.openxmlformats-officedocument";
const UPLOAD_MEDIA_TYPES: Record<string, string> = {
  txt: "text/plain",
  md: "text/markdown",
  mdz: "application/vnd.rag-kb.markdown-bundle+zip",
  html: "text/html",
  csv: "text/csv",
  pdf: "application/pdf",
  docx: `${OOXML}.wordprocessingml.document`,
  pptx: `${OOXML}.presentationml.presentation`,
  xlsx: `${OOXML}.spreadsheetml.sheet`,
};

interface RuntimeConfig {
  api_base_url: string;
}

export class ApiClientError extends Error {
  readonly status: number | null;
  readonly code: string | null;
  readonly retryable: boolean;
  readonly traceId: string | null;
  readonly fieldErrors: ApiProblem["errors"];

  constructor(
    message: string,
    options: {
      status?: number;
      code?: string;
      retryable?: boolean;
      traceId?: string;
      fieldErrors?: ApiProblem["errors"];
    } = {},
  ) {
    super(message);
    this.name = "ApiClientError";
    this.status = options.status ?? null;
    this.code = options.code ?? null;
    this.retryable = options.retryable ?? false;
    this.traceId = options.traceId ?? null;
    this.fieldErrors = options.fieldErrors ?? null;
  }
}

export async function loadRuntimeConfig(): Promise<RuntimeConfig> {
  const response = await fetch("/runtime-config.json", { cache: "no-store" });
  if (!response.ok) {
    throw new ApiClientError("无法读取本地运行配置。", {
      status: response.status,
    });
  }
  const value = await response.json() as Partial<RuntimeConfig>;
  if (typeof value.api_base_url !== "string") {
    throw new ApiClientError("本地运行配置缺少 API 地址。");
  }
  validateApiBaseUrl(value.api_base_url);
  return { api_base_url: value.api_base_url.replace(/\/$/, "") };
}

export class ApiClient {
  readonly apiBaseUrl: string;
  readonly apiOrigin: string;

  constructor(config: RuntimeConfig) {
    const parsed = validateApiBaseUrl(config.api_base_url);
    this.apiBaseUrl = parsed.toString().replace(/\/$/, "");
    this.apiOrigin = parsed.origin;
  }

  listKnowledgeBases(cursor?: string): Promise<Page<KnowledgeBase>> {
    return this.request(this.withQuery("/knowledge-bases", {
      limit: String(UI_POLICY.knowledgeBasePageSize),
      sort: "name",
      cursor,
    }));
  }

  createKnowledgeBase(
    name: string,
    parsingPreset: ParsingPreset,
    chunkingPreset: ChunkingPreset,
    embedding: KnowledgeBaseEmbeddingSelection,
    idempotencyKey: UUID,
    autoQa: { enabled: boolean; model_profile_revision_id?: UUID | null } = {
      enabled: false,
    },
  ): Promise<KnowledgeBase> {
    return this.request("/knowledge-bases", {
      method: "POST",
      headers: this.jsonHeaders(idempotencyKey),
      body: JSON.stringify({
        name,
        parsing: { preset: parsingPreset },
        chunking: { preset: chunkingPreset },
        embedding,
        auto_qa: autoQa.enabled
          ? {
            enabled: true,
            model_profile_revision_id: autoQa.model_profile_revision_id,
          }
          : { enabled: false },
      }),
    });
  }

  deleteKnowledgeBase(kbId: UUID, idempotencyKey: UUID): Promise<{
    id: UUID;
    name: string;
    deleted_at: string;
  }> {
    return this.request(`/knowledge-bases/${kbId}`, {
      method: "DELETE",
      headers: { "Idempotency-Key": idempotencyKey },
    });
  }

  listDocuments(kbId: UUID, cursor?: string): Promise<Page<DocumentRecord>> {
    return this.request(this.withQuery(`/knowledge-bases/${kbId}/documents`, {
      limit: String(UI_POLICY.documentPageSize),
      sort: "-created_at",
      cursor,
    }));
  }

  getDocument(documentId: UUID): Promise<DocumentDetail> {
    return this.request(`/documents/${documentId}`);
  }

  getDocumentChunks(
    documentId: UUID,
    cursor?: string,
  ): Promise<DocumentChunkInspection> {
    return this.request(this.withQuery(`/documents/${documentId}/chunks`, {
      limit: String(UI_POLICY.chunkPageSize),
      cursor,
    }));
  }

  uploadDocument(
    kbId: UUID,
    file: File,
    idempotencyKey: UUID,
  ): Promise<DocumentUpload> {
    return this.upload(
      `/knowledge-bases/${kbId}/documents`,
      file,
      file.name,
      idempotencyKey,
    );
  }

  uploadDocumentVersion(
    documentId: UUID,
    file: File,
    displayName: string,
    idempotencyKey: UUID,
  ): Promise<DocumentUpload> {
    return this.upload(
      `/documents/${documentId}/versions`,
      file,
      displayName,
      idempotencyKey,
    );
  }

  deleteDocument(documentId: UUID, idempotencyKey: UUID): Promise<unknown> {
    return this.request(`/documents/${documentId}`, {
      method: "DELETE",
      headers: { "Idempotency-Key": idempotencyKey },
    });
  }

  deleteDocumentChunk(documentId: UUID, chunkId: UUID): Promise<{
    document_id: UUID;
    chunk_id: UUID;
    excluded_at: string;
  }> {
    return this.request(`/documents/${documentId}/chunks/${chunkId}`, {
      method: "DELETE",
    });
  }

  listIndexingJobs(kbId: UUID): Promise<Page<IndexingJob>> {
    return this.request(this.withQuery(`/knowledge-bases/${kbId}/indexing-jobs`, {
      limit: String(UI_POLICY.indexingJobPageSize),
    }));
  }

  retryIndexingJob(jobId: UUID, idempotencyKey: UUID): Promise<IndexingJob> {
    return this.request(`/indexing-jobs/${jobId}/retry`, {
      method: "POST",
      headers: { "Idempotency-Key": idempotencyKey },
    });
  }

  queryRetrievalDebug(
    knowledgeBaseId: UUID,
    query: string,
    topK: number,
    strategy: "exact_vector" | "hybrid",
    rerankMode: RerankMode,
  ): Promise<RetrievalEvidencePack> {
    return this.request("/retrieval/query", {
      method: "POST",
      headers: this.jsonHeaders(),
      body: JSON.stringify({
        knowledge_base_id: knowledgeBaseId,
        query,
        top_k: topK,
        strategy,
        rerank_mode: rerankMode,
        include_debug: true,
      }),
    });
  }

  getGraphConfig(knowledgeBaseId: UUID): Promise<GraphConfig> {
    return this.request(`/knowledge-bases/${knowledgeBaseId}/graph-config`);
  }

  getGraphSchemaProfiles(): Promise<GraphSchemaProfile[]> {
    return this.request("/graph-schema-profiles");
  }

  updateGraphConfig(
    knowledgeBaseId: UUID,
    payload: GraphConfigUpdate,
  ): Promise<GraphConfig> {
    return this.request(`/knowledge-bases/${knowledgeBaseId}/graph-config`, {
      method: "PUT",
      headers: this.jsonHeaders(),
      body: JSON.stringify(payload),
    });
  }

  getModelSettings(): Promise<ModelSettings> {
    return this.request("/model-settings");
  }

  createModelProvider(payload: {
    name: string;
    protocol: ModelProviderProtocol;
    base_url: string;
    api_key: string;
    timeout_seconds: number;
    max_retries: number;
    max_concurrency: number;
  }): Promise<ModelProvider> {
    return this.request("/model-providers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  updateModelProvider(providerId: UUID, payload: {
    name?: string;
    protocol?: ModelProviderProtocol;
    base_url?: string;
    api_key?: string;
    timeout_seconds?: number;
    max_retries?: number;
    max_concurrency?: number;
    enabled?: boolean;
  }): Promise<ModelProvider> {
    return this.request(`/model-providers/${providerId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  listProviderModels(providerId: UUID): Promise<ModelCatalog> {
    return this.request(`/model-providers/${providerId}/models`);
  }

  createModelProfile(payload: {
    provider_id: UUID;
    name: string;
    kind: ModelKind;
    model: string;
    parameters: Record<string, unknown>;
  }): Promise<ModelProfile> {
    return this.request("/model-profiles", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  updateModelProfile(profileId: UUID, payload: {
    provider_id?: UUID;
    name?: string;
    model?: string;
    parameters?: Record<string, unknown>;
    enabled?: boolean;
  }): Promise<ModelProfile> {
    return this.request(`/model-profiles/${profileId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  validateModelProfile(profileId: UUID): Promise<ModelProfile> {
    return this.request(`/model-profiles/${profileId}/validate`, {
      method: "POST",
    });
  }

  updateModelSelection(payload: {
    chat_profile_revision_id: UUID | null;
    text_embedding_profile_revision_id: UUID | null;
    multimodal_embedding_profile_revision_id: UUID | null;
  }): Promise<ModelSelection> {
    return this.request("/model-selection", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chat_profile_revision_id: payload.chat_profile_revision_id,
        text_embedding_profile_revision_id: payload.text_embedding_profile_revision_id,
        multimodal_embedding_profile_revision_id: payload.multimodal_embedding_profile_revision_id,
      }),
    });
  }

  listChatSessions(
    knowledgeBaseId: UUID,
    cursor?: string,
  ): Promise<Page<ChatSession>> {
    return this.request(this.withQuery("/chat/sessions", {
      knowledge_base_id: knowledgeBaseId,
      limit: String(UI_POLICY.chatPageSize),
      sort: "-updated_at",
      cursor,
    }));
  }

  createChatSession(
    knowledgeBaseId: UUID,
    title: string,
  ): Promise<ChatSession> {
    return this.request("/chat/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        knowledge_base_id: knowledgeBaseId,
        title,
      }),
    });
  }

  listChatMessages(
    sessionId: UUID,
    cursor?: string,
  ): Promise<Page<ChatMessage>> {
    return this.request(this.withQuery(`/chat/sessions/${sessionId}/messages`, {
      limit: String(UI_POLICY.chatPageSize),
      sort: "created_at",
      cursor,
    }));
  }

  createChatRun(
    payload: ChatRunCreate,
    idempotencyKey: UUID,
  ): Promise<ChatRun> {
    return this.request("/chat/runs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": idempotencyKey,
      },
      body: JSON.stringify(payload),
    });
  }

  getChatRun(runIdOrUrl: UUID | string): Promise<ChatRun> {
    const path = runIdOrUrl.startsWith("/")
      || runIdOrUrl.startsWith("http://")
      ? runIdOrUrl
      : `/chat/runs/${runIdOrUrl}`;
    return this.request(path);
  }

  subscribeChatRun(
    eventsUrl: string,
    handlers: {
      completed: (event: ChatTerminalEvent) => void;
      failed: (event: ChatTerminalEvent) => void;
      progress: (event: ChatProgressSnapshot) => void;
      progressInvalid: () => void;
      error: () => void;
      open?: () => void;
    },
  ): () => void {
    const source = new EventSource(this.resolvePublicApiUrl(eventsUrl));
    const completed = (event: Event) => {
      const value = parseSseData(event);
      value ? handlers.completed(value) : handlers.error();
    };
    const failed = (event: Event) => {
      const value = parseSseData(event);
      value ? handlers.failed(value) : handlers.error();
    };
    const progress = (event: Event) => {
      const value = parseProgressSnapshot(event);
      value ? handlers.progress(value) : handlers.progressInvalid();
    };
    const open = () => handlers.open?.();
    const error = () => handlers.error();
    source.addEventListener("answer.completed", completed);
    source.addEventListener("run.failed", failed);
    source.addEventListener("agent.progress", progress);
    source.addEventListener("open", open);
    source.addEventListener("error", error);
    return () => {
      source.removeEventListener("answer.completed", completed);
      source.removeEventListener("run.failed", failed);
      source.removeEventListener("agent.progress", progress);
      source.removeEventListener("open", open);
      source.removeEventListener("error", error);
      source.close();
    };
  }

  resolvePublicApiUrl(pathOrUrl: string): string {
    const candidate = pathOrUrl.startsWith("http://")
      ? new URL(pathOrUrl)
      : pathOrUrl.startsWith(API_PATH)
        ? new URL(pathOrUrl, this.apiOrigin)
        : new URL(`${this.apiBaseUrl}/${pathOrUrl.replace(/^\//, "")}`);
    if (candidate.protocol !== "http:") {
      throw new ApiClientError("API 返回了不安全的资源地址。");
    }
    if (candidate.origin !== this.apiOrigin || !isApiPath(candidate.pathname)) {
      throw new ApiClientError("API 返回了范围外的资源地址。");
    }
    return candidate.toString();
  }

  private withQuery(
    path: string,
    values: Record<string, string | undefined>,
  ): string {
    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(values)) {
      if (value !== undefined) query.set(key, value);
    }
    return `${path}?${query.toString()}`;
  }

  private upload(
    path: string,
    file: File,
    displayName: string,
    idempotencyKey: UUID,
  ): Promise<DocumentUpload> {
    const extension = file.name.toLowerCase().split(".").pop() ?? "";
    const mediaType = UPLOAD_MEDIA_TYPES[extension];
    if (!mediaType) {
      throw new ApiClientError(
        "不支持该文件类型。请选择 TXT、Markdown、HTML、CSV、PDF、DOCX、PPTX、XLSX 或 MDZ 文件。",
        { code: "FRONTEND_FILE_TYPE_UNSUPPORTED" },
      );
    }
    return this.request(path, {
      method: "POST",
      headers: {
        "Content-Type": mediaType,
        "Idempotency-Key": idempotencyKey,
        "X-Document-Metadata": encodeUploadMetadata(file.name, displayName),
      },
      body: file,
    });
  }

  private jsonHeaders(idempotencyKey?: UUID): Record<string, string> {
    return {
      "Content-Type": "application/json",
      ...(idempotencyKey ? { "Idempotency-Key": idempotencyKey } : {}),
    };
  }

  private async request<T>(pathOrUrl: string, init: RequestInit = {}): Promise<T> {
    let response: Response;
    try {
      response = await fetch(this.resolvePublicApiUrl(pathOrUrl), {
        ...init,
        cache: "no-store",
      });
    } catch (error) {
      if (error instanceof ApiClientError) throw error;
      throw new ApiClientError("无法完成 API 请求，请检查本地服务状态或跨域配置。", {
        retryable: true,
      });
    }
    if (!response.ok) {
      let problem: ApiProblem = {};
      try {
        problem = await response.json() as ApiProblem;
      } catch {
        // The status code remains the authoritative fallback.
      }
      throw new ApiClientError(problemMessage(response.status, problem), {
        status: response.status,
        code: problem.code,
        retryable: Boolean(problem.retryable),
        traceId: problem.trace_id,
        fieldErrors: problem.errors,
      });
    }
    return response.json() as Promise<T>;
  }
}

function encodeUploadMetadata(filename: string, displayName: string): string {
  const bytes = new TextEncoder().encode(JSON.stringify({
    v: 1,
    filename,
    display_name: displayName,
  }));
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary)
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
}

function validateApiBaseUrl(value: string): URL {
  const parsed = new URL(value);
  const loopback = parsed.hostname === "127.0.0.1"
    || parsed.hostname === "::1"
    || parsed.hostname === "localhost";
  if (
    parsed.protocol !== "http:"
    || !loopback
    || parsed.username
    || parsed.password
    || parsed.search
    || parsed.hash
    || parsed.pathname.replace(/\/$/, "") !== API_PATH
  ) {
    throw new ApiClientError("API 地址必须是本机的 /api/v1 地址。");
  }
  return parsed;
}

function isApiPath(path: string): boolean {
  return path === API_PATH || path.startsWith(`${API_PATH}/`);
}

function parseSseData(event: Event): ChatTerminalEvent | null {
  if (!(event instanceof MessageEvent) || typeof event.data !== "string") {
    return null;
  }
  try {
    const value = JSON.parse(event.data) as Partial<ChatTerminalEvent>;
    return typeof value.run_id === "string" && typeof value.status_url === "string"
      ? value as ChatTerminalEvent
      : null;
  } catch {
    return null;
  }
}

function parseProgressSnapshot(event: Event): ChatProgressSnapshot | null {
  const value = parseJsonObject(event);
  if (
    !value
    || !hasExactKeys(value, [
      "run_id",
      "attempt",
      "seq",
      "active_stage",
      "activity",
      "completed_stages",
      "status",
      "facts",
    ])
    || typeof value.run_id !== "string"
    || !isPositiveInteger(value.attempt)
    || !isPositiveInteger(value.seq)
  ) return null;
  const stages = new Set([
    "understand_query",
    "retrieve_evidence",
    "prepare_visual_evidence",
    "generate_answer",
    "validate_answer",
    "persist_result",
  ]);
  const activities = new Set([
    "load_context",
    "tool_decision",
    "search_knowledge_base",
    "calculate",
    "submit_answer",
    "retrieval_complete",
    "prepare_visual_evidence",
    "generate_answer",
    "validate_answer",
    "persist_result",
  ]);
  if (
    typeof value.active_stage !== "string"
    || !stages.has(value.active_stage)
    || typeof value.activity !== "string"
    || !activities.has(value.activity)
    || !isStringArray(value.completed_stages, 8, stages)
    || !["active", "completed"].includes(String(value.status))
    || !isProgressFacts(value.facts)
  ) return null;
  return value as unknown as ChatProgressSnapshot;
}

function isProgressFacts(value: unknown): value is ChatProgressSnapshot["facts"] {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  const facts = value as Record<string, unknown>;
  return hasExactKeys(facts, [
    "objective",
    "queries",
    "evidence_count",
    "new_evidence_count",
    "retrieval_calls",
    "covered_aspects",
    "missing_aspects",
    "conflict_count",
  ])
    && isNullableBoundedString(facts.objective)
    && isStringArray(facts.queries, 3)
    && isNullableCounter(facts.evidence_count)
    && isNullableCounter(facts.new_evidence_count)
    && isNullableCounter(facts.retrieval_calls)
    && isStringArray(facts.covered_aspects, 6)
    && isStringArray(facts.missing_aspects, 6)
    && isNullableCounter(facts.conflict_count);
}

function isStringArray(
  value: unknown,
  maximum: number,
  allowed?: Set<string>,
): value is string[] {
  return Array.isArray(value)
    && value.length <= maximum
    && value.every((item) => (
      typeof item === "string"
      && item.length > 0
      && item.length <= 160
      && (!allowed || allowed.has(item))
    ));
}

function isNullableBoundedString(value: unknown): value is string | null {
  return value === null
    || (typeof value === "string" && value.length > 0 && value.length <= 160);
}

function isNullableCounter(value: unknown): value is number | null {
  return value === null
    || (Number.isInteger(value) && Number(value) >= 0 && Number(value) <= 1000);
}

function parseJsonObject(event: Event): Record<string, unknown> | null {
  if (!(event instanceof MessageEvent) || typeof event.data !== "string") {
    return null;
  }
  try {
    const value: unknown = JSON.parse(event.data);
    return typeof value === "object" && value !== null && !Array.isArray(value)
      ? value as Record<string, unknown>
      : null;
  } catch {
    return null;
  }
}

function hasExactKeys(
  value: Record<string, unknown>,
  keys: string[],
): boolean {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return actual.length === expected.length
    && actual.every((key, index) => key === expected[index]);
}

function isPositiveInteger(value: unknown): value is number {
  return Number.isInteger(value) && Number(value) >= 1;
}

function problemMessage(status: number, problem: ApiProblem): string {
  if (problem.code === "CHAT_SESSION_BUSY") {
    return "这个会话仍在生成回答，请稍候。";
  }
  if (problem.code === "RESOURCE_NAME_CONFLICT") {
    return "该名称已存在，请修改名称，或编辑现有配置。";
  }
  const detail = typeof problem.detail === "string" ? problem.detail : null;
  const fieldDetail = problem.errors?.map((item) => item.message).join("；");
  if (status === 404) return detail || "请求的本地内容已不存在。";
  if (status === 409) return detail || "当前状态暂时不能完成此操作。";
  if (status === 422) return fieldDetail || detail || "提交内容不符合要求，请检查后重试。";
  if (status >= 500) return "本地知识库服务暂时不可用，请稍后重试。";
  return detail || (typeof problem.title === "string" ? problem.title : "请求未能完成。");
}
