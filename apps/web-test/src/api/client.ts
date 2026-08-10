import type {
  ChatAnswerCompletedEvent,
  ChatMessage,
  ChatPreviewDeltaEvent,
  ChatPreviewResetEvent,
  ChatRun,
  ChatRunCreate,
  ChatRunFinalContext,
  ChatRunFailedEvent,
  ChatSession,
  DocumentRecord,
  DocumentDetail,
  DocumentChunkInspection,
  DocumentUpload,
  EvidencePack,
  IndexingJob,
  KnowledgeBase,
  KnowledgeBaseEmbeddingSelection,
  ModelSettingsSummary,
  ParsingPreset,
  ChunkingPreset,
  Page,
  ProblemDetails,
  RuntimeConfig,
  RetrievalCapabilities,
  UUID,
} from "./types";

const API_PATH = "/api/v1";
const SAFE_LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]"]);

const OOXML = "application/vnd.openxmlformats-officedocument";

/** The upload media types the API admits, keyed by lowercase extension. */
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

export class ApiClientError extends Error {
  readonly status: number | null;
  readonly code: string;
  readonly retryable: boolean;
  readonly traceId: string | null;
  readonly fieldErrors: ProblemDetails["errors"];

  constructor(
    message: string,
    options: {
      status?: number | null;
      code?: string;
      retryable?: boolean;
      traceId?: string | null;
      fieldErrors?: ProblemDetails["errors"];
    } = {},
  ) {
    super(message);
    this.name = "ApiClientError";
    this.status = options.status ?? null;
    this.code = options.code ?? "FRONTEND_REQUEST_FAILED";
    this.retryable = options.retryable ?? false;
    this.traceId = options.traceId ?? null;
    this.fieldErrors = options.fieldErrors ?? null;
  }
}

export async function loadRuntimeConfig(): Promise<RuntimeConfig> {
  let response: Response;
  try {
    response = await fetch("/runtime-config.json", {
      cache: "no-store",
      credentials: "omit",
    });
  } catch {
    throw new ApiClientError("The local frontend configuration is unavailable.");
  }
  if (!response.ok) {
    throw new ApiClientError("The local frontend configuration is unavailable.");
  }
  let value: unknown;
  try {
    value = await response.json();
  } catch {
    throw new ApiClientError("The local frontend configuration is invalid.");
  }
  if (!isObject(value) || typeof value.api_base_url !== "string") {
    throw new ApiClientError("The local frontend configuration is invalid.");
  }
  validateApiBaseUrl(value.api_base_url);
  return { api_base_url: value.api_base_url };
}

export class ApiClient {
  readonly apiBaseUrl: string;
  readonly apiOrigin: string;

  constructor(config: RuntimeConfig) {
    const base = validateApiBaseUrl(config.api_base_url);
    this.apiBaseUrl = base.toString().replace(/\/$/, "");
    this.apiOrigin = base.origin;
  }

  listKnowledgeBases(cursor?: string): Promise<Page<KnowledgeBase>> {
    return this.request(this.withQuery("/knowledge-bases", {
      limit: "100",
      sort: "name",
      cursor,
    }));
  }

  getRetrievalCapabilities(): Promise<RetrievalCapabilities> {
    return this.request("/retrieval/capabilities");
  }

  getModelSettings(): Promise<ModelSettingsSummary> {
    return this.request("/model-settings");
  }

  createKnowledgeBase(
    name: string,
    preset: ChunkingPreset,
    parsingPreset: ParsingPreset,
    embedding: KnowledgeBaseEmbeddingSelection,
    idempotencyKey: UUID,
  ): Promise<KnowledgeBase> {
    return this.request("/knowledge-bases", {
      method: "POST",
      headers: this.jsonHeaders(idempotencyKey),
      body: JSON.stringify({
        name,
        parsing: { preset: parsingPreset },
        chunking: { preset },
        embedding,
      }),
    });
  }

  listDocuments(kbId: UUID, cursor?: string): Promise<Page<DocumentRecord>> {
    return this.request(this.withQuery(`/knowledge-bases/${kbId}/documents`, {
      limit: "100",
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
      limit: "100",
      cursor,
    }));
  }

  uploadDocument(
    kbId: UUID,
    file: File,
    displayName: string,
    idempotencyKey: UUID,
  ): Promise<DocumentUpload> {
    return this.upload(
      `/knowledge-bases/${kbId}/documents`,
      file,
      displayName,
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

  getIndexingJob(jobId: UUID): Promise<IndexingJob> {
    return this.request(`/indexing-jobs/${jobId}`);
  }

  retryIndexingJob(jobId: UUID, idempotencyKey: UUID): Promise<IndexingJob> {
    return this.request(`/indexing-jobs/${jobId}/retry`, {
      method: "POST",
      headers: { "Idempotency-Key": idempotencyKey },
    });
  }

  listChatSessions(cursor?: string): Promise<Page<ChatSession>> {
    return this.request(this.withQuery("/chat/sessions", {
      limit: "100",
      sort: "-updated_at",
      cursor,
    }));
  }

  createChatSession(
    knowledgeBaseId: UUID,
    title: string | null,
  ): Promise<ChatSession> {
    return this.request("/chat/sessions", {
      method: "POST",
      headers: this.jsonHeaders(),
      body: JSON.stringify({ knowledge_base_id: knowledgeBaseId, title }),
    });
  }

  listChatMessages(sessionId: UUID, cursor?: string): Promise<Page<ChatMessage>> {
    return this.request(this.withQuery(`/chat/sessions/${sessionId}/messages`, {
      limit: "100",
      sort: "created_at",
      cursor,
    }));
  }

  createChatRun(payload: ChatRunCreate, idempotencyKey: UUID): Promise<ChatRun> {
    return this.request("/chat/runs", {
      method: "POST",
      headers: this.jsonHeaders(idempotencyKey),
      body: JSON.stringify(payload),
    });
  }

  getChatRun(runIdOrUrl: UUID | string): Promise<ChatRun> {
    const path = runIdOrUrl.startsWith("/")
      ? runIdOrUrl
      : `/chat/runs/${runIdOrUrl}`;
    return this.request(path);
  }

  getChatRunFinalContext(runIdOrUrl: UUID | string): Promise<ChatRunFinalContext> {
    const path = runIdOrUrl.startsWith("/")
      ? runIdOrUrl
      : `/chat/runs/${runIdOrUrl}/final-context`;
    return this.request(path);
  }

  queryRetrievalDebug(
    knowledgeBaseId: UUID,
    query: string,
    topK: number,
    strategy: "exact_vector" | "hybrid",
  ): Promise<EvidencePack> {
    return this.request("/retrieval/query", {
      method: "POST",
      headers: this.jsonHeaders(),
      body: JSON.stringify({
        knowledge_base_id: knowledgeBaseId,
        query,
        top_k: topK,
        strategy,
        rerank_mode: "classic",
        include_debug: true,
      }),
    });
  }

  subscribeChatRun(
    eventsUrl: string,
    handlers: {
      completed: (event: ChatAnswerCompletedEvent) => void;
      failed: (event: ChatRunFailedEvent) => void;
      previewDelta: (event: ChatPreviewDeltaEvent) => void;
      previewReset: (event: ChatPreviewResetEvent) => void;
      previewInvalid: () => void;
      error: () => void;
      open?: () => void;
    },
  ): () => void {
    const source = new EventSource(this.resolvePublicApiUrl(eventsUrl));
    const completed = (event: Event) => {
      const value = parseSseData<ChatAnswerCompletedEvent>(event, "answer.completed");
      if (value) handlers.completed(value);
      else handlers.error();
    };
    const failed = (event: Event) => {
      const value = parseSseData<ChatRunFailedEvent>(event, "run.failed");
      if (value) handlers.failed(value);
      else handlers.error();
    };
    const previewDelta = (event: Event) => {
      const value = parsePreviewDelta(event);
      if (value) handlers.previewDelta(value);
      else handlers.previewInvalid();
    };
    const previewReset = (event: Event) => {
      const value = parsePreviewReset(event);
      if (value) handlers.previewReset(value);
      else handlers.previewInvalid();
    };
    const open = () => handlers.open?.();
    const error = () => handlers.error();
    source.addEventListener("answer.completed", completed);
    source.addEventListener("run.failed", failed);
    source.addEventListener("answer.preview.delta", previewDelta);
    source.addEventListener("answer.preview.reset", previewReset);
    source.addEventListener("open", open);
    source.addEventListener("error", error);
    return () => {
      source.removeEventListener("answer.completed", completed);
      source.removeEventListener("run.failed", failed);
      source.removeEventListener("answer.preview.delta", previewDelta);
      source.removeEventListener("answer.preview.reset", previewReset);
      source.removeEventListener("open", open);
      source.removeEventListener("error", error);
      source.close();
    };
  }

  resolvePublicApiUrl(pathOrUrl: string): string {
    const candidate = pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")
      ? new URL(pathOrUrl)
      : pathOrUrl.startsWith(API_PATH)
        ? new URL(pathOrUrl, this.apiOrigin)
        : new URL(`${this.apiBaseUrl}/${pathOrUrl.replace(/^\//, "")}`);
    if (candidate.origin !== this.apiOrigin || !isApiPath(candidate.pathname)) {
      throw new ApiClientError("The API returned an unsafe resource URL.");
    }
    return candidate.toString();
  }

  private upload(
    path: string,
    file: File,
    displayName: string,
    idempotencyKey: UUID,
  ): Promise<DocumentUpload> {
    const extension = file.name.toLowerCase().split(".").pop() ?? "";
    const mediaType = UPLOAD_MEDIA_TYPES[extension] ?? null;
    if (!mediaType) {
      throw new ApiClientError(
        "Choose a .txt, .md, .mdz, .html, .csv, .pdf, .docx, .pptx, or .xlsx file.",
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

  private async request<T>(pathOrUrl: string, init: RequestInit = {}): Promise<T> {
    let response: Response;
    try {
      response = await fetch(this.resolvePublicApiUrl(pathOrUrl), {
        ...init,
        cache: "no-store",
        credentials: "omit",
        headers: {
          Accept: "application/json",
          ...(init.headers ?? {}),
        },
      });
    } catch (error) {
      if (error instanceof ApiClientError) throw error;
      throw new ApiClientError("The API could not be reached.", {
        code: "FRONTEND_NETWORK_ERROR",
        retryable: true,
      });
    }
    if (!response.ok) {
      throw await problemFromResponse(response);
    }
    try {
      return (await response.json()) as T;
    } catch {
      throw new ApiClientError("The API returned an invalid response.", {
        status: response.status,
        code: "FRONTEND_INVALID_RESPONSE",
      });
    }
  }
}

function validateApiBaseUrl(value: string): URL {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new ApiClientError("The local API address is invalid.");
  }
  if (
    url.protocol !== "http:" ||
    !SAFE_LOOPBACK_HOSTS.has(url.hostname) ||
    url.pathname.replace(/\/$/, "") !== API_PATH ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  ) {
    throw new ApiClientError("The local API address is outside the loopback boundary.");
  }
  return url;
}

function isApiPath(pathname: string): boolean {
  return pathname === API_PATH || pathname.startsWith(`${API_PATH}/`);
}

function encodeUploadMetadata(filename: string, displayName: string): string {
  const bytes = new TextEncoder().encode(JSON.stringify({
    v: 1,
    filename,
    display_name: displayName,
  }));
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary)
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
}

async function problemFromResponse(response: Response): Promise<ApiClientError> {
  let value: unknown = null;
  try {
    value = await response.json();
  } catch {
    // Do not expose an unknown response body.
  }
  if (isProblemDetails(value)) {
    return new ApiClientError(`${value.title}: ${value.detail}`, {
      status: value.status,
      code: value.code,
      retryable: value.retryable,
      traceId: value.trace_id,
      fieldErrors: value.errors,
    });
  }
  return new ApiClientError(`The API request failed with status ${response.status}.`, {
    status: response.status,
    code: "FRONTEND_HTTP_ERROR",
    retryable: response.status >= 500,
  });
}

function isProblemDetails(value: unknown): value is ProblemDetails {
  return isObject(value)
    && typeof value.title === "string"
    && typeof value.detail === "string"
    && typeof value.status === "number"
    && typeof value.code === "string"
    && typeof value.trace_id === "string"
    && typeof value.retryable === "boolean";
}

function parseSseData<T>(event: Event, expectedEvent: string): T | null {
  if (!(event instanceof MessageEvent) || event.type !== expectedEvent) return null;
  try {
    const value: unknown = JSON.parse(String(event.data));
    if (!isObject(value) || typeof value.run_id !== "string") return null;
    return value as T;
  } catch {
    return null;
  }
}

function parsePreviewDelta(event: Event): ChatPreviewDeltaEvent | null {
  const value = parseSseObject(event, "answer.preview.delta");
  if (
    !value
    || !hasExactKeys(value, ["run_id", "attempt", "seq", "delta"])
    || typeof value.run_id !== "string"
    || !isPositiveInteger(value.attempt)
    || !isPositiveInteger(value.seq)
    || typeof value.delta !== "string"
    || value.delta.length === 0
  ) return null;
  return value as unknown as ChatPreviewDeltaEvent;
}

function parsePreviewReset(event: Event): ChatPreviewResetEvent | null {
  const value = parseSseObject(event, "answer.preview.reset");
  const reasons = new Set([
    "generation_failed",
    "validation_repair",
    "preview_invalid",
  ]);
  if (
    !value
    || !hasExactKeys(value, ["run_id", "attempt", "seq", "reason"])
    || typeof value.run_id !== "string"
    || !isPositiveInteger(value.attempt)
    || !isPositiveInteger(value.seq)
    || typeof value.reason !== "string"
    || !reasons.has(value.reason)
  ) return null;
  return value as unknown as ChatPreviewResetEvent;
}

function parseSseObject(
  event: Event,
  expectedEvent: string,
): Record<string, unknown> | null {
  if (
    !(event instanceof MessageEvent)
    || event.type !== expectedEvent
    || typeof event.data !== "string"
  ) return null;
  try {
    const value: unknown = JSON.parse(event.data);
    return isObject(value) ? value : null;
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

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
