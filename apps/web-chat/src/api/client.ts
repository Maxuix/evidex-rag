import type {
  ApiProblem,
  ChatMessage,
  ChatPreviewDeltaEvent,
  ChatPreviewResetEvent,
  ChatRun,
  ChatRunCreate,
  ChatSession,
  ChatTerminalEvent,
  KnowledgeBase,
  Page,
  UUID,
} from "./types";

const API_PATH = "/api/v1";

interface RuntimeConfig {
  api_base_url: string;
}

export class ApiClientError extends Error {
  readonly status: number | null;
  readonly code: string | null;
  readonly retryable: boolean;

  constructor(
    message: string,
    options: {
      status?: number;
      code?: string;
      retryable?: boolean;
    } = {},
  ) {
    super(message);
    this.name = "ApiClientError";
    this.status = options.status ?? null;
    this.code = options.code ?? null;
    this.retryable = options.retryable ?? false;
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
      limit: "100",
      sort: "name",
      cursor,
    }));
  }

  listChatSessions(
    knowledgeBaseId: UUID,
    cursor?: string,
  ): Promise<Page<ChatSession>> {
    return this.request(this.withQuery("/chat/sessions", {
      knowledge_base_id: knowledgeBaseId,
      limit: "50",
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
      limit: "50",
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
      previewDelta: (event: ChatPreviewDeltaEvent) => void;
      previewReset: (event: ChatPreviewResetEvent) => void;
      previewInvalid: () => void;
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
    const previewDelta = (event: Event) => {
      const value = parsePreviewDelta(event);
      value ? handlers.previewDelta(value) : handlers.previewInvalid();
    };
    const previewReset = (event: Event) => {
      const value = parsePreviewReset(event);
      value ? handlers.previewReset(value) : handlers.previewInvalid();
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

  private async request<T>(pathOrUrl: string, init: RequestInit = {}): Promise<T> {
    let response: Response;
    try {
      response = await fetch(this.resolvePublicApiUrl(pathOrUrl), {
        ...init,
        cache: "no-store",
      });
    } catch (error) {
      if (error instanceof ApiClientError) throw error;
      throw new ApiClientError("无法连接本地知识库服务，请确认服务已启动。", {
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
      });
    }
    return response.json() as Promise<T>;
  }
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

function parsePreviewDelta(event: Event): ChatPreviewDeltaEvent | null {
  const value = parseJsonObject(event);
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
  const value = parseJsonObject(event);
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
  if (status === 404) return "请求的本地内容已不存在。";
  if (status === 409) return "当前状态暂时不能完成此操作。";
  if (status === 422) return "提交内容不符合要求，请检查后重试。";
  if (status >= 500) return "本地知识库服务暂时不可用，请稍后重试。";
  return typeof problem.title === "string" ? problem.title : "请求未能完成。";
}
