import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiClient, ApiClientError } from "./client";

const client = new ApiClient({ api_base_url: "http://localhost/api/v1" });

function response(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/problem+json" },
  });
}

async function rejected(operation: () => Promise<unknown>): Promise<ApiClientError> {
  try {
    await operation();
  } catch (error) {
    expect(error).toBeInstanceOf(ApiClientError);
    return error as ApiClientError;
  }
  throw new Error("expected the API operation to fail");
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("API error behavior", () => {
  it("keeps the chat-busy code, retryability, and trace id while showing the user message", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response(409, {
      code: "CHAT_SESSION_BUSY",
      detail: "internal busy detail",
      retryable: true,
      trace_id: "trace-busy",
    })));

    const error = await rejected(() => client.createChatSession("kb-a", "新会话"));

    expect(error.message).toBe("这个会话仍在生成回答，请稍候。");
    expect(error.code).toBe("CHAT_SESSION_BUSY");
    expect(error.retryable).toBe(true);
    expect(error.traceId).toBe("trace-busy");
  });

  it("retains field violations from validation problems for form handling", async () => {
    const fieldErrors = [{
      location: ["body", "name"],
      message: "Field required",
      error_type: "missing",
    }];
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response(422, {
      code: "REQUEST_VALIDATION_FAILED",
      errors: fieldErrors,
    })));

    const error = await rejected(() => client.createChatSession("kb-a", ""));

    expect(error.message).toBe("Field required");
    expect(error.code).toBe("REQUEST_VALIDATION_FAILED");
    expect(error.fieldErrors).toEqual(fieldErrors);
    expect(error.retryable).toBe(false);
  });

  it("explains duplicate resource names instead of exposing the API detail", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response(409, {
      code: "RESOURCE_NAME_CONFLICT",
      detail: "A resource with the requested name already exists.",
      retryable: false,
    })));

    const error = await rejected(() => client.createModelProfile({
      provider_id: "provider-a",
      name: "mimo-v2.5",
      kind: "chat",
      model: "mimo-v2.5",
      parameters: {},
    }));

    expect(error.message).toBe("该名称已存在，请修改名称，或编辑现有配置。");
    expect(error.code).toBe("RESOURCE_NAME_CONFLICT");
    expect(error.retryable).toBe(false);
  });

  it("marks transport failures as retryable with a safe local message", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("network detail")));

    const error = await rejected(() => client.listKnowledgeBases());

    expect(error.message).toBe("无法完成 API 请求，请检查本地服务状态或跨域配置。");
    expect(error.retryable).toBe(true);
    expect(error.code).toBeNull();
  });
});

describe("knowledge-base Auto-QA create payload", () => {
  it("omits a chat profile when Auto-QA is disabled", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      id: "kb-1",
    }), { status: 201, headers: { "Content-Type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);

    await client.createKnowledgeBase(
      "资料",
      "text_local_v1",
      "structural_balanced_v2",
      { strategy: "text_only", text_profile_revision_id: "emb-1" },
      "idem-1",
    );

    const body = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    expect(body.auto_qa).toEqual({ enabled: false });
  });

  it("requires the selected chat profile when Auto-QA is enabled", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      id: "kb-1",
    }), { status: 201, headers: { "Content-Type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);

    await client.createKnowledgeBase(
      "资料",
      "text_local_v1",
      "structural_balanced_v2",
      { strategy: "text_only", text_profile_revision_id: "emb-1" },
      "idem-1",
      { enabled: true, model_profile_revision_id: "chat-1" },
    );

    const body = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    expect(body.retrieval_defaults).toBeUndefined();
    expect(body.auto_qa).toEqual({
      enabled: true,
      model_profile_revision_id: "chat-1",
    });
  });
});

describe("activity delivery", () => {
  it("decodes observations, isolates unknown versions, and removes listeners on close", async () => {
    const { eventFixture } = await import("../execution/activityFixtures");
    const stream = new EventTarget();
    const close = vi.fn();
    vi.stubGlobal("EventSource", class { constructor() { return Object.assign(stream, { close }); } });
    const activity = vi.fn(), activityInvalid = vi.fn();
    const stop = client.subscribeChatRun("/api/v1/chat/runs/run/events", { completed: vi.fn(), failed: vi.fn(), progress: vi.fn(), progressInvalid: vi.fn(), error: vi.fn(), activity, activityInvalid });
    const valid = eventFixture();
    stream.dispatchEvent(new MessageEvent("agent.activity", { data: JSON.stringify(valid) }));
    stream.dispatchEvent(new MessageEvent("agent.activity", { data: JSON.stringify({ ...valid, version: "unknown" }) }));
    expect(activity).toHaveBeenCalledExactlyOnceWith(valid);
    expect(activityInvalid).toHaveBeenCalledOnce();
    stop();
    stream.dispatchEvent(new MessageEvent("agent.activity", { data: JSON.stringify(valid) }));
    expect(activity).toHaveBeenCalledOnce(); expect(close).toHaveBeenCalledOnce();
  });
  it("preserves a server-reported unavailable trace when normalizing a run", async () => {
    const { runFixture } = await import("../execution/activityFixtures");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response(200, runFixture({ activities: [], activity_unavailable: true }))));
    expect((await client.getChatRun("/api/v1/chat/runs/run")).activity_unavailable).toBe(true);
  });
});
