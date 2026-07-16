import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiClient, ApiClientError, loadRuntimeConfig } from "./client";
import { completedRun, ids } from "../test/fixtures";

const client = new ApiClient({ api_base_url: "http://127.0.0.1:8000/api/v1" });

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("ApiClient", () => {
  it("uploads the raw file with only the frozen public headers", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ job_id: ids.job }, 202));
    vi.stubGlobal("fetch", fetchMock);
    const file = new File(["# Guide"], "guide.md", { type: "text/markdown" });

    await client.uploadDocument(ids.kb, file, "guide.md", ids.sourceChange);

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`http://127.0.0.1:8000/api/v1/knowledge-bases/${ids.kb}/documents`);
    expect(init.body).toBe(file);
    expect(init.credentials).toBe("omit");
    expect(init.headers).toMatchObject({
      Accept: "application/json",
      "Content-Type": "text/markdown",
      "Idempotency-Key": ids.sourceChange,
      "X-Document-Filename": "guide.md",
      "X-Document-Display-Name": "guide.md",
    });
    expect(JSON.stringify(init.headers)).not.toContain("Authorization");
  });

  it("rejects non-ASCII upload headers before fetch", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    expect(() => client.uploadDocument(
      ids.kb,
      new File(["text"], "中文.txt"),
      "中文.txt",
      ids.sourceChange,
    )).toThrowError("printable ASCII");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("normalizes Problem Details and does not expose unknown bodies", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({
      type: "about:blank",
      title: "Index unavailable",
      status: 503,
      detail: "Try again later.",
      instance: "/api/v1/retrieval/query",
      code: "EMBEDDING_PROVIDER_UNAVAILABLE",
      trace_id: "trace-safe",
      retryable: true,
      errors: null,
    }, 503)));

    await expect(client.queryRetrievalDebug(ids.kb, "query", 10)).rejects.toMatchObject({
      status: 503,
      code: "EMBEDDING_PROVIDER_UNAVAILABLE",
      retryable: true,
      traceId: "trace-safe",
    });

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(
      "provider-secret-stack",
      { status: 500, headers: { "Content-Type": "text/plain" } },
    )));
    const unknownBodyError = await client.getIndexingJob(ids.job).catch((error: unknown) => error);
    expect(unknownBodyError).toMatchObject({
      status: 500,
      code: "FRONTEND_HTTP_ERROR",
      retryable: true,
    });
    expect(String(unknownBodyError)).not.toContain("provider-secret-stack");
  });

  it("resolves backend resource URLs against the API origin and blocks escapes", () => {
    expect(client.resolvePublicApiUrl(`/api/v1/chat/runs/${ids.run}`)).toBe(
      `http://127.0.0.1:8000/api/v1/chat/runs/${ids.run}`,
    );
    expect(() => client.resolvePublicApiUrl("https://attacker.example/api/v1/chat/runs/x"))
      .toThrow(ApiClientError);
  });

  it("uses named terminal events and closes the EventSource", () => {
    const sources: MockEventSource[] = [];
    vi.stubGlobal("EventSource", class extends MockEventSource {
      constructor(url: string) {
        super(url);
        sources.push(this);
      }
    });
    const completed = vi.fn();
    const failed = vi.fn();
    const error = vi.fn();
    const close = client.subscribeChatRun(completedRun.events_url, {
      completed,
      failed,
      error,
    });

    sources[0].emit("answer.completed", {
      run_id: ids.run,
      status_url: completedRun.status_url,
    });
    expect(completed).toHaveBeenCalledOnce();
    expect(failed).not.toHaveBeenCalled();
    close();
    expect(sources[0].closed).toBe(true);
  });
});

describe("loadRuntimeConfig", () => {
  it("accepts only the loopback /api/v1 boundary", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({
      api_base_url: "http://127.0.0.1:8123/api/v1",
    })));
    await expect(loadRuntimeConfig()).resolves.toEqual({
      api_base_url: "http://127.0.0.1:8123/api/v1",
    });

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({
      api_base_url: "https://example.com/api/v1",
    })));
    await expect(loadRuntimeConfig()).rejects.toThrow("loopback boundary");
  });
});

class MockEventSource {
  readonly url: string;
  closed = false;
  private readonly listeners = new Map<string, Set<EventListener>>();

  constructor(url: string) {
    this.url = url;
  }

  addEventListener(type: string, listener: EventListenerOrEventListenerObject): void {
    const callback = typeof listener === "function" ? listener : listener.handleEvent.bind(listener);
    const values = this.listeners.get(type) ?? new Set<EventListener>();
    values.add(callback);
    this.listeners.set(type, values);
  }

  removeEventListener(type: string, listener: EventListenerOrEventListenerObject): void {
    if (typeof listener === "function") this.listeners.get(type)?.delete(listener);
  }

  close(): void {
    this.closed = true;
  }

  emit(type: string, data: unknown): void {
    const event = new MessageEvent(type, { data: JSON.stringify(data) });
    for (const listener of this.listeners.get(type) ?? []) listener(event);
  }
}

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
