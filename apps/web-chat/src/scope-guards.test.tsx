import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { eventFixture, runFixture, snapshotFixture, stepFixture } from "./execution/activityFixtures";

import { KnowledgeChat } from "./App";
import { KnowledgeBaseManagementPage } from "./KnowledgeBaseManagementPage";
import type { ApiClient } from "./api/client";
import type {
  DocumentRecord,
  IndexingJob,
  KnowledgeBase,
  ModelSettings,
  Page,
  ChatSession,
} from "./api/types";

const NOW = "2026-08-25T00:00:00Z";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function knowledgeBase(id: string, name: string): KnowledgeBase {
  return {
    id,
    name,
    source_change_seq: 0,
    active_index_revision_id: "index-${id}",
    embedding_space_id: "space-${id}",
    embedding: {
      strategy: "text_only",
      text: {
        embedding_space_id: "space-${id}",
        profile_revision_id: null,
        dimension: 1024,
      },
      cross_modal: null,
    },
    parsing: { preset: "text_local_v1", profile: "docling_text_local_v1" },
    chunking: {
      preset: "structural_balanced_v2",
      profile: "structural_by_title_token_v4",
    },
    retrieval_defaults: { strategy: "exact_vector", top_k: 10, rerank_mode: "classic" },
    answer_policy_defaults: { answer_style: "concise", insufficiency_policy: "refuse" },
    auto_qa: {
      enabled: false,
      questions_per_chunk: 5,
      model_profile_revision_id: null,
      model_name: null,
      model_revision: null,
    },
    provisioned_at: NOW,
    created_at: NOW,
    updated_at: NOW,
  };
}

function document(kbId: string, id: string, name: string): DocumentRecord {
  return {
    id,
    kb_id: kbId,
    display_name: name,
    current_version: null,
    deleted_at: null,
    created_at: NOW,
    updated_at: NOW,
  };
}

function session(kbId: string, id: string, title: string): ChatSession {
  return {
    id,
    knowledge_base_id: kbId,
    title,
    created_at: NOW,
    updated_at: NOW,
  };
}

function modelSettings(): ModelSettings {
  return {
    providers: [],
    profiles: [{
      id: "profile-a",
      revision_id: "profile-a-r1",
      revision: 1,
      provider_id: "provider-a",
      provider_revision_id: "provider-a-r1",
      name: "不可用模型",
      kind: "chat",
      model: "local-model",
      parameters: {
        type: "chat",
        temperature: 0,
        top_p: null,
        sampling_top_k: null,
        max_output_tokens: 128,
        reasoning_effort: "off",
        structured_output_mode: "json_object",
        vision_enabled: false,
      },
      enabled: true,
      provider_secret_available: false,
      validation_status: "valid",
      validation_error_code: null,
      validated_at: NOW,
      configuration_fingerprint: "fingerprint",
      capability_fingerprint: "capability",
      compatibility_fingerprint: null,
      embedding_validation: null,
      created_at: NOW,
      updated_at: NOW,
    }],
    selection: {
      chat_profile_revision_id: "profile-a-r1",
      text_embedding_profile_revision_id: null,
      multimodal_embedding_profile_revision_id: null,
      updated_at: NOW,
    },
  };
}

function managementProps(client: ApiClient, knowledgeBases: KnowledgeBase[], selected: string) {
  return {
    client,
    knowledgeBases,
    selectedKnowledgeBaseId: selected,
    modelSettings: null,
    graphConfig: null,
    graphSchemaProfiles: [],
    graphSchemaProfilesError: null,
    graphConfigLoading: false,
    graphConfigError: null,
    onRefreshGraphConfig: vi.fn().mockResolvedValue(null),
    onUpdateGraphConfig: vi.fn(),
    onKnowledgeBaseCreated: vi.fn(),
    onKnowledgeBaseDeleted: vi.fn(),
    onOpenModelSettings: vi.fn(),
    onOpenMobileSidebar: vi.fn(),
  };
}

afterEach(() => {
  cleanup();
});

describe("component request scopes", () => {
  it("keeps terminal activity authoritative over delayed polling and stream callbacks", async () => {
    const kb = knowledgeBase("kb-a", "知识库 A");
    const run = runFixture({ knowledge_base_id: kb.id, session_id: "session-a" });
    const callbacks: Parameters<ApiClient["subscribeChatRun"]>[1][] = [];
    const poll = deferred<ReturnType<typeof runFixture>>();
    const getChatRun = vi.fn().mockResolvedValue(run);
    const message = { id: "assistant-a", session_id: run.session_id, run_id: run.run_id, role: "assistant", assistant_status: "generating", content: "", created_at: NOW };
    const client = {
      getModelSettings: vi.fn().mockResolvedValue(modelSettings()),
      listKnowledgeBases: vi.fn().mockResolvedValue({ items: [kb], next_cursor: null }),
      getGraphConfig: vi.fn().mockResolvedValue(null), getGraphSchemaProfiles: vi.fn().mockResolvedValue([]),
      listChatSessions: vi.fn().mockResolvedValue({ items: [session(kb.id, run.session_id, "验收会话")], next_cursor: null }),
      listChatMessages: vi.fn().mockResolvedValue({ items: [message], next_cursor: null }),
      getChatRun,
      subscribeChatRun: vi.fn((_url, handlers) => { callbacks.push(handlers); return vi.fn(); }),
    } as unknown as ApiClient;
    render(<KnowledgeChat client={client} />);
    await userEvent.click(await screen.findByRole("button", { name: "验收会话" }));
    await waitFor(() => expect(callbacks.length).toBeGreaterThan(0));
    const stream = callbacks[callbacks.length - 1];
    act(() => { stream.activity?.(eventFixture()); });
    expect(screen.getByText("2025 年营业收入与同比增幅")).toBeTruthy();
    getChatRun.mockImplementationOnce(() => poll.promise);
    act(() => { stream.error(); });
    expect(screen.getByText(/实时连接中断，正在查询后台状态/)).toBeTruthy();
    const saved = snapshotFixture([stepFixture({ status: "succeeded", ended_offset_ms: 900, returned_count: 2, new_evidence_count: 2 })]);
    getChatRun.mockResolvedValue({ ...run, status: "completed", activities: [saved] });
    await act(async () => { stream.completed({ run_id: run.run_id, status_url: run.status_url } as Parameters<typeof stream.completed>[0]); });
    await waitFor(() => expect(screen.getByText("返回 2 条资料 · 合并新增 2 条")).toBeTruthy());
    await act(async () => { poll.resolve(run); stream.activity?.(eventFixture(stepFixture({ seq: 100, queries: ["迟到的错误更新"] }))); });
    expect(screen.queryByText("迟到的错误更新")).toBeNull();
    expect(screen.queryByText(/实时连接中断，正在查询后台状态/)).toBeNull();
    expect(screen.getByText("返回 2 条资料 · 合并新增 2 条")).toBeTruthy();
  });

  it("does not let an old management-page document response replace the new KB", async () => {
    const kbA = knowledgeBase("kb-a", "知识库 A");
    const kbB = knowledgeBase("kb-b", "知识库 B");
    const documentsA = deferred<Page<DocumentRecord>>();
    const documentsB = deferred<Page<DocumentRecord>>();
    const client = {
      listDocuments: vi.fn((id: string) => id === kbA.id
        ? documentsA.promise
        : documentsB.promise),
      listIndexingJobs: vi.fn().mockResolvedValue({ items: [], next_cursor: null }),
    } as unknown as ApiClient;
    const view = render(
      <KnowledgeBaseManagementPage {...managementProps(client, [kbA, kbB], kbA.id)} />,
    );

    view.rerender(
      <KnowledgeBaseManagementPage {...managementProps(client, [kbA, kbB], kbB.id)} />,
    );
    documentsB.resolve({
      items: [document(kbB.id, "doc-b", "B 文档")],
      next_cursor: null,
    });
    await waitFor(() => expect(screen.getByText("B 文档")).toBeTruthy());

    documentsA.resolve({
      items: [document(kbA.id, "doc-a", "A 文档")],
      next_cursor: null,
    });
    await waitFor(() => expect(screen.queryByText("A 文档")).toBeNull());
    expect(screen.getByText("B 文档")).toBeTruthy();
  });

  it("keeps late session results out of the selected KB and fails closed for a missing secret", async () => {
    const kbA = knowledgeBase("kb-a", "知识库 A");
    const kbB = knowledgeBase("kb-b", "知识库 B");
    const sessionsA = deferred<Page<ChatSession>>();
    const sessionsB = deferred<Page<ChatSession>>();
    const client = {
      getModelSettings: vi.fn().mockResolvedValue(modelSettings()),
      listKnowledgeBases: vi.fn().mockResolvedValue({
        items: [kbA, kbB],
        next_cursor: null,
      }),
      getGraphConfig: vi.fn().mockResolvedValue(null),
      getGraphSchemaProfiles: vi.fn().mockResolvedValue([]),
      listChatSessions: vi.fn((id: string) => id === kbA.id
        ? sessionsA.promise
        : sessionsB.promise),
      listChatMessages: vi.fn().mockResolvedValue({ items: [], next_cursor: null }),
      subscribeChatRun: vi.fn().mockReturnValue(() => undefined),
    } as unknown as ApiClient;

    render(
      <KnowledgeChat client={client} />,
    );
    await waitFor(() => expect(screen.getByRole("combobox", { name: "知识库" })).toBeTruthy());
    const selector = screen.getByRole("combobox", { name: "知识库" });
    await userEvent.selectOptions(selector, kbB.id);
    sessionsB.resolve({
      items: [session(kbB.id, "session-b", "B 会话")],
      next_cursor: null,
    });
    await waitFor(() => expect(screen.getByRole("button", { name: "B 会话" })).toBeTruthy());

    sessionsA.resolve({
      items: [session(kbA.id, "session-a", "A 会话")],
      next_cursor: null,
    });
    await waitFor(() => expect(screen.queryByText("A 会话")).toBeNull());
    expect((screen.getByLabelText("输入问题") as HTMLTextAreaElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "发送问题" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("shows the server's actionable indexing failure and retry affordance", async () => {
    const kb = knowledgeBase("kb-a", "知识库 A");
    const doc = document(kb.id, "doc-a", "故障文档");
    const failedJob: IndexingJob = {
      job_id: "job-a",
      kb_id: kb.id,
      document_id: doc.id,
      document_version_id: "version-a",
      indexed_document_version_id: "version-a",
      index_revision_id: "index-a",
      status: "failed",
      phase: "failed",
      progress: null,
      attempt: 1,
      build_status: "failed",
      serving_status: "candidate",
      claimed_at: null,
      heartbeat_at: null,
      next_attempt_at: null,
      error: {
        code: "EMBEDDING_PROVIDER_UNAVAILABLE",
        detail: { provider: "local" },
      },
      can_retry: true,
      created_at: NOW,
      updated_at: NOW,
    };
    const client = {
      listDocuments: vi.fn().mockResolvedValue({ items: [doc], next_cursor: null }),
      listIndexingJobs: vi.fn().mockResolvedValue({
        items: [failedJob],
        next_cursor: null,
      }),
    } as unknown as ApiClient;

    render(
      <KnowledgeBaseManagementPage {...managementProps(client, [kb], kb.id)} />,
    );

    await waitFor(() => expect(screen.getByText(/Embedding 服务不可用/)).toBeTruthy());
    expect(screen.getByRole("button", { name: "重试索引" })).toBeTruthy();
  });
});


describe("saved multi-library selection", () => {
  it("restores concrete IDs, selects all pages, and keeps empty selection local", async () => {
    const a=knowledgeBase("kb-a","知识库 A"), b=knowledgeBase("kb-b","知识库 B");
    const saved={...session(a.id,"multi-session","多库会话"),knowledge_base_id:null,knowledge_base_ids:[a.id]};
    const updateChatScope=vi.fn(async (_id:string,ids:string[]) => ({...saved,knowledge_base_ids:ids}));
    const createChatRun=vi.fn();
    const listKnowledgeBases=vi.fn(async (cursor?:string) => cursor ? {items:[b],next_cursor:null} : {items:[a],next_cursor:"second-page"});
    const api={getModelSettings:vi.fn().mockResolvedValue(modelSettings()),listKnowledgeBases,getGraphConfig:vi.fn().mockResolvedValue(null),getGraphSchemaProfiles:vi.fn().mockResolvedValue([]),listChatSessions:vi.fn().mockResolvedValue({items:[saved],next_cursor:null}),listChatMessages:vi.fn().mockResolvedValue({items:[],next_cursor:null}),updateChatScope,createChatRun} as unknown as ApiClient;
    render(<KnowledgeChat client={api} />);
    await screen.findByRole("button",{name:"多库会话"});
    await userEvent.click(screen.getByRole("button",{name:/搜索范围：/}));
    const scope=screen.getByRole("dialog",{name:"选择知识库"});
    await userEvent.click(within(scope).getByRole("button",{name:"全选"}));
    await waitFor(()=>expect(updateChatScope).toHaveBeenCalledWith("multi-session",["kb-a","kb-b"]));
    expect(listKnowledgeBases).toHaveBeenCalledWith("second-page");
    expect((within(scope).getByRole("checkbox",{name:"知识库 B"}) as HTMLInputElement).checked).toBe(true);
    await userEvent.click(within(scope).getByRole("button",{name:"清空"}));
    expect(screen.getByText("请至少选择一个知识库后发送。")).toBeTruthy();
    expect(updateChatScope).toHaveBeenCalledTimes(1);
    expect(createChatRun).not.toHaveBeenCalled();
  });
});
