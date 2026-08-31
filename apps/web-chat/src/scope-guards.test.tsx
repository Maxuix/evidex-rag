import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { KnowledgeChat } from "./App";
import { KnowledgeBaseManagementPage } from "./KnowledgeBaseManagementPage";
import type { ApiClient } from "./api/client";
import type {
  DocumentRecord,
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
});
