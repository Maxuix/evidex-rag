/** Local visual QA entry. Vite's production entry remains index.html/main.tsx.
 * Uses the real UI with in-memory examples; never connects to the application's API.
 */
import { createRoot } from "react-dom/client";
import { KnowledgeChat } from "./App";
import { ApiClient } from "./api/client";
import type { ChatMessage, ChatRun, ChatRunCreate, ChatSession, KnowledgeBase, ModelSettings } from "./api/types";
import { runFixture, snapshotFixture, stepFixture } from "./execution/activityFixtures";
import "./styles.css";

const atLocalTime = (hours: number, minutes: number, daysAgo = 0) => {
  const date = new Date();
  date.setDate(date.getDate() - daysAgo);
  date.setHours(hours, minutes, 0, 0);
  return date.toISOString();
};
const today = atLocalTime(9, 30);
const kb: KnowledgeBase = {
  id: "preview-kb", name: "企业知识库", description: "本地经营报告与业务资料",
  source_change_seq: 0, active_index_revision_id: "preview-index", embedding_space_id: "preview-space",
  embedding: { strategy: "text_only", text: { embedding_space_id: "preview-space", profile_revision_id: null, dimension: 1024 }, cross_modal: null },
  parsing: { preset: "text_local_v1", profile: "docling_text_local_v1" },
  chunking: { preset: "structural_balanced_v2", profile: "structural_by_title_token_v4" },
  retrieval_defaults: { strategy: "exact_vector", top_k: 10, rerank_mode: "classic" },
  answer_policy_defaults: {}, auto_qa: { enabled: false, questions_per_chunk: 5, model_profile_revision_id: null, model_name: null, model_revision: null },
  provisioned_at: today, created_at: today, updated_at: today,
};
// The optional scope example makes README screenshots reproducible while the
// default preview keeps the selected single-library visual direction.
const knowledgeBases = new URLSearchParams(window.location.search).has("scope-demo")
  ? [kb,
    { ...kb, id: "preview-product-kb", name: "产品与技术", description: "产品规划、技术方案与发布记录" },
    { ...kb, id: "preview-team-kb", name: "团队流程", description: "团队协作规范与常用操作流程" },
  ]
  : [kb];
const settings: ModelSettings = {
  providers: [], profiles: [{
    id: "preview-model", revision_id: "preview-model-r1", revision: 1, provider_id: "preview-provider", provider_revision_id: "preview-provider-r1",
    name: "模型", kind: "chat", model: "preview", enabled: true, provider_secret_available: true,
    parameters: { type: "chat", temperature: 0.2, top_p: null, sampling_top_k: null, max_output_tokens: 4096, reasoning_effort: "off", structured_output_mode: "json_object", vision_enabled: false },
    validation_status: "valid", validation_error_code: null, validated_at: today, configuration_fingerprint: "preview",
    capability_fingerprint: "preview", compatibility_fingerprint: null, embedding_validation: null, created_at: today, updated_at: today,
  }],
  selection: { chat_profile_revision_id: "preview-model-r1", text_embedding_profile_revision_id: null, multimodal_embedding_profile_revision_id: null, updated_at: today },
};
const sessions: ChatSession[] = [
  { id: "preview-session", knowledge_base_id: kb.id, knowledge_base_ids: [kb.id], title: "年度营收分析", created_at: today, updated_at: today },
  { id: "preview-planning", knowledge_base_id: kb.id, knowledge_base_ids: [kb.id], title: "产品规划的主要内容？", created_at: today, updated_at: atLocalTime(9, 12) },
  { id: "preview-history", knowledge_base_id: kb.id, knowledge_base_ids: [kb.id], title: "公司核心业务有哪些？", created_at: atLocalTime(9, 20, 1), updated_at: atLocalTime(9, 20, 1) },
];
const answer = "## 营业收入同比增长 25%\n\n2025 年营业收入为 120 亿元，较上一年的 96 亿元增长 25%。[1]\n\n计算方式：(120 − 96) ÷ 96 = 25%。";
const initialRun = runFixture({
  run_id: "preview-run", knowledge_base_id: kb.id, knowledge_base_ids: [kb.id], session_id: sessions[0].id,
  status: "completed", assistant_status: "completed", answer, created_at: today, updated_at: today, completed_at: today,
  citations: [{ ordinal: 0, index_chunk_id: "preview-chunk", document_id: "preview-document", document_version_id: "preview-version",
    document_display_name: "2025 年度经营报告", document_original_filename: "2025-report.pdf",
    quoted_text: "2025 年公司营业收入为 120 亿元，上年同期为 96 亿元，同比增长 25%。",
    source_location: { surface_type: "page", surface_start: 12, surface_end: 12, surface_label: "营业收入" },
    score: 0.92, modality: "text", asset: null, matched_representations: ["text"] }],
  activities: [snapshotFixture([
    stepFixture({ status: "succeeded", ended_offset_ms: 2200, returned_count: 4, new_evidence_count: 4, queries: ["2025 年营业收入与同比增长"] }),
  ])],
});
const runs = new Map<string, ChatRun>([[initialRun.run_id, initialRun]]);
const messages = new Map<string, ChatMessage[]>([[sessions[0].id, [
  { id: "preview-question", session_id: sessions[0].id, run_id: initialRun.run_id, role: "user", assistant_status: null, content: "2025 年营业收入是多少？比上一年增长了多少？", created_at: today },
  { id: "preview-answer", session_id: sessions[0].id, run_id: initialRun.run_id, role: "assistant", assistant_status: "completed", content: answer, created_at: today },
]]]);
const page = <T,>(items: T[]) => ({ items, next_cursor: null });
const client = Object.assign(new ApiClient({ api_base_url: "http://127.0.0.1:9/api/v1" }), {
  listKnowledgeBases: async () => page(knowledgeBases),
  getModelSettings: async () => settings,
  getGraphConfig: async () => null,
  getGraphSchemaProfiles: async () => [],
  listChatSessions: async () => page([...sessions]),
  listChatMessages: async (id: string) => page(messages.get(id) ?? []),
  getChatRun: async (id: string) => runs.get(id) ?? initialRun,
  listDocuments: async () => page([{ id: "preview-document", kb_id: kb.id, display_name: "2025 年度经营报告", current_version: null, deleted_at: null, created_at: today, updated_at: today }]),
  listIndexingJobs: async () => page([]),
  updateChatScope: async (id: string, ids: string[]) => {
    const session = sessions.find((item) => item.id === id);
    if (!session) throw new Error("示例会话不存在");
    session.knowledge_base_ids = [...ids];
    return { ...session };
  },
  createChatSession: async (_kbId: string, title: string) => {
    const session: ChatSession = { ...sessions[0], id: crypto.randomUUID(), title };
    sessions.unshift(session);
    return session;
  },
  createChatRun: async (payload: ChatRunCreate) => {
    const run = { ...initialRun, run_id: crypto.randomUUID(), session_id: payload.session_id };
    runs.set(run.run_id, run);
    const existing = messages.get(payload.session_id) ?? [];
    messages.set(payload.session_id, [...existing,
      { id: crypto.randomUUID(), session_id: payload.session_id, run_id: run.run_id, role: "user", assistant_status: null, content: payload.message, created_at: today },
      { id: crypto.randomUUID(), session_id: payload.session_id, run_id: run.run_id, role: "assistant", assistant_status: "completed", content: answer, created_at: today },
    ]);
    return run;
  },
});

// Only this preview entry uses the example client; main.tsx uses the real runtime config.
createRoot(document.getElementById("root")!).render(<KnowledgeChat client={client} />);
