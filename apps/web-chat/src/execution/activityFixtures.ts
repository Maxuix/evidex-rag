import type { ChatRun } from "../api/types";
import type { ActivityEvent, ActivitySnapshot, ActivitySource, ActivityStep } from "./activityTypes";

export const RUN_ID = "01900000-0000-7000-8000-000000000001";
export const sourceFixture: ActivitySource = {
  document_id: "01900000-0000-7000-8000-000000000002", document_version_id: "01900000-0000-7000-8000-000000000003",
  index_chunk_id: "01900000-0000-7000-8000-000000000004", title: "2025 年度经营报告", ref: "ev_1", location: "第 12 页 · 营业收入",
};
export function stepFixture(overrides: Partial<ActivityStep> = {}): ActivityStep {
  return {
    step_id: "step_1", ordinal: 1, seq: 1, kind: "tool", name: "semantic_search", status: "running", round: 1,
    started_offset_ms: 100, ended_offset_ms: null, queries: ["2025 年营业收入与同比增幅"], refs: [], expression: null,
    include_outline: null, top_k: 10, returned_count: null, new_evidence_count: null, document_count: null,
    citation_count: null, image_count: null, path_count: null, hop1_count: null, hop2_count: null, hop3_count: null,
    result_value: null, result_code: null, sources: [], details_truncated: false, ...overrides,
  };
}
export function eventFixture(step = stepFixture(), attempt = 1): ActivityEvent {
  return { version: "chat_activity_v1", run_id: RUN_ID, attempt, seq: step.seq, step, elapsed_ms: 3200 };
}
export function snapshotFixture(steps: ActivityStep[], attempt = 1): ActivitySnapshot {
  return { version: "chat_activity_v1", attempt, steps, total_steps: Math.max(0, ...steps.map(step => step.ordinal)), omitted_step_count: 0, status: "completed", elapsed_ms: 12300 };
}
export function runFixture(overrides: Partial<ChatRun> = {}): ChatRun {
  return {
    run_id: RUN_ID, knowledge_base_id: "kb-1", session_id: "session-1", index_revision_id: "index-1", status: "running",
    assistant_status: "generating", attempt: 1, answer: null, citations: [], status_url: "/run", events_url: "/events",
    effective_answer_policy: {}, agent: { version: "native_tool_calling_agent_v6", budget: { max_total_tokens: 100000 }, trace: null },
    retrieval: { mode: "text", profile_version: "text_v1", strategy: "exact_vector", top_k: 10, rerank_mode: "none" },
    model: { profile_revision_id: null, profile_name: "mimo-v2.5", provider_name: "OpenCode Go", model: "mimo-v2.5", revision: null, temperature: 0.2, top_p: null, sampling_top_k: null, max_output_tokens: 4096, reasoning_effort: "off" },
    error: null, created_at: "2026-09-05T12:00:00Z", updated_at: "2026-09-05T12:00:00Z", completed_at: null, ...overrides,
  };
}
