export type ActivityStatus = "pending" | "running" | "processing" | "succeeded" | "failed" | "rejected" | "cancelled";
export interface ActivitySource {
  document_id: string;
  document_version_id: string;
  title: string;
  index_chunk_id: string | null;
  ref: string | null;
  location: string | null;
}
export interface ActivityStep {
  step_id: string;
  ordinal: number;
  seq: number;
  kind: "model" | "tool" | "system";
  name: string;
  status: ActivityStatus;
  round: number | null;
  started_offset_ms: number | null;
  ended_offset_ms: number | null;
  queries: string[];
  refs: string[];
  expression: string | null;
  include_outline: boolean | null;
  top_k: number | null;
  returned_count: number | null;
  new_evidence_count: number | null;
  document_count: number | null;
  citation_count: number | null;
  image_count: number | null;
  path_count: number | null;
  hop1_count: number | null;
  hop2_count: number | null;
  hop3_count: number | null;
  result_value: string | null;
  result_code: string | null;
  sources: ActivitySource[];
  details_truncated: boolean;
}
export interface ActivitySnapshot {
  version: "chat_activity_v1";
  attempt: number;
  steps: ActivityStep[];
  total_steps: number;
  omitted_step_count: number;
  status: "running" | "completed" | "failed" | "cancelled";
  elapsed_ms: number;
}
export interface ActivityEvent {
  version: "chat_activity_v1";
  run_id: string;
  attempt: number;
  seq: number;
  step: ActivityStep;
  elapsed_ms: number;
}
export const ACTIVE_STATUSES: ReadonlySet<string> = new Set(["pending", "running", "processing"]);
