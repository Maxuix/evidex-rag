import type { ActivityEvent, ActivitySnapshot, ActivitySource, ActivityStep } from "./activityTypes";

interface AttemptActivity {
  steps: Record<string, ActivityStep>;
  lastSeq: number;
  incomplete: boolean;
  elapsedMs: number;
  receivedAt: number;
  omitted: number;
}
export interface ActivityState {
  invalid?: boolean;
  runId: string | null;
  attempts: Record<number, AttemptActivity>;
  mode: "idle" | "live" | "disconnected";
}
export const emptyActivity = (runId: string | null): ActivityState => ({ runId, attempts: {}, mode: "idle" });
export function applyActivity(state: ActivityState, runId: string, attempt: number, event: ActivityEvent, now = Date.now()): ActivityState {
  if (event.run_id !== runId || event.attempt < attempt) return state;
  const base = state.runId === runId ? state : emptyActivity(runId);
  const knownAttempts = Object.keys(base.attempts).map(Number);
  if (event.attempt < Math.max(attempt, ...knownAttempts)) return base;
  const current = base.attempts[event.attempt] ?? { steps: {}, lastSeq: 0, incomplete: false, elapsedMs: 0, receivedAt: now, omitted: 0 };
  if ((current.steps[event.step.step_id]?.seq ?? 0) >= event.seq) return base;
  const steps = { ...current.steps, [event.step.step_id]: event.step };
  let omitted = current.omitted;
  const ordered = Object.values(steps).sort((a, b) => a.ordinal - b.ordinal);
  for (const step of ordered.slice(0, Math.max(0, ordered.length - 1024))) {
    delete steps[step.step_id]; omitted += 1;
  }
  return {
    runId, mode: "live", invalid: base.invalid, attempts: { ...base.attempts, [event.attempt]: {
      steps, omitted, lastSeq: Math.max(current.lastSeq, event.seq),
      incomplete: current.incomplete || event.seq > current.lastSeq + 1,
      elapsedMs: event.seq > current.lastSeq ? event.elapsed_ms : current.elapsedMs,
      receivedAt: event.seq > current.lastSeq ? now : current.receivedAt,
    } },
  };
}
export function disconnectActivity(state: ActivityState, runId: string): ActivityState {
  const base = state.runId === runId ? state : emptyActivity(runId);
  return { ...base, mode: "disconnected", attempts: Object.fromEntries(Object.entries(base.attempts).map(([key, value]) => [key, { ...value, incomplete: true }])) };
}

const number = (value: unknown, min = 0): value is number => Number.isSafeInteger(value) && Number(value) >= min;
const text = (value: unknown, max: number): value is string => typeof value === "string" && value.trim().length > 0 && value.length <= max;
const nullableText = (value: unknown, max: number) => value === null || text(value, max);
const uuid = (value: unknown) => typeof value === "string" && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value);
function object(value: unknown): value is Record<string, unknown> { return !!value && typeof value === "object" && !Array.isArray(value); }
function keys(value: Record<string, unknown>, expected: string[]) { return Object.keys(value).length === expected.length && expected.every(key => key in value); }
function source(value: unknown): value is ActivitySource {
  if (!object(value)) return false;
  const {knowledge_base_id, knowledge_base_name, index_revision_id, ...legacy} = value;
  return (knowledge_base_id == null || uuid(knowledge_base_id))
    && (index_revision_id == null || uuid(index_revision_id))
    && (knowledge_base_name == null || text(knowledge_base_name, 256))
    && keys(legacy, ["document_id", "document_version_id", "title", "index_chunk_id", "ref", "location"])
    && uuid(value.document_id) && uuid(value.document_version_id) && text(value.title, 512)
    && (value.index_chunk_id === null || uuid(value.index_chunk_id))
    && (value.ref === null || (typeof value.ref === "string" && /^ev_[1-9][0-9]*$/.test(value.ref)))
    && nullableText(value.location, 256);
}
const names: Record<string, ReadonlySet<string>> = {
  model: new Set(["model_round"]),
  tool: new Set(["semantic_search", "keyword_search", "read_chunk_context", "list_documents", "search_graph_relations", "calculate", "unknown"]),
  system: new Set(["load_context", "prepare_visuals", "resolve_citations", "persist_result", "close_search", "token_wrap_up"]),
};
const counts = ["round", "started_offset_ms", "ended_offset_ms", "top_k", "returned_count", "new_evidence_count", "document_count", "citation_count", "image_count", "path_count", "hop1_count", "hop2_count", "hop3_count"];
export function parseActivityStep(value: unknown): ActivityStep | null {
  if (!object(value)) return null;
  const {scope_results, ...legacy} = value;
  if (scope_results !== undefined && (!Array.isArray(scope_results) || scope_results.length > 100 || !scope_results.every(scope => object(scope) && keys(scope, ["knowledge_base_id", "name", "status", "query", "retrieved_count", "admitted_count", "displayed_count", "omitted_count"]) && uuid(scope.knowledge_base_id) && text(scope.name, 255) && text(scope.status, 80) && /^[A-Za-z0-9_]+$/.test(scope.status) && nullableText(scope.query, 2048) && [scope.retrieved_count, scope.admitted_count, scope.displayed_count, scope.omitted_count].every(count => count === null || number(count, 0))))) return null;
  if (!keys(legacy, ["step_id", "ordinal", "seq", "kind", "name", "status", ...counts, "queries", "refs", "expression", "include_outline", "result_value", "result_code", "sources", "details_truncated"])) return null;
  if (!number(value.ordinal, 1) || value.step_id !== `step_${value.ordinal}` || !number(value.seq, 1)
    || typeof value.kind !== "string" || typeof value.name !== "string" || !names[value.kind]?.has(value.name)
    || !["pending", "running", "processing", "succeeded", "failed", "rejected", "cancelled"].includes(String(value.status))) return null;
  if (!counts.every(key => value[key] === null || number(value[key], key === "round" || key === "top_k" ? 1 : 0))) return null;
  if (value.ended_offset_ms !== null && (value.started_offset_ms === null || Number(value.ended_offset_ms) < Number(value.started_offset_ms))) return null;
  if (!Array.isArray(value.queries) || value.queries.length > 3 || !value.queries.every(item => text(item, 2048))
    || !Array.isArray(value.refs) || value.refs.length > 3 || !value.refs.every(item => text(item, 128))
    || !nullableText(value.expression, 512) || !nullableText(value.result_value, 1024)
    || !(value.result_code === null || (text(value.result_code, 80) && /^[A-Za-z0-9_]+$/.test(value.result_code)))
    || !(value.include_outline === null || typeof value.include_outline === "boolean")
    || typeof value.details_truncated !== "boolean" || !Array.isArray(value.sources) || value.sources.length > 100 || !value.sources.every(source)) return null;
  return value as unknown as ActivityStep;
}
export function parseActivityEvent(value: unknown): ActivityEvent | null {
  if (!object(value) || !keys(value, ["version", "run_id", "attempt", "seq", "step", "elapsed_ms"]) || value.version !== "chat_activity_v1"
    || !uuid(value.run_id) || !number(value.attempt, 1) || !number(value.seq, 1) || !number(value.elapsed_ms)) return null;
  const step = parseActivityStep(value.step);
  return step && step.seq === value.seq ? { ...value, step } as ActivityEvent : null;
}
export function parseActivitySnapshot(value: unknown): ActivitySnapshot | null {
  if (!object(value) || !keys(value, ["version", "attempt", "steps", "total_steps", "omitted_step_count", "status", "elapsed_ms"])
    || value.version !== "chat_activity_v1" || !number(value.attempt, 1) || !number(value.total_steps) || !number(value.omitted_step_count) || !number(value.elapsed_ms)
    || !["running", "completed", "failed", "cancelled"].includes(String(value.status)) || !Array.isArray(value.steps) || value.steps.length > 1024
    || value.steps.length + value.omitted_step_count !== value.total_steps) return null;
  const steps = value.steps.map(parseActivityStep);
  if (steps.some((step, index) => !step || step.ordinal > Number(value.total_steps) || (index > 0 && step.ordinal <= (steps[index - 1]?.ordinal ?? 0)))) return null;
  return { ...value, steps } as ActivitySnapshot;
}
