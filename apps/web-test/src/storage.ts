import type { UUID } from "./api/types";

const STORAGE_KEY = "rag-kb-observation-state.v1";
const MAX_TRACKED_JOBS = 100;

export interface TrackedJobReference {
  jobId: UUID;
  knowledgeBaseId: UUID;
  documentId: UUID;
  documentVersionId: UUID;
  indexedDocumentVersionId: UUID;
  indexRevisionId: UUID;
  observedAt: string;
}

export interface ActiveRunReference {
  runId: UUID;
  knowledgeBaseId: UUID;
  sessionId: UUID;
  observedAt: string;
}

interface StoredState {
  selectedKnowledgeBaseId?: UUID;
  jobs: TrackedJobReference[];
  activeRun?: ActiveRunReference;
}

export function readSelectedKnowledgeBaseId(): UUID | null {
  return readState().selectedKnowledgeBaseId ?? null;
}

export function storeSelectedKnowledgeBaseId(value: UUID | null): void {
  const state = readState();
  if (value) state.selectedKnowledgeBaseId = value;
  else delete state.selectedKnowledgeBaseId;
  writeState(state);
}

export function readTrackedJobs(knowledgeBaseId: UUID): TrackedJobReference[] {
  return readState().jobs.filter((job) => job.knowledgeBaseId === knowledgeBaseId);
}

export function trackJob(reference: Omit<TrackedJobReference, "observedAt">): void {
  const state = readState();
  const next = {
    ...reference,
    observedAt: new Date().toISOString(),
  };
  state.jobs = [next, ...state.jobs.filter((job) => job.jobId !== next.jobId)]
    .slice(0, MAX_TRACKED_JOBS);
  writeState(state);
}

export function forgetTrackedJob(jobId: UUID): void {
  const state = readState();
  state.jobs = state.jobs.filter((job) => job.jobId !== jobId);
  writeState(state);
}

export function readActiveRun(knowledgeBaseId: UUID): ActiveRunReference | null {
  const active = readState().activeRun;
  return active?.knowledgeBaseId === knowledgeBaseId ? active : null;
}

export function storeActiveRun(
  reference: Omit<ActiveRunReference, "observedAt"> | null,
): void {
  const state = readState();
  if (reference) {
    state.activeRun = { ...reference, observedAt: new Date().toISOString() };
  } else {
    delete state.activeRun;
  }
  writeState(state);
}

function readState(): StoredState {
  try {
    const value: unknown = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "null");
    if (!isRecord(value) || !Array.isArray(value.jobs)) return { jobs: [] };
    const jobs = value.jobs.filter(isTrackedJobReference).slice(0, MAX_TRACKED_JOBS);
    const state: StoredState = { jobs };
    if (typeof value.selectedKnowledgeBaseId === "string") {
      state.selectedKnowledgeBaseId = value.selectedKnowledgeBaseId;
    }
    if (isActiveRunReference(value.activeRun)) state.activeRun = value.activeRun;
    return state;
  } catch {
    return { jobs: [] };
  }
}

function writeState(state: StoredState): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // Observation state is optional; API business state remains authoritative.
  }
}

function isTrackedJobReference(value: unknown): value is TrackedJobReference {
  return isRecord(value)
    && typeof value.jobId === "string"
    && typeof value.knowledgeBaseId === "string"
    && typeof value.documentId === "string"
    && typeof value.documentVersionId === "string"
    && typeof value.indexedDocumentVersionId === "string"
    && typeof value.indexRevisionId === "string"
    && typeof value.observedAt === "string";
}

function isActiveRunReference(value: unknown): value is ActiveRunReference {
  return isRecord(value)
    && typeof value.runId === "string"
    && typeof value.knowledgeBaseId === "string"
    && typeof value.sessionId === "string"
    && typeof value.observedAt === "string";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
