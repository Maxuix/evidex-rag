import { describe, expect, it } from "vitest";

import {
  readActiveRun,
  readTrackedJobs,
  storeActiveRun,
  trackJob,
} from "./storage";
import { ids } from "./test/fixtures";

describe("local observation storage", () => {
  it("persists identifiers only for job and run recovery", () => {
    trackJob({
      jobId: ids.job,
      knowledgeBaseId: ids.kb,
      documentId: ids.document,
      documentVersionId: ids.version,
      indexedDocumentVersionId: ids.indexedVersion,
      indexRevisionId: ids.revision,
    });
    storeActiveRun({
      runId: ids.run,
      knowledgeBaseId: ids.kb,
      sessionId: ids.session,
    });

    expect(readTrackedJobs(ids.kb)).toHaveLength(1);
    expect(readActiveRun(ids.kb)?.runId).toBe(ids.run);
    const raw = window.localStorage.getItem("rag-kb-observation-state.v1") ?? "";
    expect(raw).not.toContain("question");
    expect(raw).not.toContain("answer");
    expect(raw).not.toContain("quoted_text");
  });

  it("fails closed on corrupt storage", () => {
    window.localStorage.setItem("rag-kb-observation-state.v1", "not-json");
    expect(readTrackedJobs(ids.kb)).toEqual([]);
    expect(readActiveRun(ids.kb)).toBeNull();
  });
});
