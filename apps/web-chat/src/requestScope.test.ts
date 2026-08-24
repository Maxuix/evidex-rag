import { describe, expect, it } from "vitest";

import {
  isChatViewScopeCurrent,
  isDocumentScopeCurrent,
  isManagementKbScopeCurrent,
  isRequestSequenceCurrent,
} from "./requestScope";

describe("request scope guards", () => {
  it("rejects stale chat knowledge-base or session responses", () => {
    const token = { generation: 4, knowledgeBaseId: "kb-a", sessionId: "session-a" };

    expect(isChatViewScopeCurrent(token, token)).toBe(true);
    expect(isChatViewScopeCurrent(token, {
      generation: 5,
      knowledgeBaseId: "kb-a",
      sessionId: "session-a",
    })).toBe(false);
    expect(isChatViewScopeCurrent(token, {
      generation: 4,
      knowledgeBaseId: "kb-b",
      sessionId: "session-a",
    })).toBe(false);
    expect(isChatViewScopeCurrent(token, {
      generation: 4,
      knowledgeBaseId: "kb-a",
      sessionId: "session-b",
    })).toBe(false);
  });

  it("keeps document and request-sequence guards independent", () => {
    const management = { generation: 2, knowledgeBaseId: "kb-a" };
    const document = { ...management, documentId: "doc-a" };

    expect(isManagementKbScopeCurrent(management, management)).toBe(true);
    expect(isManagementKbScopeCurrent(management, {
      generation: 2,
      knowledgeBaseId: "kb-b",
    })).toBe(false);
    expect(isDocumentScopeCurrent(document, document)).toBe(true);
    expect(isDocumentScopeCurrent(document, {
      ...management,
      documentId: "doc-b",
    })).toBe(false);
    expect(isRequestSequenceCurrent(7, 7)).toBe(true);
    expect(isRequestSequenceCurrent(7, 8)).toBe(false);
  });
});
