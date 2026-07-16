import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { RetrievalView } from "./RetrievalView";
import type { ApiClient } from "./api/client";
import { evidencePack, ids, knowledgeBase } from "./test/fixtures";

describe("RetrievalView", () => {
  it("uses the locked exact-debug request and preserves API evidence order", async () => {
    const client = {
      queryRetrievalDebug: vi.fn().mockResolvedValue(evidencePack),
    } as unknown as ApiClient;
    const user = userEvent.setup();
    render(
      <RetrievalView
        client={client}
        knowledgeBase={knowledgeBase}
        onOpenDocument={vi.fn()}
      />,
    );

    await user.type(screen.getByLabelText("Retrieval query"), "serving evidence");
    await user.click(screen.getByRole("button", { name: "Inspect serving evidence" }));

    expect(await screen.findByText("First serving evidence.")).toBeInTheDocument();
    const first = screen.getByText("First serving evidence.");
    const second = screen.getByText("Second serving evidence.");
    expect(first.compareDocumentPosition(second) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(client.queryRetrievalDebug).toHaveBeenCalledWith(
      ids.kb,
      "serving evidence",
      10,
    );
    expect(screen.getByText("current version only", { exact: false })).toBeInTheDocument();
    expect(screen.queryByRole("checkbox")).toBeNull();
  });

  it("clears an old evidence snapshot when the selected knowledge base changes", async () => {
    const client = {
      queryRetrievalDebug: vi.fn().mockResolvedValue(evidencePack),
    } as unknown as ApiClient;
    const user = userEvent.setup();
    const { rerender } = render(
      <RetrievalView client={client} knowledgeBase={knowledgeBase} onOpenDocument={vi.fn()} />,
    );
    await user.type(screen.getByLabelText("Retrieval query"), "serving evidence");
    await user.click(screen.getByRole("button", { name: "Inspect serving evidence" }));
    expect(await screen.findByText("First serving evidence.")).toBeInTheDocument();

    rerender(
      <RetrievalView
        client={client}
        knowledgeBase={{
          ...knowledgeBase,
          id: "00000000-0000-4000-8000-000000000101",
          retrieval_defaults: { strategy: "exact_vector", top_k: 7 },
        }}
        onOpenDocument={vi.fn()}
      />,
    );

    expect(await screen.findByText("No retrieval snapshot")).toBeInTheDocument();
    expect(screen.queryByText("First serving evidence.")).toBeNull();
    expect(screen.getByLabelText("Retrieval query")).toHaveValue("");
    expect(screen.getByLabelText("Top K")).toHaveValue(7);
  });
});
