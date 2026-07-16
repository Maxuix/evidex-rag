import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ObservationApp } from "./App";
import { ApiClientError, type ApiClient } from "./api/client";
import { readSelectedKnowledgeBaseId, storeSelectedKnowledgeBaseId } from "./storage";
import { ids, knowledgeBase } from "./test/fixtures";

function frontendClient(overrides: Record<string, unknown> = {}): ApiClient {
  return {
    apiOrigin: "http://127.0.0.1:8000",
    listKnowledgeBases: vi.fn().mockResolvedValue({
      items: [knowledgeBase],
      next_cursor: null,
    }),
    listDocuments: vi.fn().mockResolvedValue({ items: [], next_cursor: null }),
    getIndexingJob: vi.fn(),
    getDocument: vi.fn(),
    ...overrides,
  } as unknown as ApiClient;
}

describe("ObservationApp", () => {
  it("locks global navigation while an idempotent write remains unconfirmed", async () => {
    const failure = new ApiClientError("The response was not received.", {
      code: "FRONTEND_REQUEST_FAILED",
      retryable: true,
    });
    const client = frontendClient({
      uploadDocument: vi.fn().mockRejectedValue(failure),
    });
    const user = userEvent.setup();

    render(<ObservationApp client={client} />);
    await screen.findByText("No documents");
    const file = new File(["# Guide"], "guide.md", { type: "text/markdown" });
    await user.upload(screen.getByLabelText(/^Text file/), file);
    const upload = screen.getByRole("button", { name: "Upload and index" });
    fireEvent.submit(upload.closest("form")!);

    expect(await screen.findByText("Upload was not confirmed")).toBeInTheDocument();
    expect(screen.getByLabelText("Knowledge base")).toBeDisabled();
    expect(screen.getByLabelText("Create a local knowledge base")).toBeDisabled();
    expect(screen.getByRole("button", { name: /Chat/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Retrieval Debug/ })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Discard and edit" }));
    expect(screen.getByLabelText("Knowledge base")).toBeEnabled();
    expect(screen.getByRole("button", { name: /Chat/ })).toBeEnabled();
  });

  it("preserves a stored knowledge-base selection until later pages are loaded", async () => {
    const storedKnowledgeBase = {
      ...knowledgeBase,
      id: "00000000-0000-4000-8000-000000000101",
      name: "Stored later page",
    };
    storeSelectedKnowledgeBaseId(storedKnowledgeBase.id);
    const listKnowledgeBases = vi.fn().mockImplementation((cursor?: string) => (
      Promise.resolve(cursor
        ? { items: [storedKnowledgeBase], next_cursor: null }
        : { items: [knowledgeBase], next_cursor: "page-2" })
    ));
    const client = frontendClient({ listKnowledgeBases });
    const user = userEvent.setup();

    render(<ObservationApp client={client} />);
    await user.click(await screen.findByRole("button", { name: "Load more" }));

    expect((await screen.findAllByText("Stored later page")).length).toBe(2);
    expect(readSelectedKnowledgeBaseId()).toBe(storedKnowledgeBase.id);
    expect(listKnowledgeBases).toHaveBeenLastCalledWith("page-2");
    await waitFor(() => expect(client.listDocuments).toHaveBeenCalledWith(
      storedKnowledgeBase.id,
      undefined,
    ));
    expect(ids.kb).not.toBe(storedKnowledgeBase.id);
  });

  it("does not let a stale list response erase a confirmed knowledge-base create", async () => {
    let resolveList!: (value: {
      items: typeof knowledgeBase[];
      next_cursor: null;
    }) => void;
    const listResponse = new Promise<{
      items: typeof knowledgeBase[];
      next_cursor: null;
    }>((resolve) => {
      resolveList = resolve;
    });
    const created = {
      ...knowledgeBase,
      id: "00000000-0000-4000-8000-000000000140",
      name: "Confirmed creation",
    };
    const client = frontendClient({
      listKnowledgeBases: vi.fn().mockReturnValue(listResponse),
      createKnowledgeBase: vi.fn().mockResolvedValue(created),
    });
    const user = userEvent.setup();

    render(<ObservationApp client={client} />);
    await user.type(
      screen.getByLabelText("Create a local knowledge base"),
      created.name,
    );
    await user.click(screen.getByRole("button", { name: "Create" }));
    expect((await screen.findAllByText(created.name)).length).toBe(2);

    await act(async () => {
      resolveList({ items: [knowledgeBase], next_cursor: null });
      await listResponse;
    });
    expect(screen.getAllByText(created.name)).toHaveLength(2);
    expect(screen.getByLabelText("Knowledge base")).toHaveValue(created.id);
  });
});
