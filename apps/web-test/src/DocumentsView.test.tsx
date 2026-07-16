import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { DocumentsView } from "./DocumentsView";
import { ApiClientError, type ApiClient } from "./api/client";
import { trackJob } from "./storage";
import {
  documentUpload,
  documentRecord,
  failedJob,
  ids,
  knowledgeBase,
} from "./test/fixtures";

describe("DocumentsView", () => {
  it("uploads raw text, restores the observed job, and exposes eligible retry", async () => {
    const user = userEvent.setup();
    const client = {
      listDocuments: vi.fn().mockResolvedValue({ items: [], next_cursor: null }),
      uploadDocument: vi.fn().mockResolvedValue(documentUpload),
      getIndexingJob: vi.fn().mockResolvedValue(failedJob),
      retryIndexingJob: vi.fn().mockResolvedValue({
        ...failedJob,
        status: "queued",
        build_status: "queued",
        error: null,
        can_retry: false,
      }),
      getDocument: vi.fn(),
    } as unknown as ApiClient;

    render(
      <DocumentsView
        client={client}
        knowledgeBase={knowledgeBase}
        focusedDocumentId={null}
        focusedDocumentVersionId={null}
        onMutationPendingChange={vi.fn()}
      />,
    );
    await screen.findByText("No documents");
    const file = new File(["# Guide"], "guide.md", { type: "text/markdown" });
    await user.upload(screen.getByLabelText(/^Text file/), file);
    const submit = screen.getByRole("button", { name: "Upload and index" });
    expect(submit).toBeEnabled();
    // jsdom does not apply an uploaded File to native form validity, so submit
    // the already-enabled form directly and exercise the React handler.
    fireEvent.submit(submit.closest("form")!);

    await waitFor(() => expect(client.uploadDocument).toHaveBeenCalledWith(
      ids.kb,
      file,
      "guide.md",
      expect.any(String),
    ));
    expect(await screen.findByText("EMBEDDING_PROVIDER_UNAVAILABLE")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Retry failed job" }));
    await waitFor(() => expect(client.retryIndexingJob).toHaveBeenCalledWith(
      ids.job,
      expect.any(String),
    ));
    await waitFor(() => expect(client.getIndexingJob).toHaveBeenCalledTimes(2));
  });

  it("freezes and reuses an unconfirmed upload request until it is discarded", async () => {
    const callback = vi.fn();
    const failure = new ApiClientError("The response was not received.", {
      code: "FRONTEND_REQUEST_FAILED",
      retryable: true,
    });
    const client = {
      listDocuments: vi.fn().mockResolvedValue({ items: [], next_cursor: null }),
      uploadDocument: vi.fn().mockRejectedValue(failure),
      getIndexingJob: vi.fn(),
      getDocument: vi.fn(),
    } as unknown as ApiClient;
    const user = userEvent.setup();

    render(
      <DocumentsView
        client={client}
        knowledgeBase={knowledgeBase}
        focusedDocumentId={null}
        focusedDocumentVersionId={null}
        onMutationPendingChange={callback}
      />,
    );
    await screen.findByText("No documents");
    const file = new File(["# Guide"], "guide.md", { type: "text/markdown" });
    await user.upload(screen.getByLabelText(/^Text file/), file);
    const submit = screen.getByRole("button", { name: "Upload and index" });
    fireEvent.submit(submit.closest("form")!);

    expect(await screen.findByText("Upload was not confirmed")).toBeInTheDocument();
    expect(submit).toBeDisabled();
    expect(screen.getByLabelText(/^Text file/)).toBeDisabled();
    expect(callback).toHaveBeenLastCalledWith(true);
    const firstKey = vi.mocked(client.uploadDocument).mock.calls[0][3];

    await user.click(screen.getByRole("button", { name: "Retry same request" }));
    await waitFor(() => expect(client.uploadDocument).toHaveBeenCalledTimes(2));
    expect(vi.mocked(client.uploadDocument).mock.calls[1][3]).toBe(firstKey);

    await user.click(screen.getByRole("button", { name: "Discard and edit" }));
    expect(submit).toBeEnabled();
    expect(callback).toHaveBeenLastCalledWith(false);
  });

  it("allows only one indexing retry mutation to own the recovery slot", async () => {
    const secondJobId = "00000000-0000-4000-8000-000000000130";
    for (const jobId of [ids.job, secondJobId]) {
      trackJob({
        jobId,
        knowledgeBaseId: ids.kb,
        documentId: ids.document,
        documentVersionId: ids.version,
        indexedDocumentVersionId: ids.indexedVersion,
        indexRevisionId: ids.revision,
      });
    }
    let resolveRetry!: (value: typeof failedJob) => void;
    const retryResponse = new Promise<typeof failedJob>((resolve) => {
      resolveRetry = resolve;
    });
    const callback = vi.fn();
    const client = {
      listDocuments: vi.fn().mockResolvedValue({ items: [], next_cursor: null }),
      getIndexingJob: vi.fn().mockImplementation((jobId: string) => Promise.resolve({
        ...failedJob,
        job_id: jobId,
      })),
      retryIndexingJob: vi.fn().mockReturnValue(retryResponse),
      getDocument: vi.fn(),
    } as unknown as ApiClient;
    const user = userEvent.setup();

    render(
      <DocumentsView
        client={client}
        knowledgeBase={knowledgeBase}
        focusedDocumentId={null}
        focusedDocumentVersionId={null}
        onMutationPendingChange={callback}
      />,
    );
    const retryButtons = await screen.findAllByRole("button", {
      name: "Retry failed job",
    });
    expect(retryButtons).toHaveLength(2);
    await user.click(retryButtons[0]);

    await waitFor(() => expect(client.retryIndexingJob).toHaveBeenCalledOnce());
    for (const button of screen.getAllByRole("button", { name: /Retry/ })) {
      expect(button).toBeDisabled();
    }
    expect(screen.getByLabelText(/^Text file/)).toBeDisabled();
    expect(callback).toHaveBeenLastCalledWith(true);

    const retriedJobId = vi.mocked(client.retryIndexingJob).mock.calls[0][0];
    await act(async () => {
      resolveRetry({
        ...failedJob,
        job_id: retriedJobId,
        status: "queued",
        build_status: "queued",
        error: null,
        can_retry: false,
      });
      await retryResponse;
    });
    await waitFor(() => expect(callback).toHaveBeenLastCalledWith(false));
  });

  it("serializes list reads with writes and reflects authoritative removals", async () => {
    let resolveRefresh!: (value: {
      items: typeof documentRecord[];
      next_cursor: null;
    }) => void;
    const refreshResponse = new Promise<{
      items: typeof documentRecord[];
      next_cursor: null;
    }>((resolve) => {
      resolveRefresh = resolve;
    });
    const client = {
      listDocuments: vi.fn()
        .mockResolvedValueOnce({ items: [documentRecord], next_cursor: null })
        .mockReturnValueOnce(refreshResponse),
      getIndexingJob: vi.fn(),
      getDocument: vi.fn(),
    } as unknown as ApiClient;
    const user = userEvent.setup();

    render(
      <DocumentsView
        client={client}
        knowledgeBase={knowledgeBase}
        focusedDocumentId={null}
        focusedDocumentVersionId={null}
        onMutationPendingChange={vi.fn()}
      />,
    );
    await screen.findByRole("heading", { name: documentRecord.display_name });
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    expect(screen.getByLabelText(/^Text file/)).toBeDisabled();
    expect(screen.getByRole("button", { name: "Upload and index" })).toBeDisabled();

    await act(async () => {
      resolveRefresh({ items: [], next_cursor: null });
      await refreshResponse;
    });
    expect(await screen.findByText("No documents")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: documentRecord.display_name })).toBeNull();
  });
});
