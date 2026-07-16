import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ChatView } from "./ChatView";
import { ApiClientError, type ApiClient } from "./api/client";
import type { ChatRun } from "./api/types";
import { readActiveRun, storeActiveRun } from "./storage";
import {
  chatSession,
  completedRun,
  ids,
  knowledgeBase,
} from "./test/fixtures";

describe("ChatView", () => {
  it("recovers a run from safe identifiers and renders committed policy and citations", async () => {
    storeActiveRun({
      runId: ids.run,
      knowledgeBaseId: ids.kb,
      sessionId: ids.session,
    });
    const openDocument = vi.fn();
    const client = {
      listChatSessions: vi.fn().mockResolvedValue({
        items: [chatSession],
        next_cursor: null,
      }),
      listChatMessages: vi.fn().mockResolvedValue({ items: [], next_cursor: null }),
      getChatRun: vi.fn().mockResolvedValue(completedRun),
      createChatSession: vi.fn(),
      createChatRun: vi.fn(),
      subscribeChatRun: vi.fn(),
    } as unknown as ApiClient;

    render(
      <ChatView
        client={client}
        knowledgeBase={knowledgeBase}
        onOpenCitationDocument={openDocument}
        onMutationPendingChange={vi.fn()}
      />,
    );

    expect(await screen.findByText(completedRun.answer!)).toBeInTheDocument();
    expect(screen.getByText("evidence_only")).toBeInTheDocument();
    expect(screen.getByText("[1]")).toBeInTheDocument();
    expect(screen.getByText("Retained evidence text.")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", {
      name: "View current document metadata",
    }));
    expect(openDocument).toHaveBeenCalledWith(ids.document, ids.version);
    await waitFor(() => expect(client.subscribeChatRun).not.toHaveBeenCalled());
  });

  it("renders untrusted answer text without creating HTML elements", async () => {
    storeActiveRun({
      runId: ids.run,
      knowledgeBaseId: ids.kb,
      sessionId: ids.session,
    });
    const malicious = {
      ...completedRun,
      answer: "<img src=x onerror=alert(1)>",
      citations: [{ ...completedRun.citations[0], quoted_text: "<script>alert(1)</script>" }],
    };
    const client = {
      listChatSessions: vi.fn().mockResolvedValue({ items: [chatSession], next_cursor: null }),
      listChatMessages: vi.fn().mockResolvedValue({ items: [], next_cursor: null }),
      getChatRun: vi.fn().mockResolvedValue(malicious),
    } as unknown as ApiClient;
    const { container } = render(
      <ChatView
        client={client}
        knowledgeBase={knowledgeBase}
        onOpenCitationDocument={vi.fn()}
        onMutationPendingChange={vi.fn()}
      />,
    );

    expect(await screen.findByText("<img src=x onerror=alert(1)>")).toBeInTheDocument();
    expect(screen.getByText("<script>alert(1)</script>")).toBeInTheDocument();
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("script")).toBeNull();
  });

  it("closes a disconnected SSE stream and recovers the terminal status by polling", async () => {
    storeActiveRun({
      runId: ids.run,
      knowledgeBaseId: ids.kb,
      sessionId: ids.session,
    });
    const queuedRun = {
      ...completedRun,
      status: "queued" as const,
      assistant_status: "generating" as const,
      answer: null,
      citations: [],
      usage: null,
      timing: null,
      completed_at: null,
    };
    let streamHandlers: { error: () => void } | null = null;
    const closeStream = vi.fn();
    const client = {
      listChatSessions: vi.fn().mockResolvedValue({ items: [chatSession], next_cursor: null }),
      listChatMessages: vi.fn().mockResolvedValue({ items: [], next_cursor: null }),
      getChatRun: vi.fn()
        .mockResolvedValueOnce(queuedRun)
        .mockResolvedValueOnce(completedRun),
      subscribeChatRun: vi.fn().mockImplementation((_url, handlers) => {
        streamHandlers = handlers;
        return closeStream;
      }),
    } as unknown as ApiClient;

    render(
      <ChatView
        client={client}
        knowledgeBase={knowledgeBase}
        onOpenCitationDocument={vi.fn()}
        onMutationPendingChange={vi.fn()}
      />,
    );
    await waitFor(() => expect(client.subscribeChatRun).toHaveBeenCalledOnce());
    act(() => streamHandlers!.error());

    expect(await screen.findByText(completedRun.answer!)).toBeInTheDocument();
    expect(client.getChatRun).toHaveBeenCalledTimes(2);
    expect(closeStream).toHaveBeenCalled();
    await waitFor(() => expect(readActiveRun(ids.kb)).toBeNull());
  });

  it("paginates session history without replacing the messages already shown", async () => {
    const firstMessage = {
      id: ids.userMessage,
      session_id: ids.session,
      run_id: ids.run,
      role: "user" as const,
      assistant_status: null,
      content: "First retained question",
      created_at: "2026-07-15T08:00:00Z",
    };
    const secondMessage = {
      id: ids.assistantMessage,
      session_id: ids.session,
      run_id: ids.run,
      role: "assistant" as const,
      assistant_status: "completed" as const,
      content: "Second retained answer",
      created_at: "2026-07-15T08:01:00Z",
    };
    const client = {
      listChatSessions: vi.fn().mockResolvedValue({ items: [chatSession], next_cursor: null }),
      listChatMessages: vi.fn()
        .mockResolvedValueOnce({ items: [firstMessage], next_cursor: "next-history" })
        .mockResolvedValueOnce({ items: [secondMessage], next_cursor: null }),
    } as unknown as ApiClient;
    const user = userEvent.setup();

    render(
      <ChatView
        client={client}
        knowledgeBase={knowledgeBase}
        onOpenCitationDocument={vi.fn()}
        onMutationPendingChange={vi.fn()}
      />,
    );
    expect(await screen.findByText(firstMessage.content)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Load more messages" }));

    expect(await screen.findByText(secondMessage.content)).toBeInTheDocument();
    expect(screen.getByText(firstMessage.content)).toBeInTheDocument();
    expect(client.listChatMessages).toHaveBeenLastCalledWith(ids.session, "next-history");
  });

  it("freezes and reuses an unconfirmed ChatRun request until it is discarded", async () => {
    const callback = vi.fn();
    const failure = new ApiClientError("The response was not received.", {
      code: "FRONTEND_REQUEST_FAILED",
      retryable: true,
    });
    const client = {
      listChatSessions: vi.fn().mockResolvedValue({ items: [chatSession], next_cursor: null }),
      listChatMessages: vi.fn().mockResolvedValue({ items: [], next_cursor: null }),
      createChatRun: vi.fn().mockRejectedValue(failure),
    } as unknown as ApiClient;
    const user = userEvent.setup();

    render(
      <ChatView
        client={client}
        knowledgeBase={knowledgeBase}
        onOpenCitationDocument={vi.fn()}
        onMutationPendingChange={callback}
      />,
    );
    await screen.findByRole("option", { name: chatSession.title! });
    const question = screen.getByLabelText("Question");
    await user.type(question, "What is retained?");
    await user.click(screen.getByRole("button", { name: "Ask with evidence" }));

    expect(await screen.findByText("ChatRun creation was not confirmed")).toBeInTheDocument();
    expect(question).toBeDisabled();
    expect(screen.getByLabelText("Current session")).toBeDisabled();
    expect(callback).toHaveBeenLastCalledWith(true);
    const firstKey = vi.mocked(client.createChatRun).mock.calls[0][1];

    await user.click(screen.getByRole("button", { name: "Retry same request" }));
    await waitFor(() => expect(client.createChatRun).toHaveBeenCalledTimes(2));
    expect(vi.mocked(client.createChatRun).mock.calls[1][1]).toBe(firstKey);

    await user.click(screen.getByRole("button", { name: "Discard and edit" }));
    expect(question).toBeEnabled();
    expect(question).toHaveValue("What is retained?");
    expect(callback).toHaveBeenLastCalledWith(false);
  });

  it("renders a cancellation without implying that the terminal run is pending", async () => {
    storeActiveRun({
      runId: ids.run,
      knowledgeBaseId: ids.kb,
      sessionId: ids.session,
    });
    const cancelledRun = {
      ...completedRun,
      status: "cancelled" as const,
      assistant_status: "failed" as const,
      answer: null,
      citations: [],
      error: null,
    };
    const client = {
      listChatSessions: vi.fn().mockResolvedValue({ items: [chatSession], next_cursor: null }),
      listChatMessages: vi.fn().mockResolvedValue({ items: [], next_cursor: null }),
      getChatRun: vi.fn().mockResolvedValue(cancelledRun),
    } as unknown as ApiClient;

    render(
      <ChatView
        client={client}
        knowledgeBase={knowledgeBase}
        onOpenCitationDocument={vi.fn()}
        onMutationPendingChange={vi.fn()}
      />,
    );

    expect(await screen.findByText("The run was cancelled without a committed answer.")).toBeInTheDocument();
    expect(screen.queryByText("Waiting for a committed terminal result…")).toBeNull();
  });

  it("keeps the selected session history when an older request resolves last", async () => {
    const secondSession = {
      ...chatSession,
      id: "00000000-0000-4000-8000-000000000109",
      title: "Second session",
      updated_at: "2026-07-15T07:00:00Z",
    };
    let resolveFirst!: (value: {
      items: Array<{
        id: string;
        session_id: string;
        run_id: null;
        role: "user";
        assistant_status: null;
        content: string;
        created_at: string;
      }>;
      next_cursor: null;
    }) => void;
    const firstPage = new Promise<Parameters<typeof resolveFirst>[0]>((resolve) => {
      resolveFirst = resolve;
    });
    const secondMessage = {
      id: "00000000-0000-4000-8000-000000000110",
      session_id: secondSession.id,
      run_id: null,
      role: "user" as const,
      assistant_status: null,
      content: "Second session history",
      created_at: "2026-07-15T08:01:00Z",
    };
    const client = {
      listChatSessions: vi.fn().mockResolvedValue({
        items: [chatSession, secondSession],
        next_cursor: null,
      }),
      listChatMessages: vi.fn().mockImplementation((sessionId: string) => (
        sessionId === ids.session
          ? firstPage
          : Promise.resolve({ items: [secondMessage], next_cursor: null })
      )),
    } as unknown as ApiClient;
    const user = userEvent.setup();

    render(
      <ChatView
        client={client}
        knowledgeBase={knowledgeBase}
        onOpenCitationDocument={vi.fn()}
        onMutationPendingChange={vi.fn()}
      />,
    );
    await waitFor(() => expect(client.listChatMessages).toHaveBeenCalledWith(
      ids.session,
      undefined,
    ));
    await user.selectOptions(screen.getByLabelText("Current session"), secondSession.id);
    expect(await screen.findByText(secondMessage.content)).toBeInTheDocument();

    await act(async () => {
      resolveFirst({
        items: [{
          id: ids.userMessage,
          session_id: ids.session,
          run_id: null,
          role: "user",
          assistant_status: null,
          content: "Stale first-session history",
          created_at: "2026-07-15T08:00:00Z",
        }],
        next_cursor: null,
      });
      await firstPage;
    });

    expect(screen.getByText(secondMessage.content)).toBeInTheDocument();
    expect(screen.queryByText("Stale first-session history")).toBeNull();
  });

  it("retains a recoverable run identifier after a transient load failure", async () => {
    storeActiveRun({
      runId: ids.run,
      knowledgeBaseId: ids.kb,
      sessionId: ids.session,
    });
    const client = {
      listChatSessions: vi.fn().mockResolvedValue({ items: [chatSession], next_cursor: null }),
      listChatMessages: vi.fn().mockResolvedValue({ items: [], next_cursor: null }),
      getChatRun: vi.fn()
        .mockRejectedValueOnce(new ApiClientError("Temporarily unavailable.", {
          status: 503,
          code: "SERVICE_UNAVAILABLE",
          retryable: true,
        }))
        .mockResolvedValueOnce(completedRun),
    } as unknown as ApiClient;
    const user = userEvent.setup();

    render(
      <ChatView
        client={client}
        knowledgeBase={knowledgeBase}
        onOpenCitationDocument={vi.fn()}
        onMutationPendingChange={vi.fn()}
      />,
    );

    expect(await screen.findByText("Temporarily unavailable.")).toBeInTheDocument();
    expect(readActiveRun(ids.kb)?.runId).toBe(ids.run);
    await user.click(screen.getByRole("button", { name: "Retry same request" }));
    expect(await screen.findByText(completedRun.answer!)).toBeInTheDocument();
    await waitFor(() => expect(readActiveRun(ids.kb)).toBeNull());
  });

  it("ignores an older run inspection response that resolves last", async () => {
    const olderRunId = "00000000-0000-4000-8000-000000000120";
    const newerRunId = "00000000-0000-4000-8000-000000000121";
    const history = [{
      id: "00000000-0000-4000-8000-000000000122",
      session_id: ids.session,
      run_id: olderRunId,
      role: "user" as const,
      assistant_status: null,
      content: "Inspect older run",
      created_at: "2026-07-15T08:00:00Z",
    }, {
      id: "00000000-0000-4000-8000-000000000123",
      session_id: ids.session,
      run_id: newerRunId,
      role: "user" as const,
      assistant_status: null,
      content: "Inspect newer run",
      created_at: "2026-07-15T08:01:00Z",
    }];
    let resolveOlder!: (value: ChatRun) => void;
    let resolveNewer!: (value: ChatRun) => void;
    const olderResponse = new Promise<ChatRun>((resolve) => {
      resolveOlder = resolve;
    });
    const newerResponse = new Promise<ChatRun>((resolve) => {
      resolveNewer = resolve;
    });
    const client = {
      listChatSessions: vi.fn().mockResolvedValue({ items: [chatSession], next_cursor: null }),
      listChatMessages: vi.fn().mockResolvedValue({ items: history, next_cursor: null }),
      getChatRun: vi.fn().mockImplementation((runId: string) => (
        runId === olderRunId ? olderResponse : newerResponse
      )),
    } as unknown as ApiClient;
    const user = userEvent.setup();

    render(
      <ChatView
        client={client}
        knowledgeBase={knowledgeBase}
        onOpenCitationDocument={vi.fn()}
        onMutationPendingChange={vi.fn()}
      />,
    );
    const inspectButtons = await screen.findAllByRole("button", { name: "Inspect run" });
    await user.click(inspectButtons[0]);
    await user.click(inspectButtons[1]);

    await act(async () => {
      resolveNewer({
        ...completedRun,
        run_id: newerRunId,
        answer: "Newer inspected result",
      });
      await newerResponse;
    });
    expect(await screen.findByText("Newer inspected result")).toBeInTheDocument();

    await act(async () => {
      resolveOlder({
        ...completedRun,
        run_id: olderRunId,
        answer: "Stale older result",
      });
      await olderResponse;
    });
    expect(screen.getByText("Newer inspected result")).toBeInTheDocument();
    expect(screen.queryByText("Stale older result")).toBeNull();
  });

  it("does not let a stale session list erase a confirmed creation", async () => {
    let resolveRefresh!: (value: {
      items: typeof chatSession[];
      next_cursor: null;
    }) => void;
    const staleRefresh = new Promise<{
      items: typeof chatSession[];
      next_cursor: null;
    }>((resolve) => {
      resolveRefresh = resolve;
    });
    const client = {
      listChatSessions: vi.fn()
        .mockResolvedValueOnce({ items: [], next_cursor: null })
        .mockReturnValueOnce(staleRefresh),
      listChatMessages: vi.fn().mockResolvedValue({ items: [], next_cursor: null }),
      createChatSession: vi.fn().mockResolvedValue(chatSession),
    } as unknown as ApiClient;
    const user = userEvent.setup();

    render(
      <ChatView
        client={client}
        knowledgeBase={knowledgeBase}
        onOpenCitationDocument={vi.fn()}
        onMutationPendingChange={vi.fn()}
      />,
    );
    await screen.findByRole("option", { name: "No session yet" });
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    await user.type(screen.getByLabelText(/New session title/), chatSession.title!);
    await user.click(screen.getByRole("button", { name: "Create session" }));
    expect(await screen.findByRole("option", { name: chatSession.title! })).toBeInTheDocument();

    await act(async () => {
      resolveRefresh({ items: [], next_cursor: null });
      await staleRefresh;
    });
    expect(screen.getByRole("option", { name: chatSession.title! })).toBeInTheDocument();
  });
});
