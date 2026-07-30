import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiClient, ApiClientError } from "./api/client";
import type {
  AnswerStyle,
  ChatMessage,
  ChatPreviewDeltaEvent,
  ChatPreviewResetEvent,
  ChatRun,
  ChatRunCreate,
  ChatRunFinalContext,
  ChatSession,
  InsufficiencyPolicy,
  KnowledgeBase,
} from "./api/types";
import {
  AssetPreview,
  EmptyState,
  JsonDetails,
  KeyValueGrid,
  ProblemNotice,
  StatusBadge,
  formatDate,
  shortId,
} from "./components";
import { readActiveRun, storeActiveRun } from "./storage";

interface PendingRunSubmission {
  payload: ChatRunCreate;
  idempotencyKey: string;
}

interface RunLoadRequest {
  runId: string;
  clearStoredOnNotFound: boolean;
}

type DeliveryMode = "idle" | "sse-connecting" | "sse" | "polling" | "terminal";

interface PreviewDiagnostics {
  runId: string | null;
  attempt: number;
  lastSeq: number;
  content: string;
  phase: "idle" | "streaming" | "verifying" | "discarded" | "replaced";
  deltaEvents: number;
  deltaBytes: number;
  resetEvents: number;
  gapDetected: boolean;
  lastResetReason: ChatPreviewResetEvent["reason"] | null;
}

export function ChatView({
  client,
  knowledgeBase,
  onOpenCitationDocument,
  onMutationPendingChange,
}: {
  client: ApiClient;
  knowledgeBase: KnowledgeBase;
  onOpenCitationDocument: (documentId: string, versionId: string) => void;
  onMutationPendingChange: (pending: boolean) => void;
}) {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [sessionsCursor, setSessionsCursor] = useState<string | null>(null);
  const [sessionsLoading, setSessionsLoading] = useState(true);
  const [sessionsError, setSessionsError] = useState<unknown | null>(null);
  const [selectedSessionId, setSelectedSessionId] = useState("");
  const [sessionTitle, setSessionTitle] = useState("");
  const [creatingSession, setCreatingSession] = useState(false);
  const [pendingSessionTitle, setPendingSessionTitle] = useState<string | null>(null);
  const [sessionCreateError, setSessionCreateError] = useState<unknown | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [messagesCursor, setMessagesCursor] = useState<string | null>(null);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [messagesError, setMessagesError] = useState<unknown | null>(null);
  const [message, setMessage] = useState("");
  const [answerStyle, setAnswerStyle] = useState<AnswerStyle>(
    knowledgeBase.answer_policy_defaults.answer_style,
  );
  const [insufficiencyPolicy, setInsufficiencyPolicy] = useState<InsufficiencyPolicy>(
    knowledgeBase.answer_policy_defaults.insufficiency_policy,
  );
  const [topK, setTopK] = useState(knowledgeBase.retrieval_defaults.top_k);
  const [submitting, setSubmitting] = useState(false);
  const [pendingSubmission, setPendingSubmission] = useState<PendingRunSubmission | null>(null);
  const [submissionError, setSubmissionError] = useState<unknown | null>(null);
  const [run, setRun] = useState<ChatRun | null>(null);
  const [finalContext, setFinalContext] = useState<ChatRunFinalContext | null>(null);
  const [finalContextLoading, setFinalContextLoading] = useState(false);
  const [finalContextError, setFinalContextError] = useState<unknown | null>(null);
  const [runLoadError, setRunLoadError] = useState<unknown | null>(null);
  const [runLoadRequest, setRunLoadRequest] = useState<RunLoadRequest | null>(null);
  const [deliveryMode, setDeliveryMode] = useState<DeliveryMode>("idle");
  const [deliveryError, setDeliveryError] = useState<unknown | null>(null);
  const [preview, setPreview] = useState<PreviewDiagnostics>(
    () => emptyPreviewDiagnostics(null),
  );
  const forcePolling = useRef<(() => void) | null>(null);
  const selectedSessionIdRef = useRef("");
  const messageRequestGeneration = useRef(0);
  const runRequestGeneration = useRef(0);
  selectedSessionIdRef.current = selectedSessionId;
  const runIsTerminal = run !== null && isTerminal(run);
  const mutationPending = pendingSessionTitle !== null || pendingSubmission !== null;
  const sessionBusy = (
    (run !== null && !runIsTerminal && run.session_id === selectedSessionId)
    || messages.some((item) => item.assistant_status === "generating")
  );

  useEffect(() => {
    onMutationPendingChange(mutationPending);
    return () => onMutationPendingChange(false);
  }, [mutationPending, onMutationPendingChange]);

  useEffect(() => () => {
    messageRequestGeneration.current += 1;
    runRequestGeneration.current += 1;
  }, []);

  const visibleSessions = useMemo(
    () => sessions.filter((session) => session.knowledge_base_id === knowledgeBase.id),
    [knowledgeBase.id, sessions],
  );

  const loadSessions = useCallback(async (cursor?: string) => {
    setSessionsLoading(true);
    setSessionsError(null);
    try {
      const page = await client.listChatSessions(cursor);
      // Session creation can complete while a previously-started list read is
      // still in flight. Merge the snapshot so it cannot erase that result.
      setSessions((current) => mergeSessions(current, page.items));
      setSessionsCursor(page.next_cursor);
    } catch (error) {
      setSessionsError(error);
    } finally {
      setSessionsLoading(false);
    }
  }, [client]);

  const loadMessages = useCallback(async (sessionId: string, cursor?: string) => {
    if (!sessionId) {
      setMessages([]);
      setMessagesCursor(null);
      return;
    }
    if (selectedSessionIdRef.current !== sessionId) return;
    const generation = ++messageRequestGeneration.current;
    setMessagesLoading(true);
    setMessagesError(null);
    try {
      const page = await client.listChatMessages(sessionId, cursor);
      if (
        generation !== messageRequestGeneration.current
        || selectedSessionIdRef.current !== sessionId
      ) return;
      setMessages((current) => cursor
        ? mergeMessages(current, page.items)
        : page.items);
      setMessagesCursor(page.next_cursor);
    } catch (error) {
      if (
        generation === messageRequestGeneration.current
        && selectedSessionIdRef.current === sessionId
      ) setMessagesError(error);
    } finally {
      if (
        generation === messageRequestGeneration.current
        && selectedSessionIdRef.current === sessionId
      ) setMessagesLoading(false);
    }
  }, [client]);

  const loadRun = useCallback(async (request: RunLoadRequest) => {
    const generation = ++runRequestGeneration.current;
    setRunLoadRequest(request);
    setRunLoadError(null);
    try {
      const value = await client.getChatRun(request.runId);
      if (generation !== runRequestGeneration.current) return;
      if (value.knowledge_base_id !== knowledgeBase.id) {
        throw new ApiClientError("The run is outside the selected knowledge base.", {
          code: "FRONTEND_RUN_SCOPE_MISMATCH",
        });
      }
      setRun(value);
      setSelectedSessionId(value.session_id);
      setRunLoadRequest(null);
      if (!isTerminal(value)) {
        storeActiveRun({
          runId: value.run_id,
          knowledgeBaseId: value.knowledge_base_id,
          sessionId: value.session_id,
        });
      }
    } catch (error) {
      if (generation !== runRequestGeneration.current) return;
      if (
        request.clearStoredOnNotFound
        && error instanceof ApiClientError
        && error.status === 404
      ) {
        const active = readActiveRun(knowledgeBase.id);
        if (active?.runId === request.runId) storeActiveRun(null);
      }
      setRunLoadError(error);
    }
  }, [client, knowledgeBase.id]);

  useEffect(() => {
    setSessions([]);
    setSelectedSessionId("");
    setMessages([]);
    setMessagesCursor(null);
    setRun(null);
    setPreview(emptyPreviewDiagnostics(null));
    setRunLoadError(null);
    setDeliveryMode("idle");
    setAnswerStyle(knowledgeBase.answer_policy_defaults.answer_style);
    setInsufficiencyPolicy(knowledgeBase.answer_policy_defaults.insufficiency_policy);
    setTopK(knowledgeBase.retrieval_defaults.top_k);
    void loadSessions();

    const active = readActiveRun(knowledgeBase.id);
    if (active) {
      setSelectedSessionId(active.sessionId);
      void loadRun({ runId: active.runId, clearStoredOnNotFound: true });
    }
  }, [knowledgeBase, loadRun, loadSessions]);

  useEffect(() => {
    setFinalContext(null);
    setFinalContextLoading(false);
    setFinalContextError(null);
  }, [run?.run_id]);

  useEffect(() => {
    if (!selectedSessionId && visibleSessions.length > 0) {
      setSelectedSessionId(visibleSessions[0].id);
    }
  }, [selectedSessionId, visibleSessions]);

  useEffect(() => {
    messageRequestGeneration.current += 1;
    setMessages([]);
    setMessagesCursor(null);
    setMessagesLoading(false);
    setMessagesError(null);
    if (selectedSessionId) void loadMessages(selectedSessionId);
  }, [loadMessages, selectedSessionId]);

  useEffect(() => {
    setPreview(emptyPreviewDiagnostics(run?.run_id ?? null));
  }, [run?.run_id]);

  useEffect(() => {
    if (!run || runIsTerminal) {
      forcePolling.current = null;
      if (run && runIsTerminal) {
        setPreview((current) => terminalPreviewDiagnostics(current, run.run_id));
        setDeliveryMode("terminal");
        const active = readActiveRun(knowledgeBase.id);
        if (active?.runId === run.run_id) storeActiveRun(null);
        void loadMessages(run.session_id);
      }
      return;
    }

    let cancelled = false;
    let closeStream: (() => void) | null = null;
    let pollTimer: number | null = null;
    let pollingStarted = false;

    const schedulePoll = (delay = 0) => {
      if (cancelled) return;
      if (pollTimer !== null) window.clearTimeout(pollTimer);
      pollTimer = window.setTimeout(poll, delay);
    };

    const poll = async () => {
      if (cancelled) return;
      try {
        const next = await client.getChatRun(run.run_id);
        if (cancelled) return;
        setRun(next);
        setDeliveryError(null);
        if (!isTerminal(next)) schedulePoll(1100 + Math.floor(Math.random() * 350));
      } catch (error) {
        if (cancelled) return;
        setDeliveryError(error);
        schedulePoll(2200);
      }
    };

    const beginPolling = () => {
      if (cancelled || pollingStarted) return;
      pollingStarted = true;
      closeStream?.();
      closeStream = null;
      setPreview((current) => discardPreviewDiagnostics(current, run.run_id));
      setDeliveryMode("polling");
      schedulePoll();
    };

    const settleFromStatus = async (statusUrl: string) => {
      closeStream?.();
      closeStream = null;
      setPreview((current) => terminalPreviewDiagnostics(current, run.run_id));
      try {
        const next = await client.getChatRun(statusUrl);
        if (!cancelled) {
          setRun(next);
          if (!isTerminal(next)) beginPolling();
        }
      } catch (error) {
        if (!cancelled) {
          setDeliveryError(error);
          beginPolling();
        }
      }
    };

    forcePolling.current = beginPolling;
    setDeliveryMode("sse-connecting");
    setDeliveryError(null);
    closeStream = client.subscribeChatRun(run.events_url, {
      open: () => {
        if (!cancelled) setDeliveryMode("sse");
      },
      completed: (event) => {
        if (!cancelled) void settleFromStatus(event.status_url);
      },
      failed: (event) => {
        if (!cancelled) void settleFromStatus(event.status_url);
      },
      previewDelta: (event) => {
        if (!cancelled) {
          setPreview(
            (current) => applyPreviewDeltaDiagnostics(current, run.run_id, event),
          );
        }
      },
      previewReset: (event) => {
        if (!cancelled) {
          setPreview(
            (current) => applyPreviewResetDiagnostics(current, run.run_id, event),
          );
        }
      },
      previewInvalid: () => {
        if (!cancelled) {
          setPreview(
            (current) => discardPreviewDiagnostics(current, run.run_id),
          );
        }
      },
      error: beginPolling,
    });

    return () => {
      cancelled = true;
      forcePolling.current = null;
      closeStream?.();
      if (pollTimer !== null) window.clearTimeout(pollTimer);
    };
  }, [client, knowledgeBase.id, loadMessages, run?.run_id, runIsTerminal]);

  const createSession = async (event: React.FormEvent) => {
    event.preventDefault();
    if (mutationPending) return;
    const requestedTitle = sessionTitle.trim();
    setPendingSessionTitle(requestedTitle);
    setCreatingSession(true);
    setSessionCreateError(null);
    try {
      const value = await client.createChatSession(
        knowledgeBase.id,
        requestedTitle || null,
      );
      setSessions((current) => mergeSessions(current, [value]));
      setSelectedSessionId(value.id);
      setSessionTitle("");
      setPendingSessionTitle(null);
    } catch (error) {
      setSessionCreateError(error);
    } finally {
      setCreatingSession(false);
    }
  };

  const submitRun = async (event: React.FormEvent) => {
    event.preventDefault();
    if (mutationPending || sessionBusy || !selectedSessionId || !message.trim()) return;
    const pending: PendingRunSubmission = {
      idempotencyKey: crypto.randomUUID(),
      payload: {
        session_id: selectedSessionId,
        knowledge_base_id: knowledgeBase.id,
        message: message.trim(),
        answer_policy: {
          answer_style: answerStyle,
          insufficiency_policy: insufficiencyPolicy,
        },
        retrieval: { mode: "vector", top_k: topK, rerank: true },
      },
    };
    setPendingSubmission(pending);
    await performRunSubmission(pending);
  };

  const performRunSubmission = async (pending: PendingRunSubmission) => {
    // A newly submitted authoritative write supersedes any outstanding history
    // inspection read. Its late response must not replace the created run.
    runRequestGeneration.current += 1;
    setRunLoadRequest(null);
    setRunLoadError(null);
    setSubmitting(true);
    setSubmissionError(null);
    setDeliveryError(null);
    try {
      const value = await client.createChatRun(pending.payload, pending.idempotencyKey);
      setRun(value);
      setSelectedSessionId(value.session_id);
      setMessage("");
      setPendingSubmission(null);
      storeActiveRun(isTerminal(value) ? null : {
        runId: value.run_id,
        knowledgeBaseId: value.knowledge_base_id,
        sessionId: value.session_id,
      });
      await loadMessages(value.session_id);
    } catch (error) {
      setSubmissionError(error);
    } finally {
      setSubmitting(false);
    }
  };

  const openRunFromHistory = async (runId: string) => {
    await loadRun({ runId, clearStoredOnNotFound: false });
  };

  const loadFinalContext = async () => {
    if (!run || finalContextLoading) return;
    setFinalContextLoading(true);
    setFinalContextError(null);
    try {
      const value = await client.getChatRunFinalContext(run.final_context_url);
      if (value.run_id !== run.run_id) {
        throw new ApiClientError("The final context belongs to a different ChatRun.", {
          code: "FRONTEND_FINAL_CONTEXT_SCOPE_MISMATCH",
        });
      }
      setFinalContext(value);
    } catch (error) {
      setFinalContextError(error);
    } finally {
      setFinalContextLoading(false);
    }
  };

  return (
    <div className="view-stack chat-layout">
      <section className="panel sessions-panel">
        <div className="panel-heading split-heading">
          <div>
            <p className="eyebrow">Conversation facts</p>
            <h2>Sessions</h2>
          </div>
          <button className="button subtle" type="button" onClick={() => void loadSessions()}>
            Refresh
          </button>
        </div>
        {sessionsError ? (
          <ProblemNotice error={sessionsError} onRetry={() => void loadSessions()} />
        ) : null}
        <label>
          Current session
          <select
            value={selectedSessionId}
            onChange={(event) => setSelectedSessionId(event.target.value)}
            disabled={mutationPending || (sessionsLoading && visibleSessions.length === 0)}
          >
            {visibleSessions.length === 0 ? <option value="">No session yet</option> : null}
            {visibleSessions.map((session) => (
              <option value={session.id} key={session.id}>
                {session.title || `Session ${shortId(session.id)}`}
              </option>
            ))}
          </select>
        </label>
        {sessionsCursor ? (
          <button
            className="button text-button"
            type="button"
            onClick={() => void loadSessions(sessionsCursor)}
          >
            Load older sessions
          </button>
        ) : null}
        <form className="compact-form" onSubmit={createSession}>
          <label>
            New session title <span className="optional">optional</span>
            <input
              value={sessionTitle}
              onChange={(event) => setSessionTitle(event.target.value)}
              maxLength={512}
              placeholder="Implementation notes"
              disabled={mutationPending}
            />
          </label>
          <button className="button secondary" type="submit" disabled={mutationPending}>
            {creatingSession ? "Creating…" : "Create session"}
          </button>
        </form>
        {sessionCreateError ? (
          <ProblemNotice
            error={sessionCreateError}
            title="Session creation was not confirmed"
            onDiscard={() => {
              setPendingSessionTitle(null);
              setSessionCreateError(null);
            }}
            discardLabel="Acknowledge and create another"
          />
        ) : null}
        <div className="history-list" aria-busy={messagesLoading}>
          <div className="subheading-row">
            <h3>History</h3>
            <span>{messages.length} messages</span>
          </div>
          {messagesError ? <ProblemNotice error={messagesError} /> : null}
          {messagesLoading && messages.length === 0 ? <p>Loading history…</p> : null}
          {!messagesLoading && messages.length === 0 ? (
            <EmptyState title="No messages" description="Create a run to begin this session." />
          ) : (
            messages.map((item) => (
              <article className={`history-message role-${item.role}`} key={item.id}>
                <div className="message-meta">
                  <strong>{item.role}</strong>
                  <span>{formatDate(item.created_at)}</span>
                </div>
                <p>{item.content || (item.assistant_status === "generating" ? "Generating…" : "")}</p>
                {item.run_id ? (
                  <button
                    className="button text-button"
                    type="button"
                    disabled={mutationPending}
                    onClick={() => void openRunFromHistory(item.run_id!)}
                  >
                    Inspect run
                  </button>
                ) : null}
              </article>
            ))
          )}
          {messagesCursor ? (
            <button
              className="button text-button"
              type="button"
              onClick={() => void loadMessages(selectedSessionId, messagesCursor)}
            >
              Load more messages
            </button>
          ) : null}
        </div>
      </section>

      <div className="chat-main-stack">
        <section className="panel">
          <div className="panel-heading split-heading">
            <div>
              <p className="eyebrow">Evidence-only answering</p>
              <h2>Create a durable ChatRun</h2>
              <p>The server freezes retrieval and policy inputs before worker execution.</p>
            </div>
            <span className="policy-lock">P1 policy</span>
          </div>
          <form className="form-grid chat-form" onSubmit={submitRun}>
            <label className="wide-field">
              Question
              <textarea
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                maxLength={32768}
                rows={4}
                placeholder="What evidence supports the current policy?"
                required
                disabled={mutationPending || sessionBusy}
              />
            </label>
            <label>
              Answer style
              <select
                value={answerStyle}
                onChange={(event) => setAnswerStyle(event.target.value as AnswerStyle)}
                disabled={mutationPending || sessionBusy}
              >
                <option value="concise">Concise</option>
                <option value="summary">Summary</option>
              </select>
            </label>
            <label>
              If evidence is incomplete
              <select
                value={insufficiencyPolicy}
                onChange={(event) => setInsufficiencyPolicy(
                  event.target.value as InsufficiencyPolicy,
                )}
                disabled={mutationPending || sessionBusy}
              >
                <option value="refuse">Refuse</option>
                <option value="partial_answer">Give a partial answer</option>
              </select>
            </label>
            <label>
              Retrieval Top K
              <input
                type="number"
                min={1}
                max={100}
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
                disabled={mutationPending || sessionBusy}
              />
            </label>
            <div className="form-actions">
              <button
                className="button primary"
                type="submit"
                disabled={mutationPending || sessionBusy || !selectedSessionId || !message.trim()}
              >
                {submitting ? "Creating run…" : "Ask with evidence"}
              </button>
            </div>
          </form>
          {sessionBusy ? (
            <p className="delivery-label">This session is processing a ChatRun; wait for its terminal state.</p>
          ) : null}
          {submissionError ? (
            <ProblemNotice
              error={submissionError}
              title="ChatRun creation was not confirmed"
              onRetry={pendingSubmission
                ? () => void performRunSubmission(pendingSubmission)
                : undefined}
              onDiscard={pendingSubmission ? () => {
                setPendingSubmission(null);
                setSubmissionError(null);
              } : undefined}
              discardLabel="Discard and edit"
            />
          ) : null}
        </section>

        <section className="panel run-panel" aria-live="polite">
          <div className="panel-heading split-heading">
            <div>
              <p className="eyebrow">Authoritative run state</p>
              <h2>Run inspector</h2>
            </div>
            {run ? (
              <div className="badge-row">
                <StatusBadge value={run.status} />
                <span className="delivery-label">{deliveryLabel(deliveryMode)}</span>
              </div>
            ) : null}
          </div>
          {runLoadError ? (
            <ProblemNotice
              error={runLoadError}
              onRetry={runLoadRequest ? () => void loadRun(runLoadRequest) : undefined}
              onDiscard={() => {
                setRunLoadRequest(null);
                setRunLoadError(null);
              }}
            />
          ) : null}
          {deliveryError ? (
            <ProblemNotice
              error={deliveryError}
              title="Live delivery is unavailable; status recovery continues"
            />
          ) : null}
          {!run ? (
            <EmptyState
              title="No run selected"
              description="Create a run or inspect one from session history."
            />
          ) : (
            <>
              <KeyValueGrid values={[
                ["Run ID", shortId(run.run_id)],
                ["Revision", shortId(run.index_revision_id)],
                ["Attempt", run.attempt],
                ["Assistant", run.assistant_status],
                ["Updated", formatDate(run.updated_at)],
                ["Completed", formatDate(run.completed_at)],
              ]} />
              <section className="preview-diagnostics">
                <div className="subheading-row">
                  <div>
                    <p className="eyebrow">Ephemeral delivery</p>
                    <h3>Unvalidated answer preview</h3>
                  </div>
                  <span className={`preview-phase phase-${preview.phase}`}>
                    {preview.phase}
                  </span>
                </div>
                <KeyValueGrid values={[
                  ["Attempt", preview.attempt || "none"],
                  ["Last sequence", preview.lastSeq || "none"],
                  ["Delta events", preview.deltaEvents],
                  ["Preview bytes", preview.deltaBytes],
                  ["Reset events", preview.resetEvents],
                  ["Gap / invalid", preview.gapDetected ? "yes" : "no"],
                  ["Last reset", preview.lastResetReason || "none"],
                ]} />
                {preview.content ? (
                  <div className="diagnostic-preview-copy">{preview.content}</div>
                ) : (
                  <p className="context-disclosure">
                    {preview.phase === "verifying"
                      ? "Preview cleared while the server validates or repairs the final answer."
                      : preview.phase === "replaced"
                        ? "The authoritative terminal result replaced all preview content."
                        : "No replayable preview content is available."}
                  </p>
                )}
              </section>
              {!isTerminal(run) && deliveryMode !== "polling" ? (
                <button
                  className="button text-button"
                  type="button"
                  onClick={() => forcePolling.current?.()}
                >
                  Use status polling
                </button>
              ) : null}
              <div className="policy-card">
                <div>
                  <p className="eyebrow">Effective policy</p>
                  <strong>{run.effective_answer_policy.policy_version.toUpperCase()}</strong>
                </div>
                <KeyValueGrid values={[
                  ["Grounding", run.effective_answer_policy.grounding_policy],
                  ["Style", run.effective_answer_policy.answer_style],
                  ["Insufficiency", run.effective_answer_policy.insufficiency_policy],
                  ["Citations", run.effective_answer_policy.citation_granularity],
                ]} />
              </div>
              <div className="policy-card">
                <div>
                  <p className="eyebrow">Query context</p>
                  <strong>{run.query_context.status}</strong>
                </div>
                <KeyValueGrid values={[
                  ["Strategy", run.query_context.strategy],
                  ["History turns", run.query_context.history_turn_count],
                  ["History tokens", run.query_context.history_token_count],
                  ["Truncated", run.query_context.history_truncated ? "yes" : "no"],
                  ["Rewrite source", run.query_context.rewrite_source || "pending/legacy"],
                ]} />
                {run.query_context.standalone_query ? (
                  <p>{run.query_context.standalone_query}</p>
                ) : null}
              </div>
              {run.error ? (
                <div className="run-failure">
                  <p className="eyebrow">{run.error.code}</p>
                  <h3>The run ended without a fabricated answer.</h3>
                  <p>{run.error.retryable
                    ? "The recorded failure is retryable; create a new run when ready."
                    : "The recorded failure is terminal for this run."}</p>
                  <JsonDetails label="Failure detail" value={run.error.detail} />
                </div>
              ) : null}
              {run.status === "cancelled" && !run.error ? (
                <div className="run-failure">
                  <p className="eyebrow">RUN_CANCELLED</p>
                  <h3>The run was cancelled without a committed answer.</h3>
                  <p>The authoritative terminal status is cancelled.</p>
                </div>
              ) : null}
              {run.answer ? (
                <article className="answer-card">
                  <p className="eyebrow">Committed answer</p>
                  <div className="answer-copy">{run.answer}</div>
                </article>
              ) : !run.error && !isTerminal(run) ? (
                <div className="pending-answer">
                  <span className="pulse-dot" aria-hidden="true" />
                  Waiting for a committed terminal result…
                </div>
              ) : null}
              {runIsTerminal ? (
                <section className="final-context-panel">
                  <div className="subheading-row">
                    <div>
                      <p className="eyebrow">Diagnostic model input</p>
                      <h3>Final LLM context</h3>
                    </div>
                    {!finalContext ? (
                      <button
                        className="button secondary"
                        type="button"
                        onClick={() => void loadFinalContext()}
                        disabled={finalContextLoading}
                      >
                        {finalContextLoading ? "Loading…" : "View final context"}
                      </button>
                    ) : null}
                  </div>
                  <p className="context-disclosure">
                    Shows the exact text messages and authorized media descriptors sent in the final model call. Media bytes and Data URLs are never stored here.
                  </p>
                  {finalContextError ? (
                    <ProblemNotice error={finalContextError} onRetry={() => void loadFinalContext()} />
                  ) : null}
                  {finalContext && !finalContext.available ? (
                    <EmptyState
                      title="No final model request"
                      description="This run used a deterministic result or completed before final-context snapshots were introduced."
                    />
                  ) : null}
                  {finalContext?.available ? (
                    <div className="final-context-content">
                      <KeyValueGrid values={[
                        ["Final operation", finalContext.operation],
                        ["Output schema", finalContext.output_schema],
                        ["Max output tokens", finalContext.max_output_tokens],
                        ["Media attachments", finalContext.media.length],
                      ]} />
                      <div className="context-message-list">
                        {finalContext.messages.map((item, index) => (
                          <article className="context-message" key={`${item.role}-${index}`}>
                            <div className="message-meta">
                              <strong>{item.role}</strong>
                              <span>message {index + 1}</span>
                            </div>
                            <pre>{item.content}</pre>
                          </article>
                        ))}
                      </div>
                      {finalContext.media.length ? (
                        <div className="context-media-list">
                          <h4>Authorized media</h4>
                          {finalContext.media.map((media, index) => (
                            <article className="context-media" key={`${media.asset.id}-${index}`}>
                              <AssetPreview
                                client={client}
                                asset={media.asset}
                                alt={`Final model context media ${index + 1}`}
                              />
                              <KeyValueGrid values={[
                                ["Attached to message", media.message_index + 1],
                                ["Citation IDs", media.citation_ids.join(", ")],
                                ["Media type", media.asset.media_type],
                                ["Dimensions", media.asset.width && media.asset.height
                                  ? `${media.asset.width} × ${media.asset.height}`
                                  : null],
                              ]} />
                            </article>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                </section>
              ) : null}
              {run.citations.length ? (
                <div className="citation-list">
                  <div className="subheading-row">
                    <h3>Citation snapshots</h3>
                    <span>{run.citations.length} cited</span>
                  </div>
                  {run.citations.map((citation) => (
                    <article className="citation-card" key={citation.ordinal}>
                      <div className="citation-number">[{citation.ordinal + 1}]</div>
                      <div>
                        {citation.asset ? (
                          <AssetPreview
                            client={client}
                            asset={citation.asset}
                            alt={`${citation.modality} citation ${citation.ordinal + 1}`}
                          />
                        ) : null}
                        <blockquote>{citation.quoted_text || "Visual asset citation"}</blockquote>
                        <KeyValueGrid values={[
                          ["Modality", citation.modality],
                          ["Representations", citation.matched_representations.join(", ")],
                          ["Document", shortId(citation.document_id)],
                          ["Version", shortId(citation.document_version_id)],
                          ["Chunk", shortId(citation.index_chunk_id)],
                          ["Score", citation.score?.toFixed(4)],
                          ["Visual unit", shortId(citation.asset?.visual_unit_id)],
                          ["Parent citation", citation.asset?.parent_citation_id],
                          ["Relation", citation.asset?.relation_type],
                          ["Selection reason", citation.asset?.selection_reason],
                        ]} />
                        <JsonDetails label="Source location" value={citation.source_location} />
                        <button
                          className="button secondary"
                          type="button"
                          onClick={() => onOpenCitationDocument(
                            citation.document_id,
                            citation.document_version_id,
                          )}
                        >
                          View current document metadata
                        </button>
                      </div>
                    </article>
                  ))}
                </div>
              ) : null}
              <div className="run-detail-row">
                {run.timing ? <JsonDetails label="Timing" value={run.timing} /> : null}
                {run.usage ? <JsonDetails label="Usage" value={run.usage} /> : null}
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
}

function isTerminal(run: ChatRun): boolean {
  return run.status === "completed" || run.status === "failed" || run.status === "cancelled";
}

function emptyPreviewDiagnostics(runId: string | null): PreviewDiagnostics {
  return {
    runId,
    attempt: 0,
    lastSeq: 0,
    content: "",
    phase: "idle",
    deltaEvents: 0,
    deltaBytes: 0,
    resetEvents: 0,
    gapDetected: false,
    lastResetReason: null,
  };
}

function applyPreviewDeltaDiagnostics(
  current: PreviewDiagnostics,
  activeRunId: string,
  event: ChatPreviewDeltaEvent,
): PreviewDiagnostics {
  if (event.run_id !== activeRunId) return current;
  const base = current.runId === activeRunId
    ? current
    : emptyPreviewDiagnostics(activeRunId);
  const received = {
    ...base,
    deltaEvents: base.deltaEvents + 1,
    deltaBytes: base.deltaBytes + new TextEncoder().encode(event.delta).byteLength,
  };
  if (event.attempt < base.attempt) return received;
  if (event.attempt > base.attempt) {
    return event.seq === 1
      ? {
        ...received,
        attempt: event.attempt,
        lastSeq: 1,
        content: event.delta,
        phase: "streaming",
      }
      : discardPreviewDiagnostics(received, activeRunId, event.attempt);
  }
  if (base.phase === "discarded" || base.phase === "replaced") return received;
  if (base.attempt === 0) {
    return event.seq === 1
      ? {
        ...received,
        attempt: event.attempt,
        lastSeq: 1,
        content: event.delta,
        phase: "streaming",
      }
      : discardPreviewDiagnostics(received, activeRunId, event.attempt);
  }
  if (event.seq !== base.lastSeq + 1) {
    return discardPreviewDiagnostics(received, activeRunId, event.attempt);
  }
  return {
    ...received,
    lastSeq: event.seq,
    content: base.content + event.delta,
    phase: "streaming",
  };
}

function applyPreviewResetDiagnostics(
  current: PreviewDiagnostics,
  activeRunId: string,
  event: ChatPreviewResetEvent,
): PreviewDiagnostics {
  if (event.run_id !== activeRunId) return current;
  const base = current.runId === activeRunId
    ? current
    : emptyPreviewDiagnostics(activeRunId);
  const received = {
    ...base,
    resetEvents: base.resetEvents + 1,
    lastResetReason: event.reason,
  };
  if (event.attempt < base.attempt) return received;
  if (
    event.attempt === base.attempt
    && (base.phase === "discarded" || base.phase === "replaced")
  ) return received;
  const expected = event.attempt > base.attempt
    ? event.seq === 1
    : event.seq === base.lastSeq + 1;
  if (!expected) {
    return discardPreviewDiagnostics(received, activeRunId, event.attempt);
  }
  return {
    ...received,
    attempt: event.attempt,
    lastSeq: event.seq,
    content: "",
    phase: "verifying",
  };
}

function discardPreviewDiagnostics(
  current: PreviewDiagnostics,
  runId: string,
  attempt = current.attempt,
): PreviewDiagnostics {
  return {
    ...current,
    runId,
    attempt,
    content: "",
    phase: "discarded",
    gapDetected: true,
  };
}

function terminalPreviewDiagnostics(
  current: PreviewDiagnostics,
  runId: string,
): PreviewDiagnostics {
  const base = current.runId === runId
    ? current
    : emptyPreviewDiagnostics(runId);
  return {
    ...base,
    content: "",
    phase: "replaced",
  };
}

function mergeSessions(current: ChatSession[], incoming: ChatSession[]): ChatSession[] {
  const byId = new Map(current.map((item) => [item.id, item]));
  for (const item of incoming) byId.set(item.id, item);
  return [...byId.values()].sort((left, right) =>
    right.updated_at.localeCompare(left.updated_at)
  );
}

function mergeMessages(current: ChatMessage[], incoming: ChatMessage[]): ChatMessage[] {
  const byId = new Map(current.map((item) => [item.id, item]));
  for (const item of incoming) byId.set(item.id, item);
  return [...byId.values()].sort((left, right) =>
    left.created_at.localeCompare(right.created_at)
  );
}

function deliveryLabel(mode: DeliveryMode): string {
  const labels: Record<DeliveryMode, string> = {
    idle: "status",
    "sse-connecting": "SSE connecting",
    sse: "terminal SSE",
    polling: "status polling",
    terminal: "committed",
  };
  return labels[mode];
}
