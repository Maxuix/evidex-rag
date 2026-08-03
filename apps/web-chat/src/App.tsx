import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { ApiClient, ApiClientError, loadRuntimeConfig } from "./api/client";
import type {
  ChatMessage,
  ChatPreviewDeltaEvent,
  ChatPreviewResetEvent,
  ChatRun,
  ChatRunCreate,
  ChatSession,
  KnowledgeBase,
  RetrievalCapabilities,
} from "./api/types";
import {
  AnswerText,
  EvidenceDrawer,
  Notice,
  SourcesButton,
} from "./components";
import {
  formatTime,
  questionTitle,
  sessionGroup,
  sessionTitle,
} from "./format";
import {
  readKnowledgeBaseId,
  readSessionId,
  readSidebarCollapsed,
  storeKnowledgeBaseId,
  storeSessionId,
  storeSidebarCollapsed,
} from "./storage";

interface PendingRun {
  payload: ChatRunCreate;
  idempotencyKey: string;
}

interface EvidenceSelection {
  run: ChatRun | null;
  runId: string;
  ordinal: number;
}

interface ChatPreviewState {
  runId: string | null;
  attempt: number;
  lastSeq: number;
  content: string;
  mode: "idle" | "streaming" | "verifying" | "discarded";
}

export function App() {
  const [client, setClient] = useState<ApiClient | null>(null);
  const [startupError, setStartupError] = useState<string | null>(null);
  const [retrievalCapabilities, setRetrievalCapabilities] =
    useState<RetrievalCapabilities | null>(null);
  const [retrievalCapabilitiesLoading, setRetrievalCapabilitiesLoading] =
    useState(false);
  const [retrievalCapabilitiesError, setRetrievalCapabilitiesError] =
    useState<unknown | null>(null);

  useEffect(() => {
    void loadRuntimeConfig()
      .then((config) => setClient(new ApiClient(config)))
      .catch((error) => setStartupError(errorMessage(error)));
  }, []);

  useEffect(() => {
    if (!client) return;
    let cancelled = false;
    setRetrievalCapabilitiesLoading(true);
    setRetrievalCapabilitiesError(null);
    void client.getRetrievalCapabilities().then((value) => {
      if (!cancelled) setRetrievalCapabilities(value);
    }).catch((error) => {
      if (!cancelled) setRetrievalCapabilitiesError(error);
    }).finally(() => {
      if (!cancelled) setRetrievalCapabilitiesLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [client]);

  if (startupError) {
    return (
      <main className="startup-screen">
        <div className="brand-symbol">K</div>
        <h1>Knowledge Chat 无法启动</h1>
        <p>{startupError}</p>
        <button type="button" onClick={() => window.location.reload()}>重新加载</button>
      </main>
    );
  }
  if (!client) {
    return (
      <main className="startup-screen" aria-busy="true">
        <div className="brand-symbol">K</div>
        <div className="loading-line" />
      </main>
    );
  }
  return (
    <KnowledgeChat
      client={client}
      retrievalCapabilities={retrievalCapabilities}
      retrievalCapabilitiesLoading={retrievalCapabilitiesLoading}
      retrievalCapabilitiesError={retrievalCapabilitiesError}
    />
  );
}

function KnowledgeChat({
  client,
  retrievalCapabilities,
  retrievalCapabilitiesLoading,
  retrievalCapabilitiesError,
}: {
  client: ApiClient;
  retrievalCapabilities: RetrievalCapabilities | null;
  retrievalCapabilitiesLoading: boolean;
  retrievalCapabilitiesError: unknown | null;
}) {
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [knowledgeBaseCursor, setKnowledgeBaseCursor] = useState<string | null>(null);
  const [selectedKnowledgeBaseId, setSelectedKnowledgeBaseId] = useState(
    () => readKnowledgeBaseId() || "",
  );
  const [knowledgeBasesLoading, setKnowledgeBasesLoading] = useState(true);
  const [knowledgeBasesError, setKnowledgeBasesError] = useState<string | null>(null);

  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [sessionsCursor, setSessionsCursor] = useState<string | null>(null);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [sessionsError, setSessionsError] = useState<string | null>(null);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [messagesCursor, setMessagesCursor] = useState<string | null>(null);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [messagesError, setMessagesError] = useState<string | null>(null);

  const [draft, setDraft] = useState("");
  const [retrievalMode, setRetrievalMode] = useState<"vector" | "hybrid">("vector");
  const [submitting, setSubmitting] = useState(false);
  const [pendingRun, setPendingRun] = useState<PendingRun | null>(null);
  const [submissionError, setSubmissionError] = useState<string | null>(null);
  const [currentRun, setCurrentRun] = useState<ChatRun | null>(null);
  const [deliveryMode, setDeliveryMode] = useState<"idle" | "sse" | "polling">("idle");
  const [preview, setPreview] = useState<ChatPreviewState>(
    () => emptyPreview(null),
  );

  const [runCache, setRunCache] = useState<Record<string, ChatRun>>({});
  const [evidence, setEvidence] = useState<EvidenceSelection | null>(null);
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);
  const evidenceTrigger = useRef<HTMLButtonElement | null>(null);

  const [sidebarCollapsed, setSidebarCollapsed] = useState(readSidebarCollapsed);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messageGeneration = useRef(0);

  const selectedKnowledgeBase = useMemo(
    () => knowledgeBases.find((item) => item.id === selectedKnowledgeBaseId) ?? null,
    [knowledgeBases, selectedKnowledgeBaseId],
  );
  const selectedSession = useMemo(
    () => sessions.find((item) => item.id === selectedSessionId) ?? null,
    [sessions, selectedSessionId],
  );
  const sessionBusy = Boolean(
    currentRun
    && currentRun.session_id === selectedSessionId
    && !isTerminal(currentRun),
  ) || messages.some((item) => item.assistant_status === "generating");
  const hybridEnabled = retrievalCapabilities?.modes.some(
    (item) => item.mode === "hybrid" && item.enabled,
  ) ?? false;

  const loadKnowledgeBases = useCallback(async (cursor?: string) => {
    setKnowledgeBasesLoading(true);
    setKnowledgeBasesError(null);
    try {
      const page = await client.listKnowledgeBases(cursor);
      setKnowledgeBases((current) => cursor
        ? mergeById(current, page.items)
        : page.items);
      setKnowledgeBaseCursor(page.next_cursor);
      if (!cursor) {
        setSelectedKnowledgeBaseId((current) => {
          if (current && page.items.some((item) => item.id === current)) return current;
          return page.items[0]?.id ?? "";
        });
      }
    } catch (error) {
      setKnowledgeBasesError(errorMessage(error));
    } finally {
      setKnowledgeBasesLoading(false);
    }
  }, [client]);

  const loadSessions = useCallback(async (
    knowledgeBaseId: string,
    cursor?: string,
  ) => {
    setSessionsLoading(true);
    setSessionsError(null);
    try {
      const page = await client.listChatSessions(knowledgeBaseId, cursor);
      setSessions((current) => cursor
        ? mergeById(current, page.items)
        : page.items);
      setSessionsCursor(page.next_cursor);
      if (!cursor) {
        const stored = readSessionId(knowledgeBaseId);
        const next = stored && page.items.some((item) => item.id === stored)
          ? stored
          : page.items[0]?.id ?? null;
        setSelectedSessionId(next);
      }
    } catch (error) {
      setSessionsError(errorMessage(error));
    } finally {
      setSessionsLoading(false);
    }
  }, [client]);

  const loadMessages = useCallback(async (
    sessionId: string,
    cursor?: string,
  ) => {
    const generation = ++messageGeneration.current;
    setMessagesLoading(true);
    setMessagesError(null);
    try {
      const page = await client.listChatMessages(sessionId, cursor);
      if (generation !== messageGeneration.current) return;
      setMessages((current) => cursor
        ? mergeMessages(current, page.items)
        : page.items);
      setMessagesCursor(page.next_cursor);
    } catch (error) {
      if (generation === messageGeneration.current) {
        setMessagesError(errorMessage(error));
      }
    } finally {
      if (generation === messageGeneration.current) setMessagesLoading(false);
    }
  }, [client]);

  useEffect(() => {
    void loadKnowledgeBases();
  }, [loadKnowledgeBases]);

  useEffect(() => {
    storeKnowledgeBaseId(selectedKnowledgeBaseId || null);
    setRetrievalMode("vector");
    setSessions([]);
    setSelectedSessionId(null);
    setMessages([]);
    setCurrentRun(null);
    setRunCache({});
    closeEvidence();
    if (selectedKnowledgeBaseId) {
      void loadSessions(selectedKnowledgeBaseId);
    }
  }, [loadSessions, selectedKnowledgeBaseId]);

  useEffect(() => {
    if (!hybridEnabled && retrievalMode === "hybrid") setRetrievalMode("vector");
  }, [hybridEnabled, retrievalMode]);

  useEffect(() => {
    messageGeneration.current += 1;
    setMessages([]);
    setMessagesCursor(null);
    setCurrentRun(null);
    closeEvidence();
    if (selectedKnowledgeBaseId) {
      storeSessionId(selectedKnowledgeBaseId, selectedSessionId);
    }
    if (selectedSessionId) void loadMessages(selectedSessionId);
  }, [loadMessages, selectedKnowledgeBaseId, selectedSessionId]);

  useEffect(() => {
    if (!selectedSessionId || currentRun || messagesLoading) return;
    const generating = [...messages].reverse().find(
      (item) => item.role === "assistant"
        && item.assistant_status === "generating"
        && item.run_id,
    );
    if (!generating?.run_id) return;
    void client.getChatRun(generating.run_id)
      .then((run) => {
        if (run.session_id === selectedSessionId) setCurrentRun(run);
      })
      .catch((error) => setMessagesError(errorMessage(error)));
  }, [client, currentRun, messages, messagesLoading, selectedSessionId]);

  useEffect(() => {
    setPreview(emptyPreview(currentRun?.run_id ?? null));
  }, [currentRun?.run_id]);

  useEffect(() => {
    if (!currentRun) {
      setDeliveryMode("idle");
      return;
    }
    setRunCache((current) => ({ ...current, [currentRun.run_id]: currentRun }));
    if (isTerminal(currentRun)) {
      setPreview(emptyPreview(currentRun.run_id));
      setDeliveryMode("idle");
      if (currentRun.session_id === selectedSessionId) {
        void loadMessages(currentRun.session_id);
      }
      return;
    }
    let cancelled = false;
    let closeStream: (() => void) | null = null;
    let pollTimer: number | null = null;
    let polling = false;
    const poll = async () => {
      if (cancelled) return;
      try {
        const next = await client.getChatRun(currentRun.run_id);
        if (cancelled) return;
        setCurrentRun(next);
        if (!isTerminal(next)) {
          pollTimer = window.setTimeout(poll, 1200 + Math.random() * 350);
        }
      } catch {
        if (!cancelled) pollTimer = window.setTimeout(poll, 2200);
      }
    };
    const beginPolling = () => {
      if (cancelled || polling) return;
      polling = true;
      closeStream?.();
      closeStream = null;
      setPreview((current) => discardPreview(current, currentRun.run_id));
      setDeliveryMode("polling");
      void poll();
    };
    const settle = async (statusUrl: string) => {
      closeStream?.();
      closeStream = null;
      setPreview(emptyPreview(currentRun.run_id));
      try {
        const next = await client.getChatRun(statusUrl);
        if (!cancelled) setCurrentRun(next);
      } catch {
        beginPolling();
      }
    };
    closeStream = client.subscribeChatRun(currentRun.events_url, {
      open: () => !cancelled && setDeliveryMode("sse"),
      completed: (event) => void settle(event.status_url),
      failed: (event) => void settle(event.status_url),
      previewDelta: (event) => setPreview(
        (current) => applyPreviewDelta(current, currentRun.run_id, event),
      ),
      previewReset: (event) => setPreview(
        (current) => applyPreviewReset(current, currentRun.run_id, event),
      ),
      previewInvalid: () => setPreview(
        (current) => discardPreview(current, currentRun.run_id),
      ),
      error: beginPolling,
    });
    return () => {
      cancelled = true;
      closeStream?.();
      if (pollTimer !== null) window.clearTimeout(pollTimer);
    };
  }, [client, currentRun?.run_id, currentRun?.status, loadMessages, selectedSessionId]);

  useEffect(() => {
    const field = textareaRef.current;
    if (!field) return;
    field.style.height = "auto";
    field.style.height = `${Math.min(field.scrollHeight, 180)}px`;
  }, [draft]);

  const chooseKnowledgeBase = (value: string) => {
    if (value === selectedKnowledgeBaseId) return;
    if (draft.trim() && value !== selectedKnowledgeBaseId) {
      const discard = window.confirm("切换知识库会清除当前未发送的问题，是否继续？");
      if (!discard) return;
      setDraft("");
    }
    // A failed request is frozen to its original KB and idempotency key. Do
    // not leave that retryable payload attached to the newly selected KB.
    setPendingRun(null);
    setSubmissionError(null);
    setSelectedKnowledgeBaseId(value);
    setMobileSidebarOpen(false);
  };

  const beginNewConversation = () => {
    setSelectedSessionId(null);
    setMessages([]);
    setCurrentRun(null);
    setSubmissionError(null);
    closeEvidence();
    setMobileSidebarOpen(false);
    window.setTimeout(() => textareaRef.current?.focus(), 0);
  };

  const submit = async () => {
    if (
      !selectedKnowledgeBase
      || !draft.trim()
      || submitting
      || sessionBusy
    ) return;
    const question = draft.trim();
    setSubmitting(true);
    setSubmissionError(null);
    let sessionId = selectedSessionId;
    try {
      if (!sessionId) {
        const created = await client.createChatSession(
          selectedKnowledgeBase.id,
          questionTitle(question),
        );
        sessionId = created.id;
        setSessions((current) => mergeById([created], current));
        setSelectedSessionId(created.id);
        storeSessionId(selectedKnowledgeBase.id, created.id);
      }
      const pending: PendingRun = {
        idempotencyKey: crypto.randomUUID(),
        payload: {
          session_id: sessionId,
          knowledge_base_id: selectedKnowledgeBase.id,
          message: question,
          answer_policy: {
            answer_style: selectedKnowledgeBase.answer_policy_defaults.answer_style,
            insufficiency_policy: (
              selectedKnowledgeBase.answer_policy_defaults.insufficiency_policy
            ),
          },
          retrieval: {
            mode: retrievalMode,
            top_k: selectedKnowledgeBase.retrieval_defaults.top_k,
            rerank: true,
          },
        },
      };
      setPendingRun(pending);
      await performRun(pending);
    } catch (error) {
      setSubmissionError(errorMessage(error));
    } finally {
      setSubmitting(false);
    }
  };

  const performRun = async (pending: PendingRun) => {
    setSubmitting(true);
    setSubmissionError(null);
    try {
      const run = await client.createChatRun(
        pending.payload,
        pending.idempotencyKey,
      );
      setCurrentRun(run);
      setRunCache((current) => ({ ...current, [run.run_id]: run }));
      setDraft("");
      setPendingRun(null);
      await loadMessages(run.session_id);
    } catch (error) {
      setSubmissionError(errorMessage(error));
      throw error;
    } finally {
      setSubmitting(false);
    }
  };

  const changeRetrievalMode = (next: "vector" | "hybrid") => {
    if (next === "hybrid" && !hybridEnabled) return;
    if (pendingRun) {
      setPendingRun(null);
      setSubmissionError(null);
    }
    setRetrievalMode(next);
  };

  const openEvidence = async (
    runId: string,
    ordinal: number,
    trigger: HTMLButtonElement,
  ) => {
    evidenceTrigger.current = trigger;
    setEvidenceError(null);
    const cached = runCache[runId];
    if (cached) {
      setEvidence({ run: cached, runId, ordinal });
      return;
    }
    setEvidence({ run: null, runId, ordinal });
    setEvidenceLoading(true);
    try {
      const run = await client.getChatRun(runId);
      if (run.knowledge_base_id !== selectedKnowledgeBaseId) {
        throw new ApiClientError("来源不属于当前知识库。");
      }
      setRunCache((current) => ({ ...current, [runId]: run }));
      setEvidence({ run, runId, ordinal });
    } catch (error) {
      setEvidenceError(errorMessage(error));
    } finally {
      setEvidenceLoading(false);
    }
  };

  function closeEvidence() {
    setEvidence(null);
    setEvidenceError(null);
    setEvidenceLoading(false);
    const trigger = evidenceTrigger.current;
    evidenceTrigger.current = null;
    if (trigger) window.setTimeout(() => trigger.focus(), 0);
  }

  const groupedSessions = groupSessions(sessions);
  const title = selectedSession ? sessionTitle(selectedSession) : "新对话";

  return (
    <div className={`app${sidebarCollapsed ? " sidebar-collapsed" : ""}`}>
      {mobileSidebarOpen ? (
        <button
          className="mobile-backdrop"
          type="button"
          aria-label="关闭会话列表"
          onClick={() => setMobileSidebarOpen(false)}
        />
      ) : null}
      <aside className={`sidebar${mobileSidebarOpen ? " mobile-open" : ""}`}>
        <div className="sidebar-top">
          <div className="brand">
            <span className="brand-symbol">K</span>
            <span className="brand-name">Knowledge Chat</span>
          </div>
          <button
            className="icon-button collapse-button"
            type="button"
            aria-label={sidebarCollapsed ? "展开侧栏" : "收起侧栏"}
            onClick={() => {
              const next = !sidebarCollapsed;
              setSidebarCollapsed(next);
              storeSidebarCollapsed(next);
            }}
          >
            {sidebarCollapsed ? "›" : "‹"}
          </button>
        </div>

        <label className="knowledge-select">
          <span>知识库</span>
          <select
            value={selectedKnowledgeBaseId}
            disabled={knowledgeBasesLoading || !knowledgeBases.length}
            onChange={(event) => chooseKnowledgeBase(event.target.value)}
          >
            {!knowledgeBases.length ? <option value="">暂无知识库</option> : null}
            {knowledgeBases.map((item) => (
              <option key={item.id} value={item.id}>{item.name}</option>
            ))}
          </select>
        </label>
        {knowledgeBaseCursor ? (
          <button
            className="sidebar-text-action"
            type="button"
            onClick={() => void loadKnowledgeBases(knowledgeBaseCursor)}
          >
            加载更多知识库
          </button>
        ) : null}

        <button
          className="new-chat-button"
          type="button"
          disabled={!selectedKnowledgeBase}
          onClick={beginNewConversation}
        >
          <span aria-hidden="true">＋</span>
          <span>新对话</span>
        </button>

        <nav className="session-navigation" aria-label="会话历史">
          {sessionsLoading && !sessions.length ? (
            <div className="sidebar-loading">正在加载会话…</div>
          ) : null}
          {sessionsError ? (
            <Notice
              message={sessionsError}
              action="重试"
              onAction={() => selectedKnowledgeBaseId
                && void loadSessions(selectedKnowledgeBaseId)}
            />
          ) : null}
          {!sessionsLoading && !sessions.length ? (
            <p className="no-sessions">还没有历史对话</p>
          ) : null}
          {[...groupedSessions.entries()].map(([group, items]) => (
            <div className="session-group" key={group}>
              <p>{group}</p>
              {items.map((session) => (
                <button
                  className={`session-item${
                    session.id === selectedSessionId ? " active" : ""
                  }`}
                  type="button"
                  key={session.id}
                  title={sessionTitle(session)}
                  onClick={() => {
                    setSelectedSessionId(session.id);
                    setMobileSidebarOpen(false);
                  }}
                >
                  <span className="session-dot" aria-hidden="true" />
                  <span>{sessionTitle(session)}</span>
                </button>
              ))}
            </div>
          ))}
          {sessionsCursor ? (
            <button
              className="load-more"
              type="button"
              onClick={() => selectedKnowledgeBaseId
                && void loadSessions(selectedKnowledgeBaseId, sessionsCursor)}
            >
              加载更早会话
            </button>
          ) : null}
        </nav>
        <div className="local-boundary">
          <span className="status-dot" aria-hidden="true" />
          <span>本地试用</span>
        </div>
      </aside>

      <main className="chat-main">
        <header className="chat-header">
          <button
            className="icon-button mobile-menu"
            type="button"
            aria-label="打开会话列表"
            onClick={() => setMobileSidebarOpen(true)}
          >
            ☰
          </button>
          <div className="chat-heading">
            <h1>{title}</h1>
            <p>{selectedKnowledgeBase?.name || "请选择知识库"}</p>
          </div>
          {deliveryMode !== "idle" ? (
            <span className="delivery-status">
              <span className="status-dot" aria-hidden="true" />
              {deliveryMode === "sse" ? "正在回答" : "正在恢复连接"}
            </span>
          ) : <span />}
        </header>

        <div className="conversation">
          {knowledgeBasesError ? (
            <Notice
              message={knowledgeBasesError}
              action="重试"
              onAction={() => void loadKnowledgeBases()}
            />
          ) : null}
          {!knowledgeBasesLoading && !knowledgeBases.length ? (
            <EmptyKnowledgeBase />
          ) : messagesLoading && !messages.length && selectedSessionId ? (
            <div className="center-state">
              <span className="loading-ring" aria-hidden="true" />
              正在打开会话…
            </div>
          ) : !messages.length ? (
            <Welcome
              knowledgeBase={selectedKnowledgeBase}
              onSuggestion={(value) => {
                setDraft(value);
                window.setTimeout(() => textareaRef.current?.focus(), 0);
              }}
            />
          ) : (
            <div className="message-list">
              {messagesCursor ? (
                <button
                  className="older-messages"
                  type="button"
                  onClick={() => selectedSessionId
                    && void loadMessages(selectedSessionId, messagesCursor)}
                >
                  加载更早消息
                </button>
              ) : null}
              {messages.map((message) => (
                <Message
                  key={message.id}
                  message={message}
                  run={message.run_id ? runCache[message.run_id] ?? null : null}
                  preview={
                    message.run_id
                    && message.run_id === currentRun?.run_id
                    ? preview
                    : null
                  }
                  onCitation={(ordinal, trigger) => {
                    if (message.run_id) {
                      void openEvidence(message.run_id, ordinal, trigger);
                    }
                  }}
                />
              ))}
              {messagesError ? (
                <Notice
                  message={messagesError}
                  action="重试"
                  onAction={() => selectedSessionId
                    && void loadMessages(selectedSessionId)}
                />
              ) : null}
              {currentRun?.error && currentRun.session_id === selectedSessionId ? (
                <Notice message="这次回答没有完成，你可以稍后再次提问。" />
              ) : null}
            </div>
          )}
        </div>

        <div className="composer-zone">
          {submissionError ? (
            <div className="composer-error">
              <span>{submissionError}</span>
              {pendingRun ? (
                <button
                  type="button"
                  onClick={() => void performRun(pendingRun).catch(() => undefined)}
                >
                  重试发送
                </button>
              ) : null}
            </div>
          ) : null}
          <div className="composer">
            <label className="retrieval-mode-control">
              <span>检索模式</span>
              <select
                value={retrievalMode}
                onChange={(event) => changeRetrievalMode(
                  event.target.value as "vector" | "hybrid",
                )}
                disabled={submitting || sessionBusy}
                aria-describedby="web-chat-retrieval-mode-help"
              >
                <option value="vector">精确向量</option>
                <option value="hybrid" disabled={!hybridEnabled}>
                  混合（关键词 + 语义）
                </option>
              </select>
              <small id="web-chat-retrieval-mode-help">
                {retrievalCapabilitiesLoading
                  ? "能力状态加载中，已保持精确检索。"
                  : retrievalCapabilitiesError || !retrievalCapabilities
                    ? "能力状态不可用，混合模式已禁用。"
                    : hybridEnabled
                      ? "结合关键词与语义，可能更慢。"
                      : "当前 API 未启用混合模式。"}
              </small>
            </label>
            <textarea
              ref={textareaRef}
              value={draft}
              rows={1}
              maxLength={32768}
              disabled={!selectedKnowledgeBase || submitting || sessionBusy}
              placeholder={
                selectedKnowledgeBase
                  ? "询问这个知识库中的内容…"
                  : "请先选择知识库"
              }
              aria-label="输入问题"
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (
                  event.key === "Enter"
                  && !event.shiftKey
                  && !event.nativeEvent.isComposing
                ) {
                  event.preventDefault();
                  void submit();
                }
              }}
            />
            <button
              className="send-button"
              type="button"
              aria-label="发送问题"
              disabled={
                !selectedKnowledgeBase
                || !draft.trim()
                || submitting
                || sessionBusy
              }
              onClick={() => void submit()}
            >
              {submitting || sessionBusy ? (
                <span className="send-pulse" aria-hidden="true" />
              ) : (
                <span aria-hidden="true">↑</span>
              )}
            </button>
          </div>
          <p className="composer-disclaimer">
            回答仅基于当前知识库内容，请核对重要信息。
          </p>
        </div>
      </main>

      {evidence ? (
        <EvidenceDrawer
          client={client}
          run={evidence.run}
          selectedOrdinal={evidence.ordinal}
          loading={evidenceLoading}
          error={evidenceError}
          onSelect={(ordinal) => setEvidence((current) => current
            ? { ...current, ordinal }
            : null)}
          onClose={closeEvidence}
        />
      ) : null}
    </div>
  );
}

function Message({
  message,
  run,
  preview,
  onCitation,
}: {
  message: ChatMessage;
  run: ChatRun | null;
  preview: ChatPreviewState | null;
  onCitation: (ordinal: number, trigger: HTMLButtonElement) => void;
}) {
  if (message.role === "user") {
    return (
      <article className="message user-message">
        <div>{message.content}</div>
        <time>{formatTime(message.created_at)}</time>
      </article>
    );
  }
  const generating = message.assistant_status === "generating" && !message.content;
  const failed = message.assistant_status === "failed";
  return (
    <article className="message assistant-message">
      <div className="assistant-mark" aria-hidden="true">K</div>
      <div className="assistant-content">
        {generating && preview?.mode === "streaming" && preview.content ? (
          <div className="answer-preview" aria-live="polite">
            <div className="preview-label">
              <span>未验证预览</span>
              <span>最终回答可能调整</span>
            </div>
            <div className="preview-copy">
              {preview.content}
              <span className="preview-cursor" aria-hidden="true" />
            </div>
          </div>
        ) : generating && preview?.mode === "verifying" ? (
          <div className="thinking" aria-live="polite">
            <span /><span /><span />
            <strong>正在校验最终回答</strong>
          </div>
        ) : generating ? (
          <div className="thinking" aria-live="polite">
            <span /><span /><span />
            <strong>正在查找资料并整理回答</strong>
          </div>
        ) : failed ? (
          <p className="failed-answer">这次回答没有完成，请稍后再次提问。</p>
        ) : (
          <>
            <AnswerText
              content={message.content}
              citationCount={run ? run.citations.length : null}
              onCitation={message.run_id ? onCitation : undefined}
            />
            {message.run_id ? (
              <SourcesButton content={message.content} onOpen={onCitation} />
            ) : null}
          </>
        )}
        <time>{formatTime(message.created_at)}</time>
      </div>
    </article>
  );
}

function Welcome({
  knowledgeBase,
  onSuggestion,
}: {
  knowledgeBase: KnowledgeBase | null;
  onSuggestion: (value: string) => void;
}) {
  const suggestions = [
    "总结这个知识库的核心内容",
    "有哪些重要规定需要注意？",
    "帮我查找关键流程和操作要求",
  ];
  return (
    <section className="welcome">
      <div className="welcome-mark">K</div>
      <p>{knowledgeBase?.name || "Knowledge Chat"}</p>
      <h2>想从知识库中了解什么？</h2>
      <div className="suggestions">
        {suggestions.map((value) => (
          <button type="button" key={value} onClick={() => onSuggestion(value)}>
            {value}
            <span aria-hidden="true">↗</span>
          </button>
        ))}
      </div>
    </section>
  );
}

function EmptyKnowledgeBase() {
  return (
    <section className="welcome">
      <div className="welcome-mark">K</div>
      <h2>还没有可用的知识库</h2>
      <p>请先在本地诊断界面创建知识库并添加文档。</p>
    </section>
  );
}

function groupSessions(sessions: ChatSession[]): Map<string, ChatSession[]> {
  const groups = new Map<string, ChatSession[]>();
  for (const session of sessions) {
    const group = sessionGroup(session.updated_at);
    groups.set(group, [...(groups.get(group) ?? []), session]);
  }
  return groups;
}

function mergeById<T extends { id: string }>(left: T[], right: T[]): T[] {
  const values = new Map(left.map((item) => [item.id, item]));
  for (const item of right) values.set(item.id, item);
  return [...values.values()];
}

function mergeMessages(left: ChatMessage[], right: ChatMessage[]): ChatMessage[] {
  return mergeById(left, right).sort(
    (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
  );
}

function isTerminal(run: ChatRun): boolean {
  return ["completed", "failed", "cancelled"].includes(run.status);
}

function emptyPreview(runId: string | null): ChatPreviewState {
  return {
    runId,
    attempt: 0,
    lastSeq: 0,
    content: "",
    mode: "idle",
  };
}

function applyPreviewDelta(
  current: ChatPreviewState,
  activeRunId: string,
  event: ChatPreviewDeltaEvent,
): ChatPreviewState {
  if (event.run_id !== activeRunId) return current;
  const base = current.runId === activeRunId
    ? current
    : emptyPreview(activeRunId);
  if (event.attempt < base.attempt) return base;
  if (event.attempt > base.attempt) {
    return event.seq === 1
      ? {
        runId: activeRunId,
        attempt: event.attempt,
        lastSeq: 1,
        content: event.delta,
        mode: "streaming",
      }
      : discardPreview(base, activeRunId, event.attempt);
  }
  if (base.mode === "discarded") return base;
  if (base.attempt === 0) {
    return event.seq === 1
      ? {
        runId: activeRunId,
        attempt: event.attempt,
        lastSeq: 1,
        content: event.delta,
        mode: "streaming",
      }
      : discardPreview(base, activeRunId, event.attempt);
  }
  if (event.seq !== base.lastSeq + 1) {
    return discardPreview(base, activeRunId, event.attempt);
  }
  return {
    ...base,
    lastSeq: event.seq,
    content: base.content + event.delta,
    mode: "streaming",
  };
}

function applyPreviewReset(
  current: ChatPreviewState,
  activeRunId: string,
  event: ChatPreviewResetEvent,
): ChatPreviewState {
  if (event.run_id !== activeRunId) return current;
  const base = current.runId === activeRunId
    ? current
    : emptyPreview(activeRunId);
  if (event.attempt < base.attempt) return base;
  if (event.attempt === base.attempt && base.mode === "discarded") return base;
  const expected = event.attempt > base.attempt
    ? event.seq === 1
    : event.seq === base.lastSeq + 1;
  if (!expected) {
    return discardPreview(base, activeRunId, event.attempt);
  }
  return {
    runId: activeRunId,
    attempt: event.attempt,
    lastSeq: event.seq,
    content: "",
    mode: "verifying",
  };
}

function discardPreview(
  current: ChatPreviewState,
  runId: string,
  attempt = current.attempt,
): ChatPreviewState {
  return {
    runId,
    attempt,
    lastSeq: current.lastSeq,
    content: "",
    mode: "discarded",
  };
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "操作未能完成，请稍后重试。";
}
