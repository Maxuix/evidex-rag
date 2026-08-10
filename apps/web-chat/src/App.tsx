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
  ChatProgressSnapshot,
  ChatProgressStage,
  ChatPreviewDeltaEvent,
  ChatPreviewResetEvent,
  ChatRun,
  ChatRunCreate,
  ChatSession,
  ChatWorkflowCapabilities,
  ChatWorkflowMode,
  KnowledgeBase,
  ModelSettings,
  RerankMode,
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
import { ModelSettingsDialog } from "./ModelSettingsDialog";
import { KnowledgeBaseManagementPage } from "./KnowledgeBaseManagementPage";

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

interface ChatProgressState {
  runId: string | null;
  attempt: number;
  lastSeq: number;
  snapshot: ChatProgressSnapshot | null;
  stageRecords: Partial<Record<ChatProgressStage, ChatProgressSnapshot>>;
  mode: "idle" | "live" | "disconnected";
}

interface ComposerMenuOption<T extends string> {
  value: T;
  label: string;
  description: string;
  disabled?: boolean;
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
  const [workflowCapabilities, setWorkflowCapabilities] =
    useState<ChatWorkflowCapabilities | null>(null);
  const [workflowCapabilitiesLoading, setWorkflowCapabilitiesLoading] =
    useState(false);
  const [workflowCapabilitiesError, setWorkflowCapabilitiesError] =
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

  useEffect(() => {
    if (!client) return;
    let cancelled = false;
    setWorkflowCapabilitiesLoading(true);
    setWorkflowCapabilitiesError(null);
    void client.getChatWorkflowCapabilities().then((value) => {
      if (!cancelled) setWorkflowCapabilities(value);
    }).catch((error) => {
      if (!cancelled) setWorkflowCapabilitiesError(error);
    }).finally(() => {
      if (!cancelled) setWorkflowCapabilitiesLoading(false);
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
      workflowCapabilities={workflowCapabilities}
      workflowCapabilitiesLoading={workflowCapabilitiesLoading}
      workflowCapabilitiesError={workflowCapabilitiesError}
    />
  );
}

function KnowledgeChat({
  client,
  retrievalCapabilities,
  retrievalCapabilitiesLoading,
  retrievalCapabilitiesError,
  workflowCapabilities,
  workflowCapabilitiesLoading,
  workflowCapabilitiesError,
}: {
  client: ApiClient;
  retrievalCapabilities: RetrievalCapabilities | null;
  retrievalCapabilitiesLoading: boolean;
  retrievalCapabilitiesError: unknown | null;
  workflowCapabilities: ChatWorkflowCapabilities | null;
  workflowCapabilitiesLoading: boolean;
  workflowCapabilitiesError: unknown | null;
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
  const [rerankMode, setRerankMode] = useState<RerankMode>("classic");
  const [workflowMode, setWorkflowMode] = useState<ChatWorkflowMode>("simple");
  const [modelSettings, setModelSettings] = useState<ModelSettings | null>(null);
  const [modelSettingsLoading, setModelSettingsLoading] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [selectedChatModelRevisionId, setSelectedChatModelRevisionId] =
    useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [pendingRun, setPendingRun] = useState<PendingRun | null>(null);
  const [submissionError, setSubmissionError] = useState<string | null>(null);
  const [currentRun, setCurrentRun] = useState<ChatRun | null>(null);
  const [deliveryMode, setDeliveryMode] = useState<"idle" | "sse" | "polling">("idle");
  const [preview, setPreview] = useState<ChatPreviewState>(
    () => emptyPreview(null),
  );
  const [progress, setProgress] = useState<ChatProgressState>(
    () => emptyProgress(null),
  );

  const [runCache, setRunCache] = useState<Record<string, ChatRun>>({});
  const [evidence, setEvidence] = useState<EvidenceSelection | null>(null);
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);
  const evidenceTrigger = useRef<HTMLButtonElement | null>(null);

  const [sidebarCollapsed, setSidebarCollapsed] = useState(readSidebarCollapsed);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [activePage, setActivePage] = useState<"chat" | "knowledge-base">("chat");
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
  const agentEnabled = workflowCapabilities?.modes.some(
    (item) => item.mode === "agent" && item.enabled,
  ) ?? false;
  const autoEnabled = workflowCapabilities?.modes.some(
    (item) => item.mode === "auto" && item.enabled,
  ) ?? false;
  const chatModels = modelSettings?.profiles.filter((profile) => (
    profile.kind === "chat"
    && profile.enabled
    && profile.validation_status === "valid"
  )) ?? [];
  const chatModelConfigured = Boolean(
    selectedChatModelRevisionId
    && chatModels.some(
      (profile) => profile.revision_id === selectedChatModelRevisionId,
    ),
  );

  const acceptModelSettings = useCallback((value: ModelSettings) => {
    setModelSettings(value);
    setSelectedChatModelRevisionId((current) => (
      current && value.profiles.some((profile) => profile.revision_id === current)
        ? current
        : value.selection.chat_profile_revision_id
    ));
  }, []);

  useEffect(() => {
    let cancelled = false;
    setModelSettingsLoading(true);
    void client.getModelSettings().then((value) => {
      if (!cancelled) acceptModelSettings(value);
    }).catch(() => {
      // The composer stays fail-closed; the settings dialog can retry visibly.
    }).finally(() => {
      if (!cancelled) setModelSettingsLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [acceptModelSettings, client]);

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
    setWorkflowMode("simple");
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
    if (selectedKnowledgeBase) {
      setRerankMode(selectedKnowledgeBase.retrieval_defaults.rerank_mode);
    }
  }, [
    selectedKnowledgeBase?.id,
    selectedKnowledgeBase?.retrieval_defaults.rerank_mode,
  ]);

  useEffect(() => {
    if (
      (workflowMode === "agent" && !agentEnabled)
      || (workflowMode === "auto" && !autoEnabled)
    ) {
      setWorkflowMode("simple");
    }
    if (workflowMode !== "simple" && rerankMode === "local_minilm_v1") {
      setRerankMode("classic");
    }
  }, [agentEnabled, autoEnabled, rerankMode, workflowMode]);

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
    if (!selectedSessionId || messagesLoading) return;
    const missingRunIds = [...new Set(messages.flatMap((item) => (
      item.role === "assistant" && item.run_id && !runCache[item.run_id]
        ? [item.run_id]
        : []
    )))];
    if (!missingRunIds.length) return;
    let cancelled = false;
    void Promise.allSettled(
      missingRunIds.map((runId) => client.getChatRun(runId)),
    ).then((results) => {
      if (cancelled) return;
      const loaded: Record<string, ChatRun> = {};
      for (const result of results) {
        if (
          result.status === "fulfilled"
          && result.value.session_id === selectedSessionId
          && result.value.knowledge_base_id === selectedKnowledgeBaseId
        ) {
          loaded[result.value.run_id] = result.value;
        }
      }
      if (Object.keys(loaded).length) {
        setRunCache((current) => ({ ...current, ...loaded }));
      }
    });
    return () => {
      cancelled = true;
    };
  }, [
    client,
    messages,
    messagesLoading,
    runCache,
    selectedKnowledgeBaseId,
    selectedSessionId,
  ]);

  useEffect(() => {
    setPreview(emptyPreview(currentRun?.run_id ?? null));
    setProgress(emptyProgress(currentRun?.run_id ?? null));
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
      setProgress((current) => disconnectProgress(current, currentRun.run_id));
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
      progress: (event) => setProgress(
        (current) => applyProgress(current, currentRun.run_id, event),
      ),
      previewInvalid: () => setPreview(
        (current) => discardPreview(current, currentRun.run_id),
      ),
      progressInvalid: () => setProgress(
        (current) => disconnectProgress(current, currentRun.run_id),
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
    setActivePage("chat");
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
      || !chatModelConfigured
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
          workflow: {
            mode: workflowMode,
          },
          retrieval: {
            mode: retrievalMode,
            top_k: selectedKnowledgeBase.retrieval_defaults.top_k,
            rerank_mode: rerankMode,
          },
          model_profile_revision_id: selectedChatModelRevisionId,
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
    if (next === "hybrid" && rerankMode === "none") setRerankMode("classic");
    setRetrievalMode(next);
  };

  const changeWorkflowMode = (next: ChatWorkflowMode) => {
    if (next === "agent" && !agentEnabled) return;
    if (next === "auto" && !autoEnabled) return;
    if (pendingRun) {
      setPendingRun(null);
      setSubmissionError(null);
    }
    if (next !== "simple" && rerankMode === "local_minilm_v1") {
      setRerankMode("classic");
    }
    setWorkflowMode(next);
  };

  const changeRerankMode = (next: RerankMode) => {
    if (next === "none" && retrievalMode === "hybrid") return;
    if (
      next === "local_minilm_v1"
      && (
        workflowMode !== "simple"
        || (selectedKnowledgeBase?.retrieval_defaults.top_k ?? 100) > 20
      )
    ) return;
    if (pendingRun) {
      setPendingRun(null);
      setSubmissionError(null);
    }
    setRerankMode(next);
  };

  const changeChatModel = (next: string) => {
    if (pendingRun) {
      setPendingRun(null);
      setSubmissionError(null);
    }
    setSelectedChatModelRevisionId(next || null);
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

        <nav className="sidebar-primary-nav" aria-label="主要页面">
          <button
            className={activePage === "chat" ? "active" : ""}
            type="button"
            title="对话"
            onClick={() => {
              setActivePage("chat");
              setMobileSidebarOpen(false);
            }}
          >
            <span aria-hidden="true">◌</span>
            <span>对话</span>
          </button>
          <button
            className={activePage === "knowledge-base" ? "active" : ""}
            type="button"
            title="知识库管理"
            onClick={() => {
              setActivePage("knowledge-base");
              setMobileSidebarOpen(false);
            }}
          >
            <span aria-hidden="true">▤</span>
            <span>知识库管理</span>
          </button>
        </nav>

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

        {activePage === "chat" ? (
          <button
            className="new-chat-button"
            type="button"
            disabled={!selectedKnowledgeBase}
            onClick={beginNewConversation}
          >
            <span aria-hidden="true">＋</span>
            <span>新对话</span>
          </button>
        ) : null}

        {activePage === "chat" ? <nav className="session-navigation" aria-label="会话历史">
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
        </nav> : (
          <div className="sidebar-management-note">
            <strong>知识库工作区</strong>
            <p>创建、导入、查看解析结果并测试检索。</p>
          </div>
        )}
        <div className="local-boundary">
          <span className="status-dot" aria-hidden="true" />
          <span>本地试用</span>
        </div>
      </aside>

      {activePage === "knowledge-base" ? (
        <KnowledgeBaseManagementPage
          client={client}
          knowledgeBases={knowledgeBases}
          selectedKnowledgeBaseId={selectedKnowledgeBaseId}
          modelSettings={modelSettings}
          hybridEnabled={hybridEnabled}
          onKnowledgeBaseCreated={(created) => {
            setKnowledgeBases((current) => (
              [...current.filter((item) => item.id !== created.id), created]
                .sort((left, right) => left.name.localeCompare(right.name, "zh-CN"))
            ));
            setSelectedKnowledgeBaseId(created.id);
          }}
          onKnowledgeBaseDeleted={(id) => {
            const remaining = knowledgeBases.filter((item) => item.id !== id);
            setKnowledgeBases(remaining);
            setSelectedKnowledgeBaseId(remaining[0]?.id ?? "");
          }}
          onOpenModelSettings={() => setSettingsOpen(true)}
          onOpenMobileSidebar={() => setMobileSidebarOpen(true)}
        />
      ) : <main className="chat-main">
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
            <EmptyKnowledgeBase onManage={() => setActivePage("knowledge-base")} />
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
                  progress={
                    message.run_id
                    && message.run_id === currentRun?.run_id
                    ? progress
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
            <textarea
              ref={textareaRef}
              value={draft}
              rows={1}
              maxLength={32768}
              disabled={!selectedKnowledgeBase || !chatModelConfigured || submitting || sessionBusy}
              placeholder={
                !chatModelConfigured
                  ? "请先在右下角齿轮中选择对话模型"
                  : selectedKnowledgeBase
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
            <div className="composer-toolbar">
              <div className="composer-options">
                <label
                  className="composer-model-control"
                  title={modelSettingsLoading
                    ? "正在读取模型设置…"
                    : chatModels.length
                      ? "选择本次对话使用的模型修订版"
                      : "请在齿轮设置中添加并验证模型"}
                >
                  <ComposerModelIcon />
                  <select
                    aria-label="对话模型"
                    value={selectedChatModelRevisionId ?? ""}
                    onChange={(event) => changeChatModel(event.target.value)}
                    disabled={submitting || sessionBusy}
                  >
                    <option value="">未选择模型</option>
                    {chatModels.map((profile) => (
                      <option key={profile.revision_id} value={profile.revision_id}>
                        {profile.name} · r{profile.revision}
                      </option>
                    ))}
                  </select>
                  <span className="composer-model-chevron" aria-hidden="true">⌄</span>
                </label>
                <ComposerOptionMenu
                  kind="workflow"
                  label="回答工作流"
                  value={workflowMode}
                  disabled={submitting || sessionBusy}
                  options={[
                    {
                      value: "simple",
                      label: "Simple",
                      description: "单次检索，速度最快。",
                    },
                    {
                      value: "agent",
                      label: "Agent",
                      description: agentEnabled
                        ? "多视角、多跳检索，通常更慢。"
                        : workflowCapabilitiesLoading
                          ? "能力状态加载中。"
                          : "当前服务未启用 Agent。",
                      disabled: !agentEnabled,
                    },
                    {
                      value: "auto",
                      label: "Auto",
                      description: autoEnabled
                        ? "自动判断问题复杂度并选择工作流。"
                        : workflowCapabilitiesError || !workflowCapabilities
                          ? "能力状态不可用。"
                          : "当前服务未启用 Auto。",
                      disabled: !autoEnabled,
                    },
                  ]}
                  onChange={changeWorkflowMode}
                />
                <ComposerOptionMenu
                  kind="retrieval"
                  label="检索模式"
                  value={retrievalMode}
                  disabled={submitting || sessionBusy}
                  options={[
                    {
                      value: "vector",
                      label: "精确向量",
                      description: "按语义相似度检索。",
                    },
                    {
                      value: "hybrid",
                      label: "混合检索",
                      description: hybridEnabled
                        ? "结合关键词与语义，可能更慢。"
                        : retrievalCapabilitiesLoading
                          ? "能力状态加载中。"
                          : retrievalCapabilitiesError || !retrievalCapabilities
                            ? "能力状态不可用。"
                            : "当前服务未启用混合检索。",
                      disabled: !hybridEnabled,
                    },
                  ]}
                  onChange={changeRetrievalMode}
                />
                <ComposerOptionMenu
                  kind="rerank"
                  label="精排方式"
                  value={rerankMode}
                  disabled={submitting || sessionBusy}
                  options={[
                    {
                      value: "none",
                      label: "不精排",
                      description: retrievalMode === "hybrid"
                        ? "混合检索必须保留精排。"
                        : "直接使用向量检索顺序，资源开销最低。",
                      disabled: retrievalMode === "hybrid",
                    },
                    {
                      value: "classic",
                      label: "经典精排",
                      description: "使用现有关键词、向量与去重规则。",
                    },
                    {
                      value: "local_minilm_v1",
                      label: "本地 MiniLM",
                      description: workflowMode !== "simple"
                        ? "首版仅支持 Simple 工作流。"
                        : (selectedKnowledgeBase?.retrieval_defaults.top_k ?? 100) > 20
                          ? "本地模型要求知识库 Top K 不超过 20。"
                          : "本机离线 CrossEncoder 精排，相关性更强但更慢。",
                      disabled: workflowMode !== "simple"
                        || (selectedKnowledgeBase?.retrieval_defaults.top_k ?? 100) > 20,
                    },
                  ]}
                  onChange={changeRerankMode}
                />
              </div>
              <button
                className="send-button"
                type="button"
                aria-label="发送问题"
                disabled={
                  !selectedKnowledgeBase
                  || !draft.trim()
                  || submitting
                  || sessionBusy
                  || !chatModelConfigured
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
          </div>
          <p className="composer-disclaimer">
            回答仅基于当前知识库内容，请核对重要信息。
          </p>
        </div>
      </main>}

      {activePage === "chat" ? <button
        className="settings-gear"
        type="button"
        aria-label="打开模型设置"
        title="模型设置"
        onClick={() => setSettingsOpen(true)}
      >
        ⚙
      </button> : null}

      {settingsOpen ? (
        <ModelSettingsDialog
          client={client}
          initial={modelSettings}
          onChange={acceptModelSettings}
          onClose={() => setSettingsOpen(false)}
        />
      ) : null}

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

function ComposerOptionMenu<T extends string>({
  kind,
  label,
  value,
  options,
  disabled,
  onChange,
}: {
  kind: "workflow" | "retrieval" | "rerank";
  label: string;
  value: T;
  options: readonly ComposerMenuOption<T>[];
  disabled: boolean;
  onChange: (value: T) => void;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const selected = options.find((option) => option.value === value) ?? options[0];

  useEffect(() => {
    if (!open) return;
    const closeOnPointerDown = (event: PointerEvent) => {
      if (
        rootRef.current
        && event.target instanceof Node
        && !rootRef.current.contains(event.target)
      ) {
        setOpen(false);
      }
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", closeOnPointerDown);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnPointerDown);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  return (
    <div className="composer-option-menu" ref={rootRef}>
      <button
        className={`composer-icon-button${open ? " active" : ""}`}
        type="button"
        aria-label={`${label}：${selected.label}`}
        aria-haspopup="menu"
        aria-expanded={open}
        title={`${label}：${selected.label}`}
        disabled={disabled}
        onClick={() => setOpen((current) => !current)}
      >
        <ComposerMenuIcon kind={kind} />
      </button>
      {open ? (
        <div className="composer-popover" role="menu" aria-label={label}>
          <div className="composer-popover-title">{label}</div>
          {options.map((option) => (
            <button
              key={option.value}
              className="composer-popover-option"
              type="button"
              role="menuitemradio"
              aria-checked={option.value === value}
              disabled={option.disabled}
              onClick={() => {
                onChange(option.value);
                setOpen(false);
              }}
            >
              <span className="composer-option-check" aria-hidden="true">
                {option.value === value ? "✓" : ""}
              </span>
              <span>
                <strong>{option.label}</strong>
                <small>{option.description}</small>
              </span>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function ComposerModelIcon() {
  return (
    <svg className="composer-model-icon" viewBox="0 0 20 20" aria-hidden="true">
      <path d="M10 2.75 16 6.2v7.6l-6 3.45-6-3.45V6.2L10 2.75Z" />
      <circle cx="10" cy="10" r="2.1" />
    </svg>
  );
}

function ComposerMenuIcon({ kind }: {
  kind: "workflow" | "retrieval" | "rerank";
}) {
  if (kind === "retrieval") {
    return (
      <svg viewBox="0 0 20 20" aria-hidden="true">
        <circle cx="8.5" cy="8.5" r="4.75" />
        <path d="m12 12 4.25 4.25" />
      </svg>
    );
  }
  if (kind === "rerank") {
    return (
      <svg viewBox="0 0 20 20" aria-hidden="true">
        <path d="M4 5h12M4 10h12M4 15h12" />
        <circle cx="8" cy="5" r="1.5" />
        <circle cx="13" cy="10" r="1.5" />
        <circle cx="7" cy="15" r="1.5" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true">
      <circle cx="5" cy="5" r="1.75" />
      <circle cx="15" cy="5" r="1.75" />
      <circle cx="10" cy="15" r="1.75" />
      <path d="M6.7 5h2.05A1.25 1.25 0 0 1 10 6.25v6.9M13.3 5h-2.05A1.25 1.25 0 0 0 10 6.25" />
    </svg>
  );
}

function Message({
  message,
  run,
  preview,
  progress,
  onCitation,
}: {
  message: ChatMessage;
  run: ChatRun | null;
  preview: ChatPreviewState | null;
  progress: ChatProgressState | null;
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
        {run ? <WorkflowSummary workflow={run.workflow} model={run.model} /> : null}
        {run ? (
          <ExecutionTrace
            run={run}
            progress={progress}
            generating={generating}
          />
        ) : null}
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

function WorkflowSummary({ workflow, model }: {
  workflow: ChatRun["workflow"];
  model: ChatRun["model"];
}) {
  const requested = workflowModeLabel(workflow.requested_mode);
  const resolved = workflow.resolved_mode === "pending"
    ? "路由中"
    : workflowModeLabel(workflow.resolved_mode);
  const modeLabel = workflow.requested_mode === "auto"
    ? `Auto → ${resolved}`
    : requested;
  const result = workflow.research_result;
  const trace = workflow.search_trace;
  const fallback = workflow.route_status === "fallback"
    ? workflow.route_reason_codes.includes("router_invalid")
      ? "路由结果无效，已回退"
      : "路由服务不可用，已回退"
    : null;
  return (
    <div className="workflow-summary" aria-label={`回答工作流：${modeLabel}`}>
      <span className="workflow-badge">{modeLabel}</span>
      <span className="workflow-badge">{model.profile_name || model.model}</span>
      {fallback ? <span>{fallback}</span> : null}
      {result && trace ? (
        <span>
          {researchStatusLabel(result.status)} · {terminationLabel(
            result.termination_reason,
          )} · 检索 {trace.retrieval_calls} 次 · 证据 {trace.evidence_count} 项
          {result.missing_aspects.length
            ? ` · 待补 ${result.missing_aspects.length} 项`
            : ""}
          {result.conflicts.length ? ` · 冲突 ${result.conflicts.length} 项` : ""}
        </span>
      ) : null}
    </div>
  );
}

const PROGRESS_STAGES: ChatProgressStage[] = [
  "understand_query",
  "select_workflow",
  "retrieve_evidence",
  "assess_evidence",
  "prepare_visual_evidence",
  "generate_answer",
  "validate_answer",
  "persist_result",
];

function ExecutionTrace({
  run,
  progress,
  generating,
}: {
  run: ChatRun;
  progress: ChatProgressState | null;
  generating: boolean;
}) {
  const [selectedStage, setSelectedStage] = useState<ChatProgressStage | null>(null);
  useEffect(() => setSelectedStage(null), [run.run_id]);

  const terminal = isTerminal(run);
  const overview = run.status === "completed"
    ? completedProgress(run)
    : progress?.snapshot ?? null;
  const disconnected = generating && progress?.mode === "disconnected";
  const activeStage = overview?.active_stage ?? null;
  const availableStages = new Set<ChatProgressStage>([
    ...(overview?.completed_stages ?? []),
    ...Object.keys(progress?.stageRecords ?? {}) as ChatProgressStage[],
  ]);
  if (activeStage) availableStages.add(activeStage);
  const archivedStage = selectedStage
    && selectedStage !== activeStage
    && availableStages.has(selectedStage)
    ? selectedStage
    : null;
  const viewedStage = archivedStage ?? activeStage;
  const viewedSnapshot = viewedStage
    ? stageProgress(run, progress, overview, viewedStage)
    : null;
  const followingCurrent = archivedStage === null;
  const showViewedSnapshot = Boolean(
    viewedSnapshot
    && (!disconnected || !followingCurrent || terminal),
  );
  const content = (
    <div className="execution-trace-body">
      {disconnected ? (
        <div className="trace-connection-note" role="status">
          实时轨迹连接已中断，回答仍在后台运行；当前节点暂不确定。
        </div>
      ) : null}
      {run.status === "failed" || run.status === "cancelled" ? (
        <div className="trace-connection-note terminal">
          执行在完成前中止；这里只保留已确认的工作流状态。
        </div>
      ) : null}
      <div className="trace-stage-rail" aria-label="回答执行阶段">
        {PROGRESS_STAGES.map((stage, index) => {
          const complete = overview?.completed_stages.includes(stage) ?? false;
          const active = Boolean(
            overview
            && !disconnected
            && overview.status === "active"
            && overview.active_stage === stage,
          );
          const available = availableStages.has(stage);
          const selected = viewedStage === stage;
          return (
            <div className="trace-stage-wrap" key={stage}>
              <button
                type="button"
                className={`trace-stage${complete ? " complete" : ""}${
                active ? " active" : ""
              }${selected ? " selected" : ""}`}
                disabled={!available}
                aria-pressed={selected}
                aria-label={complete
                  ? `查看${progressStageLabel(stage)}阶段记录`
                  : active
                    ? `${progressStageLabel(stage)}，当前阶段`
                    : progressStageLabel(stage)}
                onClick={() => setSelectedStage(stage === activeStage ? null : stage)}
              >
                <span aria-hidden="true">{complete ? "✓" : index + 1}</span>
                <strong>{progressStageLabel(stage)}</strong>
              </button>
              {index < PROGRESS_STAGES.length - 1 ? (
                <span className={`trace-arrow${complete ? " complete" : ""}`}>
                  →
                </span>
              ) : null}
            </div>
          );
        })}
      </div>
      {showViewedSnapshot && viewedSnapshot && viewedStage ? (
        <div
          className="trace-current"
          key={`${viewedStage}-${viewedSnapshot.activity}-${viewedSnapshot.seq}`}
          aria-live={followingCurrent && !terminal ? "polite" : "off"}
        >
          <div className="trace-current-heading">
            <div>
              <span>{followingCurrent
                ? viewedSnapshot.status === "completed" ? "执行完成" : "当前阶段"
                : "阶段记录"}</span>
              <strong>{activityLabel(viewedSnapshot.activity)}</strong>
            </div>
            {!followingCurrent && !terminal ? (
              <button type="button" onClick={() => setSelectedStage(null)}>
                返回当前阶段
              </button>
            ) : null}
          </div>
          <p>{activityDescription(viewedSnapshot.activity)}</p>
          {viewedSnapshot.requested_mode === "auto"
            && viewedSnapshot.resolved_mode !== "pending" ? (
              <div className="trace-route-choice">
                Auto 已选择 <strong>{workflowModeLabel(
                  viewedSnapshot.resolved_mode,
                )}</strong>
              </div>
            ) : null}
          <ProgressFacts facts={viewedSnapshot.facts} />
        </div>
      ) : !terminal && !disconnected ? (
        <div className="trace-current waiting" aria-live="polite">
          正在等待第一个执行节点…
        </div>
      ) : null}
      {viewedStage === "retrieve_evidence"
        && run.workflow.search_trace?.steps.length ? (
        <div className="trace-search-history">
          <h4>检索决策记录</h4>
          {run.workflow.search_trace.steps.map((step) => (
            <div className="trace-search-step" key={step.observation_id}>
              <div>
                <strong>{step.objective}</strong>
                <span>{searchResultLabel(step.result)} · 新增证据 {
                  step.new_evidence_count
                } 项</span>
              </div>
              {step.queries.length ? (
                <ul>
                  {step.queries.map((query) => <li key={query}>{query}</li>)}
                </ul>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
  if (terminal) {
    return (
      <details className="execution-trace terminal-trace">
        <summary>
          <span>执行轨迹</span>
          <strong>{run.status === "completed" ? "已完成" : "未完成"}</strong>
        </summary>
        {content}
      </details>
    );
  }
  return (
    <section className="execution-trace" aria-label="实时执行轨迹">
      <header>
        <span>执行轨迹</span>
        <strong>决策过程 · 实时</strong>
      </header>
      {content}
    </section>
  );
}

function stageProgress(
  run: ChatRun,
  progress: ChatProgressState | null,
  overview: ChatProgressSnapshot | null,
  stage: ChatProgressStage,
): ChatProgressSnapshot {
  const recorded = progress?.stageRecords[stage];
  const complete = overview?.completed_stages.includes(stage) ?? false;
  if (recorded) {
    return {
      ...recorded,
      completed_stages: overview?.completed_stages ?? recorded.completed_stages,
      status: complete ? "completed" : recorded.status,
    };
  }
  return {
    run_id: run.run_id,
    attempt: run.attempt,
    seq: overview?.seq ?? 0,
    active_stage: stage,
    activity: stageActivity(run, stage),
    completed_stages: overview?.completed_stages ?? [],
    status: complete ? "completed" : "active",
    requested_mode: run.workflow.requested_mode,
    resolved_mode: run.workflow.resolved_mode,
    facts: stageFacts(run, stage),
  };
}

function stageActivity(
  run: ChatRun,
  stage: ChatProgressStage,
): ChatProgressSnapshot["activity"] {
  if (stage === "retrieve_evidence") {
    if (run.workflow.research_result) return "research_complete";
    if (run.workflow.search_trace?.steps.length) return "retrieval_complete";
    return "simple_search";
  }
  const activities: Record<Exclude<ChatProgressStage, "retrieve_evidence">,
    ChatProgressSnapshot["activity"]> = {
      understand_query: "contextualize_query",
      select_workflow: "route_decision",
      assess_evidence: "assess_evidence",
      prepare_visual_evidence: "prepare_visual_evidence",
      generate_answer: "generate_answer",
      validate_answer: "validate_answer",
      persist_result: "persist_result",
    };
  return activities[stage];
}

function stageFacts(
  run: ChatRun,
  stage: ChatProgressStage,
): ChatProgressSnapshot["facts"] {
  const result = run.workflow.research_result;
  const trace = run.workflow.search_trace;
  const lastSearch = trace?.steps.at(-1);
  const facts: ChatProgressSnapshot["facts"] = {
    objective: null,
    queries: [],
    evidence_count: null,
    new_evidence_count: null,
    retrieval_calls: null,
    route_status: run.workflow.route_status,
    route_reason_codes: run.workflow.route_reason_codes,
    research_status: null,
    covered_aspects: [],
    missing_aspects: [],
    conflict_count: null,
    decision: null,
  };
  if (stage === "select_workflow" && run.workflow.resolved_mode !== "pending") {
    facts.decision = run.workflow.resolved_mode === "agent"
      ? "select_agent"
      : "select_simple";
  }
  if (stage === "retrieve_evidence") {
    facts.objective = lastSearch?.objective ?? null;
    facts.queries = lastSearch?.queries ?? [];
    facts.evidence_count = trace?.evidence_count ?? null;
    facts.new_evidence_count = lastSearch?.new_evidence_count ?? null;
    facts.retrieval_calls = trace?.retrieval_calls ?? null;
    facts.research_status = result?.status ?? null;
    facts.decision = result ? "finish_research" : null;
  }
  if (stage === "assess_evidence") {
    facts.evidence_count = trace?.evidence_count ?? null;
    facts.retrieval_calls = trace?.retrieval_calls ?? null;
    facts.research_status = result?.status ?? null;
    facts.covered_aspects = result?.covered_aspects.slice(0, 6) ?? [];
    facts.missing_aspects = result?.missing_aspects.slice(0, 6) ?? [];
    facts.conflict_count = result?.conflicts.length ?? null;
    facts.decision = result ? "finish_research" : null;
  }
  return facts;
}

function ProgressFacts({ facts }: { facts: ChatProgressSnapshot["facts"] }) {
  const routeReasons = facts.route_reason_codes.map(routeReasonLabel);
  return (
    <div className="trace-facts">
      {facts.objective ? (
        <div><span>检索目标</span><strong>{facts.objective}</strong></div>
      ) : null}
      {facts.queries.length ? (
        <div>
          <span>查询</span>
          <ul>{facts.queries.map((query) => <li key={query}>{query}</li>)}</ul>
        </div>
      ) : null}
      {routeReasons.length ? (
        <div><span>路由依据</span><strong>{routeReasons.join("、")}</strong></div>
      ) : null}
      {facts.decision ? (
        <div><span>当前决定</span><strong>{decisionLabel(facts.decision)}</strong></div>
      ) : null}
      {facts.evidence_count !== null ? (
        <div><span>可用证据</span><strong>{facts.evidence_count} 项</strong></div>
      ) : null}
      {facts.retrieval_calls !== null ? (
        <div><span>检索次数</span><strong>{facts.retrieval_calls} 次</strong></div>
      ) : null}
      {facts.new_evidence_count !== null ? (
        <div><span>本轮新增</span><strong>{facts.new_evidence_count} 项</strong></div>
      ) : null}
      {facts.research_status ? (
        <div>
          <span>研究结论</span>
          <strong>{researchStatusLabel(facts.research_status)}</strong>
        </div>
      ) : null}
      {facts.covered_aspects.length ? (
        <div><span>已覆盖</span><strong>{facts.covered_aspects.join("、")}</strong></div>
      ) : null}
      {facts.missing_aspects.length ? (
        <div><span>仍缺少</span><strong>{facts.missing_aspects.join("、")}</strong></div>
      ) : null}
      {facts.conflict_count ? (
        <div><span>冲突</span><strong>{facts.conflict_count} 项</strong></div>
      ) : null}
    </div>
  );
}

function completedProgress(run: ChatRun): ChatProgressSnapshot {
  return {
    run_id: run.run_id,
    attempt: run.attempt,
    seq: 1,
    active_stage: "persist_result",
    activity: "persist_result",
    completed_stages: [...PROGRESS_STAGES],
    status: "completed",
    requested_mode: run.workflow.requested_mode,
    resolved_mode: run.workflow.resolved_mode,
    facts: stageFacts(run, "persist_result"),
  };
}

function progressStageLabel(stage: ChatProgressStage): string {
  const labels: Record<ChatProgressStage, string> = {
    understand_query: "理解问题",
    select_workflow: "选择工作流",
    retrieve_evidence: "检索证据",
    assess_evidence: "评估证据",
    prepare_visual_evidence: "准备素材",
    generate_answer: "生成回答",
    validate_answer: "校验回答",
    persist_result: "保存结果",
  };
  return labels[stage];
}

function activityLabel(activity: ChatProgressSnapshot["activity"]): string {
  const labels: Record<ChatProgressSnapshot["activity"], string> = {
    load_context: "读取对话上下文",
    contextualize_query: "理解并改写问题",
    route_decision: "判断问题复杂度",
    simple_search: "执行单次检索",
    agent_decision: "规划下一步检索",
    agent_search: "执行 Agent 查询",
    retrieval_complete: "整理本轮检索结果",
    verify_coverage: "Verifier 检查覆盖度",
    research_complete: "研究阶段结束",
    assess_evidence: "评估证据是否足够",
    prepare_visual_evidence: "准备可引用的视觉证据",
    generate_answer: "基于证据生成回答",
    validate_answer: "校验结构与引用",
    persist_result: "保存最终回答",
  };
  return labels[activity];
}

function activityDescription(activity: ChatProgressSnapshot["activity"]): string {
  const descriptions: Record<ChatProgressSnapshot["activity"], string> = {
    load_context: "读取本次问题、会话上下文与冻结配置。",
    contextualize_query: "把当前问题整理成可独立检索的查询。",
    route_decision: "根据问题是否需要多视角或多跳信息选择 Simple / Agent。",
    simple_search: "使用冻结的检索配置查找最相关证据。",
    agent_decision: "根据已有观察决定继续搜索还是进入覆盖度验证。",
    agent_search: "按受控目标执行最多三条并行查询。",
    retrieval_complete: "合并并去重本轮结果，只统计可用证据。",
    verify_coverage: "独立检查证据覆盖、缺口与冲突，并决定是否继续检索。",
    research_complete: "检索决策已结束，固定用于回答的证据集合。",
    assess_evidence: "依据回答策略判断充分、部分覆盖或拒答。",
    prepare_visual_evidence: "选择与文字证据相关的图片或表格素材。",
    generate_answer: "只使用已选证据组织回答和引用。",
    validate_answer: "检查回答结构、引用编号和证据约束。",
    persist_result: "把最终回答和可核验事实写入本地数据库。",
  };
  return descriptions[activity];
}

function decisionLabel(decision: NonNullable<
  ChatProgressSnapshot["facts"]["decision"]
>): string {
  return {
    select_simple: "选择 Simple",
    select_agent: "选择 Agent",
    search_evidence: "继续执行检索",
    continue_search: "覆盖仍不足，继续检索",
    finish_research: "证据研究结束，进入回答",
  }[decision];
}

function routeReasonLabel(reason: ChatRun["workflow"]["route_reason_codes"][number]) {
  return {
    single_lookup: "单点查询",
    direct_summary: "直接总结",
    multi_view_required: "需要多视角",
    multi_hop_required: "需要多跳检索",
    evidence_uncertain: "证据不确定",
    router_invalid: "路由结果无效，已回退",
    router_unavailable: "路由不可用，已回退",
  }[reason];
}

function searchResultLabel(result: ChatSearchResult): string {
  return {
    evidence_found: "找到新证据",
    no_evidence: "未找到新证据",
    verification_gap: "根据覆盖缺口继续",
  }[result];
}

type ChatSearchResult = "evidence_found" | "no_evidence" | "verification_gap";

function workflowModeLabel(mode: ChatWorkflowMode): string {
  if (mode === "agent") return "Agent";
  if (mode === "auto") return "Auto";
  return "Simple";
}

function researchStatusLabel(status: NonNullable<
  ChatRun["workflow"]["research_result"]
>["status"]): string {
  const labels: Record<string, string> = {
    sufficient: "研究充分",
    partial: "部分覆盖",
    no_evidence: "未找到证据",
    conflict: "证据冲突",
    premise_unsupported: "前提无依据",
  };
  return labels[String(status)] ?? "研究完成";
}

function terminationLabel(reason: NonNullable<
  ChatRun["workflow"]["research_result"]
>["termination_reason"]): string {
  const labels: Record<string, string> = {
    sufficient: "覆盖完成",
    partial: "部分完成",
    no_evidence: "无证据",
    no_progress: "无新增证据",
    budget_exhausted: "达到检索上限",
    conflict_unresolved: "冲突未解决",
    premise_unsupported: "问题前提不成立",
  };
  return labels[reason] ?? reason;
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

function EmptyKnowledgeBase({ onManage }: { onManage: () => void }) {
  return (
    <section className="welcome">
      <div className="welcome-mark">K</div>
      <h2>还没有可用的知识库</h2>
      <p>创建知识库并添加文档后，就可以开始基于资料提问。</p>
      <button className="primary-button" type="button" onClick={onManage}>创建知识库</button>
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

function emptyProgress(runId: string | null): ChatProgressState {
  return {
    runId,
    attempt: 0,
    lastSeq: 0,
    snapshot: null,
    stageRecords: {},
    mode: "idle",
  };
}

function applyProgress(
  current: ChatProgressState,
  activeRunId: string,
  event: ChatProgressSnapshot,
): ChatProgressState {
  if (event.run_id !== activeRunId) return current;
  const base = current.runId === activeRunId
    ? current
    : emptyProgress(activeRunId);
  if (event.attempt < base.attempt) return base;
  if (event.attempt === base.attempt && event.seq <= base.lastSeq) return base;
  const stageRecords = event.attempt > base.attempt
    ? { [event.active_stage]: event }
    : { ...base.stageRecords, [event.active_stage]: event };
  return {
    runId: activeRunId,
    attempt: event.attempt,
    lastSeq: event.seq,
    snapshot: event,
    stageRecords,
    mode: "live",
  };
}

function disconnectProgress(
  current: ChatProgressState,
  runId: string,
): ChatProgressState {
  const base = current.runId === runId ? current : emptyProgress(runId);
  return { ...base, mode: "disconnected" };
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "操作未能完成，请稍后重试。";
}
