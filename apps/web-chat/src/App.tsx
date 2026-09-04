import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { ApiClient, ApiClientError, loadRuntimeConfig } from "./api/client";
import type {
  ChatAgentTraceEvent,
  ChatMessage,
  ChatProgressSnapshot,
  ChatProgressStage,
  ChatRun,
  ChatRunCreate,
  ChatSession,
  GraphConfig,
  GraphConfigUpdate,
  GraphSchemaProfile,
  KnowledgeBase,
  ModelSettings,
  RerankMode,
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
import { UI_POLICY } from "./uiPolicy";
import {
  isChatViewScopeCurrent,
  isRequestSequenceCurrent,
  type ChatViewScope,
} from "./requestScope";
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

  useEffect(() => {
    void loadRuntimeConfig()
      .then((config) => setClient(new ApiClient(config)))
      .catch((error) => setStartupError(errorMessage(error)));
  }, []);

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
    <KnowledgeChat client={client} />
  );
}

export function KnowledgeChat({
  client,
}: {
  client: ApiClient;
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
  const [retrievalMode, setRetrievalMode] = useState<"text" | "auto" | "graph">("auto");
  const [rerankMode, setRerankMode] = useState<RerankMode>("classic");
  const [graphConfig, setGraphConfig] = useState<GraphConfig | null>(null);
  const [graphConfigLoading, setGraphConfigLoading] = useState(false);
  const [graphConfigError, setGraphConfigError] = useState<string | null>(null);
  const [graphSchemaProfiles, setGraphSchemaProfiles] = useState<GraphSchemaProfile[]>([]);
  const [graphSchemaProfilesError, setGraphSchemaProfilesError] = useState<string | null>(null);
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
  const chatGeneration = useRef(0);
  const sessionRequestSequence = useRef(0);
  const graphConfigGeneration = useRef(0);
  const selectedKnowledgeBaseIdRef = useRef(selectedKnowledgeBaseId);
  const selectedSessionIdRef = useRef(selectedSessionId);
  const previousChatKnowledgeBaseIdRef = useRef(selectedKnowledgeBaseId);
  const previousChatSessionIdRef = useRef(selectedSessionId);
  const skipSessionScopeInvalidationRef = useRef(false);
  selectedKnowledgeBaseIdRef.current = selectedKnowledgeBaseId;
  selectedSessionIdRef.current = selectedSessionId;

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
  const graphReady = Boolean(
    graphConfig?.enabled && graphConfig.status === "ready",
  );
  const graphTopK = Math.min(
    20,
    Math.max(4, selectedKnowledgeBase?.retrieval_defaults.top_k ?? 4),
  );
  const chatModels = modelSettings?.profiles.filter((profile) => (
    profile.kind === "chat"
    && profile.enabled
    && profile.validation_status === "valid"
    && profile.provider_secret_available
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
    const token: ChatViewScope = {
      generation: chatGeneration.current,
      knowledgeBaseId,
      sessionId: null,
    };
    const sequence = ++sessionRequestSequence.current;
    const isCurrent = () => isChatViewScopeCurrent(token, {
      generation: chatGeneration.current,
      knowledgeBaseId: selectedKnowledgeBaseIdRef.current,
      sessionId: null,
    }) && isRequestSequenceCurrent(sequence, sessionRequestSequence.current);
    setSessionsLoading(true);
    setSessionsError(null);
    try {
      const page = await client.listChatSessions(knowledgeBaseId, cursor);
      if (!isCurrent()) return;
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
      if (isCurrent()) setSessionsError(errorMessage(error));
    } finally {
      if (isCurrent()) setSessionsLoading(false);
    }
  }, [client]);

  const loadMessages = useCallback(async (
    sessionId: string,
    cursor?: string,
  ) => {
    const generation = ++messageGeneration.current;
    const token: ChatViewScope = {
      generation: chatGeneration.current,
      knowledgeBaseId: selectedKnowledgeBaseIdRef.current,
      sessionId,
    };
    const isCurrent = () => isChatViewScopeCurrent(token, {
      generation: chatGeneration.current,
      knowledgeBaseId: selectedKnowledgeBaseIdRef.current,
      sessionId: selectedSessionIdRef.current,
    }) && isRequestSequenceCurrent(generation, messageGeneration.current);
    setMessagesLoading(true);
    setMessagesError(null);
    try {
      const page = await client.listChatMessages(sessionId, cursor);
      if (!isCurrent()) return;
      setMessages((current) => cursor
        ? mergeMessages(current, page.items)
        : page.items);
      setMessagesCursor(page.next_cursor);
    } catch (error) {
      if (isCurrent()) {
        setMessagesError(errorMessage(error));
      }
    } finally {
      if (isCurrent()) setMessagesLoading(false);
    }
  }, [client]);

  const loadGraphConfig = useCallback(async (
    knowledgeBaseId: string,
    quiet = false,
  ): Promise<GraphConfig | null> => {
    const generation = ++graphConfigGeneration.current;
    if (!quiet) {
      setGraphConfigLoading(true);
      setGraphConfigError(null);
    }
    try {
      const value = await client.getGraphConfig(knowledgeBaseId);
      if (
        generation === graphConfigGeneration.current
        && selectedKnowledgeBaseIdRef.current === knowledgeBaseId
      ) {
        setGraphConfig(value);
        setGraphConfigError(null);
      }
      return value;
    } catch (error) {
      if (
        generation === graphConfigGeneration.current
        && selectedKnowledgeBaseIdRef.current === knowledgeBaseId
      ) {
        setGraphConfigError(errorMessage(error));
      }
      return null;
    } finally {
      if (
        !quiet
        && generation === graphConfigGeneration.current
        && selectedKnowledgeBaseIdRef.current === knowledgeBaseId
      ) {
        setGraphConfigLoading(false);
      }
    }
  }, [client]);

  const updateGraphConfig = useCallback(async (
    payload: GraphConfigUpdate,
  ): Promise<GraphConfig> => {
    const knowledgeBaseId = selectedKnowledgeBaseIdRef.current;
    if (!knowledgeBaseId) throw new Error("请先选择知识库。");
    const value = await client.updateGraphConfig(knowledgeBaseId, payload);
    if (selectedKnowledgeBaseIdRef.current === knowledgeBaseId) {
      ++graphConfigGeneration.current;
      setGraphConfig(value);
      setGraphConfigError(null);
      setGraphConfigLoading(false);
    }
    return value;
  }, [client]);

  useEffect(() => {
    let cancelled = false;
    void client.getGraphSchemaProfiles().then((profiles) => {
      if (!cancelled) {
        setGraphSchemaProfiles(profiles);
        setGraphSchemaProfilesError(null);
      }
    }).catch((error) => {
      if (!cancelled) setGraphSchemaProfilesError(errorMessage(error));
    });
    return () => {
      cancelled = true;
    };
  }, [client]);

  useEffect(() => {
    void loadKnowledgeBases();
  }, [loadKnowledgeBases]);

  useEffect(() => {
    ++graphConfigGeneration.current;
    setGraphConfig(null);
    setGraphConfigError(null);
    if (!selectedKnowledgeBaseId) {
      setGraphConfigLoading(false);
      return;
    }
    void loadGraphConfig(selectedKnowledgeBaseId);
  }, [loadGraphConfig, selectedKnowledgeBaseId]);

  useEffect(() => {
    if (
      !selectedKnowledgeBaseId
      || graphConfig?.knowledge_base_id !== selectedKnowledgeBaseId
      || graphConfig.status !== "building"
    ) return;
    const timer = window.setInterval(() => {
      void loadGraphConfig(selectedKnowledgeBaseId, true);
    }, UI_POLICY.graphPollMs);
    return () => window.clearInterval(timer);
  }, [graphConfig, loadGraphConfig, selectedKnowledgeBaseId]);

  useEffect(() => {
    ++chatGeneration.current;
    storeKnowledgeBaseId(selectedKnowledgeBaseId || null);
    setSubmitting(false);
    setPendingRun(null);
    setSubmissionError(null);
    setRetrievalMode("auto");
    setSessions([]);
    setSessionsLoading(false);
    setSessionsError(null);
    setSelectedSessionId(null);
    setMessages([]);
    setMessagesLoading(false);
    setMessagesError(null);
    setCurrentRun(null);
    setRunCache({});
    closeEvidence();
    if (selectedKnowledgeBaseId) {
      void loadSessions(selectedKnowledgeBaseId);
    }
  }, [loadSessions, selectedKnowledgeBaseId]);

  useEffect(() => {
    if (!graphReady && retrievalMode === "graph") setRetrievalMode("auto");
  }, [graphReady, retrievalMode]);

  useEffect(() => {
    if (selectedKnowledgeBase) {
      setRerankMode(selectedKnowledgeBase.retrieval_defaults.rerank_mode);
    }
  }, [
    selectedKnowledgeBase?.id,
    selectedKnowledgeBase?.retrieval_defaults.rerank_mode,
  ]);

  useEffect(() => {
    const knowledgeBaseChanged = (
      previousChatKnowledgeBaseIdRef.current !== selectedKnowledgeBaseId
    );
    previousChatKnowledgeBaseIdRef.current = selectedKnowledgeBaseId;
    if (knowledgeBaseChanged) {
      skipSessionScopeInvalidationRef.current = true;
      previousChatSessionIdRef.current = selectedSessionId;
      return;
    }
    if (skipSessionScopeInvalidationRef.current) {
      skipSessionScopeInvalidationRef.current = false;
      previousChatSessionIdRef.current = selectedSessionId;
      return;
    }
    if (previousChatSessionIdRef.current === selectedSessionId) return;
    previousChatSessionIdRef.current = selectedSessionId;
    ++chatGeneration.current;
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
    const generation = chatGeneration.current;
    const token: ChatViewScope = {
      generation,
      knowledgeBaseId: selectedKnowledgeBaseId,
      sessionId: selectedSessionId,
    };
    const generating = [...messages].reverse().find(
      (item) => item.role === "assistant"
        && item.assistant_status === "generating"
        && item.run_id,
    );
    if (!generating?.run_id) return;
    void client.getChatRun(generating.run_id)
      .then((run) => {
        if (
          isChatViewScopeCurrent(token, {
            generation: chatGeneration.current,
            knowledgeBaseId: selectedKnowledgeBaseIdRef.current,
            sessionId: selectedSessionIdRef.current,
          })
          && run.session_id === selectedSessionId
          && run.knowledge_base_id === selectedKnowledgeBaseId
        ) setCurrentRun(run);
      })
      .catch((error) => {
        if (isChatViewScopeCurrent(token, {
          generation: chatGeneration.current,
          knowledgeBaseId: selectedKnowledgeBaseIdRef.current,
          sessionId: selectedSessionIdRef.current,
        })) setMessagesError(errorMessage(error));
      });
  }, [client, currentRun, messages, messagesLoading, selectedKnowledgeBaseId, selectedSessionId]);

  useEffect(() => {
    if (!selectedSessionId || messagesLoading) return;
    const missingRunIds = [...new Set(messages.flatMap((item) => (
      item.role === "assistant" && item.run_id && !runCache[item.run_id]
        ? [item.run_id]
        : []
    )))];
    if (!missingRunIds.length) return;
    const token: ChatViewScope = {
      generation: chatGeneration.current,
      knowledgeBaseId: selectedKnowledgeBaseId,
      sessionId: selectedSessionId,
    };
    let cancelled = false;
    void Promise.allSettled(
      missingRunIds.map((runId) => client.getChatRun(runId)),
    ).then((results) => {
      if (cancelled || !isChatViewScopeCurrent(token, {
        generation: chatGeneration.current,
        knowledgeBaseId: selectedKnowledgeBaseIdRef.current,
        sessionId: selectedSessionIdRef.current,
      })) return;
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
    setProgress(emptyProgress(currentRun?.run_id ?? null));
  }, [currentRun?.run_id]);

  useEffect(() => {
    if (!currentRun) {
      setDeliveryMode("idle");
      return;
    }
    const token: ChatViewScope = {
      generation: chatGeneration.current,
      knowledgeBaseId: currentRun.knowledge_base_id,
      sessionId: currentRun.session_id,
    };
    const isCurrent = () => isChatViewScopeCurrent(token, {
      generation: chatGeneration.current,
      knowledgeBaseId: selectedKnowledgeBaseIdRef.current,
      sessionId: selectedSessionIdRef.current,
    });
    if (!isCurrent()) return;
    setRunCache((current) => ({ ...current, [currentRun.run_id]: currentRun }));
    if (isTerminal(currentRun)) {
      setDeliveryMode("idle");
      if (isCurrent() && currentRun.session_id === selectedSessionIdRef.current) {
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
        if (cancelled || !isCurrent() || next.knowledge_base_id !== token.knowledgeBaseId || next.session_id !== token.sessionId) return;
        setCurrentRun(next);
        if (!isTerminal(next)) {
          pollTimer = window.setTimeout(
            poll,
            UI_POLICY.runPollInitialMs + Math.random() * UI_POLICY.runPollJitterMs,
          );
        }
      } catch {
        if (!cancelled) pollTimer = window.setTimeout(poll, UI_POLICY.runPollRetryMs);
      }
    };
    const beginPolling = () => {
      if (cancelled || polling) return;
      polling = true;
      closeStream?.();
      closeStream = null;
      setProgress((current) => disconnectProgress(current, currentRun.run_id));
      setDeliveryMode("polling");
      void poll();
    };
    const settle = async (statusUrl: string) => {
      closeStream?.();
      closeStream = null;
      try {
        const next = await client.getChatRun(statusUrl);
        if (!cancelled && isCurrent() && next.knowledge_base_id === token.knowledgeBaseId && next.session_id === token.sessionId) setCurrentRun(next);
      } catch {
        beginPolling();
      }
    };
    closeStream = client.subscribeChatRun(currentRun.events_url, {
      open: () => !cancelled && isCurrent() && setDeliveryMode("sse"),
      completed: (event) => void settle(event.status_url),
      failed: (event) => void settle(event.status_url),
      progress: (event) => {
        if (isCurrent()) setProgress(
          (current) => applyProgress(current, currentRun.run_id, event),
        );
      },
      progressInvalid: () => {
        if (isCurrent()) setProgress(
          (current) => disconnectProgress(current, currentRun.run_id),
        );
      },
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
    if (submitting || pendingRun) return;
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
    if (submitting || pendingRun) return;
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
    const submissionKnowledgeBaseId = selectedKnowledgeBase.id;
    setSubmitting(true);
    setSubmissionError(null);
    let sessionId = selectedSessionId;
    try {
      if (!sessionId) {
        const created = await client.createChatSession(
          selectedKnowledgeBase.id,
          questionTitle(question),
        );
        if (selectedKnowledgeBaseIdRef.current !== submissionKnowledgeBaseId) return;
        sessionId = created.id;
        selectedSessionIdRef.current = created.id;
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
          retrieval: {
            mode: retrievalMode,
            top_k: retrievalMode === "graph"
              ? graphTopK
              : selectedKnowledgeBase.retrieval_defaults.top_k,
            rerank_mode: rerankMode,
          },
          model_profile_revision_id: selectedChatModelRevisionId,
        },
      };
      setPendingRun(pending);
      await performRun(pending);
    } catch (error) {
      if (selectedKnowledgeBaseIdRef.current === submissionKnowledgeBaseId) {
        setSubmissionError(errorMessage(error));
      }
    } finally {
      if (selectedKnowledgeBaseIdRef.current === submissionKnowledgeBaseId) {
        setSubmitting(false);
      }
    }
  };

  const performRun = async (pending: PendingRun) => {
    const submissionKnowledgeBaseId = pending.payload.knowledge_base_id;
    const submissionSessionId = pending.payload.session_id;
    const isCurrent = () => (
      selectedKnowledgeBaseIdRef.current === submissionKnowledgeBaseId
      && selectedSessionIdRef.current === submissionSessionId
    );
    setSubmitting(true);
    setSubmissionError(null);
    try {
      const run = await client.createChatRun(
        pending.payload,
        pending.idempotencyKey,
      );
      if (!isCurrent()) return;
      setCurrentRun(run);
      setRunCache((current) => ({ ...current, [run.run_id]: run }));
      setDraft("");
      setPendingRun(null);
      await loadMessages(run.session_id);
    } catch (error) {
      if (isCurrent()) setSubmissionError(errorMessage(error));
      throw error;
    } finally {
      if (isCurrent()) setSubmitting(false);
    }
  };

  const changeRetrievalMode = (next: "text" | "auto" | "graph") => {
    if (next === "graph" && !graphReady) return;
    if (pendingRun) {
      setPendingRun(null);
      setSubmissionError(null);
    }
    if (next === "graph") setRerankMode("classic");
    setRetrievalMode(next);
  };

  const changeRerankMode = (next: RerankMode) => {
    if (
      next === "none" && retrievalMode === "graph"
    ) return;
    if (
      next === "local_minilm_v1"
      && (retrievalMode === "graph"
        ? graphTopK
        : selectedKnowledgeBase?.retrieval_defaults.top_k ?? 100) > 20
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
    const token: ChatViewScope = {
      generation: chatGeneration.current,
      knowledgeBaseId: selectedKnowledgeBaseId,
      sessionId: selectedSessionId,
    };
    const isCurrent = () => isChatViewScopeCurrent(token, {
      generation: chatGeneration.current,
      knowledgeBaseId: selectedKnowledgeBaseIdRef.current,
      sessionId: selectedSessionIdRef.current,
    });
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
      if (!isCurrent() || run.knowledge_base_id !== selectedKnowledgeBaseIdRef.current) {
        throw new ApiClientError("来源不属于当前知识库。");
      }
      setRunCache((current) => ({ ...current, [runId]: run }));
      setEvidence({ run, runId, ordinal });
    } catch (error) {
      if (isCurrent()) setEvidenceError(errorMessage(error));
    } finally {
      if (isCurrent()) setEvidenceLoading(false);
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
            disabled={knowledgeBasesLoading || !knowledgeBases.length || submitting || Boolean(pendingRun)}
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
            disabled={!selectedKnowledgeBase || submitting || Boolean(pendingRun)}
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
                && !submitting
                && !pendingRun
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
                    if (submitting || pendingRun) return;
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
                && !submitting
                && !pendingRun
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
          graphConfig={graphConfig}
          graphSchemaProfiles={graphSchemaProfiles}
          graphSchemaProfilesError={graphSchemaProfilesError}
          graphConfigLoading={graphConfigLoading}
          graphConfigError={graphConfigError}
          onRefreshGraphConfig={() => selectedKnowledgeBaseId
            ? loadGraphConfig(selectedKnowledgeBaseId)
            : Promise.resolve(null)}
          onUpdateGraphConfig={updateGraphConfig}
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
                  kind="retrieval"
                  label="检索模式"
                  value={retrievalMode}
                  disabled={submitting || sessionBusy}
                  options={[
                    {
                      value: "text",
                      label: "文档检索模式",
                      description: "仅查文档分块：语义、关键词与邻域检索，不使用图谱。",
                    },
                    {
                      value: "auto",
                      label: "智能自适应模式（推荐）",
                      description: graphReady
                        ? "开放全部检索通道，由模型自主选择文档或实体关系。"
                        : "图谱未就绪时自动使用文档检索，不会自动开始建图。",
                    },
                    {
                      value: "graph",
                      label: "图谱优先模式",
                      description: graphReady
                        ? "强制图路径优先打包，再用混合文档证据回填。"
                        : graphConfigLoading
                          ? "正在读取当前知识库的图谱状态。"
                          : graphConfigError
                            ? "图谱状态不可用。"
                            : graphConfig?.status === "building"
                              ? "图谱正在构建完成前不可用。"
                              : graphConfig?.status === "failed"
                                ? "图谱构建失败，请先修复或重试。"
                                : "当前知识库尚未启用或完成图谱构建。",
                      disabled: !graphReady,
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
                      description: retrievalMode === "graph"
                        ? "图谱优先模式必须启用精排。"
                        : "直接使用原始检索顺序，资源开销最低。",
                      disabled: retrievalMode === "graph",
                    },
                    {
                      value: "classic",
                      label: "经典精排",
                      description: "使用现有关键词、向量与去重规则。",
                    },
                    {
                      value: "local_minilm_v1",
                      label: "本地 MiniLM",
                      description: (retrievalMode === "graph"
                        ? graphTopK
                        : selectedKnowledgeBase?.retrieval_defaults.top_k ?? 100) > 20
                          ? "本地模型要求知识库 Top K 不超过 20。"
                          : "本机离线 CrossEncoder 精排，相关性更强但更慢。",
                      disabled: (retrievalMode === "graph"
                        ? graphTopK
                        : selectedKnowledgeBase?.retrieval_defaults.top_k ?? 100) > 20,
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
  kind: "retrieval" | "rerank";
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
  kind: "retrieval" | "rerank";
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
  progress,
  onCitation,
}: {
  message: ChatMessage;
  run: ChatRun | null;
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
        {run ? (
          <ExecutionTrace
            run={run}
            progress={progress}
            generating={generating}
          />
        ) : null}
        {generating ? (
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

function ExecutionTrace({
  run,
  progress,
  generating,
}: {
  run: ChatRun;
  progress: ChatProgressState | null;
  generating: boolean;
}) {
  const terminal = isTerminal(run);
  const [expanded, setExpanded] = useState(false);
  useEffect(() => setExpanded(false), [run.run_id]);

  const disconnected = generating && progress?.mode === "disconnected";
  const metrics = answerProcessMetrics(run);
  const content = (
    <div className="execution-trace-body">
      {disconnected ? (
        <div className="answer-process-notice" role="status">
          实时进度连接已中断，回答仍在后台运行；这里保留最后一次确认的状态。
        </div>
      ) : null}
      {run.status === "completed" ? (
        <CompletedAnswerProcess run={run} metrics={metrics} />
      ) : (
        <ActiveAnswerProcess
          run={run}
          snapshot={progress?.snapshot ?? null}
          disconnected={disconnected}
        />
      )}
      {terminal ? <AnswerProcessTechnicalDetails run={run} metrics={metrics} /> : null}
    </div>
  );
  if (terminal) {
    return (
      <section className={`execution-trace terminal-trace${expanded ? " expanded" : ""}`}>
        <button
          className="answer-process-toggle"
          type="button"
          aria-expanded={expanded}
          onClick={() => setExpanded((value) => !value)}
        >
          <span>回答过程</span>
          <strong className={run.status === "completed" ? "complete" : "incomplete"}>
            {run.status === "completed" ? "已完成" : "未完成"}
          </strong>
        </button>
        {expanded ? content : null}
      </section>
    );
  }
  return (
    <section className="execution-trace" aria-label="实时回答过程">
      <header>
        <span>回答过程</span>
        <strong className="active">进行中</strong>
      </header>
      {content}
    </section>
  );
}

interface AnswerProcessMetrics {
  retrievalCalls: number;
  candidateCount: number;
  citationCount: number;
  calculationCalls: number;
  modelRounds: number;
  graphRelationsStatus: string | null;
  graphRelationsNewEvidenceCount: number;
  graphRelationsCallCount: number;
  graphRelationsHopCounts: {
    hop1Count: number;
    hop2Count: number;
    hop3Count: number;
  };
}

interface AnswerProcessStep {
  key: string;
  title: string;
  description: string;
  meta: string;
}

function CompletedAnswerProcess({
  run,
  metrics,
}: {
  run: ChatRun;
  metrics: AnswerProcessMetrics;
}) {
  const summary = completedAnswerSummary(run, metrics);
  const steps = completedAnswerSteps(run, metrics);
  return (
    <>
      <section className="answer-process-summary" aria-labelledby={`answer-summary-${run.run_id}`}>
        <h3 id={`answer-summary-${run.run_id}`}>{summary.title}</h3>
        <p>{summary.description}</p>
      </section>
      <AnswerProcessMetricList metrics={metrics} />
      <h3 className="answer-process-section-title">本次回答经历了什么</h3>
      <ol className="answer-process-steps">
        {steps.map((step, index) => (
          <li key={step.key}>
            <span className="answer-process-step-number" aria-hidden="true">
              {index + 1}
            </span>
            <div>
              <h4>{step.title}</h4>
              <p>{step.description}</p>
              <span className="answer-process-step-meta">{step.meta}</span>
            </div>
          </li>
        ))}
      </ol>
    </>
  );
}

function AnswerProcessMetricList({ metrics }: { metrics: AnswerProcessMetrics }) {
  return (
    <ul className="answer-process-metrics" aria-label="本次回答摘要">
      <li className="retrieval"><span>检索</span><strong>{metrics.retrievalCalls} 次</strong></li>
      <li className="candidate"><span>候选资料</span><strong>{metrics.candidateCount} 条</strong></li>
      <li className="citation"><span>最终引用</span><strong>{metrics.citationCount} 条</strong></li>
      {metrics.graphRelationsStatus ? (
        <li className="retrieval"><span>图谱关系检索</span><strong>{metrics.graphRelationsStatus}</strong></li>
      ) : null}
    </ul>
  );
}

function ActiveAnswerProcess({
  run,
  snapshot,
  disconnected,
}: {
  run: ChatRun;
  snapshot: ChatProgressSnapshot | null;
  disconnected: boolean;
}) {
  const failed = run.status === "failed" || run.status === "cancelled";
  const activity = liveActivityView(snapshot?.activity ?? "load_context");
  const meta = liveProgressMeta(snapshot);
  return (
    <>
      <section className={`answer-process-summary${failed ? " failed" : ""}`}>
        <h3>{failed
          ? run.status === "cancelled" ? "这次回答已停止" : "这次回答未能完成"
          : "正在查找资料并整理回答"}</h3>
        <p>{failed
          ? "这里只展示停止前已确认的状态；未完成的步骤不会被标记为完成。"
          : "进度会随着已确认的工作更新，不会预先补齐尚未发生的步骤。"}</p>
      </section>
      <div
        className={`answer-process-live-step${failed ? " failed" : ""}`}
        aria-live={failed || disconnected ? "off" : "polite"}
      >
        <span className="answer-process-step-number" aria-hidden="true">
          {activity.step}
        </span>
        <div>
          <span className="answer-process-live-label">
            {failed ? "最后确认的状态" : disconnected ? "最后收到的进度" : "当前进度"}
          </span>
          <h4>{activity.title}</h4>
          <p>{activity.description}</p>
          {meta ? <span className="answer-process-step-meta">{meta}</span> : null}
        </div>
      </div>
    </>
  );
}

function AnswerProcessTechnicalDetails({
  run,
  metrics,
}: {
  run: ChatRun;
  metrics: AnswerProcessMetrics;
}) {
  const [expanded, setExpanded] = useState(false);
  const modelName = run.model.profile_name || run.model.model;
  const hiddenDiagnosticCount = run.agent.trace?.events.filter(
    (event) => event.tool === "protocol" || event.status !== "ok",
  ).length ?? 0;
  return (
    <>
      <section className={`answer-process-technical${expanded ? " expanded" : ""}`}>
        <button
          type="button"
          aria-expanded={expanded}
          onClick={() => setExpanded((value) => !value)}
        >
          <span>查看技术详情</span>
          <small>
            {modelName} · 模型 {metrics.modelRounds} 轮 · 诊断信息已{expanded ? "展开" : "收起"}
          </small>
        </button>
        {expanded ? (
          <dl>
            <div><dt>Agent</dt><dd>Native Tool-Calling</dd></div>
            <div><dt>模型</dt><dd>{modelName}</dd></div>
            <div><dt>检索调用</dt><dd>{metrics.retrievalCalls} 次</dd></div>
            {metrics.graphRelationsStatus ? (
              <div><dt>图谱关系检索</dt><dd>{metrics.graphRelationsStatus}</dd></div>
            ) : null}
            {metrics.graphRelationsCallCount > 0 ? (
              <div><dt>图谱调用</dt><dd>{metrics.graphRelationsCallCount} 次</dd></div>
            ) : null}
            <div><dt>计算调用</dt><dd>{metrics.calculationCalls} 次</dd></div>
            {hiddenDiagnosticCount > 0 ? (
              <div><dt>校验调整</dt><dd>{hiddenDiagnosticCount} 次</dd></div>
            ) : null}
          </dl>
        ) : null}
      </section>
      <p className="answer-process-privacy-note">
        默认隐藏工具 ID、event ref 与协议状态；这些内部标记不会作为回答结果展示。
      </p>
    </>
  );
}

function answerProcessMetrics(run: ChatRun): AnswerProcessMetrics {
  const trace = run.agent.trace;
  const searchEvents = trace?.events.filter(
    (event) =>
      (
        event.tool === "search_knowledge_base"
        || event.tool === "semantic_search"
        || event.tool === "keyword_search"
        || event.tool === "read_chunk_context"
      )
      && event.status === "ok",
  ) ?? [];
  const eventCandidateCount = Math.max(0, ...searchEvents.map((event) => event.count));
  const graphEvents = trace?.events.filter(
    (event) => event.tool === "search_graph_relations",
  ) ?? [];
  const lastGraphEvent = graphEvents[graphEvents.length - 1];
  const adaptiveProfile = run.retrieval.mode === "auto";
  const graphRelationsStatus = adaptiveProfile
    ? graphRelationsStatusLabel(
      lastGraphEvent?.route_result_code,
      lastGraphEvent?.new_evidence_count ?? 0,
    )
    : null;
  return {
    retrievalCalls: safeUsageCount(run, "retrieval_calls", searchEvents.length),
    candidateCount: safeUsageCount(run, "evidence_refs", eventCandidateCount),
    citationCount: run.citations.length,
    calculationCalls: safeUsageCount(
      run,
      "calculation_calls",
      trace?.events.filter(
        (event) => event.tool === "calculate" && event.status === "ok",
      ).length ?? 0,
    ),
    modelRounds: safeUsageCount(run, "model_rounds", 0),
    graphRelationsStatus,
    graphRelationsNewEvidenceCount: lastGraphEvent?.new_evidence_count ?? 0,
    graphRelationsCallCount: graphEvents.length,
    graphRelationsHopCounts: {
      hop1Count: lastGraphEvent?.hop1_count ?? 0,
      hop2Count: lastGraphEvent?.hop2_count ?? 0,
      hop3Count: lastGraphEvent?.hop3_count ?? 0,
    },
  };
}

function graphRelationsStatusLabel(
  status: ChatAgentTraceEvent["route_result_code"] | undefined,
  count: number,
): string {
  switch (status) {
    case "admitted":
      return `已返回 · 新增 ${count} 条`;
    case "no_evidence":
      return "已检索但无新增证据";
    case "not_ready":
      return "图谱当前未就绪";
    case "timeout":
      return "图谱检索超时";
    case "unavailable":
      return "图谱当前不可用";
    case "rejected":
      return "未执行";
    case "not_requested":
    case undefined:
      return "未请求";
    default:
      return "未请求";
  }
}

function safeUsageCount(run: ChatRun, key: string, fallback: number): number {
  const value = run.agent.trace?.usage[key];
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(0, Math.trunc(value))
    : fallback;
}

function completedAnswerSummary(
  run: ChatRun,
  metrics: AnswerProcessMetrics,
): { title: string; description: string } {
  const outcome = run.agent.trace?.outcome ?? (run.answer ? "answered" : "refused");
  if (outcome === "clarify") {
    return {
      title: "系统需要先和你确认问题的具体指向",
      description: "问题存在歧义，系统没有猜测你的意图，请根据追问补充说明。",
    };
  }
  if (outcome === "refused") {
    return {
      title: "这次回答没有找到足够可靠的支持材料",
      description: metrics.retrievalCalls > 0
        ? `系统检索了 ${metrics.retrievalCalls} 次，但没有用不可靠的候选内容拼凑答案。`
        : "系统没有生成缺少可靠依据的推测性回答。",
    };
  }
  if (outcome === "partial") {
    return {
      title: `这次回答只保留了有可靠来源的部分内容，并使用 ${metrics.citationCount} 条来源`,
      description: `${candidateSummaryDescription(metrics)}${graphitiSummaryDescription(metrics)}`,
    };
  }
  return {
    title: metrics.retrievalCalls > 0
      ? `这次回答查找了 ${metrics.retrievalCalls} 次资料，最终使用 ${metrics.citationCount} 条来源`
      : `这次回答已经完成，最终使用 ${metrics.citationCount} 条来源`,
    description: `${candidateSummaryDescription(metrics)}${graphitiSummaryDescription(metrics)}`,
  };
}

function candidateSummaryDescription(metrics: AnswerProcessMetrics): string {
  if (metrics.retrievalCalls === 0) {
    return "本次没有调用知识库检索；最终引用数量仍以回答实际采用的来源为准。";
  }
  if (metrics.candidateCount === 0) {
    return "这次检索没有产生可用候选资料，因此不会把内部调用记录当作引用展示。";
  }
  return `检索到的 ${metrics.candidateCount} 条内容只是候选资料；只有经过整理并被最终回答采用的来源才会显示为引用。`;
}

function graphitiSummaryDescription(metrics: AnswerProcessMetrics): string {
  const { graphRelationsStatus: status } = metrics;
  if (!status) return "";
  const hopSummary = [
    metrics.graphRelationsHopCounts.hop1Count > 0
      ? `一跳 ${metrics.graphRelationsHopCounts.hop1Count} 条`
      : "",
    metrics.graphRelationsHopCounts.hop2Count > 0
      ? `两跳 ${metrics.graphRelationsHopCounts.hop2Count} 条`
      : "",
    metrics.graphRelationsHopCounts.hop3Count > 0
      ? `三跳 ${metrics.graphRelationsHopCounts.hop3Count} 条`
      : "",
  ].filter(Boolean).join("、");
  const hopText = metrics.graphRelationsCallCount > 0 && hopSummary
    ? `（${hopSummary}）`
    : "";
  if (metrics.graphRelationsNewEvidenceCount > 0) {
    return ` 图谱关系检索新增 ${metrics.graphRelationsNewEvidenceCount} 条来源候选${hopText}。`;
  }
  return ` 图谱关系检索（第 ${metrics.graphRelationsCallCount} 次）：${status}${hopText}。`;
}

function completedAnswerSteps(
  run: ChatRun,
  metrics: AnswerProcessMetrics,
): AnswerProcessStep[] {
  const outcome = run.agent.trace?.outcome ?? (run.answer ? "answered" : "refused");
  const searchTitle = metrics.retrievalCalls > 1
    ? `查找资料 · 第 1–${metrics.retrievalCalls} 次`
    : metrics.retrievalCalls === 1 ? "查找资料 · 1 次" : "评估可用资料";
  const searchDescription = metrics.retrievalCalls > 0
    ? `共找到 ${metrics.candidateCount} 条候选内容；系统会继续筛选，候选资料不等于最终引用。${graphitiSummaryDescription(metrics)}`
    : "本次没有调用知识库检索，也不会虚构检索阶段或候选数量。";
  const verificationDescription = outcome === "refused"
    ? "没有足够可靠的来源支持结论，因此没有生成推测性回答。"
    : outcome === "clarify"
      ? "问题指向存在歧义，系统没有基于猜测生成回答。"
      : metrics.citationCount > 0
        ? `最终回答实际采用 ${metrics.citationCount} 条来源；未采用的候选内容不会显示为引用。`
        : "回答没有附带来源；界面不会把候选内容误标为最终引用。";
  const resultDescription = outcome === "refused"
    ? "本次以说明资料不足结束，没有输出无依据的结论。"
    : outcome === "clarify"
      ? "本次以追问结束，请补充说明后系统会继续回答。"
      : outcome === "partial"
        ? `已生成部分回答，并附上实际采用的 ${metrics.citationCount} 条来源。`
        : `回答已生成，并附上实际采用的 ${metrics.citationCount} 条来源。`;
  return [
    {
      key: "understand",
      title: "理解问题",
      description: "结合本次问题与会话上下文，确定需要查找和核对的知识范围。",
      meta: "输入已就绪",
    },
    {
      key: "search",
      title: searchTitle,
      description: searchDescription,
      meta: metrics.retrievalCalls > 0
        ? `${metrics.retrievalCalls} 次检索 · ${metrics.candidateCount} 条候选资料`
        : "未调用知识库检索",
    },
    {
      key: "verify",
      title: "整理并核验回答",
      description: verificationDescription,
      meta: metrics.citationCount > 0
        ? `${metrics.citationCount} 条最终引用来源`
        : "没有最终引用来源",
    },
    {
      key: "result",
      title: "完成结果",
      description: resultDescription,
      meta: outcome === "refused"
        ? "完成 · 未生成推测性结论"
        : outcome === "clarify"
          ? "完成 · 等待你的补充说明"
          : `完成 · 最终引用 ${metrics.citationCount} 条`,
    },
  ];
}

function liveActivityView(
  activity: ChatProgressSnapshot["activity"],
): { step: number; title: string; description: string } {
  const views: Record<ChatProgressSnapshot["activity"] | "semantic_search" | "keyword_search" | "read_chunk_context" | "list_documents", {
    step: number;
    title: string;
    description: string;
  }> = {
    load_context: {
      step: 1,
      title: "理解问题",
      description: "正在读取本次问题与会话上下文。",
    },
    tool_decision: {
      step: 1,
      title: "确定下一步",
      description: "正在判断是否需要查找资料、核对计算或整理回答。",
    },
    search_knowledge_base: {
      step: 2,
      title: "查找资料",
      description: "正在知识库中查找与问题相关的候选内容。",
    },
    semantic_search: {
      step: 2,
      title: "语义检索",
      description: "正在按概念和释义查找相关内容。",
    },
    keyword_search: {
      step: 2,
      title: "关键词检索",
      description: "正在按专名、编号或精确短语查找相关内容。",
    },
    read_chunk_context: {
      step: 2,
      title: "阅读相邻片段",
      description: "正在读取已命中片段的相邻上下文。",
    },
    list_documents: {
      step: 2,
      title: "盘点文档",
      description: "正在列出知识库中的文档清单。",
    },
    search_graph_relations: {
      step: 2,
      title: "图谱关系检索",
      description: "正在图谱中查找完整的关系路径证据。",
    },
    retrieval_complete: {
      step: 2,
      title: "整理候选资料",
      description: "正在合并并去除重复的检索结果。",
    },
    prepare_visual_evidence: {
      step: 2,
      title: "核对可引用素材",
      description: "正在确认与文字证据相关的图片或表格。",
    },
    calculate: {
      step: 3,
      title: "核对计算",
      description: "正在基于已找到的资料检查计算结果。",
    },
    submit_answer: {
      step: 3,
      title: "整理并核验回答",
      description: "正在提交回答并核对来源是否有效。",
    },
    generate_answer: {
      step: 3,
      title: "整理回答",
      description: "正在根据可用资料生成回答。",
    },
    validate_answer: {
      step: 3,
      title: "核验回答",
      description: "正在检查回答内容与引用来源。",
    },
    persist_result: {
      step: 4,
      title: "完成结果",
      description: "正在保存最终回答和引用来源。",
    },
  };
  return views[activity];
}

function liveProgressMeta(snapshot: ChatProgressSnapshot | null): string | null {
  if (!snapshot) return null;
  const facts = snapshot.facts;
  if (facts.retrieval_calls !== null && facts.evidence_count !== null) {
    return `${facts.retrieval_calls} 次检索 · ${facts.evidence_count} 条候选资料`;
  }
  if (facts.retrieval_calls !== null) return `已检索 ${facts.retrieval_calls} 次`;
  if (facts.evidence_count !== null) return `已找到 ${facts.evidence_count} 条候选资料`;
  return null;
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
