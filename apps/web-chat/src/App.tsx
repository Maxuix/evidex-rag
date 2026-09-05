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
import { ExecutionTimeline } from "./execution/ExecutionTimeline";
import { applyActivity, disconnectActivity, emptyActivity, type ActivityState } from "./execution/activityState";

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
  const [selectedScopeIds, setSelectedScopeIds] = useState<string[]>(() => readKnowledgeBaseId() ? [readKnowledgeBaseId()!] : []);
  const [scopeSaving, setScopeSaving] = useState(false);
  const selectedScopeIdsRef = useRef(selectedScopeIds);
  selectedScopeIdsRef.current = selectedScopeIds;
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
  const [scopeGraphReady, setScopeGraphReady] = useState<Record<string, boolean>>({});
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

  const [activity, setActivity] = useState<ActivityState>(() => emptyActivity(null));

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
  const graphReady = selectedScopeIds.some(id => id === selectedKnowledgeBaseId
    ? Boolean(graphConfig?.enabled && graphConfig.status === "ready") : scopeGraphReady[id]);
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
        const restored = page.items.find(item => item.id === next);
        if (restored) setSelectedScopeIds(restored.knowledge_base_ids ?? (restored.knowledge_base_id ? [restored.knowledge_base_id] : []));
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
    setSelectedScopeIds(selectedKnowledgeBaseId ? [selectedKnowledgeBaseId] : []);
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
    let cancelled = false;
    const identifiers = selectedScopeIds.filter(id => id !== selectedKnowledgeBaseId);
    void Promise.all(identifiers.map(async id => {
      try { const config = await client.getGraphConfig(id); return [id, Boolean(config?.enabled && config.status === "ready")] as const; }
      catch { return [id, false] as const; }
    })).then(entries => { if (!cancelled) setScopeGraphReady(Object.fromEntries(entries)); });
    return () => { cancelled = true; };
  }, [client, selectedKnowledgeBaseId, selectedScopeIds]);

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
      // The KB effect resets to null. A subsequent non-null selection must
      // still load its messages, even if the null reset was batched away.
      previousChatSessionIdRef.current = null;
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
          && runScopeKey(run) === scopeKey(selectedScopeIdsRef.current)
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
    setActivity(emptyActivity(currentRun?.run_id ?? null));
  }, [currentRun?.run_id]);

  useEffect(() => {
    if (!currentRun) {
      setDeliveryMode("idle");
      return;
    }
    const token: ChatViewScope = {
      generation: chatGeneration.current,
      knowledgeBaseId: selectedKnowledgeBaseIdRef.current,
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
    let finished = false;
    let closeStream: (() => void) | null = null;
    let pollTimer: number | null = null;
    let reconnectTimer: number | null = null;
    let reconnectAttempts = 0;
    let polling = false;
    const alive = () => !cancelled && !finished && isCurrent();
    const accept = (next: ChatRun) => {
      if (!alive() || next.run_id !== currentRun.run_id || runScopeKey(next) !== runScopeKey(currentRun) || next.session_id !== token.sessionId) return false;
      if (isTerminal(next)) {
        finished = true;
        closeStream?.(); closeStream = null;
        if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
        if (pollTimer !== null) window.clearTimeout(pollTimer);
      }
      setCurrentRun(next);
      return true;
    };
    const poll = async () => {
      if (!alive()) return;
      try {
        const next = await client.getChatRun(currentRun.run_id);
        if (accept(next) && !finished) pollTimer = window.setTimeout(poll, UI_POLICY.runPollInitialMs + Math.random() * UI_POLICY.runPollJitterMs);
      } catch {
        if (alive()) pollTimer = window.setTimeout(poll, UI_POLICY.runPollRetryMs);
      }
    };
    const beginPolling = () => {
      if (!alive()) return;
      closeStream?.(); closeStream = null;
      setProgress(current => disconnectProgress(current, currentRun.run_id));
      setActivity(current => disconnectActivity(current, currentRun.run_id));
      setDeliveryMode("polling");
      if (!polling) { polling = true; void poll(); }
      if (reconnectTimer === null && reconnectAttempts < 3) {
        const delay = [1000, 2500, 5000][reconnectAttempts++];
        reconnectTimer = window.setTimeout(() => { reconnectTimer = null; if (alive()) connect(); }, delay);
      }
    };
    const settle = async (event: { run_id: string; status_url: string }) => {
      if (!alive() || event.run_id !== currentRun.run_id) return;
      closeStream?.(); closeStream = null;
      try { accept(await client.getChatRun(event.status_url)); }
      catch { beginPolling(); }
    };
    const connect = () => {
      closeStream = client.subscribeChatRun(currentRun.events_url, {
        open: () => {
          if (!alive()) return;
          setDeliveryMode("sse");
          setProgress(current => ({ ...current, mode: current.snapshot ? "live" : "idle" }));
          setActivity(current => ({ ...current, mode: "live" }));
        },
        completed: event => void settle(event),
        failed: event => void settle(event),
        progress: event => { if (alive()) setProgress(current => applyProgress(current, currentRun.run_id, event)); },
        progressInvalid: () => { if (alive()) setProgress(current => disconnectProgress(current, currentRun.run_id)); },
        activity: event => { if (alive()) setActivity(current => applyActivity(current, currentRun.run_id, currentRun.attempt, event)); },
        activityInvalid: () => { if (alive()) setActivity(current => ({ ...current, invalid: true })); },
        error: beginPolling,
      });
    };
    connect();
    return () => {
      cancelled = true;
      closeStream?.();
      if (pollTimer !== null) window.clearTimeout(pollTimer);
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
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

  const changeScope = async (ids: string[]) => {
    if (scopeSaving || submitting || pendingRun) return;
    const previous = selectedScopeIds;
    const selected = [...new Set(ids)].sort();
    const sessionId = selectedSessionId;
    setSelectedScopeIds(selected);
    selectedScopeIdsRef.current = selected;
    setSubmissionError(null);
    // Empty selection is a local draft; no retrieval request can be submitted.
    if (!sessionId || !selected.length) return;
    setScopeSaving(true);
    try {
      const updated = await client.updateChatScope(sessionId, selected);
      if (selectedSessionIdRef.current === sessionId) setSessions(current => mergeById(current, [updated]));
    } catch (error) {
      if (selectedSessionIdRef.current === sessionId) {
        setSelectedScopeIds(previous);
        setSubmissionError(errorMessage(error));
      }
    } finally { setScopeSaving(false); }
  };

  const selectAllScopes = async () => {
    if (scopeSaving || submitting || pendingRun) return;
    const generation = chatGeneration.current;
    const sessionId = selectedSessionIdRef.current;
    setScopeSaving(true);
    try {
      const all: KnowledgeBase[] = [];
      let cursor: string | undefined;
      const seen = new Set<string>();
      do {
        const page = await client.listKnowledgeBases(cursor);
        if (generation !== chatGeneration.current || sessionId !== selectedSessionIdRef.current) return;
        all.push(...page.items);
        cursor = page.next_cursor ?? undefined;
        if (cursor && seen.has(cursor)) throw new Error("知识库分页未能完成，请稍后重试。");
        if (cursor) seen.add(cursor);
      } while (cursor);
      const ids = [...new Set(all.map(kb => kb.id))].sort();
      if (sessionId && ids.length) {
        const updated = await client.updateChatScope(sessionId, ids);
        if (generation !== chatGeneration.current || sessionId !== selectedSessionIdRef.current) return;
        setSessions(current => mergeById(current, [updated]));
      }
      setKnowledgeBases(all);
      setKnowledgeBaseCursor(null);
      setSelectedScopeIds(ids);
      selectedScopeIdsRef.current = ids;
    } catch (error) {
      if (generation === chatGeneration.current) setSubmissionError(errorMessage(error));
    } finally { setScopeSaving(false); }
  };

  const submit = async () => {
    if (
      !selectedKnowledgeBase
      || !selectedScopeIds.length
      || scopeSaving
      || !draft.trim()
      || submitting
      || sessionBusy
      || !chatModelConfigured
    ) return;
    const question = draft.trim();
    const submissionKnowledgeBaseId = selectedKnowledgeBase.id;
    const submittedScope = [...selectedScopeIds];
    setSubmitting(true);
    setSubmissionError(null);
    let sessionId = selectedSessionId;
    try {
      if (!sessionId) {
        const created = await client.createChatSession(
          submittedScope,
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
          knowledge_base_ids: submittedScope,
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
    const submissionScopeKey = scopeKey(pending.payload.knowledge_base_ids ?? (pending.payload.knowledge_base_id ? [pending.payload.knowledge_base_id] : []));
    const submissionSessionId = pending.payload.session_id;
    const isCurrent = () => (
      scopeKey(selectedScopeIdsRef.current) === submissionScopeKey
      && selectedSessionIdRef.current === submissionSessionId
    );
    setSubmitting(true);
    setSubmissionError(null);
    try {
      const run = await client.createChatRun(
        pending.payload,
        pending.idempotencyKey,
      );
      if (!isCurrent() || runScopeKey(run) !== submissionScopeKey || run.session_id !== submissionSessionId) return;
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
      if (!isCurrent() || run.session_id !== selectedSessionIdRef.current) {
        throw new ApiClientError("来源不属于当前会话。");
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
                    setSelectedScopeIds(session.knowledge_base_ids ?? (session.knowledge_base_id ? [session.knowledge_base_id] : []));
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
            <p>{selectedScopeIds.length > 1 ? `搜索范围：${selectedScopeIds.length} 个知识库` : selectedScopeIds.length === 1 ? knowledgeBases.find(kb => kb.id === selectedScopeIds[0])?.name ?? "已选择 1 个知识库" : "请选择知识库"}</p>
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
                  client={client}
                  activity={message.run_id === currentRun?.run_id ? activity : null}
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
            <fieldset className="search-scope" disabled={scopeSaving || submitting || Boolean(pendingRun)}>
              <legend>搜索范围{scopeSaving ? " · 保存中" : ""}</legend>
              <details>
                <summary>{selectedScopeIds.length ? selectedScopeIds.map(id => knowledgeBases.find(kb => kb.id === id)?.name ?? id).join("、") : "请选择知识库"}</summary>
                <div className="scope-options">
                  <button type="button" onClick={() => void selectAllScopes()}>全选</button>
                  <button type="button" onClick={() => void changeScope([])}>清空</button>
                  {knowledgeBases.map(kb => <label key={kb.id} title={kb.description || kb.name}>
                    <input type="checkbox" checked={selectedScopeIds.includes(kb.id)} onChange={event => void changeScope(event.target.checked ? [...selectedScopeIds, kb.id] : selectedScopeIds.filter(id => id !== kb.id))} />
                    {kb.name}
                  </label>)}
                  {knowledgeBaseCursor ? <button type="button" onClick={() => void loadKnowledgeBases(knowledgeBaseCursor)}>加载更多知识库</button> : null}
                </div>
              </details>
              {!selectedScopeIds.length ? <small>请至少选择一个知识库后发送。</small> : null}
              {currentRun && !isTerminal(currentRun) ? <small>本轮范围：{currentRun.knowledge_bases?.map(kb => kb.name).join("、") || runScopeKey(currentRun)}。选择变更用于下一轮。</small> : null}
            </fieldset>
            <textarea
              ref={textareaRef}
              value={draft}
              rows={1}
              maxLength={32768}
              disabled={!selectedKnowledgeBase || !selectedScopeIds.length || scopeSaving || !chatModelConfigured || submitting || sessionBusy}
              placeholder={
                !chatModelConfigured
                  ? "请先在右下角齿轮中选择对话模型"
                  : selectedKnowledgeBase
                  ? "询问所选知识库中的内容…"
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
                        ? "逐库优先检索局部图谱，未就绪的库会标明不可用。"
                        : graphConfigLoading
                          ? "正在读取当前知识库的图谱状态。"
                          : graphConfigError
                            ? "图谱状态不可用。"
                            : graphConfig?.status === "building"
                              ? "图谱正在构建完成前不可用。"
                              : graphConfig?.status === "failed"
                                ? "图谱构建失败，请先修复或重试。"
                                : "所选知识库尚未启用或完成图谱构建。",
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
            回答依据本轮所选知识库，请核对重要信息。
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
  client,
  activity,
  run,
  progress,
  onCitation,
}: {
  message: ChatMessage;
  client: ApiClient;
  activity: ActivityState | null;
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
          <ExecutionTimeline key={run.run_id} run={run} activity={activity} progress={progress} client={client} onCitation={onCitation} />
        ) : null}
        {generating ? (!run ? (
          <div className="thinking" aria-live="polite">
            <span /><span /><span />
            <strong>正在查找资料并整理回答</strong>
          </div>
        ) : null) : failed ? (
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

function scopeKey(ids: readonly string[]): string { return [...new Set(ids)].sort().join(","); }
function runScopeKey(run: ChatRun): string { return scopeKey(run.knowledge_base_ids ?? run.knowledge_bases?.map(kb => kb.knowledge_base_id) ?? (run.knowledge_base_id ? [run.knowledge_base_id] : [])); }
