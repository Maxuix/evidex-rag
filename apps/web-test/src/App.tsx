import { useCallback, useEffect, useMemo, useState } from "react";

import { ChatView } from "./ChatView";
import { DocumentsView } from "./DocumentsView";
import { RetrievalView } from "./RetrievalView";
import { ApiClient, loadRuntimeConfig } from "./api/client";
import type {
  ChunkingPreset,
  GraphConfig,
  KnowledgeBase,
  KnowledgeBaseEmbeddingSelection,
  ModelSettingsSummary,
  ParsingPreset,
  RetrievalCapabilities,
} from "./api/types";
import {
  EmptyState,
  KnowledgeBaseSelector,
  ProblemNotice,
  shortId,
} from "./components";
import {
  readSelectedKnowledgeBaseId,
  storeSelectedKnowledgeBaseId,
} from "./storage";

type ViewName = "documents" | "chat" | "retrieval";
type PendingKnowledgeBaseCreate = {
  name: string;
  preset: ChunkingPreset;
  parsingPreset: ParsingPreset;
  embedding: KnowledgeBaseEmbeddingSelection;
  idempotencyKey: string;
};

export function App() {
  const [client, setClient] = useState<ApiClient | null>(null);
  const [configurationError, setConfigurationError] = useState<unknown | null>(null);
  const [retrievalCapabilities, setRetrievalCapabilities] =
    useState<RetrievalCapabilities | null>(null);
  const [retrievalCapabilitiesLoading, setRetrievalCapabilitiesLoading] =
    useState(false);
  const [retrievalCapabilitiesError, setRetrievalCapabilitiesError] =
    useState<unknown | null>(null);

  useEffect(() => {
    void loadRuntimeConfig().then((config) => {
      setClient(new ApiClient(config));
    }).catch(setConfigurationError);
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

  if (configurationError) {
    return (
      <main className="startup-shell">
        <ProblemNotice
          error={configurationError}
          title="Observation Studio could not start"
        />
      </main>
    );
  }
  if (!client) {
    return (
      <main className="startup-shell" aria-busy="true">
        <div className="startup-mark">RAG KB</div>
        <h1>Opening the local observation boundary…</h1>
      </main>
    );
  }
  return (
    <ObservationApp
      client={client}
      retrievalCapabilities={retrievalCapabilities}
      retrievalCapabilitiesLoading={retrievalCapabilitiesLoading}
      retrievalCapabilitiesError={retrievalCapabilitiesError}
    />
  );
}

export function ObservationApp({
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
  const [selectedId, setSelectedId] = useState<string | null>(
    () => readSelectedKnowledgeBaseId(),
  );
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [listError, setListError] = useState<unknown | null>(null);
  const [newKnowledgeBaseName, setNewKnowledgeBaseName] = useState("");
  const [newChunkingPreset, setNewChunkingPreset] = useState<ChunkingPreset>(
    "structural_balanced_v2",
  );
  const [newParsingPreset, setNewParsingPreset] = useState<ParsingPreset>(
    "text_local_v1",
  );
  const [modelSettings, setModelSettings] = useState<ModelSettingsSummary | null>(null);
  const [graphConfig, setGraphConfig] = useState<GraphConfig | null>(null);
  const [graphConfigLoading, setGraphConfigLoading] = useState(false);
  const [graphConfigError, setGraphConfigError] = useState<unknown | null>(null);
  const [newEmbeddingStrategy, setNewEmbeddingStrategy] = useState<
    "dual_space" | "unified_multimodal"
  >("dual_space");
  const [newTextProfileRevisionId, setNewTextProfileRevisionId] = useState("");
  const [newMultimodalProfileRevisionId, setNewMultimodalProfileRevisionId] = useState("");
  const [newUnifiedProfileRevisionId, setNewUnifiedProfileRevisionId] = useState("");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<unknown | null>(null);
  const [pendingCreate, setPendingCreate] = useState<PendingKnowledgeBaseCreate | null>(null);
  const [view, setView] = useState<ViewName>("documents");
  const [viewMutationPending, setViewMutationPending] = useState(false);
  const [focusedDocument, setFocusedDocument] = useState<{
    documentId: string;
    versionId: string | null;
  } | null>(null);

  const selectedKnowledgeBase = useMemo(
    () => knowledgeBases.find((item) => item.id === selectedId) ?? null,
    [knowledgeBases, selectedId],
  );
  const textProfiles = useMemo(
    () => (modelSettings?.profiles ?? []).filter((profile) =>
      profile.kind === "text_embedding"
      && profile.enabled
      && profile.validation_status === "valid"
    ),
    [modelSettings],
  );
  const multimodalProfiles = useMemo(
    () => (modelSettings?.profiles ?? []).filter((profile) =>
      profile.kind === "multimodal_embedding"
      && profile.enabled
      && profile.validation_status === "valid"
    ),
    [modelSettings],
  );
  const unifiedProfiles = useMemo(
    () => multimodalProfiles.filter((profile) => {
      const validation = profile.embedding_validation;
      return Boolean(
        validation?.shared_text_image_space_confirmed
        && validation.input_capabilities.includes("text_document")
        && validation.input_capabilities.includes("text_query")
        && validation.input_capabilities.includes("image")
        && validation.normalization === "client_l2_v1"
      );
    }),
    [multimodalProfiles],
  );
  const interactionLocked = pendingCreate !== null || viewMutationPending;
  const handleMutationPendingChange = useCallback((pending: boolean) => {
    setViewMutationPending(pending);
  }, []);

  const loadKnowledgeBases = useCallback(async (cursor?: string) => {
    setLoading(true);
    setListError(null);
    try {
      const page = await client.listKnowledgeBases(cursor);
      // A list response may have started before a confirmed create. Merge it
      // so that a late snapshot cannot erase the newly returned business fact.
      setKnowledgeBases((current) => mergeKnowledgeBases(current, page.items));
      setNextCursor(page.next_cursor);
    } catch (error) {
      setListError(error);
    } finally {
      setLoading(false);
    }
  }, [client]);

  useEffect(() => {
    void loadKnowledgeBases();
  }, [loadKnowledgeBases]);

  useEffect(() => {
    let cancelled = false;
    void client.getModelSettings().then((value) => {
      if (!cancelled) setModelSettings(value);
    }).catch(() => {
      if (!cancelled) setModelSettings(null);
    });
    return () => {
      cancelled = true;
    };
  }, [client]);

  useEffect(() => {
    if (!selectedId) {
      setGraphConfig(null);
      setGraphConfigError(null);
      return;
    }
    let cancelled = false;
    setGraphConfigLoading(true);
    setGraphConfigError(null);
    void client.getGraphConfig(selectedId).then((value) => {
      if (!cancelled) setGraphConfig(value);
    }).catch((error) => {
      if (!cancelled) {
        setGraphConfig(null);
        setGraphConfigError(error);
      }
    }).finally(() => {
      if (!cancelled) setGraphConfigLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [client, selectedId]);

  useEffect(() => {
    if (!modelSettings) return;
    setNewTextProfileRevisionId((current) =>
      current || modelSettings.selection.text_embedding_profile_revision_id || ""
    );
    setNewMultimodalProfileRevisionId((current) =>
      current || modelSettings.selection.multimodal_embedding_profile_revision_id || ""
    );
    setNewUnifiedProfileRevisionId((current) => {
      if (current && unifiedProfiles.some((profile) => profile.revision_id === current)) {
        return current;
      }
      const selected = modelSettings.selection.multimodal_embedding_profile_revision_id;
      return unifiedProfiles.find((profile) => profile.revision_id === selected)?.revision_id
        ?? unifiedProfiles[0]?.revision_id
        ?? "";
    });
  }, [modelSettings, unifiedProfiles]);

  useEffect(() => {
    if (loading) return;
    setSelectedId((selection) =>
      selection && knowledgeBases.some((item) => item.id === selection)
        ? selection
        : selection && nextCursor
          ? selection
        : knowledgeBases[0]?.id ?? null
    );
  }, [knowledgeBases, loading, nextCursor]);

  useEffect(() => {
    storeSelectedKnowledgeBaseId(selectedId);
  }, [selectedId]);

  const createKnowledgeBase = async (event: React.FormEvent) => {
    event.preventDefault();
    if (interactionLocked) return;
    const name = newKnowledgeBaseName.trim();
    if (!name) return;
    const embedding: KnowledgeBaseEmbeddingSelection =
      newParsingPreset === "text_local_v1"
        ? {
          strategy: "text_only",
          text_profile_revision_id: newTextProfileRevisionId || null,
        }
        : newEmbeddingStrategy === "unified_multimodal"
          ? {
            strategy: "unified_multimodal",
            profile_revision_id: newUnifiedProfileRevisionId || null,
          }
          : {
            strategy: "dual_space",
            text_profile_revision_id: newTextProfileRevisionId || null,
            multimodal_profile_revision_id: newMultimodalProfileRevisionId || null,
          };
    const pending = {
      name,
      preset: newChunkingPreset,
      parsingPreset: newParsingPreset,
      embedding,
      idempotencyKey: crypto.randomUUID(),
    };
    setPendingCreate(pending);
    await performCreate(pending);
  };

  const performCreate = async (pending: PendingKnowledgeBaseCreate) => {
    setCreating(true);
    setCreateError(null);
    try {
      const value = await client.createKnowledgeBase(
        pending.name,
        pending.preset,
        pending.parsingPreset,
        pending.embedding,
        pending.idempotencyKey,
      );
      setKnowledgeBases((current) => mergeKnowledgeBases(current, [value]));
      setSelectedId(value.id);
      setNewKnowledgeBaseName("");
      setPendingCreate(null);
    } catch (error) {
      setCreateError(error);
    } finally {
      setCreating(false);
    }
  };

  const openDocument = (documentId: string, versionId?: string) => {
    if (interactionLocked) return;
    setFocusedDocument({ documentId, versionId: versionId ?? null });
    setView("documents");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true">R</div>
          <div>
            <p className="eyebrow">Enterprise Knowledge Base</p>
            <h1>Observation Studio</h1>
          </div>
        </div>
        <div className="runtime-boundary">
          <span className="runtime-dot" aria-hidden="true" />
          <div>
            <strong>Local development only</strong>
            <span>{client.apiOrigin}</span>
          </div>
        </div>
      </header>

      <section className="control-deck" aria-label="Knowledge base controls">
        <KnowledgeBaseSelector
          knowledgeBases={knowledgeBases}
          selectedId={selectedId}
          loading={loading}
          disabled={interactionLocked}
          nextCursor={nextCursor}
          onSelect={(value) => {
            setSelectedId(value);
            setFocusedDocument(null);
          }}
          onLoadMore={() => nextCursor && void loadKnowledgeBases(nextCursor)}
        />
        <form className="create-kb-form" onSubmit={createKnowledgeBase}>
          <label htmlFor="new-kb">Create a local knowledge base</label>
          <div className="inline-controls">
            <input
              id="new-kb"
              value={newKnowledgeBaseName}
              onChange={(event) => setNewKnowledgeBaseName(event.target.value)}
              maxLength={255}
              placeholder="Product handbook"
              required
              disabled={interactionLocked}
            />
            <select
              aria-label="Parsing preset"
              value={newParsingPreset}
              onChange={(event) =>
                setNewParsingPreset(event.target.value as ParsingPreset)
              }
              disabled={interactionLocked}
            >
              <option value="text_local_v1">Text-only local</option>
              <option value="multimodal_local_v2">
                Multimodal local (Markdown media)
              </option>
            </select>
            <select
              aria-label="Chunking preset"
              value={newChunkingPreset}
              onChange={(event) =>
                setNewChunkingPreset(event.target.value as ChunkingPreset)
              }
              disabled={interactionLocked}
            >
              <option value="structural_balanced_v2">Structural balanced</option>
              <option value="semantic_balanced_v1">Semantic balanced</option>
            </select>
            <button
              className="button secondary"
              type="submit"
              disabled={
                creating
                || interactionLocked
                || !newKnowledgeBaseName.trim()
                || (
                  newParsingPreset === "multimodal_local_v2"
                  && newEmbeddingStrategy === "unified_multimodal"
                  && !newUnifiedProfileRevisionId
                )
              }
            >
              {creating ? "Creating…" : "Create"}
            </button>
          </div>
          <div className="inline-controls" aria-label="Embedding strategy">
            {newParsingPreset === "text_local_v1" ? (
              <>
                <span>Text-only embedding</span>
                <select
                  aria-label="Text embedding profile"
                  value={newTextProfileRevisionId}
                  onChange={(event) => setNewTextProfileRevisionId(event.target.value)}
                  disabled={interactionLocked}
                >
                  <option value="">Workspace default text model</option>
                  {textProfiles.map((profile) => (
                    <option key={profile.revision_id} value={profile.revision_id}>
                      {profile.name} · {profile.embedding_validation?.selected_dimension ?? "?"}d · r{profile.revision}
                    </option>
                  ))}
                </select>
              </>
            ) : (
              <>
                <select
                  aria-label="Embedding strategy"
                  value={newEmbeddingStrategy}
                  onChange={(event) => setNewEmbeddingStrategy(
                    event.target.value as "dual_space" | "unified_multimodal"
                  )}
                  disabled={interactionLocked}
                >
                  <option value="dual_space">Dedicated dual models (default)</option>
                  <option value="unified_multimodal" disabled={!unifiedProfiles.length}>
                    Unified multimodal model
                  </option>
                </select>
                {newEmbeddingStrategy === "dual_space" ? (
                  <>
                    <select
                      aria-label="Text embedding profile"
                      value={newTextProfileRevisionId}
                      onChange={(event) => setNewTextProfileRevisionId(event.target.value)}
                      disabled={interactionLocked}
                    >
                      <option value="">Workspace default text model</option>
                      {textProfiles.map((profile) => (
                        <option key={profile.revision_id} value={profile.revision_id}>
                          Text · {profile.name} · {profile.embedding_validation?.selected_dimension ?? "?"}d
                        </option>
                      ))}
                    </select>
                    <select
                      aria-label="Multimodal embedding profile"
                      value={newMultimodalProfileRevisionId}
                      onChange={(event) => setNewMultimodalProfileRevisionId(event.target.value)}
                      disabled={interactionLocked}
                    >
                      <option value="">Workspace default multimodal model</option>
                      {multimodalProfiles.map((profile) => (
                        <option key={profile.revision_id} value={profile.revision_id}>
                          Visual · {profile.name} · {profile.embedding_validation?.selected_dimension ?? "?"}d
                        </option>
                      ))}
                    </select>
                  </>
                ) : (
                  <select
                    aria-label="Unified multimodal embedding profile"
                    required
                    value={newUnifiedProfileRevisionId}
                    onChange={(event) => setNewUnifiedProfileRevisionId(event.target.value)}
                    disabled={interactionLocked || !unifiedProfiles.length}
                  >
                    <option value="">Select an eligible unified model</option>
                    {unifiedProfiles.map((profile) => (
                      <option key={profile.revision_id} value={profile.revision_id}>
                        {profile.name} · {profile.embedding_validation?.selected_dimension ?? "?"}d · r{profile.revision}
                      </option>
                    ))}
                  </select>
                )}
              </>
            )}
          </div>
          <small className="field-help">
            {newParsingPreset === "multimodal_local_v2"
              ? "Snapshots Markdown images, including folder resources and public remote URLs; requires the configured multimodal provider. "
              : "Uses the existing text-only parser. "}
            {newChunkingPreset === "structural_balanced_v2"
              ? "Uses titles and token windows."
              : "Uses additional embeddings to find semantic breakpoints."}
            {newParsingPreset === "multimodal_local_v2" && !unifiedProfiles.length
              ? " Unified is unavailable until a validated multimodal profile confirms one shared text/image semantic space."
              : ""}
          </small>
        </form>
      </section>

      {listError ? (
        <div className="page-notice">
          <ProblemNotice error={listError} onRetry={() => void loadKnowledgeBases()} />
        </div>
      ) : null}
      {createError ? (
        <div className="page-notice">
          <ProblemNotice
            error={createError}
            title="Knowledge-base creation was not confirmed"
            onRetry={pendingCreate ? () => void performCreate(pendingCreate) : undefined}
            onDiscard={pendingCreate ? () => {
              setPendingCreate(null);
              setCreateError(null);
            } : undefined}
            discardLabel="Discard and edit"
          />
        </div>
      ) : null}

      {retrievalCapabilitiesLoading || retrievalCapabilitiesError ? (
        <div className="page-notice" role="status">
          {retrievalCapabilitiesLoading
            ? "Retrieval capability status is loading; hybrid remains disabled."
            : "Retrieval capability status is unavailable; exact vector remains available and hybrid is disabled."}
        </div>
      ) : null}

      <nav className="view-tabs" aria-label="Observation views">
        {([
          ["documents", "Documents", "Upload and index"],
          ["chat", "Chat", "Run and cite"],
          ["retrieval", "Retrieval Debug", "Inspect evidence"],
        ] as Array<[ViewName, string, string]>).map(([name, label, caption], index) => (
          <button
            type="button"
            key={name}
            className={view === name ? "active" : ""}
            aria-current={view === name ? "page" : undefined}
            onClick={() => setView(name)}
            disabled={interactionLocked}
          >
            <span className="tab-number">0{index + 1}</span>
            <span><strong>{label}</strong><small>{caption}</small></span>
          </button>
        ))}
      </nav>

      <main className="main-content">
        {loading && knowledgeBases.length === 0 ? (
          <section className="panel"><p className="loading-line">Loading knowledge bases…</p></section>
        ) : !selectedKnowledgeBase ? (
          <section className="panel">
            <EmptyState
              title="Create the first knowledge base"
              description="The public API will provision its active revision and selected embedding role bindings."
            />
          </section>
        ) : (
          <>
            <section className="context-strip">
              <div>
                <span>Selected knowledge base</span>
                <strong>{selectedKnowledgeBase.name}</strong>
              </div>
              <div>
                <span>Active revision</span>
                <strong>{shortId(selectedKnowledgeBase.active_index_revision_id)}</strong>
              </div>
              <div>
                <span>Source sequence</span>
                <strong>{selectedKnowledgeBase.source_change_seq}</strong>
              </div>
              <div>
                <span>Parsing</span>
                <strong>
                  {selectedKnowledgeBase.parsing.preset === "multimodal_local_v2"
                    ? "Multimodal local"
                      : "Text-only local"}
                </strong>
              </div>
              <div>
                <span>Chunking preset</span>
                <strong>
                  {selectedKnowledgeBase.chunking.preset === "semantic_balanced_v1"
                    ? "Semantic balanced"
                    : selectedKnowledgeBase.chunking.preset === "structural_balanced_v2"
                      ? "Structural balanced"
                      : "Legacy (read-only)"}
                </strong>
              </div>
              <div>
                <span>Embedding strategy</span>
                <strong>{selectedKnowledgeBase.embedding.strategy.replaceAll("_", " ")}</strong>
              </div>
              <div>
                <span>Embedding dimensions</span>
                <strong>
                  Text {selectedKnowledgeBase.embedding.text.dimension}d
                  {selectedKnowledgeBase.embedding.cross_modal
                    ? ` · Visual ${selectedKnowledgeBase.embedding.cross_modal.dimension}d`
                    : ""}
                </strong>
              </div>
            </section>
            {view === "documents" ? (
              <DocumentsView
                key={selectedKnowledgeBase.id}
                client={client}
                knowledgeBase={selectedKnowledgeBase}
                focusedDocumentId={focusedDocument?.documentId ?? null}
                focusedDocumentVersionId={focusedDocument?.versionId ?? null}
                onMutationPendingChange={handleMutationPendingChange}
              />
            ) : null}
            {view === "chat" ? (
              <ChatView
                key={selectedKnowledgeBase.id}
                client={client}
                knowledgeBase={selectedKnowledgeBase}
                retrievalCapabilities={retrievalCapabilities}
                retrievalCapabilitiesLoading={retrievalCapabilitiesLoading}
                retrievalCapabilitiesError={retrievalCapabilitiesError}
                onOpenCitationDocument={openDocument}
                onMutationPendingChange={handleMutationPendingChange}
              />
            ) : null}
            {view === "retrieval" ? (
              <RetrievalView
                key={selectedKnowledgeBase.id}
                client={client}
                knowledgeBase={selectedKnowledgeBase}
                retrievalCapabilities={retrievalCapabilities}
                retrievalCapabilitiesLoading={retrievalCapabilitiesLoading}
                retrievalCapabilitiesError={retrievalCapabilitiesError}
                graphConfig={graphConfig}
                graphConfigLoading={graphConfigLoading}
                graphConfigError={graphConfigError}
                onOpenDocument={openDocument}
              />
            ) : null}
          </>
        )}
      </main>

      <footer className="site-footer">
        <p>
          Fixed development identity · Public <code>/api/v1</code> only · No enterprise
          authentication, backup, high availability, or multi-tenant guarantee
        </p>
      </footer>
    </div>
  );
}

function mergeKnowledgeBases(
  current: KnowledgeBase[],
  incoming: KnowledgeBase[],
): KnowledgeBase[] {
  const byId = new Map(current.map((item) => [item.id, item]));
  for (const item of incoming) byId.set(item.id, item);
  return [...byId.values()].sort((left, right) => left.name.localeCompare(right.name));
}
