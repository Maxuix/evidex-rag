import { useCallback, useEffect, useMemo, useState } from "react";

import { ChatView } from "./ChatView";
import { DocumentsView } from "./DocumentsView";
import { RetrievalView } from "./RetrievalView";
import { ApiClient, loadRuntimeConfig } from "./api/client";
import type {
  ChunkingPreset,
  KnowledgeBase,
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
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<unknown | null>(null);
  const [pendingCreate, setPendingCreate] = useState<{
    name: string;
    preset: ChunkingPreset;
    parsingPreset: ParsingPreset;
    idempotencyKey: string;
  } | null>(null);
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
    const pending = {
      name,
      preset: newChunkingPreset,
      parsingPreset: newParsingPreset,
      idempotencyKey: crypto.randomUUID(),
    };
    setPendingCreate(pending);
    await performCreate(pending);
  };

  const performCreate = async (pending: {
    name: string;
    preset: ChunkingPreset;
    parsingPreset: ParsingPreset;
    idempotencyKey: string;
  }) => {
    setCreating(true);
    setCreateError(null);
    try {
      const value = await client.createKnowledgeBase(
        pending.name,
        pending.preset,
        pending.parsingPreset,
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
              disabled={creating || interactionLocked || !newKnowledgeBaseName.trim()}
            >
              {creating ? "Creating…" : "Create"}
            </button>
          </div>
          <small className="field-help">
            {newParsingPreset === "multimodal_local_v2"
              ? "Snapshots Markdown images, including folder resources and public remote URLs; requires the configured multimodal provider. "
              : "Uses the existing text-only parser. "}
            {newChunkingPreset === "structural_balanced_v2"
              ? "Uses titles and token windows."
              : "Uses additional embeddings to find semantic breakpoints."}
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
              description="The public API will provision its active revision and fixed embedding space."
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
