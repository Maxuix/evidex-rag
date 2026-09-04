import {
  type ChangeEvent,
  type FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { ApiClient, ApiClientError } from "./api/client";
import { UI_POLICY } from "./uiPolicy";
import {
  isDocumentScopeCurrent,
  isManagementKbScopeCurrent,
  isRequestSequenceCurrent,
  type DocumentScope,
  type ManagementKbScope,
} from "./requestScope";
import type {
  ChunkingPreset,
  DocumentChunk,
  DocumentChunkInspection,
  DocumentDetail,
  DocumentRecord,
  GraphConfig,
  GraphConfigUpdate,
  GraphSchemaProfile,
  IndexingJob,
  KnowledgeBase,
  KnowledgeBaseEmbeddingSelection,
  ModelProfile,
  ModelSettings,
  ParsingPreset,
  RerankMode,
  RetrievalEvidencePack,
} from "./api/types";

const ACCEPTED_EXTENSIONS = ".txt,.md,.mdz,.html,.csv,.pdf,.docx,.pptx,.xlsx";
const TERMINAL_JOB_STATUSES = new Set(["completed", "failed", "cancelled"]);
const LONG_RUNNING_JOB_SECONDS = 10 * 60;
const INDEX_PHASES = [
  "queued",
  "claimed",
  "source_read",
  "parsing",
  "asset_extraction",
  "enrichment",
  "semantic_analysis",
  "embedding",
  "multimodal_embedding",
  "auto_qa_generation",
  "persisting",
  "validating",
  "completed",
] as const;

interface UploadItem {
  id: string;
  file: File;
  status: "waiting" | "uploading" | "accepted" | "failed";
  jobId: string | null;
  documentId: string | null;
  error: string | null;
}

interface Confirmation {
  kind: "knowledge-base" | "document" | "chunk";
  id: string;
  name: string;
  documentId?: string;
}

export function KnowledgeBaseManagementPage({
  client,
  knowledgeBases,
  selectedKnowledgeBaseId,
  modelSettings,
  graphConfig,
  graphSchemaProfiles,
  graphSchemaProfilesError,
  graphConfigLoading,
  graphConfigError,
  onRefreshGraphConfig,
  onUpdateGraphConfig,
  onKnowledgeBaseCreated,
  onKnowledgeBaseDeleted,
  onOpenModelSettings,
  onOpenMobileSidebar,
}: {
  client: ApiClient;
  knowledgeBases: KnowledgeBase[];
  selectedKnowledgeBaseId: string;
  modelSettings: ModelSettings | null;
  graphConfig: GraphConfig | null;
  graphSchemaProfiles: GraphSchemaProfile[];
  graphSchemaProfilesError: string | null;
  graphConfigLoading: boolean;
  graphConfigError: string | null;
  onRefreshGraphConfig: () => Promise<GraphConfig | null>;
  onUpdateGraphConfig: (payload: GraphConfigUpdate) => Promise<GraphConfig>;
  onKnowledgeBaseCreated: (value: KnowledgeBase) => void;
  onKnowledgeBaseDeleted: (id: string) => void;
  onOpenModelSettings: () => void;
  onOpenMobileSidebar: () => void;
}) {
  const knowledgeBase = knowledgeBases.find(
    (item) => item.id === selectedKnowledgeBaseId,
  ) ?? null;
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [documentsError, setDocumentsError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<IndexingJob[]>([]);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);
  const [documentDetail, setDocumentDetail] = useState<DocumentDetail | null>(null);
  const [chunks, setChunks] = useState<DocumentChunkInspection | null>(null);
  const [chunksLoading, setChunksLoading] = useState(false);
  const [chunksError, setChunksError] = useState<string | null>(null);
  const [uploadItems, setUploadItems] = useState<UploadItem[]>([]);
  const [uploading, setUploading] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [confirmation, setConfirmation] = useState<Confirmation | null>(null);
  const [confirmName, setConfirmName] = useState("");
  const [confirming, setConfirming] = useState(false);
  const uploadInput = useRef<HTMLInputElement>(null);
  const kbGeneration = useRef(0);
  const documentsSequence = useRef(0);
  const jobsSequence = useRef(0);
  const chunksSequence = useRef(0);
  const knowledgeBaseIdRef = useRef(selectedKnowledgeBaseId);
  const selectedDocumentIdRef = useRef(selectedDocumentId);
  knowledgeBaseIdRef.current = selectedKnowledgeBaseId;
  selectedDocumentIdRef.current = selectedDocumentId;

  const loadDocuments = useCallback(async (kbId: string) => {
    const token: ManagementKbScope = {
      generation: kbGeneration.current,
      knowledgeBaseId: kbId,
    };
    const sequence = ++documentsSequence.current;
    const isCurrent = () => isManagementKbScopeCurrent(token, {
      generation: kbGeneration.current,
      knowledgeBaseId: knowledgeBaseIdRef.current,
    }) && isRequestSequenceCurrent(sequence, documentsSequence.current);
    setDocumentsLoading(true);
    setDocumentsError(null);
    try {
      const page = await client.listDocuments(kbId);
      if (!isCurrent()) return;
      setDocuments(page.items);
      setSelectedDocumentId((current) => (
        current && page.items.some((item) => item.id === current)
          ? current
          : null
      ));
    } catch (error) {
      if (isCurrent()) setDocumentsError(managementError(error));
    } finally {
      if (isCurrent()) setDocumentsLoading(false);
    }
  }, [client]);

  const loadJobs = useCallback(async (kbId: string, quiet = false) => {
    const token: ManagementKbScope = {
      generation: kbGeneration.current,
      knowledgeBaseId: kbId,
    };
    const sequence = ++jobsSequence.current;
    const isCurrent = () => isManagementKbScopeCurrent(token, {
      generation: kbGeneration.current,
      knowledgeBaseId: knowledgeBaseIdRef.current,
    }) && isRequestSequenceCurrent(sequence, jobsSequence.current);
    if (!quiet) setJobsError(null);
    try {
      const page = await client.listIndexingJobs(kbId);
      if (!isCurrent()) return;
      setJobs(page.items);
    } catch (error) {
      if (!quiet && isCurrent()) setJobsError(managementError(error));
    }
  }, [client]);

  useEffect(() => {
    ++kbGeneration.current;
    documentsSequence.current += 1;
    jobsSequence.current += 1;
    chunksSequence.current += 1;
    setDocuments([]);
    setDocumentsLoading(false);
    setJobs([]);
    setUploadItems([]);
    setUploading(false);
    setSelectedDocumentId(null);
    setDocumentDetail(null);
    setChunks(null);
    setChunksLoading(false);
    setActionError(null);
    setConfirmation(null);
    setConfirming(false);
    if (!selectedKnowledgeBaseId) return;
    void loadDocuments(selectedKnowledgeBaseId);
    void loadJobs(selectedKnowledgeBaseId);
  }, [loadDocuments, loadJobs, selectedKnowledgeBaseId]);

  const hasActiveJobs = jobs.some((job) => !jobSettled(job));
  useEffect(() => {
    if (!selectedKnowledgeBaseId || !hasActiveJobs) return;
    const timer = window.setInterval(() => {
      void loadJobs(selectedKnowledgeBaseId, true);
    }, UI_POLICY.indexingPollMs);
    return () => window.clearInterval(timer);
  }, [hasActiveJobs, loadJobs, selectedKnowledgeBaseId]);

  const loadChunks = useCallback(async (documentId: string, cursor?: string) => {
    const token: DocumentScope = {
      generation: kbGeneration.current,
      knowledgeBaseId: knowledgeBaseIdRef.current,
      documentId,
    };
    const sequence = ++chunksSequence.current;
    const isCurrent = () => isDocumentScopeCurrent(token, {
      generation: kbGeneration.current,
      knowledgeBaseId: knowledgeBaseIdRef.current,
      documentId: selectedDocumentIdRef.current ?? "",
    }) && isRequestSequenceCurrent(sequence, chunksSequence.current);
    setChunksLoading(true);
    setChunksError(null);
    try {
      const [detail, inspection] = await Promise.all([
        cursor ? Promise.resolve(null) : client.getDocument(documentId),
        client.getDocumentChunks(documentId, cursor),
      ]);
      if (!isCurrent()) return;
      if (cursor && chunks?.next_cursor !== cursor) return;
      if (detail) setDocumentDetail(detail);
      setChunks((current) => cursor && current
        ? {
          ...inspection,
          items: mergeChunks(current.items, inspection.items),
        }
        : inspection);
    } catch (error) {
      if (isCurrent()) {
        setChunksError(managementError(error));
        if (!cursor) setChunks(null);
      }
    } finally {
      if (isCurrent()) setChunksLoading(false);
    }
  }, [client]);

  const inspectDocument = (documentId: string, job: IndexingJob | null) => {
    if (!chunkPreviewReady(job)) return;
    selectedDocumentIdRef.current = documentId;
    setSelectedDocumentId(documentId);
    setDocumentDetail(null);
    setChunks(null);
    void loadChunks(documentId);
  };

  const uploadFiles = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (!knowledgeBase || files.length === 0 || uploading) return;
    const target: ManagementKbScope = {
      generation: kbGeneration.current,
      knowledgeBaseId: knowledgeBase.id,
    };
    const isCurrent = () => isManagementKbScopeCurrent(target, {
      generation: kbGeneration.current,
      knowledgeBaseId: knowledgeBaseIdRef.current,
    });
    const items: UploadItem[] = files.map((file) => ({
      id: crypto.randomUUID(),
      file,
      status: "waiting",
      jobId: null,
      documentId: null,
      error: null,
    }));
    setUploadItems(items);
    setUploading(true);
    setActionError(null);
    const uploadOrder = [...items].sort(
      (left, right) => left.file.size - right.file.size,
    );
    for (const item of uploadOrder) {
      if (!isCurrent()) return;
      setUploadItems((current) => updateUpload(current, item.id, {
        status: "uploading",
      }));
      try {
        const accepted = await client.uploadDocument(
          knowledgeBase.id,
          item.file,
          crypto.randomUUID(),
        );
        if (!isCurrent()) return;
        setUploadItems((current) => updateUpload(current, item.id, {
          status: "accepted",
          jobId: accepted.job_id,
          documentId: accepted.document.id,
        }));
      } catch (error) {
        if (!isCurrent()) return;
        setUploadItems((current) => updateUpload(current, item.id, {
          status: "failed",
          error: managementError(error),
        }));
      }
    }
    if (!isCurrent()) return;
    setUploading(false);
    await Promise.all([
      loadDocuments(knowledgeBase.id),
      loadJobs(knowledgeBase.id),
    ]);
  };

  const updateDocument = async (
    document: DocumentRecord,
    event: ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || uploading) return;
    const target: DocumentScope = {
      generation: kbGeneration.current,
      knowledgeBaseId: knowledgeBaseIdRef.current,
      documentId: document.id,
    };
    const isCurrent = () => isDocumentScopeCurrent(target, {
      generation: kbGeneration.current,
      knowledgeBaseId: knowledgeBaseIdRef.current,
      documentId: selectedDocumentIdRef.current ?? document.id,
    });
    setUploading(true);
    setActionError(null);
    try {
      await client.uploadDocumentVersion(
        document.id,
        file,
        document.display_name,
        crypto.randomUUID(),
      );
      if (!isCurrent()) return;
      if (selectedDocumentId === document.id) {
        setSelectedDocumentId(null);
        setDocumentDetail(null);
        setChunks(null);
      }
      if (knowledgeBase && isCurrent()) {
        await Promise.all([
          loadDocuments(knowledgeBase.id),
          loadJobs(knowledgeBase.id),
        ]);
      }
    } catch (error) {
      if (isCurrent()) setActionError(`更新“${document.display_name}”失败：${managementError(error)}`);
    } finally {
      if (isCurrent()) setUploading(false);
    }
  };

  const runConfirmation = async () => {
    if (!confirmation) return;
    const targetKbId = knowledgeBaseIdRef.current;
    const targetGeneration = kbGeneration.current;
    const isCurrent = () => isManagementKbScopeCurrent(
      { generation: targetGeneration, knowledgeBaseId: targetKbId },
      { generation: kbGeneration.current, knowledgeBaseId: knowledgeBaseIdRef.current },
    );
    setConfirming(true);
    setActionError(null);
    try {
      if (confirmation.kind === "knowledge-base") {
        await client.deleteKnowledgeBase(confirmation.id, crypto.randomUUID());
        if (!isCurrent()) return;
        onKnowledgeBaseDeleted(confirmation.id);
      } else if (confirmation.kind === "document") {
        await client.deleteDocument(confirmation.id, crypto.randomUUID());
        if (!isCurrent()) return;
        if (knowledgeBase) {
          await Promise.all([
            loadDocuments(knowledgeBase.id),
            loadJobs(knowledgeBase.id),
          ]);
        }
        if (selectedDocumentId === confirmation.id) {
          setSelectedDocumentId(null);
          setDocumentDetail(null);
          setChunks(null);
        }
      } else {
        await client.deleteDocumentChunk(
          confirmation.documentId!,
          confirmation.id,
        );
        if (isCurrent()) await loadChunks(confirmation.documentId!);
      }
      if (!isCurrent()) return;
      setConfirmation(null);
      setConfirmName("");
    } catch (error) {
      if (isCurrent()) setActionError(managementError(error));
    } finally {
      if (isCurrent()) setConfirming(false);
    }
  };

  const retryJob = async (job: IndexingJob) => {
    const target: ManagementKbScope = {
      generation: kbGeneration.current,
      knowledgeBaseId: knowledgeBaseIdRef.current,
    };
    const isCurrent = () => isManagementKbScopeCurrent(target, {
      generation: kbGeneration.current,
      knowledgeBaseId: knowledgeBaseIdRef.current,
    });
    setActionError(null);
    try {
      await client.retryIndexingJob(job.job_id, crypto.randomUUID());
      if (knowledgeBase && isCurrent()) await loadJobs(knowledgeBase.id);
    } catch (error) {
      if (isCurrent()) setActionError(`重试失败：${managementError(error)}`);
    }
  };

  return (
    <main className="management-main">
      <header className="chat-header management-header">
        <button
          className="icon-button mobile-menu"
          type="button"
          aria-label="打开侧栏"
          onClick={onOpenMobileSidebar}
        >
          ☰
        </button>
        <div className="chat-heading management-heading">
          <h1>知识库管理</h1>
          <p>{knowledgeBase ? knowledgeBase.name : "创建并导入你的本地资料"}</p>
        </div>
        <button className="quiet-button" type="button" onClick={onOpenModelSettings}>
          模型设置
        </button>
      </header>

      <div className="management-scroll">
        <div className="management-content">
          <KnowledgeBaseCreator
            modelSettings={modelSettings}
            onCreate={async (value) => {
              const created = await client.createKnowledgeBase(
                value.name,
                value.parsing,
                value.chunking,
                value.embedding,
                crypto.randomUUID(),
                value.autoQa,
              );
              onKnowledgeBaseCreated(created);
            }}
            onOpenModelSettings={onOpenModelSettings}
          />

          {knowledgeBase ? (
            <>
              <section className="management-overview">
                <div>
                  <span className="eyebrow">当前知识库</span>
                  <h2>{knowledgeBase.name}</h2>
                  <p>
                    {parsingLabel(knowledgeBase.parsing.preset)} · {chunkingLabel(knowledgeBase.chunking.preset)} · {embeddingLabel(knowledgeBase.embedding.strategy)}
                  </p>
                  <p>{autoQaStatus(knowledgeBase)}</p>
                </div>
                <button
                  className="danger-button"
                  type="button"
                  onClick={() => {
                    setConfirmName("");
                    setConfirmation({
                      kind: "knowledge-base",
                      id: knowledgeBase.id,
                      name: knowledgeBase.name,
                    });
                  }}
                >
                  删除知识库
                </button>
              </section>

              {actionError ? <InlineError message={actionError} /> : null}

              <GraphSettingsPanel
                knowledgeBase={knowledgeBase}
                modelSettings={modelSettings}
                config={graphConfig}
                schemaProfiles={graphSchemaProfiles}
                schemaProfilesError={graphSchemaProfilesError}
                loading={graphConfigLoading}
                error={graphConfigError}
                onRefresh={onRefreshGraphConfig}
                onUpdate={onUpdateGraphConfig}
                onOpenModelSettings={onOpenModelSettings}
              />

              <section className="management-panel upload-panel-chat">
                <div className="management-panel-heading">
                  <div>
                    <span className="eyebrow">导入与索引</span>
                    <h2>添加文档</h2>
                    <p>支持批量选择；上传后会自动解析、切分并建立索引。</p>
                  </div>
                  <button
                    className="primary-button"
                    type="button"
                    disabled={uploading}
                    onClick={() => uploadInput.current?.click()}
                  >
                    {uploading ? "正在上传…" : "选择文档"}
                  </button>
                  <input
                    ref={uploadInput}
                    className="visually-hidden"
                    type="file"
                    accept={ACCEPTED_EXTENSIONS}
                    multiple
                    onChange={(event) => void uploadFiles(event)}
                  />
                </div>
                <p className="file-support-note">
                  TXT、Markdown、HTML、CSV、PDF、DOCX、PPTX、XLSX、MDZ
                </p>
                {uploadItems.length ? (
                  <UploadProgress items={uploadItems} jobs={jobs} />
                ) : (
                  <button
                    className="upload-dropzone"
                    type="button"
                    onClick={() => uploadInput.current?.click()}
                  >
                    <strong>批量导入文档</strong>
                    <span>点击选择本地文件；每个文件单独报告成功或失败原因</span>
                  </button>
                )}
              </section>

              <section className="management-panel">
                <div className="management-panel-heading">
                  <div>
                    <span className="eyebrow">文档</span>
                    <h2>已导入文档</h2>
                    <p>{documents.length} 个文档，进度来自服务端索引任务。</p>
                  </div>
                  <button
                    className="quiet-button"
                    type="button"
                    disabled={documentsLoading}
                    onClick={() => {
                      void loadDocuments(knowledgeBase.id);
                      void loadJobs(knowledgeBase.id);
                    }}
                  >
                    刷新
                  </button>
                </div>
                {documentsError ? <InlineError message={documentsError} /> : null}
                {jobsError ? <InlineError message={jobsError} /> : null}
                {documentsLoading && !documents.length ? (
                  <EmptyState text="正在读取文档…" />
                ) : !documents.length ? (
                  <EmptyState text="还没有文档，从上方选择文件开始导入。" />
                ) : (
                  <div className="document-list">
                    {documents.map((document) => {
                      const job = newestJob(jobs, document.id);
                      const previewReady = chunkPreviewReady(job);
                      return (
                        <DocumentRow
                          key={document.id}
                          document={document}
                          job={job}
                          selected={selectedDocumentId === document.id}
                          disabled={uploading}
                          previewReady={previewReady}
                          onInspect={() => inspectDocument(document.id, job)}
                          onUpdate={(event) => void updateDocument(document, event)}
                          onDelete={() => setConfirmation({
                            kind: "document",
                            id: document.id,
                            name: document.display_name,
                          })}
                          onRetry={() => job && void retryJob(job)}
                        />
                      );
                    })}
                  </div>
                )}
              </section>

              {selectedDocumentId ? (
                <ChunkPreview
                  client={client}
                  detail={documentDetail}
                  inspection={chunks}
                  loading={chunksLoading}
                  error={chunksError}
                  onClose={() => {
                    setSelectedDocumentId(null);
                    setDocumentDetail(null);
                    setChunks(null);
                  }}
                  onLoadMore={() => chunks?.next_cursor
                    && void loadChunks(selectedDocumentId, chunks.next_cursor)}
                  onDelete={(chunk) => setConfirmation({
                    kind: "chunk",
                    id: chunk.id,
                    name: `Chunk ${chunk.ordinal + 1}`,
                    documentId: selectedDocumentId,
                  })}
                />
              ) : null}

              <RetrievalDebugger
                client={client}
                knowledgeBase={knowledgeBase}
              />
            </>
          ) : (
            <section className="management-panel empty-management">
              <span className="welcome-mark">K</span>
              <h2>先创建一个知识库</h2>
              <p>创建时确定解析、切分和 embedding 配置，之后即可批量导入文档。</p>
            </section>
          )}
        </div>
      </div>

      {confirmation ? (
        <ConfirmationDialog
          confirmation={confirmation}
          value={confirmName}
          busy={confirming}
          onChange={setConfirmName}
          onCancel={() => {
            setConfirmation(null);
            setConfirmName("");
          }}
          onConfirm={() => void runConfirmation()}
        />
      ) : null}
    </main>
  );
}

type GraphAction = "configure" | "retry" | "force-rebuild" | "disable" | "refresh";

function GraphSettingsPanel({
  knowledgeBase,
  modelSettings,
  config,
  schemaProfiles,
  schemaProfilesError,
  loading,
  error,
  onRefresh,
  onUpdate,
  onOpenModelSettings,
}: {
  knowledgeBase: KnowledgeBase;
  modelSettings: ModelSettings | null;
  config: GraphConfig | null;
  schemaProfiles: GraphSchemaProfile[];
  schemaProfilesError: string | null;
  loading: boolean;
  error: string | null;
  onRefresh: () => Promise<GraphConfig | null>;
  onUpdate: (payload: GraphConfigUpdate) => Promise<GraphConfig>;
  onOpenModelSettings: () => void;
}) {
  const currentConfig = config?.knowledge_base_id === knowledgeBase.id ? config : null;
  const chatProfiles = useMemo(
    () => validProfiles(modelSettings, "chat"),
    [modelSettings],
  );
  const [profileRevisionId, setProfileRevisionId] = useState("");
  const [schemaProfileKey, setSchemaProfileKey] = useState("");
  const [busyAction, setBusyAction] = useState<GraphAction | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  useEffect(() => {
    const configured = currentConfig?.chat_profile_revision_id;
    const preferred = modelSettings?.selection.chat_profile_revision_id;
    setProfileRevisionId(
      chatProfiles.some((profile) => profile.revision_id === configured)
        ? configured!
        : chatProfiles.some((profile) => profile.revision_id === preferred)
          ? preferred!
          : chatProfiles[0]?.revision_id ?? "",
    );
    setActionError(null);
  }, [
    chatProfiles,
    currentConfig?.chat_profile_revision_id,
    knowledgeBase.id,
    modelSettings?.selection.chat_profile_revision_id,
  ]);

  useEffect(() => {
    const configured = currentConfig?.schema_profile_key;
    const preferred = schemaProfiles.find((profile) => profile.is_default)?.key
      ?? "generic_open_domain_v1";
    setSchemaProfileKey(
      schemaProfiles.some((profile) => profile.key === configured)
        ? configured!
        : preferred,
    );
  }, [currentConfig?.schema_profile_key, knowledgeBase.id, schemaProfiles]);

  const runAction = async (action: GraphAction) => {
    if (busyAction) return;
    if (action === "configure" && !profileRevisionId) return;
    setBusyAction(action);
    setActionError(null);
    try {
      if (action === "refresh") {
        await onRefresh();
      } else if (action === "configure") {
        await onUpdate({
          enabled: true,
          chat_profile_revision_id: profileRevisionId,
          schema_profile_key: schemaProfileKey,
        });
      } else if (action === "retry") {
        await onUpdate({ enabled: true, retry: true });
      } else if (action === "force-rebuild") {
        await onUpdate({ enabled: true, retry: true, force_rebuild: true });
      } else {
        await onUpdate({ enabled: false });
      }
    } catch (caught) {
      setActionError(managementError(caught));
    } finally {
      setBusyAction(null);
    }
  };

  const status = currentConfig?.status ?? null;
  const statusLabel = status
    ? graphStatusLabel(status, currentConfig?.requires_rebuild ?? false)
    : loading ? "读取中" : "未读取";
  const profileChanged = Boolean(
    currentConfig?.enabled
    && profileRevisionId
    && profileRevisionId !== currentConfig.chat_profile_revision_id,
  );
  const schemaProfileChanged = Boolean(
    currentConfig?.enabled
    && schemaProfileKey
    && schemaProfileKey !== currentConfig.schema_profile_key,
  );
  const selectedSchemaProfile = schemaProfiles.find(
    (profile) => profile.key === schemaProfileKey,
  );
  const progress = currentConfig
    ? currentConfig.eligible_chunk_count > 0
      ? Math.min(100, Math.round(
        (currentConfig.processed_chunk_count / currentConfig.eligible_chunk_count) * 100,
      ))
      : currentConfig.status === "ready" ? 100 : 0
    : 0;
  const controlsBusy = busyAction !== null;

  return (
    <section className="management-panel graph-settings-panel">
      <div className="management-panel-heading graph-settings-heading">
        <div>
          <span className="eyebrow">检索增强</span>
          <h2>实体图谱 Graph</h2>
          <p>抽取实体及其关系，用于补充普通向量与关键词检索难以连接的多跳证据。</p>
        </div>
        <span
          className={`graph-status-badge ${status ?? "unknown"}`}
          role="status"
          aria-label={`Graph 状态：${statusLabel}`}
        >
          {statusLabel}
        </span>
      </div>

      <div className="graph-compatibility-note">
        <strong>现有知识库可直接启用</strong>
        <span>
          无需重新上传或重建普通索引；Graph 会在索引空闲时读取当前可检索的文本与表格 chunks，增量完成回填。
        </span>
      </div>

      {loading && !currentConfig ? (
        <div className="graph-loading" aria-live="polite">正在读取 Graph 配置…</div>
      ) : null}

      {error ? <InlineError message={`Graph 配置读取失败：${error}`} /> : null}
      {schemaProfilesError ? <InlineError message={`Graph 类型读取失败：${schemaProfilesError}`} /> : null}
      {actionError ? <InlineError message={`Graph 操作失败：${actionError}`} /> : null}

      {currentConfig ? (
        <div className="graph-settings-body">
          <div className="graph-model-row">
            <ModelSelect
              label={currentConfig.enabled ? "Graph 抽取模型" : "选择 Graph 抽取模型"}
              value={profileRevisionId}
              profiles={chatProfiles}
              disabled={controlsBusy}
              onChange={setProfileRevisionId}
            />
            {currentConfig.enabled ? (
              <div className="graph-current-model">
                <span>当前构建使用</span>
                <strong>{currentConfig.profile_name ?? "已保存的 Chat Profile"}</strong>
                <small>
                  {[currentConfig.provider_name, currentConfig.model]
                    .filter(Boolean)
                    .join(" · ") || "模型信息不可用"}
                </small>
              </div>
            ) : (
              <div className="graph-current-model muted">
                <span>启用后</span>
                <strong>后台增量回填</strong>
                <small>构建期间普通检索和聊天仍可继续使用</small>
              </div>
            )}
          </div>

          <div className="graph-model-row graph-schema-row">
            <label className="management-field">
              知识图谱类型
              <select
                value={schemaProfileKey}
                disabled={controlsBusy || !schemaProfiles.length}
                onChange={(event) => setSchemaProfileKey(event.target.value)}
                aria-describedby="graph-schema-profile-help"
              >
                {schemaProfiles.map((profile) => (
                  <option key={profile.key} value={profile.key}>
                    {profile.display_name}{profile.is_default ? "（默认）" : ""}
                  </option>
                ))}
              </select>
            </label>
            <div className="graph-current-model">
              <span>当前选择</span>
              <strong>{selectedSchemaProfile?.display_name ?? currentConfig.schema_profile_name}</strong>
              <small id="graph-schema-profile-help">
                {selectedSchemaProfile?.description ?? "Profile 信息不可用"}
              </small>
            </div>
          </div>

          {schemaProfileChanged ? (
            <div className="graph-runtime-note">
              更换知识图谱类型会创建新的 Graph 构建；旧的 READY 构建会继续服务，普通向量/关键词索引不受影响。
            </div>
          ) : null}

          {!chatProfiles.length ? (
            <div className="model-required-note graph-model-required">
              Graph 需要一个已启用且验证通过的 Chat 模型。
              <button type="button" onClick={onOpenModelSettings}>打开模型设置</button>
            </div>
          ) : null}

          {currentConfig.enabled ? (
            <>
              <div className="graph-progress-block">
                <div>
                  <strong>{graphProgressTitle(currentConfig)}</strong>
                  <span>{progress}%</span>
                </div>
                <progress max={100} value={progress} />
                {currentConfig.last_error_code ? (
                  <small>错误码：{currentConfig.last_error_code}</small>
                ) : null}
                <small>
                  目标类型：{currentConfig.schema_profile_name}
                  {currentConfig.active_build_schema_profile_key
                    ? ` · 当前服务：${currentConfig.active_build_schema_profile_key}`
                    : " · 尚无 READY 构建"}
                </small>
              </div>
              <dl className="graph-stat-grid">
                <div>
                  <dt>已处理 / 可处理</dt>
                  <dd>{currentConfig.processed_chunk_count} / {currentConfig.eligible_chunk_count}</dd>
                </div>
                <div>
                  <dt>抽取到图谱</dt>
                  <dd>{currentConfig.extracted_chunk_count}</dd>
                </div>
                <div>
                  <dt>无实体关系</dt>
                  <dd>{currentConfig.empty_chunk_count}</dd>
                </div>
                <div>
                  <dt>协议跳过 / 资源跳过</dt>
                  <dd>{currentConfig.protocol_skipped_count} / {currentConfig.resource_skipped_count}</dd>
                </div>
                <div>
                  <dt>总跳过 / 允许上限</dt>
                  <dd>{currentConfig.protocol_skipped_count + currentConfig.resource_skipped_count} / {currentConfig.allowed_skipped_count}</dd>
                </div>
              </dl>
            </>
          ) : null}

          <div className="graph-actions">
            {!currentConfig.enabled ? (
              <button
                className="primary-button"
                type="button"
                disabled={!profileRevisionId || controlsBusy}
                onClick={() => void runAction("configure")}
              >
                {busyAction === "configure" ? "正在启用…" : "启用并开始构建"}
              </button>
            ) : (
              <>
                {profileChanged || schemaProfileChanged ? (
                  <button
                    className="primary-button"
                    type="button"
                    disabled={controlsBusy}
                    onClick={() => void runAction("configure")}
                  >
                    {busyAction === "configure" ? "正在应用…" : "应用配置并重新构建"}
                  </button>
                ) : null}
                {currentConfig.requires_rebuild ? (
                  <button
                    className="primary-button"
                    type="button"
                    disabled={controlsBusy}
                    onClick={() => void runAction("force-rebuild")}
                  >
                    {busyAction === "force-rebuild" ? "正在代际重建…" : "重建为当前 Graph 代际"}
                  </button>
                ) : null}
                {currentConfig.status === "failed" && !currentConfig.requires_rebuild ? (
                  <button
                    className="primary-button"
                    type="button"
                    disabled={controlsBusy}
                    onClick={() => void runAction("retry")}
                  >
                    {busyAction === "retry" ? "正在重试…" : "重试构建"}
                  </button>
                ) : null}
                {currentConfig.status !== "building" && !currentConfig.requires_rebuild ? (
                  <button
                    className="quiet-button"
                    type="button"
                    title="生成新的 Graph 构建并重新抽取所有可处理 chunks"
                    disabled={controlsBusy}
                    onClick={() => void runAction("force-rebuild")}
                  >
                    {busyAction === "force-rebuild" ? "正在重建…" : "强制重建"}
                  </button>
                ) : null}
                <button
                  className="quiet-button"
                  type="button"
                  disabled={controlsBusy}
                  onClick={() => void runAction("refresh")}
                >
                  {busyAction === "refresh" ? "正在刷新…" : "刷新状态"}
                </button>
                <button
                  className="danger-button"
                  type="button"
                  disabled={controlsBusy}
                  onClick={() => void runAction("disable")}
                >
                  {busyAction === "disable" ? "正在禁用…" : "禁用 Graph"}
                </button>
              </>
            )}
          </div>
        </div>
      ) : !loading ? (
        <div className="graph-actions graph-retry-load">
          <button className="quiet-button" type="button" onClick={() => void runAction("refresh")}>重新读取</button>
        </div>
      ) : null}
    </section>
  );
}

function KnowledgeBaseCreator({
  modelSettings,
  onCreate,
  onOpenModelSettings,
}: {
  modelSettings: ModelSettings | null;
  onCreate: (value: {
    name: string;
    parsing: ParsingPreset;
    chunking: ChunkingPreset;
    embedding: KnowledgeBaseEmbeddingSelection;
    autoQa: { enabled: boolean; model_profile_revision_id?: string | null };
  }) => Promise<void>;
  onOpenModelSettings: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [parsing, setParsing] = useState<ParsingPreset>("text_local_v1");
  const [chunking, setChunking] = useState<ChunkingPreset>("structural_balanced_v2");
  const [multimodalStrategy, setMultimodalStrategy] = useState<"dual_space" | "unified_multimodal">("dual_space");
  const [textModel, setTextModel] = useState("");
  const [multimodalModel, setMultimodalModel] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoQaEnabled, setAutoQaEnabled] = useState(false);
  const [autoQaModel, setAutoQaModel] = useState("");
  const textModels = useMemo(
    () => validProfiles(modelSettings, "text_embedding"),
    [modelSettings],
  );
  const multimodalModels = useMemo(
    () => validProfiles(modelSettings, "multimodal_embedding"),
    [modelSettings],
  );
  const unifiedModels = useMemo(
    () => multimodalModels.filter(unifiedEligible),
    [multimodalModels],
  );
  const chatModels = useMemo(
    () => validProfiles(modelSettings, "chat"),
    [modelSettings],
  );

  useEffect(() => {
    if (!textModels.some((profile) => profile.revision_id === textModel)) {
      const preferred = modelSettings?.selection.text_embedding_profile_revision_id;
      setTextModel(textModels.some((profile) => profile.revision_id === preferred)
        ? preferred!
        : textModels[0]?.revision_id ?? "");
    }
    if (!multimodalModels.some((profile) => profile.revision_id === multimodalModel)) {
      const preferred = modelSettings?.selection.multimodal_embedding_profile_revision_id;
      setMultimodalModel(multimodalModels.some((profile) => profile.revision_id === preferred)
        ? preferred!
        : multimodalModels[0]?.revision_id ?? "");
    }
    if (!chatModels.some((profile) => profile.revision_id === autoQaModel)) {
      const preferred = modelSettings?.selection.chat_profile_revision_id;
      setAutoQaModel(chatModels.some((profile) => profile.revision_id === preferred)
        ? preferred!
        : chatModels[0]?.revision_id ?? "");
    }
  }, [autoQaModel, chatModels, modelSettings, multimodalModel, multimodalModels, textModel, textModels]);

  const selectedMultimodalIsUnified = unifiedModels.some(
    (profile) => profile.revision_id === multimodalModel,
  );
  const embeddingValid = parsing === "text_local_v1"
    ? Boolean(textModel)
    : multimodalStrategy === "dual_space"
      ? Boolean(textModel && multimodalModel)
      : Boolean(multimodalModel && selectedMultimodalIsUnified);
  const autoQaValid = !autoQaEnabled || Boolean(autoQaModel);
  const selectionValid = embeddingValid && autoQaValid;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!name.trim() || !selectionValid || creating) return;
    setCreating(true);
    setError(null);
    const embedding: KnowledgeBaseEmbeddingSelection = parsing === "text_local_v1"
      ? { strategy: "text_only", text_profile_revision_id: textModel }
      : multimodalStrategy === "dual_space"
        ? {
          strategy: "dual_space",
          text_profile_revision_id: textModel,
          multimodal_profile_revision_id: multimodalModel,
        }
        : { strategy: "unified_multimodal", profile_revision_id: multimodalModel };
    try {
      await onCreate({
        name: name.trim(),
        parsing,
        chunking,
        embedding,
        autoQa: autoQaEnabled
          ? { enabled: true, model_profile_revision_id: autoQaModel }
          : { enabled: false },
      });
      setName("");
      setOpen(false);
    } catch (caught) {
      setError(managementError(caught));
    } finally {
      setCreating(false);
    }
  };

  return (
    <section className={`management-panel kb-creator${open ? " open" : ""}`}>
      <div className="management-panel-heading">
        <div>
          <span className="eyebrow">知识库配置</span>
          <h2>创建知识库</h2>
          <p>解析、chunk 和 embedding 配置会固化到索引版本中。</p>
        </div>
        <button className="primary-button" type="button" onClick={() => setOpen((value) => !value)}>
          {open ? "收起" : "新建知识库"}
        </button>
      </div>
      {open ? (
        <form className="kb-create-form" onSubmit={(event) => void submit(event)}>
          <label className="management-field full-field">
            名称
            <input
              value={name}
              maxLength={255}
              placeholder="例如：产品资料"
              onChange={(event) => setName(event.target.value)}
            />
          </label>
          <fieldset className="choice-group">
            <legend>解析模式</legend>
            <ChoiceCard
              active={parsing === "text_local_v1"}
              title="文本解析"
              description="提取文档中的文本、标题与表格文本。"
              onClick={() => setParsing("text_local_v1")}
            />
            <ChoiceCard
              active={parsing === "multimodal_local_v2"}
              title="多模态解析"
              description="额外抽取图片和视觉单元，适合图文资料。"
              onClick={() => setParsing("multimodal_local_v2")}
            />
          </fieldset>
          <fieldset className="choice-group">
            <legend>Chunk 模式</legend>
            <ChoiceCard
              active={chunking === "structural_balanced_v2"}
              title="结构切分"
              description="遵循标题和文档结构，结果更可解释。"
              onClick={() => setChunking("structural_balanced_v2")}
            />
            <ChoiceCard
              active={chunking === "semantic_balanced_v1"}
              title="语义切分"
              description="根据语义边界切分，适合长篇连续文本。"
              onClick={() => setChunking("semantic_balanced_v1")}
            />
          </fieldset>
          {parsing === "multimodal_local_v2" ? (
            <fieldset className="choice-group">
              <legend>Embedding 空间</legend>
              <ChoiceCard
                active={multimodalStrategy === "dual_space"}
                title="双空间"
                description="文本与图片分别使用对应模型。"
                onClick={() => setMultimodalStrategy("dual_space")}
              />
              <ChoiceCard
                active={multimodalStrategy === "unified_multimodal"}
                title="统一多模态"
                description="一个已验证的共享空间同时编码文字和图片。"
                onClick={() => setMultimodalStrategy("unified_multimodal")}
              />
            </fieldset>
          ) : null}
          <div className="model-selection-grid">
            {(parsing === "text_local_v1" || multimodalStrategy === "dual_space") ? (
              <ModelSelect
                label="文本 Embedding 模型"
                value={textModel}
                profiles={textModels}
                onChange={setTextModel}
              />
            ) : null}
            {parsing === "multimodal_local_v2" ? (
              <ModelSelect
                label={multimodalStrategy === "dual_space" ? "多模态 Embedding 模型" : "统一多模态模型"}
                value={multimodalModel}
                profiles={multimodalStrategy === "dual_space" ? multimodalModels : unifiedModels}
                onChange={setMultimodalModel}
              />
            ) : null}
          </div>
          <fieldset className="choice-group">
            <legend>Auto-QA 问句索引</legend>
            <ChoiceCard
              active={!autoQaEnabled}
              title="关闭"
              description="保持现有正文索引，不额外调用 Chat 模型。"
              onClick={() => setAutoQaEnabled(false)}
            />
            <ChoiceCard
              active={autoQaEnabled}
              title="开启"
              description="每个文本 Chunk 最多保留 5 个通过原文支持校验的问题，允许零个。生成和校验会增加一次性模型用量，问句补充仅在显式选择本地模型重排时生效，不作为答案或引用。"
              onClick={() => setAutoQaEnabled(true)}
            />
          </fieldset>
          {autoQaEnabled ? (
            <div className="model-selection-grid">
              <ModelSelect
                label="Auto-QA Chat 模型"
                value={autoQaModel}
                profiles={chatModels}
                onChange={setAutoQaModel}
              />
            </div>
          ) : null}
          {!embeddingValid ? (
            <div className="model-required-note">
              没有可用且验证通过的 Embedding 模型。
              <button type="button" onClick={onOpenModelSettings}>打开模型设置</button>
            </div>
          ) : null}
          {autoQaEnabled && !autoQaValid ? (
            <div className="model-required-note">
              开启 Auto-QA 需要可用且验证通过的 Chat 模型。
              <button type="button" onClick={onOpenModelSettings}>打开模型设置</button>
            </div>
          ) : null}
          {error ? <InlineError message={error} /> : null}
          <div className="form-actions-chat">
            <button className="primary-button" type="submit" disabled={!name.trim() || !selectionValid || creating}>
              {creating ? "正在创建…" : "创建知识库"}
            </button>
          </div>
        </form>
      ) : null}
    </section>
  );
}

function ChoiceCard({
  active,
  title,
  description,
  onClick,
}: {
  active: boolean;
  title: string;
  description: string;
  onClick: () => void;
}) {
  return (
    <button className={`choice-card${active ? " active" : ""}`} type="button" onClick={onClick}>
      <span className="choice-radio" aria-hidden="true" />
      <strong>{title}</strong>
      <small>{description}</small>
    </button>
  );
}

function ModelSelect({
  label,
  value,
  profiles,
  disabled = false,
  onChange,
}: {
  label: string;
  value: string;
  profiles: ModelProfile[];
  disabled?: boolean;
  onChange: (value: string) => void;
}) {
  return (
    <label className="management-field">
      {label}
      <select
        value={profiles.some((item) => item.revision_id === value) ? value : ""}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      >
        <option value="">请选择模型</option>
        {profiles.map((profile) => (
          <option key={profile.revision_id} value={profile.revision_id}>
            {profile.name} · r{profile.revision} · {profile.model}
          </option>
        ))}
      </select>
    </label>
  );
}

function UploadProgress({ items, jobs }: { items: UploadItem[]; jobs: IndexingJob[] }) {
  const completed = items.filter((item) => {
    if (item.status === "failed") return true;
    const job = item.jobId ? jobs.find((candidate) => candidate.job_id === item.jobId) : null;
    return Boolean(job && jobSettled(job));
  }).length;
  return (
    <div className="batch-progress" aria-live="polite">
      <div className="batch-progress-heading">
        <strong>{completed} / {items.length} 个文件处理完成</strong>
        <span>{Math.round((completed / items.length) * 100)}%</span>
      </div>
      <progress max={items.length} value={completed} />
      <div className="upload-item-list">
        {items.map((item) => {
          const job = item.jobId ? jobs.find((candidate) => candidate.job_id === item.jobId) : null;
          return (
            <div className="upload-item-chat" key={item.id}>
              <div>
                <strong>{item.file.name}</strong>
                <span>{formatBytes(item.file.size)}</span>
              </div>
              <JobState item={item} job={job ?? null} />
            </div>
          );
        })}
      </div>
    </div>
  );
}

function JobState({ item, job }: { item: UploadItem; job: IndexingJob | null }) {
  if (item.error) return <span className="status-text failed">{item.error}</span>;
  if (item.status === "waiting") return <span className="status-text">等待上传</span>;
  if (item.status === "uploading") return <span className="status-text active">正在上传</span>;
  if (!job) return <span className="status-text active">已接收，等待任务状态</span>;
  if (job.status === "failed") {
    return <span className="status-text failed">{jobFailure(job)}</span>;
  }
  if (job.status === "cancelled") return <span className="status-text failed">索引已取消</span>;
  if (job.status === "completed") {
    const message = job.serving_status === "serving"
      ? "解析和索引完成"
      : job.serving_status === "retired"
        ? "索引完成，但该版本已被替代"
        : "索引完成，正在切换 serving";
    return <span className="status-text success">{message}</span>;
  }
  return (
    <span className="status-text active">
      {phaseLabel(job.phase)} · {jobProgress(job)}%{autoQaProgressSuffix(job)}
    </span>
  );
}

function DocumentRow({
  document,
  job,
  selected,
  disabled,
  previewReady,
  onInspect,
  onUpdate,
  onDelete,
  onRetry,
}: {
  document: DocumentRecord;
  job: IndexingJob | null;
  selected: boolean;
  disabled: boolean;
  previewReady: boolean;
  onInspect: () => void;
  onUpdate: (event: ChangeEvent<HTMLInputElement>) => void;
  onDelete: () => void;
  onRetry: () => void;
}) {
  const progress = job ? jobProgress(job) : 0;
  return (
    <article className={`document-row${selected ? " selected" : ""}`}>
      <button
        className="document-main"
        type="button"
        disabled={!previewReady}
        title={previewReady ? "查看 Chunk 预览" : previewAvailability(job)}
        onClick={onInspect}
      >
        <span className="document-icon" aria-hidden="true">▤</span>
        <span className="document-copy">
          <strong>{document.display_name}</strong>
          <small>
            {document.current_version
              ? `v${document.current_version.version_number} · ${formatBytes(document.current_version.size_bytes)} · ${formatDate(document.updated_at)}`
              : "没有可用版本"}
          </small>
          <span className={`preview-availability${previewReady ? " ready" : ""}`}>
            {previewReady ? "查看 Chunk" : previewAvailability(job)}
          </span>
        </span>
      </button>
      <div className="document-progress">
        <div>
          <span>{job ? `${phaseLabel(job.phase)}${autoQaProgressSuffix(job)}` : "等待任务"}</span>
          <strong>{job?.status === "completed" && job.serving_status === "serving" ? "可检索" : `${progress}%`}</strong>
        </div>
        <progress max={100} value={progress} />
        {job?.status === "running" ? (
          <small className={`job-activity${jobLongRunning(job) ? " long-running" : ""}`}>
            {runningActivity(job)}
          </small>
        ) : null}
        {job?.status === "failed" ? <small>{jobFailure(job)}</small> : null}
      </div>
      <div className="document-actions">
        {job?.can_retry ? <button type="button" onClick={onRetry}>重试索引</button> : null}
        <label className={disabled ? "disabled" : ""}>
          更新文档
          <input type="file" accept={ACCEPTED_EXTENSIONS} disabled={disabled} onChange={onUpdate} />
        </label>
        <button className="danger-text" type="button" onClick={onDelete}>删除</button>
      </div>
    </article>
  );
}

function ChunkPreview({
  client,
  detail,
  inspection,
  loading,
  error,
  onClose,
  onLoadMore,
  onDelete,
}: {
  client: ApiClient;
  detail: DocumentDetail | null;
  inspection: DocumentChunkInspection | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
  onLoadMore: () => void;
  onDelete: (chunk: DocumentChunk) => void;
}) {
  return (
    <section className="management-panel chunk-preview-panel">
      <div className="management-panel-heading">
        <div>
          <span className="eyebrow">解析结果</span>
          <h2>{detail?.display_name ?? "Chunk 预览"}</h2>
          <p>{inspection ? `共 ${inspection.total_chunks} 个 chunk` : "读取当前 serving 索引中的切分结果。"}</p>
        </div>
        <button className="quiet-button" type="button" onClick={onClose}>关闭</button>
      </div>
      {error ? <InlineError message={error} /> : null}
      {loading && !inspection ? <EmptyState text="正在读取解析结果…" /> : null}
      {inspection ? (
        <div className="chunk-list">
          {inspection.items.map((chunk) => (
            <article className={`chunk-card${chunk.excluded_at ? " excluded" : ""}`} key={chunk.id}>
              <header>
                <div>
                  <span>#{chunk.ordinal + 1}</span>
                  <span>{modalityLabel(chunk.modality)}</span>
                  <span>{chunk.token_count} tokens</span>
                  {chunk.excluded_at ? <span className="excluded-badge">已从检索移除</span> : null}
                </div>
                <button
                  className="danger-text"
                  type="button"
                  disabled={Boolean(chunk.excluded_at)}
                  onClick={() => onDelete(chunk)}
                >
                  {chunk.excluded_at ? "已删除" : "删除 chunk"}
                </button>
              </header>
              {chunk.asset?.media_type.startsWith("image/") ? (
                <img
                  className="chunk-asset"
                  src={client.resolvePublicApiUrl(chunk.asset.content_url)}
                  alt={`Chunk ${chunk.ordinal + 1} 视觉内容`}
                />
              ) : null}
              <p>{chunk.content || "该视觉 chunk 没有文本表示。"}</p>
              <details className="chunk-questions">
                <summary>生成问题</summary>
                {chunk.generated_questions.length ? (
                  <ol>
                    {chunk.generated_questions.map((question) => (
                      <li key={question}>{question}</li>
                    ))}
                  </ol>
                ) : (
                  <p>未生成</p>
                )}
              </details>
              <details>
                <summary>位置与结构信息</summary>
                <pre>{JSON.stringify({
                  source_location: chunk.source_location,
                  hierarchy: chunk.hierarchy,
                  representations: chunk.representations,
                }, null, 2)}</pre>
              </details>
            </article>
          ))}
          {inspection.next_cursor ? (
            <button className="quiet-button load-chunks" type="button" disabled={loading} onClick={onLoadMore}>
              {loading ? "正在加载…" : "加载更多 chunk"}
            </button>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

function RetrievalDebugger({
  client,
  knowledgeBase,
}: {
  client: ApiClient;
  knowledgeBase: KnowledgeBase;
}) {
  const scopeGeneration = useRef(0);
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(knowledgeBase.retrieval_defaults.top_k);
  const [strategy, setStrategy] = useState<"exact_vector" | "hybrid">("exact_vector");
  const [rerankMode, setRerankMode] = useState<RerankMode>(
    knowledgeBase.retrieval_defaults.rerank_mode,
  );
  const [result, setResult] = useState<RetrievalEvidencePack | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    ++scopeGeneration.current;
    setQuery("");
    setResult(null);
    setError(null);
    setTopK(knowledgeBase.retrieval_defaults.top_k);
    setRerankMode(knowledgeBase.retrieval_defaults.rerank_mode);
  }, [
    knowledgeBase.id,
    knowledgeBase.retrieval_defaults.rerank_mode,
    knowledgeBase.retrieval_defaults.top_k,
  ]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!query.trim() || loading) return;
    const token: ManagementKbScope = {
      generation: scopeGeneration.current,
      knowledgeBaseId: knowledgeBase.id,
    };
    const isCurrent = () => isManagementKbScopeCurrent(token, {
      generation: scopeGeneration.current,
      knowledgeBaseId: knowledgeBase.id,
    });
    setLoading(true);
    setError(null);
    try {
      const value = await client.queryRetrievalDebug(
        knowledgeBase.id,
        query.trim(),
        topK,
        strategy,
        rerankMode,
      );
      if (isCurrent()) setResult(value);
    } catch (caught) {
      if (isCurrent()) {
        setError(managementError(caught));
        setResult(null);
      }
    } finally {
      if (isCurrent()) setLoading(false);
    }
  };

  return (
    <section className="management-panel debug-panel">
      <div className="management-panel-heading">
        <div>
          <span className="eyebrow">Debug</span>
          <h2>测试索引结果</h2>
          <p>直接查看当前 serving 索引返回的 chunk、排序和分数。</p>
        </div>
      </div>
      <form className="debug-form" onSubmit={(event) => void submit(event)}>
        <textarea
          rows={3}
          value={query}
          placeholder="输入问题或关键词…"
          onChange={(event) => setQuery(event.target.value)}
        />
        <div className="debug-controls">
          <label>
            Top K
            <input
              type="number"
              min={1}
              max={rerankMode === "local_minilm_v1" ? 20 : 100}
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
            />
          </label>
          <label>
            检索策略
            <select value={strategy} onChange={(event) => {
              const next = event.target.value as "exact_vector" | "hybrid";
              setStrategy(next);
              if (next === "hybrid" && rerankMode === "none") {
                setRerankMode("classic");
              }
            }}>
              <option value="exact_vector">精确向量</option>
              <option value="hybrid">混合检索</option>
            </select>
          </label>
          <label>
            精排方式
            <select value={rerankMode} onChange={(event) => {
              const next = event.target.value as RerankMode;
              setRerankMode(next);
              if (next === "local_minilm_v1" && topK > 20) setTopK(20);
            }}>
              <option value="none" disabled={strategy === "hybrid"}>不精排</option>
              <option value="classic">经典精排</option>
              <option value="local_minilm_v1">本地 MiniLM</option>
            </select>
          </label>
          <button className="primary-button" type="submit" disabled={!query.trim() || loading}>
            {loading ? "检索中…" : "运行测试"}
          </button>
        </div>
      </form>
      {error ? <InlineError message={error} /> : null}
      {result ? (
        <div className="debug-results">
          <div className="debug-summary">
            <strong>{result.evidence.length} 个结果</strong>
            <span>策略 {result.strategy === "hybrid" ? "混合检索" : "精确向量"}</span>
            {result.debug ? <span>候选 {result.debug.text_candidate_count ?? 0} / {result.debug.lexical_candidate_count ?? 0}</span> : null}
            {result.debug?.model_rerank_candidate_count !== null
              && result.debug?.model_rerank_candidate_count !== undefined ? (
                <span>
                  模型精排 {result.debug.model_rerank_candidate_count} 个候选 / {result.debug.model_rerank_window_count ?? 0} 个窗口
                </span>
              ) : null}
          </div>
          {result.evidence.length ? result.evidence.map((evidence) => (
            <article className="debug-result" key={evidence.index_chunk_id}>
              <div className="debug-rank">{evidence.rank}</div>
              <div>
                <header>
                  <strong>{modalityLabel(evidence.modality)} · Chunk #{evidence.ordinal + 1}</strong>
                  <span>{evidence.score_kind} {evidence.score.toFixed(5)}</span>
                  {evidence.model_rerank_score !== null ? (
                    <span>
                      MiniLM #{evidence.model_rerank_rank} {evidence.model_rerank_score.toFixed(5)} · {evidence.model_rerank_window_count} 窗口
                    </span>
                  ) : null}
                </header>
                <p>{evidence.text || "该结果没有文本表示。"}</p>
                <small>文档 {evidence.document_id.slice(0, 8)} · {evidence.matched_representations.join(" / ")}</small>
                {result.debug?.matched_questions?.find((item) => item.index_chunk_id === evidence.index_chunk_id) ? (
                  <small>
                    命中问句：{result.debug.matched_questions.find((item) => item.index_chunk_id === evidence.index_chunk_id)?.question}
                  </small>
                ) : null}
              </div>
            </article>
          )) : <EmptyState text="当前 serving 索引没有返回匹配 chunk。" />}
        </div>
      ) : null}
    </section>
  );
}

function ConfirmationDialog({
  confirmation,
  value,
  busy,
  onChange,
  onCancel,
  onConfirm,
}: {
  confirmation: Confirmation;
  value: string;
  busy: boolean;
  onChange: (value: string) => void;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const typed = confirmation.kind === "knowledge-base";
  const noun = confirmation.kind === "knowledge-base"
    ? "知识库"
    : confirmation.kind === "document" ? "文档" : "chunk";
  return (
    <div className="settings-backdrop" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget && !busy) onCancel();
    }}>
      <section className="confirm-dialog" role="alertdialog" aria-modal="true" aria-labelledby="confirm-title">
        <span className="danger-mark" aria-hidden="true">!</span>
        <h2 id="confirm-title">删除{noun}？</h2>
        <p>
          {typed
            ? "知识库及其中所有文档将不再可访问，源文件会进入后台清理队列。"
            : confirmation.kind === "chunk"
              ? "该 chunk 会保留在解析预览中，但不会再参与检索。"
              : "文档及当前可检索版本将被移除。"}
        </p>
        <strong>{confirmation.name}</strong>
        {typed ? (
          <label className="management-field">
            输入知识库名称以确认
            <input autoFocus value={value} onChange={(event) => onChange(event.target.value)} />
          </label>
        ) : null}
        <div className="confirm-actions">
          <button className="quiet-button" type="button" disabled={busy} onClick={onCancel}>取消</button>
          <button className="danger-button solid" type="button" disabled={busy || (typed && value !== confirmation.name)} onClick={onConfirm}>
            {busy ? "正在删除…" : "确认删除"}
          </button>
        </div>
      </section>
    </div>
  );
}

function InlineError({ message }: { message: string }) {
  return <div className="management-error" role="alert">{message}</div>;
}

function EmptyState({ text }: { text: string }) {
  return <div className="management-empty">{text}</div>;
}

function graphStatusLabel(
  status: GraphConfig["status"],
  requiresRebuild = false,
): string {
  if (status === "ready" && requiresRebuild) return "需重建";
  const labels: Record<GraphConfig["status"], string> = {
    disabled: "未启用",
    building: "构建中",
    ready: "可用",
    failed: "构建失败",
  };
  return labels[status];
}

function graphProgressTitle(config: GraphConfig): string {
  if (config.status === "ready" && config.requires_rebuild) {
    return "历史构建已完成，请重建为当前 Graph 代际";
  }
  if (config.status === "ready") return "构建完成，Graph 检索已可用";
  if (config.status === "failed") return "构建中止，可重试或更换模型";
  if (config.eligible_chunk_count === 0) return "正在验证模型并扫描现有 chunks";
  return `正在构建，已处理 ${config.processed_chunk_count} 个 chunks`;
}

function autoQaStatus(knowledgeBase: KnowledgeBase): string {
  if (!knowledgeBase.auto_qa.enabled) {
    return "Auto-QA 关闭。如需开启，需要未来的整库重建功能。";
  }
  const model = knowledgeBase.auto_qa.model_name
    ? `${knowledgeBase.auto_qa.model_name} / r${knowledgeBase.auto_qa.model_revision ?? "?"}`
    : "已冻结 Chat 模型";
  return `Auto-QA · 最多 ${knowledgeBase.auto_qa.questions_per_chunk} 问/Chunk · ${model}。此知识库创建后不可原地开关。`;
}

function autoQaProgressSuffix(job: IndexingJob): string {
  if (job.progress?.schema_version !== "auto_qa_generation_v1") return "";
  return ` · ${job.progress.processed_chunks}/${job.progress.eligible_chunks} chunks · ${job.progress.question_count} 问`;
}

function validProfiles(
  settings: ModelSettings | null,
  kind: ModelProfile["kind"],
): ModelProfile[] {
  return settings?.profiles.filter((profile) => (
    profile.kind === kind
    && profile.enabled
    && profile.validation_status === "valid"
    && profile.provider_secret_available
  )) ?? [];
}

function unifiedEligible(profile: ModelProfile): boolean {
  const validation = profile.embedding_validation;
  return Boolean(
    validation?.shared_text_image_space_confirmed
    && ["text_document", "text_query", "image"].every(
      (capability) => validation.input_capabilities.includes(
        capability as "text_document" | "text_query" | "image",
      ),
    ),
  );
}

function newestJob(jobs: IndexingJob[], documentId: string): IndexingJob | null {
  return jobs.find((job) => job.document_id === documentId) ?? null;
}

function jobSettled(job: IndexingJob): boolean {
  if (!TERMINAL_JOB_STATUSES.has(job.status)) return false;
  return job.status !== "completed" || job.serving_status !== "candidate";
}

function chunkPreviewReady(job: IndexingJob | null): boolean {
  return Boolean(
    job?.status === "completed"
    && job.build_status === "ready"
    && job.serving_status === "serving",
  );
}

function previewAvailability(job: IndexingJob | null): string {
  if (!job) return "等待索引任务";
  if (job.status === "failed") return "索引失败，暂不可预览";
  if (job.status === "cancelled") return "索引已取消，暂不可预览";
  if (job.status === "completed" && job.serving_status === "candidate") {
    return "正在切换可检索版本";
  }
  if (job.status === "completed" && job.serving_status === "retired") {
    return "该版本已被替代";
  }
  return `${phaseLabel(job.phase)}，完成后可预览`;
}

function jobProgress(job: IndexingJob): number {
  if (job.status === "completed") return 100;
  if (job.status === "failed" || job.status === "cancelled") {
    return Math.max(4, phaseProgress(job.phase));
  }
  return phaseProgress(job.phase);
}

function phaseProgress(phase: string): number {
  const index = INDEX_PHASES.indexOf(phase as typeof INDEX_PHASES[number]);
  if (index < 0) return 8;
  return Math.round((index / (INDEX_PHASES.length - 1)) * 100);
}

function phaseLabel(phase: string): string {
  const labels: Record<string, string> = {
    queued: "等待处理",
    claimed: "任务已领取",
    source_read: "读取源文件",
    parsing: "解析文档",
    asset_extraction: "提取视觉资源",
    enrichment: "内容增强",
    semantic_analysis: "分析语义边界",
    embedding: "生成文本向量",
    multimodal_embedding: "生成多模态向量",
    auto_qa_generation: "生成问句索引",
    persisting: "写入索引",
    validating: "校验索引",
    completed: "处理完成",
    failed: "处理失败",
    cancelled: "已取消",
  };
  return labels[phase] ?? phase.replaceAll("_", " ");
}

function runningActivity(job: IndexingJob): string {
  const heartbeat = job.heartbeat_at ? Date.parse(job.heartbeat_at) : Number.NaN;
  const elapsedSeconds = jobElapsedSeconds(job);
  const heartbeatFresh = Number.isFinite(heartbeat)
    && Date.now() - heartbeat < 45_000;
  const elapsed = elapsedSeconds === null
    ? ""
    : ` · 已运行 ${formatElapsed(elapsedSeconds)}`;
  if (jobLongRunning(job)) {
    return `处理时间较长，任务未完成但仍会继续运行 · ${heartbeatFresh ? "Worker 正常" : "等待 Worker 心跳"}${elapsed}`;
  }
  return `${heartbeatFresh ? "Worker 正常" : "等待 Worker 心跳"}${elapsed}`;
}

function jobLongRunning(job: IndexingJob): boolean {
  const elapsed = jobElapsedSeconds(job);
  return elapsed !== null && elapsed >= LONG_RUNNING_JOB_SECONDS;
}

function jobElapsedSeconds(job: IndexingJob): number | null {
  const claimedAt = job.claimed_at ? Date.parse(job.claimed_at) : Number.NaN;
  const currentAttempt = Number.isFinite(claimedAt)
    ? Math.max(0, Math.floor((Date.now() - claimedAt) / 1000))
    : 0;
  const parsing = job.progress?.schema_version === "pdf_parsing_progress_v1"
    ? Math.max(0, Math.floor(job.progress.elapsed_ms / 1000))
    : 0;
  const elapsed = Math.max(currentAttempt, parsing);
  return elapsed > 0 ? elapsed : null;
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds} 秒`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return remainder ? `${minutes} 分 ${remainder} 秒` : `${minutes} 分钟`;
}

function jobFailure(job: IndexingJob): string {
  if (!job.error) return "索引失败，服务端未返回更多诊断。";
  const labels: Record<string, string> = {
    PARSER_NOT_CONFIGURED: "解析器未配置",
    PARSER_CHUNK_LIMIT_EXCEEDED: "文档切分数量超限",
    PARSER_RESOURCE_LIMIT: "解析资源达到限制",
    PARSER_CRASHED: "解析进程异常退出",
    PARSER_OUTPUT_INVALID: "解析结果无效",
    SEMANTIC_CHUNKING_FAILED: "语义切分失败",
    SOURCE_FILE_MISSING: "源文件缺失",
    SOURCE_FILE_INTEGRITY: "源文件完整性校验失败",
    EMBEDDING_PROVIDER_UNAVAILABLE: "Embedding 服务不可用",
    EMBEDDING_RESPONSE_INVALID: "Embedding 返回结果无效",
    EMBEDDING_SPACE_MISMATCH: "Embedding 空间不兼容",
    INDEX_PERSISTENCE_FAILED: "索引写入失败",
    INDEX_INCOMPLETE: "索引完整性校验失败",
    AUTO_QA_MODEL_UNAVAILABLE: "Auto-QA 所用 Chat 模型不可用",
    AUTO_QA_RESPONSE_INVALID: "Auto-QA 模型输出无效",

  };
  const legacyTimeLimit = job.error.detail.limit_name;
  if (
    (job.error.code === "PARSER_RESOURCE_LIMIT"
      && ["document_timeout", "pdf_segment_timeout", "pdf_total_timeout"].includes(
        typeof legacyTimeLimit === "string" ? legacyTimeLimit : "",
      ))
    || job.error.code === "INDEXING_DEADLINE_EXCEEDED"
  ) {
    return "此任务此前因旧版耗时上限中止；当前版本已取消耗时上限，请点击“重试索引”继续处理。";
  }
  const detail = Object.entries(job.error.detail)
    .filter(([, value]) => typeof value === "string" || typeof value === "number")
    .map(([key, value]) => `${key}: ${String(value)}`)
    .join("；");
  return `${labels[job.error.code] ?? "索引失败"}（${job.error.code}）${detail ? `：${detail}` : ""}`;
}

function managementError(error: unknown): string {
  if (!(error instanceof ApiClientError)) {
    return error instanceof Error ? error.message : "操作未能完成，请稍后重试。";
  }
  const codeLabels: Record<string, string> = {
    FILE_MEDIA_TYPE_UNSUPPORTED: "文件类型不受支持",
    FILE_MEDIA_TYPE_MISMATCH: "文件扩展名与内容类型不匹配",
    FILE_TOO_LARGE: "文件超过允许大小",
    FILE_INVALID_UTF8: "文本文件不是有效 UTF-8",
    FILE_LINE_LIMIT_EXCEEDED: "文本行数超过限制",
    FILE_STRUCTURE_LIMIT_EXCEEDED: "文档结构复杂度超过限制",
    FILE_ARCHIVE_LIMIT_EXCEEDED: "压缩文档内容超过限制",
    FILE_CONTENT_INVALID: "文件内容无法读取或与扩展名不符",
    SOURCE_NOT_AVAILABLE: "源文件不可用",
    RESOURCE_NAME_CONFLICT: "同名知识库已经存在",
    RESOURCE_STATE_CONFLICT: "当前资源状态不允许该操作",
    REQUEST_VALIDATION_FAILED: "提交内容不符合要求",
  };
  const prefix = error.code ? codeLabels[error.code] : null;
  const trace = error.traceId ? `（追踪号 ${error.traceId}）` : "";
  return `${prefix ? `${prefix}：` : ""}${error.message}${trace}`;
}

function updateUpload(
  items: UploadItem[],
  id: string,
  patch: Partial<UploadItem>,
): UploadItem[] {
  return items.map((item) => item.id === id ? { ...item, ...patch } : item);
}

function mergeChunks(left: DocumentChunk[], right: DocumentChunk[]): DocumentChunk[] {
  const values = new Map(left.map((item) => [item.id, item]));
  for (const item of right) values.set(item.id, item);
  return [...values.values()].sort((a, b) => a.ordinal - b.ordinal);
}

function parsingLabel(value: ParsingPreset): string {
  return value === "text_local_v1" ? "文本解析" : "多模态解析";
}

function chunkingLabel(value: ChunkingPreset): string {
  return value === "structural_balanced_v2" ? "结构切分" : "语义切分";
}

function embeddingLabel(value: KnowledgeBase["embedding"]["strategy"]): string {
  if (value === "text_only") return "文本 Embedding";
  return value === "dual_space" ? "双空间 Embedding" : "统一多模态 Embedding";
}

function modalityLabel(value: string): string {
  return value === "image" ? "图片" : value === "table" ? "表格" : "文本";
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}
