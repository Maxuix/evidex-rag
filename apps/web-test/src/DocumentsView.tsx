import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiClient, ApiClientError } from "./api/client";
import { DocumentChunksView } from "./DocumentChunksView";
import type {
  DocumentRecord,
  DocumentDetail,
  DocumentUpload,
  IndexingJob,
  KnowledgeBase,
} from "./api/types";
import {
  EmptyState,
  JsonDetails,
  KeyValueGrid,
  ProblemNotice,
  StatusBadge,
  formatDate,
  shortId,
} from "./components";
import {
  forgetTrackedJob,
  readTrackedJobs,
  trackJob,
  type TrackedJobReference,
} from "./storage";
import {
  buildMarkdownBundle,
  inspectMarkdownFolder,
} from "./markdownBundle";

type UploadItemStatus = "pending" | "uploading" | "queued" | "failed";

interface UploadItem {
  file: File;
  displayName: string;
  documentId: string | null;
  idempotencyKey: string;
  status: UploadItemStatus;
  error: unknown | null;
}

interface JobObservation {
  value: IndexingJob | null;
  error: unknown | null;
}

export function DocumentsView({
  client,
  knowledgeBase,
  focusedDocumentId,
  focusedDocumentVersionId,
  onMutationPendingChange,
}: {
  client: ApiClient;
  knowledgeBase: KnowledgeBase;
  focusedDocumentId: string | null;
  focusedDocumentVersionId: string | null;
  onMutationPendingChange: (pending: boolean) => void;
}) {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [listError, setListError] = useState<unknown | null>(null);
  const [trackedJobs, setTrackedJobs] = useState<TrackedJobReference[]>([]);
  const [jobs, setJobs] = useState<Record<string, JobObservation>>({});
  const [uploadMode, setUploadMode] = useState<"new" | "version">("new");
  const [versionDocumentId, setVersionDocumentId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [sourceMode, setSourceMode] = useState<"file" | "markdown-folder">("file");
  const [folderFiles, setFolderFiles] = useState<File[]>([]);
  const [markdownEntrypoints, setMarkdownEntrypoints] = useState<string[]>([]);
  const [markdownEntrypoint, setMarkdownEntrypoint] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [uploadItems, setUploadItems] = useState<UploadItem[]>([]);
  const [uploadError, setUploadError] = useState<unknown | null>(null);
  const [retryingJobId, setRetryingJobId] = useState<string | null>(null);
  const [pendingRetry, setPendingRetry] = useState<{
    jobId: string;
    key: string;
  } | null>(null);
  const [retryError, setRetryError] = useState<unknown | null>(null);
  const [focusedDocument, setFocusedDocument] = useState<DocumentDetail | null>(null);
  const [focusError, setFocusError] = useState<unknown | null>(null);
  const [inspectedDocumentId, setInspectedDocumentId] = useState<string | null>(null);
  const pollGeneration = useRef(0);
  const uploading = uploadItems.some((item) => item.status === "uploading");
  const mutationPending = uploadItems.some((item) =>
    item.status === "pending"
    || item.status === "uploading"
    || item.status === "failed"
  ) || pendingRetry !== null;
  const markdownMediaEnabled =
    knowledgeBase.parsing.preset === "multimodal_local_v2";

  useEffect(() => {
    onMutationPendingChange(mutationPending);
    return () => onMutationPendingChange(false);
  }, [mutationPending, onMutationPendingChange]);

  const refreshDocuments = useCallback(async (cursor?: string) => {
    setLoading(true);
    setListError(null);
    try {
      const page = await client.listDocuments(knowledgeBase.id, cursor);
      setDocuments((current) => cursor
        ? mergeDocuments(current, page.items)
        : page.items);
      setNextCursor(page.next_cursor);
    } catch (error) {
      setListError(error);
    } finally {
      setLoading(false);
    }
  }, [client, knowledgeBase.id]);

  useEffect(() => {
    setDocuments([]);
    setNextCursor(null);
    setFile(null);
    setSelectedFiles([]);
    setSourceMode("file");
    setFolderFiles([]);
    setMarkdownEntrypoints([]);
    setMarkdownEntrypoint("");
    setDisplayName("");
    setUploadItems([]);
    setUploadError(null);
    setVersionDocumentId("");
    setInspectedDocumentId(null);
    setTrackedJobs(readTrackedJobs(knowledgeBase.id));
    void refreshDocuments();
  }, [knowledgeBase.id, refreshDocuments]);

  useEffect(() => {
    const generation = ++pollGeneration.current;
    let timer: number | null = null;

    const poll = async () => {
      const results = await Promise.all(trackedJobs.map(async (reference) => {
        try {
          return [reference.jobId, {
            value: await client.getIndexingJob(reference.jobId),
            error: null,
          }] as const;
        } catch (error) {
          return [reference.jobId, { value: null, error }] as const;
        }
      }));
      if (generation !== pollGeneration.current) return;
      const next = Object.fromEntries(results) as Record<string, JobObservation>;
      setJobs(next);
      const shouldContinue = Object.values(next).some(({ value, error }) =>
        isRetryableObservationError(error) || (!error && value !== null && (
          value.status === "queued"
          || value.status === "running"
          || (value.status === "completed" && value.serving_status === "candidate")
        )));
      if (shouldContinue) {
        timer = window.setTimeout(poll, 1800);
      }
    };

    if (trackedJobs.length) void poll();
    else setJobs({});
    return () => {
      pollGeneration.current += 1;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [client, trackedJobs]);

  useEffect(() => {
    if (!focusedDocumentId) {
      setFocusedDocument(null);
      setFocusError(null);
      return;
    }
    let cancelled = false;
    setFocusError(null);
    void client.getDocument(focusedDocumentId).then((value) => {
      if (!cancelled) setFocusedDocument(value);
    }).catch((error) => {
      if (!cancelled) {
        setFocusedDocument(null);
        setFocusError(error);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [client, focusedDocumentId]);

  const activeJobCount = useMemo(() => Object.values(jobs).filter(({ value }) =>
    value?.status === "queued" || value?.status === "running"
  ).length, [jobs]);
  const batchSelection = uploadMode === "new"
    && sourceMode === "file"
    && selectedFiles.length > 1;
  const queuedUploadCount = uploadItems.filter((item) => item.status === "queued").length;
  const failedUploadCount = uploadItems.filter((item) => item.status === "failed").length;
  const hasUploadFile = sourceMode === "markdown-folder"
    ? file !== null
    : selectedFiles.length > 0;

  const resetUploadInputs = () => {
    setFile(null);
    setSelectedFiles([]);
    setDisplayName("");
    setUploadError(null);
    const input = document.getElementById("document-file") as HTMLInputElement | null;
    if (input) input.value = "";
    const folderInput = document.getElementById(
      "markdown-folder",
    ) as HTMLInputElement | null;
    if (folderInput) folderInput.value = "";
  };

  const updateUploadItem = (
    idempotencyKey: string,
    update: Partial<UploadItem>,
  ) => {
    setUploadItems((current) => current.map((item) =>
      item.idempotencyKey === idempotencyKey ? { ...item, ...update } : item
    ));
  };

  const submitUploadItem = async (item: UploadItem): Promise<boolean> => {
    updateUploadItem(item.idempotencyKey, { status: "uploading", error: null });
    try {
      const result = item.documentId
        ? await client.uploadDocumentVersion(
            item.documentId,
            item.file,
            item.displayName,
            item.idempotencyKey,
          )
        : await client.uploadDocument(
            knowledgeBase.id,
            item.file,
            item.displayName,
            item.idempotencyKey,
          );
      rememberUpload(result);
      setDocuments((current) => mergeDocuments(current, [result.document]));
      updateUploadItem(item.idempotencyKey, { status: "queued", error: null });
      return true;
    } catch (error) {
      updateUploadItem(item.idempotencyKey, { status: "failed", error });
      return false;
    }
  };

  const processUploadBatch = async (items: UploadItem[]) => {
    let allQueued = true;
    for (const item of items) {
      const succeeded = await submitUploadItem(item);
      if (!succeeded) allQueued = false;
    }
    if (allQueued) {
      setUploadItems([]);
      resetUploadInputs();
    }
  };

  const submitUpload = async (event: React.FormEvent) => {
    event.preventDefault();
    const selected = sourceMode === "markdown-folder"
      ? (file ? [file] : [])
      : selectedFiles;
    const uploadFiles = uploadMode === "version" ? selected.slice(0, 1) : selected;
    if (loading || mutationPending || uploadFiles.length === 0) return;
    const batchMode = uploadMode === "new"
      && sourceMode === "file"
      && uploadFiles.length > 1;
    const items: UploadItem[] = uploadFiles.map((selectedFile) => ({
      file: selectedFile,
      displayName: batchMode
        ? selectedFile.name
        : displayName.trim() || selectedFile.name,
      documentId: uploadMode === "version" ? versionDocumentId : null,
      idempotencyKey: crypto.randomUUID(),
      status: "pending",
      error: null,
    }));
    setUploadItems(items);
    await processUploadBatch(items);
  };

  const retryUploadItem = async (item: UploadItem) => {
    if (loading || uploadItems.some((candidate) =>
      candidate.status === "pending" || candidate.status === "uploading"
    )) return;
    const succeeded = await submitUploadItem(item);
    if (!succeeded) return;
    const allQueued = uploadItems.every((candidate) =>
      candidate.idempotencyKey === item.idempotencyKey
      || candidate.status === "queued"
    );
    if (allQueued) {
      setUploadItems([]);
      resetUploadInputs();
    }
  };

  const discardUploadItem = (idempotencyKey: string) => {
    if (uploadItems.some((item) =>
      item.status === "pending" || item.status === "uploading"
    )) return;
    const target = uploadItems.find((item) => item.idempotencyKey === idempotencyKey);
    if (!target || target.status !== "failed") return;
    const remaining = uploadItems.filter((item) => item.idempotencyKey !== idempotencyKey);
    if (remaining.length === 0 || remaining.every((item) => item.status === "queued")) {
      setUploadItems([]);
      resetUploadInputs();
    } else {
      setUploadItems(remaining);
    }
  };

  const rememberUpload = (result: DocumentUpload) => {
    const reference = {
      jobId: result.job_id,
      knowledgeBaseId: knowledgeBase.id,
      documentId: result.document.id,
      documentVersionId: result.document_version_id,
      indexedDocumentVersionId: result.indexed_document_version_id,
      indexRevisionId: result.index_revision_id,
    };
    trackJob(reference);
    setTrackedJobs(readTrackedJobs(knowledgeBase.id));
  };

  const retryJob = async (jobId: string, existingKey?: string) => {
    if ((loading || mutationPending) && !existingKey) return;
    const key = existingKey ?? crypto.randomUUID();
    setRetryingJobId(jobId);
    setPendingRetry({ jobId, key });
    setRetryError(null);
    try {
      const value = await client.retryIndexingJob(jobId, key);
      setJobs((current) => ({ ...current, [jobId]: { value, error: null } }));
      // A failed job has no outstanding poll timer. Refresh the observed
      // references so the polling effect resumes for the re-queued job.
      setTrackedJobs((current) => [...current]);
      setPendingRetry(null);
    } catch (error) {
      setRetryError(error);
    } finally {
      setRetryingJobId(null);
    }
  };

  const forgetJob = (jobId: string) => {
    forgetTrackedJob(jobId);
    setTrackedJobs(readTrackedJobs(knowledgeBase.id));
  };

  const selectMarkdownFolder = async (selected: File[]) => {
    setUploadError(null);
    try {
      const inspected = inspectMarkdownFolder(selected);
      if (inspected.entrypoints.length === 0) {
        throw new Error("The selected folder does not contain a Markdown file.");
      }
      const entrypoint = inspected.entrypoints[0];
      setFolderFiles(selected);
      setMarkdownEntrypoints(inspected.entrypoints);
      setMarkdownEntrypoint(entrypoint);
      const bundle = await buildMarkdownBundle(selected, entrypoint);
      setFile(bundle);
      if (!displayName) setDisplayName(entrypoint.split("/").pop() ?? entrypoint);
    } catch (error) {
      setFolderFiles([]);
      setMarkdownEntrypoints([]);
      setMarkdownEntrypoint("");
      setFile(null);
      setUploadError(error);
    }
  };

  const selectMarkdownEntrypoint = async (entrypoint: string) => {
    setMarkdownEntrypoint(entrypoint);
    setUploadError(null);
    try {
      setFile(await buildMarkdownBundle(folderFiles, entrypoint));
    } catch (error) {
      setFile(null);
      setUploadError(error);
    }
  };

  return (
    <div className="view-stack">
      {focusedDocumentId ? (
        <section className="panel focus-panel" aria-live="polite">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Citation navigation</p>
              <h2>Current document metadata</h2>
            </div>
          </div>
          {focusError ? (
            <ProblemNotice
              error={focusError}
              title="Current document metadata is unavailable"
            />
          ) : focusedDocument ? (
            <>
              <p className="supporting-copy">
                Citations remain authoritative snapshots. This panel shows only the
                document&apos;s current public metadata and may refer to a newer version.
              </p>
              <KeyValueGrid values={[
                ["Document", focusedDocument.display_name],
                ["Document ID", focusedDocument.id],
                ["Cited version ID", shortId(focusedDocumentVersionId)],
                ["Current version", focusedDocument.current_version?.version_number],
                ["Version ID", focusedDocument.current_version?.id],
                [
                  "Cited version is current",
                  focusedDocumentVersionId
                    ? focusedDocument.current_version?.id === focusedDocumentVersionId
                    : null,
                ],
                ["Source status", focusedDocument.current_version?.source_status],
                ["Index build status", focusedDocument.index?.build_status],
                ["Serving status", focusedDocument.index?.serving_status],
                ["Evidence units", focusedDocument.index?.unit_count],
                ["Composite text chunks", focusedDocument.index?.composite_chunk_count],
                ["Visual units", focusedDocument.index?.visual_unit_count],
                ["Assets", focusedDocument.index?.asset_count],
                ["Asset relations", focusedDocument.index?.relation_count],
                ["Representations", focusedDocument.index?.representation_count],
                ["Text representations", focusedDocument.index?.text_representation_count],
                ["Native-image representations", focusedDocument.index?.native_image_representation_count],
                ["Table representations", focusedDocument.index?.table_representation_count],
              ]} />
            </>
          ) : <p>Loading document metadata…</p>}
        </section>
      ) : null}

      {inspectedDocumentId ? (
        <DocumentChunksView
          client={client}
          documentId={inspectedDocumentId}
          documentName={documents.find((item) => item.id === inspectedDocumentId)?.display_name ?? shortId(inspectedDocumentId)}
          onClose={() => setInspectedDocumentId(null)}
        />
      ) : null}

      <section className="panel upload-panel">
        <div className="panel-heading split-heading">
          <div>
            <p className="eyebrow">Source admission</p>
            <h2>Upload a document</h2>
            <p>Raw UTF-8 text, queued for durable indexing.</p>
          </div>
          <div className="metric-chip">
            <span>Active jobs</span>
            <strong>{activeJobCount}</strong>
          </div>
        </div>
        <form className="form-grid" onSubmit={submitUpload}>
          <fieldset className="segmented-field">
            <legend>Upload mode</legend>
            <label>
              <input
                type="radio"
                name="upload-mode"
                checked={uploadMode === "new"}
                onChange={() => setUploadMode("new")}
                disabled={loading || mutationPending}
              />
              New document
            </label>
            <label>
              <input
                type="radio"
                name="upload-mode"
                checked={uploadMode === "version"}
                onChange={() => {
                  setUploadMode("version");
                  const first = selectedFiles[0] ?? null;
                  setSelectedFiles(first ? [first] : []);
                  if (first && !displayName) setDisplayName(first.name);
                }}
                disabled={loading || documents.length === 0 || mutationPending}
              />
              New version
            </label>
          </fieldset>
          {markdownMediaEnabled ? (
            <fieldset className="segmented-field">
              <legend>Source input</legend>
              <label>
                <input
                  type="radio"
                  name="source-mode"
                  checked={sourceMode === "file"}
                  onChange={() => {
                    setSourceMode("file");
                    setFile(null);
                    setSelectedFiles([]);
                    setUploadError(null);
                  }}
                  disabled={loading || mutationPending}
                />
                Single file
              </label>
              <label>
                <input
                  type="radio"
                  name="source-mode"
                  checked={sourceMode === "markdown-folder"}
                  onChange={() => {
                    setSourceMode("markdown-folder");
                    setFile(null);
                    setSelectedFiles([]);
                    setUploadError(null);
                  }}
                  disabled={loading || mutationPending}
                />
                Markdown folder
              </label>
            </fieldset>
          ) : null}
          {uploadMode === "version" ? (
            <label>
              Existing document
              <select
                value={versionDocumentId}
                onChange={(event) => setVersionDocumentId(event.target.value)}
                required
                disabled={loading || mutationPending}
              >
                <option value="">Select a document</option>
                {documents.filter((item) => !item.deleted_at).map((item) => (
                  <option key={item.id} value={item.id}>{item.display_name}</option>
                ))}
              </select>
            </label>
          ) : null}
          {sourceMode === "markdown-folder" && markdownMediaEnabled ? (
            <>
              <label>
                Markdown folder
                <input
                  id="markdown-folder"
                  type="file"
                  required
                  disabled={loading || mutationPending}
                  ref={(node) => node?.setAttribute("webkitdirectory", "")}
                  onChange={(event) =>
                    void selectMarkdownFolder(
                      Array.from(event.target.files ?? []),
                    )
                  }
                />
                <span className="field-hint">
                  The browser packages the Markdown entrypoint with local PNG,
                  JPEG, WebP, and supported static raster resources; no manual
                  ZIP step is required.
                </span>
              </label>
              {markdownEntrypoints.length > 1 ? (
                <label>
                  Markdown entrypoint
                  <select
                    value={markdownEntrypoint}
                    disabled={loading || mutationPending}
                    onChange={(event) =>
                      void selectMarkdownEntrypoint(event.target.value)
                    }
                  >
                    {markdownEntrypoints.map((entrypoint) => (
                      <option key={entrypoint} value={entrypoint}>
                        {entrypoint}
                      </option>
                    ))}
                  </select>
                </label>
              ) : null}
            </>
          ) : (
            <label>
              Document file
              <input
                id="document-file"
                type="file"
                multiple={uploadMode === "new" && sourceMode === "file"}
                accept=".txt,.md,.mdz,.html,.csv,.pdf,.docx,.pptx,.xlsx"
                required
                disabled={loading || mutationPending}
                onChange={(event) => {
                  const selected = Array.from(event.target.files ?? []);
                  setFile(null);
                  setSelectedFiles(selected);
                  setUploadError(null);
                  if (selected.length > 1) {
                    setDisplayName("");
                  } else if (selected[0] && !displayName) {
                    setDisplayName(selected[0].name);
                  }
                }}
              />
              <span className="field-hint">
                Supports UTF-8 .txt/.md/.html/.csv, .pdf, .docx/.pptx/.xlsx,
                and v2 Markdown .mdz bundles. Remote images in a v2 .md file
                are snapshotted automatically. If a .md references local
                ./images or ../images paths, use Markdown folder instead.
              </span>
            </label>
          )}
          <label>
            Display name
            <input
              value={displayName}
              onChange={(event) => setDisplayName(event.target.value)}
              maxLength={255}
              placeholder="quarterly-notes.md"
              disabled={loading || mutationPending || batchSelection}
            />
            {batchSelection ? (
              <span className="field-hint">
                Each selected file uses its original filename.
              </span>
            ) : null}
          </label>
          <div className="form-actions">
            <button
              className="button primary"
              type="submit"
              disabled={
                uploading
                || loading
                || mutationPending
                || !hasUploadFile
                || (uploadMode === "version" && !versionDocumentId)
              }
            >
              {uploading ? "Submitting…" : "Upload and index"}
            </button>
          </div>
        </form>
        {batchSelection && uploadItems.length === 0 ? (
          <section className="upload-selection" aria-live="polite">
            <div className="upload-progress-heading">
              <div>
                <p className="eyebrow">Batch selection</p>
                <h3>{selectedFiles.length} files selected</h3>
              </div>
              <span className="count-label">Ready to upload</span>
            </div>
            <ul className="upload-items">
              {selectedFiles.map((selectedFile, index) => (
                <li
                  className="upload-item"
                  key={`${selectedFile.name}-${selectedFile.lastModified}-${index}`}
                >
                  <strong>{selectedFile.name}</strong>
                </li>
              ))}
            </ul>
          </section>
        ) : null}
        {uploadItems.length > 0 ? (
          <section
            className="upload-progress"
            aria-label="Batch upload progress"
            aria-live="polite"
            aria-busy={uploading}
          >
            <div className="upload-progress-heading">
              <div>
                <p className="eyebrow">Batch upload</p>
                <h3>{queuedUploadCount} of {uploadItems.length} queued</h3>
              </div>
              <span className="count-label">
                {failedUploadCount ? `${failedUploadCount} failed` : "Indexing remains asynchronous"}
              </span>
            </div>
            <progress
              className="upload-progress-bar"
              value={queuedUploadCount}
              max={uploadItems.length}
              aria-label={`${queuedUploadCount} of ${uploadItems.length} uploads queued`}
            />
            <p className="field-hint">
              Queued means the upload was accepted; indexing status is tracked below.
            </p>
            <ul className="upload-items">
              {uploadItems.map((item) => (
                <li className="upload-item" key={item.idempotencyKey}>
                  <div className="upload-item-heading">
                    <div className="upload-item-name">
                      <strong>{item.file.name}</strong>
                      {item.displayName !== item.file.name ? (
                        <span className="field-hint">Display name: {item.displayName}</span>
                      ) : null}
                    </div>
                    <StatusBadge value={item.status} />
                  </div>
                  {item.status === "failed" && item.error ? (
                    <ProblemNotice
                      error={item.error}
                      title={`Upload failed for ${item.file.name}`}
                      onRetry={() => void retryUploadItem(item)}
                      onDiscard={() => discardUploadItem(item.idempotencyKey)}
                      discardLabel="Discard file"
                    />
                  ) : null}
                </li>
              ))}
            </ul>
          </section>
        ) : null}
        {uploadError ? (
          <ProblemNotice
            error={uploadError}
            title="Upload selection was not confirmed"
          />
        ) : null}
      </section>

      <section className="panel">
        <div className="panel-heading split-heading">
          <div>
            <p className="eyebrow">Durable work</p>
            <h2>Observed indexing jobs</h2>
            <p>Only jobs observed by this browser can be recovered after refresh.</p>
          </div>
          <span className="count-label">{trackedJobs.length} tracked</span>
        </div>
        {retryError && pendingRetry ? (
          <ProblemNotice
            error={retryError}
            title="Indexing retry was not confirmed"
            onRetry={() => void retryJob(pendingRetry.jobId, pendingRetry.key)}
            onDiscard={() => {
              setPendingRetry(null);
              setRetryError(null);
            }}
          />
        ) : null}
        {trackedJobs.length === 0 ? (
          <EmptyState
            title="No observed jobs"
            description="Upload a document to begin tracking its indexing state."
          />
        ) : (
          <div className="job-list">
            {trackedJobs.map((reference) => {
              const observation = jobs[reference.jobId];
              const job = observation?.value;
              const documentName = documents.find((item) => item.id === reference.documentId)?.display_name;
              return (
                <article className="job-card" key={reference.jobId}>
                  <div className="job-card-heading">
                    <div>
                      <p className="eyebrow">{documentName ?? shortId(reference.documentId)}</p>
                      <h3>{job?.phase ?? "Loading job state…"}</h3>
                    </div>
                    <div className="badge-row">
                      {job ? <StatusBadge value={job.status} /> : null}
                      {job ? <StatusBadge value={job.serving_status} /> : null}
                    </div>
                  </div>
                  {observation?.error ? (
                    <ProblemNotice
                      error={observation.error}
                      title="Job status is unavailable"
                      onRetry={() => setTrackedJobs((current) => [...current])}
                    />
                  ) : job ? (
                    <>
                      <KeyValueGrid values={[
                        ["Build", job.build_status],
                        ["Serving", job.serving_status],
                        ["Attempt", job.attempt],
                        ["Updated", formatDate(job.updated_at)],
                        ["Job ID", shortId(job.job_id)],
                        ["Version ID", shortId(job.document_version_id)],
                      ]} />
                      {job.error ? (
                        <div className="inline-error">
                          <strong>{job.error.code}</strong>
                          <JsonDetails label="Failure detail" value={job.error.detail} />
                        </div>
                      ) : null}
                    </>
                  ) : null}
                  <div className="card-actions">
                    {job?.can_retry ? (
                      <button
                        className="button secondary"
                        type="button"
                        disabled={
                          loading
                          || mutationPending
                          || retryingJobId === job.job_id
                        }
                        onClick={() => void retryJob(job.job_id)}
                      >
                        {retryingJobId === job.job_id ? "Retrying…" : "Retry failed job"}
                      </button>
                    ) : null}
                    <button
                      className="button text-button"
                      type="button"
                      disabled={loading || mutationPending}
                      onClick={() => forgetJob(reference.jobId)}
                    >
                      Forget locally
                    </button>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </section>

      <section className="panel">
        <div className="panel-heading split-heading">
          <div>
            <p className="eyebrow">Public content state</p>
            <h2>Documents</h2>
          </div>
          <button
            className="button subtle"
            type="button"
            disabled={loading || mutationPending}
            onClick={() => void refreshDocuments()}
          >
            Refresh
          </button>
        </div>
        {listError ? (
          <ProblemNotice
            error={listError}
            onRetry={mutationPending ? undefined : () => void refreshDocuments()}
          />
        ) : loading && documents.length === 0 ? (
          <p className="loading-line">Loading documents…</p>
        ) : documents.length === 0 ? (
          <EmptyState
            title="No documents"
            description="The first accepted upload will appear here immediately."
          />
        ) : (
          <div className="document-grid">
            {documents.map((item) => (
              <article className="document-card" key={item.id}>
                <div className="document-title-row">
                  <h3>{item.display_name}</h3>
                  {item.current_version ? (
                    <StatusBadge value={item.current_version.source_status} />
                  ) : null}
                </div>
                <KeyValueGrid values={[
                  ["Version", item.current_version?.version_number],
                  ["Bytes", item.current_version?.size_bytes],
                  ["Media", item.current_version?.media_type],
                  ["Updated", formatDate(item.updated_at)],
                  ["Document ID", shortId(item.id)],
                  ["Version ID", shortId(item.current_version?.id)],
                ]} />
                <button
                  className="button secondary document-inspector-button"
                  type="button"
                  disabled={loading || mutationPending || !item.current_version || Boolean(item.deleted_at)}
                  onClick={() => setInspectedDocumentId(item.id)}
                >
                  Inspect chunks
                </button>
              </article>
            ))}
          </div>
        )}
        {nextCursor ? (
          <button
            className="button secondary load-more"
            type="button"
            disabled={loading || mutationPending}
            onClick={() => void refreshDocuments(nextCursor)}
          >
            Load more documents
          </button>
        ) : null}
      </section>
    </div>
  );
}

function mergeDocuments(
  current: DocumentRecord[],
  incoming: DocumentRecord[],
): DocumentRecord[] {
  const byId = new Map(current.map((item) => [item.id, item]));
  for (const item of incoming) byId.set(item.id, item);
  return [...byId.values()].sort((left, right) =>
    right.created_at.localeCompare(left.created_at)
  );
}

function isRetryableObservationError(error: unknown | null): boolean {
  if (!error) return false;
  if (!(error instanceof ApiClientError)) return true;
  return error.status === null || error.retryable || error.status >= 500;
}
