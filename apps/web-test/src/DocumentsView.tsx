import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiClient, ApiClientError } from "./api/client";
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

interface PendingUpload {
  file: File;
  displayName: string;
  documentId: string | null;
  idempotencyKey: string;
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
  const [displayName, setDisplayName] = useState("");
  const [uploading, setUploading] = useState(false);
  const [pendingUpload, setPendingUpload] = useState<PendingUpload | null>(null);
  const [uploadError, setUploadError] = useState<unknown | null>(null);
  const [retryingJobId, setRetryingJobId] = useState<string | null>(null);
  const [pendingRetry, setPendingRetry] = useState<{
    jobId: string;
    key: string;
  } | null>(null);
  const [retryError, setRetryError] = useState<unknown | null>(null);
  const [focusedDocument, setFocusedDocument] = useState<DocumentDetail | null>(null);
  const [focusError, setFocusError] = useState<unknown | null>(null);
  const pollGeneration = useRef(0);
  const mutationPending = pendingUpload !== null || pendingRetry !== null;

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
    setDisplayName("");
    setUploadError(null);
    setPendingUpload(null);
    setVersionDocumentId("");
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

  const submitUpload = async (event: React.FormEvent) => {
    event.preventDefault();
    if (loading || mutationPending || !file) return;
    const pending: PendingUpload = {
      file,
      displayName: displayName.trim() || file.name,
      documentId: uploadMode === "version" ? versionDocumentId : null,
      idempotencyKey: crypto.randomUUID(),
    };
    setPendingUpload(pending);
    await performUpload(pending);
  };

  const performUpload = async (pending: PendingUpload) => {
    setUploading(true);
    setUploadError(null);
    try {
      const result = pending.documentId
        ? await client.uploadDocumentVersion(
            pending.documentId,
            pending.file,
            pending.displayName,
            pending.idempotencyKey,
          )
        : await client.uploadDocument(
            knowledgeBase.id,
            pending.file,
            pending.displayName,
            pending.idempotencyKey,
          );
      rememberUpload(result);
      setDocuments((current) => mergeDocuments(current, [result.document]));
      setFile(null);
      setDisplayName("");
      setPendingUpload(null);
      setUploadError(null);
      const input = document.getElementById("document-file") as HTMLInputElement | null;
      if (input) input.value = "";
    } catch (error) {
      setUploadError(error);
    } finally {
      setUploading(false);
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
                ["Assets", focusedDocument.index?.asset_count],
                ["Representations", focusedDocument.index?.representation_count],
              ]} />
            </>
          ) : <p>Loading document metadata…</p>}
        </section>
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
                onChange={() => setUploadMode("version")}
                disabled={loading || documents.length === 0 || mutationPending}
              />
              New version
            </label>
          </fieldset>
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
          <label>
            Document file
            <input
              id="document-file"
              type="file"
              accept=".txt,.md,.pdf,.docx,text/plain,text/markdown,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
              required
              disabled={loading || mutationPending}
              onChange={(event) => {
                const selected = event.target.files?.[0] ?? null;
                setFile(selected);
                if (selected && !displayName) setDisplayName(selected.name);
              }}
            />
            <span className="field-hint">
              Supports UTF-8 .txt/.md, text-based .pdf, and .docx files.
            </span>
          </label>
          <label>
            Display name
            <input
              value={displayName}
              onChange={(event) => setDisplayName(event.target.value)}
              maxLength={255}
              placeholder="quarterly-notes.md"
              disabled={loading || mutationPending}
            />
          </label>
          <div className="form-actions">
            <button
              className="button primary"
              type="submit"
              disabled={
                uploading
                || loading
                || mutationPending
                || !file
                || (uploadMode === "version" && !versionDocumentId)
              }
            >
              {uploading ? "Submitting…" : "Upload and index"}
            </button>
          </div>
        </form>
        {uploadError ? (
          <ProblemNotice
            error={uploadError}
            title="Upload was not confirmed"
            onRetry={pendingUpload ? () => void performUpload(pendingUpload) : undefined}
            onDiscard={pendingUpload ? () => {
              setPendingUpload(null);
              setUploadError(null);
            } : undefined}
            discardLabel="Discard and edit"
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
