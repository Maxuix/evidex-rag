import { useEffect, useMemo, useState } from "react";

import { ApiClient } from "./api/client";
import type { DocumentChunk, DocumentChunkInspection } from "./api/types";
import {
  AssetPreview,
  EmptyState,
  JsonDetails,
  KeyValueGrid,
  ProblemNotice,
  shortId,
} from "./components";

type ModalityFilter = "all" | DocumentChunk["modality"];

export function DocumentChunksView({
  client,
  documentId,
  documentName,
  onClose,
}: {
  client: ApiClient;
  documentId: string;
  documentName: string;
  onClose: () => void;
}) {
  const [inspection, setInspection] = useState<DocumentChunkInspection | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<unknown | null>(null);
  const [filter, setFilter] = useState<ModalityFilter>("all");

  const load = async (cursor?: string) => {
    if (cursor) setLoadingMore(true);
    else setLoading(true);
    setError(null);
    try {
      const page = await client.getDocumentChunks(documentId, cursor);
      setInspection((current) => cursor && current
        ? { ...page, items: [...current.items, ...page.items] }
        : page);
    } catch (caught) {
      setError(caught);
      if (!cursor) setInspection(null);
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  };

  useEffect(() => {
    setInspection(null);
    setFilter("all");
    void load();
  }, [documentId]); // The document identity is the immutable inspection selection.

  const visibleChunks = useMemo(() => (
    inspection?.items.filter((item) => filter === "all" || item.modality === filter) ?? []
  ), [filter, inspection]);
  const counts = useMemo(() => ({
    text: inspection?.items.filter((item) => item.modality === "text").length ?? 0,
    image: inspection?.items.filter((item) => item.modality === "image").length ?? 0,
    table: inspection?.items.filter((item) => item.modality === "table").length ?? 0,
    tokens: inspection?.items.reduce((total, item) => total + item.token_count, 0) ?? 0,
  }), [inspection]);

  return (
    <section className="panel chunk-inspector" aria-live="polite">
      <div className="panel-heading split-heading">
        <div>
          <p className="eyebrow">Stable current-version snapshot</p>
          <h2>Chunk inspector · {documentName}</h2>
          <p>Current source version and active index revision only. Indexing must be ready.</p>
        </div>
        <button className="button subtle" type="button" onClick={onClose}>Close inspector</button>
      </div>
      {error ? <ProblemNotice error={error} title="Chunk snapshot is not available" onRetry={() => void load()} /> : null}
      {loading ? <p className="loading-line">Loading stable chunk snapshot…</p> : null}
      {inspection ? (
        <>
          <KeyValueGrid values={[
            ["All chunks", inspection.total_chunks],
            ["Loaded", inspection.items.length],
            ["Loaded tokens", counts.tokens],
            ["Loaded text", counts.text],
            ["Loaded images", counts.image],
            ["Loaded tables", counts.table],
            ["Version", shortId(inspection.document_version_id)],
            ["Revision", shortId(inspection.index_revision_id)],
          ]} />
          <div className="chunk-timeline-wrap">
            <div className="subheading-row">
              <div>
                <p className="eyebrow">Ordinal map</p>
                <h3>Chunk timeline</h3>
              </div>
              <span className="count-label">{inspection.items.length} loaded</span>
            </div>
            <div className="chunk-timeline" aria-label="Chunk ordinal timeline">
              {inspection.items.map((item) => (
                <a
                  className={`chunk-timeline-item modality-${item.modality}`}
                  href={`#chunk-${item.id}`}
                  key={item.id}
                  style={{ flexBasis: `${Math.min(156, Math.max(28, Math.sqrt(item.token_count + 1) * 8))}px` }}
                  title={`#${item.ordinal} · ${item.modality} · ${item.token_count} tokens`}
                >
                  <span>#{item.ordinal}</span>
                  <strong>{item.token_count}</strong>
                </a>
              ))}
            </div>
            <p className="field-help">Block width approximates token count. Timeline grows as more pages are loaded.</p>
          </div>
          <div className="chunk-filter-row" role="group" aria-label="Filter chunks by modality">
            {(["all", "text", "image", "table"] as ModalityFilter[]).map((value) => (
              <button
                className={`button ${filter === value ? "primary" : "secondary"}`}
                type="button"
                key={value}
                onClick={() => setFilter(value)}
              >
                {value}
              </button>
            ))}
          </div>
          {visibleChunks.length === 0 ? (
            <EmptyState title="No matching loaded chunks" description="Change the modality filter or load the next page." />
          ) : (
            <div className="chunk-list">
              {visibleChunks.map((chunk) => <ChunkCard client={client} chunk={chunk} key={chunk.id} />)}
            </div>
          )}
          {inspection.next_cursor ? (
            <button
              className="button secondary load-more"
              type="button"
              disabled={loadingMore}
              onClick={() => void load(inspection.next_cursor ?? undefined)}
            >
              {loadingMore ? "Loading chunks…" : "Load next chunks"}
            </button>
          ) : inspection.items.length < inspection.total_chunks ? (
            <p className="field-help">The snapshot changed while paging. Refresh the inspector to obtain a new stable view.</p>
          ) : null}
        </>
      ) : null}
    </section>
  );
}

function ChunkCard({ client, chunk }: { client: ApiClient; chunk: DocumentChunk }) {
  return (
    <article className="chunk-card" id={`chunk-${chunk.id}`}>
      <div className="chunk-card-heading">
        <div><span className={`status-badge modality-badge modality-${chunk.modality}`}>{chunk.modality}</span><h3>Chunk #{chunk.ordinal}</h3></div>
        <span className="count-label">{chunk.token_count} tokens</span>
      </div>
      {chunk.asset ? <AssetPreview client={client} asset={chunk.asset} alt={`Chunk ${chunk.ordinal} asset`} /> : null}
      <pre className="chunk-content">{chunk.content}</pre>
      <KeyValueGrid values={[
        ["Chunk", shortId(chunk.id)],
        ["Evidence group", chunk.evidence_group_key],
        ["Representations", chunk.representations.join(", ") || "—"],
        ["Related visuals", chunk.related_visuals.length],
      ]} />
      {chunk.related_visuals.length ? (
        <div className="chunk-relations">
          <strong>Related visual assets</strong>
          {chunk.related_visuals.map((relation) => (
            <div className="chunk-relation" key={`${relation.visual_unit_id}:${relation.asset.id}:${relation.relation_type}`}>
              <AssetPreview client={client} asset={relation.asset} alt={relation.figure_label ?? relation.relation_type} />
              <span>{relation.relation_type} · {(relation.confidence_micros / 1_000_000).toFixed(2)} · {relation.figure_label ?? "unlabeled"}</span>
            </div>
          ))}
        </div>
      ) : null}
      <div className="evidence-details">
        <JsonDetails label="Source location" value={chunk.source_location} />
        <JsonDetails label="Hierarchy" value={chunk.hierarchy} />
        <JsonDetails label="Source metadata" value={chunk.source_metadata} />
      </div>
    </article>
  );
}
