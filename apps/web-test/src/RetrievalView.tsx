import { useEffect, useState } from "react";

import { ApiClient, ApiClientError } from "./api/client";
import type {
  EvidencePack,
  KnowledgeBase,
  RelatedVisualEvidence,
  RetrievalCapabilities,
} from "./api/types";
import {
  AssetPreview,
  EmptyState,
  JsonDetails,
  KeyValueGrid,
  ProblemNotice,
  StatusBadge,
  shortId,
} from "./components";

export function RetrievalView({
  client,
  knowledgeBase,
  retrievalCapabilities,
  retrievalCapabilitiesLoading,
  retrievalCapabilitiesError,
  onOpenDocument,
}: {
  client: ApiClient;
  knowledgeBase: KnowledgeBase;
  retrievalCapabilities: RetrievalCapabilities | null;
  retrievalCapabilitiesLoading: boolean;
  retrievalCapabilitiesError: unknown | null;
  onOpenDocument: (documentId: string, versionId: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(knowledgeBase.retrieval_defaults.top_k);
  const [strategy, setStrategy] = useState<"exact_vector" | "hybrid">("exact_vector");
  const [result, setResult] = useState<EvidencePack | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown | null>(null);
  const hybridEnabled = retrievalCapabilities?.modes.some(
    (item) => item.mode === "hybrid" && item.enabled,
  ) ?? false;

  useEffect(() => {
    if (!hybridEnabled && strategy === "hybrid") setStrategy("exact_vector");
  }, [hybridEnabled, strategy]);

  useEffect(() => {
    setQuery("");
    setTopK(knowledgeBase.retrieval_defaults.top_k);
    setStrategy("exact_vector");
    setResult(null);
    setError(null);
  }, [knowledgeBase.id, knowledgeBase.retrieval_defaults.top_k]);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!query.trim()) return;
    if (strategy === "hybrid" && !hybridEnabled) return;
    setLoading(true);
    setError(null);
    try {
      const value = await client.queryRetrievalDebug(
        knowledgeBase.id,
        query.trim(),
        topK,
        strategy,
      );
      if (!value.debug) {
        throw new ApiClientError("The authorized debug contract was not returned.", {
          code: "FRONTEND_DEBUG_CONTRACT_MISSING",
        });
      }
      setResult(value);
    } catch (caught) {
      setError(caught);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="view-stack retrieval-layout">
      <section className="panel retrieval-query-panel">
        <div className="panel-heading split-heading">
          <div>
            <p className="eyebrow">Authorized evidence inspection</p>
            <h2>Retrieval Debug</h2>
            <p>Inspect the exact serving snapshot without weakening its filters.</p>
          </div>
          <span className="policy-lock">{strategy === "hybrid" ? "Hybrid FTS" : "Exact vector"}</span>
        </div>
        <form className="form-grid retrieval-form" onSubmit={submit}>
          <label className="wide-field">
            Retrieval query
            <textarea
              rows={3}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Enter an identifier, phrase, or question"
              required
            />
          </label>
          <label>
            Top K
            <input
              type="number"
              min={1}
              max={100}
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
            />
          </label>
          <label>
            Strategy
            <select
              value={strategy}
              onChange={(event) => setStrategy(event.target.value as "exact_vector" | "hybrid")}
            >
              <option value="exact_vector">Exact vector</option>
              <option value="hybrid" disabled={!hybridEnabled}>
                Hybrid FTS + dense{hybridEnabled ? "" : " (disabled)"}
              </option>
            </select>
            <span className="field-hint">
              {retrievalCapabilitiesLoading
                ? "Capability status is loading; exact vector remains available."
                : retrievalCapabilitiesError || !retrievalCapabilities
                  ? "Capability status is unavailable; hybrid is disabled."
                  : hybridEnabled
                    ? "Hybrid combines keywords and semantic search and may be slower."
                    : "Hybrid is not enabled for this API process."}
            </span>
          </label>
          <div className="locked-settings" aria-label="Locked retrieval settings">
            <div><span>Strategy</span><strong>{strategy}</strong></div>
            <div><span>Rerank</span><strong>disabled</strong></div>
            <div><span>Debug</span><strong>authorized</strong></div>
          </div>
          <div className="form-actions">
            <button className="button primary" type="submit" disabled={loading || !query.trim()}>
              {loading ? "Retrieving…" : "Inspect serving evidence"}
            </button>
          </div>
        </form>
        {error ? <ProblemNotice error={error} /> : null}
      </section>

      {!result ? (
        <section className="panel">
          <EmptyState
            title="No retrieval snapshot"
            description="Run a query to inspect evidence, ordering, and mandatory filters."
          />
        </section>
      ) : (
        <>
          <section className="panel query-plan-panel">
            <div className="panel-heading split-heading">
              <div>
                <p className="eyebrow">Resolved query plan</p>
                <h2>Mandatory serving filters</h2>
              </div>
              <div className="badge-row">
                <StatusBadge value={result.debug!.query_plan.build_status} />
                <StatusBadge value={result.debug!.query_plan.serving_status} />
              </div>
            </div>
            <KeyValueGrid values={[
              ["Strategy", result.debug!.query_plan.strategy],
              ["Top K", result.debug!.query_plan.top_k],
              ["Revision selector", result.debug!.query_plan.revision_selector],
              ["Current version only", result.debug!.query_plan.current_document_version_only],
              ["Distance", result.debug!.query_plan.distance_metric],
              ["Iterative scan", result.debug!.query_plan.iterative_scan],
              ["Rerank", result.debug!.query_plan.rerank],
              ["Result count", result.debug!.result_count],
              ["Text candidates", result.debug!.text_candidate_count],
              ["Lexical candidates", result.debug!.lexical_candidate_count],
              ["Cross-modal candidates", result.debug!.cross_modal_candidate_count],
              ["Lexical analyzer", result.debug!.lexical_analyzer_version],
              ["Lexical manifests", result.debug!.lexical_manifest_target_count],
              ["Hydrated relations", result.debug!.hydrated_relation_count],
              ["Evidence groups", result.debug!.evidence_group_count],
              ["Active revision", shortId(result.debug!.resolved_active_revision_id)],
              ["Workspace", shortId(result.debug!.query_plan.workspace_id)],
            ]} />
          </section>

          <section className="panel">
            <div className="panel-heading split-heading">
              <div>
                <p className="eyebrow">Ranked evidence</p>
                <h2>Serving chunks</h2>
                <p>Displayed in the exact order returned by the API.</p>
              </div>
              <span className="count-label">{result.evidence.length} results</span>
            </div>
            {result.evidence.length === 0 ? (
              <EmptyState
                title="No serving evidence"
                description="The authorized serving snapshot returned no matching chunks."
              />
            ) : (
              <div className="evidence-list">
                {result.evidence.map((evidence) => (
                  <article className="evidence-card" key={evidence.index_chunk_id}>
                    <div className="rank-column">
                      <span>Rank</span>
                      <strong>{evidence.rank}</strong>
                    </div>
                    <div className="evidence-body">
                      <div className="evidence-score-row">
                        <span>{evidence.modality} · {evidence.score_kind}</span>
                        <strong>{evidence.score.toFixed(6)}</strong>
                      </div>
                      {evidence.asset ? (
                        <AssetPreview
                          client={client}
                          asset={evidence.asset}
                          alt={`${evidence.modality} evidence at rank ${evidence.rank}`}
                        />
                      ) : null}
                      <p className="evidence-text">
                        {evidence.text || "No text representation is available for this visual asset."}
                      </p>
                      {evidence.related_visuals.length ? (
                        <div className="evidence-details">
                          <strong>Related visual descriptors</strong>
                          {evidence.related_visuals.map((visual) => (
                            <RelatedVisualDescriptor
                              key={`${visual.visual_unit_id}:${visual.asset.id}`}
                              client={client}
                              visual={visual}
                            />
                          ))}
                        </div>
                      ) : null}
                      <KeyValueGrid values={[
                        ["Modality", evidence.modality],
                        ["Representations", evidence.matched_representations.join(", ")],
                        ["Text lane rank", evidence.text_space_rank],
                        ["Lexical lane rank", evidence.lexical_rank],
                        ["Cross-modal rank", evidence.cross_modal_rank],
                        ["Fusion score", evidence.fusion_score?.toFixed(8)],
                        ["Chunk", shortId(evidence.index_chunk_id)],
                        ["Ordinal", evidence.ordinal],
                        ["Document", shortId(evidence.document_id)],
                        ["Version", shortId(evidence.document_version_id)],
                        ["Revision", shortId(evidence.index_revision_id)],
                      ]} />
                      <div className="evidence-details">
                        <JsonDetails label="Source location" value={evidence.source_location} />
                        <JsonDetails label="Source metadata" value={evidence.source_metadata} />
                        <JsonDetails label="Hierarchy" value={evidence.hierarchy} />
                      </div>
                      <button
                        className="button secondary"
                        type="button"
                        onClick={() => onOpenDocument(
                          evidence.document_id,
                          evidence.document_version_id,
                        )}
                      >
                        View current document metadata
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
}

function RelatedVisualDescriptor({
  client,
  visual,
}: {
  client: ApiClient;
  visual: RelatedVisualEvidence;
}) {
  let previewUrl: string | null = null;
  try {
    previewUrl = client.resolvePublicApiUrl(visual.asset.content_url);
  } catch {
    previewUrl = null;
  }
  return (
    <div className="citation-card">
      <KeyValueGrid values={[
        ["Figure", visual.figure_label],
        ["Relation", visual.relation_type],
        ["Provenance", visual.relation_provenance],
        ["Confidence micros", visual.relation_confidence_micros],
        ["Visual unit", shortId(visual.visual_unit_id)],
        ["Group", visual.evidence_group_key],
        ["Text lane rank", visual.text_space_rank],
        ["Lexical lane rank", visual.lexical_rank],
        ["Cross-modal rank", visual.cross_modal_rank],
      ]} />
      {previewUrl ? (
        <a className="button secondary" href={previewUrl} target="_blank" rel="noreferrer">
          Open authorized preview
        </a>
      ) : null}
    </div>
  );
}
