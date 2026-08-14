import { useEffect, useState } from "react";

import { ApiClient, ApiClientError } from "./api/client";
import type {
  EvidencePack,
  GraphConfig,
  GraphDebug,
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
  graphConfig,
  graphConfigLoading,
  graphConfigError,
  onOpenDocument,
}: {
  client: ApiClient;
  knowledgeBase: KnowledgeBase;
  retrievalCapabilities: RetrievalCapabilities | null;
  retrievalCapabilitiesLoading: boolean;
  retrievalCapabilitiesError: unknown | null;
  graphConfig: GraphConfig | null;
  graphConfigLoading: boolean;
  graphConfigError: unknown | null;
  onOpenDocument: (documentId: string, versionId: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(knowledgeBase.retrieval_defaults.top_k);
  const [strategy, setStrategy] = useState<"exact_vector" | "hybrid" | "graph">("exact_vector");
  const [result, setResult] = useState<EvidencePack | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown | null>(null);
  const hybridEnabled = retrievalCapabilities?.modes.some(
    (item) => item.mode === "hybrid" && item.enabled,
  ) ?? false;
  const graphCapabilityEnabled = retrievalCapabilities?.modes.some(
    (item) => item.mode === "graph" && item.enabled,
  ) ?? false;
  const graphReady = graphCapabilityEnabled && graphConfig?.status === "ready";

  useEffect(() => {
    if (!hybridEnabled && strategy === "hybrid") setStrategy("exact_vector");
    if (!graphReady && strategy === "graph") setStrategy("exact_vector");
  }, [graphReady, hybridEnabled, strategy]);

  useEffect(() => {
    setQuery("");
    setTopK(knowledgeBase.retrieval_defaults.top_k);
    setStrategy("exact_vector");
    setResult(null);
    setError(null);
  }, [knowledgeBase.id, knowledgeBase.retrieval_defaults.top_k]);

  useEffect(() => {
    if (strategy !== "graph") return;
    setTopK((current) => Math.min(20, Math.max(4, current)));
  }, [strategy]);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!query.trim()) return;
    if (strategy === "hybrid" && !hybridEnabled) return;
    if (strategy === "graph" && !graphReady) return;
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
          <span className="policy-lock">
            {strategy === "graph"
              ? "Graph · Hybrid seed · Classic"
              : strategy === "hybrid" ? "Hybrid FTS" : "Exact vector"}
          </span>
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
              min={strategy === "graph" ? 4 : 1}
              max={strategy === "graph" ? 20 : 100}
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
            />
          </label>
          <label>
            Strategy
            <select
              value={strategy}
              onChange={(event) => setStrategy(
                event.target.value as "exact_vector" | "hybrid" | "graph",
              )}
            >
              <option value="exact_vector">Exact vector</option>
              <option value="hybrid" disabled={!hybridEnabled}>
                Hybrid FTS + dense{hybridEnabled ? "" : " (disabled)"}
              </option>
              <option value="graph" disabled={!graphReady}>
                Entity Graph{graphReady ? "" : " (not ready)"}
              </option>
            </select>
            <span className="field-hint">
              {retrievalCapabilitiesLoading
                ? "Capability status is loading; exact vector remains available."
                : retrievalCapabilitiesError || !retrievalCapabilities
                  ? "Capability status is unavailable; hybrid is disabled."
                  : strategy === "graph"
                    ? "Graph uses dense + lexical seeds, bounded 1–2 hop paths, and fixed Classic reranking."
                    : graphConfigLoading
                      ? "Graph configuration is loading."
                      : graphConfigError
                        ? "Graph configuration is unavailable; Graph remains disabled."
                        : graphConfig?.status !== "ready"
                          ? `Graph is ${graphConfig?.status ?? "disabled"}; complete the build before querying.`
                  : hybridEnabled
                    ? "Hybrid combines keywords and semantic search and may be slower."
                    : "Hybrid is not enabled for this API process."}
            </span>
          </label>
          <div className="locked-settings" aria-label="Locked retrieval settings">
            <div><span>Strategy</span><strong>{strategy}</strong></div>
            <div><span>Rerank</span><strong>{strategy === "graph" ? "classic (fixed)" : "disabled"}</strong></div>
            <div><span>Debug</span><strong>authorized</strong></div>
          </div>
          <div className="form-actions">
            <button
              className="button primary"
              type="submit"
              disabled={loading || !query.trim() || (strategy === "graph" && !graphReady)}
            >
              {loading ? "Retrieving…" : "Inspect serving evidence"}
            </button>
          </div>
        </form>
        {error ? <ProblemNotice error={error} /> : null}
      </section>

      <section className="panel graph-config-panel">
        <div className="panel-heading split-heading">
          <div>
            <p className="eyebrow">Derived capability</p>
            <h2>Entity Graph configuration</h2>
            <p>Graph is built from current serving text/table chunks and never changes ordinary indexing readiness.</p>
          </div>
          {graphConfig ? <StatusBadge value={graphConfig.status} /> : null}
        </div>
        {graphConfigLoading ? <p className="field-hint">Loading Graph configuration…</p> : null}
        {graphConfigError ? <ProblemNotice error={graphConfigError} /> : null}
        {graphConfig ? (
          <KeyValueGrid values={[
            ["Status", graphConfig.status],
            ["Profile", graphConfig.profile_name ?? "not selected"],
            ["Provider / model", graphConfig.provider_name && graphConfig.model
              ? `${graphConfig.provider_name} · ${graphConfig.model}`
              : "not selected"],
            ["Progress", `${graphConfig.processed_chunk_count} / ${graphConfig.eligible_chunk_count} chunks`],
            ["Extracted / empty", `${graphConfig.extracted_chunk_count} / ${graphConfig.empty_chunk_count}`],
            ["Protocol skipped", graphConfig.protocol_skipped_count],
            ["Resource skipped", graphConfig.resource_skipped_count],
            ["Last error", graphConfig.last_error_code ?? "none"],
          ]} />
        ) : null}
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
                <h2>Effective retrieval request</h2>
              </div>
            </div>
            <KeyValueGrid values={[
              ["Strategy", result.debug!.query_plan.strategy],
              ["Top K", result.debug!.query_plan.top_k],
              ["Distance", result.debug!.query_plan.distance_metric],
              ["Rerank", result.debug!.query_plan.rerank_mode],
              ["Result count", result.debug!.result_count],
              ["Text candidates", result.debug!.text_candidate_count],
              ["Lexical candidates", result.debug!.lexical_candidate_count],
              ["Cross-modal candidates", result.debug!.cross_modal_candidate_count],
              ["Lexical analyzer", result.debug!.lexical_analyzer_version],
              ["Lexical manifests", result.debug!.lexical_manifest_target_count],
              ["Hydrated relations", result.debug!.hydrated_relation_count],
              ["Evidence groups", result.debug!.evidence_group_count],
              ["Model candidates", result.debug!.model_rerank_candidate_count],
              ["Model windows", result.debug!.model_rerank_window_count],
              ["Active revision", shortId(result.debug!.resolved_active_revision_id)],
              ["Workspace", shortId(result.debug!.query_plan.workspace_id)],
            ]} />
          </section>

          {result.debug!.graph ? (
            <GraphDebugPanel debug={result.debug!.graph} />
          ) : null}

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

function GraphDebugPanel({ debug }: { debug: GraphDebug }) {
  return (
    <section className="panel graph-debug-panel">
      <div className="panel-heading split-heading">
        <div>
          <p className="eyebrow">Bounded graph evidence</p>
          <h2>Entity paths and bundles</h2>
          <p>Only the server-authorized path and bundle metadata is shown.</p>
        </div>
        <span className="count-label">{debug.paths.length} paths</span>
      </div>
      <KeyValueGrid values={[
        ["Dense seeds", debug.dense_seed_count],
        ["Lexical seeds", debug.lexical_seed_count],
        ["Fused seeds", debug.fused_seed_count],
        ["Query entities", debug.query_entity_count],
        ["1-hop / 2-hop paths", `${debug.one_hop_path_count} / ${debug.two_hop_path_count}`],
        ["Rejected paths", debug.rejected_path_count],
        ["Bundles", debug.bundle_count],
        ["Protocol / resource skips", `${debug.protocol_skipped_count} / ${debug.resource_skipped_count}`],
      ]} />
      {debug.paths.length ? (
        <div className="graph-path-list">
          {debug.paths.map((path) => (
            <article className="citation-card" key={path.path_id}>
              <KeyValueGrid values={[
                ["Path", path.path_id],
                ["Entry entity", path.entry_entity_key],
                ["Hops", path.hop_count],
                ["Seed entry", path.seed_entry ? "yes" : "no"],
                ["Rank", path.rank],
                ["Anchor chunk", shortId(path.anchor_chunk_id)],
                ["Support counts", path.support_counts.join(", ") || "none"],
                ["Source chunks", path.source_chunk_ids.map(shortId).join(", ")],
              ]} />
            </article>
          ))}
        </div>
      ) : (
        <EmptyState
          title="No graph paths"
          description="The bounded graph plan found no eligible path for this query."
        />
      )}
      {debug.bundles.length ? (
        <div className="evidence-details">
          <strong>Evidence bundles</strong>
          {debug.bundles.map((bundle) => (
            <p key={`${bundle.path_id}:${bundle.chunk_ids.join(",")}`}>
              {bundle.path_id}: {bundle.chunk_ids.map(shortId).join(", ")}
            </p>
          ))}
        </div>
      ) : null}
    </section>
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
