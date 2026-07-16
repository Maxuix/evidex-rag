import type { JsonMap, KnowledgeBase } from "./api/types";
import { ApiClientError } from "./api/client";

export function KnowledgeBaseSelector({
  knowledgeBases,
  selectedId,
  loading,
  disabled,
  nextCursor,
  onSelect,
  onLoadMore,
}: {
  knowledgeBases: KnowledgeBase[];
  selectedId: string | null;
  loading: boolean;
  disabled: boolean;
  nextCursor: string | null;
  onSelect: (id: string) => void;
  onLoadMore: () => void;
}) {
  return (
    <div className="kb-selector">
      <label htmlFor="knowledge-base">Knowledge base</label>
      <div className="inline-controls">
        <select
          id="knowledge-base"
          value={selectedId ?? ""}
          onChange={(event) => onSelect(event.target.value)}
          disabled={disabled || loading || knowledgeBases.length === 0}
        >
          {knowledgeBases.length === 0 ? (
            <option value="">No knowledge bases yet</option>
          ) : null}
          {knowledgeBases.map((knowledgeBase) => (
            <option value={knowledgeBase.id} key={knowledgeBase.id}>
              {knowledgeBase.name}
            </option>
          ))}
        </select>
        {nextCursor ? (
          <button
            type="button"
            className="button subtle"
            onClick={onLoadMore}
            disabled={disabled || loading}
          >
            Load more
          </button>
        ) : null}
      </div>
    </div>
  );
}

export function ProblemNotice({
  error,
  title = "Request could not be completed",
  onRetry,
  onDiscard,
  discardLabel = "Discard pending request",
}: {
  error: unknown;
  title?: string;
  onRetry?: () => void;
  onDiscard?: () => void;
  discardLabel?: string;
}) {
  const known = error instanceof ApiClientError ? error : null;
  return (
    <section className="notice error-notice" role="alert">
      <div>
        <p className="eyebrow">{known?.code ?? "FRONTEND_ERROR"}</p>
        <h3>{title}</h3>
        <p>{known?.message ?? "An unexpected local frontend error occurred."}</p>
        {known?.traceId ? <p className="trace">Trace {known.traceId}</p> : null}
        {known?.fieldErrors?.length ? (
          <ul className="field-errors">
            {known.fieldErrors.map((item, index) => (
              <li key={`${item.location.join(".")}-${index}`}>
                {item.location.join(".")}: {item.message}
              </li>
            ))}
          </ul>
        ) : null}
      </div>
      {onRetry || onDiscard ? (
        <div className="notice-actions">
          {onRetry ? (
            <button className="button secondary" type="button" onClick={onRetry}>
              Retry same request
            </button>
          ) : null}
          {onDiscard ? (
            <button className="button text-button" type="button" onClick={onDiscard}>
              {discardLabel}
            </button>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

export function StatusBadge({ value }: { value: string }) {
  const normalized = value.toLowerCase().replaceAll("_", "-");
  return <span className={`status-badge status-${normalized}`}>{value}</span>;
}

export function EmptyState({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <div className="empty-state">
      <span aria-hidden="true">○</span>
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}

export function KeyValueGrid({
  values,
}: {
  values: Array<[string, string | number | boolean | null | undefined]>;
}) {
  return (
    <dl className="key-value-grid">
      {values.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{formatScalar(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

export function JsonDetails({ label, value }: { label: string; value: JsonMap }) {
  return (
    <details className="json-details">
      <summary>{label}</summary>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return "—";
  const parsed = new Date(value);
  return Number.isNaN(parsed.valueOf()) ? value : parsed.toLocaleString();
}

export function shortId(value: string | null | undefined): string {
  if (!value) return "—";
  return value.length > 16 ? `${value.slice(0, 8)}…${value.slice(-6)}` : value;
}

function formatScalar(value: string | number | boolean | null | undefined): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return String(value);
}
