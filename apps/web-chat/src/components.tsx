import { useEffect, useRef, useState, type ReactNode } from "react";
import { CaretRight, Check, Copy, FileText, X } from "@phosphor-icons/react";

import type { ApiClient } from "./api/client";
import type { ChatRun } from "./api/types";
import {
  citationOrdinals,
  documentName,
  formatSourceLocation,
} from "./format";

export function AnswerText({
  content,
  citationCount,
  onCitation,
}: {
  content: string;
  citationCount: number | null;
  onCitation?: (ordinal: number, trigger: HTMLButtonElement) => void;
}) {
  const paragraphs = content.split(/\n{2,}/);
  return (
    <div className="answer-text">
      {paragraphs.map((paragraph, paragraphIndex) => {
        const key = `${paragraphIndex}-${paragraph.slice(0, 20)}`;
        const heading = /^(#{1,3})\s+([^\n]+)$/.exec(paragraph);
        return heading ? <h3 key={key}>{renderInline(heading[2], citationCount, onCitation)}</h3>
          : <p key={key}>{renderInline(paragraph, citationCount, onCitation)}</p>;
      })}
    </div>
  );
}

export function SourcesButton({
  content,
  citations,
  onOpen,
}: {
  content: string;
  citations?: ChatRun["citations"];
  onOpen: (ordinal: number, trigger: HTMLButtonElement) => void;
}) {
  const ordinals = citationOrdinals(content);
  if (!ordinals.length) return null;
  const first = citations?.find((citation) => citation.ordinal === ordinals[0]);
  return (
    <button
      className="sources-button"
      type="button"
      aria-label={`${ordinals.length} 个来源`}
      onClick={(event) => onOpen(ordinals[0], event.currentTarget)}
    >
      <FileText size={19} aria-hidden="true" />
      <span className="source-summary">
        {first ? <><span>[{first.ordinal + 1}] {documentName(first.document_display_name, first.document_original_filename)}</span>
          <small>{formatSourceLocation(first.source_location)}{ordinals.length > 1 ? ` · 共 ${ordinals.length} 个来源` : ""}</small></>
          : <span>{ordinals.length} 个来源</span>}
      </span>
      <CaretRight size={16} aria-hidden="true" />
    </button>
  );
}

export function EvidenceDrawer({
  client,
  run,
  selectedOrdinal,
  loading,
  error,
  onSelect,
  onClose,
}: {
  client: ApiClient;
  run: ChatRun | null;
  selectedOrdinal: number;
  loading: boolean;
  error: string | null;
  onSelect: (ordinal: number) => void;
  onClose: () => void;
}) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const drawerRef = useRef<HTMLElement>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  useEffect(() => {
    closeRef.current?.focus();
    const handleKey = (event: KeyboardEvent) => {
      if (document.querySelector(".settings-dialog, .retrieval-settings-panel, .scope-popover")) return;
      if (event.key === "Escape") onCloseRef.current();
      if (event.key !== "Tab" || window.matchMedia("(min-width: 1200px)").matches) return;
      const buttons = drawerRef.current?.querySelectorAll<HTMLButtonElement>("button:not(:disabled)");
      if (!buttons?.length) return;
      const first = buttons[0];
      const last = buttons[buttons.length - 1];
      if (!drawerRef.current?.contains(document.activeElement) || (event.shiftKey && document.activeElement === first)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, []);

  useEffect(() => {
    document.getElementById(`evidence-${selectedOrdinal}`)?.scrollIntoView({
      block: "nearest",
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "auto"
        : "smooth",
    });
  }, [selectedOrdinal, run]);

  return (
    <>
      <button
        className="drawer-backdrop"
        type="button"
        aria-label="关闭证据来源"
        onClick={onClose}
      />
      <aside ref={drawerRef} className="evidence-drawer" aria-label="证据来源">
        <header className="evidence-header">
          <div>
            <h2>引用来源</h2>
          </div>
          <button
            ref={closeRef}
            className="icon-button"
            type="button"
            aria-label="关闭证据来源"
            onClick={onClose}
          >
            <X size={21} aria-hidden="true" />
          </button>
        </header>

        {loading ? (
          <div className="drawer-state" aria-live="polite">
            <span className="loading-ring" aria-hidden="true" />
            正在读取来源…
          </div>
        ) : null}
        {error ? <div className="drawer-error">{error}</div> : null}
        {run && !run.citations.length ? (
          <div className="drawer-state">这条回答没有引用来源。</div>
        ) : null}
        <div className="evidence-list">
          {run?.citations.length ? <p className="evidence-intro">本次回答的依据</p> : null}
          {run?.citations.map((citation) => (
            <EvidenceCard
              key={citation.ordinal}
              client={client}
              runId={run.run_id}
              citation={citation}
              selected={citation.ordinal === selectedOrdinal}
              onSelect={() => onSelect(citation.ordinal)}
            />
          ))}
        </div>
      </aside>
    </>
  );
}

function EvidenceCard({
  client,
  citation,
  selected,
  onSelect,
}: {
  client: ApiClient;
  runId: string;
  citation: ChatRun["citations"][number];
  selected: boolean;
  onSelect: () => void;
}) {
  const [imageFailed, setImageFailed] = useState(false);
  const [copied, setCopied] = useState(false);
  const name = documentName(
    citation.document_display_name,
    citation.document_original_filename,
  );
  let imageUrl: string | null = null;
  try {
    imageUrl = citation.asset
      ? client.resolvePublicApiUrl(citation.asset.content_url)
      : null;
  } catch {
    imageUrl = null;
  }
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(citation.quoted_text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };
  return (
    <article
      id={`evidence-${citation.ordinal}`}
      className={`evidence-card${selected ? " selected" : ""}`}
      onClick={onSelect}
    >
      <div className="evidence-number">[{citation.ordinal + 1}]</div>
      <div className="evidence-card-body">
        <div className="evidence-title-row">
          <div>
            {citation.knowledge_base_name ? <p className="citation-kb">{citation.knowledge_base_name}</p> : null}
            <h3>{name}</h3>
            <p>{formatSourceLocation(citation.source_location)}</p>
          </div>
          {citation.modality !== "text" ? <span className="modality-label">
            {citation.modality === "image"
              ? "图片"
              : citation.modality === "table"
                ? "表格"
                : "文本"}
          </span> : null}
        </div>
        {imageUrl && !imageFailed ? (
          <img
            className="evidence-image"
            src={imageUrl}
            alt={`${name} 的引用图片`}
            loading="lazy"
            onError={() => setImageFailed(true)}
          />
        ) : null}
        <blockquote>
          {citation.quoted_text || "该引用来自一项视觉证据。"}
        </blockquote>
        <button
          className="copy-button"
          type="button"
          onClick={(event) => {
            event.stopPropagation();
            void copy();
          }}
        >
          {copied ? <Check size={14} aria-hidden="true" /> : <Copy size={14} aria-hidden="true" />}
          {copied ? "已复制" : "复制引用"}
        </button>
      </div>
    </article>
  );
}

export function Notice({
  message,
  action,
  onAction,
}: {
  message: string;
  action?: string;
  onAction?: () => void;
}) {
  return (
    <div className="notice" role="alert">
      <span>{message}</span>
      {action && onAction ? (
        <button type="button" onClick={onAction}>{action}</button>
      ) : null}
    </div>
  );
}

function renderInline(
  value: string,
  citationCount: number | null,
  onCitation?: (ordinal: number, trigger: HTMLButtonElement) => void,
) {
  const output: ReactNode[] = [];
  const pattern = /\[([1-9]\d*)\]/g;
  let cursor = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(value)) !== null) {
    if (match.index > cursor) output.push(value.slice(cursor, match.index));
    const ordinal = Number(match[1]) - 1;
    const valid = citationCount === null || ordinal < citationCount;
    output.push(
      valid && onCitation ? (
        <button
          className="citation-marker"
          type="button"
          aria-label={`查看来源 ${ordinal + 1}`}
          key={`${match.index}-${ordinal}`}
          onClick={(event) => onCitation(ordinal, event.currentTarget)}
        >
          {ordinal + 1}
        </button>
      ) : (
        match[0]
      ),
    );
    cursor = pattern.lastIndex;
  }
  if (cursor < value.length) output.push(value.slice(cursor));
  return output;
}
