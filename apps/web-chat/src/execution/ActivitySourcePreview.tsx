import { useEffect, useRef, useState } from "react";
import type { ApiClient } from "../api/client";
import type { ChatRun, DocumentChunk } from "../api/types";
import type { ActivitySource } from "./activityTypes";

export function ActivitySourcePreview({ client, run, source, trigger, onClose }: {
  client: ApiClient; run: ChatRun; source: ActivitySource; trigger: HTMLButtonElement; onClose: () => void;
}) {
  const [chunk, setChunk] = useState<DocumentChunk | null>(null);
  const [error, setError] = useState<string | null>(null);
  const panel = useRef<HTMLElement>(null);
  const close = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    let cancelled = false;
    close.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
      if (event.key !== "Tab") return;
      const items = panel.current?.querySelectorAll<HTMLElement>('button, a[href], [tabindex="0"]');
      if (!items?.length) return;
      const first = items[0], last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    window.addEventListener("keydown", onKey);
    void (async () => {
      try {
        const document = await client.getDocument(source.document_id);
        if (cancelled) return;
        if (document.kb_id !== run.knowledge_base_id || document.deleted_at || document.current_version?.id !== source.document_version_id || !run.index_revision_id || document.index?.index_revision_id !== run.index_revision_id) throw new Error("这份资料的原版本已不可用，不能用当前版本替代历史来源。");
        let cursor: string | undefined;
        const seen = new Set<string>();
        do {
          const page = await client.getDocumentChunks(source.document_id, cursor);
          if (cancelled) return;
          if (page.document_id !== source.document_id || page.document_version_id !== source.document_version_id || page.index_revision_id !== run.index_revision_id) throw new Error("资料版本已变化，无法读取这次调用的原片段。");
          const found = page.items.find(item => item.id === source.index_chunk_id && !item.excluded_at);
          if (found) { setChunk(found); return; }
          cursor = page.next_cursor ?? undefined;
          if (cursor && seen.has(cursor)) break;
          if (cursor) seen.add(cursor);
        } while (cursor);
        throw new Error("该片段已不存在或已被排除。");
      } catch (cause) { if (!cancelled) setError(cause instanceof Error ? cause.message : "暂时无法读取资料。"); }
    })();
    return () => { cancelled = true; window.removeEventListener("keydown", onKey); if (trigger.isConnected) trigger.focus(); };
  }, [client, run.run_id, run.knowledge_base_id, run.index_revision_id, source, trigger, onClose]);
  return <><button className="drawer-backdrop" aria-label="关闭检索资料" onClick={onClose} /><aside ref={panel} className="evidence-drawer activity-source-drawer" role="dialog" aria-modal="true" aria-label="检索资料">
    <header className="evidence-header"><div><p className="overline">本次调用返回的资料</p><h2>检索片段</h2></div><button ref={close} className="icon-button" type="button" aria-label="关闭检索资料" onClick={onClose}>×</button></header>
    <div className="activity-source-body"><h3>{source.title}</h3>{source.location ? <p>{source.location}</p> : null}{error ? <p role="alert">{error}</p> : chunk ? <div className="activity-source-text">{chunk.content}</div> : <p role="status">正在读取片段…</p>}</div>
  </aside></>;
}
