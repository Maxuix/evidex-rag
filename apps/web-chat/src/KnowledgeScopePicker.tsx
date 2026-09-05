import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { KnowledgeBase } from "./api/types";

interface Props {
  knowledgeBases: KnowledgeBase[];
  selectedIds: string[];
  disabled: boolean;
  saving: boolean;
  loading: boolean;
  hasMore: boolean;
  activeRun: boolean;
  onChange: (ids: string[]) => Promise<void>;
  onSelectAll: () => Promise<void>;
  onLoadMore: () => Promise<void>;
}

export function KnowledgeScopePicker({ knowledgeBases, selectedIds, disabled, saving, loading, hasMore,
  activeRun, onChange, onSelectAll, onLoadMore }: Props) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [initialSelection, setInitialSelection] = useState<string[]>([]);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const panelId = useId();
  const selectedNames = selectedIds.map(id => knowledgeBases.find(kb => kb.id === id)?.name ?? "已选知识库");
  const name = selectedNames[0] ?? "选择知识库";
  const label = selectedIds.length ? `搜索范围：${selectedNames.join("、")}` : "搜索范围：请选择知识库";
  const busy = disabled || saving;
  const matches = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    return knowledgeBases.filter(kb => `${kb.name} ${kb.description ?? ""}`.toLocaleLowerCase().includes(needle))
      // Keep the opening order while toggling so rows do not move under the pointer.
      .sort((a, b) => Number(initialSelection.includes(b.id)) - Number(initialSelection.includes(a.id)));
  }, [knowledgeBases, query, initialSelection]);

  const close = (restoreFocus = false) => {
    setOpen(false);
    if (restoreFocus) triggerRef.current?.focus();
  };

  useEffect(() => {
    if (!open) return;
    searchRef.current?.focus();
    const outside = (event: PointerEvent) => {
      if (event.target instanceof Node && !rootRef.current?.contains(event.target)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setOpen(false);
        triggerRef.current?.focus();
      }
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("keydown", escape);
    };
  }, [open]);

  return <div className="knowledge-scope" ref={rootRef} onBlur={event => {
    if (event.relatedTarget instanceof Node && !event.currentTarget.contains(event.relatedTarget)) close();
  }}>
    <button type="button" ref={triggerRef} className={`scope-trigger${open ? " is-open" : ""}${selectedIds.length ? "" : " is-empty"}${selectedIds.length > 1 ? " is-multiple" : ""}`}
      aria-label={label} title={label} aria-expanded={open} aria-haspopup="dialog" aria-controls={open ? panelId : undefined}
      disabled={disabled} onClick={() => {
        if (open) close();
        else { setQuery(""); setInitialSelection(selectedIds); setOpen(true); }
      }}>
      <span className="scope-trigger-kind">范围</span>
      <span className="scope-trigger-name">{name}</span>
      {selectedIds.length > 1 ? <span className="scope-count">+{selectedIds.length - 1}</span> : null}
      {selectedIds.length > 1 ? <span className="scope-mobile-count">{selectedIds.length} 个库</span> : null}
      <span className="scope-trigger-chevron" aria-hidden="true">⌄</span>
    </button>
    {open ? <div id={panelId} role="dialog" aria-label="选择知识库" className="scope-popover">
      <div className="scope-popover-heading">
        <strong>搜索范围</strong>
        <span role="status">{saving ? "保存中…" : `已选 ${selectedIds.length} 个`}</span>
      </div>
      <input ref={searchRef} className="scope-search" type="search" value={query}
        aria-label="搜索知识库" placeholder="搜索知识库名称或描述" onChange={event => setQuery(event.target.value)} />
      <div className="scope-list-actions">
        <span>知识库</span>
        <button type="button" disabled={busy || loading} onClick={() => void onSelectAll()}>全选</button>
        <button type="button" disabled={busy || !selectedIds.length} onClick={() => void onChange([])}>清空</button>
      </div>
      <div className="scope-list" role="group" aria-label="知识库列表" aria-busy={loading}>
        {matches.map(kb => <label key={kb.id} className={`scope-row${selectedIds.includes(kb.id) ? " is-selected" : ""}`} title={kb.description ? `${kb.name}\n${kb.description}` : kb.name}>
          <input type="checkbox" aria-label={kb.name} checked={selectedIds.includes(kb.id)} disabled={busy}
            onChange={event => void onChange(event.target.checked ? [...selectedIds, kb.id] : selectedIds.filter(id => id !== kb.id))} />
          <span className="scope-row-copy"><strong>{kb.name}</strong>{kb.description ? <small>{kb.description}</small> : null}</span>
        </label>)}
        {!matches.length ? <p className="scope-list-empty">{loading ? "正在加载知识库…" : hasMore ? "当前列表未找到匹配项，可加载更多知识库。" : "没有找到匹配的知识库"}</p> : null}
        {hasMore ? <button type="button" className="scope-load-more" disabled={busy || loading} onClick={() => void onLoadMore()}>{loading ? "加载中…" : "加载更多知识库"}</button> : null}
      </div>
      <div className="scope-popover-footer">
        <small>{!selectedIds.length ? "请至少选择一个知识库后发送。" : activeRun ? "修改用于下一轮，当前回答范围不变" : "仅从所选知识库查找资料"}</small>
        <button type="button" className="scope-done" onClick={() => close(true)}>完成</button>
      </div>
    </div> : null}
  </div>;
}
