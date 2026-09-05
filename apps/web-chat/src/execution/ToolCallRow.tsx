import { useState } from "react";
import type { ChatRun } from "../api/types";
import { ACTIVE_STATUSES, type ActivitySource } from "./activityTypes";
import { duration, inputText, labels, resultText, statusLabels, type TimelineStep } from "./activityView";

export function ToolCallRow({ step, steps, live, elapsedMs, run, attempt, onCitation, onSource }: {
  step: TimelineStep; steps: TimelineStep[]; live: boolean; elapsedMs: number; run: ChatRun; attempt: number;
  onCitation: (ordinal: number, trigger: HTMLButtonElement) => void;
  onSource: (source: ActivitySource, trigger: HTMLButtonElement) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const active = ACTIVE_STATUSES.has(step.status);
  const input = inputText(step, steps);
  const elapsed = step.legacy_duration_ms ?? (step.started_offset_ms !== null && (step.ended_offset_ms !== null || (active && live))
    ? Math.max(0, (step.ended_offset_ms ?? elapsedMs) - step.started_offset_ms) : null);
  const status = active && !live ? "未确认完成" : statusLabels[step.status];
  const detailId = `activity-${run.run_id}-${attempt}-${step.step_id}-${step.round ?? "system"}`;
  if (step.kind !== "tool") return (
    <li className={`activity-system ${active && live ? "is-active" : ""}`}>
      <span className="activity-dot" aria-hidden="true" />
      <div><span>{labels[step.name] ?? step.name}</span><small>{resultText(step, live)}</small></div>
      <span className="activity-system-status">{active ? status : elapsed !== null ? duration(elapsed) : status}</span>
    </li>
  );
  return (
    <li className={`activity-tool ${step.status}${active && live ? " is-active" : ""}`}>
      <button className="activity-call-toggle" type="button" aria-expanded={expanded} aria-controls={detailId} onClick={() => setExpanded(value => !value)}>
        <span className="activity-dot" aria-hidden="true" />
        <span className="activity-call-content">
          <span className="activity-call-heading"><strong>{labels[step.name] ?? step.name}</strong><code>{step.name}</code></span>
          {input ? <span className="activity-input-preview">{input.length > 140 ? `${input.slice(0, 140)}…` : input}</span> : null}
          <span className="activity-result-preview">{resultText(step, live)}</span>
        </span>
        <span className="activity-call-meta"><span className="activity-status">{status}</span>{elapsed !== null ? <time>{duration(elapsed)}</time> : null}<span className="activity-disclosure">{expanded ? "收起" : "展开"}</span></span>
      </button>
      {expanded ? <div className="activity-details" id={detailId}>
        {step.legacy ? <p className="activity-note">历史记录未保存输入和轮次；耗时仅在原记录提供时展示。</p> : <dl className="activity-parameters">
          {step.queries.length ? <><dt>查询{step.queries.length > 1 ? `（${step.queries.length} 个）` : ""}</dt><dd>{step.queries.map((query, index) => <p key={index}>{query}</p>)}</dd></> : null}
          {step.top_k !== null ? <><dt>每个查询最多返回</dt><dd>{step.top_k} 条</dd></> : null}
          {step.refs.length ? <><dt>上下文锚点</dt><dd>{input}</dd></> : null}
          {step.include_outline !== null ? <><dt>包含大纲</dt><dd>{step.include_outline ? "是" : "否"}</dd></> : null}
          {step.expression ? <><dt>表达式</dt><dd><code>{step.expression}</code></dd></> : null}
          {step.result_value !== null ? <><dt>计算结果</dt><dd><strong>{step.result_value}</strong></dd></> : null}
        </dl>}
        {step.scope_results?.length ? <div className="activity-scope-results"><h4>各知识库结果</h4><ul>{step.scope_results.map((scope, index) => <li key={`${scope.knowledge_base_id}-${index}`}>
          <strong>{scope.name}</strong> · {scope.status === "ok" || scope.status === "admitted" ? "已检索" : scope.status === "empty" || scope.status === "no_evidence" ? "未命中" : scope.status === "chat_revision_mismatch" ? "索引或图版本已变化" : scope.status === "index_unavailable" ? "索引不可用" : scope.status}
          {scope.query ? <p>{scope.query}</p> : null}
          {scope.retrieved_count !== null ? <small>命中 {scope.retrieved_count} · 准入 {scope.admitted_count ?? 0} · 展示 {scope.displayed_count ?? 0}{scope.omitted_count ? ` · 预算未展示 ${scope.omitted_count}` : ""}</small> : null}
        </li>)}</ul></div> : null}
        {step.sources.length ? <div className="activity-sources"><h4>{step.name === "list_documents" ? "文档清单" : "返回的资料"}</h4><ul>{step.sources.map((source, index) => {
          const citation = attempt === run.attempt && source.index_chunk_id ? run.citations.find(item =>
            item.index_chunk_id === source.index_chunk_id && item.document_id === source.document_id
            && item.document_version_id === source.document_version_id
            && (!source.knowledge_base_id || item.knowledge_base_id === source.knowledge_base_id)
            && (!source.index_revision_id || item.index_revision_id === source.index_revision_id)) : undefined;
          return <li key={`${source.ref ?? source.document_id}-${index}`}>
            <div><strong>{source.knowledge_base_name ? `${source.knowledge_base_name} · ` : ""}{source.title}</strong>{source.location ? <span>{source.location}</span> : null}</div>
            {citation ? <button type="button" onClick={event => onCitation(citation.ordinal, event.currentTarget)}>引用来源 {citation.ordinal + 1}</button>
              : source.index_chunk_id ? <button type="button" onClick={event => onSource(source, event.currentTarget)}>查看片段</button> : null}
          </li>;
        })}</ul>{step.name !== "list_documents" ? <p className="activity-note">检索结果可能重复；最终采用的资料会标记为引用来源。</p> : null}</div> : null}
        {step.details_truncated ? <p className="activity-note">当前详情已裁剪。实时预览会在回答结束后以保存的记录校准；记录超限的详情可能仍不完整。</p> : null}
        <details className="activity-diagnostics"><summary>调用信息</summary><dl><dt>工具</dt><dd>{step.name}</dd><dt>状态</dt><dd>{status}</dd>{step.result_code ? <><dt>结果码</dt><dd>{step.result_code}</dd></> : null}</dl></details>
      </div> : null}
    </li>
  );
}
