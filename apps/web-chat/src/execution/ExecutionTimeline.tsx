import { useCallback, useEffect, useRef, useState } from "react";
import type { ApiClient } from "../api/client";
import type { ChatProgressSnapshot, ChatRun } from "../api/types";
import type { ActivityState } from "./activityState";
import { ACTIVE_STATUSES, type ActivitySource } from "./activityTypes";
import { duration, labels, legacySteps, type TimelineStep } from "./activityView";
import { ToolCallRow } from "./ToolCallRow";
import { ActivitySourcePreview } from "./ActivitySourcePreview";

export function ExecutionTimeline({ run, activity, progress, client, onCitation }: {
  run: ChatRun; activity: ActivityState | null; progress: { snapshot: ChatProgressSnapshot | null; mode: string } | null;
  client: ApiClient; onCitation: (ordinal: number, trigger: HTMLButtonElement) => void;
}) {
  const terminal = ["completed", "failed", "cancelled"].includes(run.status);
  const [expanded, setExpanded] = useState(!terminal);
  const [now, setNow] = useState(Date.now());
  const [following, setFollowing] = useState(!terminal);
  const [selection, setSelection] = useState<{ source: ActivitySource; trigger: HTMLButtonElement } | null>(null);
  const closeSource = useCallback(() => setSelection(null), []);
  const body = useRef<HTMLDivElement>(null);
  const scope = activity?.runId === run.run_id ? activity : null;
  const persisted = run.activities ?? [];
  const currentAttempt = terminal ? run.attempt : Math.max(run.attempt, ...Object.keys(scope?.attempts ?? {}).map(Number));
  const saved = persisted.find(item => item.attempt === currentAttempt);
  const live = scope?.attempts[currentAttempt];
  const legacy = !saved && !live && terminal;
  const steps: TimelineStep[] = saved?.steps ?? (live ? Object.values(live.steps).sort((a, b) => a.ordinal - b.ordinal) : legacy ? legacySteps(run) : []);
  const disconnected = !terminal && (scope && scope.mode !== "idle" ? scope.mode === "disconnected" : progress?.mode === "disconnected");
  const elapsedMs = saved?.elapsed_ms ?? (live ? live.elapsedMs + (!terminal && !disconnected ? Math.max(0, now - live.receivedAt) : 0) : run.agent.trace?.diagnostics?.elapsed_ms ?? 0);
  const activeTools = steps.filter(step => step.kind === "tool" && ACTIVE_STATUSES.has(step.status));
  const toolCount = steps.filter(step => step.kind === "tool").length;
  const omitted = saved?.omitted_step_count ?? live?.omitted ?? 0;
  const latest = [...steps].reverse().find(step => ACTIVE_STATUSES.has(step.status));
  const activeText = activeTools.length > 1 ? `正在并行执行 ${activeTools.length} 个工具` : latest ? `${labels[latest.name] ?? latest.name}${latest.round ? ` · 第 ${latest.round} 轮` : ""}` : "等待执行进度";
  const stateText = run.status === "completed" ? "已完成" : run.status === "failed" ? "未完成" : run.status === "cancelled" ? "已停止" : disconnected ? "连接中断" : run.status === "queued" ? "等待执行" : activeText;
  const sequence = steps.map(step => `${step.step_id}:${step.seq}`).join("|");
  useEffect(() => { if (terminal) return; const timer = window.setInterval(() => setNow(Date.now()), 1000); return () => window.clearInterval(timer); }, [terminal]);
  useEffect(() => { if (following && body.current) body.current.scrollTop = body.current.scrollHeight; }, [sequence, following, expanded]);
  const source = (value: ActivitySource, trigger: HTMLButtonElement) => setSelection({ source: value, trigger });
  const renderSteps = (items: TimelineStep[], isLive: boolean, attempt: number, attemptElapsed = elapsedMs) => {
    const groups: { key: string; round: number | null; steps: TimelineStep[] }[] = [];
    for (const step of items) {
      const previous = groups[groups.length - 1];
      if (previous && previous.round === step.round) previous.steps.push(step);
      else groups.push({ key: step.step_id, round: step.round, steps: [step] });
    }
    return groups.map(group => <section className="activity-round" key={group.key} aria-label={group.round ? `第 ${group.round} 轮` : "执行记录"}>
      {group.round ? <div className="activity-round-label">第 {group.round} 轮{group.steps.filter(step => step.kind === "tool").length > 1 ? <span>{group.steps.filter(step => step.kind === "tool").length} 个工具并行</span> : null}</div> : null}
      <ol>{group.steps.map(step => <ToolCallRow key={`${attempt}:${step.step_id}`} step={step} steps={items} live={isLive} elapsedMs={attemptElapsed} run={run} attempt={attempt} onCitation={onCitation} onSource={source} />)}</ol>
    </section>);
  };
  const priorAttempts = new Map(persisted.filter(item => item.attempt < currentAttempt).map(item => [item.attempt, { steps: item.steps, omitted: item.omitted_step_count, elapsedMs: item.elapsed_ms }]));
  for (const [attempt, record] of Object.entries(scope?.attempts ?? {})) if (+attempt < currentAttempt && !priorAttempts.has(+attempt)) priorAttempts.set(+attempt, { steps: Object.values(record.steps).sort((a, b) => a.ordinal - b.ordinal), omitted: record.omitted, elapsedMs: record.elapsedMs });
  return <section className={`execution-timeline${terminal ? " is-terminal" : ""}`} aria-label="回答执行过程">
    <button className="activity-header" type="button" aria-expanded={expanded} aria-controls={`timeline-${run.run_id}`} onClick={() => setExpanded(value => !value)}>
      <span><strong>回答过程</strong><span className="activity-header-state">{stateText}</span></span>
      <span className="activity-header-meta">{toolCount ? `${toolCount} 次${terminal && saved && !omitted ? "" : "已记录的"}工具调用` : ""}{run.citations.length ? ` · ${run.citations.length} 条引用` : ""}{elapsedMs > 0 ? ` · ${duration(elapsedMs)}` : ""}<span>{expanded ? "收起" : "展开"}</span></span>
    </button>
    <span className="sr-only" role="status" aria-live="polite">{disconnected ? "实时连接已中断" : stateText}</span>
    {expanded ? <div id={`timeline-${run.run_id}`}>
      {disconnected ? <p className="activity-notice" role="status">实时连接中断，正在查询后台状态；已保留最后确认的记录。</p> : null}
      {!terminal && run.live_progress_available === false ? <p className="activity-notice">服务端未开启实时进度，回答结束后可查看保存的执行记录。</p> : null}
      {run.activity_unavailable || scope?.invalid ? <p className="activity-notice">部分执行记录的格式暂不支持，回答仍可正常查看。</p> : null}
      {live?.incomplete && !saved && !terminal && !disconnected ? <p className="activity-note activity-inset">运行中的记录可能不完整，完成后将读取保存的轨迹。</p> : null}
      {omitted > 0 ? <p className="activity-notice">记录已裁剪，省略 {omitted} 个较早步骤。</p> : null}
      {legacy && steps.length ? <p className="activity-note activity-inset">历史记录按保存顺序展示；输入、轮次和部分耗时未保存，较早事件可能不完整。</p> : null}
      <div className="activity-scroll" ref={body} onScroll={event => { const el = event.currentTarget; setFollowing(el.scrollHeight - el.clientHeight - el.scrollTop < 32); }}>
        {[...priorAttempts].sort(([a], [b]) => a - b).map(([attempt, items]) => <details className="activity-attempt" key={attempt}><summary>第 {attempt} 次尝试 · 已结束</summary>{items.omitted ? <p className="activity-note">记录已裁剪，省略 {items.omitted} 个较早步骤。</p> : null}{renderSteps(items.steps, false, attempt, items.elapsedMs)}</details>)}
        {currentAttempt > 1 ? <p className="activity-attempt-label">第 {currentAttempt} 次尝试</p> : null}
        {steps.length ? renderSteps(steps, !terminal && !disconnected, currentAttempt) : <p className="activity-empty">{terminal ? "这次回答没有保存逐步执行记录。" : run.status === "queued" ? "问题已提交，等待后台开始执行。" : progress?.snapshot ? "已收到阶段进度，等待工具调用记录。" : "等待后台发送执行进度…"}</p>}
      </div>
      {!following && !terminal ? <button className="activity-follow" type="button" onClick={() => setFollowing(true)}>回到最新进度</button> : null}
      {terminal ? <details className="activity-run-details"><summary>运行信息</summary><dl><dt>模型</dt><dd>{run.model.profile_name || run.model.model}</dd><dt>模型轮次</dt><dd>{run.agent.trace?.usage.model_rounds ?? "未记录"}</dd><dt>查询执行数</dt><dd>{run.agent.trace?.usage.retrieval_queries ?? run.agent.trace?.usage.retrieval_calls ?? "未记录"}</dd><dt>结束原因</dt><dd>{run.agent.trace?.diagnostics?.stop_reason ?? run.error?.code ?? run.status}</dd></dl></details> : null}
    </div> : null}
    {selection ? <ActivitySourcePreview key={`${run.run_id}:${selection.source.index_chunk_id}`} client={client} run={run} source={selection.source} trigger={selection.trigger} onClose={closeSource} /> : null}
  </section>;
}
