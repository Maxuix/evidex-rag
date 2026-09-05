import type { ChatRun } from "../api/types";
import { ACTIVE_STATUSES, type ActivityStep } from "./activityTypes";
export type TimelineStep = ActivityStep & { legacy?: boolean; legacy_duration_ms?: number };
export const labels: Record<string, string> = {
  semantic_search: "语义检索", keyword_search: "关键词检索", read_chunk_context: "阅读上下文",
  list_documents: "查看文档目录", search_graph_relations: "图谱关系检索", calculate: "计算",
  unknown: "未识别的工具", search_knowledge_base: "知识库检索", model_round: "模型处理",
  load_context: "读取问题与会话", prepare_visuals: "准备图片与表格素材", resolve_citations: "解析引用",
  persist_result: "保存回答", close_search: "结束检索", token_wrap_up: "整理已有结果",
  verifier: "历史核验步骤", submit_answer: "提交回答（历史）",
};
export const statusLabels: Record<string, string> = {
  pending: "待执行", running: "进行中", processing: "合并中", succeeded: "已完成",
  failed: "执行失败", rejected: "未执行", cancelled: "已停止",
};
const codes: Record<string, string> = {
  no_evidence: "没有新增的图谱证据", not_ready: "图谱尚未就绪", timeout: "图谱检索超时",
  unavailable: "图谱服务暂不可用", rejected: "调用未执行", cancelled: "执行已停止",
  invalid_tool_arguments: "参数未通过检查，调用未执行", tool_not_available: "当前轮未提供此工具",
  keyword_unavailable: "关键词索引当前不可用", invalid_arguments: "参数未通过检查",
  chat_provider_unavailable: "服务暂不可用", CHAT_PROVIDER_UNAVAILABLE: "服务暂不可用",
  tool_execution_failed: "工具执行失败", execution_failed: "执行发生错误",
  deadline_exceeded: "本次回答运行超时", CHAT_PIPELINE_DEADLINE_EXCEEDED: "本次回答运行超时",
  no_new_evidence: "连续两轮没有新增证据，结束检索", token_budget: "已达到 token 预算，使用已有资料整理回答",
};
export function duration(ms: number): string { return ms < 1000 ? `${Math.round(ms)} 毫秒` : `${(ms / 1000).toFixed(1)} 秒`; }
export function inputText(step: TimelineStep, steps: TimelineStep[]): string {
  if (step.queries.length) return step.queries.join("；");
  if (step.expression) return step.expression;
  if (step.refs.length) return step.refs.map(ref => steps.flatMap(item => item.sources).find(source => source.ref === ref)?.title ?? ref).join("、");
  if (step.include_outline !== null) return step.include_outline ? "文档清单与章节大纲" : "知识库文档清单";
  return "";
}
export function resultText(step: TimelineStep, live: boolean): string {
  if (ACTIVE_STATUSES.has(step.status)) {
    if (!live) return "尚未收到该步骤的完成确认";
    if (step.status === "processing") return `${step.returned_count !== null ? `已返回 ${step.returned_count} 条，` : ""}正在合并资料`;
    return step.status === "pending" ? "等待执行" : "正在执行";
  }
  if (["failed", "rejected", "cancelled"].includes(step.status)) return codes[step.result_code ?? ""] ?? (step.result_code ? `未完成 · ${step.result_code}` : statusLabels[step.status]);
  if (step.kind === "model") return step.result_code === "final_text" ? "已返回回答文本" : `已返回 ${step.returned_count ?? 0} 个工具调用`;
  if (codes[step.result_code ?? ""]) return codes[step.result_code!];
  if (step.result_value !== null) return `结果 ${step.result_value}`;
  if (step.document_count !== null) return `列出 ${step.document_count} 份文档`;
  if (step.citation_count !== null) return `采用 ${step.citation_count} 条引用来源`;
  if (step.image_count !== null) return `已准备 ${step.image_count} 张图片`;
  const parts = [];
  if (step.returned_count !== null) parts.push(step.returned_count ? `返回 ${step.returned_count} 条资料` : "没有返回匹配资料");
  if (step.new_evidence_count !== null) parts.push(`合并新增 ${step.new_evidence_count} 条`);
  if (step.path_count !== null) parts.push(`${step.path_count} 条关系路径`);
  if (step.hop1_count !== null) parts.push(`一跳 ${step.hop1_count} / 二跳 ${step.hop2_count ?? 0} / 三跳 ${step.hop3_count ?? 0}`);
  if (parts.length) return parts.join(" · ");
  return step.legacy && step.kind === "tool" ? "历史记录未保存结果详情" : "已完成";
}
export function legacySteps(run: ChatRun): TimelineStep[] {
  return (run.agent.trace?.events ?? []).map((event, index) => {
    const system = ["protocol", "verifier", "submit_answer"].includes(event.tool);
    const search = ["search_knowledge_base", "semantic_search", "keyword_search", "read_chunk_context", "search_graph_relations"].includes(event.tool);
    return {
      step_id: `legacy_${index}`, ordinal: index + 1, seq: index + 1, round: null,
      kind: system ? "system" : "tool", name: event.tool === "protocol" ? "resolve_citations" : event.tool,
      status: ["rejected", "refused"].includes(event.status) ? "rejected" : "succeeded",
      started_offset_ms: null, ended_offset_ms: null,
      queries: [], refs: [], expression: null, include_outline: null, top_k: null,
      returned_count: search ? event.count : null, new_evidence_count: event.new_evidence_count ?? null,
      document_count: null, citation_count: event.tool === "protocol" ? run.citations.length : null,
      image_count: null, hop1_count: event.hop1_count ?? null, hop2_count: event.hop2_count ?? null, hop3_count: event.hop3_count ?? null, path_count: event.path_count ?? null, result_value: null,
      result_code: event.route_result_code ?? null, sources: [], details_truncated: false,
      legacy: true, legacy_duration_ms: event.duration_ms,
    };
  });
}
