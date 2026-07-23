import type { ChatSession, JsonMap } from "./api/types";

export function sessionTitle(session: ChatSession): string {
  return session.title?.trim() || `会话 ${session.id.slice(0, 8)}`;
}

export function questionTitle(question: string): string {
  const normalized = question.replace(/\s+/g, " ").trim();
  return normalized.length <= 48 ? normalized : `${normalized.slice(0, 47)}…`;
}

export function formatTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export function sessionGroup(value: string): string {
  const date = new Date(value);
  const now = new Date();
  const start = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const target = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const days = Math.round((start.getTime() - target.getTime()) / 86_400_000);
  if (days <= 0) return "今天";
  if (days === 1) return "昨天";
  if (days < 7) return "最近 7 天";
  return "更早";
}

export function documentName(
  displayName: string | null,
  originalFilename: string | null,
  documentId: string,
): string {
  return displayName?.trim()
    || originalFilename?.trim()
    || `文档 ${documentId.slice(0, 8)}`;
}

export function formatSourceLocation(value: JsonMap): string {
  const parts: string[] = [];
  const page = firstScalar(value, ["page_number", "page", "page_index"]);
  const section = firstScalar(value, ["section", "section_title", "title"]);
  const paragraph = firstScalar(value, ["paragraph", "paragraph_number"]);
  const table = firstScalar(value, ["table", "table_number"]);
  if (page !== null) parts.push(`第 ${page} 页`);
  if (section !== null) parts.push(String(section));
  if (paragraph !== null) parts.push(`第 ${paragraph} 段`);
  if (table !== null) parts.push(`表格 ${table}`);
  return parts.length ? parts.join(" · ") : "位置未标注";
}

export function citationOrdinals(content: string): number[] {
  const values = new Set<number>();
  for (const match of content.matchAll(/\[([1-9]\d*)\]/g)) {
    values.add(Number(match[1]) - 1);
  }
  return [...values].sort((left, right) => left - right);
}

function firstScalar(value: JsonMap, keys: string[]): string | number | null {
  for (const key of keys) {
    const candidate = value[key];
    if (
      (typeof candidate === "string" && candidate.trim())
      || (typeof candidate === "number" && Number.isFinite(candidate))
    ) {
      return candidate;
    }
  }
  return null;
}
