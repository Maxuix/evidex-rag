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
  displayName: string,
  originalFilename: string,
): string {
  return displayName.trim() || originalFilename.trim();
}

export function formatSourceLocation(value: JsonMap): string {
  const type = typeof value.surface_type === "string"
    ? value.surface_type
    : "logical";
  const start = finiteNumber(value.surface_start);
  const end = finiteNumber(value.surface_end);
  const labels = [
    ...(typeof value.surface_label === "string" ? [value.surface_label] : []),
    ...(Array.isArray(value.surface_labels)
      ? value.surface_labels.filter((item): item is string => typeof item === "string")
      : []),
  ];
  const typeLabel = {
    page: "页",
    slide: "幻灯片",
    sheet: "工作表",
    logical: "逻辑位置",
  }[type] || type;
  const range = start === null
    ? null
    : start === end || end === null
      ? `${typeLabel} ${start}`
      : `${typeLabel} ${start}–${end}`;
  return [...new Set(labels.map((item) => item.trim()).filter(Boolean)), range]
    .filter((item): item is string => Boolean(item))
    .join(" · ") || "逻辑位置";
}

export function citationOrdinals(content: string): number[] {
  const values = new Set<number>();
  for (const match of content.matchAll(/\[([1-9]\d*)\]/g)) {
    values.add(Number(match[1]) - 1);
  }
  return [...values].sort((left, right) => left - right);
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
