const KNOWLEDGE_BASE_KEY = "rag-kb.user-chat.knowledge-base";
const SESSION_PREFIX = "rag-kb.user-chat.session.";
const SIDEBAR_KEY = "rag-kb.user-chat.sidebar-collapsed";

export function readKnowledgeBaseId(): string | null {
  return safeRead(KNOWLEDGE_BASE_KEY);
}

export function storeKnowledgeBaseId(value: string | null): void {
  safeWrite(KNOWLEDGE_BASE_KEY, value);
}

export function readSessionId(knowledgeBaseId: string): string | null {
  return safeRead(`${SESSION_PREFIX}${knowledgeBaseId}`);
}

export function storeSessionId(
  knowledgeBaseId: string,
  value: string | null,
): void {
  safeWrite(`${SESSION_PREFIX}${knowledgeBaseId}`, value);
}

export function readSidebarCollapsed(): boolean {
  return safeRead(SIDEBAR_KEY) === "true";
}

export function storeSidebarCollapsed(value: boolean): void {
  safeWrite(SIDEBAR_KEY, String(value));
}

function safeRead(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeWrite(key: string, value: string | null): void {
  try {
    if (value === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch {
    // Storage is a preference only; the product remains usable without it.
  }
}
