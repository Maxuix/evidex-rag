export type ChatViewScope = {
  generation: number;
  knowledgeBaseId: string;
  sessionId: string | null;
};

export type ManagementKbScope = {
  generation: number;
  knowledgeBaseId: string;
};

export type DocumentScope = ManagementKbScope & {
  documentId: string;
};

export function isChatViewScopeCurrent(
  token: ChatViewScope,
  current: ChatViewScope,
): boolean {
  return token.generation === current.generation
    && token.knowledgeBaseId === current.knowledgeBaseId
    && token.sessionId === current.sessionId;
}

export function isManagementKbScopeCurrent(
  token: ManagementKbScope,
  current: ManagementKbScope,
): boolean {
  return token.generation === current.generation
    && token.knowledgeBaseId === current.knowledgeBaseId;
}

export function isDocumentScopeCurrent(
  token: DocumentScope,
  current: DocumentScope,
): boolean {
  return isManagementKbScopeCurrent(token, current)
    && token.documentId === current.documentId;
}

export function isRequestSequenceCurrent(
  token: number,
  current: number,
): boolean {
  return token === current;
}
