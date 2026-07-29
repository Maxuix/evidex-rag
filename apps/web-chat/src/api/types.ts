export type UUID = string;
export type IsoDate = string;
export type JsonMap = Record<string, unknown>;

export interface Page<T> {
  items: T[];
  next_cursor: string | null;
}

export interface KnowledgeBase {
  id: UUID;
  name: string;
  active_index_revision_id: UUID;
  retrieval_defaults: {
    strategy: "exact_vector";
    top_k: number;
    rerank: boolean;
  };
  answer_policy_defaults: {
    answer_style: "concise" | "summary";
    insufficiency_policy: "refuse" | "partial_answer";
  };
  created_at: IsoDate;
  updated_at: IsoDate;
}

export interface ChatSession {
  id: UUID;
  knowledge_base_id: UUID;
  title: string | null;
  created_at: IsoDate;
  updated_at: IsoDate;
}

export interface ChatMessage {
  id: UUID;
  session_id: UUID;
  run_id: UUID | null;
  role: "user" | "assistant";
  assistant_status: "generating" | "completed" | "failed" | null;
  content: string;
  created_at: IsoDate;
}

export interface CitationAsset {
  id: UUID;
  media_type: string;
  checksum_sha256: string;
  content_url: string;
  width: number | null;
  height: number | null;
}

export interface ChatCitation {
  ordinal: number;
  index_chunk_id: UUID | null;
  document_id: UUID;
  document_version_id: UUID;
  document_display_name: string;
  document_original_filename: string;
  quoted_text: string;
  source_location: JsonMap;
  score: number | null;
  modality: "text" | "image" | "table";
  asset: CitationAsset | null;
  matched_representations: string[];
}

export interface ChatRunError {
  code: string;
  retryable: boolean;
}

export interface ChatRun {
  run_id: UUID;
  knowledge_base_id: UUID;
  session_id: UUID;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  assistant_status: "generating" | "completed" | "failed";
  answer: string | null;
  citations: ChatCitation[];
  status_url: string;
  events_url: string;
  effective_answer_policy: {
    answer_style: "concise" | "summary";
    insufficiency_policy: "refuse" | "partial_answer";
  };
  error: ChatRunError | null;
  created_at: IsoDate;
  updated_at: IsoDate;
  completed_at: IsoDate | null;
}

export interface ChatRunCreate {
  session_id: UUID;
  knowledge_base_id: UUID;
  message: string;
  answer_policy: {
    answer_style: "concise" | "summary";
    insufficiency_policy: "refuse" | "partial_answer";
  };
  retrieval: {
    mode: "vector";
    top_k: number;
    rerank: boolean;
  };
}

export interface ChatTerminalEvent {
  run_id: UUID;
  status_url: string;
}

export interface ApiProblem {
  code?: string;
  title?: string;
  detail?: unknown;
  retryable?: boolean;
}
