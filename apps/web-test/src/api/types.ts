export type UUID = string;
export type IsoDate = string;
export type JsonMap = Record<string, unknown>;

export interface RuntimeConfig {
  api_base_url: string;
}

export interface FieldViolation {
  location: Array<string | number>;
  message: string;
  error_type: string;
}

export interface ProblemDetails {
  type: string;
  title: string;
  status: number;
  detail: string;
  instance: string;
  code: string;
  trace_id: string;
  retryable: boolean;
  errors: FieldViolation[] | null;
}

export interface Page<T> {
  items: T[];
  next_cursor: string | null;
}

export type AnswerStyle = "concise" | "summary";
export type InsufficiencyPolicy = "refuse" | "partial_answer";
export type ChunkingPreset =
  | "structural_balanced_v2"
  | "semantic_balanced_v1";
export type ParsingPreset = "text_local_v1" | "multimodal_local_v1";

export interface KnowledgeBase {
  id: UUID;
  name: string;
  source_change_seq: number;
  active_index_revision_id: UUID;
  embedding_space_id: UUID;
  parsing: {
    preset: ParsingPreset;
    profile: "unstructured_local_v1" | "unstructured_multimodal_local_v1" | "unstructured_multimodal_local_v2";
  };
  chunking: {
    preset: ChunkingPreset | "legacy_incompatible";
    profile:
      | "unstructured_by_title_token_v2"
      | "semantic_breakpoint_v1"
      | "unstructured_title_semantic_qwen_v1";
  };
  retrieval_defaults: {
    strategy: "exact_vector";
    top_k: number;
    rerank: boolean;
  };
  answer_policy_defaults: {
    answer_style: AnswerStyle;
    insufficiency_policy: InsufficiencyPolicy;
  };
  provisioned_at: IsoDate;
  created_at: IsoDate;
  updated_at: IsoDate;
}

export interface DocumentVersion {
  id: UUID;
  version_number: number;
  source_status: "available" | "unavailable" | "deleted";
  checksum_sha256: string;
  original_filename: string;
  media_type: string;
  size_bytes: number;
  created_at: IsoDate;
}

export interface DocumentRecord {
  id: UUID;
  kb_id: UUID;
  display_name: string;
  current_version: DocumentVersion | null;
  deleted_at: IsoDate | null;
  created_at: IsoDate;
  updated_at: IsoDate;
}

export interface DocumentIndexSummary {
  indexed_document_version_id: UUID;
  index_revision_id: UUID;
  build_status: "queued" | "processing" | "ready" | "failed";
  serving_status: "candidate" | "serving" | "retired";
  unit_count: number | null;
  asset_count: number | null;
  representation_count: number | null;
}

export interface DocumentDetail extends DocumentRecord {
  index: DocumentIndexSummary | null;
}

export interface DocumentUpload {
  document: DocumentRecord;
  document_version_id: UUID;
  source_change_id: UUID;
  source_change_seq: number;
  indexed_document_version_id: UUID;
  index_revision_id: UUID;
  job_id: UUID;
  job_status: "queued";
}

export interface IndexingJob {
  job_id: UUID;
  kb_id: UUID;
  document_id: UUID;
  document_version_id: UUID;
  indexed_document_version_id: UUID;
  index_revision_id: UUID;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  phase: string;
  attempt: number;
  build_status: "queued" | "processing" | "ready" | "failed";
  serving_status: "candidate" | "serving" | "retired";
  claimed_at: IsoDate | null;
  heartbeat_at: IsoDate | null;
  next_attempt_at: IsoDate | null;
  error: { code: string; detail: JsonMap } | null;
  can_retry: boolean;
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

export interface EffectiveAnswerPolicy {
  grounding_policy: "evidence_only";
  answer_style: AnswerStyle;
  insufficiency_policy: InsufficiencyPolicy;
  citation_required: true;
  citation_granularity: "claim_level";
  answer_task: "answer";
  policy_version: "p1";
}

export interface ChatRunError {
  code: string;
  detail: JsonMap;
  retryable: boolean;
}

export interface ChatCitation {
  ordinal: number;
  index_chunk_id: UUID | null;
  document_id: UUID;
  document_version_id: UUID;
  quoted_text: string;
  source_location: JsonMap;
  score: number | null;
  modality: "text" | "image" | "table";
  asset: EvidenceAsset | null;
  matched_representations: string[];
}

export interface ChatRun {
  run_id: UUID;
  knowledge_base_id: UUID;
  session_id: UUID;
  user_message_id: UUID;
  assistant_message_id: UUID;
  index_revision_id: UUID;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  assistant_status: "generating" | "completed" | "failed";
  answer: string | null;
  citations: ChatCitation[];
  status_url: string;
  events_url: string;
  effective_answer_policy: EffectiveAnswerPolicy;
  retrieval: {
    strategy: "exact_vector";
    top_k: number;
    rerank: boolean;
  };
  query_context: {
    strategy: "recent_completed_turns_v1";
    status: "pending" | "original" | "contextualized" | "needs_clarification";
    history_turn_count: number;
    history_token_count: number;
    history_truncated: boolean;
    standalone_query: string | null;
    rewrite_source: "original" | "model" | "repair" | "fallback" | null;
  };
  attempt: number;
  error: ChatRunError | null;
  usage: JsonMap | null;
  timing: JsonMap | null;
  created_at: IsoDate;
  updated_at: IsoDate;
  completed_at: IsoDate | null;
}

export interface ChatRunCreate {
  session_id: UUID;
  knowledge_base_id: UUID;
  message: string;
  answer_policy: {
    answer_style: AnswerStyle;
    insufficiency_policy: InsufficiencyPolicy;
  };
  retrieval: {
    mode: "vector";
    top_k: number;
    rerank?: boolean;
  };
}

export interface ChatAnswerCompletedEvent {
  run_id: UUID;
  message_id: UUID;
  answer: string;
  citations: ChatCitation[];
  effective_answer_policy: EffectiveAnswerPolicy;
  status_url: string;
}

export interface ChatRunFailedEvent {
  run_id: UUID;
  status: "failed" | "cancelled";
  error: ChatRunError;
  effective_answer_policy: EffectiveAnswerPolicy;
  status_url: string;
}

export interface RetrievalQueryPlan {
  workspace_id: UUID;
  knowledge_base_id: UUID;
  strategy: "exact_vector";
  top_k: number;
  revision_selector: "active";
  current_document_version_only: true;
  build_status: "ready";
  serving_status: "serving";
  distance_metric: "cosine";
  candidate_count: number | null;
  ef_search: number | null;
  iterative_scan: "disabled";
  rerank: boolean;
}

export interface Evidence {
  rank: number;
  index_chunk_id: UUID;
  indexed_document_version_id: UUID;
  document_id: UUID;
  document_version_id: UUID;
  index_revision_id: UUID;
  ordinal: number;
  text: string;
  source_location: JsonMap;
  hierarchy: JsonMap;
  source_metadata: JsonMap;
  score: number;
  score_kind: "cosine_similarity" | "hybrid_rerank" | "reciprocal_rank_fusion";
  vector_similarity: number | null;
  lexical_score: number;
  lexical_coverage: number;
  modality: "text" | "image" | "table";
  asset: EvidenceAsset | null;
  evidence_group_key: string | null;
  matched_representations: string[];
  text_space_rank: number | null;
  cross_modal_rank: number | null;
  fusion_score: number | null;
}

export interface EvidenceAsset {
  id: UUID;
  media_type: string;
  checksum_sha256: string;
  content_url: string;
  width: number | null;
  height: number | null;
}

export interface EvidencePack {
  knowledge_base_id: UUID;
  index_revision_id: UUID;
  strategy: "exact_vector";
  evidence: Evidence[];
  debug: {
    query_plan: RetrievalQueryPlan;
    resolved_active_revision_id: UUID;
    result_count: number;
  } | null;
}
