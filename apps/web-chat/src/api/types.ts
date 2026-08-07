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

export interface ChatRunRetrieval {
  profile_version: "exact_vector_v1" | "hybrid_fts_rrf_v1";
  strategy: "exact_vector" | "hybrid";
  top_k: number;
  rerank: boolean;
  dense_candidate_count: number;
  lexical_candidate_count: number;
  cross_modal_candidate_count: number;
  lexical_analyzer_version: string | null;
  lexical_query_version: string | null;
  rrf_k: number;
  dense_weight_micros: number;
  lexical_weight_micros: number;
  cross_modal_weight_micros: number;
  min_cosine_similarity: number;
  min_rerank_score: number;
  cross_modal_min_cosine_similarity: number;
  rerank_vector_weight: number;
  rerank_lexical_weight: number;
  mmr_lambda: number;
}

export interface RetrievalCapability {
  mode: "vector" | "hybrid";
  strategy: "exact_vector" | "hybrid";
  profile_version: "exact_vector_v1" | "hybrid_fts_rrf_v1";
  enabled: boolean;
}

export interface RetrievalCapabilities {
  default_mode: "vector";
  modes: RetrievalCapability[];
}

export type ChatWorkflowMode = "simple" | "agent" | "auto";

export interface ChatWorkflowCapability {
  mode: ChatWorkflowMode;
  enabled: boolean;
}

export interface ChatWorkflowCapabilities {
  version: "chat_workflow_v1";
  default_mode: "simple";
  modes: ChatWorkflowCapability[];
}

export interface ChatResearchResult {
  version: "research_result_v1";
  status:
    | "sufficient"
    | "partial"
    | "no_evidence"
    | "conflict"
    | "premise_unsupported";
  selected_evidence_keys: string[];
  aspects: Array<{
    aspect: string;
    status: "supported" | "partial" | "missing" | "conflict";
    evidence_keys: string[];
  }>;
  covered_aspects: string[];
  missing_aspects: string[];
  conflicts: string[];
  termination_reason:
    | "sufficient"
    | "partial"
    | "no_evidence"
    | "no_progress"
    | "budget_exhausted"
    | "conflict_unresolved"
    | "premise_unsupported";
}

export interface ChatSearchTrace {
  version: "search_trace_v1";
  steps: Array<{
    observation_id: string;
    objective: string;
    queries: string[];
    based_on_observation_ids: string[];
    result: "evidence_found" | "no_evidence" | "verification_gap";
    new_evidence_count: number;
  }>;
  decision_rounds: number;
  retrieval_calls: number;
  verifier_calls: number;
  evidence_count: number;
}

export interface ChatWorkflow {
  version: "chat_workflow_v1";
  requested_mode: ChatWorkflowMode;
  resolved_mode: "pending" | "simple" | "agent";
  route_status: "not_applicable" | "pending" | "resolved" | "fallback";
  route_reason_codes: Array<
    | "single_lookup"
    | "direct_summary"
    | "multi_view_required"
    | "multi_hop_required"
    | "evidence_uncertain"
    | "router_invalid"
    | "router_unavailable"
  >;
  research_result: ChatResearchResult | null;
  search_trace: ChatSearchTrace | null;
}

export interface ChatRun {
  run_id: UUID;
  knowledge_base_id: UUID;
  session_id: UUID;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  assistant_status: "generating" | "completed" | "failed";
  attempt: number;
  answer: string | null;
  citations: ChatCitation[];
  status_url: string;
  events_url: string;
  effective_answer_policy: {
    answer_style: "concise" | "summary";
    insufficiency_policy: "refuse" | "partial_answer";
  };
  workflow: ChatWorkflow;
  retrieval: ChatRunRetrieval;
  model: {
    profile_revision_id: UUID | null;
    profile_name: string | null;
    provider_name: string;
    model: string;
    revision: number | null;
    temperature: number;
    top_p: number | null;
    sampling_top_k: number | null;
    max_output_tokens: number;
    reasoning_effort: "off" | "low" | "medium" | "high";
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
  workflow: {
    mode: ChatWorkflowMode;
  };
  retrieval: {
    mode: "vector" | "hybrid";
    top_k: number;
    rerank: boolean;
  };
  model_profile_revision_id?: UUID | null;
}

export type ModelProviderProtocol = "openai_compatible" | "tongyi_multimodal";
export type ModelKind = "chat" | "text_embedding" | "multimodal_embedding";
export type ModelValidationStatus = "unverified" | "valid" | "invalid";

export interface ModelProvider {
  id: UUID;
  revision_id: UUID;
  revision: number;
  name: string;
  protocol: ModelProviderProtocol;
  base_url: string;
  timeout_seconds: number;
  max_retries: number;
  max_concurrency: number;
  enabled: boolean;
  api_key_configured: boolean;
  configuration_fingerprint: string;
  created_at: IsoDate;
  updated_at: IsoDate;
}

export interface ChatModelParameters {
  type: "chat";
  temperature: number;
  top_p: number | null;
  sampling_top_k: number | null;
  max_output_tokens: number;
  reasoning_effort: "off" | "low" | "medium" | "high";
  structured_output_mode: "json_object" | "json_schema";
  vision_enabled: boolean;
}

export interface EmbeddingModelParameters {
  type: "embedding";
  dimension: 768 | 1024;
  max_batch_size: number;
  distance_metric: "cosine";
  vector_data_type: "float32";
  normalization: "l2";
}

export interface ModelProfile {
  id: UUID;
  revision_id: UUID;
  revision: number;
  provider_id: UUID;
  provider_revision_id: UUID;
  name: string;
  kind: ModelKind;
  model: string;
  parameters: ChatModelParameters | EmbeddingModelParameters;
  enabled: boolean;
  validation_status: ModelValidationStatus;
  validation_error_code: string | null;
  validated_at: IsoDate | null;
  configuration_fingerprint: string;
  capability_fingerprint: string;
  compatibility_fingerprint: string | null;
  created_at: IsoDate;
  updated_at: IsoDate;
}

export interface ModelSelection {
  chat_profile_revision_id: UUID | null;
  text_embedding_profile_revision_id: UUID | null;
  multimodal_embedding_profile_revision_id: UUID | null;
  updated_at: IsoDate;
}

export interface ModelSettings {
  providers: ModelProvider[];
  profiles: ModelProfile[];
  selection: ModelSelection;
}

export interface ModelCatalog {
  models: string[];
}

export interface ChatTerminalEvent {
  run_id: UUID;
  status_url: string;
}

export interface ChatPreviewDeltaEvent {
  run_id: UUID;
  attempt: number;
  seq: number;
  delta: string;
}

export interface ChatPreviewResetEvent {
  run_id: UUID;
  attempt: number;
  seq: number;
  reason: "generation_failed" | "validation_repair" | "preview_invalid";
}

export type ChatProgressStage =
  | "understand_query"
  | "select_workflow"
  | "retrieve_evidence"
  | "assess_evidence"
  | "prepare_visual_evidence"
  | "generate_answer"
  | "validate_answer"
  | "persist_result";

export type ChatProgressActivity =
  | "load_context"
  | "contextualize_query"
  | "route_decision"
  | "simple_search"
  | "agent_decision"
  | "agent_search"
  | "retrieval_complete"
  | "verify_coverage"
  | "research_complete"
  | "assess_evidence"
  | "prepare_visual_evidence"
  | "generate_answer"
  | "validate_answer"
  | "persist_result";

export interface ChatProgressFacts {
  objective: string | null;
  queries: string[];
  evidence_count: number | null;
  new_evidence_count: number | null;
  retrieval_calls: number | null;
  route_status: "not_applicable" | "pending" | "resolved" | "fallback" | null;
  route_reason_codes: ChatWorkflow["route_reason_codes"];
  research_status: ChatResearchResult["status"] | null;
  covered_aspects: string[];
  missing_aspects: string[];
  conflict_count: number | null;
  decision:
    | "select_simple"
    | "select_agent"
    | "search_evidence"
    | "continue_search"
    | "finish_research"
    | null;
}

export interface ChatProgressSnapshot {
  run_id: UUID;
  attempt: number;
  seq: number;
  active_stage: ChatProgressStage;
  activity: ChatProgressActivity;
  completed_stages: ChatProgressStage[];
  status: "active" | "completed";
  requested_mode: ChatWorkflowMode | null;
  resolved_mode: "pending" | "simple" | "agent";
  facts: ChatProgressFacts;
}

export interface ApiProblem {
  code?: string;
  title?: string;
  detail?: unknown;
  retryable?: boolean;
}
