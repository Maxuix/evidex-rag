export type UUID = string;
export type IsoDate = string;
export type JsonMap = Record<string, unknown>;
export type RerankMode = "none" | "classic" | "local_minilm_v1";

export interface Page<T> {
  items: T[];
  next_cursor: string | null;
}

export interface KnowledgeBase {
  id: UUID;
  name: string;
  source_change_seq: number;
  active_index_revision_id: UUID;
  embedding_space_id: UUID;
  embedding: {
    strategy: "text_only" | "dual_space" | "unified_multimodal";
    text: {
      embedding_space_id: UUID;
      profile_revision_id: UUID | null;
      dimension: number;
    };
    cross_modal: {
      embedding_space_id: UUID;
      profile_revision_id: UUID | null;
      dimension: number;
    } | null;
  };
  parsing: {
    preset: ParsingPreset;
    profile:
      | "docling_text_local_v1"
      | "docling_multimodal_local_v2"
      | "docling_text_local_v2"
      | "docling_multimodal_local_v3"
      | "docling_text_local_v3"
      | "docling_multimodal_local_v4"
      | "docling_text_local_v4"
      | "docling_multimodal_local_v5";
  };
  chunking: {
    preset: ChunkingPreset;
    profile: "structural_by_title_token_v4" | "semantic_breakpoint_v3" | "semantic_breakpoint_v4";
  };
  retrieval_defaults: {
    strategy: "exact_vector";
    top_k: number;
    rerank_mode: RerankMode;
  };
  answer_policy_defaults: Record<string, unknown>; // Historical, read-only.
  auto_qa: {
    enabled: boolean;
    questions_per_chunk: number;
    model_profile_revision_id: UUID | null;
    model_name: string | null;
    model_revision: number | null;
  };
  provisioned_at: IsoDate;
  created_at: IsoDate;
  updated_at: IsoDate;
}

export type ParsingPreset = "text_local_v1" | "multimodal_local_v2";
export type ChunkingPreset = "structural_balanced_v2" | "semantic_balanced_v1";

export type KnowledgeBaseEmbeddingSelection =
  | {
    strategy: "text_only";
    text_profile_revision_id?: UUID | null;
  }
  | {
    strategy: "dual_space";
    text_profile_revision_id?: UUID | null;
    multimodal_profile_revision_id?: UUID | null;
  }
  | {
    strategy: "unified_multimodal";
    profile_revision_id?: UUID | null;
  };

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
  composite_chunk_count: number | null;
  visual_unit_count: number | null;
  relation_count: number | null;
  text_representation_count: number | null;
  native_image_representation_count: number | null;
  table_representation_count: number | null;
}

export interface DocumentDetail extends DocumentRecord {
  index: DocumentIndexSummary | null;
}

export interface DocumentChunkAsset {
  id: UUID;
  media_type: string;
  checksum_sha256: string;
  content_url: string;
  width: number | null;
  height: number | null;
}

export interface DocumentChunkRelation {
  visual_unit_id: UUID;
  asset: DocumentChunkAsset;
  relation_type: string;
  confidence_micros: number;
  provenance: string;
  figure_label: string | null;
}

export interface DocumentChunk {
  id: UUID;
  ordinal: number;
  modality: "text" | "image" | "table";
  content: string;
  token_count: number;
  source_location: JsonMap;
  hierarchy: JsonMap;
  source_metadata: JsonMap;
  evidence_group_key: string | null;
  representations: string[];
  asset: DocumentChunkAsset | null;
  related_visuals: DocumentChunkRelation[];
  excluded_at: IsoDate | null;
  generated_questions: string[];
}

export interface DocumentChunkInspection {
  document_id: UUID;
  document_version_id: UUID;
  indexed_document_version_id: UUID;
  index_revision_id: UUID;
  total_chunks: number;
  items: DocumentChunk[];
  next_cursor: string | null;
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

export type IndexingProgress =
  | {
    schema_version: "pdf_parsing_progress_v1";
    stage: string;
    total_pages: number;
    completed_pages: number;
    segment_number: number;
    segment_count: number;
    page_from: number;
    page_to: number;
    stage_pages: Record<string, number>;
    ocr_pages: number;
    ocr_regions: number;
    table_candidates: number;
    elapsed_ms: number;
    child_peak_rss_bytes: number | null;
  }
  | {
    schema_version: "auto_qa_generation_v1";
    eligible_chunks: number;
    processed_chunks: number;
    question_count: number;
    model_calls: number;
    prompt_tokens: number;
    completion_tokens: number;
  };

export interface IndexingJob {
  job_id: UUID;
  kb_id: UUID;
  document_id: UUID;
  document_version_id: UUID;
  indexed_document_version_id: UUID;
  index_revision_id: UUID;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  phase: string;
  progress: IndexingProgress | null;
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

export interface RetrievalEvidence {
  rank: number;
  index_chunk_id: UUID;
  document_id: UUID;
  document_version_id: UUID;
  ordinal: number;
  text: string;
  source_location: JsonMap;
  hierarchy: JsonMap;
  score: number;
  score_kind: "cosine_similarity" | "hybrid_rerank" | "reciprocal_rank_fusion" | "graph_path";
  vector_similarity: number | null;
  lexical_score: number;
  modality: "text" | "image" | "table";
  asset: DocumentChunkAsset | null;
  matched_representations: string[];
  model_rerank_score: number | null;
  model_rerank_rank: number | null;
  model_rerank_window_count: number | null;
  model_rerank_winning_window_index: number | null;
}

export interface RetrievalEvidencePack {
  knowledge_base_id: UUID;
  index_revision_id: UUID;
  strategy: "exact_vector" | "hybrid";
  evidence: RetrievalEvidence[];
  debug: {
    result_count: number;
    text_candidate_count: number | null;
    lexical_candidate_count: number | null;
    cross_modal_candidate_count: number | null;
    hydrated_relation_count: number | null;
    evidence_group_count: number | null;
    model_rerank_candidate_count: number | null;
    model_rerank_window_count: number | null;
    matched_questions?: {
      index_chunk_id: UUID;
      ordinal: number;
      question: string;
    }[];
  } | null;
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
  mode: "text" | "auto" | "graph";
  profile_version: string;
  strategy: "exact_vector" | "hybrid";
  top_k: number;
  rerank_mode: RerankMode;
}

export interface GraphConfig {
  knowledge_base_id: UUID;
  enabled: boolean;
  status: "disabled" | "building" | "ready" | "failed";
  build_id: UUID;
  chat_profile_revision_id: UUID | null;
  profile_name: string | null;
  provider_name: string | null;
  model: string | null;
  extractor_version: string;
  schema_profile_key: string;
  schema_profile_name: string;
  schema_profile_digest: string;
  active_build_schema_profile_key: string | null;
  active_build_schema_profile_digest: string | null;
  last_error_code: string | null;
  eligible_chunk_count: number;
  processed_chunk_count: number;
  extracted_chunk_count: number;
  empty_chunk_count: number;
  protocol_skipped_count: number;
  resource_skipped_count: number;
  allowed_skipped_count: number;
  requires_rebuild: boolean;
}

export interface GraphConfigUpdate {
  enabled: boolean;
  chat_profile_revision_id?: UUID | null;
  schema_profile_key?: string | null;
  retry?: boolean;
  force_rebuild?: boolean;
}

export interface GraphSchemaProfile {
  key: string;
  display_name: string;
  description: string;
  is_default: boolean;
}

export interface ChatAgentTraceEvent {
  // "verifier" is retained only for displaying historical runs.
  tool: "search_knowledge_base" | "semantic_search" | "keyword_search" | "read_chunk_context" | "list_documents" | "search_graph_relations" | "calculate" | "submit_answer" | "verifier" | "protocol";
  status: "ok" | "rejected" | "salvaged" | "refused";
  tool_call_id: string;
  refs: string[];
  count: number;
  retrieval_lane?: "simple" | "semantic" | "keyword" | "chunk_context" | "document_list" | "graph_relations";
  route_reason_code?:
    | "direct_relation"
    | "relation_chain"
    | "entity_alias"
    | "cross_document_relation";
  route_result_code?:
    | "not_requested"
    | "admitted"
    | "no_evidence"
    | "not_ready"
    | "timeout"
    | "unavailable"
    | "rejected";
  new_evidence_count?: number;
  call_index?: number;
  invocation_source?: "agent" | "legacy_guard";
  duration_ms?: number;
  candidate_count?: number;
  path_count?: number;
  hydrated_chunk_count?: number;
  returned_chunk_count?: number;
  hop1_count?: number;
  hop2_count?: number;
  hop3_count?: number;
}

export interface ChatAgent {
  version:
    | "native_tool_calling_agent_v3"
    | "native_tool_calling_agent_v4"
    | "native_tool_calling_agent_v5"
    | "native_tool_calling_agent_v6";
  budget: {
    max_total_tokens?: number;
    // Legacy v3/v4 fields, present only on historical runs.
    max_model_rounds?: number;
    max_graph_calls?: number;
    max_evidence_items?: number;
    max_retrieval_calls?: number;
  };
  trace: {
    version:
      | "native_tool_calling_agent_v3"
      | "native_tool_calling_agent_v4"
      | "native_tool_calling_agent_v5"
      | "native_tool_calling_agent_v6";
    events: ChatAgentTraceEvent[];
    budget: ChatAgent["budget"];
    usage: Record<string, number>;
    diagnostics?: {
      stop_reason:
        | "submitted"
        | "token_budget"
        | "no_new_evidence"
        | "protocol_error"
        | "deadline_exceeded"
        // Historical v3/v4 traces only.
        | "retrieval_query_budget"
        | "evidence_budget"
        | "model_round_limit"
        | "submit_protocol_invalid";
      forced_finalize: boolean;
      consecutive_no_new_evidence: number;
      elapsed_ms: number | null;
      deadline_ms: number | null;
      deadline_remaining_ms: number | null;
      deadline_exceeded: boolean;
    };
    outcome: "answered" | "partial" | "refused" | "clarify";
  } | null;
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
  effective_answer_policy: Record<string, unknown>; // Historical, read-only.
  agent: ChatAgent;
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
  retrieval: {
    mode: "text" | "auto" | "graph";
    top_k: number;
    rerank_mode: RerankMode;
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
  dimension: "auto" | number;
  max_batch_size: number;
  shared_text_image_space_confirmed: boolean;
}

export interface EmbeddingValidationSnapshot {
  schema_version: "embedding_validation_v1";
  provider_supported_dimensions: number[] | null;
  verified_dimensions: number[];
  provider_default_dimension: number | null;
  recommended_dimension: number | null;
  selected_dimension: number;
  selection_source:
    | "provider_recommended"
    | "provider_default"
    | "automatic_1024"
    | "automatic_above_1024"
    | "automatic_below_1024"
    | "provider_observed_default"
    | "user_probe"
    | "legacy_explicit";
  dimension_request_mode: "explicit" | "omitted";
  input_capabilities: Array<"text_document" | "text_query" | "image">;
  shared_text_image_space_confirmed: boolean;
  distance_metric: "cosine";
  vector_data_type: "float32";
  normalization: "l2" | "client_l2_v1";
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
  provider_secret_available: boolean;
  validation_status: ModelValidationStatus;
  validation_error_code: string | null;
  validated_at: IsoDate | null;
  configuration_fingerprint: string;
  capability_fingerprint: string;
  compatibility_fingerprint: string | null;
  embedding_validation: EmbeddingValidationSnapshot | null;
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

export type ChatProgressStage =
  | "understand_query"
  | "retrieve_evidence"
  | "prepare_visual_evidence"
  | "generate_answer"
  | "validate_answer"
  | "persist_result";

export type ChatProgressActivity =
  | "load_context"
  | "tool_decision"
  | "search_knowledge_base"
  | "search_graph_relations"
  | "calculate"
  | "submit_answer"
  | "retrieval_complete"
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
  covered_aspects: string[];
  missing_aspects: string[];
  conflict_count: number | null;
}

export interface ChatProgressSnapshot {
  run_id: UUID;
  attempt: number;
  seq: number;
  active_stage: ChatProgressStage;
  activity: ChatProgressActivity;
  completed_stages: ChatProgressStage[];
  status: "active" | "completed";
  facts: ChatProgressFacts;
}

export interface ApiProblem {
  code?: string;
  title?: string;
  detail?: unknown;
  retryable?: boolean;
  trace_id?: string;
  errors?: Array<{
    location: Array<string | number>;
    message: string;
    error_type: string;
  }> | null;
}
