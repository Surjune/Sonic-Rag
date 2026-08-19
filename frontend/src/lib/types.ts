/** Response shapes mirrored from the FastAPI backend. */

export type Language = 'en' | 'hi' | 'ta'

/** Pipeline state, which also drives the visualizer's colour and motion. */
export type PipelineState =
  | 'idle'
  | 'listening'
  | 'processing'
  | 'grounded'
  | 'refused'
  | 'error'

export interface LatencyBreakdown {
  guardrail_input?: number
  translate?: number
  stt?: number
  embed?: number
  faiss?: number
  guardrail_grounding?: number
  llm?: number
  llm_ttft?: number
  total: number
}

export interface ContextHit {
  chunk_id: string
  score: number
  text_english: string
  display_text: string
  is_selected: boolean
  above_threshold: boolean
}

export interface QueryResponse {
  answer: string
  grounded: boolean
  blocked: boolean
  generated?: boolean
  /** The model declined for lack of usable context, despite retrieval passing. */
  model_refused?: boolean
  stage?: string
  code?: string
  message?: string
  matched?: string
  language: Language
  query?: { raw: string; english: string }
  transcript?: { native: string; english: string; detected_language?: string }
  top_score?: number
  threshold?: number
  contexts?: ContextHit[]
  model?: string
  within_budget?: boolean
  latency: LatencyBreakdown
}

export interface HealthResponse {
  status: string
  index_loaded: boolean
  index_size: number
  index_meta: Record<string, unknown>
  groq_configured: boolean
  groq_model: string
  circuit: string
  sarvam_configured: boolean
  similarity_threshold: number
  ttft_budget_ms: number
}

export interface AuditEntry {
  stage: string
  code: string
  allowed: boolean
  latency_ms: number
  detail: string
  query_preview: string
}

export interface AuditResponse {
  entries: AuditEntry[]
  blocked_count: number
  total_count: number
}

export interface StatsResponse {
  index_size: number
  meta: Record<string, unknown>
  threshold: number
  supported_languages: string[]
}

export interface StrategyChunk {
  index: number
  text: string
  embed_text: string
  char_start: number
  char_end: number
  length: number
}

export interface StrategyPreview {
  strategy: string
  description: string
  count: number
  mean_chars: number
  min_chars: number
  max_chars: number
  chunks: StrategyChunk[]
}

export interface PreviewResponse {
  source_chars: number
  strategies: StrategyPreview[]
  latency: Record<string, number>
}

export interface StrategyScore {
  name: string
  description: string
  chunks: number
  vector_bytes: number
  mean_chunk_chars: number
  build_ms: number
  search_ms_p50: number
  recall: Record<string, number>
  mrr5: number
}

export interface CompareResponse {
  available: boolean
  message?: string
  how_to_generate?: string
  generated?: string
  model?: string
  queries?: number
  passages?: number
  baseline?: string
  strategies?: StrategyScore[]
}

export interface ToolCallRecord {
  name: string
  ok: boolean
  latency_ms: number
  content: string
}

/** One completed request, kept client-side to build latency distributions. */
export interface LatencySample {
  at: number
  kind: 'text' | 'voice'
  grounded: boolean
  blocked: boolean
  latency: LatencyBreakdown
}
