/** Typed API client. Every request goes through here, never fetch() inline. */

import type {
  AuditResponse,
  CompareResponse,
  HealthResponse,
  Language,
  PreviewResponse,
  QueryResponse,
  StatsResponse,
} from './types'

const BASE = import.meta.env.VITE_API_BASE ?? ''

export class ApiError extends Error {
  constructor(
    message: string,
    readonly code: string,
    readonly status: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function parse<T>(response: Response): Promise<T> {
  const text = await response.text()
  let body: unknown
  try {
    body = JSON.parse(text)
  } catch {
    throw new ApiError(text.slice(0, 200) || response.statusText, 'BAD_RESPONSE', response.status)
  }

  // The backend wraps typed failures as { error: { code, message } }. A 400
  // guardrail block is NOT an error: it is a real, expected outcome that the
  // interface renders, so it is passed through as data.
  if (!response.ok) {
    const envelope = body as { error?: { code: string; message: string }; blocked?: boolean }
    if (envelope.blocked) return body as T
    if (envelope.error) {
      throw new ApiError(envelope.error.message, envelope.error.code, response.status)
    }
    throw new ApiError(response.statusText, 'HTTP_ERROR', response.status)
  }
  return body as T
}

export async function getHealth(): Promise<HealthResponse> {
  return parse<HealthResponse>(await fetch(`${BASE}/health`))
}

export async function getAudit(limit = 50): Promise<AuditResponse> {
  return parse<AuditResponse>(await fetch(`${BASE}/api/audit?limit=${limit}`))
}

export async function getStats(): Promise<StatsResponse> {
  return parse<StatsResponse>(await fetch(`${BASE}/api/stats`))
}

export async function postQuery(
  query: string,
  options: {
    language?: Language | null
    topK?: number
    generate?: boolean
    useTools?: boolean
  } = {},
): Promise<QueryResponse> {
  const response = await fetch(`${BASE}/api/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      language: options.language ?? null,
      top_k: options.topK ?? 5,
      generate: options.generate ?? true,
      use_tools: options.useTools ?? false,
    }),
  })
  return parse<QueryResponse>(response)
}

export async function previewChunking(
  text: string,
  lang = 'en',
): Promise<PreviewResponse> {
  const response = await fetch(`${BASE}/api/chunking/preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, lang }),
  })
  return parse<PreviewResponse>(response)
}

export async function getChunkingComparison(): Promise<CompareResponse> {
  return parse<CompareResponse>(await fetch(`${BASE}/api/chunking/compare`))
}

export async function postVoice(
  audio: Blob,
  options: { language?: Language | null; topK?: number } = {},
): Promise<QueryResponse> {
  const form = new FormData()
  form.append('file', audio, 'recording.wav')
  if (options.language) form.append('language', options.language)
  form.append('top_k', String(options.topK ?? 5))
  return parse<QueryResponse>(await fetch(`${BASE}/api/voice`, { method: 'POST', body: form }))
}

interface SseHandlers {
  onMeta?: (data: QueryResponse) => void
  onTranscript?: (data: QueryResponse) => void
  onToken?: (piece: string) => void
  onBlocked?: (data: QueryResponse) => void
  onDone?: (data: {
    latency: QueryResponse['latency']
    grounded: boolean
    model_refused: boolean
  }) => void
  onError?: (error: { code: string; message: string }) => void
}

/** Read an SSE body and dispatch each frame. Shared by the text and voice streams. */
async function consumeSse(response: Response, handlers: SseHandlers): Promise<void> {
  if (!response.body) throw new ApiError('No response stream', 'NO_STREAM', response.status)

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // Frames are separated by a blank line; a partial frame stays buffered.
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''

    for (const frame of frames) {
      let event = 'message'
      let data = ''
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data += line.slice(5).trim()
      }
      if (!data) continue

      let parsed: unknown
      try {
        parsed = JSON.parse(data)
      } catch {
        continue
      }

      switch (event) {
        case 'transcript':
          handlers.onTranscript?.(parsed as QueryResponse)
          break
        case 'meta':
          handlers.onMeta?.(parsed as QueryResponse)
          break
        case 'token':
          handlers.onToken?.((parsed as { t: string }).t)
          break
        case 'blocked':
          handlers.onBlocked?.(parsed as QueryResponse)
          break
        case 'done':
          handlers.onDone?.(
            parsed as { latency: QueryResponse['latency']; grounded: boolean; model_refused: boolean },
          )
          break
        case 'error':
          handlers.onError?.(parsed as { code: string; message: string })
          break
      }
    }
  }
}

/**
 * Streaming voice query.
 *
 * The transcript arrives long before the answer, so the user sees their own
 * words while retrieval and generation are still running -- measured at 405ms
 * against 1588ms for the blocking endpoint.
 */
export async function streamVoice(
  audio: Blob,
  handlers: SseHandlers,
  options: { language?: Language | null; topK?: number } = {},
): Promise<void> {
  const form = new FormData()
  form.append('file', audio, 'recording.wav')
  if (options.language) form.append('language', options.language)
  form.append('top_k', String(options.topK ?? 5))

  await consumeSse(
    await fetch(`${BASE}/api/voice/stream`, { method: 'POST', body: form }),
    handlers,
  )
}

/** Streaming query. Callbacks fire as server-sent events arrive. */
export async function streamQuery(
  query: string,
  handlers: SseHandlers,
  options: { language?: Language | null; topK?: number } = {},
): Promise<void> {
  await consumeSse(
    await fetch(`${BASE}/api/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        language: options.language ?? null,
        top_k: options.topK ?? 5,
      }),
    }),
    handlers,
  )
}
