/** Live Playground: voice + text input, the 3D orb, and the latency HUD. */

import { AdaptiveDpr } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { AnimatePresence, motion } from 'framer-motion'
import {
  AlertTriangle,
  ChevronDown,
  Loader2,
  Mic,
  Send,
  ShieldAlert,
  Square,
} from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { Orb } from '../components/Orb'
import { Badge, Metric, Panel, ScoreMeter, SectionTitle } from '../components/ui'
import { ApiError, postQuery, streamVoice } from '../lib/api'
import { startRecording, type RecorderHandle } from '../lib/audio'
import type { Language, LatencySample, PipelineState, QueryResponse } from '../lib/types'

const LANGUAGES: { value: Language | 'auto'; label: string }[] = [
  { value: 'auto', label: 'Auto' },
  { value: 'en', label: 'English' },
  { value: 'hi', label: 'हिन्दी' },
  { value: 'ta', label: 'தமிழ்' },
]

const STATE_LABEL: Record<PipelineState, string> = {
  idle: 'Idle',
  listening: 'Listening',
  processing: 'Processing',
  grounded: 'Grounded',
  refused: 'Refused',
  error: 'Error',
}

interface PlaygroundProps {
  threshold: number
  onSample: (sample: LatencySample) => void
}

export function Playground({ threshold, onSample }: PlaygroundProps) {
  const [state, setState] = useState<PipelineState>('idle')
  const [level, setLevel] = useState(0)
  const [text, setText] = useState('')
  const [language, setLanguage] = useState<Language | 'auto'>('auto')
  const [response, setResponse] = useState<QueryResponse | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [contextsOpen, setContextsOpen] = useState(true)

  const recorder = useRef<RecorderHandle | null>(null)
  const animationFrame = useRef<number | null>(null)

  useEffect(() => {
    return () => {
      if (animationFrame.current !== null) cancelAnimationFrame(animationFrame.current)
      recorder.current?.cancel()
    }
  }, [])

  const settle = useCallback(
    (result: QueryResponse, kind: 'text' | 'voice') => {
      setResponse(result)
      setState(result.blocked || !result.grounded ? 'refused' : 'grounded')
      onSample({
        at: Date.now(),
        kind,
        grounded: result.grounded,
        blocked: result.blocked,
        latency: result.latency,
      })
    },
    [onSample],
  )

  const fail = useCallback((error: unknown) => {
    setState('error')
    setErrorMessage(
      error instanceof ApiError
        ? `${error.code}: ${error.message}`
        : error instanceof Error
          ? error.message
          : 'Request failed',
    )
  }, [])

  const submitText = useCallback(async () => {
    const query = text.trim()
    if (!query) return
    setErrorMessage(null)
    setResponse(null)
    setState('processing')
    try {
      const result = await postQuery(query, {
        language: language === 'auto' ? null : language,
      })
      settle(result, 'text')
    } catch (error) {
      fail(error)
    }
  }, [text, language, settle, fail])

  const pollLevel = useCallback(() => {
    const handle = recorder.current
    if (!handle) return
    setLevel(handle.getLevel())
    animationFrame.current = requestAnimationFrame(pollLevel)
  }, [])

  const beginRecording = useCallback(async () => {
    setErrorMessage(null)
    setResponse(null)
    try {
      recorder.current = await startRecording()
      setState('listening')
      pollLevel()
    } catch (error) {
      // Denied permission is the common case and deserves a plain explanation
      // rather than a raw DOMException.
      setState('error')
      setErrorMessage(
        error instanceof DOMException && error.name === 'NotAllowedError'
          ? 'Microphone permission denied. Allow access, or type your question instead.'
          : `Could not start the microphone: ${(error as Error).message}`,
      )
    }
  }, [pollLevel])

  const finishRecording = useCallback(async () => {
    const handle = recorder.current
    if (!handle) return
    recorder.current = null
    if (animationFrame.current !== null) cancelAnimationFrame(animationFrame.current)
    setLevel(0)
    setState('processing')

    try {
      const wav = await handle.stop()
      // Streamed: the transcript lands around 405ms and the first token around
      // 1073ms, against 1588ms before anything at all appeared.
      let answer = ''
      let partial: QueryResponse | null = null
      let blocked = false

      await streamVoice(
        wav,
        {
          onTranscript: (data) => {
            partial = { ...(partial ?? {}), ...data, answer: '' } as QueryResponse
            setResponse(partial)
          },
          onMeta: (data) => {
            partial = { ...(partial ?? {}), ...data, answer } as QueryResponse
            setResponse(partial)
          },
          onToken: (piece) => {
            answer += piece
            setResponse((previous) =>
              previous ? { ...previous, answer } : previous,
            )
          },
          onBlocked: (data) => {
            blocked = true
            partial = { ...(partial ?? {}), ...data, blocked: true } as QueryResponse
            setResponse(partial)
            setState('refused')
          },
          onError: (error) => {
            throw new ApiError(error.message, error.code, 502)
          },
          onDone: (data) => {
            const final = {
              ...(partial ?? {}),
              answer,
              grounded: data.grounded,
              model_refused: data.model_refused,
              blocked: false,
              latency: data.latency,
            } as QueryResponse
            setResponse(final)
            setState(data.grounded ? 'grounded' : 'refused')
            onSample({
              at: Date.now(),
              kind: 'voice',
              grounded: data.grounded,
              blocked: false,
              latency: data.latency,
            })
          },
        },
        { language: language === 'auto' ? null : language },
      )

      if (blocked && partial) {
        const refused = partial as QueryResponse
        onSample({
          at: Date.now(),
          kind: 'voice',
          grounded: false,
          blocked: true,
          latency: refused.latency ?? { total: 0 },
        })
      }
    } catch (error) {
      fail(error)
    }
  }, [language, fail, onSample])

  const busy = state === 'processing'
  const recording = state === 'listening'
  const latency = response?.latency

  return (
    <div className="grid h-full grid-cols-1 gap-4 overflow-hidden lg:grid-cols-[minmax(0,1fr)_400px]">
      {/* Visualizer + input */}
      <div className="flex min-h-0 flex-col gap-4">
        {/* panel-glass so the beach illustration reads as the orb's setting
            rather than being hidden behind an opaque surface. */}
        <Panel className="panel-glass relative min-h-0 flex-1 overflow-hidden">
          {/*
            DPR is capped and adaptive. On an integrated GPU a 2x device pixel
            ratio quadruples fragment work, and during a local demo the browser
            and the retrieval backend share the same two cores -- so an
            over-eager visualizer inflates the very latency it is displaying.
            AdaptiveDpr drops resolution automatically when frames slow down.
          */}
          <Canvas
            camera={{ position: [0, 0, 4.4], fov: 50 }}
            dpr={[1, 1.25]}
            performance={{ min: 0.4, debounce: 200 }}
            gl={{ antialias: false, powerPreference: 'high-performance' }}
          >
            <AdaptiveDpr pixelated />
            <Orb state={state} level={level} />
          </Canvas>

          <div className="pointer-events-none absolute inset-x-0 top-0 flex items-center justify-between p-4">
            <div className="flex items-center gap-2">
              <span className="relative flex h-2 w-2">
                <span
                  className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-70"
                  style={{ background: 'currentColor' }}
                />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-current" />
              </span>
              <span className="font-mono text-xs tracking-widest text-slate-300 uppercase">
                {STATE_LABEL[state]}
              </span>
            </div>
            {response?.model && <Badge tone="cyan">{response.model}</Badge>}
          </div>

          {/* Latency HUD */}
          <div className="pointer-events-none absolute inset-x-0 bottom-0 p-4">
            <div className="panel grid grid-cols-3 gap-3 rounded-lg px-4 py-3 sm:grid-cols-6">
              <Metric label="STT" value={latency?.stt} />
              <Metric label="Translate" value={latency?.translate} />
              <Metric label="Embed" value={latency?.embed} />
              <Metric label="FAISS" value={latency?.faiss} budgetMs={5} />
              <Metric label="Groq TTFT" value={latency?.llm_ttft} />
              <Metric label="Total" value={latency?.total} />
            </div>
          </div>
        </Panel>

        <Panel className="p-3">
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={recording ? finishRecording : beginRecording}
              disabled={busy}
              className={`flex items-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium transition ${
                recording
                  ? 'bg-fuchsia-600 text-white hover:bg-fuchsia-500'
                  : 'bg-cyan-500/15 text-cyan-300 hover:bg-cyan-500/25 disabled:opacity-40'
              }`}
            >
              {recording ? <Square size={15} /> : <Mic size={15} />}
              {recording ? 'Stop' : 'Speak'}
            </button>

            <input
              value={text}
              onChange={(event) => setText(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  void submitText()
                }
              }}
              disabled={busy || recording}
              placeholder="Ask in English, हिन्दी, or தமிழ்…"
              className="min-w-40 flex-1 rounded-lg border border-slate-700 bg-slate-900/60 px-3 py-2.5 text-sm text-slate-100 placeholder:text-slate-600 focus:border-cyan-500/60 focus:outline-none disabled:opacity-40"
            />

            <select
              value={language}
              onChange={(event) => setLanguage(event.target.value as Language | 'auto')}
              className="rounded-lg border border-slate-700 bg-slate-900/60 px-2 py-2.5 text-sm text-slate-300 focus:border-cyan-500/60 focus:outline-none"
            >
              {LANGUAGES.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>

            <button
              type="button"
              onClick={() => void submitText()}
              disabled={busy || recording || !text.trim()}
              className="flex items-center gap-2 rounded-lg bg-cyan-500 px-4 py-2.5 text-sm font-medium text-slate-950 transition hover:bg-cyan-400 disabled:opacity-30"
            >
              {busy ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} />}
              Ask
            </button>
          </div>
        </Panel>
      </div>

      {/* Answer + retrieved context */}
      <div className="flex min-h-0 flex-col gap-4 overflow-y-auto pr-1">
        <AnimatePresence mode="wait">
          {errorMessage && (
            <motion.div
              key="error"
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <Panel className="border-rose-500/40 p-4">
                <div className="flex gap-2 text-sm text-rose-300">
                  <AlertTriangle size={16} className="mt-0.5 shrink-0" />
                  <span>{errorMessage}</span>
                </div>
              </Panel>
            </motion.div>
          )}
        </AnimatePresence>

        {response && (
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
            <Panel className="p-4">
              <div className="mb-3 flex flex-wrap items-center gap-2">
                <SectionTitle title="Answer" />
                {response.blocked ? (
                  <Badge tone="rose">{response.code ?? 'BLOCKED'}</Badge>
                ) : response.model_refused ? (
                  // Retrieval cleared the threshold but the model still found
                  // the passages unusable. Showing GROUNDED here would
                  // contradict the refusal printed directly below it.
                  <Badge tone="amber">MODEL UNGROUNDED</Badge>
                ) : (
                  <Badge tone="emerald">GROUNDED</Badge>
                )}
                {response.top_score !== undefined && (
                  <Badge tone="slate">score {response.top_score.toFixed(4)}</Badge>
                )}
                {response.within_budget !== undefined && (
                  <Badge tone={response.within_budget ? 'emerald' : 'amber'}>
                    {response.within_budget ? 'within budget' : 'over budget'}
                  </Badge>
                )}
              </div>

              {response.transcript && (
                <div className="mb-3 rounded-lg border border-slate-800 bg-slate-900/50 p-3 text-xs">
                  <div className="mb-1 text-slate-500">Heard</div>
                  <div className="text-slate-200">{response.transcript.native}</div>
                  {response.transcript.english !== response.transcript.native && (
                    <div className="mt-1 text-slate-500">→ {response.transcript.english}</div>
                  )}
                </div>
              )}

              <p
                className={`text-sm leading-relaxed ${
                  response.blocked ? 'text-rose-300' : 'text-slate-100'
                }`}
              >
                {response.blocked && response.stage === 'input' ? (
                  <span className="flex items-start gap-2">
                    <ShieldAlert size={15} className="mt-0.5 shrink-0" />
                    {response.message || 'Blocked by the input guardrail.'}
                  </span>
                ) : (
                  response.answer || '—'
                )}
              </p>

              {response.matched && (
                <div className="mt-2 rounded border border-rose-500/30 bg-rose-500/5 p-2 font-mono text-[11px] text-rose-300">
                  matched: {response.matched}
                </div>
              )}
            </Panel>
          </motion.div>
        )}

        {response?.contexts && response.contexts.length > 0 && (
          <Panel className="p-4">
            <button
              type="button"
              onClick={() => setContextsOpen((open) => !open)}
              className="flex w-full items-center justify-between"
            >
              <SectionTitle
                title="Retrieved context"
                hint={`${response.contexts.length} chunks · threshold ${response.threshold ?? threshold}`}
              />
              <ChevronDown
                size={16}
                className={`text-slate-500 transition ${contextsOpen ? 'rotate-180' : ''}`}
              />
            </button>

            {contextsOpen && (
              <div className="mt-3 space-y-3">
                {response.contexts.map((hit) => (
                  <div
                    key={hit.chunk_id}
                    className="rounded-lg border border-slate-800 bg-slate-900/40 p-3"
                  >
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <span className="font-mono text-[10px] text-slate-500">{hit.chunk_id}</span>
                      <div className="flex items-center gap-1.5">
                        {hit.is_selected && <Badge tone="cyan">gold</Badge>}
                        <Badge tone={hit.above_threshold ? 'emerald' : 'rose'}>
                          {hit.score.toFixed(4)}
                        </Badge>
                      </div>
                    </div>
                    <ScoreMeter score={hit.score} threshold={response.threshold ?? threshold} />
                    <p className="mt-2 line-clamp-4 text-xs leading-relaxed text-slate-300">
                      {hit.display_text}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </Panel>
        )}
      </div>
    </div>
  )
}
