/**
 * Chunking Explorer: shows how a query resolves to parent passages and the
 * child chunks that were actually embedded, plus the index composition.
 *
 * Runs retrieval with generate=false, so exploring the index costs no model
 * tokens and returns in the retrieval-only latency.
 */

import { Boxes, Layers, Loader2, Scissors, Search } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { ChunkPreview } from '../components/ChunkPreview'
import { StrategyMatrix } from '../components/StrategyMatrix'
import { Badge, EmptyState, Panel, ScoreMeter, SectionTitle } from '../components/ui'
import { getChunkingComparison, getStats, postQuery, previewChunking } from '../lib/api'
import type {
  CompareResponse,
  PreviewResponse,
  QueryResponse,
  StatsResponse,
} from '../lib/types'

/** Passages that show each strategy's behaviour, including a Devanagari case. */
const SAMPLES: { label: string; lang: string; text: string }[] = [
  {
    label: 'English passage',
    lang: 'en',
    text:
      'A corporation is a company or group of people authorized to act as a single entity and ' +
      'recognized as such in law. Early incorporated entities were established by charter, i.e. ' +
      'by an ad hoc act granted by a monarch or passed by a parliament. Most jurisdictions now ' +
      'allow the creation of new corporations through registration. Corporations are owned by ' +
      'their stockholders, who share in profits and losses generated through the firm.',
  },
  {
    label: 'Hindi passage (danda)',
    lang: 'hi',
    text:
      'निगम एक कंपनी या लोगों का समूह होता है जिसे एक एकल इकाई के रूप में कार्य करने के लिए अधिकृत ' +
      'किया गया है। प्रारंभिक निगमित संस्थाएँ चार्टर द्वारा स्थापित की गई थीं। अधिकांश ' +
      'क्षेत्राधिकार अब पंजीकरण के माध्यम से नए निगमों के निर्माण की अनुमति देते हैं। निगम अपने ' +
      'शेयरधारकों के स्वामित्व में होते हैं।',
  },
  {
    label: 'Abbreviations & decimals',
    lang: 'en',
    text:
      'Dr. Rao filed the report on Jan. 3rd. The growth rate was 3.14 percent versus 2.71 the ' +
      'prior year. Mr. Sharma of A. B. Industries Ltd. disagreed with the U.S. figures. ' +
      'The board met at 9 a.m. and adjourned at 4.30 p.m. without a decision.',
  },
]

/** Mirrors the backend's boundary-aware splitter closely enough to illustrate it. */
const SENTENCE_BOUNDARY = /(?<=[।॥.!?])\s+/u

export function ChunkExplorer({ threshold }: { threshold: number }) {
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [query, setQuery] = useState('what is a corporation?')
  const [result, setResult] = useState<QueryResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [comparison, setComparison] = useState<CompareResponse | null>(null)
  const [preview, setPreview] = useState<PreviewResponse | null>(null)
  const [sampleIndex, setSampleIndex] = useState(0)
  const [previewing, setPreviewing] = useState(false)

  useEffect(() => {
    getStats()
      .then(setStats)
      .catch(() => setStats(null))
    getChunkingComparison()
      .then(setComparison)
      .catch(() => setComparison({ available: false, message: 'Comparison unavailable.' }))
  }, [])

  const runPreview = useCallback(async (index: number) => {
    const sample = SAMPLES[index]
    setPreviewing(true)
    try {
      setPreview(await previewChunking(sample.text, sample.lang))
    } catch {
      setPreview(null)
    } finally {
      setPreviewing(false)
    }
  }, [])

  useEffect(() => {
    void runPreview(sampleIndex)
  }, [sampleIndex, runPreview])

  const explore = async () => {
    const text = query.trim()
    if (!text) return
    setLoading(true)
    setError(null)
    try {
      // generate=false: retrieval only, so browsing the index spends no tokens.
      setResult(await postQuery(text, { generate: false }))
    } catch (caught) {
      setError((caught as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const meta = (stats?.meta ?? {}) as Record<string, unknown>

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto pr-1">
      <Panel className="p-4">
        <SectionTitle
          icon={<Boxes size={14} className="text-cyan-400" />}
          title="Index composition"
        />
        <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            ['Vectors', stats ? stats.index_size.toLocaleString() : '—'],
            ['Dimensions', String(meta.dim ?? '—')],
            ['Vector space', String(meta.vector_space ?? '—')],
            ['HNSW M', String(meta.hnsw_m ?? '—')],
          ].map(([label, value]) => (
            <div key={label} className="rounded-lg border border-slate-800 bg-slate-900/40 p-3">
              <div className="text-[10px] tracking-widest text-slate-500 uppercase">{label}</div>
              <div className="mt-1 font-mono text-lg text-slate-100 tabular-nums">{value}</div>
            </div>
          ))}
        </div>
        <p className="mt-3 text-xs leading-relaxed text-slate-500">
          The corpus is parallel, so English is embedded once and the Hindi and Tamil passages ride
          along as display payloads. Indexing each language separately would embed the same English
          text three times for no recall gain.
        </p>
      </Panel>

      <StrategyMatrix report={comparison} />

      <Panel className="p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <SectionTitle
            icon={<Scissors size={14} className="text-cyan-400" />}
            title="Split a passage"
            hint="compare where each strategy cuts"
          />
          <div className="flex flex-wrap gap-1">
            {SAMPLES.map((sample, index) => (
              <button
                key={sample.label}
                type="button"
                onClick={() => setSampleIndex(index)}
                disabled={previewing}
                className={`rounded border px-2 py-1 text-[11px] transition disabled:opacity-40 ${
                  sampleIndex === index
                    ? 'border-cyan-500/40 bg-cyan-500/10 text-cyan-300'
                    : 'border-slate-700 text-slate-400 hover:text-slate-200'
                }`}
              >
                {sample.label}
              </button>
            ))}
          </div>
        </div>
        <p className="mt-2 rounded border border-slate-800 bg-slate-950/50 p-2 text-[11px] leading-relaxed text-slate-400">
          {SAMPLES[sampleIndex].text}
        </p>
      </Panel>

      <ChunkPreview preview={preview} />

      <Panel className="p-4">
        <SectionTitle
          icon={<Search size={14} className="text-cyan-400" />}
          title="Explore retrieval"
          hint="retrieval only — no model call"
        />
        <div className="mt-3 flex gap-2">
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => event.key === 'Enter' && void explore()}
            className="flex-1 rounded-lg border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-600 focus:border-cyan-500/60 focus:outline-none"
            placeholder="Query the index…"
          />
          <button
            type="button"
            onClick={() => void explore()}
            disabled={loading}
            className="flex items-center gap-2 rounded-lg bg-cyan-500 px-4 py-2 text-sm font-medium text-slate-950 transition hover:bg-cyan-400 disabled:opacity-40"
          >
            {loading ? <Loader2 size={14} className="animate-spin" /> : <Search size={14} />}
            Search
          </button>
        </div>
        {error && <p className="mt-2 text-xs text-rose-400">{error}</p>}
        {result?.latency && (
          <div className="mt-3 flex flex-wrap gap-2 font-mono text-[11px] text-slate-400">
            <Badge tone="slate">embed {result.latency.embed?.toFixed(1) ?? '—'}ms</Badge>
            <Badge tone="emerald">faiss {result.latency.faiss?.toFixed(2) ?? '—'}ms</Badge>
            <Badge tone="slate">total {result.latency.total.toFixed(1)}ms</Badge>
          </div>
        )}
      </Panel>

      {result?.contexts && result.contexts.length > 0 ? (
        <Panel className="p-4">
          <SectionTitle
            icon={<Layers size={14} className="text-cyan-400" />}
            title="Parent / child hierarchy"
            hint={`${result.contexts.length} matches`}
          />
          <div className="mt-3 space-y-4">
            {result.contexts.map((hit) => {
              // chunk_id is queryId:lang:passageIndex:childIndex
              const parts = hit.chunk_id.split(':')
              const parentId = parts.slice(0, 3).join(':')
              const childIndex = parts[3] ?? '0'
              const units = hit.display_text.split(SENTENCE_BOUNDARY).filter(Boolean)

              return (
                <div key={hit.chunk_id} className="rounded-lg border border-slate-800 bg-slate-900/40">
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 px-3 py-2">
                    <span className="font-mono text-[11px] text-cyan-300">PARENT {parentId}</span>
                    <div className="flex items-center gap-1.5">
                      {hit.is_selected && <Badge tone="cyan">gold</Badge>}
                      <Badge tone={hit.above_threshold ? 'emerald' : 'rose'}>
                        {hit.score.toFixed(4)}
                      </Badge>
                    </div>
                  </div>

                  <div className="px-3 py-2">
                    <ScoreMeter score={hit.score} threshold={result.threshold ?? threshold} />
                  </div>

                  <div className="px-3 pb-2">
                    <div className="mb-1 text-[10px] tracking-widest text-slate-500 uppercase">
                      Embedded child #{childIndex}
                    </div>
                    <p className="rounded border border-cyan-500/20 bg-cyan-500/5 p-2 text-xs leading-relaxed text-slate-200">
                      {hit.text_english}
                    </p>
                  </div>

                  <div className="px-3 pb-3">
                    <div className="mb-1 text-[10px] tracking-widest text-slate-500 uppercase">
                      Parent passage · {units.length} semantic unit{units.length === 1 ? '' : 's'}
                    </div>
                    <div className="space-y-1">
                      {units.map((unit, index) => (
                        <div
                          key={index}
                          className="flex gap-2 rounded border border-slate-800 bg-slate-950/50 p-2 text-xs leading-relaxed text-slate-400"
                        >
                          <span className="shrink-0 font-mono text-[10px] text-slate-600">
                            {String(index).padStart(2, '0')}
                          </span>
                          <span>{unit}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </Panel>
      ) : (
        <Panel className="min-h-40 flex-1">
          <EmptyState message="Run a search to inspect how passages split into embedded child chunks." />
        </Panel>
      )}
    </div>
  )
}
