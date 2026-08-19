import { AnimatePresence, motion } from 'framer-motion'
import { Activity, AudioLines, Layers, Mic, ShieldCheck } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { Badge } from './components/ui'
import { getHealth } from './lib/api'
import type { HealthResponse, LatencySample } from './lib/types'
import { Analytics } from './tabs/Analytics'
import { ChunkExplorer } from './tabs/ChunkExplorer'
import { Guardrails } from './tabs/Guardrails'
import { Playground } from './tabs/Playground'

const TABS = [
  { id: 'playground', label: 'Live Playground', icon: Mic },
  { id: 'analytics', label: 'Latency Analytics', icon: Activity },
  { id: 'chunks', label: 'Chunking Explorer', icon: Layers },
  { id: 'guardrails', label: 'Guardrails & Audit', icon: ShieldCheck },
] as const

type TabId = (typeof TABS)[number]['id']

export default function App() {
  const [tab, setTab] = useState<TabId>('playground')
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [samples, setSamples] = useState<LatencySample[]>([])

  useEffect(() => {
    const load = () =>
      getHealth()
        .then((result) => {
          setHealth(result)
          setHealthError(null)
        })
        .catch((error: Error) => setHealthError(error.message))
    void load()
    const timer = setInterval(() => void load(), 10000)
    return () => clearInterval(timer)
  }, [])

  const addSample = useCallback((sample: LatencySample) => {
    setSamples((previous) => [...previous, sample])
  }, [])

  const threshold = health?.similarity_threshold ?? 0.65

  return (
    <div className="grid-backdrop flex h-full flex-col">
      <header className="flex flex-wrap items-center gap-3 border-b border-slate-800/80 px-5 py-3">
        <div className="flex items-center gap-2.5">
          <AudioLines className="text-cyan-400" size={20} />
          <div>
            <h1 className="text-sm font-semibold tracking-tight text-slate-100">Sonic-RAG</h1>
            <p className="text-[10px] tracking-widest text-slate-500 uppercase">
              Voice-enabled Indic retrieval
            </p>
          </div>
        </div>

        <div className="ml-auto flex flex-wrap items-center gap-2">
          {healthError ? (
            <Badge tone="rose">backend unreachable</Badge>
          ) : health ? (
            <>
              <Badge tone={health.index_loaded ? 'emerald' : 'rose'}>
                {health.index_loaded ? `${health.index_size.toLocaleString()} vectors` : 'no index'}
              </Badge>
              <Badge tone={health.groq_configured ? 'cyan' : 'amber'}>{health.groq_model}</Badge>
              <Badge tone={health.circuit === 'CLOSED' ? 'emerald' : 'rose'}>
                circuit {health.circuit}
              </Badge>
              <Badge tone="slate">θ {health.similarity_threshold}</Badge>
            </>
          ) : (
            <Badge tone="slate">connecting…</Badge>
          )}
        </div>
      </header>

      <nav className="flex gap-1 overflow-x-auto border-b border-slate-800/80 px-3 py-2">
        {TABS.map(({ id, label, icon: Icon }) => {
          const active = tab === id
          return (
            <button
              key={id}
              type="button"
              onClick={() => setTab(id)}
              className={`relative flex shrink-0 items-center gap-2 rounded-lg px-3.5 py-2 text-xs font-medium transition ${
                active ? 'text-cyan-300' : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              {active && (
                <motion.span
                  layoutId="tab-highlight"
                  className="absolute inset-0 rounded-lg border border-cyan-500/30 bg-cyan-500/10"
                  transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                />
              )}
              <Icon size={14} className="relative" />
              <span className="relative">{label}</span>
            </button>
          )
        })}
      </nav>

      <main className="min-h-0 flex-1 p-4">
        <AnimatePresence mode="wait">
          <motion.div
            key={tab}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.16 }}
            className="h-full"
          >
            {tab === 'playground' && <Playground threshold={threshold} onSample={addSample} />}
            {tab === 'analytics' && (
              <Analytics samples={samples} onClear={() => setSamples([])} />
            )}
            {tab === 'chunks' && <ChunkExplorer threshold={threshold} />}
            {tab === 'guardrails' && <Guardrails threshold={threshold} />}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  )
}
