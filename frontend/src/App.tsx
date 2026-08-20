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

/** Numbered like the Task 1 site's nav, so both submissions read as one identity. */
const TABS = [
  { id: 'playground', num: '01', label: 'Live Playground', icon: Mic },
  { id: 'analytics', num: '02', label: 'Latency Analytics', icon: Activity },
  { id: 'chunks', num: '03', label: 'Chunking Explorer', icon: Layers },
  { id: 'guardrails', num: '04', label: 'Guardrails & Audit', icon: ShieldCheck },
] as const

const EVENT = {
  name: 'HH Goa 2026',
  full: 'Hacker House Goa 2026',
  task: 'Task 2 · Voice-Enabled Indic RAG',
  team: 'Lightning Logics',
  dates: 'Goa, India · 28–31 Oct 2026',
}


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
      <header className="flex flex-wrap items-center gap-3 border-b border-white/10 px-5 py-3">
        <div className="flex items-center gap-3">
          <AudioLines style={{ color: 'var(--color-goa-yellow)' }} size={22} />
          <div>
            <div className="flex flex-wrap items-baseline gap-2">
              <h1 className="display-type text-xl leading-none">SONIC&nbsp;RAG</h1>
              <span
                className="rounded-full px-2 py-0.5 font-mono text-[9px] tracking-widest uppercase"
                style={{ background: 'var(--color-goa-pink)', color: 'var(--color-goa-yellow)' }}
              >
                {EVENT.name}
              </span>
            </div>
            <p className="mt-0.5 font-mono text-[10px] tracking-[0.22em] text-emerald-100/60 uppercase">
              {EVENT.task}
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

      <nav className="flex items-center gap-1 overflow-x-auto border-b border-white/10 px-3 py-2">
        {TABS.map(({ id, num, label, icon: Icon }) => {
          const active = tab === id
          return (
            <button
              key={id}
              type="button"
              onClick={() => setTab(id)}
              className={`relative flex shrink-0 items-center gap-2 rounded-lg px-3.5 py-2 text-xs font-medium transition ${
                active ? '' : 'text-emerald-100/45 hover:text-emerald-100/80'
              }`}
              style={active ? { color: 'var(--color-goa-yellow)' } : undefined}
            >
              {active && (
                <motion.span
                  layoutId="tab-highlight"
                  className="absolute inset-0 rounded-lg border"
                  style={{
                    borderColor: 'color-mix(in srgb, var(--color-goa-yellow) 35%, transparent)',
                    background: 'color-mix(in srgb, var(--color-goa-yellow) 10%, transparent)',
                  }}
                  transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                />
              )}
              {/* Numbered sections, as on the event site's nav. */}
              <span className="relative font-mono text-[10px] opacity-70">{num}</span>
              <Icon size={14} className="relative" />
              <span className="relative">{label}</span>
            </button>
          )
        })}

        <div className="ml-auto hidden shrink-0 items-center gap-2 pr-2 sm:flex">
          <span
            className="font-mono text-[10px] tracking-[0.2em] uppercase"
            style={{ color: 'var(--color-goa-pink)' }}
          >
            {EVENT.team}
          </span>
          <span className="h-3 w-px bg-white/15" />
          <span className="font-mono text-[10px] text-emerald-100/40">{EVENT.dates}</span>
        </div>
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
