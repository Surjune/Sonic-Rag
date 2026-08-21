import { AnimatePresence, motion } from 'framer-motion'
import { Activity, AudioLines, Layers, Mic, ShieldCheck } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { ProviderSwitch } from './components/ProviderSwitch'
import { Badge } from './components/ui'
import { getHealth, getProviders } from './lib/api'
import type { HealthResponse, LatencySample, ProvidersResponse } from './lib/types'
import { Analytics } from './tabs/Analytics'
import { ChunkExplorer } from './tabs/ChunkExplorer'
import { Guardrails } from './tabs/Guardrails'
import { Playground } from './tabs/Playground'

/** Numbered like the Task 1 site's nav, so both submissions read as one identity. */
const TABS = [
  { id: 'playground', num: '01', label: 'Live Playground', short: 'Ask', icon: Mic },
  { id: 'analytics', num: '02', label: 'Latency Analytics', short: 'Latency', icon: Activity },
  { id: 'chunks', num: '03', label: 'Chunking Explorer', short: 'Chunks', icon: Layers },
  { id: 'guardrails', num: '04', label: 'Guardrails & Audit', short: 'Guards', icon: ShieldCheck },
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
  const [providers, setProviders] = useState<ProvidersResponse | null>(null)
  // Which backend answers. Remembered, so choosing local once does not have to
  // be chosen again on every reload.
  const [provider, setProvider] = useState<string>(() => {
    try {
      return localStorage.getItem('sonic-rag.provider') || 'groq'
    } catch {
      return 'groq'
    }
  })

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

  const refreshProviders = useCallback(() => {
    getProviders()
      .then((result) => {
        setProviders(result)
        // A remembered choice that is no longer reachable -- Ollama stopped,
        // model removed -- silently falls back rather than failing every query.
        setProvider((current) =>
          result.available.includes(current) ? current : result.default,
        )
      })
      .catch(() => setProviders(null))
  }, [])

  useEffect(() => refreshProviders(), [refreshProviders])

  useEffect(() => {
    try {
      localStorage.setItem('sonic-rag.provider', provider)
    } catch {
      // Storage being unavailable is not a reason to break the switch.
    }
  }, [provider])

  const addSample = useCallback((sample: LatencySample) => {
    setSamples((previous) => [...previous, sample])
  }, [])

  const threshold = health?.similarity_threshold ?? 0.65

  return (
    <div className="grid-backdrop flex min-h-[100dvh] flex-col md:h-full">
      {/*
        One row: product on the left, event wordmark centred, status on the
        right. The wordmark is absolutely positioned rather than placed in the
        flow, because the two side blocks have different widths and a flex
        `mx-auto` would push it off-centre by the difference. Hidden below xl,
        where the sides would otherwise collide with it.
      */}
      <header className="relative flex shrink-0 flex-wrap items-center gap-x-3 gap-y-2 border-b border-white/10 px-3 py-2 md:flex-nowrap md:px-5 md:py-2.5">
        <div className="flex min-w-0 items-center gap-2.5">
          <AudioLines style={{ color: 'var(--color-goa-yellow)' }} size={20} className="shrink-0" />
          <h1 className="display-type text-lg leading-none whitespace-nowrap">SONIC&nbsp;RAG</h1>
          <span
            className="shrink-0 rounded-full px-2 py-0.5 font-mono text-[9px] tracking-widest uppercase"
            style={{ background: 'var(--color-goa-pink)', color: 'var(--color-goa-yellow)' }}
          >
            {EVENT.name}
          </span>
          <span className="hidden h-3.5 w-px shrink-0 bg-white/15 lg:block" />
          <p className="hidden truncate font-mono text-[10px] tracking-[0.2em] text-emerald-100/55 uppercase lg:block">
            {EVENT.task}
          </p>
        </div>

        <div
          className="pointer-events-none absolute left-1/2 hidden -translate-x-1/2 xl:block"
          aria-hidden="true"
        >
          <div className="goa-wordmark">
            <span>HACKER</span>
            <span className="goa-chip">गोवा</span>
            <span>HOUSE</span>
          </div>
        </div>

        <div className="ml-auto flex shrink-0 flex-wrap items-center gap-2">
          {healthError ? (
            <Badge tone="rose">backend unreachable</Badge>
          ) : health ? (
            <>
              <Badge tone={health.index_loaded ? 'emerald' : 'rose'}>
                {health.index_loaded ? `${health.index_size.toLocaleString()} vectors` : 'no index'}
              </Badge>
              <ProviderSwitch
                providers={providers}
                value={provider}
                onChange={setProvider}
                onRefresh={refreshProviders}
              />
              <span className="hidden sm:contents">
                <Badge tone={health.circuit === 'CLOSED' ? 'emerald' : 'rose'}>
                  circuit {health.circuit}
                </Badge>
                <Badge tone="slate">θ {health.similarity_threshold}</Badge>
              </span>
            </>
          ) : (
            <Badge tone="slate">connecting…</Badge>
          )}
        </div>
      </header>

      <nav className="hidden items-center gap-1 overflow-x-auto border-b border-white/10 px-3 py-2 md:flex">
        {TABS.map(({ id, num, label, icon: Icon }) => {
          const active = tab === id
          return (
            <button
              key={id}
              type="button"
              onClick={() => setTab(id)}
              className={`relative flex shrink-0 items-center gap-1.5 rounded-lg px-2.5 py-2 text-xs font-medium transition sm:gap-2 sm:px-3.5 ${
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
              <span className="relative hidden font-mono text-[10px] opacity-70 sm:inline">{num}</span>
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

      <main
        className={`min-h-0 flex-1 p-3 md:p-4 md:pb-4 ${
          tab === 'playground' ? 'pb-[10.5rem]' : 'pb-24'
        }`}
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={tab}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.16 }}
            className="h-full"
          >
            {tab === 'playground' && (
              <Playground
                threshold={threshold}
                onSample={addSample}
                provider={provider}
                onProviderChange={setProvider}
                providers={providers}
              />
            )}
            {tab === 'analytics' && (
              <Analytics samples={samples} onClear={() => setSamples([])} />
            )}
            {tab === 'chunks' && <ChunkExplorer threshold={threshold} />}
            {tab === 'guardrails' && <Guardrails threshold={threshold} />}
          </motion.div>
        </AnimatePresence>
      </main>

      {/*
        Bottom navigation, phones only.
        A horizontal strip of four tabs at the top of a 360px screen either
        overflows or shrinks past legibility, and it sits at the far end of the
        reach of a thumb. Down here each tab is a quarter of the width with a
        real tap target, which is where a phone user expects to change section.

        pb-[env(safe-area-inset-bottom)] keeps it clear of the home indicator on
        an iPhone, where the last 34px belong to the system.
      */}
      <nav className="fixed inset-x-0 bottom-0 z-40 flex border-t border-white/10 bg-[#04170f]/95 pb-[env(safe-area-inset-bottom)] backdrop-blur-md md:hidden">
        {TABS.map(({ id, short, icon: Icon }) => {
          const active = tab === id
          return (
            <button
              key={id}
              type="button"
              onClick={() => setTab(id)}
              className="relative flex flex-1 flex-col items-center justify-center gap-1 py-2.5 transition"
              style={{ color: active ? 'var(--color-goa-yellow)' : 'rgba(209,250,229,0.45)' }}
            >
              <Icon size={18} />
              <span className="font-mono text-[10px] tracking-wider uppercase">{short}</span>
              {active && (
                <motion.span
                  layoutId="tab-highlight-mobile"
                  className="absolute inset-x-3 top-0 h-0.5 rounded-full"
                  style={{ background: 'var(--color-goa-yellow)' }}
                  transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                />
              )}
            </button>
          )
        })}
      </nav>
    </div>
  )
}
