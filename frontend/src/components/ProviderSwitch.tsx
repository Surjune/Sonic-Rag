/**
 * Choose which model answers: hosted Groq, or one running on this machine.
 *
 * The two options are a genuine trade rather than a good and a bad choice, so
 * the control says what each costs. Groq needs nothing installed and pays a
 * network round trip on every question; local answers roughly five times
 * sooner and needs a 2GB download and a GPU to be worth having.
 *
 * The local option is offered only when it can actually answer. Installed,
 * running and model-pulled are three different states, and a switch that
 * silently fails is worse than no switch -- so when it is not ready the
 * control says which of the three is missing and, where it can, fixes it.
 */

import { Check, Cloud, Cpu, Download, Info, Loader2 } from 'lucide-react'
import { useState } from 'react'
import { pullLocalModel, type PullProgress } from '../lib/api'
import type { ProvidersResponse } from '../lib/types'

interface ProviderSwitchProps {
  providers: ProvidersResponse | null
  value: string
  onChange: (provider: string) => void
  /** Re-probe after a download, so the switch turns on by itself. */
  onRefresh: () => void
}

function formatBytes(bytes: number): string {
  if (!bytes) return ''
  return `${(bytes / 1e9).toFixed(2)} GB`
}

export function ProviderSwitch({ providers, value, onChange, onRefresh }: ProviderSwitchProps) {
  const [open, setOpen] = useState(false)
  const [progress, setProgress] = useState<PullProgress | null>(null)
  const [pulling, setPulling] = useState(false)
  const [pullError, setPullError] = useState<string | null>(null)
  const [seen, setSeen] = useState(() => {
    try {
      return localStorage.getItem('sonic-rag.providerSeen') === 'yes'
    } catch {
      return false
    }
  })

  if (!providers) return null

  const local = providers.ollama
  const localReady = local.ready
  // Ollama answered but does not have the model: the one case the interface
  // can resolve itself, without asking anyone to open a terminal.
  const canPull = !localReady && (local.detail ?? '').includes('not pulled')

  // Worth pointing at only while it is true: a faster backend is sitting
  // there unused, or is one download away, and nobody has looked yet. Once
  // local is chosen or the explainer has been read, the hint retires rather
  // than becoming decoration people learn to ignore.
  const unnoticed = !seen && value !== 'ollama' && (localReady || canPull)

  const markSeen = () => {
    setSeen(true)
    try {
      localStorage.setItem('sonic-rag.providerSeen', 'yes')
    } catch {
      // Storage being unavailable only means the hint shows again later.
    }
  }

  const startPull = async () => {
    setPulling(true)
    setPullError(null)
    setProgress(null)
    try {
      await pullLocalModel({
        onProgress: setProgress,
        onDone: () => {
          setPulling(false)
          onRefresh()
          onChange('ollama')
        },
        onError: (message) => {
          setPullError(message)
          setPulling(false)
        },
      })
    } catch {
      setPullError('Download failed')
      setPulling(false)
    }
  }

  const options = [
    { info: providers.groq, icon: Cloud, label: 'groq', enabled: providers.groq.ready },
    { info: local, icon: Cpu, label: 'local', enabled: localReady },
  ]

  return (
    <div className="relative flex items-center gap-1">
      <div className="flex items-center rounded-lg border border-white/10 p-0.5">
        {options.map(({ info, icon: Icon, label, enabled }) => {
          const active = value === info.id
          return (
            <button
              key={info.id}
              type="button"
              onClick={() => (enabled ? onChange(info.id) : setOpen(true))}
              title={
                enabled
                  ? `${info.label} — ${info.model}`
                  : info.detail || 'Not available on this machine'
              }
              className={`flex items-center gap-1.5 rounded-md px-2 py-1 font-mono text-[10px] tracking-wider whitespace-nowrap uppercase transition ${
                active
                  ? 'bg-white/10 text-emerald-100'
                  : enabled
                    ? 'text-emerald-100/45 hover:text-emerald-100/80'
                    : 'text-emerald-100/25 hover:text-emerald-100/45'
              }`}
            >
              <Icon size={12} />
              {label}
            </button>
          )
        })}
      </div>

      {/*
        The reason to care, on the control itself. A switch labelled only
        "groq / local" says nothing about why anyone would move it, and the
        thing worth knowing is the number. Shown only while local is unused,
        so it reads as an offer rather than a permanent ornament.
      */}
      {unnoticed && localReady && (
        <button
          type="button"
          onClick={() => {
            onChange('ollama')
            markSeen()
          }}
          className="animate-pulse rounded-md border border-emerald-300/45 bg-emerald-300/10 px-1.5 py-1 font-mono text-[9px] tracking-wider whitespace-nowrap text-emerald-200 uppercase transition hover:bg-emerald-300/20"
        >
          5× faster
        </button>
      )}
      {unnoticed && !localReady && canPull && (
        <button
          type="button"
          onClick={() => {
            setOpen(true)
            markSeen()
          }}
          className="animate-pulse rounded-md border border-emerald-300/45 bg-emerald-300/10 px-1.5 py-1 font-mono text-[9px] tracking-wider whitespace-nowrap text-emerald-200 uppercase transition hover:bg-emerald-300/20"
        >
          get 5× faster
        </button>
      )}

      <button
        type="button"
        onClick={() => {
          setOpen((o) => !o)
          markSeen()
        }}
        title="Where is the answer generated?"
        className={`relative rounded-md border p-1 transition ${
          unnoticed
            ? 'border-emerald-300/50 text-emerald-200'
            : 'border-white/10 text-emerald-100/40 hover:text-emerald-100/80'
        }`}
      >
        <Info size={12} />
        {/*
          The switch is easy to miss, and the thing worth noticing is that a
          five-times-faster backend is sitting there unused. The dot appears
          only while that is true: once local is selected, or once someone has
          opened this and read it, there is nothing left to point at.
        */}
        {unnoticed && (
          <span className="absolute -top-0.5 -right-0.5 flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-300 opacity-75" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-300" />
          </span>
        )}
      </button>

      {open && (
        <div className="absolute top-full right-0 z-50 mt-2 w-[23rem] rounded-xl border border-white/20 bg-[#031109] p-4 text-left shadow-[0_24px_60px_-12px_rgba(0,0,0,0.9)]">
          <p className="mb-3 font-mono text-[10px] tracking-[0.2em] text-emerald-100/45 uppercase">
            Where the answer is generated
          </p>
          {/* The trade, stated plainly, because neither option is simply better. */}
          <div className="mb-3 space-y-2">
            <div className="flex gap-2">
              <Cloud size={13} className="mt-0.5 shrink-0 text-emerald-100/50" />
              <p className="text-xs leading-relaxed text-emerald-100/70">
                <strong className="text-emerald-100">Groq</strong> — nothing to
                install, works anywhere. Every question pays a network round
                trip: <strong>~440ms</strong> before the first word.
              </p>
            </div>
            <div className="flex gap-2">
              <Cpu size={13} className="mt-0.5 shrink-0 text-emerald-200/70" />
              <p className="text-xs leading-relaxed text-emerald-100/70">
                <strong className="text-emerald-100">Local</strong> — runs on
                your own machine, no round trip:{' '}
                <strong className="text-emerald-200">~20-80ms</strong> to first
                word, about five times sooner. Needs a 2GB download and a GPU to
                be quick, and it is a 3B model rather than a 20B one.
              </p>
            </div>
          </div>

          {localReady ? (
            <p className="flex items-center gap-1.5 border-t border-white/10 pt-2 text-[11px] text-emerald-200/80">
              <Check size={12} /> {local.model} is ready on this machine.
            </p>
          ) : (
            <div className="border-t border-white/10 pt-2">
              <p className="mb-2 text-[11px] leading-relaxed text-emerald-100/60">
                {local.detail || 'Ollama is not available.'}
              </p>

              {canPull ? (
                <>
                  {/* The one case the page can fix by itself. */}
                  <button
                    type="button"
                    onClick={() => void startPull()}
                    disabled={pulling}
                    className="flex w-full items-center justify-center gap-2 rounded-md border border-emerald-300/40 px-3 py-2 text-xs font-medium text-emerald-200 transition hover:bg-emerald-300/10 disabled:opacity-60"
                  >
                    {pulling ? <Loader2 size={13} className="animate-spin" /> : <Download size={13} />}
                    {pulling ? 'Downloading…' : `Download ${local.model} (2.0 GB)`}
                  </button>

                  {progress && (
                    <div className="mt-2">
                      <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/10">
                        <div
                          className="h-full rounded-full bg-emerald-300 transition-[width] duration-300"
                          style={{ width: `${progress.percent ?? 0}%` }}
                        />
                      </div>
                      <p className="mt-1 font-mono text-[10px] text-emerald-100/50">
                        {progress.status}
                        {progress.total
                          ? ` · ${formatBytes(progress.completed)} / ${formatBytes(progress.total)}`
                          : ''}
                      </p>
                    </div>
                  )}
                </>
              ) : (
                <ol className="space-y-1.5 text-[11px] text-emerald-100/70">
                  <li>
                    1. Install{' '}
                    <a
                      href={local.install_url}
                      target="_blank"
                      rel="noreferrer"
                      className="underline decoration-dotted underline-offset-2 hover:text-emerald-200"
                    >
                      ollama.com/download
                    </a>{' '}
                    <span className="text-emerald-100/40">
                      — on the machine running this backend
                    </span>
                  </li>
                  <li>
                    2. Reload this page. If the model is missing, a download
                    button appears here.
                  </li>
                </ol>
              )}

              {pullError && (
                <p className="mt-2 text-[11px] text-rose-300/80">{pullError}</p>
              )}
            </div>
          )}

          <button
            type="button"
            onClick={() => setOpen(false)}
            className="mt-3 font-mono text-[10px] tracking-wider text-emerald-100/40 uppercase hover:text-emerald-100/70"
          >
            close
          </button>
        </div>
      )}
    </div>
  )
}
