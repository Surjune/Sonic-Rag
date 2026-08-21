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

  if (!providers) return null

  const local = providers.ollama
  const localReady = local.ready
  // Ollama answered but does not have the model: the one case the interface
  // can resolve itself, without asking anyone to open a terminal.
  const canPull = !localReady && (local.detail ?? '').includes('not pulled')

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

      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        title="What is the difference?"
        className={`rounded-md border p-1 transition ${
          localReady
            ? 'border-white/10 text-emerald-100/40 hover:text-emerald-100/80'
            : 'border-emerald-300/30 text-emerald-200/70 hover:text-emerald-200'
        }`}
      >
        <Info size={12} />
      </button>

      {open && (
        <div className="absolute top-full right-0 z-50 mt-2 w-[22rem] rounded-lg border border-white/15 bg-[#04170f] p-3 text-left shadow-xl">
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
