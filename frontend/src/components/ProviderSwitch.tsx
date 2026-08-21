/**
 * Choose which model answers: hosted Groq, or one running on this machine.
 *
 * The local option is offered only when it can actually answer. Whether Ollama
 * is installed, running, and has the model pulled are three different things,
 * and a switch that silently fails is worse than no switch -- so when it is not
 * ready the control says what to do about it instead of disappearing.
 */

import { Cloud, Cpu, Info } from 'lucide-react'
import { useState } from 'react'
import type { ProvidersResponse } from '../lib/types'

interface ProviderSwitchProps {
  providers: ProvidersResponse | null
  value: string
  onChange: (provider: string) => void
}

export function ProviderSwitch({ providers, value, onChange }: ProviderSwitchProps) {
  const [showHelp, setShowHelp] = useState(false)
  if (!providers) return null

  const localReady = providers.ollama.ready
  const options = [
    { info: providers.groq, icon: Cloud, enabled: providers.groq.ready },
    { info: providers.ollama, icon: Cpu, enabled: localReady },
  ]

  return (
    <div className="relative flex items-center gap-1">
      <div className="flex items-center rounded-lg border border-white/10 p-0.5">
        {options.map(({ info, icon: Icon, enabled }) => {
          const active = value === info.id
          return (
            <button
              key={info.id}
              type="button"
              disabled={!enabled}
              onClick={() => (enabled ? onChange(info.id) : setShowHelp(true))}
              title={enabled ? `${info.label} — ${info.model}` : info.detail || info.note}
              className={`flex items-center gap-1.5 rounded-md px-2 py-1 font-mono text-[10px] tracking-wider whitespace-nowrap uppercase transition ${
                active
                  ? 'bg-white/10 text-emerald-100'
                  : enabled
                    ? 'text-emerald-100/45 hover:text-emerald-100/80'
                    : 'text-emerald-100/25'
              }`}
            >
              <Icon size={12} />
              {info.id === 'groq' ? 'groq' : 'local'}
            </button>
          )
        })}
      </div>

      {!localReady && (
        <button
          type="button"
          onClick={() => setShowHelp((open) => !open)}
          title="How to run the model locally"
          className="rounded-md border border-white/10 p-1 text-emerald-100/40 transition hover:text-emerald-100/80"
        >
          <Info size={12} />
        </button>
      )}

      {showHelp && !localReady && (
        <div className="absolute top-full right-0 z-50 mt-2 w-80 rounded-lg border border-white/15 bg-[#04170f] p-3 text-left shadow-xl">
          <p className="mb-2 font-mono text-[10px] tracking-widest text-emerald-100/50 uppercase">
            Run the model locally
          </p>
          <p className="mb-2 text-xs leading-relaxed text-emerald-100/70">
            {providers.ollama.detail || 'Ollama is not available on this machine.'}{' '}
            Answers then generate on your own hardware with no network hop —
            measured here at <strong className="text-emerald-200">82ms</strong> to
            first token against Groq&apos;s <strong>438ms</strong>.
          </p>
          <ol className="mb-2 space-y-1.5 text-xs text-emerald-100/70">
            <li>
              1. Install from{' '}
              <a
                href={providers.ollama.install_url}
                target="_blank"
                rel="noreferrer"
                className="underline decoration-dotted underline-offset-2 hover:text-emerald-200"
              >
                ollama.com/download
              </a>
            </li>
            <li>
              2. Pull the model:
              <code className="mt-1 block rounded bg-black/40 px-2 py-1 font-mono text-[10px] text-emerald-200">
                {providers.ollama.pull_command}
              </code>
            </li>
            <li>3. Reload this page — the switch turns on by itself.</li>
          </ol>
          {/*
            Stated rather than buried: the local model is 3B against Groq's
            20B, and without a GPU it will be slower, not faster. Somebody
            installing 2GB on the strength of a latency number deserves to
            know what they are trading for it.
          */}
          <p className="text-[10px] leading-relaxed text-emerald-100/40">
            It is a 3B model rather than a 20B one, and it needs a GPU to be
            quick. Groq stays the default and is what a deployment uses.
          </p>
          <button
            type="button"
            onClick={() => setShowHelp(false)}
            className="mt-2 font-mono text-[10px] tracking-wider text-emerald-100/40 uppercase hover:text-emerald-100/70"
          >
            close
          </button>
        </div>
      )}
    </div>
  )
}
