/**
 * Latency Analytics: percentiles and distribution over the current session.
 *
 * Samples come from real requests made in this session, so an empty state is
 * genuinely empty rather than seeded with invented numbers.
 */

import { Activity, Timer, Trash2 } from 'lucide-react'
import { useMemo } from 'react'
import { Badge, EmptyState, Panel, SectionTitle } from '../components/ui'
import type { LatencySample } from '../lib/types'

/** Nearest-rank percentile: every reported value is one that actually occurred. */
function percentile(values: number[], fraction: number): number {
  if (values.length === 0) return 0
  const ordered = [...values].sort((a, b) => a - b)
  if (fraction >= 1) return ordered[ordered.length - 1]
  const rank = Math.max(1, Math.min(ordered.length, Math.ceil(fraction * ordered.length)))
  return ordered[rank - 1]
}

const STAGES: { key: keyof LatencySample['latency']; label: string; budget?: number }[] = [
  { key: 'guardrail_input', label: 'Guardrail in', budget: 0.5 },
  { key: 'stt', label: 'STT' },
  { key: 'translate', label: 'Translate' },
  { key: 'embed', label: 'Embed' },
  { key: 'faiss', label: 'FAISS', budget: 5 },
  { key: 'guardrail_grounding', label: 'Grounding', budget: 2 },
  { key: 'llm_ttft', label: 'Groq TTFT' },
  { key: 'total', label: 'Total' },
]

function PercentileCard({
  label,
  value,
  tone,
}: {
  label: string
  value: number
  tone: 'cyan' | 'amber' | 'rose'
}) {
  const ring = {
    cyan: 'border-cyan-500/40 text-cyan-300',
    amber: 'border-amber-500/40 text-amber-300',
    rose: 'border-rose-500/40 text-rose-300',
  }[tone]
  return (
    <Panel className={`flex flex-col gap-1 border p-4 ${ring}`}>
      <span className="text-[10px] tracking-widest text-slate-500 uppercase">{label}</span>
      <span className="font-mono text-2xl tabular-nums">{value.toFixed(1)}</span>
      <span className="text-[10px] text-slate-500">milliseconds</span>
    </Panel>
  )
}

/** Histogram of total latency. SVG keeps it dependency-free. */
function Distribution({ values }: { values: number[] }) {
  const { bars, max, min } = useMemo(() => {
    if (values.length === 0) return { bars: [] as number[], max: 0, min: 0 }
    const lo = Math.min(...values)
    const hi = Math.max(...values)
    const buckets = 24
    const span = hi - lo || 1
    const counts = new Array<number>(buckets).fill(0)
    for (const value of values) {
      const index = Math.min(buckets - 1, Math.floor(((value - lo) / span) * buckets))
      counts[index] += 1
    }
    return { bars: counts, max: hi, min: lo }
  }, [values])

  if (bars.length === 0) return null
  const peak = Math.max(...bars, 1)

  return (
    <div>
      <div className="flex h-32 items-end gap-0.5">
        {bars.map((count, index) => (
          <div
            key={index}
            className="flex-1 rounded-t bg-gradient-to-t from-cyan-500/30 to-cyan-400/80"
            style={{ height: `${Math.max(2, (count / peak) * 100)}%` }}
            title={`${count} sample${count === 1 ? '' : 's'}`}
          />
        ))}
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10px] text-slate-500">
        <span>{min.toFixed(0)}ms</span>
        <span>{max.toFixed(0)}ms</span>
      </div>
    </div>
  )
}

export function Analytics({
  samples,
  onClear,
}: {
  samples: LatencySample[]
  onClear: () => void
}) {
  const totals = useMemo(() => samples.map((s) => s.latency.total), [samples])

  const stageRows = useMemo(() => {
    return STAGES.map((stage) => {
      const values = samples
        .map((sample) => sample.latency[stage.key])
        .filter((value): value is number => typeof value === 'number')
      return {
        ...stage,
        n: values.length,
        p50: percentile(values, 0.5),
        p70: percentile(values, 0.7),
        p90: percentile(values, 0.9),
        p100: percentile(values, 1),
      }
    }).filter((row) => row.n > 0)
  }, [samples])

  if (samples.length === 0) {
    return (
      <Panel className="h-full">
        <EmptyState message="No requests yet. Ask something in the Playground and the percentiles will build here from real measurements." />
      </Panel>
    )
  }

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto pr-1">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <PercentileCard label="P50 total" value={percentile(totals, 0.5)} tone="cyan" />
        <PercentileCard label="P70 total" value={percentile(totals, 0.7)} tone="cyan" />
        <PercentileCard label="P90 total" value={percentile(totals, 0.9)} tone="amber" />
        <PercentileCard label="P100 total" value={percentile(totals, 1)} tone="rose" />
      </div>

      <Panel className="p-4">
        <div className="mb-3 flex items-center justify-between">
          <SectionTitle
            icon={<Activity size={14} className="text-cyan-400" />}
            title="Total latency distribution"
            hint={`${samples.length} request${samples.length === 1 ? '' : 's'} this session`}
          />
          <button
            type="button"
            onClick={onClear}
            className="flex items-center gap-1.5 rounded border border-slate-700 px-2 py-1 text-[11px] text-slate-400 transition hover:border-rose-500/40 hover:text-rose-300"
          >
            <Trash2 size={12} /> Clear
          </button>
        </div>
        <Distribution values={totals} />
      </Panel>

      <Panel className="p-4">
        <SectionTitle
          icon={<Timer size={14} className="text-cyan-400" />}
          title="Per-stage percentiles"
          hint="nearest-rank"
        />
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-[10px] tracking-widest text-slate-500 uppercase">
              <tr>
                <th className="py-2 pr-3 font-medium">Stage</th>
                <th className="py-2 pr-3 text-right font-medium">n</th>
                <th className="py-2 pr-3 text-right font-medium">P50</th>
                <th className="py-2 pr-3 text-right font-medium">P70</th>
                <th className="py-2 pr-3 text-right font-medium">P90</th>
                <th className="py-2 text-right font-medium">P100</th>
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              {stageRows.map((row) => (
                <tr key={row.key} className="border-t border-slate-800/70">
                  <td className="py-2 pr-3 font-sans text-slate-300">
                    <span className="flex items-center gap-2">
                      {row.label}
                      {row.budget !== undefined && (
                        <Badge tone={row.p50 <= row.budget ? 'emerald' : 'rose'}>
                          ≤{row.budget}ms
                        </Badge>
                      )}
                    </span>
                  </td>
                  <td className="py-2 pr-3 text-right text-slate-500">{row.n}</td>
                  <td className="py-2 pr-3 text-right text-slate-200">{row.p50.toFixed(2)}</td>
                  <td className="py-2 pr-3 text-right text-slate-400">{row.p70.toFixed(2)}</td>
                  <td className="py-2 pr-3 text-right text-slate-400">{row.p90.toFixed(2)}</td>
                  <td className="py-2 text-right text-slate-500">{row.p100.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel className="p-4">
        <SectionTitle title="Recent requests" />
        <div className="mt-3 space-y-1.5">
          {samples
            .slice(-12)
            .reverse()
            .map((sample) => (
              <div
                key={sample.at}
                className="flex items-center justify-between gap-2 rounded border border-slate-800/70 px-2.5 py-1.5 text-xs"
              >
                <div className="flex items-center gap-2">
                  <Badge tone={sample.kind === 'voice' ? 'cyan' : 'slate'}>{sample.kind}</Badge>
                  <Badge tone={sample.blocked ? 'rose' : sample.grounded ? 'emerald' : 'amber'}>
                    {sample.blocked ? 'blocked' : sample.grounded ? 'grounded' : 'ungrounded'}
                  </Badge>
                </div>
                <span className="font-mono tabular-nums text-slate-400">
                  {sample.latency.total.toFixed(1)}ms
                </span>
              </div>
            ))}
        </div>
      </Panel>
    </div>
  )
}
