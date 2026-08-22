/**
 * Chunking strategy comparison: retrieval quality against index cost.
 *
 * Recall and index size are shown side by side deliberately. A strategy that
 * wins recall by tripling the vector count has not won for free, and on a
 * zero-budget host memory is the binding constraint.
 */

import { Award, Database } from 'lucide-react'
import { Badge, EmptyState, Panel, SectionTitle } from './ui'
import type { CompareResponse, StrategyScore } from '../lib/types'

const LABELS: Record<string, string> = {
  fixed: 'Fixed size',
  fixed_overlap: 'Fixed + overlap',
  semantic: 'Semantic',
  hierarchical: 'Hierarchical',
}

/** Bar scaled against the best score in the column. */
function ScoreBar({ value, best }: { value: number; best: number }) {
  const share = best > 0 ? (value / best) * 100 : 0
  const isBest = value >= best && best > 0
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-14 overflow-hidden rounded-full bg-slate-800">
        <div
          className={`h-full rounded-full ${isBest ? 'bg-emerald-400' : 'bg-cyan-500/60'}`}
          style={{ width: `${Math.max(2, share)}%` }}
        />
      </div>
      <span
        className={`font-mono text-xs tabular-nums ${
          isBest ? 'text-emerald-300' : 'text-slate-300'
        }`}
      >
        {value.toFixed(3)}
      </span>
    </div>
  )
}

export function StrategyMatrix({ report }: { report: CompareResponse | null }) {
  if (!report) {
    return (
      <Panel className="p-4">
        <EmptyState message="Loading strategy comparison…" />
      </Panel>
    )
  }

  if (!report.available || !report.strategies?.length) {
    return (
      <Panel className="p-4">
        <SectionTitle
          icon={<Award size={14} className="text-amber-400" />}
          title="Strategy comparison"
        />
        <p className="mt-2 text-xs leading-relaxed text-slate-400">
          {report.message ?? 'No comparison available.'} Embedding four indices takes minutes, so
          it is computed offline rather than served from a request — and absent results say so
          rather than showing invented numbers.
        </p>
        {report.how_to_generate && (
          <pre className="mt-2 overflow-x-auto rounded border border-slate-800 bg-slate-950 p-2 font-mono text-[11px] text-cyan-300">
            {report.how_to_generate}
          </pre>
        )}
      </Panel>
    )
  }

  const strategies: StrategyScore[] = report.strategies
  const bestRecall = Math.max(...strategies.map((s) => s.recall['@5'] ?? 0))
  const bestMrr = Math.max(...strategies.map((s) => s.mrr5))
  const smallestIndex = Math.min(...strategies.map((s) => s.chunks))
  const baseline = strategies.find((s) => s.name === report.baseline)

  return (
    <Panel className="p-4">
      <SectionTitle
        icon={<Award size={14} className="text-amber-400" />}
        title="Strategy comparison"
        hint={`${report.queries} queries · ${report.passages} passages · gold labels from is_selected`}
      />

      {/*
        Seven columns will not fit a phone and should not try -- squeezed to
        393px the recall bars become slivers and the descriptions become one
        word per line. It scrolls sideways instead, which is fine as long as
        you can tell that it does: before this the table was simply cut off
        mid-number at the right edge, which reads as a broken table rather than
        as more table. The fade says there is more, and the line underneath
        says what.
      */}
      <div className="relative mt-3">
        <div className="overflow-x-auto">
        <table className="w-full min-w-[680px] text-left text-xs">
          <thead className="text-[10px] tracking-widest text-slate-500 uppercase">
            <tr>
              <th className="py-2 pr-3 font-medium">Strategy</th>
              <th className="py-2 pr-3 text-right font-medium">Chunks</th>
              <th className="py-2 pr-3 text-right font-medium">Vectors MB</th>
              <th className="py-2 pr-3 text-right font-medium">Avg chars</th>
              <th className="py-2 pr-3 font-medium">Recall@5</th>
              <th className="py-2 pr-3 font-medium">MRR@5</th>
              <th className="py-2 text-right font-medium">Search</th>
            </tr>
          </thead>
          <tbody>
            {strategies.map((entry) => {
              const recall5 = entry.recall['@5'] ?? 0
              const growth = baseline ? entry.chunks / baseline.chunks : 1
              return (
                <tr key={entry.name} className="border-t border-slate-800/70 align-middle">
                  <td className="py-2.5 pr-3">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-slate-200">
                        {LABELS[entry.name] ?? entry.name}
                      </span>
                      {entry.name === report.baseline && <Badge tone="slate">baseline</Badge>}
                      {entry.chunks === smallestIndex && <Badge tone="cyan">smallest</Badge>}
                    </div>
                    <div className="mt-0.5 max-w-sm text-[10px] leading-snug text-slate-500">
                      {entry.description}
                    </div>
                  </td>
                  <td className="py-2.5 pr-3 text-right font-mono tabular-nums text-slate-300">
                    {entry.chunks.toLocaleString()}
                    {baseline && growth !== 1 && (
                      <span
                        className={`ml-1 text-[10px] ${
                          growth > 1 ? 'text-amber-400' : 'text-emerald-400'
                        }`}
                      >
                        {growth > 1 ? '+' : ''}
                        {((growth - 1) * 100).toFixed(0)}%
                      </span>
                    )}
                  </td>
                  <td className="py-2.5 pr-3 text-right font-mono tabular-nums text-slate-400">
                    {(entry.vector_bytes / 1e6).toFixed(2)}
                  </td>
                  <td className="py-2.5 pr-3 text-right font-mono tabular-nums text-slate-400">
                    {entry.mean_chunk_chars.toFixed(0)}
                  </td>
                  <td className="py-2.5 pr-3">
                    <ScoreBar value={recall5} best={bestRecall} />
                  </td>
                  <td className="py-2.5 pr-3">
                    <ScoreBar value={entry.mrr5} best={bestMrr} />
                  </td>
                  <td className="py-2.5 text-right font-mono tabular-nums text-slate-400">
                    {entry.search_ms_p50.toFixed(2)}ms
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        </div>

        {/* Sits over the scroll container, not inside it, so it stays pinned to
            the right edge instead of scrolling away with the content. */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-y-0 right-0 w-12 bg-gradient-to-l from-[color-mix(in_srgb,var(--color-panel)_92%,transparent)] to-transparent lg:hidden"
        />
      </div>

      <p className="mt-1.5 font-mono text-[10px] tracking-wider text-slate-500 uppercase lg:hidden">
        Swipe the table for recall, MRR and search time →
      </p>

      <div className="mt-3 flex items-start gap-2 rounded border border-slate-800 bg-slate-900/40 p-2.5">
        <Database size={13} className="mt-0.5 shrink-0 text-cyan-400" />
        <p className="text-[11px] leading-relaxed text-slate-400">
          Ground truth is the corpus's own <span className="font-mono">is_selected</span> flag, so
          nothing here is scored against the system's own opinion. Recall is reported next to index
          size because overlap buys coverage by duplicating text — a win on recall that triples
          memory is not free on a zero-budget host.
        </p>
      </div>
    </Panel>
  )
}
