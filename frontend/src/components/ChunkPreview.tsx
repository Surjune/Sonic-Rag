/**
 * Side-by-side chunk preview across strategies.
 *
 * Runs on the server as pure string work with no embedding, so the same
 * passage can be re-split across all four strategies instantly and for free.
 * Seeing where fixed-size cuts mid-word next to where semantic stops at a
 * danda makes the difference concrete in a way a recall table cannot.
 */

import { Scissors } from 'lucide-react'
import { Badge, Panel, SectionTitle } from './ui'
import type { PreviewResponse, StrategyPreview } from '../lib/types'

const LABELS: Record<string, string> = {
  fixed: 'Fixed size',
  fixed_overlap: 'Fixed + overlap',
  semantic: 'Semantic',
  hierarchical: 'Hierarchical',
}

/** Alternating tints so chunk boundaries are visible at a glance. */
const TINTS = [
  'border-cyan-500/30 bg-cyan-500/5',
  'border-violet-500/30 bg-violet-500/5',
  'border-emerald-500/30 bg-emerald-500/5',
  'border-amber-500/30 bg-amber-500/5',
]

function StrategyColumn({ preview }: { preview: StrategyPreview }) {
  return (
    <div className="flex min-w-0 flex-col rounded-lg border border-slate-800 bg-slate-900/40">
      <div className="border-b border-slate-800 px-3 py-2">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-medium text-slate-200">
            {LABELS[preview.strategy] ?? preview.strategy}
          </span>
          <Badge tone="slate">{preview.count} chunks</Badge>
        </div>
        <div className="mt-1 font-mono text-[10px] text-slate-500">
          avg {preview.mean_chars} · min {preview.min_chars} · max {preview.max_chars}
        </div>
      </div>

      {/*
        The height cap belongs to the side-by-side view.

        Four columns abreast need to agree on a height or the row is ragged, so
        320px with a scrollbar is right from lg. Stacked on a phone it is
        wrong: each strategy became its own little scroll box inside a page
        that already scrolls, and the card at the boundary was sliced through
        the middle with nothing to say it continued. The passages here are
        short, so below lg the column simply shows all of them.
      */}
      <div className="space-y-1.5 p-2 lg:max-h-80 lg:overflow-y-auto">
        {preview.chunks.map((chunk) => (
          <div
            key={chunk.index}
            className={`rounded border p-2 ${TINTS[chunk.index % TINTS.length]}`}
          >
            <div className="mb-1 flex items-center justify-between font-mono text-[9px] text-slate-500">
              <span>#{chunk.index}</span>
              <span>
                {chunk.char_start}–{chunk.char_end} · {chunk.length}c
              </span>
            </div>
            <p className="text-[11px] leading-relaxed break-words text-slate-300">{chunk.text}</p>
            {chunk.embed_text !== `[en] ${chunk.text}` && (
              // Hierarchical injects parent context; showing the embedded form
              // makes clear that what is searched is not what is displayed.
              <p className="mt-1.5 border-t border-slate-700/60 pt-1.5 text-[10px] leading-relaxed text-slate-500 italic">
                embedded: {chunk.embed_text}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

export function ChunkPreview({ preview }: { preview: PreviewResponse | null }) {
  if (!preview) return null

  return (
    <Panel className="p-4">
      <SectionTitle
        icon={<Scissors size={14} className="text-cyan-400" />}
        title="Chunk boundaries by strategy"
        hint={`${preview.source_chars} source chars · no embedding, no cost`}
      />
      <div className="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-2 xl:grid-cols-4">
        {preview.strategies.map((strategy) => (
          <StrategyColumn key={strategy.strategy} preview={strategy} />
        ))}
      </div>
      {/*
        Two even columns on a phone rather than a wrapping row. The four labels
        are different lengths, so wrapping left a ragged right edge and put the
        break in a different place at every width; a grid gives them one edge
        to line up on.
      */}
      <div className="mt-3 grid grid-cols-2 gap-2 font-mono text-[10px] text-slate-500 sm:flex sm:flex-wrap">
        {Object.entries(preview.latency)
          .filter(([key]) => key !== 'total')
          .map(([name, ms]) => (
            <Badge key={name} tone="slate">
              {name} {ms.toFixed(3)}ms
            </Badge>
          ))}
      </div>
    </Panel>
  )
}
