/** Shared primitives, so repeated class strings live in one place. */

import type { ReactNode } from 'react'

export function Panel({
  children,
  className = '',
}: {
  children: ReactNode
  className?: string
}) {
  return <div className={`panel rounded-xl ${className}`}>{children}</div>
}

export function SectionTitle({ icon, title, hint }: { icon?: ReactNode; title: string; hint?: string }) {
  /*
    The hint wraps to its own line rather than competing for the title's.

    Without `flex-wrap` the two shared one row and both shrank: at 360px
    "Explore retrieval" broke into two lines with "retrieval only — no model
    call" breaking into two beside it, baseline-aligned, so neither second line
    lined up with anything. Wrapping only changes what was already overflowing
    -- wherever the row fits, which is everywhere above a phone, this renders
    identically.
  */
  return (
    <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
      {icon}
      <h2 className="text-sm font-semibold tracking-wide text-slate-200 uppercase">{title}</h2>
      {hint && <span className="text-xs text-slate-500">{hint}</span>}
    </div>
  )
}

/** A latency readout. Colour is by budget, not decoration. */
export function Metric({
  label,
  value,
  unit = 'ms',
  budgetMs,
  mono = true,
  accent = false,
  className = '',
}: {
  label: string
  value: number | undefined
  unit?: string
  budgetMs?: number
  mono?: boolean
  /**
   * Marks the headline figure in a row of stages.
   *
   * Six equal readouts give no clue which one answers "how fast was that".
   * This is the one the project is judged on, so it is the one that is bigger
   * and brighter -- the others are the breakdown that explains it.
   */
  accent?: boolean
  /** Lets a caller hide a stage on small screens without wrapping it. */
  className?: string
}) {
  const missing = value === undefined
  const over = budgetMs !== undefined && !missing && value > budgetMs
  const tone = missing
    ? 'text-slate-600'
    : over
      ? 'text-rose-400'
      : accent
        ? 'text-[var(--color-goa-yellow)]'
        : 'text-emerald-300'
  return (
    <div className={`flex flex-col gap-0.5 ${className}`}>
      <span
        className={`text-[10px] uppercase tracking-widest ${
          accent ? 'text-emerald-100/70' : 'text-slate-500'
        }`}
      >
        {label}
      </span>
      <span
        className={`${tone} ${mono ? 'font-mono' : ''} tabular-nums ${
          accent ? 'text-base font-semibold leading-tight' : 'text-sm'
        }`}
      >
        {missing ? '—' : `${value < 1 ? value.toFixed(2) : value.toFixed(0)}${unit}`}
      </span>
    </div>
  )
}

export function Badge({
  children,
  tone = 'slate',
}: {
  children: ReactNode
  tone?: 'slate' | 'emerald' | 'rose' | 'amber' | 'cyan'
}) {
  const tones: Record<string, string> = {
    slate: 'bg-slate-800/60 text-slate-300 border-slate-700',
    emerald: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
    rose: 'bg-rose-500/10 text-rose-300 border-rose-500/30',
    amber: 'bg-amber-500/10 text-amber-300 border-amber-500/30',
    cyan: 'bg-cyan-500/10 text-cyan-300 border-cyan-500/30',
  }
  return (
    <span className={`rounded border px-1.5 py-0.5 font-mono text-[10px] ${tones[tone]}`}>
      {children}
    </span>
  )
}

/** Horizontal similarity meter with the grounding threshold marked. */
export function ScoreMeter({ score, threshold }: { score: number; threshold: number }) {
  const percent = Math.max(0, Math.min(100, score * 100))
  const markerPercent = Math.max(0, Math.min(100, threshold * 100))
  const passes = score >= threshold
  return (
    <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
      <div
        className={`h-full rounded-full ${passes ? 'bg-emerald-400' : 'bg-rose-400'}`}
        style={{ width: `${percent}%` }}
      />
      {/* The threshold marker makes a score legible without reading the number. */}
      <div
        className="absolute top-0 h-full w-px bg-slate-400"
        style={{ left: `${markerPercent}%` }}
        title={`threshold ${threshold}`}
      />
    </div>
  )
}

export function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex h-full items-center justify-center p-8 text-center text-sm text-slate-500">
      {message}
    </div>
  )
}
