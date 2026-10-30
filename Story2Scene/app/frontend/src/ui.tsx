import type { ReactNode } from 'react'
import type { StageState } from './types'

const cls: Record<string, string> = {
  pending: 'bg-slate-200 text-slate-700', running: 'bg-blue-100 text-blue-800',
  done: 'bg-green-100 text-green-800', failed: 'bg-red-100 text-red-800',
  awaiting: 'bg-purple-100 text-purple-800', awaiting_selection: 'bg-purple-100 text-purple-800',
  queued: 'bg-amber-100 text-amber-800',
}
export function StatusBadge({ s }: { s: StageState | string }) {
  return <span className={`rounded px-2 py-0.5 text-xs font-medium ${cls[s] ?? cls.pending}`}>{s}</span>
}
export function ModeBadge({ mode }: { mode?: string }) {
  if (!mode) return null
  const warn = /dummy|fallback/i.test(mode)
  return (
    <span className={`rounded px-2 py-0.5 text-xs ${warn ? 'bg-orange-100 text-orange-800 ring-1 ring-orange-300' : 'bg-indigo-100 text-indigo-800'}`}>
      {warn ? '⚠ ' : ''}{mode}
    </span>
  )
}
export function Panel({ title, state, mode, children }: { title: string; state: StageState; mode?: string; children: ReactNode }) {
  return (
    <section className="min-w-0 rounded-lg border border-slate-300 bg-white p-3">
      <header className="mb-2 flex flex-wrap items-center gap-2">
        <h2 className="font-semibold">{title}</h2><StatusBadge s={state} /><ModeBadge mode={mode} />
      </header>
      {children}
    </section>
  )
}
// Pipeline progress (backend estimate: finished phases + the current one, filled by its typical length)
export function ProgressBar({ p, status }: { p?: { pct: number; label: string; eta_sec: number }; status: string }) {
  if (!p) return null
  const failed = status === 'failed' || status === 'interrupted'
  const eta = p.eta_sec >= 90 ? `about ${Math.round(p.eta_sec / 60)} min left` : p.eta_sec > 0 ? 'about a minute left' : ''
  return (
    <div className="space-y-1" aria-live="polite">
      <div className="flex flex-wrap justify-between gap-2 text-xs text-slate-600">
        <span>{p.label}</span>
        <span className="tabular-nums">{Math.floor(p.pct)}%{status === 'running' && eta ? ` · ${eta}` : ''}</span>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-200" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.floor(p.pct)}>
        <div className={`h-full rounded-full transition-[width] duration-1000 ease-linear ${failed ? 'bg-red-500' : p.pct >= 100 ? 'bg-green-600' : 'bg-blue-600'} ${status === 'running' ? 'animate-pulse' : ''}`}
          style={{ width: `${Math.max(2, Math.min(100, p.pct))}%` }} />
      </div>
    </div>
  )
}
