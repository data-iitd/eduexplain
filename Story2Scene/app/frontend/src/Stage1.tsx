import { useState } from 'react'
import type { Stage1 } from './types'
import { figureUrl, isFig, submitSelection } from './api'

const short = (t?: string, n = 160) => (t && t.length > n ? t.slice(0, n) + '…' : t ?? '')

// Manual mode: every retrieved candidate with a checkbox; the ticked ones (in ticking order) become the gold set
function ManualPicker({ s, jobId }: { s: Stage1; jobId: string }) {
  const [picked, setPicked] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const suggested = new Set(s.selection?.shortlist ?? [])
  const toggle = (id: string) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]))
  async function go() {
    setBusy(true); setErr(null)
    try { await submitSelection(jobId, picked) } catch (e) { setErr(e instanceof Error ? e.message : String(e)); setBusy(false) }
  }
  return (
    <div className="space-y-2 text-sm">
      <p className="rounded bg-purple-50 p-2 text-purple-900">
        Tick the resources the storyboard should use. setCore suggests about {s.selection?.suggested_n ?? '?'};
        its shortlist is marked <i>suggested</i>.
      </p>
      <div className="flex flex-wrap items-center gap-2">
        <button onClick={() => setPicked([...(s.selection?.shortlist ?? [])])} className="rounded border px-2 py-1 text-xs">Tick suggested</button>
        <button onClick={() => setPicked([])} className="rounded border px-2 py-1 text-xs">Clear</button>
        <button onClick={go} disabled={busy || picked.length === 0} className="rounded bg-blue-600 px-3 py-1 text-white disabled:bg-slate-400">
          {busy ? 'Submitting…' : `Use ${picked.length} selected →`}
        </button>
      </div>
      {err && <p role="alert" className="text-red-700">{err}</p>}
      <ul className="max-h-[32rem] space-y-1 overflow-y-auto">
        {(s.retrieved ?? []).map((r, i) => {
          const on = picked.includes(r.resource_id)
          return (
            <li key={r.resource_id}>
              <label className={`flex cursor-pointer gap-2 rounded border p-1.5 ${on ? 'border-green-400 bg-green-50' : 'border-slate-200'}`}>
                <input type="checkbox" checked={on} onChange={() => toggle(r.resource_id)} className="mt-1 shrink-0" />
                {isFig(r.resource_id) && <img src={figureUrl(r.resource_id)} alt={r.resource_id} loading="lazy" className="h-14 w-14 shrink-0 rounded border bg-white object-contain" />}
                <span className="min-w-0">
                  <span className="flex flex-wrap items-center gap-1 text-xs">
                    <span className="text-slate-400">{i + 1}.</span>
                    <span className="font-mono font-semibold">{r.resource_id}</span>
                    <span className="text-slate-500">{r.resource_type}</span>
                    <span className="tabular-nums">{Number(r.score).toFixed(3)}</span>
                    {suggested.has(r.resource_id) && <span className="rounded bg-blue-100 px-1 text-blue-800">suggested</span>}
                    {on && <span className="rounded bg-green-200 px-1 text-green-900">#{picked.indexOf(r.resource_id) + 1}</span>}
                  </span>
                  <span className="block break-words text-xs text-slate-600">{short(r.text, 220)}</span>
                </span>
              </label>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

export default function Stage1View({ s, jobId, awaiting }: { s: Stage1 | null; jobId?: string; awaiting?: boolean }) {
  if (!s) return <p className="text-sm text-slate-500">No data yet.</p>
  if (s.awaiting_selection && awaiting && jobId) return <ManualPicker s={s} jobId={jobId} />
  const sel = new Set(s.selection?.selected ?? [])
  const short_ = new Set(s.selection?.shortlist ?? [])
  const gold = (s.gold_set?.gold_sets ?? []).flatMap((g) => g.resources_used ?? [])
  const byId = new Map(gold.map((g) => [g.resource_id, g]))
  const retrievedById = new Map((s.retrieved ?? []).map((r) => [r.resource_id, r]))
  const selectedIds = s.selection?.selected?.length ? s.selection.selected : gold.map((g) => g.resource_id)
  return (
    <div className="space-y-3 text-sm">
      {s.selection?.fell_back && <p className="rounded bg-orange-50 p-2 text-orange-800">Selection fell back{s.selection.problem ? `: ${s.selection.problem}` : ''}</p>}
      <h3 className="font-medium">Gold set ({selectedIds.length}){s.selection?.parse_path === 'manual' ? ' · chosen by you' : ''}</h3>
      <ul className="space-y-2">
        {selectedIds.map((id) => {
          const g = byId.get(id); const r = retrievedById.get(id)
          const text = g?.summary || g?.description || g?.text || r?.text
          return (
            <li key={id} className="flex gap-2 rounded border border-green-300 bg-green-50 p-2">
              {isFig(id) && <img src={figureUrl(id)} alt={id} loading="lazy" className="h-16 w-16 shrink-0 rounded border bg-white object-contain" />}
              <div className="min-w-0">
                <div className="flex flex-wrap gap-1 font-mono text-xs"><b>{id}</b><span className="text-slate-500">{g?.resource_type ?? r?.resource_type}</span></div>
                <p className="break-words">{short(text)}</p>
              </div>
            </li>
          )
        })}
      </ul>
      <h3 className="font-medium">Retrieved top {s.retrieved?.length ?? 0}{s.pool_size && s.pool_size !== s.retrieved?.length ? ` of ${s.pool_size}` : ''}</h3>
      <ul className="max-h-96 space-y-1 overflow-y-auto">
        {(s.retrieved ?? []).map((r) => (
          <li key={r.resource_id} className={`rounded border p-1.5 ${sel.has(r.resource_id) ? 'border-green-400 bg-green-50' : 'border-slate-200'}`}>
            <div className="flex flex-wrap items-center gap-1 text-xs">
              <span className="font-mono font-semibold">{r.resource_id}</span>
              <span className="text-slate-500">{r.resource_type}</span>
              <span className="tabular-nums">{Number(r.score).toFixed(3)}</span>
              {short_.has(r.resource_id) && <span className="rounded bg-blue-100 px-1 text-blue-800">shortlist</span>}
              {sel.has(r.resource_id) && <span className="rounded bg-green-200 px-1 text-green-900">selected</span>}
            </div>
            <p className="break-words text-xs text-slate-600">{short(r.text, 100)}</p>
          </li>
        ))}
      </ul>
      {s.seconds != null && <p className="text-xs text-slate-500">{s.seconds.toFixed(1)} s</p>}
    </div>
  )
}
