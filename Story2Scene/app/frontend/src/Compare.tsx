import { useEffect, useRef, useState } from 'react'
import type { Job, JobSummary } from './types'
import { apiPath, figureUrl, getJob, isFig } from './api'
import Stage2View from './Stage2'

// Watch and compare: two finished configurations of the same question side by side (video, Stage 1 set, storyboard).
// Resources in both Stage 1 sets are marked "in both"; "Play both" starts and pauses the two videos together.
const short = (t?: string, n = 120) => (t && t.length > n ? t.slice(0, n) + '…' : t ?? '')
export const configLabel = (j: JobSummary, s1: Record<string, string>, s2: Record<string, string>) =>
  `${j.stage1_label ?? s1[j.selection ?? 'auto'] ?? j.selection} + ${j.stage2_label ?? s2[j.stage2 ?? 'set2story'] ?? j.stage2}`

function selected(j: Job | null) {
  const s = j?.stage1
  const gold = (s?.gold_set?.gold_sets ?? []).flatMap((g) => g.resources_used ?? [])
  const byId = new Map(gold.map((g) => [g.resource_id, g]))
  const ids = s?.selection?.selected?.length ? s.selection.selected : gold.map((g) => g.resource_id)
  return ids.map((id) => { const g = byId.get(id); return { id, text: g?.summary || g?.description || g?.text || '' } })
}

function Side({ job, other, label, tone, vref }: { job: Job | null; other: Set<string>; label: string; tone: string; vref: (v: HTMLVideoElement | null) => void }) {
  if (!job) return <p className="text-sm text-slate-500">Loading…</p>
  const set = selected(job)
  return (
    <div className="min-w-0 space-y-2">
      <p className={`rounded px-2 py-1 text-sm font-semibold ${tone}`}>{label}</p>
      {job.video_url
        ? <video ref={vref} src={apiPath(job.video_url)} controls preload="metadata" className="aspect-video w-full rounded bg-black" />
        : <p className="text-sm text-slate-500">No video.</p>}
      <div>
        <p className="text-xs font-semibold">Stage 1 set ({set.length})</p>
        <ul className="mt-1 space-y-1">
          {set.map((r) => (
            <li key={r.id} className={`flex gap-2 rounded border p-1.5 text-xs ${other.has(r.id) ? 'border-green-300 bg-green-50' : 'border-amber-300 bg-amber-50'}`}>
              {isFig(r.id) && <img src={figureUrl(r.id)} alt={r.id} loading="lazy" className="h-10 w-10 shrink-0 rounded border bg-white object-contain" />}
              <span className="min-w-0">
                <span className="font-mono font-semibold">{r.id}</span>{' '}
                <span className={other.has(r.id) ? 'text-green-800' : 'text-amber-800'}>{other.has(r.id) ? 'in both' : 'only here'}</span>
                <span className="block break-words text-slate-600">{short(r.text)}</span>
              </span>
            </li>
          ))}
        </ul>
      </div>
      <details open className="text-sm">
        <summary className="cursor-pointer text-xs font-semibold">Stage 2 storyboard</summary>
        <div className="mt-1 max-h-[30rem] overflow-y-auto pr-1"><Stage2View s={job.stage2} /></div>
      </details>
    </div>
  )
}

export default function Compare({ job, jobs, s1, s2 }: { job: Job; jobs: JobSummary[]; s1: Record<string, string>; s2: Record<string, string> }) {
  const qid = job.spec?.query_id as string | undefined, q = String(job.spec?.query ?? '')
  const same = jobs.filter((j) => j.status === 'done' && (qid ? j.query_id === qid : j.query === q))
  const others = same.filter((j) => j.id !== job.id)
  const [leftId, setLeftId] = useState(job.id)
  const [rightId, setRightId] = useState<string | null>(others[0]?.id ?? null)
  const [left, setLeft] = useState<Job | null>(job)
  const [right, setRight] = useState<Job | null>(null)
  const va = useRef<HTMLVideoElement | null>(null), vb = useRef<HTMLVideoElement | null>(null)
  const [playing, setPlaying] = useState(false)
  // follow the job opened above
  useEffect(() => { setLeftId(job.id); setLeft(job); setRightId((r) => (r && r !== job.id ? r : others[0]?.id ?? null)) }, [job.id]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (leftId === job.id) setLeft(job); else getJob(leftId).then(setLeft).catch(() => setLeft(null)) }, [leftId, job])
  useEffect(() => { setRight(null); if (rightId) getJob(rightId).then(setRight).catch(() => setRight(null)) }, [rightId])
  if (others.length === 0) return <p className="text-xs text-slate-500">Run another Stage 1 / Stage 2 configuration of this question to compare two videos side by side.</p>
  const ids = (j: Job | null) => new Set(selected(j).map((r) => r.id))
  const lab = (id: string | null) => { const j = same.find((x) => x.id === id); return j ? configLabel(j, s1, s2) : '' }
  function both() {
    const vs = [va.current, vb.current].filter((v): v is HTMLVideoElement => !!v)
    if (playing) { vs.forEach((v) => v.pause()); setPlaying(false); return }
    vs.forEach((v) => { v.currentTime = 0; void v.play() }); setPlaying(true)
  }
  const picker = (v: string | null, set: (id: string) => void, not: string | null) => (
    <select value={v ?? ''} onChange={(e) => { set(e.target.value); setPlaying(false) }} className="min-w-0 max-w-full rounded border p-1 text-sm">
      {same.filter((j) => j.id !== not).map((j) => <option key={j.id} value={j.id}>{configLabel(j, s1, s2)}{j.premade ? '' : ' (your run)'}</option>)}
    </select>
  )
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="rounded bg-blue-100 px-1.5 font-semibold text-blue-900">A</span>{picker(leftId, setLeftId, rightId)}
        <span className="text-slate-400">vs</span>
        <span className="rounded bg-orange-100 px-1.5 font-semibold text-orange-900">B</span>{picker(rightId, setRightId, leftId)}
        <button onClick={() => { const l = leftId; setLeftId(rightId ?? l); setRightId(l); setPlaying(false) }} className="rounded border px-2 py-1 text-xs">⇄ Swap</button>
        <button onClick={both} className="rounded bg-blue-600 px-3 py-1 text-white">{playing ? '❚❚ Pause both' : '▶ Play both'}</button>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <Side job={left} other={ids(right)} label={`A · ${lab(leftId)}`} tone="bg-blue-50 text-blue-900" vref={(v) => { va.current = v }} />
        <Side job={right} other={ids(left)} label={`B · ${lab(rightId)}`} tone="bg-orange-50 text-orange-900" vref={(v) => { vb.current = v }} />
      </div>
    </div>
  )
}
