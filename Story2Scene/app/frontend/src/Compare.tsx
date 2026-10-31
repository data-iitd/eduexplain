import { useEffect, useRef, useState } from 'react'
import type { Job, JobSummary } from './types'
import { apiPath, figureUrl, getJob, isFig, listJobs } from './api'
import Stage2View from './Stage2'

// Compare tab: two finished configurations of the same question side by side (video, Stage 1 set, storyboard).
// Resources in both Stage 1 sets are marked "in both"; "Play both" starts and pauses the two videos together.
// The question picker lists questions with at least two finished runs; it starts on the job last opened in "Make a video".
const short = (t?: string, n = 120) => (t && t.length > n ? t.slice(0, n) + '…' : t ?? '')
export const S1L: Record<string, string> = { auto: 'SetCoRe-LLM', manual: 'Manual', lw: 'SetCoRe-LW', e2e: 'SetCoRe-E2E', optiset: 'OptiSet', gold: 'Benchmark gold set' }
export const S2L: Record<string, string> = { set2story: 'Set2Story', freeform: 'FreeForm', reference: 'Benchmark reference' }
export const configLabel = (j: JobSummary, s1: Record<string, string> = S1L, s2: Record<string, string> = S2L) =>
  `${j.stage1_label ?? s1[j.selection ?? 'auto'] ?? j.selection} + ${j.stage2_label ?? s2[j.stage2 ?? 'set2story'] ?? j.stage2}`
const qkey = (j: JobSummary) => j.query_id ?? j.query
// our full system first, so it is A by default
const ours = (j: JobSummary) => (j.stage1_label ?? S1L[j.selection ?? 'auto']) === 'SetCoRe-LLM' && (j.stage2_label ?? S2L[j.stage2 ?? 'set2story']) === 'Set2Story'

export default function CompareTab({ lastJob }: { lastJob: string | null }) {
  const [jobs, setJobs] = useState<JobSummary[] | null>(null)
  const [key, setKey] = useState<string | null>(null)
  useEffect(() => { listJobs().then(setJobs).catch(() => setJobs([])) }, [])
  const groups = new Map<string, JobSummary[]>()
  for (const j of jobs ?? []) if (j.status === 'done') groups.set(qkey(j), [...(groups.get(qkey(j)) ?? []), j])
  const keys = [...groups.keys()].filter((k) => groups.get(k)!.length > 1)
  const last = (jobs ?? []).find((j) => j.id === lastJob)
  const cur = key ?? (last && keys.includes(qkey(last)) ? qkey(last) : keys[0] ?? null)
  if (!jobs) return <p className="text-sm text-slate-500">Loading…</p>
  if (!cur) return <p className="text-sm text-slate-500">No question has two finished configurations yet. Run another Stage 1 / Stage 2 method of a question in "Make a video".</p>
  const list = [...groups.get(cur)!].sort((a, b) => Number(ours(b)) - Number(ours(a)))
  const a = last && qkey(last) === cur ? last.id : list[0].id
  return (
    <div className="space-y-3 rounded-lg border border-slate-300 bg-white p-3 text-sm">
      <label className="block">Question
        <select value={cur} onChange={(e) => setKey(e.target.value)} className="mt-1 block w-full min-w-0 rounded border p-1">
          {keys.map((k) => { const g = groups.get(k)!; return <option key={k} value={k}>{g[0].query.length > 110 ? g[0].query.slice(0, 110) + '…' : g[0].query}{g[0].query_id ? ` · ${g[0].query_id}` : ''} ({g.length} runs)</option> })}
        </select>
      </label>
      <Pair key={cur} jobs={list} first={a} />
    </div>
  )
}

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

// two finished runs of one question (jobs: that question's runs, first: the initial A)
function Pair({ jobs: same, first }: { jobs: JobSummary[]; first: string }) {
  const [leftId, setLeftId] = useState(first)
  const [rightId, setRightId] = useState<string | null>(same.find((j) => j.id !== first)?.id ?? null)
  const [left, setLeft] = useState<Job | null>(null)
  const [right, setRight] = useState<Job | null>(null)
  const va = useRef<HTMLVideoElement | null>(null), vb = useRef<HTMLVideoElement | null>(null)
  const [playing, setPlaying] = useState(false)
  useEffect(() => { setLeft(null); getJob(leftId).then(setLeft).catch(() => setLeft(null)) }, [leftId])
  useEffect(() => { setRight(null); if (rightId) getJob(rightId).then(setRight).catch(() => setRight(null)) }, [rightId])
  const ids = (j: Job | null) => new Set(selected(j).map((r) => r.id))
  const lab = (id: string | null) => { const j = same.find((x) => x.id === id); return j ? configLabel(j) : '' }
  function both() {
    const vs = [va.current, vb.current].filter((v): v is HTMLVideoElement => !!v)
    if (playing) { vs.forEach((v) => v.pause()); setPlaying(false); return }
    vs.forEach((v) => { v.currentTime = 0; void v.play() }); setPlaying(true)
  }
  const picker = (v: string | null, set: (id: string) => void, not: string | null) => (
    <select value={v ?? ''} onChange={(e) => { set(e.target.value); setPlaying(false) }} className="min-w-0 max-w-full rounded border p-1 text-sm">
      {same.filter((j) => j.id !== not).map((j) => <option key={j.id} value={j.id}>{configLabel(j)}{j.premade ? '' : ' (your run)'}</option>)}
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
